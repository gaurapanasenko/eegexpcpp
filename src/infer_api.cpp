#include "eegexpcpp/infer_api.h"

#include <cstring>
#include <filesystem>
#include <memory>
#include <string>

#include <torch/torch.h>

#include "eegexpcpp/checkpoint.hpp"
#include "eegexpcpp/config.hpp"
#include "eegexpcpp/dataset_h5.hpp"
#include "eegexpcpp/latent_ae_ode.hpp"

namespace {

thread_local std::string g_last_error;

void set_error(const std::string& msg) { g_last_error = msg; }

void clear_error() { g_last_error.clear(); }

namespace fs = std::filesystem;

void resolve_repo_relative(std::string& path,
                           const fs::path& config_json_path) {
  fs::path p(path);
  if (p.is_absolute()) {
    return;
  }
  const fs::path repo_guess =
      config_json_path.parent_path().parent_path();
  const fs::path candidate = (repo_guess / p).lexically_normal();
  if (fs::exists(candidate)) {
    path = candidate.string();
  }
}

struct InferSessionImpl {
  eegexpcpp::ExperimentConfig cfg;
  eegexpcpp::LatentAeOde model{nullptr};
  torch::Device device{torch::kCPU};
  torch::Tensor x_train;
  torch::Tensor dx_train;
  torch::Tensor x_val;
  torch::Tensor dx_val;
  torch::Tensor x_all;
  torch::Tensor dx_all;
  int64_t next_epoch = -1;
};

std::pair<torch::Tensor, torch::Tensor> select_split(
    const InferSessionImpl& s,
    int split) {
  switch (split) {
    case EEGEXPCPP_INFER_TRAIN:
      return {s.x_train, s.dx_train};
    case EEGEXPCPP_INFER_VAL:
      return {s.x_val, s.dx_val};
    case EEGEXPCPP_INFER_FULL:
      return {s.x_all, s.dx_all};
    default:
      throw std::runtime_error("invalid split id");
  }
}

void copy_btf_to_tf(const torch::Tensor& t,
                    float* out,
                    int64_t feat_expected) {
  auto cpu = t.detach().to(torch::kCPU, torch::kFloat32).contiguous();
  if (cpu.dim() != 3 || cpu.size(0) != 1) {
    throw std::runtime_error("expected [1,T,F] tensor for infer copy");
  }
  const int64_t time = cpu.size(1);
  const int64_t feat = cpu.size(2);
  if (feat != feat_expected) {
    throw std::runtime_error(
        "feature dim mismatch: tensor has " + std::to_string(feat) +
        ", buffer expects " + std::to_string(feat_expected));
  }
  const float* src = cpu.data_ptr<float>();
  for (int64_t ti = 0; ti < time; ++ti) {
    for (int64_t fi = 0; fi < feat; ++fi) {
      out[ti * feat + fi] = src[ti * feat + fi];
    }
  }
}

}  // namespace

struct EegexpcppInferHandle {
  std::unique_ptr<InferSessionImpl> impl;
};

extern "C" {

EegexpcppInferHandle* eegexpcpp_infer_create(const char* config_json_path,
                                             const char* checkpoint_path,
                                             const char* device) {
  clear_error();
  if (config_json_path == nullptr) {
    set_error("config_json_path is null");
    return nullptr;
  }
  try {
    auto session = std::make_unique<EegexpcppInferHandle>();
    session->impl = std::make_unique<InferSessionImpl>();
    auto& s = *session->impl;

    const fs::path cfg_abs = fs::absolute(config_json_path);
    s.cfg = eegexpcpp::load_config(cfg_abs.string());
    resolve_repo_relative(s.cfg.data.h5_path, cfg_abs);
    resolve_repo_relative(s.cfg.train.checkpoint_dir, cfg_abs);
    s.device = eegexpcpp::resolve_device(
        device != nullptr ? device : "cpu");

    auto [x_all, dx_all] = eegexpcpp::load_h5_pair_btf(
        s.cfg.data.h5_path, s.cfg.data.h5_group);
    s.x_all = x_all;
    s.dx_all = dx_all;

    auto [train_pair, val_pair] = eegexpcpp::split_train_val(
        x_all, dx_all, s.cfg.data.train_fraction);
    s.x_train = train_pair.first;
    s.dx_train = train_pair.second;
    s.x_val = val_pair.first;
    s.dx_val = val_pair.second;

    eegexpcpp::ModelConfig model_cfg = s.cfg.model;
    model_cfg.feature_dim = s.x_all.size(2);
    s.model = eegexpcpp::LatentAeOde(model_cfg);
    s.model->to(s.device);
    s.model->eval();

    std::string ckpt = checkpoint_path != nullptr
                           ? std::string(checkpoint_path)
                           : eegexpcpp::default_checkpoint_path(
                                 s.cfg.train.checkpoint_dir);
    if (checkpoint_path == nullptr) {
      resolve_repo_relative(ckpt, cfg_abs);
    }

    torch::optim::AdamW optimizer(
        s.model->parameters(),
        torch::optim::AdamWOptions(s.cfg.train.lr));
    const auto resume =
        eegexpcpp::try_resume_train(ckpt, s.model, optimizer);
    if (!resume.resumed) {
      set_error("checkpoint not found or unloadable: " + ckpt);
      return nullptr;
    }
    s.next_epoch = resume.next_epoch;
    s.model->eval();

    return session.release();
  } catch (const std::exception& e) {
    set_error(e.what());
    return nullptr;
  }
}

void eegexpcpp_infer_destroy(EegexpcppInferHandle* handle) {
  delete handle;
}

const char* eegexpcpp_infer_last_error(void) {
  return g_last_error.c_str();
}

int64_t eegexpcpp_infer_next_epoch(const EegexpcppInferHandle* handle) {
  if (handle == nullptr || handle->impl == nullptr) {
    return -1;
  }
  return handle->impl->next_epoch;
}

int64_t eegexpcpp_infer_n_latent(const EegexpcppInferHandle* handle) {
  if (handle == nullptr || handle->impl == nullptr) {
    return -1;
  }
  return handle->impl->cfg.model.n_latent;
}

int eegexpcpp_infer_sizes(const EegexpcppInferHandle* handle,
                          int split,
                          int64_t* t_out,
                          int64_t* f_out) {
  clear_error();
  if (handle == nullptr || handle->impl == nullptr) {
    set_error("null handle");
    return -1;
  }
  if (t_out == nullptr || f_out == nullptr) {
    set_error("null size output pointer");
    return -1;
  }
  try {
    const auto [x, dx] = select_split(*handle->impl, split);
    *t_out = x.size(1);
    *f_out = x.size(2);
    return 0;
  } catch (const std::exception& e) {
    set_error(e.what());
    return -1;
  }
}

int eegexpcpp_infer_forward(const EegexpcppInferHandle* handle,
                            int split,
                            float* x_out,
                            float* y_out,
                            float* z_out,
                            float* dz_out,
                            float* dz_pred_out) {
  clear_error();
  if (handle == nullptr || handle->impl == nullptr) {
    set_error("null handle");
    return -1;
  }
  if (x_out == nullptr || y_out == nullptr || z_out == nullptr ||
      dz_out == nullptr || dz_pred_out == nullptr) {
    set_error("null output buffer");
    return -1;
  }
  try {
    auto& s = *handle->impl;
    auto [x, dx] = select_split(s, split);
    x = x.to(s.device);
    dx = dx.to(s.device);

    torch::NoGradGuard no_grad;
    const auto out = s.model->forward(x, dx);

    const int64_t f = x.size(2);
    const int64_t l = s.cfg.model.n_latent;
    copy_btf_to_tf(x, x_out, f);
    copy_btf_to_tf(out.y, y_out, f);
    copy_btf_to_tf(out.z, z_out, l);
    copy_btf_to_tf(out.dz, dz_out, l);
    copy_btf_to_tf(out.dz_pred, dz_pred_out, l);
    return 0;
  } catch (const c10::Error& e) {
    set_error(e.what());
    return -1;
  } catch (const std::exception& e) {
    set_error(e.what());
    return -1;
  }
}

int eegexpcpp_infer_ode_rhs(const EegexpcppInferHandle* handle,
                            const float* z_in,
                            float* dz_out,
                            int64_t f_dim) {
  clear_error();
  if (handle == nullptr || handle->impl == nullptr) {
    set_error("null handle");
    return -1;
  }
  if (z_in == nullptr || dz_out == nullptr) {
    set_error("null vector pointer");
    return -1;
  }
  try {
    auto& s = *handle->impl;
    if (f_dim != s.cfg.model.n_latent) {
      set_error("f_dim mismatch (expected n_latent)");
      return -1;
    }
    auto z =
        torch::from_blob(const_cast<float*>(z_in), {1, 1, f_dim},
                         torch::TensorOptions().dtype(torch::kFloat32))
            .clone()
            .to(s.device);
    torch::NoGradGuard no_grad;
    auto dz_t = s.model->ode_forward(z);
    dz_t = dz_t.to(torch::kCPU).contiguous();
    const float* src = dz_t.data_ptr<float>();
    std::memcpy(dz_out, src, static_cast<size_t>(f_dim) * sizeof(float));
    return 0;
  } catch (const c10::Error& e) {
    set_error(e.what());
    return -1;
  } catch (const std::exception& e) {
    set_error(e.what());
    return -1;
  }
}

}  // extern "C"
