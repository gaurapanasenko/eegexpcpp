#include <atomic>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include <torch/torch.h>

#include "eegexpcpp/checkpoint.hpp"
#include "eegexpcpp/config.hpp"
#include "eegexpcpp/dataset_h5.hpp"
#include "eegexpcpp/latent_ae_ode.hpp"
#include "eegexpcpp/sindy_xi.hpp"
#include "eegexpcpp/train_history.hpp"

namespace fs = std::filesystem;

namespace {

std::atomic<bool> g_train_interrupted{false};

void on_train_sigint(int /*signum*/) { g_train_interrupted = true; }

constexpr int kLossWidth = 12;
constexpr int kLossPrec = 4;

int epoch_field_width(int64_t max_epoch) {
  int w = 1;
  for (int64_t x = max_epoch; x > 0; x /= 10) {
    ++w;
  }
  return std::max(w, 5);
}

void print_named_metric(const char* name, double value) {
  std::cout << ' ' << name << '='
            << std::right << std::fixed << std::setprecision(kLossPrec)
            << std::setfill(' ') << std::setw(kLossWidth) << value;
}

void apply_adamw_lr_weight_decay(torch::optim::AdamW& optimizer,
                                 const eegexpcpp::TrainConfig& train) {
  for (auto& group : optimizer.param_groups()) {
    auto& opts =
        static_cast<torch::optim::AdamWOptions&>(group.options());
    opts.lr(train.lr);
    opts.weight_decay(train.weight_decay);
  }
}

void print_loss_row(int epoch_w,
                    int64_t epoch,
                    double train_j,
                    double val_j,
                    double j_y,
                    double j_dy,
                    double j_dz,
                    double j_z1) {
  std::cout << "epoch="
            << std::right << std::setfill(' ') << std::setw(epoch_w) << epoch;
  print_named_metric("train_J", train_j);
  print_named_metric("val_J", val_j);
  print_named_metric("J_Y", j_y);
  print_named_metric("J_dY", j_dy);
  print_named_metric("J_dZ", j_dz);
  print_named_metric("J_z_1", j_z1);
  std::cout << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: eegexpcpp_train <config.json>\n";
    return 1;
  }

  const eegexpcpp::ExperimentConfig cfg =
      eegexpcpp::load_config(argv[1]);
  torch::manual_seed(cfg.train.seed);
  const auto device = eegexpcpp::resolve_device(cfg.train.device);

  auto [x_all, dx_all] = eegexpcpp::load_h5_pair_btf(
      cfg.data.h5_path, cfg.data.h5_group);
  auto [train_pair, val_pair] =
      eegexpcpp::split_train_val(x_all, dx_all, cfg.data.train_fraction);
  auto x_train = train_pair.first.to(device);
  auto dx_train = train_pair.second.to(device);
  auto x_val = val_pair.first.to(device);
  auto dx_val = val_pair.second.to(device);

  eegexpcpp::ModelConfig model_cfg = cfg.model;
  model_cfg.feature_dim = x_train.size(2);
  const auto [z_mean, z_std] = eegexpcpp::load_h5_z_affine(
      cfg.data.h5_path, cfg.data.h5_group);
  eegexpcpp::LatentAeOde model(model_cfg);
  model->to(device);
  model->train();
  eegexpcpp::print_latent_ae_ode_summary(model, model_cfg, std::cout);

  torch::optim::AdamW optimizer(
      model->parameters(),
      torch::optim::AdamWOptions(cfg.train.lr)
          .weight_decay(cfg.train.weight_decay)
          .betas({cfg.train.adamw_beta1, cfg.train.adamw_beta2})
          .eps(cfg.train.adamw_eps));

  const auto ckpt_path =
      eegexpcpp::default_checkpoint_path(cfg.train.checkpoint_dir);
  const auto resume =
      eegexpcpp::try_resume_train(ckpt_path, model, optimizer);
  model->set_latent_z_affine(z_mean, z_std);
  if (resume.resumed && resume.optimizer_state) {
    apply_adamw_lr_weight_decay(optimizer, cfg.train);
  }
  const int64_t epoch_begin = resume.next_epoch;
  const int64_t epoch_end = resume.next_epoch + cfg.train.epochs;

  if (resume.resumed) {
    std::cout << "resume: loaded " << ckpt_path << ", epochs ["
              << epoch_begin << ", " << epoch_end << ")";
    if (!resume.optimizer_state) {
      std::cout << " (weights only, optimizer reset)";
    }
    std::cout << '\n';
  }

  const int epoch_w = epoch_field_width(epoch_end - 1);

  eegexpcpp::TrainHistory history;
  if (resume.resumed) {
    history.load_from_checkpoint_file(ckpt_path);
  }
  eegexpcpp::CheckpointBests bests;
  if (fs::exists(ckpt_path)) {
    eegexpcpp::load_checkpoint_bests(ckpt_path, bests, model_cfg);
  }
  history.begin_run(epoch_begin, epoch_end, cfg.train.log_every, cfg.train,
                    cfg.loss);

  const auto& xi_cfg = cfg.train.xi_prune;
  const bool sindy_train = cfg.model.ode.type == eegexpcpp::OdeType::Sindy;
  const bool xi_prune_train =
      sindy_train && xi_cfg.prune_every > 0;
  const bool xi_weak_prune_train =
      sindy_train && xi_cfg.weak_prune_every > 0;
  const bool xi_mask_train = xi_prune_train || xi_weak_prune_train;

  std::signal(SIGINT, on_train_sigint);

  const auto save_checkpoint = [&](int64_t next_epoch, const char* reason) {
    if (xi_mask_train) {
      model->ode_module()->sindy()->enforce_trainable_mask();
    }
    eegexpcpp::save_train_checkpoint(ckpt_path, model, optimizer, next_epoch,
                                     history, bests);
    std::cout << reason << "saved " << ckpt_path
              << " next_epoch=" << next_epoch << '\n';
  };

  for (int64_t epoch = epoch_begin; epoch < epoch_end; ++epoch) {
    if (g_train_interrupted) {
      save_checkpoint(epoch, "interrupt (before epoch): ");
      return 128 + SIGINT;
    }

    optimizer.zero_grad();
    auto out = model->forward(x_train, dx_train);
    auto losses = model->compute_losses(out, x_train, dx_train, cfg.loss);
    auto loss = model->total_loss(losses, cfg.loss);
    loss.backward();
    if (cfg.train.grad_clip_norm > 0.0) {
      torch::nn::utils::clip_grad_norm_(model->parameters(),
                                        cfg.train.grad_clip_norm);
    }
    if (xi_mask_train) {
      model->ode_module()->sindy()->mask_xi_gradient();
    }
    optimizer.step();
    if (xi_mask_train) {
      auto& sindy = model->ode_module()->sindy();
      sindy->enforce_trainable_mask();
      if (xi_prune_train &&
          eegexpcpp::xi_prune_due_epoch(epoch, xi_cfg.prune_every)) {
        torch::NoGradGuard no_grad;
        sindy->prune_xi(xi_cfg.threshold);
        const int64_t nnz = sindy->xi_nnz(xi_cfg.threshold);
        std::cout << "xi_prune epoch=" << epoch << " nnz=" << nnz << '\n';
        history.append_xi_prune(epoch, nnz);
      }
      if (xi_weak_prune_train &&
          eegexpcpp::xi_prune_due_epoch(epoch, xi_cfg.weak_prune_every)) {
        torch::NoGradGuard no_grad;
        const bool was_training = model->is_training();
        model->eval();
        auto wout = model->forward(x_train, dx_train);
        auto wlosses =
            model->compute_losses(wout, x_train, dx_train, cfg.loss);
        const double train_j_now =
            model->total_loss(wlosses, cfg.loss).item<double>();
        if (was_training) {
          model->train();
        }
        const double best_before = bests.best_train_j;
        if (train_j_now <=
                best_before * xi_cfg.weak_prune_best_train_j_slack &&
            sindy->prune_xi_weakest()) {
          sindy->enforce_trainable_mask();
          const int64_t nnz = sindy->xi_active_count();
          std::cout << "xi_weak_prune epoch=" << epoch << " nnz=" << nnz
                    << " train_J=" << train_j_now
                    << " best_train_J=" << best_before << '\n';
          history.append_xi_weak_prune(epoch, nnz, train_j_now, best_before);
        }
      }
    }

    if (epoch % cfg.train.log_every == 0 || epoch == epoch_end - 1) {
      model->eval();
      torch::NoGradGuard no_grad;
      auto vout = model->forward(x_val, dx_val);
      auto vlosses =
          model->compute_losses(vout, x_val, dx_val, cfg.loss);
      auto vtotal = model->total_loss(vlosses, cfg.loss);
      model->train();

      const double train_j = loss.item<double>();
      const double val_j = vtotal.item<double>();
      const double j_y = losses.at("J_Y").item<double>();
      const double j_dy = losses.at("J_dY").item<double>();
      const double j_dz = losses.at("J_dZ").item<double>();
      const double j_z1 = losses.at("J_z_1").item<double>();
      print_loss_row(epoch_w, epoch, train_j, val_j, j_y, j_dy, j_dz, j_z1);
      history.append_loss_row(epoch, train_j, val_j, j_y, j_dy, j_dz, j_z1);
      bests.consider_train(epoch, train_j, model, model_cfg);
      bests.consider_val(epoch, val_j, model, model_cfg);
    }

    if (g_train_interrupted) {
      save_checkpoint(epoch + 1, "interrupt (after epoch): ");
      return 128 + SIGINT;
    }
  }

  save_checkpoint(epoch_end, "");
  return 0;
}
