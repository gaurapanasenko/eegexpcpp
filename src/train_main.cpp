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
#include "eegexpcpp/sindy_stlsq.hpp"
#include "eegexpcpp/sindy_table.hpp"

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

  std::signal(SIGINT, on_train_sigint);

  const auto save_checkpoint = [&](int64_t next_epoch, const char* reason) {
    eegexpcpp::save_train_checkpoint(ckpt_path, model, optimizer, next_epoch);
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
    optimizer.step();

    const auto& stlsq_cfg = cfg.model.ode.stlsq;
    if (eegexpcpp::model_ode_is_sindy(cfg.model) && stlsq_cfg.every > 0) {
      const int64_t since_begin = epoch - epoch_begin + 1;
      if (since_begin > 0 && since_begin % stlsq_cfg.every == 0) {
        torch::NoGradGuard no_grad;
        model->eval();
        auto fit_out = model->forward(x_train, dx_train);
        const auto& sindy = model->ode_module()->sindy();
        auto theta = sindy->library().build_theta(
            model->latent_z_physical(fit_out.z));
        const auto theta_nt =
            theta.reshape({-1, theta.size(-1)});
        const auto dz_nt = model->latent_dz_physical(fit_out.dz)
                               .reshape({-1, fit_out.dz.size(-1)});
        const auto fit = eegexpcpp::sindy_stlsq_fit(
            theta_nt, dz_nt, stlsq_cfg.threshold, stlsq_cfg.max_iter,
            stlsq_cfg.ridge);
        bool xi_updated = false;
        for (const auto& pair : model->named_parameters()) {
          if (pair.key() == "ode.sindy.xi") {
            pair.value().copy_(fit.xi.to(device));
            eegexpcpp::reset_adamw_momentum(optimizer, pair.value());
            xi_updated = true;
            break;
          }
        }
        if (!xi_updated) {
          throw std::runtime_error("ode.sindy.xi parameter not found");
        }
        model->train();
        std::cout << "stlsq epoch=" << epoch << " nnz=" << fit.nnz << '\n';
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

      print_loss_row(epoch_w, epoch, loss.item<double>(),
                     vtotal.item<double>(), losses.at("J_Y").item<double>(),
                     losses.at("J_dY").item<double>(),
                     losses.at("J_dZ").item<double>(),
                     losses.at("J_z_1").item<double>());
    }

    if (g_train_interrupted) {
      save_checkpoint(epoch + 1, "interrupt (after epoch): ");
      return 128 + SIGINT;
    }
  }

  save_checkpoint(epoch_end, "");
  return 0;
}
