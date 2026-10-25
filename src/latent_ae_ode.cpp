#include "eegexpcpp/latent_ae_ode.hpp"

#include <iostream>
#include <sstream>
#include <stdexcept>

namespace eegexpcpp {

namespace {

torch::Tensor mse_mean(torch::Tensor a, torch::Tensor b) {
  return torch::mse_loss(a, b, torch::Reduction::Mean);
}

}  // namespace

LatentAeOdeImpl::LatentAeOdeImpl(const ModelConfig& model_cfg)
    : n_latent_(model_cfg.n_latent),
      encoder_(model_cfg.feature_dim, model_cfg.n_latent,
               model_cfg.encoder_hidden, model_cfg.encoder_activation),
      ode_(model_cfg.n_latent, model_cfg.ode_hidden, model_cfg.ode_activation),
      decoder_(model_cfg.n_latent, model_cfg.feature_dim,
               model_cfg.decoder_hidden, model_cfg.decoder_activation) {
  if (model_cfg.feature_dim <= 0) {
    throw std::runtime_error("ModelConfig.feature_dim must be set from data");
  }
  if (model_cfg.n_latent <= 0) {
    throw std::runtime_error("ModelConfig.n_latent must be positive");
  }
  register_module("encoder", encoder_);
  register_module("ode", ode_);
  register_module("decoder", decoder_);
}

ForwardOutput LatentAeOdeImpl::forward(torch::Tensor x, torch::Tensor dx) {
  ForwardOutput out;
  std::tie(out.z, out.dz) = encoder_->forward(x, dx);
  out.dz_pred = ode_->forward(out.z);
  std::tie(out.y, out.dy_pred) = decoder_->forward(out.z, out.dz_pred);
  return out;
}

std::map<std::string, torch::Tensor> LatentAeOdeImpl::compute_losses(
    const ForwardOutput& out,
    torch::Tensor x,
    torch::Tensor dx,
    const LossConfig& loss_cfg) const {
  std::map<std::string, torch::Tensor> losses;
  losses["J_Y"] = mse_mean(out.y, x);
  losses["J_dY"] = mse_mean(out.dy_pred, dx);
  losses["J_dZ"] = mse_mean(out.dz_pred, out.dz);
  const auto x0 = x.select(1, 0);
  const auto z0 = out.z.select(1, 0);
  const int64_t n = std::min(n_latent_, x0.size(-1));
  losses["J_z_1"] = mse_mean(z0, x0.narrow(-1, 0, n));
  losses["J_reg_ae_l1"] = l1_mean_abs(*encoder_) + l1_mean_abs(*decoder_);
  return losses;
}

torch::Tensor LatentAeOdeImpl::ode_forward(torch::Tensor z) {
  return ode_->forward(std::move(z));
}

torch::Tensor LatentAeOdeImpl::total_loss(
    const std::map<std::string, torch::Tensor>& losses,
    const LossConfig& loss_cfg) const {
  return losses.at("J_Y") * loss_cfg.lambda_Y +
         losses.at("J_dY") * loss_cfg.lambda_dY +
         losses.at("J_dZ") * loss_cfg.lambda_dZ +
         losses.at("J_z_1") * loss_cfg.lambda_z1 +
         losses.at("J_reg_ae_l1") * loss_cfg.lambda_l1;
}

torch::Tensor l1_mean_abs(const torch::nn::Module& module) {
  std::vector<torch::Tensor> chunks;
  for (const auto& p : module.parameters()) {
    chunks.push_back(p.abs().reshape({-1}));
  }
  if (chunks.empty()) {
    return torch::zeros({1});
  }
  return torch::cat(chunks).mean();
}

namespace {

int64_t count_parameters_prefix(const LatentAeOde& model,
                                const std::string& prefix) {
  int64_t n = 0;
  for (const auto& pair : model->named_parameters()) {
    if (pair.key().compare(0, prefix.size(), prefix) == 0) {
      n += pair.value().numel();
    }
  }
  return n;
}

std::string format_mlp_stack(int64_t in_features,
                             const std::vector<int64_t>& hidden,
                             int64_t out_features,
                             const std::string& activation) {
  std::ostringstream line;
  line << in_features;
  for (int64_t h : hidden) {
    line << " -> " << h << " (" << activation << ')';
  }
  line << " -> " << out_features;
  return line.str();
}

}  // namespace

void print_latent_ae_ode_summary(const LatentAeOde& model,
                                 const ModelConfig& cfg,
                                 std::ostream& os) {
  const int64_t f = cfg.feature_dim;
  const int64_t l = cfg.n_latent;
  os << "model: feature_dim=" << f << " n_latent=" << l << '\n';
  os << "  encoder: "
     << format_mlp_stack(f, cfg.encoder_hidden, l, cfg.encoder_activation)
     << '\n';
  os << "  ode:     "
     << format_mlp_stack(l, cfg.ode_hidden, l, cfg.ode_activation) << '\n';
  os << "  decoder: "
     << format_mlp_stack(l, cfg.decoder_hidden, f, cfg.decoder_activation)
     << '\n';
  const int64_t enc_p = count_parameters_prefix(model, "encoder.");
  const int64_t ode_p = count_parameters_prefix(model, "ode.");
  const int64_t dec_p = count_parameters_prefix(model, "decoder.");
  const int64_t total = enc_p + ode_p + dec_p;
  os << "  parameters: encoder=" << enc_p << " ode=" << ode_p
     << " decoder=" << dec_p << " total=" << total << '\n';
}

}  // namespace eegexpcpp
