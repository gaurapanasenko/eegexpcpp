#pragma once

#include <iosfwd>
#include <map>
#include <string>

#include "eegexpcpp/config.hpp"
#include "eegexpcpp/latent_mlp.hpp"
#include "eegexpcpp/ode.hpp"

namespace eegexpcpp {

struct ForwardOutput {
  /// MLP latent (normalized); physical copies for ODE / infer export.
  torch::Tensor z_norm;
  torch::Tensor dz_norm;
  torch::Tensor dz_pred_norm;
  torch::Tensor z;
  torch::Tensor dz;
  torch::Tensor dz_pred;
  torch::Tensor y;
  torch::Tensor dy_pred;
};

class LatentAeOdeImpl : public torch::nn::Module {
 public:
  explicit LatentAeOdeImpl(const ModelConfig& model_cfg);

  ForwardOutput forward(torch::Tensor x, torch::Tensor dx);

  std::map<std::string, torch::Tensor> compute_losses(
      const ForwardOutput& out,
      torch::Tensor x,
      torch::Tensor dx,
      const LossConfig& loss_cfg) const;

  torch::Tensor total_loss(
      const std::map<std::string, torch::Tensor>& losses,
      const LossConfig& loss_cfg) const;

  torch::Tensor ode_forward(torch::Tensor z);

  /// Match `eegexpcpp.prep.zscore` affine on latent dim (broadcast).
  void set_latent_z_affine(float mean, float std);

  /// Inverse of encoder output affine (for J_z_1 in normalized space).
  torch::Tensor latent_z_normalized(torch::Tensor z) const;

  float latent_affine_mean() const;
  float latent_affine_scale() const;

  const Ode& ode_module() const { return ode_; }

  Ode& ode_module() { return ode_; }

 private:
  int64_t n_latent_{0};
  LatentMlp encoder_{nullptr};
  Ode ode_{nullptr};
  LatentMlp decoder_{nullptr};
  torch::Tensor latent_mean_{torch::zeros({1, 1, 1})};
  torch::Tensor latent_scale_{torch::ones({1, 1, 1})};
};

TORCH_MODULE(LatentAeOde);

torch::Tensor l1_mean_abs(const torch::nn::Module& module);

void print_latent_ae_ode_summary(const LatentAeOde& model,
                                 const ModelConfig& cfg,
                                 std::ostream& os);

}  // namespace eegexpcpp
