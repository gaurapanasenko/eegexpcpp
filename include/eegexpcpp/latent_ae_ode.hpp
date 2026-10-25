#pragma once

#include <iosfwd>
#include <map>
#include <string>

#include "eegexpcpp/config.hpp"
#include "eegexpcpp/decoder.hpp"
#include "eegexpcpp/encoder.hpp"
#include "eegexpcpp/ode.hpp"

namespace eegexpcpp {

struct ForwardOutput {
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

  const Ode& ode_module() const { return ode_; }

 private:
  int64_t n_latent_{0};
  Encoder encoder_{nullptr};
  Ode ode_{nullptr};
  Decoder decoder_{nullptr};
};

TORCH_MODULE(LatentAeOde);

torch::Tensor l1_mean_abs(const torch::nn::Module& module);

void print_latent_ae_ode_summary(const LatentAeOde& model,
                                 const ModelConfig& cfg,
                                 std::ostream& os);

}  // namespace eegexpcpp
