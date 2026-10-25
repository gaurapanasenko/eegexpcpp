#pragma once

#include "eegexpcpp/config.hpp"
#include "eegexpcpp/mlp.hpp"
#include "eegexpcpp/ode_sindy.hpp"

namespace eegexpcpp {

/// Z [B,T,L] -> dZ_pred [B,T,L] (MLP or SINDy library).
class OdeImpl : public torch::nn::Module {
 public:
  OdeImpl(int64_t latent_dim, const OdeConfig& ode_cfg);

  torch::Tensor forward(torch::Tensor z);

  OdeType kind() const { return kind_; }

  bool is_sindy() const { return kind_ == OdeType::Sindy; }

  const OdeSindy& sindy() const;

 private:
  OdeType kind_{OdeType::Nn};
  TimeMlp net_{nullptr};
  OdeSindy sindy_{nullptr};
};

TORCH_MODULE(Ode);

}  // namespace eegexpcpp
