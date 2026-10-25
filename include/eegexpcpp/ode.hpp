#pragma once

#include "eegexpcpp/mlp.hpp"

namespace eegexpcpp {

/// Z [B,T,L] -> dZ_pred [B,T,L].
class OdeImpl : public torch::nn::Module {
 public:
  OdeImpl(int64_t latent_dim,
          const std::vector<int64_t>& hidden,
          const std::string& activation);

  torch::Tensor forward(torch::Tensor z);

 private:
  TimeMlp net_{nullptr};
};

TORCH_MODULE(Ode);

}  // namespace eegexpcpp
