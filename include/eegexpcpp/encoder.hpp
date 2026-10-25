#pragma once

#include "eegexpcpp/mlp.hpp"

namespace eegexpcpp {

/// Z = f(X); dZ = JVP(f, X, dX) — dX is dX/dt from data.
class EncoderImpl : public torch::nn::Module {
 public:
  EncoderImpl(int64_t feature_dim,
              int64_t latent_dim,
              const std::vector<int64_t>& hidden,
              const std::string& activation);

  torch::Tensor encode(torch::Tensor x);

  std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x,
                                                  torch::Tensor dx_dt);

 private:
  TimeMlp net_{nullptr};
};

TORCH_MODULE(Encoder);

}  // namespace eegexpcpp
