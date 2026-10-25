#pragma once

#include "eegexpcpp/mlp.hpp"

namespace eegexpcpp {

/// Y = g(Z); dY = JVP(g, Z, dZ_pred) — second arg like dX on the encoder.
class DecoderImpl : public torch::nn::Module {
 public:
  DecoderImpl(int64_t latent_dim,
              int64_t feature_dim,
              const std::vector<int64_t>& hidden,
              const std::string& activation);

  torch::Tensor decode(torch::Tensor z);

  std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor z,
                                                  torch::Tensor dz_pred);

 private:
  TimeMlp net_{nullptr};
};

TORCH_MODULE(Decoder);

}  // namespace eegexpcpp
