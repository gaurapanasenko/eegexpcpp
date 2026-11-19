#pragma once

#include "eegexpcpp/mlp.hpp"

namespace eegexpcpp {

/// Shared TimeMlp wrapper for encoder (X->Z) and decoder (Z->Y).
class LatentMlpImpl : public torch::nn::Module {
 public:
  LatentMlpImpl(int64_t in_features,
                int64_t out_features,
                const std::vector<int64_t>& hidden,
                const std::string& activation);

  torch::Tensor forward(torch::Tensor x);

  std::pair<torch::Tensor, torch::Tensor> forward_jvp(torch::Tensor x,
                                                      torch::Tensor dx);

 private:
  TimeMlp net_{nullptr};
};

TORCH_MODULE(LatentMlp);

}  // namespace eegexpcpp