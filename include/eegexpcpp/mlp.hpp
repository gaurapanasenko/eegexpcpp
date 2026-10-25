#pragma once

#include <string>
#include <utility>
#include <vector>

#include <torch/torch.h>

namespace eegexpcpp {

/// MLP on last dim of [B, T, in_features] -> [B, T, out_features].
class TimeMlpImpl : public torch::nn::Module {
 public:
  TimeMlpImpl(int64_t in_features,
              const std::vector<int64_t>& hidden,
              int64_t out_features,
              const std::string& activation);

  torch::Tensor forward(torch::Tensor x);

  /// Primal forward and forward-mode JVP along the last dim (dX / dZ).
  std::pair<torch::Tensor, torch::Tensor> forward_jvp(torch::Tensor x,
                                                      torch::Tensor dx);

 private:
  torch::nn::Sequential seq_{nullptr};
};

TORCH_MODULE(TimeMlp);

}  // namespace eegexpcpp
