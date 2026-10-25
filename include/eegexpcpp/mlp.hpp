#pragma once

#include <string>
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

 private:
  torch::nn::Sequential seq_{nullptr};
};

TORCH_MODULE(TimeMlp);

}  // namespace eegexpcpp
