#include "eegexpcpp/latent_mlp.hpp"

namespace eegexpcpp {

LatentMlpImpl::LatentMlpImpl(int64_t in_features,
                             int64_t out_features,
                             const std::vector<int64_t>& hidden,
                             const std::string& activation)
    : net_(in_features, hidden, out_features, activation) {
  register_module("net", net_);
}

torch::Tensor LatentMlpImpl::forward(torch::Tensor x) {
  return net_->forward(std::move(x));
}

std::pair<torch::Tensor, torch::Tensor> LatentMlpImpl::forward_jvp(
    torch::Tensor x,
    torch::Tensor dx) {
  return net_->forward_jvp(std::move(x), std::move(dx));
}

}  // namespace eegexpcpp