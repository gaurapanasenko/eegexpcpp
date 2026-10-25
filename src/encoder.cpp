#include "eegexpcpp/encoder.hpp"

namespace eegexpcpp {

EncoderImpl::EncoderImpl(int64_t feature_dim,
                         int64_t latent_dim,
                         const std::vector<int64_t>& hidden,
                         const std::string& activation)
    : net_(feature_dim, hidden, latent_dim, activation) {
  register_module("net", net_);
}

torch::Tensor EncoderImpl::encode(torch::Tensor x) {
  return net_->forward(x);
}

std::pair<torch::Tensor, torch::Tensor> EncoderImpl::forward(
    torch::Tensor x,
    torch::Tensor dx_dt) {
  return net_->forward_jvp(std::move(x), std::move(dx_dt));
}

}  // namespace eegexpcpp
