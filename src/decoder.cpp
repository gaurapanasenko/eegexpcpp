#include "eegexpcpp/decoder.hpp"

namespace eegexpcpp {

DecoderImpl::DecoderImpl(int64_t latent_dim,
                         int64_t feature_dim,
                         const std::vector<int64_t>& hidden,
                         const std::string& activation)
    : net_(latent_dim, hidden, feature_dim, activation) {
  register_module("net", net_);
}

torch::Tensor DecoderImpl::decode(torch::Tensor z) {
  return net_->forward(z);
}

std::pair<torch::Tensor, torch::Tensor> DecoderImpl::forward(
    torch::Tensor z,
    torch::Tensor dz_pred) {
  return net_->forward_jvp(std::move(z), std::move(dz_pred));
}

}  // namespace eegexpcpp
