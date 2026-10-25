#include "eegexpcpp/ode.hpp"

namespace eegexpcpp {

OdeImpl::OdeImpl(int64_t latent_dim,
                 const std::vector<int64_t>& hidden,
                 const std::string& activation)
    : net_(latent_dim, hidden, latent_dim, activation) {
  register_module("net", net_);
}

torch::Tensor OdeImpl::forward(torch::Tensor z) {
  return net_->forward(z);
}

}  // namespace eegexpcpp
