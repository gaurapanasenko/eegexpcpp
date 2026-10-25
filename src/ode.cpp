#include "eegexpcpp/ode.hpp"

#include <stdexcept>

namespace eegexpcpp {

OdeImpl::OdeImpl(int64_t latent_dim, const OdeConfig& ode_cfg)
    : kind_(ode_cfg.type) {
  if (kind_ == OdeType::Nn) {
    net_ = TimeMlp(latent_dim, ode_cfg.nn.hidden, latent_dim,
                   ode_cfg.nn.activation);
    register_module("net", net_);
    return;
  }
  if (kind_ == OdeType::Sindy) {
    sindy_ = OdeSindy(latent_dim, ode_cfg.libraries);
    register_module("sindy", sindy_);
    return;
  }
  throw std::runtime_error("unknown ode type");
}

torch::Tensor OdeImpl::forward(torch::Tensor z) {
  if (kind_ == OdeType::Nn) {
    return net_->forward(std::move(z));
  }
  return sindy_->forward(std::move(z));
}

const OdeSindy& OdeImpl::sindy() const {
  if (kind_ != OdeType::Sindy) {
    throw std::runtime_error("ode is not sindy");
  }
  return sindy_;
}

}  // namespace eegexpcpp
