#include "eegexpcpp/ode_sindy.hpp"

#include <stdexcept>

namespace eegexpcpp {

OdeSindyImpl::OdeSindyImpl(int64_t latent_dim,
                           const std::vector<SindyLibraryConfig>& libraries)
    : latent_dim_(latent_dim), library_(latent_dim, libraries) {
  if (latent_dim <= 0) {
    throw std::runtime_error("OdeSindy latent_dim must be positive");
  }
  const int64_t n_features = library_.size();
  xi_ = register_parameter(
      "xi", torch::zeros({n_features, latent_dim_}, torch::kFloat32));
}

torch::Tensor OdeSindyImpl::forward(torch::Tensor z) {
  const torch::Tensor theta = library_.build_theta(std::move(z));
  return torch::matmul(theta, xi_);
}

torch::Tensor OdeSindyImpl::xi_matrix() const { return xi_; }

}  // namespace eegexpcpp
