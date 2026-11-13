#include "eegexpcpp/ode_sindy.hpp"

#include <stdexcept>

#include "eegexpcpp/sindy_xi.hpp"

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
  xi_trainable_ = register_buffer(
      "xi_trainable",
      torch::ones({n_features, latent_dim_}, torch::kFloat32));
}

torch::Tensor OdeSindyImpl::xi_effective() const {
  return xi_ * xi_trainable_;
}

torch::Tensor OdeSindyImpl::forward(torch::Tensor z) {
  const torch::Tensor theta = library_.build_theta(std::move(z));
  return torch::matmul(theta, xi_effective());
}

torch::Tensor OdeSindyImpl::xi_matrix() const { return xi_effective(); }

void OdeSindyImpl::set_xi_matrix(torch::Tensor xi) {
  if (!xi_.sizes().equals(xi.sizes())) {
    throw std::runtime_error("set_xi_matrix shape mismatch");
  }
  torch::NoGradGuard no_grad;
  xi_.copy_(xi.detach());
  xi_.mul_(xi_trainable_);
}

void OdeSindyImpl::prune_xi(double threshold) {
  torch::NoGradGuard no_grad;
  xi_.copy_(xi_apply_threshold(xi_, threshold));
  xi_shrink_trainable_mask(xi_trainable_, xi_, threshold);
  xi_.mul_(xi_trainable_);
}

int64_t OdeSindyImpl::xi_nnz(double threshold) const {
  return xi_count_nnz(xi_matrix(), threshold);
}

void OdeSindyImpl::mask_xi_gradient() {
  xi_mask_grad(xi_, xi_trainable_);
}

void OdeSindyImpl::enforce_trainable_mask() {
  torch::NoGradGuard no_grad;
  xi_clamp_trainable(xi_, xi_trainable_);
}

}  // namespace eegexpcpp
