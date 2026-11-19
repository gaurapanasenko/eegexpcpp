#include "eegexpcpp/sindy_xi.hpp"

#include <limits>
#include <stdexcept>

namespace eegexpcpp {

namespace {

int64_t matrix_nnz(const torch::Tensor& xi, double threshold) {
  int64_t n = 0;
  auto x = xi.detach().to(torch::kCPU).contiguous().view({-1});
  for (int64_t i = 0; i < x.numel(); ++i) {
    const double v = x[i].item<double>();
    if (v >= threshold || v <= -threshold) {
      ++n;
    }
  }
  return n;
}

}  // namespace

torch::Tensor xi_apply_threshold(torch::Tensor xi, double threshold) {
  if (threshold < 0.0) {
    throw std::runtime_error("xi_threshold must be non-negative");
  }
  xi = xi.detach();
  return torch::where(xi.abs() >= threshold, xi,
                      torch::zeros_like(xi));
}

int64_t xi_count_nnz(const torch::Tensor& xi, double threshold) {
  return matrix_nnz(xi, threshold);
}

torch::Tensor xi_active_mask(const torch::Tensor& xi, double threshold) {
  if (threshold < 0.0) {
    throw std::runtime_error("xi_threshold must be non-negative");
  }
  return (xi.detach().abs() >= threshold).to(xi.dtype());
}

void xi_shrink_trainable_mask(torch::Tensor& trainable,
                              const torch::Tensor& xi,
                              double threshold) {
  trainable.mul_(xi_active_mask(xi, threshold));
}

void xi_clamp_trainable(torch::Tensor& xi,
                        const torch::Tensor& trainable) {
  if (xi.sizes() != trainable.sizes()) {
    throw std::runtime_error("xi trainable mask shape mismatch");
  }
  torch::NoGradGuard no_grad;
  xi.mul_(trainable);
}

void xi_mask_grad(torch::Tensor& xi, const torch::Tensor& trainable) {
  if (xi.sizes() != trainable.sizes()) {
    throw std::runtime_error("xi trainable mask shape mismatch");
  }
  if (xi.grad().defined()) {
    xi.mutable_grad().mul_(trainable);
  }
}

void xi_apply_trainable_mask(torch::Tensor& xi,
                             const torch::Tensor& trainable) {
  xi_clamp_trainable(xi, trainable);
  xi_mask_grad(xi, trainable);
}

bool xi_prune_weakest(torch::Tensor& xi, torch::Tensor& trainable) {
  if (xi.sizes() != trainable.sizes()) {
    throw std::runtime_error("xi trainable mask shape mismatch");
  }
  torch::NoGradGuard no_grad;
  const auto active = trainable > 0;
  if (!active.any().item<bool>()) {
    return false;
  }
  const auto abs_xi = xi.detach().abs();
  const auto masked =
      torch::where(active, abs_xi,
                   torch::full_like(abs_xi, std::numeric_limits<double>::infinity()));
  const int64_t flat_idx = masked.view({-1}).argmin().item<int64_t>();
  auto xi_flat = xi.view({-1});
  auto trainable_flat = trainable.view({-1});
  xi_flat[flat_idx] = 0;
  trainable_flat[flat_idx] = 0;
  return true;
}

int64_t xi_count_active(const torch::Tensor& trainable) {
  return trainable.gt(0).sum().item<int64_t>();
}

}  // namespace eegexpcpp
