#pragma once

#include <ATen/Functions.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/torch.h>

#include <functional>
#include <utility>

namespace eegexpcpp {

/// Forward-mode JVP: fn(primal) and Jacobian @ tangent at primal.
inline std::pair<at::Tensor, at::Tensor> jvp(
    const std::function<at::Tensor(const at::Tensor&)>& fn,
    const at::Tensor& primal,
    const at::Tensor& tangent) {
  const uint64_t level = torch::autograd::forward_ad::enter_dual_level();
  const int64_t lvl = static_cast<int64_t>(level);
  auto dual_in = at::_make_dual(primal, tangent, lvl);
  auto dual_out = fn(dual_in);
  auto unpacked = at::_unpack_dual(dual_out, lvl);
  torch::autograd::forward_ad::exit_dual_level(level);
  return {std::get<0>(unpacked), std::get<1>(unpacked)};
}

inline std::pair<at::Tensor, at::Tensor> jvp(
    const std::function<at::Tensor(const at::Tensor&, const at::Tensor&)>& fn,
    const at::Tensor& primal_a,
    const at::Tensor& tangent_a,
    const at::Tensor& primal_b,
    const at::Tensor& tangent_b) {
  const uint64_t level = torch::autograd::forward_ad::enter_dual_level();
  const int64_t lvl = static_cast<int64_t>(level);
  auto dual_a = at::_make_dual(primal_a, tangent_a, lvl);
  auto dual_b = at::_make_dual(primal_b, tangent_b, lvl);
  auto dual_out = fn(dual_a, dual_b);
  auto unpacked = at::_unpack_dual(dual_out, lvl);
  torch::autograd::forward_ad::exit_dual_level(level);
  return {std::get<0>(unpacked), std::get<1>(unpacked)};
}

}  // namespace eegexpcpp
