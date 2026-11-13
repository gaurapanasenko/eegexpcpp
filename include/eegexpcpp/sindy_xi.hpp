#pragma once

#include <cstdint>

#include <torch/torch.h>

namespace eegexpcpp {

/// True after epoch `epoch` (0-based) when `xi_prune_every` > 0 (global index).
inline bool xi_prune_due_epoch(int64_t epoch, int64_t prune_every) {
  if (prune_every <= 0) {
    return false;
  }
  return (epoch + 1) % prune_every == 0;
}

/// Zero entries with |xi| < threshold; returns new tensor.
torch::Tensor xi_apply_threshold(torch::Tensor xi, double threshold);

/// Count entries with |xi| >= threshold.
int64_t xi_count_nnz(const torch::Tensor& xi, double threshold);

/// 1 where |xi| >= threshold, else 0 (float mask, same shape as xi).
torch::Tensor xi_active_mask(const torch::Tensor& xi, double threshold);

/// trainable &= active_mask(xi); pruned entries stay frozen thereafter.
void xi_shrink_trainable_mask(torch::Tensor& trainable,
                              const torch::Tensor& xi,
                              double threshold);

/// Zero weights on entries where trainable is 0.
void xi_clamp_trainable(torch::Tensor& xi, const torch::Tensor& trainable);

/// Zero grads on entries where trainable is 0.
void xi_mask_grad(torch::Tensor& xi, const torch::Tensor& trainable);

/// Clamp weights and grads (both helpers).
void xi_apply_trainable_mask(torch::Tensor& xi,
                             const torch::Tensor& trainable);

}  // namespace eegexpcpp
