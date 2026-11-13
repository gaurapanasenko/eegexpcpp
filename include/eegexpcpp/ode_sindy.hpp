#pragma once

#include <cstdint>
#include <vector>

#include <torch/torch.h>

#include "eegexpcpp/config.hpp"
#include "eegexpcpp/sindy_library.hpp"

namespace eegexpcpp {

/// Z [B,T,L] -> dZ_pred [B,T,L] via sparse-linear SINDy library.
class OdeSindyImpl : public torch::nn::Module {
 public:
  OdeSindyImpl(int64_t latent_dim,
               const std::vector<SindyLibraryConfig>& libraries);

  torch::Tensor forward(torch::Tensor z);

  int64_t feature_dim() const { return library_.size(); }

  const SindyFeatureLibrary& library() const { return library_; }

  torch::Tensor xi_matrix() const;

  void set_xi_matrix(torch::Tensor xi);

  /// Hard-threshold xi_, freeze mask, zero pruned storage.
  void prune_xi(double threshold);

  int64_t xi_nnz(double threshold) const;

  /// Zero gradients on frozen xi entries (after backward).
  void mask_xi_gradient();

  /// Zero xi_ storage on frozen entries (after optimizer step).
  void enforce_trainable_mask();

 private:
  torch::Tensor xi_effective() const;

  int64_t latent_dim_{0};
  SindyFeatureLibrary library_;
  torch::Tensor xi_;
  torch::Tensor xi_trainable_;
};

TORCH_MODULE(OdeSindy);

}  // namespace eegexpcpp
