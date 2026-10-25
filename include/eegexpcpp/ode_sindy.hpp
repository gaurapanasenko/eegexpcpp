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

 private:
  int64_t latent_dim_{0};
  SindyFeatureLibrary library_;
  torch::Tensor xi_;
};

TORCH_MODULE(OdeSindy);

}  // namespace eegexpcpp
