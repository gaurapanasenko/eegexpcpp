#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "eegexpcpp/config.hpp"

namespace eegexpcpp {

using SindyFeatureFn =
    std::function<torch::Tensor(torch::Tensor z)>;

struct SindyFeatureTerm {
  std::string name;
  SindyFeatureFn eval;
};

/// Feature library built once: names + column evaluators for Theta(Z).
class SindyFeatureLibrary {
 public:
  SindyFeatureLibrary(int64_t latent_dim,
                      const std::vector<SindyLibraryConfig>& libraries);

  int64_t size() const { return static_cast<int64_t>(terms_.size()); }

  const std::vector<std::string>& feature_names() const { return names_; }

  const std::vector<SindyFeatureTerm>& terms() const { return terms_; }

  torch::Tensor build_theta(torch::Tensor z) const;

 private:
  std::vector<std::string> names_;
  std::vector<SindyFeatureTerm> terms_;
};

int64_t poly_feature_count(int64_t latent_dim, const PolyLibraryConfig& cfg);

int64_t fourier_feature_count(int64_t latent_dim,
                              const FourierLibraryConfig& cfg);

torch::Tensor poly_features(torch::Tensor z, const PolyLibraryConfig& cfg);

torch::Tensor fourier_features(torch::Tensor z,
                               const FourierLibraryConfig& cfg);

int64_t sindy_library_feature_count(int64_t latent_dim,
                                    const SindyLibraryConfig& cfg);

torch::Tensor sindy_library_features(torch::Tensor z,
                                     const SindyLibraryConfig& cfg);

int64_t sindy_total_feature_count(
    int64_t latent_dim,
    const std::vector<SindyLibraryConfig>& libraries);

torch::Tensor sindy_build_theta(
    torch::Tensor z,
    const std::vector<SindyLibraryConfig>& libraries);

}  // namespace eegexpcpp
