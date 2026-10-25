#pragma once

#include <iosfwd>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "eegexpcpp/config.hpp"
#include "eegexpcpp/latent_ae_ode.hpp"

namespace eegexpcpp {

struct SindyCoefficientsTable {
  std::vector<std::string> features;
  std::vector<std::string> latent_labels;
  /// Row-major [features.size() * latent_labels.size()], Xi[f,l].
  std::vector<float> xi;
};

bool model_ode_is_sindy(const ModelConfig& cfg);

SindyCoefficientsTable build_sindy_coefficients_table(
    const LatentAeOde& model,
    const ModelConfig& cfg);

void write_sindy_coefficients_csv(const SindyCoefficientsTable& table,
                                  std::ostream& os);

std::string sindy_coefficients_csv_string(const LatentAeOde& model,
                                          const ModelConfig& cfg);

}  // namespace eegexpcpp
