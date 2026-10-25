#include "eegexpcpp/sindy_table.hpp"

#include <iomanip>
#include <sstream>
#include <stdexcept>

#include "eegexpcpp/ode.hpp"

namespace eegexpcpp {

bool model_ode_is_sindy(const ModelConfig& cfg) {
  return cfg.ode.type == OdeType::Sindy;
}

SindyCoefficientsTable build_sindy_coefficients_table(
    const LatentAeOde& model,
    const ModelConfig& cfg) {
  if (!model_ode_is_sindy(cfg)) {
    throw std::runtime_error("model ode is not sindy");
  }
  const OdeSindy& sindy = model->ode_module()->sindy();
  const SindyFeatureLibrary& lib = sindy->library();
  const int64_t n_feat = lib.size();
  const int64_t l = cfg.n_latent;
  if (n_feat != sindy->feature_dim()) {
    throw std::runtime_error("sindy feature dim mismatch");
  }
  SindyFeatureLibrary expected(l, cfg.ode.libraries);
  if (expected.size() != n_feat) {
    throw std::runtime_error("sindy library config does not match model");
  }
  const auto& names = lib.feature_names();
  if (names != expected.feature_names()) {
    throw std::runtime_error("sindy feature names do not match config");
  }

  SindyCoefficientsTable out;
  out.features = names;
  out.latent_labels.reserve(static_cast<size_t>(l));
  for (int64_t i = 0; i < l; ++i) {
    out.latent_labels.push_back("dz" + std::to_string(i));
  }

  auto xi = sindy->xi_matrix().detach().to(torch::kCPU, torch::kFloat32);
  if (xi.dim() != 2 || xi.size(0) != n_feat || xi.size(1) != l) {
    throw std::runtime_error("unexpected xi shape");
  }
  out.xi.resize(static_cast<size_t>(n_feat * l));
  const float* src = xi.data_ptr<float>();
  for (int64_t f = 0; f < n_feat; ++f) {
    for (int64_t j = 0; j < l; ++j) {
      out.xi[static_cast<size_t>(f * l + j)] = src[f * l + j];
    }
  }
  return out;
}

std::string sindy_coefficients_csv_string(const LatentAeOde& model,
                                          const ModelConfig& cfg) {
  const auto table = build_sindy_coefficients_table(model, cfg);
  std::ostringstream oss;
  write_sindy_coefficients_csv(table, oss);
  return oss.str();
}

void write_sindy_coefficients_csv(const SindyCoefficientsTable& table,
                                  std::ostream& os) {
  os << "feature";
  for (const auto& col : table.latent_labels) {
    os << ',' << col;
  }
  os << '\n';
  const int64_t l = static_cast<int64_t>(table.latent_labels.size());
  for (int64_t f = 0; f < static_cast<int64_t>(table.features.size()); ++f) {
    os << table.features[static_cast<size_t>(f)];
    for (int64_t j = 0; j < l; ++j) {
      os << ',' << std::setprecision(9)
         << table.xi[static_cast<size_t>(f * l + j)];
    }
    os << '\n';
  }
}

}  // namespace eegexpcpp
