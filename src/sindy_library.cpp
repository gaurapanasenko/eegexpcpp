#include "eegexpcpp/sindy_library.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace eegexpcpp {

namespace {

bool monomial_allowed(const std::vector<int>& exp,
                      int total_degree,
                      bool include_interaction,
                      bool interaction_only) {
  if (total_degree == 0) {
    return true;
  }
  int nonzero = 0;
  int max_e = 0;
  for (int e : exp) {
    if (e > 0) {
      ++nonzero;
    }
    max_e = std::max(max_e, e);
  }
  if (interaction_only) {
    return max_e <= 1 && nonzero == total_degree;
  }
  if (!include_interaction && nonzero > 1) {
    return false;
  }
  return true;
}

void generate_exponent_sums(int latent_dim,
                            int degree_left,
                            int idx,
                            std::vector<int>& current,
                            std::vector<std::vector<int>>& out) {
  if (idx == latent_dim) {
    if (degree_left == 0) {
      out.push_back(current);
    }
    return;
  }
  for (int e = degree_left; e >= 0; --e) {
    current[static_cast<size_t>(idx)] = e;
    generate_exponent_sums(latent_dim, degree_left - e, idx + 1, current,
                           out);
  }
}

std::vector<std::vector<int>> poly_monomial_exponents(
    int64_t latent_dim,
    const PolyLibraryConfig& cfg) {
  if (latent_dim <= 0) {
    throw std::runtime_error("poly library requires positive latent_dim");
  }
  const int L = static_cast<int>(latent_dim);
  std::vector<std::vector<int>> exponents;
  if (cfg.include_bias) {
    exponents.emplace_back(static_cast<size_t>(L), 0);
  }
  for (int deg = 1; deg <= cfg.degree; ++deg) {
    std::vector<int> current(static_cast<size_t>(L), 0);
    std::vector<std::vector<int>> raw;
    generate_exponent_sums(L, deg, 0, current, raw);
    for (const auto& exp : raw) {
      if (monomial_allowed(exp, deg, cfg.include_interaction,
                           cfg.interaction_only)) {
        exponents.push_back(exp);
      }
    }
  }
  return exponents;
}

std::string format_monomial_name(const std::vector<int>& exp) {
  std::ostringstream oss;
  bool any = false;
  for (int i = 0; i < static_cast<int>(exp.size()); ++i) {
    if (exp[static_cast<size_t>(i)] == 0) {
      continue;
    }
    if (any) {
      oss << '*';
    }
    any = true;
    const int e = exp[static_cast<size_t>(i)];
    oss << 'x' << i;
    if (e > 1) {
      oss << '^' << e;
    }
  }
  if (!any) {
    return "1";
  }
  return oss.str();
}

SindyFeatureFn make_monomial_eval(std::vector<int> exp) {
  return [exp = std::move(exp)](torch::Tensor z) -> torch::Tensor {
    const int64_t L = z.size(-1);
    torch::Tensor term =
        torch::ones({z.size(0), z.size(1)}, z.options().dtype(z.dtype()));
    for (int64_t i = 0; i < L; ++i) {
      const int e = exp[static_cast<size_t>(i)];
      if (e > 0) {
        term = term * z.select(-1, i).pow(e);
      }
    }
    return term.unsqueeze(-1);
  };
}

SindyFeatureFn make_fourier_eval(int64_t dim_index, int frequency, bool is_sin) {
  return [dim_index, frequency, is_sin](torch::Tensor z) -> torch::Tensor {
    const torch::Tensor zj = z.select(-1, dim_index);
    const float scale = static_cast<float>(frequency);
    torch::Tensor col =
        is_sin ? torch::sin(scale * zj) : torch::cos(scale * zj);
    return col.unsqueeze(-1);
  };
}

std::string format_fourier_name(int dim_index, int frequency, bool is_sin) {
  std::ostringstream oss;
  oss << (is_sin ? "sin(" : "cos(") << frequency << "*x" << dim_index << ')';
  return oss.str();
}

void append_poly_terms(int64_t latent_dim,
                       const PolyLibraryConfig& cfg,
                       std::vector<std::string>& names,
                       std::vector<SindyFeatureTerm>& terms) {
  for (const auto& exp : poly_monomial_exponents(latent_dim, cfg)) {
    SindyFeatureTerm term;
    term.name = format_monomial_name(exp);
    term.eval = make_monomial_eval(exp);
    names.push_back(term.name);
    terms.push_back(std::move(term));
  }
}

void append_fourier_terms(int64_t latent_dim,
                          const FourierLibraryConfig& cfg,
                          std::vector<std::string>& names,
                          std::vector<SindyFeatureTerm>& terms) {
  if (cfg.n_frequencies < 1) {
    throw std::runtime_error("fourier n_frequencies must be >= 1");
  }
  if (!cfg.include_sin && !cfg.include_cos) {
    throw std::runtime_error("fourier library needs include_sin or include_cos");
  }
  for (int64_t j = 0; j < latent_dim; ++j) {
    for (int k = 1; k <= cfg.n_frequencies; ++k) {
      if (cfg.include_sin) {
        SindyFeatureTerm term;
        term.name = format_fourier_name(static_cast<int>(j), k, true);
        term.eval = make_fourier_eval(j, k, true);
        names.push_back(term.name);
        terms.push_back(std::move(term));
      }
      if (cfg.include_cos) {
        SindyFeatureTerm term;
        term.name = format_fourier_name(static_cast<int>(j), k, false);
        term.eval = make_fourier_eval(j, k, false);
        names.push_back(term.name);
        terms.push_back(std::move(term));
      }
    }
  }
}

torch::Tensor eval_monomials(torch::Tensor z,
                             const std::vector<std::vector<int>>& exponents) {
  const int64_t L = z.size(-1);
  std::vector<torch::Tensor> cols;
  cols.reserve(exponents.size());
  for (const auto& exp : exponents) {
    torch::Tensor term =
        torch::ones({z.size(0), z.size(1)}, z.options().dtype(z.dtype()));
    for (int64_t i = 0; i < L; ++i) {
      const int e = exp[static_cast<size_t>(i)];
      if (e > 0) {
        term = term * z.select(-1, i).pow(e);
      }
    }
    cols.push_back(term.unsqueeze(-1));
  }
  if (cols.empty()) {
    return torch::empty({z.size(0), z.size(1), 0}, z.options());
  }
  return torch::cat(cols, -1);
}

}  // namespace

SindyFeatureLibrary::SindyFeatureLibrary(
    int64_t latent_dim,
    const std::vector<SindyLibraryConfig>& libraries) {
  if (latent_dim <= 0) {
    throw std::runtime_error("SindyFeatureLibrary needs positive latent_dim");
  }
  if (libraries.empty()) {
    throw std::runtime_error("SindyFeatureLibrary needs at least one library");
  }
  for (const auto& lib : libraries) {
    if (lib.kind == SindyLibraryConfig::Kind::Poly) {
      append_poly_terms(latent_dim, lib.poly, names_, terms_);
    } else {
      append_fourier_terms(latent_dim, lib.fourier, names_, terms_);
    }
  }
  if (terms_.empty()) {
    throw std::runtime_error("SindyFeatureLibrary produced zero features");
  }
}

torch::Tensor SindyFeatureLibrary::build_theta(torch::Tensor z) const {
  std::vector<torch::Tensor> cols;
  cols.reserve(terms_.size());
  for (const auto& term : terms_) {
    cols.push_back(term.eval(z));
  }
  if (cols.size() == 1) {
    return cols.front();
  }
  return torch::cat(cols, -1);
}

int64_t poly_feature_count(int64_t latent_dim, const PolyLibraryConfig& cfg) {
  return static_cast<int64_t>(poly_monomial_exponents(latent_dim, cfg).size());
}

int64_t fourier_feature_count(int64_t latent_dim,
                              const FourierLibraryConfig& cfg) {
  if (latent_dim <= 0) {
    throw std::runtime_error("fourier library requires positive latent_dim");
  }
  int per_freq = 0;
  if (cfg.include_sin) {
    ++per_freq;
  }
  if (cfg.include_cos) {
    ++per_freq;
  }
  if (per_freq == 0) {
    throw std::runtime_error("fourier library needs include_sin or include_cos");
  }
  if (cfg.n_frequencies < 1) {
    throw std::runtime_error("fourier n_frequencies must be >= 1");
  }
  return latent_dim * cfg.n_frequencies * per_freq;
}

torch::Tensor poly_features(torch::Tensor z, const PolyLibraryConfig& cfg) {
  const int64_t L = z.size(-1);
  return eval_monomials(z, poly_monomial_exponents(L, cfg));
}

torch::Tensor fourier_features(torch::Tensor z,
                               const FourierLibraryConfig& cfg) {
  std::vector<std::string> names;
  std::vector<SindyFeatureTerm> terms;
  append_fourier_terms(z.size(-1), cfg, names, terms);
  std::vector<torch::Tensor> cols;
  for (const auto& term : terms) {
    cols.push_back(term.eval(z));
  }
  return torch::cat(cols, -1);
}

int64_t sindy_library_feature_count(int64_t latent_dim,
                                    const SindyLibraryConfig& cfg) {
  if (cfg.kind == SindyLibraryConfig::Kind::Poly) {
    return poly_feature_count(latent_dim, cfg.poly);
  }
  return fourier_feature_count(latent_dim, cfg.fourier);
}

torch::Tensor sindy_library_features(torch::Tensor z,
                                     const SindyLibraryConfig& cfg) {
  if (cfg.kind == SindyLibraryConfig::Kind::Poly) {
    return poly_features(std::move(z), cfg.poly);
  }
  return fourier_features(std::move(z), cfg.fourier);
}

int64_t sindy_total_feature_count(
    int64_t latent_dim,
    const std::vector<SindyLibraryConfig>& libraries) {
  return SindyFeatureLibrary(latent_dim, libraries).size();
}

torch::Tensor sindy_build_theta(
    torch::Tensor z,
    const std::vector<SindyLibraryConfig>& libraries) {
  const SindyFeatureLibrary lib(z.size(-1), libraries);
  return lib.build_theta(std::move(z));
}

}  // namespace eegexpcpp
