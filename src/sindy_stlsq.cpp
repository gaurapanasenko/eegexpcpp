#include "eegexpcpp/sindy_stlsq.hpp"

#include <stdexcept>
#include <vector>

namespace eegexpcpp {

namespace {

torch::Tensor lstsq_columns(torch::Tensor theta,
                            torch::Tensor y,
                            const std::vector<int64_t>& cols,
                            double ridge) {
  const auto out_opts = theta.options().dtype(torch::kFloat32);
  if (cols.empty()) {
    return torch::zeros({theta.size(1)}, out_opts);
  }
  auto idx =
      torch::tensor(cols, torch::TensorOptions().dtype(torch::kInt64));
  auto t = theta.to(torch::kCPU, torch::kFloat64);
  auto y_vec = y.to(torch::kCPU, torch::kFloat64).reshape({-1});
  if (y_vec.dim() != 1) {
    throw std::runtime_error("stlsq lstsq: y must be a single column [N]");
  }
  if (y_vec.size(0) != t.size(0)) {
    throw std::runtime_error("stlsq lstsq: theta/y row count mismatch");
  }
  auto yt = y_vec.unsqueeze(1);
  auto sub = t.index_select(1, idx);
  const int64_t k = sub.size(1);
  auto gram = sub.transpose(0, 1).mm(sub);
  if (ridge > 0.0) {
    gram = gram + ridge * torch::eye(k, gram.options());
  }
  auto rhs = sub.transpose(0, 1).mm(yt);
  auto solved = torch::linalg_solve(gram, rhs);
  if (solved.dim() != 2 || solved.size(0) != k || solved.size(1) != 1) {
    throw std::runtime_error("stlsq lstsq: expected single-target regression");
  }
  auto xi_sub = solved.squeeze(1).flatten();
  if (xi_sub.dim() != 1 || xi_sub.size(0) != k) {
    throw std::runtime_error("stlsq lstsq: unexpected coefficient shape");
  }
  auto xi = torch::zeros({t.size(1)}, xi_sub.options());
  for (size_t i = 0; i < cols.size(); ++i) {
    xi.index_put_(
        {cols[i]},
        torch::scalar_tensor(
            xi_sub[static_cast<int64_t>(i)].item<double>(), xi.options()));
  }
  return xi.to(out_opts).reshape({t.size(1)});
}

std::vector<int64_t> active_support(const torch::Tensor& xi, double threshold) {
  std::vector<int64_t> cols;
  auto x = xi.detach().to(torch::kCPU).flatten();
  if (x.dim() != 1) {
    throw std::runtime_error("stlsq active_support expects 1d xi column");
  }
  for (int64_t k = 0; k < x.size(0); ++k) {
    const double v = x[k].item<double>();
    if (v >= threshold || v <= -threshold) {
      cols.push_back(k);
    }
  }
  return cols;
}

torch::Tensor stlsq_one_column(torch::Tensor theta,
                               torch::Tensor y,
                               double threshold,
                               int max_iter,
                               double ridge) {
  const int64_t n_feat = theta.size(1);
  std::vector<int64_t> all_cols;
  all_cols.reserve(static_cast<size_t>(n_feat));
  for (int64_t k = 0; k < n_feat; ++k) {
    all_cols.push_back(k);
  }
  torch::Tensor xi = lstsq_columns(theta, y, all_cols, ridge);
  for (int iter = 0; iter < max_iter; ++iter) {
    auto support = active_support(xi, threshold);
    if (support.empty()) {
      return torch::zeros({n_feat}, theta.options().dtype(torch::kFloat32));
    }
    torch::Tensor next = lstsq_columns(theta, y, support, ridge);
    bool same = true;
    if (next.size(0) != xi.size(0)) {
      same = false;
    } else {
      same = torch::allclose(next, xi, /*rtol=*/0.0, /*atol=*/0.0);
    }
    xi = next;
    if (same) {
      break;
    }
  }
  return xi;
}

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

SindyStlsqFitResult sindy_stlsq_fit(torch::Tensor theta,
                                    torch::Tensor dz,
                                    double threshold,
                                    int max_iter,
                                    double ridge) {
  if (theta.dim() != 2 || dz.dim() != 2) {
    throw std::runtime_error("stlsq expects theta [N,F] and dz [N,L]");
  }
  if (theta.size(0) != dz.size(0)) {
    throw std::runtime_error("stlsq theta/dz row count mismatch");
  }
  if (theta.size(1) == 0 || dz.size(1) == 0) {
    throw std::runtime_error("stlsq empty feature or latent dim");
  }
  if (max_iter <= 0) {
    throw std::runtime_error("stlsq max_iter must be positive");
  }
  if (threshold < 0.0) {
    throw std::runtime_error("stlsq threshold must be non-negative");
  }
  if (ridge < 0.0) {
    throw std::runtime_error("stlsq ridge must be non-negative");
  }

  const int64_t n_latent = dz.size(1);
  std::vector<torch::Tensor> columns;
  columns.reserve(static_cast<size_t>(n_latent));
  theta = theta.detach();
  dz = dz.detach();
  for (int64_t j = 0; j < n_latent; ++j) {
    auto y_col = dz.select(1, j).reshape({-1});
    columns.push_back(
        stlsq_one_column(theta, y_col, threshold, max_iter, ridge));
  }
  SindyStlsqFitResult out;
  out.xi = torch::stack(columns, 1);
  out.nnz = matrix_nnz(out.xi, threshold);
  return out;
}

}  // namespace eegexpcpp
