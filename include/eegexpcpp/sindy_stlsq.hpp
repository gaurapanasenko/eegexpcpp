#pragma once

#include <cstdint>

#include <torch/torch.h>

namespace eegexpcpp {

struct SindyStlsqFitResult {
  torch::Tensor xi;
  int64_t nnz = 0;
};

/// Theta [N,F], dz [N,L] -> Xi [F,L] via sequential thresholded least squares.
SindyStlsqFitResult sindy_stlsq_fit(torch::Tensor theta,
                                    torch::Tensor dz,
                                    double threshold,
                                    int max_iter,
                                    double ridge);

}  // namespace eegexpcpp
