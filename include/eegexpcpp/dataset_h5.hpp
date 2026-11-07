#pragma once

#include <optional>
#include <string>
#include <utility>

#include <torch/data/datasets/base.h>
#include <torch/torch.h>

namespace eegexpcpp {

/// Load X, dX from HDF5 group (e.g. `0` or `/lorenz/0`) as [1, T, F].
std::pair<torch::Tensor, torch::Tensor> load_h5_pair_btf(
    const std::string& h5_path,
    const std::string& h5_group);

/// Split along time (dim=1) into train and val tensors.
std::pair<std::pair<torch::Tensor, torch::Tensor>,
          std::pair<torch::Tensor, torch::Tensor>>
split_train_val(torch::Tensor x,
                torch::Tensor dx,
                double train_fraction);

/// In-memory time rows. `x` and `dx` are `[T, F]`. `get(i)` is that row.
class TimePairDataset
    : public torch::data::datasets::Dataset<TimePairDataset> {
 public:
  TimePairDataset(torch::Tensor x, torch::Tensor dx);

  torch::data::Example<> get(size_t index) override;
  std::optional<size_t> size() const override;

 private:
  torch::Tensor x_;
  torch::Tensor dx_;
};

}  // namespace eegexpcpp
