#include "eegexpcpp/dataset_h5.hpp"

#include <stdexcept>
#include <string>
#include <vector>

#include <hdf5.h>

namespace eegexpcpp {

namespace {

torch::Tensor h5_dataset_to_btf(const std::string& h5_path,
                                const std::string& dset_path) {
  hid_t file = H5Fopen(h5_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file < 0) {
    throw std::runtime_error("H5Fopen failed: " + h5_path);
  }
  hid_t dset = H5Dopen2(file, dset_path.c_str(), H5P_DEFAULT);
  if (dset < 0) {
    H5Fclose(file);
    throw std::runtime_error("H5Dopen failed: " + dset_path);
  }
  hid_t space = H5Dget_space(dset);
  const int rank = H5Sget_simple_extent_ndims(space);
  if (rank != 2) {
    H5Sclose(space);
    H5Dclose(dset);
    H5Fclose(file);
    throw std::runtime_error("expected rank-2 dataset: " + dset_path);
  }
  hsize_t dims[2];
  H5Sget_simple_extent_dims(space, dims, nullptr);
  H5Sclose(space);

  const hsize_t f_dim = dims[0];
  const hsize_t t_dim = dims[1];
  std::vector<float> buf(f_dim * t_dim);
  if (H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
              buf.data()) < 0) {
    H5Dclose(dset);
    H5Fclose(file);
    throw std::runtime_error("H5Dread failed: " + dset_path);
  }
  H5Dclose(dset);
  H5Fclose(file);

  // Stored [F, T] (Julia dim x time) -> [B, T, F].
  auto t = torch::from_blob(buf.data(), {static_cast<int64_t>(f_dim),
                                         static_cast<int64_t>(t_dim)},
                            torch::TensorOptions().dtype(torch::kFloat32))
               .clone()
               .transpose(0, 1)
               .unsqueeze(0)
               .contiguous();
  return t;
}

std::string normalize_h5_group(const std::string& group) {
  std::string g = group;
  while (!g.empty() && g.front() == '/') {
    g.erase(g.begin());
  }
  while (!g.empty() && g.back() == '/') {
    g.pop_back();
  }
  if (g.empty()) {
    throw std::runtime_error("empty h5_group");
  }
  return "/" + g;
}

}  // namespace

std::pair<torch::Tensor, torch::Tensor> load_h5_pair_btf(
    const std::string& h5_path,
    const std::string& h5_group) {
  const std::string prefix = normalize_h5_group(h5_group);
  auto x = h5_dataset_to_btf(h5_path, prefix + "/X");
  auto dx = h5_dataset_to_btf(h5_path, prefix + "/dX");
  if (!x.sizes().equals(dx.sizes())) {
    throw std::runtime_error("X and dX shape mismatch in " + prefix);
  }
  return {x, dx};
}

std::pair<std::pair<torch::Tensor, torch::Tensor>,
          std::pair<torch::Tensor, torch::Tensor>>
split_train_val(torch::Tensor x,
                torch::Tensor dx,
                double train_fraction) {
  if (train_fraction <= 0.0 || train_fraction >= 1.0) {
    throw std::runtime_error("train_fraction must be in (0, 1)");
  }
  const int64_t t = x.size(1);
  const int64_t t_train = static_cast<int64_t>(
      std::floor(static_cast<double>(t) * train_fraction));
  if (t_train < 1 || t_train >= t) {
    throw std::runtime_error("train_fraction yields empty train or val split");
  }
  auto x_tr = x.index({torch::indexing::Slice(), torch::indexing::Slice(0, t_train),
                       torch::indexing::Slice()});
  auto dx_tr = dx.index({torch::indexing::Slice(), torch::indexing::Slice(0, t_train),
                         torch::indexing::Slice()});
  auto x_va =
      x.index({torch::indexing::Slice(), torch::indexing::Slice(t_train, t),
               torch::indexing::Slice()});
  auto dx_va =
      dx.index({torch::indexing::Slice(), torch::indexing::Slice(t_train, t),
                torch::indexing::Slice()});
  return {{x_tr, dx_tr}, {x_va, dx_va}};
}

}  // namespace eegexpcpp
