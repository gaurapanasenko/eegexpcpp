#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <torch/torch.h>

namespace eegexpcpp {

struct DataConfig {
  std::string h5_path;
  std::string h5_group = "0";
  double train_fraction = 0.9;
};

struct ModelConfig {
  /// Latent dim Z; feature_dim is set from HDF5 before building the model.
  int64_t n_latent = 12;
  int64_t feature_dim = 0;
  std::vector<int64_t> encoder_hidden{256, 256};
  std::string encoder_activation = "gelu";
  std::vector<int64_t> ode_hidden{32, 32};
  std::string ode_activation = "tanh";
  std::vector<int64_t> decoder_hidden{256, 256};
  std::string decoder_activation = "gelu";
};

struct LossConfig {
  double lambda_Y = 1.0;
  double lambda_dY = 0.01;
  double lambda_dZ = 0.1;
  double lambda_z1 = 1.0;
  double lambda_l1 = 1e-5;
};

struct TrainConfig {
  int64_t epochs = 100;
  double lr = 1e-3;
  double weight_decay = 1e-2;
  double adamw_beta1 = 0.9;
  double adamw_beta2 = 0.999;
  double adamw_eps = 1e-8;
  double grad_clip_norm = 0.0;  // 0 = disabled
  std::string device = "cuda";
  int64_t seed = 42;
  int64_t log_every = 10;
  std::string checkpoint_dir = "checkpoints";
};

struct ExperimentConfig {
  DataConfig data;
  ModelConfig model;
  LossConfig loss;
  TrainConfig train;
};

ExperimentConfig load_config(const std::string& path);

torch::Device resolve_device(const std::string& name);

}  // namespace eegexpcpp
