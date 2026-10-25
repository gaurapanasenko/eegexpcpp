#include "eegexpcpp/config.hpp"

#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace eegexpcpp {

namespace {

std::vector<int64_t> read_int64_array(const nlohmann::json& j,
                                      const char* key) {
  std::vector<int64_t> out;
  for (const auto& v : j.at(key)) {
    out.push_back(v.get<int64_t>());
  }
  return out;
}

}  // namespace

ExperimentConfig load_config(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("cannot open config: " + path);
  }
  nlohmann::json j;
  in >> j;

  ExperimentConfig cfg;
  const auto& jd = j.at("data");
  cfg.data.h5_path = jd.at("h5_path").get<std::string>();
  cfg.data.h5_group = jd.at("h5_group").get<std::string>();
  cfg.data.train_fraction = jd.value("train_fraction", cfg.data.train_fraction);
  const auto& jm = j.at("model");
  cfg.model.n_latent = jm.value("n_latent", cfg.model.n_latent);
  if (jm.contains("encoder_hidden")) {
    cfg.model.encoder_hidden = read_int64_array(jm, "encoder_hidden");
  }
  cfg.model.encoder_activation =
      jm.value("encoder_activation", cfg.model.encoder_activation);
  if (jm.contains("ode_hidden")) {
    cfg.model.ode_hidden = read_int64_array(jm, "ode_hidden");
  }
  cfg.model.ode_activation =
      jm.value("ode_activation", cfg.model.ode_activation);
  if (jm.contains("decoder_hidden")) {
    cfg.model.decoder_hidden = read_int64_array(jm, "decoder_hidden");
  }
  cfg.model.decoder_activation =
      jm.value("decoder_activation", cfg.model.decoder_activation);

  const auto& jl = j.at("loss");
  cfg.loss.lambda_Y = jl.value("lambda_Y", cfg.loss.lambda_Y);
  cfg.loss.lambda_dY = jl.value("lambda_dY", cfg.loss.lambda_dY);
  cfg.loss.lambda_dZ = jl.value("lambda_dZ", cfg.loss.lambda_dZ);
  cfg.loss.lambda_z1 = jl.value("lambda_z1", cfg.loss.lambda_z1);
  cfg.loss.lambda_l1 = jl.value("lambda_l1", cfg.loss.lambda_l1);

  const auto& jt = j.at("train");
  cfg.train.epochs = jt.value("epochs", cfg.train.epochs);
  cfg.train.lr = jt.value("lr", cfg.train.lr);
  cfg.train.weight_decay = jt.value("weight_decay", cfg.train.weight_decay);
  if (jt.contains("adamw_betas")) {
    const auto& b = jt.at("adamw_betas");
    cfg.train.adamw_beta1 = b.at(0).get<double>();
    cfg.train.adamw_beta2 = b.at(1).get<double>();
  }
  cfg.train.adamw_eps = jt.value("adamw_eps", cfg.train.adamw_eps);
  cfg.train.grad_clip_norm =
      jt.value("grad_clip_norm", cfg.train.grad_clip_norm);
  cfg.train.device = jt.value("device", cfg.train.device);
  cfg.train.seed = jt.value("seed", cfg.train.seed);
  cfg.train.log_every = jt.value("log_every", cfg.train.log_every);
  cfg.train.checkpoint_dir =
      jt.value("checkpoint_dir", cfg.train.checkpoint_dir);

  return cfg;
}

torch::Device resolve_device(const std::string& name) {
  if (name == "cuda" && torch::cuda::is_available()) {
    return torch::Device(torch::kCUDA);
  }
  return torch::Device(torch::kCPU);
}

}  // namespace eegexpcpp
