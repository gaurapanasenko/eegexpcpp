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
  const auto& jo = jm.at("ode");
  const std::string ode_type = jo.at("type").get<std::string>();
  if (ode_type == "nn") {
    cfg.model.ode.type = OdeType::Nn;
    if (jo.contains("hidden")) {
      cfg.model.ode.nn.hidden = read_int64_array(jo, "hidden");
    }
    cfg.model.ode.nn.activation =
        jo.value("activation", cfg.model.ode.nn.activation);
  } else if (ode_type == "sindy") {
    cfg.model.ode.type = OdeType::Sindy;
    for (const auto& jl : jo.at("libraries")) {
      SindyLibraryConfig lib;
      const std::string lib_type = jl.at("type").get<std::string>();
      if (lib_type == "poly") {
        lib.kind = SindyLibraryConfig::Kind::Poly;
        lib.poly.degree = jl.value("degree", lib.poly.degree);
        lib.poly.include_interaction =
            jl.value("include_interaction", lib.poly.include_interaction);
        lib.poly.interaction_only =
            jl.value("interaction_only", lib.poly.interaction_only);
        lib.poly.include_bias =
            jl.value("include_bias", lib.poly.include_bias);
      } else if (lib_type == "fourier") {
        lib.kind = SindyLibraryConfig::Kind::Fourier;
        lib.fourier.n_frequencies =
            jl.value("n_frequencies", lib.fourier.n_frequencies);
        lib.fourier.include_sin =
            jl.value("include_sin", lib.fourier.include_sin);
        lib.fourier.include_cos =
            jl.value("include_cos", lib.fourier.include_cos);
      } else {
        throw std::runtime_error("unknown sindy library type: " + lib_type);
      }
      cfg.model.ode.libraries.push_back(lib);
    }
    if (cfg.model.ode.libraries.empty()) {
      throw std::runtime_error("sindy ode requires non-empty libraries");
    }
  } else {
    throw std::runtime_error("unknown ode type: " + ode_type);
  }
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
