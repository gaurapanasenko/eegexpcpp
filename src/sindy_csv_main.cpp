#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include <torch/torch.h>

#include "eegexpcpp/checkpoint.hpp"
#include "eegexpcpp/config.hpp"
#include "eegexpcpp/dataset_h5.hpp"
#include "eegexpcpp/latent_ae_ode.hpp"
#include "eegexpcpp/sindy_table.hpp"

namespace {

namespace fs = std::filesystem;

void usage() {
  std::cerr << "usage: eegexpcpp_sindy_csv <config.json> [checkpoint.pt] "
               "[-o out.csv]\n";
}

void resolve_repo_relative(std::string& path, const fs::path& config_json_path) {
  fs::path p(path);
  if (p.is_absolute()) {
    return;
  }
  const fs::path repo_guess = config_json_path.parent_path().parent_path();
  const fs::path candidate = (repo_guess / p).lexically_normal();
  if (fs::exists(candidate)) {
    path = candidate.string();
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    usage();
    return 1;
  }
  std::string config_path = argv[1];
  std::string checkpoint_path;
  std::string out_path;
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "-o" && i + 1 < argc) {
      out_path = argv[++i];
    } else if (checkpoint_path.empty()) {
      checkpoint_path = arg;
    } else {
      usage();
      return 1;
    }
  }

  try {
    const fs::path cfg_abs = fs::absolute(config_path);
    auto cfg = eegexpcpp::load_config(cfg_abs.string());
    if (!eegexpcpp::model_ode_is_sindy(cfg.model)) {
      std::cerr << "config model.ode.type is not sindy\n";
      return 1;
    }
    resolve_repo_relative(cfg.data.h5_path, cfg_abs);
    resolve_repo_relative(cfg.train.checkpoint_dir, cfg_abs);
    if (checkpoint_path.empty()) {
      checkpoint_path =
          eegexpcpp::default_checkpoint_path(cfg.train.checkpoint_dir);
    } else {
      resolve_repo_relative(checkpoint_path, cfg_abs);
    }

    const auto [x_all, dx_all] =
        eegexpcpp::load_h5_pair_btf(cfg.data.h5_path, cfg.data.h5_group);
    eegexpcpp::ModelConfig model_cfg = cfg.model;
    model_cfg.feature_dim = x_all.size(2);

    eegexpcpp::LatentAeOde model(model_cfg);
    torch::optim::AdamW optimizer(model->parameters(),
                                  torch::optim::AdamWOptions(cfg.train.lr));
    const auto resume =
        eegexpcpp::try_resume_train(checkpoint_path, model, optimizer);
    if (!resume.resumed) {
      std::cerr << "cannot load checkpoint: " << checkpoint_path << '\n';
      return 1;
    }

    const auto table =
        eegexpcpp::build_sindy_coefficients_table(model, cfg.model);
    if (out_path.empty()) {
      eegexpcpp::write_sindy_coefficients_csv(table, std::cout);
    } else {
      std::ofstream out(out_path);
      if (!out) {
        std::cerr << "cannot open output: " << out_path << '\n';
        return 1;
      }
      eegexpcpp::write_sindy_coefficients_csv(table, out);
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
