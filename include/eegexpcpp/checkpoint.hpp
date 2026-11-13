#pragma once

#include <cstdint>
#include <string>

#include <torch/torch.h>

#include "eegexpcpp/latent_ae_ode.hpp"

namespace eegexpcpp {

struct ResumeState {
  int64_t next_epoch = 0;
  bool resumed = false;
  bool optimizer_state = false;
};

std::string default_checkpoint_path(const std::string& checkpoint_dir);

ResumeState try_resume_train(const std::string& path,
                             LatentAeOde& model,
                             torch::optim::AdamW& optimizer);

void save_train_checkpoint(const std::string& path,
                           const LatentAeOde& model,
                           const torch::optim::AdamW& optimizer,
                           int64_t next_epoch);

}  // namespace eegexpcpp
