#pragma once

#include <cstdint>
#include <limits>
#include <optional>
#include <string>

#include <torch/torch.h>

#include "eegexpcpp/latent_ae_ode.hpp"
#include "eegexpcpp/train_history.hpp"

namespace eegexpcpp {

enum class CheckpointWeightSlot { Latest, BestTrain, BestVal };

CheckpointWeightSlot parse_checkpoint_weight_slot(const char* slot);

struct ResumeState {
  int64_t next_epoch = 0;
  bool resumed = false;
  bool optimizer_state = false;
};

/// Best model weights by train_J / val_J (no optimizer), stored in the .pt.
struct CheckpointBests {
  double best_train_j = std::numeric_limits<double>::infinity();
  int64_t best_train_epoch = -1;
  std::optional<LatentAeOde> best_train;

  double best_val_j = std::numeric_limits<double>::infinity();
  int64_t best_val_epoch = -1;
  std::optional<LatentAeOde> best_val;

  void consider_train(int64_t epoch,
                      double train_j,
                      const LatentAeOde& model,
                      const ModelConfig& model_cfg);
  void consider_val(int64_t epoch,
                   double val_j,
                   const LatentAeOde& model,
                   const ModelConfig& model_cfg);
};

std::string default_checkpoint_path(const std::string& checkpoint_dir);

ResumeState try_resume_train(const std::string& path,
                             LatentAeOde& model,
                             torch::optim::AdamW& optimizer);

void load_checkpoint_bests(const std::string& path,
                           CheckpointBests& bests,
                           const ModelConfig& model_cfg);

/// Load model weights for infer (best_* sub-archive, not latest).
bool load_checkpoint_infer_weights(const std::string& path,
                                   LatentAeOde& model,
                                   CheckpointWeightSlot slot,
                                   int64_t& next_epoch_out);

void save_train_checkpoint(const std::string& path,
                           const LatentAeOde& model,
                           const torch::optim::AdamW& optimizer,
                           int64_t next_epoch,
                           const TrainHistory& history,
                           const CheckpointBests& bests);

}  // namespace eegexpcpp
