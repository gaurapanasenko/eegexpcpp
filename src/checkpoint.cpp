#include "eegexpcpp/checkpoint.hpp"

#include <cstring>
#include <filesystem>

#include <torch/optim/adamw.h>
#include <torch/serialize.h>

namespace fs = std::filesystem;

namespace eegexpcpp {

namespace {

bool read_ivalue_key(torch::serialize::InputArchive& archive,
                     const char* key,
                     c10::IValue& out) {
  try {
    archive.read(key, out);
    return true;
  } catch (const c10::Error&) {
    return false;
  }
}

bool read_subarchive(torch::serialize::InputArchive& archive,
                     const char* key,
                     torch::serialize::InputArchive& sub) {
  try {
    archive.read(key, sub);
    return true;
  } catch (const c10::Error&) {
    return false;
  }
}

void copy_latent_ae_ode_weights(const LatentAeOde& src, LatentAeOde& dst) {
  torch::NoGradGuard no_grad;
  for (const auto& p : src->named_parameters()) {
    dst->named_parameters()[p.key()].copy_(p.value());
  }
  for (const auto& b : src->named_buffers()) {
    dst->named_buffers()[b.key()].copy_(b.value());
  }
}

void write_best_module(torch::serialize::OutputArchive& archive,
                       const char* key,
                       const LatentAeOde& model) {
  torch::serialize::OutputArchive sub(
      std::make_shared<torch::jit::CompilationUnit>());
  sub << model;
  archive.write(key, sub);
}

}  // namespace

std::string default_checkpoint_path(const std::string& checkpoint_dir) {
  return (fs::path(checkpoint_dir) / "latent_ae_ode.pt").string();
}

CheckpointWeightSlot parse_checkpoint_weight_slot(const char* slot) {
  if (slot == nullptr || slot[0] == '\0' ||
      std::strcmp(slot, "latest") == 0) {
    return CheckpointWeightSlot::Latest;
  }
  if (std::strcmp(slot, "best_train") == 0) {
    return CheckpointWeightSlot::BestTrain;
  }
  if (std::strcmp(slot, "best_val") == 0) {
    return CheckpointWeightSlot::BestVal;
  }
  throw std::runtime_error("unknown checkpoint weight slot");
}

bool load_checkpoint_infer_weights(const std::string& path,
                                   LatentAeOde& model,
                                   CheckpointWeightSlot slot,
                                   int64_t& next_epoch_out) {
  if (!fs::exists(path)) {
    return false;
  }
  const char* key =
      slot == CheckpointWeightSlot::BestTrain ? "best_train" : "best_val";
  try {
    torch::serialize::InputArchive archive;
    archive.load_from(path);

    next_epoch_out = -1;
    c10::IValue iv;
    if (read_ivalue_key(archive, "next_epoch", iv)) {
      next_epoch_out = iv.toInt();
    }

    torch::serialize::InputArchive sub;
    if (!read_subarchive(archive, key, sub)) {
      return false;
    }
    sub >> model;
    return true;
  } catch (const c10::Error&) {
    return false;
  }
}

ResumeState try_resume_train(const std::string& path,
                             LatentAeOde& model,
                             torch::optim::AdamW& optimizer) {
  ResumeState state;
  if (!fs::exists(path)) {
    return state;
  }

  try {
    torch::serialize::InputArchive archive;
    archive.load_from(path);
    archive >> model;
    archive >> optimizer;
    try {
      c10::IValue next_iv;
      archive.read("next_epoch", next_iv);
      state.next_epoch = next_iv.toInt();
    } catch (const c10::Error&) {
      state.next_epoch = 0;
    }
    state.resumed = true;
    state.optimizer_state = true;
    return state;
  } catch (const c10::Error&) {
    // Legacy checkpoint: model weights only (torch::save(model)).
  }

  torch::load(model, path);
  state.resumed = true;
  state.optimizer_state = false;
  return state;
}

void CheckpointBests::consider_train(int64_t epoch,
                                     double train_j,
                                     const LatentAeOde& model,
                                     const ModelConfig& model_cfg) {
  if (!(train_j < best_train_j)) {
    return;
  }
  best_train_j = train_j;
  best_train_epoch = epoch;
  if (!best_train) {
    best_train = LatentAeOde(model_cfg);
  }
  copy_latent_ae_ode_weights(model, *best_train);
}

void CheckpointBests::consider_val(int64_t epoch,
                                 double val_j,
                                 const LatentAeOde& model,
                                 const ModelConfig& model_cfg) {
  if (!(val_j < best_val_j)) {
    return;
  }
  best_val_j = val_j;
  best_val_epoch = epoch;
  if (!best_val) {
    best_val = LatentAeOde(model_cfg);
  }
  copy_latent_ae_ode_weights(model, *best_val);
}

void load_checkpoint_bests(const std::string& path,
                           CheckpointBests& bests,
                           const ModelConfig& model_cfg) {
  if (!fs::exists(path)) {
    return;
  }
  try {
    torch::serialize::InputArchive archive;
    archive.load_from(path);

    c10::IValue iv;
    if (read_ivalue_key(archive, "best_train_J", iv)) {
      bests.best_train_j = iv.toDouble();
    }
    if (read_ivalue_key(archive, "best_train_epoch", iv)) {
      bests.best_train_epoch = iv.toInt();
    }
    if (read_ivalue_key(archive, "best_val_J", iv)) {
      bests.best_val_j = iv.toDouble();
    }
    if (read_ivalue_key(archive, "best_val_epoch", iv)) {
      bests.best_val_epoch = iv.toInt();
    }

    {
      torch::serialize::InputArchive sub;
      if (read_subarchive(archive, "best_train", sub)) {
        bests.best_train = LatentAeOde(model_cfg);
        sub >> *bests.best_train;
      }
    }
    {
      torch::serialize::InputArchive sub;
      if (read_subarchive(archive, "best_val", sub)) {
        bests.best_val = LatentAeOde(model_cfg);
        sub >> *bests.best_val;
      }
    }
  } catch (const c10::Error&) {
    // Legacy checkpoint without best snapshots.
  }
}

void save_train_checkpoint(const std::string& path,
                           const LatentAeOde& model,
                           const torch::optim::AdamW& optimizer,
                           int64_t next_epoch,
                           const TrainHistory& history,
                           const CheckpointBests& bests) {
  fs::create_directories(fs::path(path).parent_path());
  torch::serialize::OutputArchive archive(
      std::make_shared<torch::jit::CompilationUnit>());
  archive << model;
  archive << optimizer;
  archive.write("next_epoch", c10::IValue(next_epoch));
  history.write_to_archive(archive);

  if (bests.best_train) {
    archive.write("best_train_J", c10::IValue(bests.best_train_j));
    archive.write("best_train_epoch", c10::IValue(bests.best_train_epoch));
    write_best_module(archive, "best_train", *bests.best_train);
  }
  if (bests.best_val) {
    archive.write("best_val_J", c10::IValue(bests.best_val_j));
    archive.write("best_val_epoch", c10::IValue(bests.best_val_epoch));
    write_best_module(archive, "best_val", *bests.best_val);
  }

  archive.save_to(path);
}

}  // namespace eegexpcpp
