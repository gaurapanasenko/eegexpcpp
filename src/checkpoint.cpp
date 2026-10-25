#include "eegexpcpp/checkpoint.hpp"

#include <filesystem>

#include <torch/serialize.h>

namespace fs = std::filesystem;

namespace eegexpcpp {

std::string default_checkpoint_path(const std::string& checkpoint_dir) {
  return (fs::path(checkpoint_dir) / "latent_ae_ode.pt").string();
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

void save_train_checkpoint(const std::string& path,
                           const LatentAeOde& model,
                           const torch::optim::AdamW& optimizer,
                           int64_t next_epoch) {
  fs::create_directories(fs::path(path).parent_path());
  torch::serialize::OutputArchive archive(
      std::make_shared<torch::jit::CompilationUnit>());
  archive << model;
  archive << optimizer;
  archive.write("next_epoch", c10::IValue(next_epoch));
  archive.save_to(path);
}

}  // namespace eegexpcpp
