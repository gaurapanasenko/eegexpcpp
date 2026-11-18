#include "eegexpcpp/train_history.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace eegexpcpp {

namespace {

nlohmann::json train_to_json(const TrainConfig& t) {
  return nlohmann::json{{"epochs", t.epochs},
                        {"lr", t.lr},
                        {"weight_decay", t.weight_decay},
                        {"adamw_beta1", t.adamw_beta1},
                        {"adamw_beta2", t.adamw_beta2},
                        {"adamw_eps", t.adamw_eps},
                        {"grad_clip_norm", t.grad_clip_norm},
                        {"device", t.device},
                        {"seed", t.seed},
                        {"log_every", t.log_every},
                        {"checkpoint_dir", t.checkpoint_dir},
                        {"xi_prune_every", t.xi_prune.prune_every},
                        {"xi_threshold", t.xi_prune.threshold}};
}

nlohmann::json loss_to_json(const LossConfig& l) {
  return nlohmann::json{{"lambda_Y", l.lambda_Y},
                        {"lambda_dY", l.lambda_dY},
                        {"lambda_dZ", l.lambda_dZ},
                        {"lambda_z1", l.lambda_z1},
                        {"lambda_l1_ae", l.lambda_l1_ae},
                        {"lambda_l1_ode", l.lambda_l1_ode}};
}

torch::Tensor empty_metrics() {
  return torch::empty({0, kTrainHistoryMetricCols}, torch::kFloat64);
}

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

}  // namespace

int64_t planned_log_rows(int64_t epoch_begin,
                         int64_t epoch_end,
                         int64_t log_every) {
  if (epoch_end <= epoch_begin) {
    return 0;
  }
  if (log_every <= 0) {
    throw std::runtime_error("log_every must be positive");
  }
  int64_t n = 0;
  for (int64_t e = epoch_begin; e < epoch_end; ++e) {
    if (e % log_every == 0 || e == epoch_end - 1) {
      ++n;
    }
  }
  return n;
}

void TrainHistory::load_from_archive(torch::serialize::InputArchive& archive) {
  committed_metrics_ = empty_metrics();
  committed_events_json_ = "[]";
  run_buf_ = empty_metrics();
  run_filled_ = 0;
  run_event_json_lines_.clear();

  c10::IValue metrics_iv;
  if (read_ivalue_key(archive, "train_history_metrics", metrics_iv)) {
    auto t = metrics_iv.toTensor();
    if (t.defined() && t.numel() > 0) {
      committed_metrics_ = t.to(torch::kFloat64).contiguous().cpu();
      if (committed_metrics_.dim() != 2 ||
          committed_metrics_.size(1) != kTrainHistoryMetricCols) {
        throw std::runtime_error("train_history_metrics has invalid shape");
      }
    }
  }

  c10::IValue events_iv;
  if (read_ivalue_key(archive, "train_history_events", events_iv) &&
      events_iv.isString()) {
    committed_events_json_ = events_iv.toStringRef();
  }
}

void TrainHistory::load_from_checkpoint_file(const std::string& path) {
  torch::serialize::InputArchive archive;
  archive.load_from(path);
  load_from_archive(archive);
}

void TrainHistory::begin_run(int64_t epoch_begin,
                             int64_t epoch_end,
                             int64_t log_every,
                             const TrainConfig& train,
                             const LossConfig& loss) {
  run_filled_ = 0;
  run_event_json_lines_.clear();

  const int64_t cap = planned_log_rows(epoch_begin, epoch_end, log_every);
  if (cap > 0) {
    run_buf_ = torch::full(
        {cap, kTrainHistoryMetricCols}, std::nan(""),
        torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
  } else {
    run_buf_ = empty_metrics();
  }

  nlohmann::json ev;
  ev["type"] = "run_start";
  ev["epoch_begin"] = epoch_begin;
  ev["epoch_end"] = epoch_end;
  ev["train"] = train_to_json(train);
  ev["loss"] = loss_to_json(loss);
  run_event_json_lines_.push_back(ev.dump());
}

void TrainHistory::append_loss_row(int64_t epoch,
                                   double train_j,
                                   double val_j,
                                   double j_y,
                                   double j_dy,
                                   double j_dz,
                                   double j_z1) {
  if (!run_buf_.defined() || run_filled_ >= run_buf_.size(0)) {
    throw std::runtime_error("train history run buffer overflow");
  }
  const double vals[kTrainHistoryMetricCols] = {
      static_cast<double>(epoch), train_j, val_j, j_y, j_dy, j_dz, j_z1};
  std::memcpy(run_buf_[run_filled_].data_ptr<double>(), vals, sizeof(vals));
  ++run_filled_;
}

void TrainHistory::append_xi_prune(int64_t epoch, int64_t nnz) {
  nlohmann::json ev;
  ev["type"] = "xi_prune";
  ev["epoch"] = epoch;
  ev["nnz"] = nnz;
  run_event_json_lines_.push_back(ev.dump());
}

int64_t TrainHistory::committed_metric_rows() const {
  return committed_metrics_.defined() ? committed_metrics_.size(0) : 0;
}

torch::Tensor TrainHistory::metrics_for_save() const {
  torch::Tensor part = empty_metrics();
  if (run_buf_.defined() && run_filled_ > 0) {
    part = run_buf_.narrow(0, 0, run_filled_).contiguous();
  }
  if ((!committed_metrics_.defined() || committed_metrics_.size(0) == 0) &&
      part.size(0) == 0) {
    return empty_metrics();
  }
  if (!committed_metrics_.defined() || committed_metrics_.size(0) == 0) {
    return part;
  }
  if (part.size(0) == 0) {
    return committed_metrics_;
  }
  return torch::cat({committed_metrics_, part}, 0);
}

std::string TrainHistory::events_json_for_save() const {
  nlohmann::json merged = nlohmann::json::array();
  if (!committed_events_json_.empty()) {
    merged = nlohmann::json::parse(committed_events_json_);
    if (!merged.is_array()) {
      merged = nlohmann::json::array();
    }
  }
  for (const auto& line : run_event_json_lines_) {
    merged.push_back(nlohmann::json::parse(line));
  }
  return merged.dump();
}

void TrainHistory::write_to_archive(
    torch::serialize::OutputArchive& archive) const {
  archive.write("train_history_version",
                c10::IValue(kTrainHistoryVersion));
  archive.write("train_history_metrics", c10::IValue(metrics_for_save()));
  archive.write("train_history_events", c10::IValue(events_json_for_save()));
}

bool read_checkpoint_train_history(const std::string& path,
                                   torch::Tensor& metrics_out,
                                   std::string& events_json_out) {
  if (!std::ifstream(path).good()) {
    return false;
  }
  try {
    TrainHistory hist;
    hist.load_from_checkpoint_file(path);
    metrics_out = hist.metrics_for_save();
    events_json_out = hist.events_json_for_save();
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

}  // namespace eegexpcpp