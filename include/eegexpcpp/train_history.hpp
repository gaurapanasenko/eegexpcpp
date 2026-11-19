#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "eegexpcpp/config.hpp"

namespace eegexpcpp {

constexpr int64_t kTrainHistoryVersion = 1;
constexpr int64_t kTrainHistoryMetricCols = 7;

/// Log rows for [epoch_begin, epoch_end) with train log_every policy.
int64_t planned_log_rows(int64_t epoch_begin,
                         int64_t epoch_end,
                         int64_t log_every);

class TrainHistory {
 public:
  static constexpr std::array<const char*, kTrainHistoryMetricCols>
      kMetricNames = {"epoch",     "train_J", "val_J", "J_Y",
                      "J_dY",      "J_dZ",    "J_z_1"};

  void load_from_archive(torch::serialize::InputArchive& archive);
  void load_from_checkpoint_file(const std::string& path);

  void begin_run(int64_t epoch_begin,
                 int64_t epoch_end,
                 int64_t log_every,
                 const TrainConfig& train,
                 const LossConfig& loss);

  void append_loss_row(int64_t epoch,
                       double train_j,
                       double val_j,
                       double j_y,
                       double j_dy,
                       double j_dz,
                       double j_z1);

  void append_xi_prune(int64_t epoch, int64_t nnz);

  void append_xi_weak_prune(int64_t epoch,
                            int64_t nnz,
                            double train_j,
                            double best_train_j);

  /// Rows already stored from prior checkpoint loads (not including current run).
  int64_t committed_metric_rows() const;

  torch::Tensor metrics_for_save() const;
  std::string events_json_for_save() const;

  void write_to_archive(torch::serialize::OutputArchive& archive) const;

  static constexpr int64_t metric_cols() { return kTrainHistoryMetricCols; }

 private:
  torch::Tensor committed_metrics_;  // [N, 7] float64
  std::string committed_events_json_ = "[]";
  torch::Tensor run_buf_;            // [cap, 7] float64
  int64_t run_filled_ = 0;
  std::vector<std::string> run_event_json_lines_;
};

bool read_checkpoint_train_history(const std::string& path,
                                   torch::Tensor& metrics_out,
                                   std::string& events_json_out);

}  // namespace eegexpcpp