#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Data split for forward (time axis).
enum EegexpcppInferSplit {
  EEGEXPCPP_INFER_TRAIN = 0,
  EEGEXPCPP_INFER_VAL = 1,
  EEGEXPCPP_INFER_FULL = 2,
};

typedef struct EegexpcppInferHandle EegexpcppInferHandle;

/// Create session from experiment JSON. checkpoint_path NULL uses
/// default_checkpoint_path(train.checkpoint_dir). device: "cpu" or "cuda".
/// Returns NULL on failure; see eegexpcpp_infer_last_error().
EegexpcppInferHandle* eegexpcpp_infer_create(const char* config_json_path,
                                             const char* checkpoint_path,
                                             const char* device);

void eegexpcpp_infer_destroy(EegexpcppInferHandle* handle);

const char* eegexpcpp_infer_last_error(void);

/// Checkpoint next_epoch (-1 if weights-only legacy load without field).
int64_t eegexpcpp_infer_next_epoch(const EegexpcppInferHandle* handle);

/// Time length T and input feature dim F for the chosen split.
int eegexpcpp_infer_sizes(const EegexpcppInferHandle* handle,
                          int split,
                          int64_t* t_out,
                          int64_t* f_out);

/// Latent dimension L (model n_latent).
int64_t eegexpcpp_infer_n_latent(const EegexpcppInferHandle* handle);

/// Row-major float32: x,y length T*F; z,dz,dz_pred length T*L (L=n_latent).
int eegexpcpp_infer_forward(const EegexpcppInferHandle* handle,
                            int split,
                            float* x_out,
                            float* y_out,
                            float* z_out,
                            float* dz_out,
                            float* dz_pred_out);

/// dz_out = ode(z), vectors length L (n_latent).
int eegexpcpp_infer_ode_rhs(const EegexpcppInferHandle* handle,
                            const float* z_in,
                            float* dz_out,
                            int64_t f_dim);

/// RK4 integrate ode(z) from z0 for n_steps with step dt.
/// z_out row-major float32 [n_steps * n_latent]; z0 length n_latent.
int eegexpcpp_infer_rk4_latent(const EegexpcppInferHandle* handle,
                               int64_t n_steps,
                               float dt,
                               const float* z0,
                               float* z_out);

#ifdef __cplusplus
}
#endif
