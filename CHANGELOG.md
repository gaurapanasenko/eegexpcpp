# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.4.0] - 2026-10-25

### Changed

- `TimeMlp::forward_jvp`: analytic JVP for Linear + gelu/relu/tanh;
  encoder/decoder use it instead of forward-mode autograd `jvp()`
- Streamlit admin: RK4 `ode_rhs` on CPU when forward uses CUDA;
  infer session cache keyed by `libeegexpcpp_infer.so` mtime; Plotly
  `width="stretch"`
- Infer: `dx` on same device as `x`; `model->to(device)` after
  checkpoint load

### Fixed

- CUDA infer forward when `dX` tensors stayed on CPU in the JVP path

## [0.3.0] - 2026-10-25

### Changed

- `LatentAeOde` built from `ModelConfig` only (`n_latent` in JSON,
  `feature_dim` from HDF5); encoder/decoder/ODE use separate F vs L
- `eegexpcpp_train` prints MLP stack and parameter counts before
  the epoch loop

### Added

- C++ `LatentAeOde`: encoder → ODE `f(Z)` → decoder `(Z, dZ_pred)`;
  `dZ` and `dY_pred` via forward-mode JVP (not extra MLP heads);
  losses `J_Y`, `J_dY` on `dY_pred`, `J_dZ`, `J_z_1`, L1 on encoder/
  decoder; AdamW training via `eegexpcpp_train` and JSON config
  (`configs/latent_ae_ode.lorenz.json`, `latent_ae_ode.s001_prepare.json`)
- HDF5 loader (`[F,T]` → tensor `[B,T,F]`); `nlohmann/json` via CMake
  `FetchContent` (not in git)
- C++ `eegexpcpp_test` (JVP, losses, HDF5, JSON); pytest
  `tests/test_latent_ae_ode.py`
- `scripts/train_latent_ae_ode.sh` — train launcher (`LD_LIBRARY_PATH`,
  HDF5 check, optional `--build`)
- Training resume: `next_epoch` from checkpoint + `epochs` per run
- Config `data.h5_group` replaces `subject`/`band`; Lorenz HDF5 script
  (256 Hz, ~60 s; groups `0..2` = x/y/z with `m=128`, `tau=1`)
- `scripts/cmake_build.sh` — Ninja configure, parallel build (`nproc`)
- Shared `libeegexpcpp_infer.so` (ctypes from Python); Streamlit admin
  `scripts/latent_ae_ode_admin.py` — reconstruction, latent 3D, RK4
  full/chunked per `configs/*.json`

## [0.2.0] - 2026-03-25

### Added

- Python package `eegexpcpp`: `calculus`, `signal`, `stft`, `prep`
- STFT/RFFT band decomposition with partition-of-unity filters and
  reconstruction tests
- HDF5 export: ICA components → bands → `prepare_embed_signal`; layout
  `{s}/{b}/` with `X`, `dX`, `X_mean`, `X_std`; band-sum check on
  export
- C++20 CMake target linked to local LibTorch; `uv` Python env
- `README.md`, `LICENSE` (LGPL-3.0-or-later), `AGENTS.md`, Cursor
  rules
- Example pipeline `scripts/save_s001_prepare.py` (`data/`, gitignored
  `.mat` / `.h5`)

### Changed

- Prep defaults in script: embedding `m=128`, `n_latent=12`, `tau=1`
