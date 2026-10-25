# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

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
