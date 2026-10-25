# Agent instructions (eegexpcpp)

## Communication

- Reply to the project owner in **Ukrainian**.
- **Code, comments, commit messages, and identifiers** stay in
  **English**.

## Formatting

- **Markdown** (`*.md`, including this file): wrap prose at **72
  characters** per line. Code blocks and URLs may exceed 72 when
  breaking would hurt copy-paste or readability.
- **Git commit messages**: subject and body wrapped at **72
  characters** per line; subject still one short line when possible.

## Ponytail (default: full)

Lazy senior dev mode: shortest correct diff, YAGNI, reuse before
write, stdlib/platform/deps before new code. Bug fixes at the root
(shared path), not symptom patches in one caller.

Intensity: `lite` | `full` (default) | `ultra`. Off only if the user
says `stop ponytail` / `normal mode`.

## Project knowledge (update this file)

Anything **global** to the repo (build layout, dependencies,
conventions, architecture decisions, env setup) belongs here. When
you learn or change something project-wide, **append or edit
AGENTS.md** in the same session — do not rely on chat memory alone.

### Versioning and changelog

- User-visible or release-worthy changes: note them under **`## [draft]`**
  in **`CHANGELOG.md`** (72-column prose) until the owner asks for a
  release.
- **Do not** bump **`version` in `pyproject.toml`** or create git
  commits unless the owner explicitly asks.
- On release (when asked): rename `[draft]` to `[x.y.z]` with date,
  bump `pyproject.toml` to match, then commit.
- Released **`pyproject.toml` version** must match the latest numbered
  `CHANGELOG.md` heading (not `[draft]`).

### Stack

- **License:** LGPL-3.0-or-later (`LICENSE`); user docs in `README.md`.
- **C++20** + **CMake**, **LibTorch** at `./libtorch` (local extract;
  listed in `.gitignore`).
- **Python tooling** via **uv** (`pyproject.toml` at repo root).

### Build (C++)

```bash
./scripts/cmake_build.sh
export LD_LIBRARY_PATH=${CMAKE_SOURCE_DIR}/libtorch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
./scripts/train_latent_ae_ode.sh
```

`scripts/cmake_build.sh` — Ninja, `-j` = усі ядра (`nproc`). Після змін
у `eegexpcpp_model` перезберіть і `eegexpcpp_infer` для Streamlit-admin
(`--target eegexpcpp_infer`; перезапуск `streamlit run`, інакше старий
`.so` лишається в процесі).

`CMAKE_PREFIX_PATH` defaults to `${CMAKE_SOURCE_DIR}/libtorch` if that
directory exists. Requires system **HDF5**; **nlohmann/json** fetched by
CMake (`FetchContent`, v3.11.3 tarball) on configure.

### C++ model (`include/eegexpcpp/`)

- **LatentAeOde:** JSON `model.n_latent`; `feature_dim` from HDF5 at
  build time (`ModelConfig` only to construct). `Z = enc(X)` with
  `Z.shape[-1]=n_latent`; `dZ` / `dY_pred` via `TimeMlp::forward_jvp`
  (Linear + gelu/relu/tanh); `dZ_pred = ode(Z)`;
  `Y,dY_pred = decoder(Z,dZ_pred)` like `enc(X,dX)`. JSON **`model.ode`**:
  **`type`** `nn` (`hidden`, `activation`) or **`sindy`** (`libraries[]`
  with `poly` / `fourier` terms; `dZ = Theta(Z) @ Xi`, AdamW + `lambda_l1`,
  no STLSQ yet). SINDy export: `eegexpcpp_sindy_csv`, infer
  infer `sindy_csv` → Python `csv` parse; admin tab «SINDy Xi».
  Tensors **`[B,T,F]`** (HDF5 `[F,T]` transposed on load).
- **Train:** AdamW, all hyperparameters in JSON (`configs/`). Resume:
  train `[next_epoch, next_epoch + epochs)`; log `epoch=` is cumulative;
  checkpoint stores `next_epoch`. Ctrl+C saves checkpoint for resume.
  On resume, Adam momentum from checkpoint; `lr` and `weight_decay` from
  JSON.
  Train/val: one forward per epoch on full split tensors `[1, T, F]`.
  Infer/admin use the same layout.
- Library target `eegexpcpp_model`; shared `libeegexpcpp_infer.so`
  (C API `infer_api.h`, forward + `ode_rhs` + `rk4_latent` for Python
  ctypes); header `rk4.hpp` (generic RK4, no LibTorch). Binaries
  `eegexpcpp` (smoke), `eegexpcpp_train`, `eegexpcpp_test`
  (JVP/loss/HDF5), `eegexpcpp_calculus_test` (RK4; pytest
  `tests/test_calculus_cpp.py`). C++ tests use **doctest** v2.4.11
  (`FetchContent` header + `doctest.cpp` main, alias
  `doctest::doctest_with_main`).

### Python (uv)

- Package root: `python/eegexpcpp/` (`calculus`, `signal`, `prep`,
  `stft` — band decomposition).
- Example prep HDF5: `data/S001R01_8_prepare.h5` (inputs under `data/`,
  `*.mat` / `*.h5` gitignored) — all ICA components, each
  band-decomposed then embedded; groups `{s}/{b}/` with `X`, `dX`,
  `X_mean`, `X_std` (attr `band_name` on group). C++ train JSON:
  `data.h5_path`, `data.h5_group` (e.g. `0` or `lorenz/0`). Lorenz
  demo: `scripts/save_lorenz_h5.py` → `data/lorenz.h5` groups `0..2`
  (one Lorenz state dimension each, delay embed `m=128`, `tau=1`,
  root attr `n_latent=3`).
- Tests: `uv run pytest` (see `tests/`).
- Training admin (Streamlit + Plotly, no pip torch): build
  `eegexpcpp_infer`, then:

```bash
uv sync --all-groups
export LD_LIBRARY_PATH=${PWD}/libtorch/lib:${PWD}/build${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
uv run pytest
uv run streamlit run scripts/latent_ae_ode_admin.py
```

### LibTorch

- Version on disk: see `libtorch/build-version` (e.g.
  `2.14.1+cu132`).
- CUDA build: ensure driver/runtime match; set `LD_LIBRARY_PATH` to
  include `libtorch/lib` when running binaries if the loader cannot
  find `.so` files.
