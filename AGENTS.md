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

- User-visible or release-worthy changes: add an entry under
  `## [Unreleased]` or a new version section in **`CHANGELOG.md`**
  (Keep a Changelog style, 72-column prose).
- **In the same change**, bump **`version` in `pyproject.toml`**
  (semver). Do not edit the changelog without bumping the version.
- Current package version must match the latest `CHANGELOG.md`
  release heading.

### Stack

- **License:** LGPL-3.0-or-later (`LICENSE`); user docs in `README.md`.
- **C++20** + **CMake**, **LibTorch** at `./libtorch` (local extract;
  listed in `.gitignore`).
- **Python tooling** via **uv** (`pyproject.toml` at repo root).

### Build (C++)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/eegexpcpp
```

`CMAKE_PREFIX_PATH` defaults to `${CMAKE_SOURCE_DIR}/libtorch` if that
directory exists.

### Python (uv)

- Package root: `python/eegexpcpp/` (`calculus`, `signal`, `prep`,
  `stft` — band decomposition).
- Example prep HDF5: `data/S001R01_8_prepare.h5` (inputs under `data/`,
  `*.mat` / `*.h5` gitignored) — all ICA components, each
  band-decomposed then embedded; groups `{s}/{b}/` with `X`, `dX`,
  `X_mean`, `X_std` (attr `band_name` on group).
- Tests: `uv run pytest` (see `tests/`).

```bash
uv sync --all-groups
uv run pytest
```

### LibTorch

- Version on disk: see `libtorch/build-version` (e.g.
  `2.14.1+cu132`).
- CUDA build: ensure driver/runtime match; set `LD_LIBRARY_PATH` to
  include `libtorch/lib` when running binaries if the loader cannot
  find `.so` files.
