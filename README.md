# eegexpcpp

EEG signal processing and embedding experiments: NumPy/Python
pipelines plus C++/LibTorch training for a latent AE+ODE model.

## Features

- **calculus** — 4th-order finite-difference derivative / integral, RK4
- **signal** — z-score, delay embedding (`embed_sig`)
- **stft** — smooth band decomposition (sum of bands = original)
- **prep** — ICA component → bands → embed; HDF5 export `{s}/{b}/`

## Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended)
- C++20 toolchain and CMake 3.18+ (optional C++ target)
- LibTorch C++ distribution extracted to `./libtorch` (gitignored)
- HDF5 C library (e.g. Arch `hdf5`)

## Python

```bash
uv sync --all-groups
uv run pytest
```

Package import path: `eegexpcpp` (sources under `python/eegexpcpp/`).

### Prepare HDF5 from EEGLAB `.mat` (v7.3)

Place `S001R01_8.mat` (or your file) in `data/`, then:

```bash
uv run python scripts/save_s001_prepare.py
```

Output `data/S001R01_8_prepare.h5` layout:

- Root attrs: `fs`, `dt`, `m`, `tau`, `n_latent`, `n_components`, `n_bands`
- Groups `/{s}/{b}/` with datasets `X`, `dX`, `X_mean`, `X_std`
- Group attr `band_name`

Export checks that band-decomposed time series sum back to each ICA
component before embedding.

## C++

```bash
./scripts/cmake_build.sh
export LD_LIBRARY_PATH=./libtorch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
./build/eegexpcpp_train configs/latent_ae_ode.lorenz.json
# or: ./scripts/train_latent_ae_ode.sh --list
#     ./scripts/train_latent_ae_ode.sh --config s001_prepare
```

`CMAKE_PREFIX_PATH` defaults to `./libtorch` when present. Tensor layout
for training is `[B, T, F]` (HDF5 stores `X`, `dX` as `[F, T]`).

## Agent / contributor notes

See [AGENTS.md](AGENTS.md) for project conventions (ponytail, formatting).

## Academic integrity and use of AI

### Українською

Програмне забезпечення розробляється в контексті дисертаційної роботи
(прикладна математика, ДНУ імені Олеся Гончара). Відповідно до
[Політики використання штучного інтелекту в ДНУ](https://www.dnu.dp.ua/view/unicheck)
(наказ № 106 від 14.04.2025) та Закону України «Про академічну
доброчесність» застосування ШІ має бути **прозорим**; **автор несе
відповідальність** за зміст, оригінальність і достовірність результатів;
ШІ не звільняє від перевірки коду та інтерпретації наукових висновків.

**Декларація використання ШІ у програмному коді**

- Частину програмного коду **редагував агент Cursor** (IDE Cursor) з
  моделлю **Composer 2.5** (генерація та рефакторинг фрагментів коду за
  запитом автора).
- **Особистий внесок автора:** постановка задачі, математична модель
  (latent AE + ODE, обробка сигналу, критерії навчання), архітектура
  системи, конфігурації експериментів, інтерпретація результатів.
- **Перевірка автором:** верифікація програмного коду (коректність,
  тести, збірка), перевірка відповідності реалізації математичній
  моделі та експериментальним даним.
- **Наукова новизна, логіка побудови системи та фінальні результати
  отримані автором самостійно.** Весь програмний код пройшов верифікацію
  та ручну валідацію.

### English

This software is developed as part of a dissertation (applied
mathematics, Oles Honchar Dnipro National University). Under the
[DNU policy on the use of artificial intelligence](https://www.dnu.dp.ua/view/unicheck)
(Order No. 106 of 14 April 2025) and the Law of Ukraine “On Academic
Integrity”, use of AI must be **transparent**; the **author is
responsible** for the content, originality, and validity of the results;
AI does not replace verification of the code or interpretation of
scientific conclusions.

**Declaration of AI use in program code**

- Part of the code was **edited by the Cursor agent** (Cursor IDE) using
  the **Composer 2.5** model (generation and refactoring of code
  fragments at the author’s request).
- **Author’s own contribution:** problem statement, mathematical model
  (latent AE + ODE, signal processing, training criteria), system
  architecture, experiment configurations, interpretation of results.
- **Verification by the author:** program code verification
  (correctness, tests, build), checking that the implementation matches
  the mathematical model and experimental data.
- **Scientific novelty, system design logic, and final results were
  obtained independently by the author.** All program code underwent
  verification and manual validation.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

LGPL-3.0-or-later. See [LICENSE](LICENSE).
