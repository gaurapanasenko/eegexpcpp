#!/usr/bin/env bash
# Run C++ LatentAeOde training (eegexpcpp_train) from repo root.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_NAME="lorenz"
DO_BUILD=0
DO_LIST=0

usage() {
  cat <<'EOF'
Usage: scripts/train_latent_ae_ode.sh [options]

  --config NAME   Preset or path (default: lorenz)
  --list          List preset names and config files
  --build         scripts/cmake_build.sh if train binary is missing
  -h, --help      Show this help

Presets live in configs/latent_ae_ode.<NAME>.json
EOF
}

list_configs() {
  local f base short
  shopt -s nullglob
  for f in "${ROOT}"/configs/latent_ae_ode.*.json; do
    base="$(basename "$f")"
    short="${base#latent_ae_ode.}"
    short="${short%.json}"
    printf "  %-16s configs/%s\n" "$short" "$base"
  done
  shopt -u nullglob
}

resolve_config() {
  local name="$1"
  local candidate

  if [[ -f "$name" ]]; then
    CONFIG="$(cd "$(dirname "$name")" && pwd)/$(basename "$name")"
    return 0
  fi
  if [[ -f "${ROOT}/${name}" ]]; then
    CONFIG="${ROOT}/${name}"
    return 0
  fi

  candidate="${ROOT}/configs/latent_ae_ode.${name}.json"
  if [[ -f "$candidate" ]]; then
    CONFIG="$candidate"
    return 0
  fi

  candidate="${ROOT}/configs/${name}.json"
  if [[ -f "$candidate" ]]; then
    CONFIG="$candidate"
    return 0
  fi

  echo "unknown config: $name (try --list)" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_NAME="$2"
      shift 2
      ;;
    --list)
      DO_LIST=1
      shift
      ;;
    --build)
      DO_BUILD=1
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "$DO_LIST" -eq 1 ]]; then
  echo "Available configs:"
  list_configs
  exit 0
fi

resolve_config "$CONFIG_NAME"

BIN="${ROOT}/build/eegexpcpp_train"
if [[ ! -x "$BIN" ]]; then
  if [[ "$DO_BUILD" -eq 1 ]]; then
    "${ROOT}/scripts/cmake_build.sh"
  else
    echo "missing ${BIN} (use --build or cmake --build build)" >&2
    exit 1
  fi
fi

if [[ ! -d "${ROOT}/libtorch/lib" ]]; then
  echo "missing ${ROOT}/libtorch (extract LibTorch C++ distribution)" >&2
  exit 1
fi

H5_REL="$(uv run --directory "$ROOT" python -c "
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
cfg = json.loads(p.read_text())
print(cfg['data']['h5_path'])
" "$CONFIG")"
H5="${ROOT}/${H5_REL}"
if [[ ! -f "$H5" ]]; then
  echo "missing HDF5: ${H5}" >&2
  echo "prep: uv run python scripts/save_lorenz_h5.py" >&2
  echo "      uv run python scripts/save_s001_prepare.py" >&2
  exit 1
fi

export LD_LIBRARY_PATH="${ROOT}/libtorch/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

cd "$ROOT"
exec "$BIN" "$CONFIG"
