#!/usr/bin/env bash
# Configure (Ninja) and build C++ targets with all CPU threads.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT}/build"

if command -v nproc >/dev/null 2>&1; then
  JOBS="$(nproc)"
elif [[ "$(uname -s)" == Darwin ]] && command -v sysctl >/dev/null 2>&1; then
  JOBS="$(sysctl -n hw.ncpu)"
else
  JOBS=4
fi

if [[ -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
  gen="$(sed -n 's/^CMAKE_GENERATOR:INTERNAL=//p' "${BUILD_DIR}/CMakeCache.txt")"
  if [[ -n "$gen" && "$gen" != "Ninja" ]]; then
    echo "Switching ${BUILD_DIR} to Ninja (removing old generator: ${gen})" >&2
    rm -rf "${BUILD_DIR}"
  fi
fi

cmake -S "$ROOT" -B "$BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" -j"${JOBS}" "$@"
