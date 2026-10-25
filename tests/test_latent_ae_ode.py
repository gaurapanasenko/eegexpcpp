"""C++ LatentAeOde + JVP integration tests."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CPP_TEST = ROOT / "build" / "eegexpcpp_test"
LIBTORCH_LIB = ROOT / "libtorch" / "lib"


def _libtorch_env() -> dict[str, str]:
    env = os.environ.copy()
    if LIBTORCH_LIB.is_dir():
        prev = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (
            f"{LIBTORCH_LIB}:{prev}" if prev else str(LIBTORCH_LIB)
        )
    return env


@pytest.fixture(scope="module")
def cpp_test_bin() -> Path:
    if not CPP_TEST.is_file():
        pytest.skip(
            "build C++ tests first: ./scripts/cmake_build.sh"
        )
    return CPP_TEST


def test_latent_ae_ode_cpp(cpp_test_bin: Path) -> None:
    proc = subprocess.run(
        [str(cpp_test_bin)],
        cwd=ROOT,
        env=_libtorch_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert "Status: SUCCESS!" in proc.stdout
