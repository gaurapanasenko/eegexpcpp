"""C++ RK4 calculus tests (no LibTorch)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CPP_TEST = ROOT / "build" / "eegexpcpp_calculus_test"


@pytest.fixture(scope="module")
def calculus_test_bin() -> Path:
    if not CPP_TEST.is_file():
        pytest.skip(
            "build C++ calculus tests first: ./scripts/cmake_build.sh"
        )
    return CPP_TEST


def test_calculus_rk4_cpp(calculus_test_bin: Path) -> None:
    proc = subprocess.run(
        [str(calculus_test_bin)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert "Status: SUCCESS!" in proc.stdout
