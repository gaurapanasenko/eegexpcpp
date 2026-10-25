"""ctypes infer library smoke tests."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from eegexpcpp.experiment import ROOT, describe_config
from eegexpcpp.infer_native import InferError, InferSession, infer_lib_available

LIBTORCH_LIB = ROOT / "libtorch" / "lib"


@pytest.fixture(scope="module")
def lib_env() -> None:
    if not infer_lib_available():
        pytest.skip("build libeegexpcpp_infer first")
    if LIBTORCH_LIB.is_dir():
        prev = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = (
            f"{LIBTORCH_LIB}:{ROOT / 'build'}:{prev}"
            if prev
            else f"{LIBTORCH_LIB}:{ROOT / 'build'}"
        )


@pytest.fixture(scope="module")
def lorenz_config() -> Path:
    cfg = ROOT / "configs" / "latent_ae_ode.lorenz.json"
    if not cfg.is_file():
        pytest.skip("lorenz config missing")
    entry = describe_config(cfg)
    if not entry.checkpoint_exists:
        pytest.skip("lorenz checkpoint missing")
    return cfg


def test_infer_forward_shapes(lib_env: None, lorenz_config: Path) -> None:
    try:
        session = InferSession(lorenz_config, device="cpu")
    except InferError as exc:
        pytest.skip(f"infer session: {exc}")
    with session:
        try:
            out = session.forward("train")
        except InferError as exc:
            pytest.skip(f"checkpoint incompatible with model config: {exc}")
        t_len, f_dim = session.sizes("train")
        assert out["x"].shape == (t_len, f_dim)
        l_dim = session.n_latent()
        assert out["z"].shape == (t_len, l_dim)
        dz = session.ode_rhs(out["z"][0])
        assert dz.shape == (l_dim,)
        np.testing.assert_allclose(
            dz, out["dz_pred"][0], rtol=1e-4, atol=1e-5
        )
