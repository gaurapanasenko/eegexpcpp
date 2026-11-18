"""Checkpoint train history C API (ctypes)."""

from __future__ import annotations

from pathlib import Path

import pytest

from eegexpcpp.checkpoint_history import (
    METRIC_NAMES,
    load_checkpoint_history,
)
from eegexpcpp.experiment import ROOT, describe_config
from eegexpcpp.infer_native import infer_lib_available


@pytest.mark.skipif(not infer_lib_available(), reason="infer .so missing")
def test_checkpoint_history_api() -> None:
    cfg = describe_config(ROOT / "configs" / "latent_ae_ode.lorenz.json")
    if not cfg.checkpoint_exists or cfg.checkpoint_path is None:
        pytest.skip("checkpoint missing")
    hist = load_checkpoint_history(cfg.checkpoint_path)
    if hist.metrics.size == 0:
        pytest.skip("checkpoint has no train history yet")
    assert hist.metrics.shape[1] == len(METRIC_NAMES)