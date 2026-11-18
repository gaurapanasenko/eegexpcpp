"""Read train_history_* keys from latent_ae_ode.pt via infer .so."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ctypes import (
    CDLL,
    POINTER,
    c_char_p,
    c_double,
    c_int,
    c_int64,
    create_string_buffer,
)

from eegexpcpp.infer_native import default_infer_lib, infer_lib_available

METRIC_NAMES = (
    "epoch",
    "train_J",
    "val_J",
    "J_Y",
    "J_dY",
    "J_dZ",
    "J_z_1",
)


class CheckpointHistoryError(RuntimeError):
    pass


@dataclass
class CheckpointHistory:
    metrics: np.ndarray
    events: list[dict]


def load_checkpoint_history(
    checkpoint_path: str | Path,
    lib_path: Path | None = None,
) -> CheckpointHistory:
    if not infer_lib_available():
        raise CheckpointHistoryError("libeegexpcpp_infer.so not found")
    lib = CDLL(str(lib_path or default_infer_lib()))
    lib.eegexpcpp_infer_last_error.argtypes = []
    lib.eegexpcpp_infer_last_error.restype = c_char_p

    lib.eegexpcpp_checkpoint_history_metric_rows.argtypes = [c_char_p]
    lib.eegexpcpp_checkpoint_history_metric_rows.restype = c_int64
    lib.eegexpcpp_checkpoint_history_metric_cols.argtypes = []
    lib.eegexpcpp_checkpoint_history_metric_cols.restype = c_int
    lib.eegexpcpp_checkpoint_history_metric_copy.argtypes = [
        c_char_p,
        POINTER(c_double),
        c_int64,
    ]
    lib.eegexpcpp_checkpoint_history_metric_copy.restype = c_int
    lib.eegexpcpp_checkpoint_history_events_json_size.argtypes = [c_char_p]
    lib.eegexpcpp_checkpoint_history_events_json_size.restype = c_int64
    lib.eegexpcpp_checkpoint_history_events_json.argtypes = [
        c_char_p,
        c_char_p,
        c_int64,
    ]
    lib.eegexpcpp_checkpoint_history_events_json.restype = c_int

    ckpt = str(Path(checkpoint_path).resolve()).encode()
    rows = lib.eegexpcpp_checkpoint_history_metric_rows(ckpt)
    if rows < 0:
        err = lib.eegexpcpp_infer_last_error()
        raise CheckpointHistoryError(
            err.decode() if err else "checkpoint_history_metric_rows failed"
        )
    cols = lib.eegexpcpp_checkpoint_history_metric_cols()
    metrics = np.empty((0, cols), dtype=np.float64)
    if rows > 0:
        buf = (c_double * (rows * cols))()
        rc = lib.eegexpcpp_checkpoint_history_metric_copy(
            ckpt, buf, rows * cols
        )
        if rc != 0:
            err = lib.eegexpcpp_infer_last_error()
            raise CheckpointHistoryError(
                err.decode() if err else "checkpoint_history_metric_copy failed"
            )
        metrics = np.frombuffer(buf, dtype=np.float64, count=rows * cols)
        metrics = metrics.reshape(rows, cols)

    ev_size = lib.eegexpcpp_checkpoint_history_events_json_size(ckpt)
    if ev_size < 0:
        err = lib.eegexpcpp_infer_last_error()
        raise CheckpointHistoryError(
            err.decode() if err else "checkpoint_history_events_json_size failed"
        )
    ev_buf = create_string_buffer(ev_size)
    rc = lib.eegexpcpp_checkpoint_history_events_json(ckpt, ev_buf, ev_size)
    if rc != 0:
        err = lib.eegexpcpp_infer_last_error()
        raise CheckpointHistoryError(
            err.decode() if err else "checkpoint_history_events_json failed"
        )
    raw = ev_buf.value.decode()
    events = json.loads(raw) if raw else []
    return CheckpointHistory(metrics=metrics, events=events)