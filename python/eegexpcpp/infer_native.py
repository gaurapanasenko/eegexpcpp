"""ctypes wrapper for libeegexpcpp_infer.so (no pip torch)."""

from __future__ import annotations

import os
from ctypes import CDLL, POINTER, c_char, c_char_p, c_float, c_int, c_int64, c_void_p
from pathlib import Path
from typing import Literal

import numpy as np

SplitName = Literal["train", "val", "full"]

_SPLIT = {"train": 0, "val": 1, "full": 2}

ROOT = Path(__file__).resolve().parents[2]


def default_infer_lib() -> Path:
    env = os.environ.get("EEGEXPCPP_INFER_LIB")
    if env:
        return Path(env)
    return ROOT / "build" / "libeegexpcpp_infer.so"


def infer_lib_available() -> bool:
    return default_infer_lib().is_file()


class InferError(RuntimeError):
    pass


class InferSession:
    """Loaded LatentAeOde checkpoint + HDF5 segment from experiment JSON."""

    def __init__(
        self,
        config_path: str | Path,
        *,
        device: str = "cpu",
        checkpoint_path: str | Path | None = None,
        lib_path: str | Path | None = None,
    ) -> None:
        lib = CDLL(str(lib_path or default_infer_lib()))
        self._lib = lib
        self._handle: c_void_p | None = None

        lib.eegexpcpp_infer_create.argtypes = [
            c_char_p,
            c_char_p,
            c_char_p,
        ]
        lib.eegexpcpp_infer_create.restype = c_void_p
        lib.eegexpcpp_infer_destroy.argtypes = [c_void_p]
        lib.eegexpcpp_infer_destroy.restype = None
        lib.eegexpcpp_infer_last_error.argtypes = []
        lib.eegexpcpp_infer_last_error.restype = c_char_p
        lib.eegexpcpp_infer_next_epoch.argtypes = [c_void_p]
        lib.eegexpcpp_infer_next_epoch.restype = c_int64
        lib.eegexpcpp_infer_sizes.argtypes = [
            c_void_p,
            c_int,
            POINTER(c_int64),
            POINTER(c_int64),
        ]
        lib.eegexpcpp_infer_sizes.restype = c_int
        lib.eegexpcpp_infer_n_latent.argtypes = [c_void_p]
        lib.eegexpcpp_infer_n_latent.restype = c_int64
        lib.eegexpcpp_infer_forward.argtypes = [
            c_void_p,
            c_int,
            POINTER(c_float),
            POINTER(c_float),
            POINTER(c_float),
            POINTER(c_float),
            POINTER(c_float),
        ]
        lib.eegexpcpp_infer_forward.restype = c_int
        lib.eegexpcpp_infer_ode_rhs.argtypes = [
            c_void_p,
            POINTER(c_float),
            POINTER(c_float),
            c_int64,
        ]
        lib.eegexpcpp_infer_ode_rhs.restype = c_int
        lib.eegexpcpp_infer_rk4_latent.argtypes = [
            c_void_p,
            c_int64,
            c_float,
            POINTER(c_float),
            POINTER(c_float),
        ]
        lib.eegexpcpp_infer_rk4_latent.restype = c_int
        lib.eegexpcpp_infer_ode_is_sindy.argtypes = [c_void_p]
        lib.eegexpcpp_infer_ode_is_sindy.restype = c_int
        lib.eegexpcpp_infer_sindy_csv_size.argtypes = [c_void_p]
        lib.eegexpcpp_infer_sindy_csv_size.restype = c_int64
        lib.eegexpcpp_infer_sindy_csv.argtypes = [
            c_void_p,
            POINTER(c_char),
            c_int64,
        ]
        lib.eegexpcpp_infer_sindy_csv.restype = c_int

        cfg = str(Path(config_path).resolve())
        ckpt = (
            str(Path(checkpoint_path).resolve())
            if checkpoint_path is not None
            else None
        )
        handle = lib.eegexpcpp_infer_create(
            cfg.encode("utf-8"),
            ckpt.encode("utf-8") if ckpt else None,
            device.encode("utf-8"),
        )
        if not handle:
            err = lib.eegexpcpp_infer_last_error()
            msg = err.decode("utf-8") if err else "unknown error"
            raise InferError(msg)
        self._handle = handle

    def _check(self, code: int) -> None:
        if code != 0:
            err = self._lib.eegexpcpp_infer_last_error()
            msg = err.decode("utf-8") if err else "unknown error"
            raise InferError(msg)

    @property
    def next_epoch(self) -> int:
        assert self._handle is not None
        return int(self._lib.eegexpcpp_infer_next_epoch(self._handle))

    def sizes(self, split: SplitName = "train") -> tuple[int, int]:
        assert self._handle is not None
        t_out = c_int64()
        f_out = c_int64()
        self._check(
            self._lib.eegexpcpp_infer_sizes(
                self._handle, _SPLIT[split], t_out, f_out
            )
        )
        return int(t_out.value), int(f_out.value)

    def n_latent(self) -> int:
        assert self._handle is not None
        return int(self._lib.eegexpcpp_infer_n_latent(self._handle))

    def forward(self, split: SplitName = "train") -> dict[str, np.ndarray]:
        assert self._handle is not None
        t_len, f_dim = self.sizes(split)
        l_dim = self.n_latent()
        n_xy = t_len * f_dim
        n_z = t_len * l_dim
        x = np.empty(n_xy, dtype=np.float32)
        y = np.empty(n_xy, dtype=np.float32)
        z = np.empty(n_z, dtype=np.float32)
        dz = np.empty(n_z, dtype=np.float32)
        dz_pred = np.empty(n_z, dtype=np.float32)
        self._check(
            self._lib.eegexpcpp_infer_forward(
                self._handle,
                _SPLIT[split],
                x.ctypes.data_as(POINTER(c_float)),
                y.ctypes.data_as(POINTER(c_float)),
                z.ctypes.data_as(POINTER(c_float)),
                dz.ctypes.data_as(POINTER(c_float)),
                dz_pred.ctypes.data_as(POINTER(c_float)),
            )
        )
        shape_f = (t_len, f_dim)
        shape_z = (t_len, l_dim)
        return {
            "x": x.reshape(shape_f),
            "y": y.reshape(shape_f),
            "z": z.reshape(shape_z),
            "dz": dz.reshape(shape_z),
            "dz_pred": dz_pred.reshape(shape_z),
        }

    def ode_is_sindy(self) -> bool:
        assert self._handle is not None
        return self._lib.eegexpcpp_infer_ode_is_sindy(self._handle) != 0

    def sindy_coefficients_csv(self) -> str:
        from eegexpcpp.sindy_table import sindy_coefficients_csv

        return sindy_coefficients_csv(self)

    def sindy_coefficients_table(self) -> dict[str, list[float | str]]:
        from eegexpcpp.sindy_table import sindy_coefficients_table

        return sindy_coefficients_table(self)

    def ode_rhs(self, z: np.ndarray) -> np.ndarray:
        assert self._handle is not None
        z1 = np.ascontiguousarray(z, dtype=np.float32).reshape(-1)
        f_dim = z1.shape[0]
        out = np.empty(f_dim, dtype=np.float32)
        self._check(
            self._lib.eegexpcpp_infer_ode_rhs(
                self._handle,
                z1.ctypes.data_as(POINTER(c_float)),
                out.ctypes.data_as(POINTER(c_float)),
                c_int64(f_dim),
            )
        )
        return out

    def rk4_latent(
        self, z0: np.ndarray, dt: float, n_steps: int
    ) -> np.ndarray:
        assert self._handle is not None
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        z0v = np.ascontiguousarray(z0, dtype=np.float32).reshape(-1)
        l_dim = z0v.shape[0]
        if l_dim != self.n_latent():
            raise ValueError("z0 length must match n_latent")
        out = np.empty(n_steps * l_dim, dtype=np.float32)
        self._check(
            self._lib.eegexpcpp_infer_rk4_latent(
                self._handle,
                c_int64(n_steps),
                c_float(dt),
                z0v.ctypes.data_as(POINTER(c_float)),
                out.ctypes.data_as(POINTER(c_float)),
            )
        )
        return out.reshape(n_steps, l_dim)

    def close(self) -> None:
        if self._handle is not None:
            self._lib.eegexpcpp_infer_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> InferSession:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
