"""RK4 latent rollout using C++ ode_rhs and custom_rk4."""

from __future__ import annotations

import numpy as np

from eegexpcpp.calculus import custom_rk4
from eegexpcpp.infer_native import InferSession


def _ode_callback(du: np.ndarray, u: np.ndarray, session: InferSession, t: float) -> None:
    du[:] = session.ode_rhs(u)


def rollout_full(
    z_true: np.ndarray,
    session: InferSession,
    dt: float,
) -> np.ndarray:
    """Integrate ode(z) from z_true[0] for T steps (same length as z_true).

    Pass a CPU ``InferSession`` when the main forward uses CUDA (many
    small ``ode_rhs`` calls).
    """
    t_len = z_true.shape[0]
    u0 = np.ascontiguousarray(z_true[0:1], dtype=np.float32)

    def f(du: np.ndarray, u: np.ndarray, p: object, t: float) -> None:
        _ode_callback(du, u, session, t)

    return custom_rk4(f, u0, None, float(dt), t_len)


def rollout_chunked(
    z_true: np.ndarray,
    session: InferSession,
    dt: float,
    chunk_samples: int,
) -> np.ndarray:
    """Per-chunk RK4 with z0 = z_true[t0] at each chunk start."""
    if chunk_samples < 1:
        raise ValueError("chunk_samples must be >= 1")
    t_len = z_true.shape[0]
    z_rk4 = np.full_like(z_true, np.nan, dtype=np.float32)

    def f(du: np.ndarray, u: np.ndarray, p: object, t: float) -> None:
        _ode_callback(du, u, session, t)

    for t0 in range(0, t_len, chunk_samples):
        t1 = min(t_len, t0 + chunk_samples)
        n = t1 - t0
        u0 = np.ascontiguousarray(z_true[t0 : t0 + 1], dtype=np.float32)
        seg = custom_rk4(f, u0, None, float(dt), n)
        z_rk4[t0:t1] = seg
    return z_rk4


def chunk_mse(z_true: np.ndarray, z_rk4: np.ndarray, chunk_samples: int) -> list[float]:
    t_len = z_true.shape[0]
    mses: list[float] = []
    for t0 in range(0, t_len, chunk_samples):
        t1 = min(t_len, t0 + chunk_samples)
        diff = z_true[t0:t1] - z_rk4[t0:t1]
        mses.append(float(np.mean(diff * diff)))
    return mses
