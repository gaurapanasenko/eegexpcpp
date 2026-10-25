"""Signal z-score and delay embedding (NumPy)."""

from __future__ import annotations

import numpy as np

__all__ = ["zscore", "zscore_inverse", "embed_sig"]


def _axis_from_dim(dim: int) -> int:
    d = 1 if dim < 1 else dim
    return d - 1


def zscore(
    x: np.ndarray, *, dim: int = 1, eps: float = 1e-8
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = _axis_from_dim(dim)
    mean = x.mean(axis=axis, keepdims=True)
    std = x.std(axis=axis, keepdims=True)
    y = (x - mean) / (std + eps)
    return y, mean, std


def zscore_inverse(
    x_norm: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    eps: float = 1e-8,
) -> np.ndarray:
    return x_norm * (std + eps) + mean


def embed_sig(
    x: np.ndarray,
    m: int,
    tau: int = 1,
    *,
    merge: bool = False,
) -> np.ndarray:
    if m <= 0 or tau <= 0:
        raise ValueError("m and tau must be positive")
    x = np.asarray(x, dtype=np.float32)
    n_in = x.shape[0]
    n_out = n_in - (m - 1) * tau
    if n_out < 1:
        raise ValueError(
            f"signal length {n_in} too short for m={m}, tau={tau}"
        )
    idx = np.arange(n_out, dtype=np.intp)[:, None] + tau * np.arange(
        m, dtype=np.intp
    )[None, :]
    y = x[idx] if x.ndim == 1 else x[idx, ...]
    y = np.ascontiguousarray(y, dtype=np.float32)
    if not merge:
        return y
    if x.ndim < 2:
        raise ValueError("merge requires x with at least 2 dimensions")
    return y.reshape(y.shape[0], y.shape[1] * y.shape[2], *y.shape[3:])
