"""Fourth-order finite-difference calculus and RK4 (NumPy)."""

from __future__ import annotations

import numpy as np

__all__ = [
    "diff4_order1",
    "diff4_order1_inplace",
    "integ4_order1",
    "integ4_order1_inplace",
    "custom_rk4",
    "custom_rk4_inplace",
    "integ_diff_loss",
]


def _axis_from_dim(dim: int) -> int:
    d = 1 if dim < 1 else dim
    return d - 1


def diff4_order1(u: np.ndarray, dt: float, *, dim: int = 1) -> np.ndarray:
    du = np.empty_like(u)
    diff4_order1_inplace(du, u, dt, dim=dim)
    return du


def diff4_order1_inplace(
    du: np.ndarray, u: np.ndarray, dt: float, *, dim: int = 1
) -> np.ndarray:
    axis = _axis_from_dim(dim)
    if du.shape[axis] != u.shape[axis]:
        raise ValueError(f"Size mismatch along dimension {axis + 1}")
    u_m = np.moveaxis(u, axis, -1)
    du_m = np.moveaxis(du, axis, -1)
    n = u_m.shape[-1]
    if n < 5:
        raise ValueError("Dimension size must be at least 5")

    inv_2dt = np.asarray(1 / (2 * dt), dtype=u.dtype)
    inv_12dt = np.asarray(1 / (12 * dt), dtype=u.dtype)

    du_m[..., 0] = (-3 * u_m[..., 0] + 4 * u_m[..., 1] - u_m[..., 2]) * inv_2dt
    du_m[..., 1] = (-3 * u_m[..., 1] + 4 * u_m[..., 2] - u_m[..., 3]) * inv_2dt
    du_m[..., 2:-2] = (
        -u_m[..., 4:]
        + 8 * u_m[..., 3:-1]
        - 8 * u_m[..., 1:-3]
        + u_m[..., :-4]
    ) * inv_12dt
    du_m[..., -2] = (
        3 * u_m[..., -2] - 4 * u_m[..., -3] + u_m[..., -4]
    ) * inv_2dt
    du_m[..., -1] = (
        3 * u_m[..., -1] - 4 * u_m[..., -2] + u_m[..., -3]
    ) * inv_2dt
    return du


def integ4_order1(
    u: np.ndarray, dt: float, u0: np.ndarray, *, dim: int = 1
) -> np.ndarray:
    u_out = np.zeros_like(u)
    integ4_order1_inplace(u_out, u, dt, u0, dim=dim)
    return u_out


def integ4_order1_inplace(
    u_out: np.ndarray,
    u: np.ndarray,
    dt: float,
    u0: np.ndarray,
    *,
    dim: int = 1,
) -> np.ndarray:
    axis = _axis_from_dim(dim)
    if u_out.shape[axis] != u.shape[axis]:
        raise ValueError(f"Size mismatch along dimension {axis + 1}")
    u_m = np.moveaxis(u, axis, -1)
    u_out_m = np.moveaxis(u_out, axis, -1)
    n = u_m.shape[-1]
    if n < 4:
        raise ValueError("Dimension size must be at least 4 for 4th order integration")

    u0_m = np.moveaxis(u0, axis, -1)
    c_dt24 = np.asarray(dt / 24, dtype=u.dtype)

    init_val = u0_m[..., 0]
    incr = np.empty(u_m.shape[:-1] + (n - 1,), dtype=u.dtype)
    incr[..., 0] = c_dt24 * (
        9 * u_m[..., 0] + 19 * u_m[..., 1] - 5 * u_m[..., 2] + u_m[..., 3]
    )
    incr[..., 1:-1] = c_dt24 * (
        -u_m[..., :-3]
        + 13 * u_m[..., 1:-2]
        + 13 * u_m[..., 2:-1]
        - u_m[..., 3:]
    )
    incr[..., -1] = c_dt24 * (
        u_m[..., -4]
        - 5 * u_m[..., -3]
        + 19 * u_m[..., -2]
        + 9 * u_m[..., -1]
    )
    u_out_m[..., 0] = init_val
    u_out_m[..., 1:] = init_val[..., np.newaxis] + np.cumsum(incr, axis=-1)
    return u_out


def custom_rk4_inplace(
    u: np.ndarray,
    f,
    u0: np.ndarray,
    p,
    dt: float,
) -> None:
    if u.shape[1] != u0.shape[1]:
        raise ValueError("Column count mismatch between u and u0")
    n = u.shape[0]
    u[0:1, :] = u0
    buf = np.zeros((7, u0.shape[1]), dtype=u.dtype)
    k1, k2, k3, k4, tmp1, tmp2, tmp3 = (buf[i] for i in range(7))
    half = np.asarray(0.5, dtype=u.dtype)
    one = np.asarray(1.0, dtype=u.dtype)
    sixth = np.asarray(dt / 6.0, dtype=u.dtype)
    # ponytail: time march is sequential for generic f; stages are vectorized
    for i in range(n - 1):
        ti = i * dt
        ui = u[i, :]
        f(k1, ui, p, ti)
        tmp1[:] = ui + half * dt * k1
        f(k2, tmp1, p, ti + half * dt)
        tmp2[:] = ui + half * dt * k2
        f(k3, tmp2, p, ti + half * dt)
        tmp3[:] = ui + one * dt * k3
        f(k4, tmp3, p, ti + dt)
        u[i + 1, :] = ui + sixth * (k1 + 2 * k2 + 2 * k3 + k4)


def custom_rk4(f, u0: np.ndarray, p, dt: float, n: int) -> np.ndarray:
    out = np.full((n, u0.shape[1]), np.nan, dtype=u0.dtype)
    custom_rk4_inplace(out, f, u0, p, dt)
    return out


def integ_diff_loss(x: np.ndarray, dt: float = 1.0, *, dims: int = 1) -> np.ndarray:
    dx = diff4_order1(x, dt, dim=dims)
    u0 = np.take(x, [0], axis=_axis_from_dim(dims))
    rec = integ4_order1(dx, dt, u0, dim=dims)
    return np.sum(np.abs(rec - x) ** 2, axis=_axis_from_dim(dims), keepdims=True)
