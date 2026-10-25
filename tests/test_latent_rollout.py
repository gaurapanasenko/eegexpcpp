"""Tests for chunked latent RK4 rollout (no C++ lib required)."""

from __future__ import annotations

import numpy as np

from eegexpcpp.latent_rollout import chunk_mse, rollout_chunked


class _FakeSession:
    """Linear field dz = A z for testing."""

    def __init__(self, scale: float = -0.1) -> None:
        self.scale = scale

    def ode_rhs(self, z: np.ndarray) -> np.ndarray:
        return (self.scale * z).astype(np.float32)


def test_rollout_chunked_resets_at_boundaries() -> None:
    dt = 0.01
    t_len = 20
    f_dim = 2
    z_true = np.random.RandomState(0).randn(t_len, f_dim).astype(np.float32)
    session = _FakeSession()
    chunk = 5
    z_rk = rollout_chunked(z_true, session, dt, chunk)
    assert z_rk.shape == z_true.shape
    for t0 in range(0, t_len, chunk):
        assert np.allclose(z_rk[t0], z_true[t0])


def test_chunk_mse_length() -> None:
    z = np.zeros((10, 3), dtype=np.float32)
    z2 = z.copy()
    mses = chunk_mse(z, z2, 4)
    assert len(mses) == 3
