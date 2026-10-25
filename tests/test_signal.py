import numpy as np

import h5py

import pytest

from eegexpcpp.prep import (
    _assert_bands_reconstruct_signal,
    prepare_embed_signal,
    save_prepare_hdf5,
)
from eegexpcpp.signal import embed_sig, zscore, zscore_inverse


class TestZscore:
    def test_zero_mean_unit_std(self):
        x = np.array([[1.0, 2.0, 3.0], [4.0, 10.0, 16.0]])
        y, _, _ = zscore(x, dim=2)
        np.testing.assert_allclose(y.mean(axis=1), 0.0, atol=1e-12)
        np.testing.assert_allclose(y.std(axis=1), 1.0, atol=1e-12)

    def test_inverse(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(4, 32))
        y, mean, std = zscore(x, dim=2)
        back = zscore_inverse(y, mean, std)
        np.testing.assert_allclose(back, x, rtol=1e-10)


class TestPrepareEmbed:
    def test_shapes(self):
        n = 100
        x = np.sin(np.linspace(0, 4 * np.pi, n)).astype(np.float32)
        fs = np.float32(10.0)
        x_mat, dx_mat, n_lat, out_dt, x_mean, x_std = prepare_embed_signal(
            x, 4, fs, m=5, tau=1
        )
        assert n_lat == 4
        assert out_dt == np.float32(1.0 / fs)
        assert x_mat.shape == (5, n - 4)
        assert dx_mat.shape == x_mat.shape
        assert x_mean.shape == (1,)
        assert x_std.shape == (1,)


class TestPrepareHdfGroups:
    def test_s_b_layout(self, tmp_path):
        fs = 64.0
        n = 256
        t = np.arange(n, dtype=np.float64) / fs
        icaact = np.stack(
            [np.sin(2 * np.pi * 5 * t), np.sin(2 * np.pi * 12 * t)], axis=1
        ).astype(np.float32)
        bands = (("low", -np.inf, 30.0), ("high", 30.0, np.inf))
        out = tmp_path / "prep.h5"
        save_prepare_hdf5(out, icaact, fs, 2, m=5, tau=1, bands=bands)
        with h5py.File(out, "r") as f:
            assert f.attrs["n_components"] == 2
            assert f.attrs["n_bands"] == 2
            g = f["1/0"]
            assert g.attrs["band_name"] == "low"
            assert g["X"].shape[0] == 5
            assert g["dX"].shape == g["X"].shape

    def test_band_sum_guard(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(128).astype(np.float32)
        bad = rng.standard_normal((128, 2)).astype(np.float32)
        with pytest.raises(ValueError, match="band sum does not match"):
            _assert_bands_reconstruct_signal(x, bad, component=3)


class TestEmbedSig:
    def test_1d_tau1(self):
        x = np.arange(10, dtype=np.float32)
        y = embed_sig(x, 3, 1)
        assert y.shape == (8, 3)
        assert y.dtype == np.float32
        np.testing.assert_allclose(y[:, 0], x[0:8])
        np.testing.assert_allclose(y[:, 1], x[1:9])
        np.testing.assert_allclose(y[:, 2], x[2:10])

    def test_2d_merge(self):
        x = np.arange(20, dtype=np.float32).reshape(10, 2)
        y = embed_sig(x, 2, 2, merge=False)
        assert y.shape == (8, 2, 2)
        y_m = embed_sig(x, 2, 2, merge=True)
        assert y_m.shape == (8, 4)
