import numpy as np

from eegexpcpp.stft import EEG_BANDS, band_filterbank, decompose_signal


FS = 256.0


def _assert_reconstructs(x: np.ndarray, atol: float = 1e-10) -> None:
    parts = decompose_signal(x, FS, EEG_BANDS, method="rfft")
    np.testing.assert_allclose(parts.sum(axis=1), x, atol=atol)


class TestDecomposeReconstruction:
    def test_random_even_length(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(4096)
        _assert_reconstructs(x)

    def test_random_odd_length(self):
        rng = np.random.default_rng(7)
        x = rng.standard_normal(4001)
        _assert_reconstructs(x)

    def test_complex_sinusoid_mix(self):
        n = 8192
        t = np.arange(n, dtype=np.float64) / FS
        x = (
            0.7 * np.sin(2 * np.pi * 1.2 * t)
            + 0.4 * np.sin(2 * np.pi * 6.0 * t)
            + 0.2 * np.sin(2 * np.pi * 10.5 * t)
            + 0.15 * np.sin(2 * np.pi * 22.0 * t)
            + 0.1 * np.sin(2 * np.pi * 45.0 * t)
        )
        _assert_reconstructs(x, atol=1e-9)

    def test_custom_two_band_partition(self):
        bands = (("low", -np.inf, 30.0), ("high", 30.0, np.inf))
        freqs = np.fft.rfftfreq(2048, d=1.0 / FS)
        bank = band_filterbank(freqs, bands)
        assert bank.shape == (2, freqs.size)
        rng = np.random.default_rng(0)
        x = rng.standard_normal(2048)
        parts = decompose_signal(x, FS, bands, method="rfft")
        np.testing.assert_allclose(parts.sum(axis=1), x, atol=1e-10)
