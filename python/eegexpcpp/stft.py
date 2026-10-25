"""STFT / RFFT band decomposition (partition-of-unity filters)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = [
    "EEG_BANDS",
    "Band",
    "hann_w",
    "low_pass",
    "high_pass",
    "smooth_bandpass",
    "band_filterbank",
    "stft_pr",
    "istft_pr",
    "decompose_spectrum",
    "decompose_signal",
]

Band = tuple[str, float, float]

EEG_BANDS: tuple[Band, ...] = (
    ("Sub-Delta", -np.inf, 0.5),
    ("Low-Delta", 0.5, 2.0),
    ("High-Delta", 2.0, 4.0),
    ("Theta", 4.0, 8.0),
    ("Alpha", 8.0, 13.0),
    ("Beta", 13.0, 30.0),
    ("Low-Gamma", 30.0, 60.0),
    ("High-Gamma", 60.0, 100.0),
    ("High-Freq", 100.0, np.inf),
)


def hann_w(x: np.ndarray, length: float) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2 * np.pi * x / length)


def low_pass(f: np.ndarray, f_cutoff: float, w: float = 0.25) -> np.ndarray:
    if np.isposinf(f_cutoff):
        return np.ones_like(f, dtype=np.float64)
    if np.isneginf(f_cutoff):
        return np.zeros_like(f, dtype=np.float64)
    out = np.empty_like(f, dtype=np.float64)
    out[f < f_cutoff - w / 2] = 1.0
    out[f > f_cutoff + w / 2] = 0.0
    mid = (f >= f_cutoff - w / 2) & (f <= f_cutoff + w / 2)
    out[mid] = hann_w(f[mid] - f_cutoff - w / 2, 2 * w)
    return out


def high_pass(f: np.ndarray, f_cutoff: float, w: float = 0.25) -> np.ndarray:
    if np.isneginf(f_cutoff):
        return np.ones_like(f, dtype=np.float64)
    if np.isposinf(f_cutoff):
        return np.zeros_like(f, dtype=np.float64)
    out = np.empty_like(f, dtype=np.float64)
    out[f < f_cutoff - w / 2] = 0.0
    out[f > f_cutoff + w / 2] = 1.0
    mid = (f >= f_cutoff - w / 2) & (f <= f_cutoff + w / 2)
    out[mid] = hann_w(f[mid] - f_cutoff - 3 * w / 2, 2 * w)
    return out


def smooth_bandpass(
    f: np.ndarray, f_start: float, f_end: float, w: float = 0.25
) -> np.ndarray:
    return high_pass(f, f_start, w=w) * low_pass(f, f_end, w=w)


def band_filterbank(
    freqs: np.ndarray,
    bands: Sequence[Band],
    *,
    w: float = 0.25,
) -> np.ndarray:
    """Shape (n_bands, n_freqs); rows sum to 1 on the frequency grid."""
    bank = np.stack(
        [smooth_bandpass(freqs, f0, f1, w=w) for _, f0, f1 in bands],
        axis=0,
    )
    total = bank.sum(axis=0)
    if not np.allclose(total, 1.0, rtol=0, atol=1e-5):
        raise ValueError("band filters do not sum to 1 on this frequency grid")
    return bank


def stft_pr(
    x: np.ndarray, fs: float, *, nfft: int = 128
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64).ravel()
    if nfft <= 0 or (nfft & (nfft - 1)) != 0:
        raise ValueError("nfft must be a positive power of two")
    win = np.hanning(nfft + 1)[:-1].astype(np.float64)
    hop = nfft // 2
    if not np.allclose(win[hop:] + win[:hop], 1.0):
        raise ValueError("window does not satisfy COLA for hop = nfft/2")
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    hop_i = int(hop)
    x_pad = np.concatenate([x[hop_i:0:-1], x, x[-2 : -hop_i - 1 : -1]])
    n_frames = (len(x_pad) - nfft) // hop + 1
    stft = np.empty((len(freqs), n_frames), dtype=np.complex128)
    for m in range(n_frames):
        idx = slice(m * hop, m * hop + nfft)
        stft[:, m] = np.fft.rfft(x_pad[idx] * win)
    return freqs, stft, win


def istft_pr(stft: np.ndarray, win: np.ndarray | None = None) -> np.ndarray:
    n_freqs, n_frames = stft.shape
    nfft = (n_freqs - 1) * 2
    if win is None:
        win = np.hanning(nfft + 1)[:-1].astype(np.float64)
    hop = nfft // 2
    x_rec = np.zeros((n_frames + 2) * hop, dtype=np.float64)
    for m in range(n_frames):
        idx = slice(m * hop, m * hop + nfft)
        frame = np.fft.irfft(stft[:, m], n=nfft)
        x_rec[idx] += frame * win
    hop_i = int(hop)
    return x_rec[hop_i : -hop_i - 1]


def decompose_spectrum(
    freqs: np.ndarray,
    spectrum: np.ndarray,
    bands: Sequence[Band],
    *,
    w: float = 0.25,
) -> np.ndarray:
    """Mask spectrum per band. spectrum 1D (n_freq,) or 2D (n_freq, n_frames)."""
    bank = band_filterbank(freqs, bands, w=w)
    if spectrum.ndim == 1:
        return spectrum[np.newaxis, :] * bank
    return spectrum[:, np.newaxis, :] * bank[:, :, np.newaxis]


def _signals_from_band_spectra(
    band_specs: np.ndarray, orig_size: int, *, stft_win: np.ndarray | None
) -> np.ndarray:
    if band_specs.ndim == 2:
        n_bands = band_specs.shape[0]
        out = np.zeros((orig_size, n_bands), dtype=np.float64)
        for i in range(n_bands):
            out[:, i] = np.fft.irfft(band_specs[i], n=orig_size)
        return out
    n_bands = band_specs.shape[2]
    out = np.zeros((orig_size, n_bands), dtype=np.float64)
    win = stft_win
    for i in range(n_bands):
        rec = istft_pr(band_specs[:, :, i], win)
        out[:, i] = rec[:orig_size]
    return out


def decompose_signal(
    x: np.ndarray,
    fs: float,
    bands: Sequence[Band],
    *,
    method: str = "rfft",
    w: float = 0.25,
    nfft: int = 128,
) -> np.ndarray:
    """
    Decompose x into one column per band. Sum along columns equals x (rfft path).
    Returns float array (n_samples, n_bands).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    orig_size = x.size
    if method == "rfft":
        freqs = np.fft.rfftfreq(orig_size, d=1.0 / fs)
        spec = np.fft.rfft(x)
        band_specs = decompose_spectrum(freqs, spec, bands, w=w)
        return _signals_from_band_spectra(band_specs, orig_size, stft_win=None)
    if method == "stft":
        freqs, stft, win = stft_pr(x, fs, nfft=nfft)
        band_specs = decompose_spectrum(freqs, stft, bands, w=w)
        return _signals_from_band_spectra(band_specs, orig_size, stft_win=win)
    raise ValueError("method must be 'rfft' or 'stft'")
