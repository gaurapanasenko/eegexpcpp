"""Embed signal preparation and HDF5 I/O."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import h5py
import numpy as np

from eegexpcpp.calculus import diff4_order1
from eegexpcpp.signal import embed_sig, zscore
from eegexpcpp.stft import EEG_BANDS, Band, decompose_signal

_BAND_RECON_RTOL = 1e-6
_BAND_RECON_ATOL = 1e-5


def _assert_bands_reconstruct_signal(
    x: np.ndarray, per_band: np.ndarray, *, component: int
) -> None:
    recon = per_band.sum(axis=1)
    if not np.allclose(
        recon, x, rtol=_BAND_RECON_RTOL, atol=_BAND_RECON_ATOL
    ):
        err = float(np.max(np.abs(recon - x)))
        raise ValueError(
            f"band sum does not match component {component} "
            f"(max |error|={err})"
        )


__all__ = [
    "prepare_embed_signal",
    "load_eeglab_icaact",
    "load_eeglab_icaact_component",
    "save_prepare_hdf5",
]


def prepare_embed_signal(
    x: np.ndarray,
    n_latent: int,
    fs: float,
    m: int,
    tau: int = 1,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, int, np.float32, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float32).ravel()
    dt = np.float32(1.0 / float(fs))
    x_norm, x_m, x_s = zscore(x, dim=1)
    dx = diff4_order1(x_norm, float(dt), dim=1).astype(np.float32, copy=False)
    x_emb = embed_sig(x_norm, m, tau)
    dx_emb = embed_sig(dx, m, tau)
    x_mat = np.ascontiguousarray(x_emb.T, dtype=np.float32)
    dx_mat = np.ascontiguousarray(dx_emb.T, dtype=np.float32)
    return x_mat, dx_mat, int(n_latent), dt, x_m.astype(np.float32), x_s.astype(
        np.float32
    )


def load_eeglab_icaact(mat_path: str | Path) -> tuple[np.ndarray, np.float32]:
    with h5py.File(mat_path, "r") as f:
        icaact = np.asarray(f["icaact"], dtype=np.float32)
        fs = float(np.asarray(f["srate"]).squeeze())
    if icaact.ndim != 2:
        raise ValueError(f"expected 2D icaact, got shape {icaact.shape}")
    return np.ascontiguousarray(icaact), np.float32(fs)


def load_eeglab_icaact_component(
    mat_path: str | Path,
    component: int = 0,
) -> tuple[np.ndarray, np.float32]:
    icaact, fs = load_eeglab_icaact(mat_path)
    return np.ascontiguousarray(icaact[:, component], dtype=np.float32), fs


def save_prepare_hdf5(
    path: str | Path,
    icaact: np.ndarray,
    fs: float,
    n_latent: int,
    m: int,
    tau: int = 1,
    bands: Sequence[Band] = EEG_BANDS,
) -> None:
    """
    Layout: /{s}/{b}/X, dX, X_mean, X_std; group attrs band_name;
    root attrs fs, dt, m, tau, n_latent, n_components, n_bands.
    """
    icaact = np.asarray(icaact, dtype=np.float32)
    if icaact.ndim != 2:
        raise ValueError(f"icaact must be (time, components), got {icaact.shape}")
    n_comp = icaact.shape[1]
    band_names = [name for name, _, _ in bands]
    n_bands = len(band_names)
    dt = float(1.0 / float(fs))

    with h5py.File(path, "w") as f:
        f.attrs["n_latent"] = int(n_latent)
        f.attrs["m"] = int(m)
        f.attrs["tau"] = int(tau)
        f.attrs["fs"] = float(fs)
        f.attrs["dt"] = dt
        f.attrs["n_components"] = n_comp
        f.attrs["n_bands"] = n_bands

        for s in range(n_comp):
            x_s = icaact[:, s]
            per_band = decompose_signal(x_s, fs, bands, method="rfft")
            _assert_bands_reconstruct_signal(x_s, per_band, component=s)
            for b in range(n_bands):
                g = f.create_group(f"{s}/{b}")
                g.attrs["band_name"] = band_names[b]
                x_mat, dx_mat, _, _, x_mean, x_std = prepare_embed_signal(
                    per_band[:, b], n_latent, fs, m, tau
                )
                g.create_dataset("X", data=x_mat)
                g.create_dataset("dX", data=dx_mat)
                g.create_dataset("X_mean", data=x_mean)
                g.create_dataset("X_std", data=x_std)
