"""Discover experiment JSON configs and HDF5 metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ConfigEntry:
    path: Path
    name: str
    is_latent_ae_ode: bool
    error: str | None
    h5_path: Path | None
    h5_group: str | None
    checkpoint_dir: Path | None
    checkpoint_path: Path | None
    checkpoint_exists: bool


def list_config_paths(configs_dir: Path | None = None) -> list[Path]:
    base = configs_dir or (ROOT / "configs")
    if not base.is_dir():
        return []
    return sorted(base.glob("*.json"))


def _resolve_from_root(rel: str) -> Path:
    return (ROOT / rel).resolve()


def describe_config(path: Path) -> ConfigEntry:
    name = path.name
    try:
        with path.open(encoding="utf-8") as f:
            j = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return ConfigEntry(
            path=path,
            name=name,
            is_latent_ae_ode=False,
            error=str(e),
            h5_path=None,
            h5_group=None,
            checkpoint_dir=None,
            checkpoint_path=None,
            checkpoint_exists=False,
        )

    required = ("data", "model", "loss", "train")
    missing = [k for k in required if k not in j]
    if missing:
        return ConfigEntry(
            path=path,
            name=name,
            is_latent_ae_ode=False,
            error=f"missing keys: {', '.join(missing)}",
            h5_path=None,
            h5_group=None,
            checkpoint_dir=None,
            checkpoint_path=None,
            checkpoint_exists=False,
        )

    data = j["data"]
    train = j["train"]
    h5_rel = data.get("h5_path")
    h5_group = data.get("h5_group", "0")
    ck_dir_rel = train.get("checkpoint_dir", "checkpoints")
    h5_path = _resolve_from_root(h5_rel) if h5_rel else None
    checkpoint_dir = _resolve_from_root(ck_dir_rel)
    checkpoint_path = checkpoint_dir / "latent_ae_ode.pt"
    return ConfigEntry(
        path=path.resolve(),
        name=name,
        is_latent_ae_ode=True,
        error=None,
        h5_path=h5_path,
        h5_group=str(h5_group),
        checkpoint_dir=checkpoint_dir,
        checkpoint_path=checkpoint_path,
        checkpoint_exists=checkpoint_path.is_file(),
    )


def h5_metadata(h5_path: Path, h5_group: str) -> dict[str, float | int | None]:
    meta: dict[str, float | int | None] = {
        "fs": None,
        "dt": None,
        "t_steps": None,
        "n_features": None,
    }
    if not h5_path.is_file():
        return meta
    with h5py.File(h5_path, "r") as f:
        if "fs" in f.attrs:
            meta["fs"] = float(f.attrs["fs"])
        if "dt" in f.attrs:
            meta["dt"] = float(f.attrs["dt"])
        grp = f[h5_group.lstrip("/")]
        if "X" in grp:
            sh = grp["X"].shape
            meta["n_features"] = int(sh[0])
            meta["t_steps"] = int(sh[1])
    if meta["fs"] is None and meta["dt"] is not None:
        meta["fs"] = 1.0 / float(meta["dt"])
    return meta
