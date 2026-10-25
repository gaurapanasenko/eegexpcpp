"""Lorenz HDF5 export layout."""

import importlib.util
from pathlib import Path

import h5py

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "save_lorenz_h5.py"


def _load_save_lorenz_module():
    spec = importlib.util.spec_from_file_location("save_lorenz_h5", _SCRIPTS)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_lorenz_three_root_groups(tmp_path, monkeypatch):
    mod = _load_save_lorenz_module()
    out = tmp_path / "lorenz.h5"
    monkeypatch.setattr(mod, "OUT", out)
    mod.main()
    m = int(mod.M)
    t_emb = mod.N_STEPS - (m - 1) * mod.TAU
    with h5py.File(out, "r") as f:
        assert f.attrs["n_groups"] == 3
        assert f.attrs["m"] == m
        assert f.attrs["tau"] == mod.TAU
        for i in range(3):
            g = f[str(i)]
            assert g["X"].shape == g["dX"].shape == (m, t_emb)
            assert "X_mean" in g and "X_std" in g
