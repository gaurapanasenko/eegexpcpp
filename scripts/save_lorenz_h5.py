"""Integrate Lorenz (RK4) and export 3 HDF5 groups (one per state dim)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from eegexpcpp.calculus import custom_rk4
from eegexpcpp.prep import prepare_embed_signal

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "lorenz.h5"

FS = 256.0
DURATION_S = 60.0
DT = np.float32(1.0 / FS)
N_STEPS = int(FS * DURATION_S)
N_GROUPS = 3
M = 128
TAU = 1
N_LATENT = 3

IC = np.array([-10.902, -14.7626, 25.0078], dtype=np.float32)
RHO = 28.0


def lorenz_du(du: np.ndarray, u: np.ndarray, p, t: float) -> None:
    sigma, rho, beta = p
    du[0] = sigma * (u[1] - u[0])
    du[1] = u[0] * (rho - u[2]) - u[1]
    du[2] = u[0] * u[1] - beta * u[2]


def trajectory(u0: np.ndarray, rho: float) -> np.ndarray:
    p = (10.0, rho, 8.0 / 3.0)
    return custom_rk4(lorenz_du, u0.reshape(1, -1), p, float(DT), N_STEPS)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    traj = trajectory(IC, RHO)
    t_emb = N_STEPS - (M - 1) * TAU
    with h5py.File(OUT, "w") as f:
        f.attrs["system"] = "lorenz"
        f.attrs["fs"] = FS
        f.attrs["dt"] = float(DT)
        f.attrs["duration_s"] = DURATION_S
        f.attrs["n_steps"] = N_STEPS
        f.attrs["n_groups"] = N_GROUPS
        f.attrs["m"] = M
        f.attrs["tau"] = TAU
        f.attrs["n_latent"] = N_LATENT
        f.attrs["ic"] = IC
        f.attrs["rho"] = RHO
        for i in range(N_GROUPS):
            x_mat, dx_mat, _, _, x_mean, x_std = prepare_embed_signal(
                traj[:, i], N_LATENT, FS, M, TAU
            )
            g = f.create_group(str(i))
            g.create_dataset("X", data=x_mat)
            g.create_dataset("dX", data=dx_mat)
            g.create_dataset("X_mean", data=x_mean)
            g.create_dataset("X_std", data=x_std)
            print(f"group {i}: X={x_mat.shape} dX={dx_mat.shape}")
    print(f"wrote {OUT} (T_emb={t_emb})")


if __name__ == "__main__":
    main()
