#!/usr/bin/env python3
"""Profile Streamlit admin compute_arrays path (Lorenz by default)."""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time
from pathlib import Path

import numpy as np

from eegexpcpp.experiment import ROOT
from eegexpcpp.infer_native import InferError, InferSession
from eegexpcpp.latent_rollout import rollout_chunked, rollout_full

DEFAULT_CFG = ROOT / "configs" / "latent_ae_ode.lorenz.json"


def ode_rhs_calls(t_len: int, chunk: int) -> tuple[int, int]:
    n_full = 4 * max(0, t_len - 1)
    n_chunk = sum(
        4 * max(0, min(chunk, t_len - t0) - 1) for t0 in range(0, t_len, chunk)
    )
    return n_full, n_chunk


def bench(
    config: Path,
    dev: str,
    split: str,
    fs_hz: float,
    chunk: int,
    repeats: int,
) -> dict[str, float | int | str]:
    dt = 1.0 / fs_hz
    t0 = time.perf_counter()
    session = InferSession(config, device=dev)
    ode = session if dev == "cpu" else InferSession(config, device="cpu")
    load_s = time.perf_counter() - t0
    t_len, f_dim = session.sizes(split)
    n_full, n_chunk = ode_rhs_calls(t_len, chunk)

    session.forward(split)
    fwd: list[float] = []
    full: list[float] = []
    chunk_t: list[float] = []
    for _ in range(repeats):
        t1 = time.perf_counter()
        out = session.forward(split)
        fwd.append(time.perf_counter() - t1)
        z = out["z"]
        t1 = time.perf_counter()
        rollout_full(z, ode, dt)
        full.append(time.perf_counter() - t1)
        t1 = time.perf_counter()
        rollout_chunked(z, ode, dt, chunk)
        chunk_t.append(time.perf_counter() - t1)

    session.close()
    if ode is not session:
        ode.close()

    med = {
        "forward": float(np.median(fwd)),
        "rk4_full": float(np.median(full)),
        "rk4_chunk": float(np.median(chunk_t)),
    }
    med["total"] = sum(med.values())
    return {
        "dev": dev,
        "split": split,
        "T": t_len,
        "F": f_dim,
        "load_s": load_s,
        "ode_full": n_full,
        "ode_chunk": n_chunk,
        **med,
    }


def profile_rk4_full(
    config: Path, dev: str, split: str, fs_hz: float
) -> str:
    dt = 1.0 / fs_hz
    session = InferSession(config, device=dev)
    ode = session if dev == "cpu" else InferSession(config, device="cpu")
    z = session.forward(split)["z"]
    pr = cProfile.Profile()
    pr.enable()
    rollout_full(z, ode, dt)
    pr.disable()
    session.close()
    if ode is not session:
        ode.close()
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("cumtime").print_stats(25)
    return buf.getvalue()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CFG)
    p.add_argument("--split", default="train", choices=("train", "val", "full"))
    p.add_argument("--fs", type=float, default=256.0)
    p.add_argument("--chunk", type=int, default=256, help="samples (admin default 1s)")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--profile", action="store_true", help="cProfile rollout_full")
    args = p.parse_args()

    print(f"config: {args.config}")
    print(f"split={args.split}  fs={args.fs}  chunk={args.chunk} samples\n")

    for dev in ("cpu", "cuda"):
        try:
            r = bench(
                args.config,
                dev,
                args.split,
                args.fs,
                args.chunk,
                args.repeats,
            )
        except InferError as exc:
            print(f"[{dev}] skipped: {exc}\n")
            continue
        rk = r["rk4_full"] + r["rk4_chunk"]
        print(f"=== forward device: {dev} (ode_rhs on CPU) ===")
        print(f"T={r['T']}  F={r['F']}  (~{r['T']/args.fs:.1f} s @ {args.fs} Hz)")
        print(f"session load: {r['load_s']:.2f} s (×2 if cuda; cached in Streamlit)")
        print(f"forward:      {r['forward']:.3f} s")
        print(f"RK4 full:     {r['rk4_full']:.3f} s  ({r['ode_full']} ode_rhs)")
        print(f"RK4 chunked:  {r['rk4_chunk']:.3f} s  ({r['ode_chunk']} ode_rhs)")
        print(f"total:        {r['total']:.3f} s  (RK4 {100*rk/r['total']:.1f}%)\n")

        if args.profile and dev == "cuda":
            print("--- cProfile rollout_full ---")
            print(profile_rk4_full(args.config, dev, args.split, args.fs))
        elif args.profile and dev == "cpu":
            print("--- cProfile rollout_full ---")
            print(profile_rk4_full(args.config, dev, args.split, args.fs))


if __name__ == "__main__":
    main()
