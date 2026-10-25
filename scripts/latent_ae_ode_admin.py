"""Streamlit admin: Latent AE-ODE training results per JSON config."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from eegexpcpp.experiment import (
    ROOT,
    describe_config,
    h5_metadata,
    list_config_paths,
)
from eegexpcpp.infer_native import InferError, InferSession, infer_lib_available
from eegexpcpp.latent_rollout import chunk_mse, rollout_chunked, rollout_full

os.chdir(ROOT)

st.set_page_config(page_title="Latent AE-ODE", layout="wide")
st.title("Latent AE-ODE — результати навчання")

if not infer_lib_available():
    st.error(
        "Не знайдено libeegexpcpp_infer.so. Зберіть: "
        "`./scripts/cmake_build.sh --target eegexpcpp_infer` "
        "і встановіть LD_LIBRARY_PATH на libtorch/lib."
    )
    st.stop()

entries = [describe_config(p) for p in list_config_paths()]
labels = []
for e in entries:
    mark = "✓" if e.checkpoint_exists else "✗"
    kind = "AE-ODE" if e.is_latent_ae_ode else "інший"
    labels.append(f"{mark} {e.name} ({kind})")

if not entries:
    st.warning("Немає JSON у configs/.")
    st.stop()

idx = st.sidebar.selectbox("Конфіг", range(len(entries)), format_func=lambda i: labels[i])
entry = entries[idx]

st.sidebar.markdown(f"**Шлях:** `{entry.path}`")
if entry.error:
    st.sidebar.warning(entry.error)
if entry.is_latent_ae_ode:
    st.sidebar.markdown(f"**Checkpoint:** `{entry.checkpoint_path}`")
    if entry.checkpoint_exists:
        st.sidebar.success("Чекпоінт знайдено")
    else:
        st.sidebar.error("Чекпоінт відсутній")

split = st.sidebar.radio("Дані", ["train", "val", "full"], index=0)
device = st.sidebar.selectbox("Device (C++)", ["cpu", "cuda"], index=0)

meta = {}
if entry.h5_path and entry.h5_group:
    meta = h5_metadata(entry.h5_path, entry.h5_group)
fs_default = float(meta["fs"] or 256.0)
fs = st.sidebar.number_input("Частота дискретизації (Hz)", min_value=1.0, value=fs_default)
dt = 1.0 / fs
chunk_sec = st.sidebar.number_input(
    "Довжина шматка (с)", min_value=0.05, value=1.0, step=0.05
)
chunk_samples = max(1, int(round(chunk_sec * fs)))

if not entry.is_latent_ae_ode or not entry.checkpoint_exists:
    st.info("Оберіть валідний Latent AE-ODE конфіг із наявним чекпоінтом.")
    st.stop()


@st.cache_resource
def load_session(config_path: str, dev: str) -> InferSession:
    return InferSession(config_path, device=dev)


@st.cache_data(show_spinner="Forward + RK4…")
def compute_arrays(
    config_path: str,
    dev: str,
    split_name: str,
    fs_hz: float,
    chunk_n: int,
) -> dict[str, np.ndarray | float | int]:
    dt_local = 1.0 / fs_hz
    session = load_session(config_path, dev)
    out = session.forward(split_name)
    z = out["z"]
    z_full = rollout_full(z, session, dt_local)
    z_chunk = rollout_chunked(z, session, dt_local, chunk_n)
    return {
            "x": out["x"],
            "y": out["y"],
            "z": z,
            "z_full": z_full,
            "z_chunk": z_chunk,
        "next_epoch": session.next_epoch,
    }


try:
    data = compute_arrays(
        str(entry.path), device, split, fs, chunk_samples
    )
except InferError as exc:
    st.error(str(exc))
    st.stop()

st.caption(f"next_epoch={data['next_epoch']} | split={split} | T={data['x'].shape[0]}")

x = data["x"]
y = data["y"]
z = data["z"]
t_axis = np.arange(x.shape[0], dtype=np.float64) / fs
f_dim = x.shape[1]
z_dim = z.shape[1]
feat = st.sidebar.slider("Канал (feature)", 0, max(0, f_dim - 1), 0)

max_plot = 8000
step = max(1, x.shape[0] // max_plot)
sl = slice(None, None, step)
t_plot = t_axis[sl]


def _downsample(a: np.ndarray) -> np.ndarray:
    return a[sl]


tab_rec, tab_3d, tab_rk4, tab_chunk = st.tabs(
    ["Реконструкція", "Латент 3D", "RK4 (повний)", "RK4 (шматки)"]
)

with tab_rec:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=t_plot, y=_downsample(x[:, feat]), name="X (оригінал)")
    )
    fig.add_trace(
        go.Scatter(x=t_plot, y=_downsample(y[:, feat]), name="Y (реконструкція)")
    )
    mse = float(np.mean((x[:, feat] - y[:, feat]) ** 2))
    fig.update_layout(
        title=f"Канал {feat} — MSE={mse:.6g}",
        xaxis_title="час (с)",
        yaxis_title="амплітуда",
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_3d:
    if z_dim < 3:
        st.warning("Потрібно щонайменше 3 латентні виміри.")
    else:
        fig3 = go.Figure(
            go.Scatter3d(
                x=z[sl, 0],
                y=z[sl, 1],
                z=z[sl, 2],
                mode="lines",
                line=dict(width=2, color=t_axis[sl], colorscale="Viridis"),
            )
        )
        fig3.update_layout(
            title="Латент Z (перші 3 dim)",
            scene=dict(xaxis_title="z0", yaxis_title="z1", zaxis_title="z2"),
            height=550,
        )
        st.plotly_chart(fig3, use_container_width=True)

with tab_rk4:
    dims = min(3, z_dim)
    fig_r = go.Figure()
    z_full = data["z_full"]
    for d in range(dims):
        fig_r.add_trace(
            go.Scatter(
                x=t_plot,
                y=_downsample(z[:, d]),
                name=f"Z{d} true",
                line=dict(dash="solid"),
            )
        )
        fig_r.add_trace(
            go.Scatter(
                x=t_plot,
                y=_downsample(z_full[:, d]),
                name=f"Z{d} RK4",
                line=dict(dash="dot"),
            )
        )
    rmse = float(np.sqrt(np.mean((z - z_full) ** 2)))
    fig_r.update_layout(
        title=f"Z vs RK4 (z0=Z[0]) — RMSE={rmse:.6g}",
        xaxis_title="час (с)",
        height=450,
    )
    st.plotly_chart(fig_r, use_container_width=True)

with tab_chunk:
    z_chunk = data["z_chunk"]
    dims = min(3, z_dim)
    fig_c = go.Figure()
    for d in range(dims):
        fig_c.add_trace(
            go.Scatter(
                x=t_plot,
                y=_downsample(z[:, d]),
                name=f"Z{d} true",
            )
        )
        fig_c.add_trace(
            go.Scatter(
                x=t_plot,
                y=_downsample(z_chunk[:, d]),
                name=f"Z{d} RK4 chunk",
                line=dict(dash="dot"),
            )
        )
    for t0 in range(chunk_samples, z.shape[0], chunk_samples):
        fig_c.add_vline(x=t0 / fs, line_width=1, line_dash="dash", line_color="gray")
    mses = chunk_mse(z, z_chunk, chunk_samples)
    fig_c.update_layout(
        title=f"Шматки {chunk_sec}s ({chunk_samples} samples)",
        xaxis_title="час (с)",
        height=450,
    )
    st.plotly_chart(fig_c, use_container_width=True)
    st.dataframe(
        {"chunk": range(len(mses)), "mse": mses},
        use_container_width=True,
        hide_index=True,
    )
