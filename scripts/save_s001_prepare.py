"""Build HDF5 prep from S001R01_8.mat (all icaact components, per band)."""

from pathlib import Path

from eegexpcpp.prep import load_eeglab_icaact, save_prepare_hdf5

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MAT = DATA / "S001R01_8.mat"
OUT = DATA / "S001R01_8_prepare.h5"

M = 128
TAU = 1
N_LATENT = 12


def main() -> None:
    icaact, fs = load_eeglab_icaact(MAT)
    save_prepare_hdf5(OUT, icaact, fs, N_LATENT, M, TAU)
    with __import__("h5py").File(OUT, "r") as f:
        x = f["0/0/X"]
        print(
            f"wrote {OUT} groups={f.attrs['n_components']}x{f.attrs['n_bands']} "
            f"example 0/0/X={x.shape} fs={f.attrs['fs']} dt={f.attrs['dt']}"
        )


if __name__ == "__main__":
    main()
