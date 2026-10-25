"""SINDy Xi CSV from libeegexpcpp_infer."""

from __future__ import annotations

import csv
from io import StringIO
from typing import TYPE_CHECKING

from ctypes import c_char, c_int, c_int64, c_void_p

if TYPE_CHECKING:
    from eegexpcpp.infer_native import InferSession


def sindy_coefficients_csv(session: InferSession) -> str:
    """Return Xi table as UTF-8 CSV (header + rows)."""
    lib = session._lib
    handle = session._handle
    assert handle is not None

    size = int(lib.eegexpcpp_infer_sindy_csv_size(handle))
    if size < 0:
        session._check(-1)
    if size == 0:
        raise ValueError("model ode is not sindy")

    buf = (c_char * size)()
    session._check(lib.eegexpcpp_infer_sindy_csv(handle, buf, c_int64(size)))
    return buf.value.decode("utf-8")


def parse_sindy_coefficients_csv(text: str) -> dict[str, list[float | str]]:
    """Parse CSV from infer into column dict for st.dataframe."""
    reader = csv.DictReader(StringIO(text))
    if reader.fieldnames is None:
        raise ValueError("empty sindy csv")
    columns: dict[str, list[float | str]] = {
        name: [] for name in reader.fieldnames
    }
    for row in reader:
        for name in reader.fieldnames:
            raw = row[name]
            if name == "feature":
                columns[name].append(raw)
            else:
                columns[name].append(float(raw))
    return columns


def sindy_coefficients_table(session: InferSession) -> dict[str, list[float | str]]:
    """Return columns for a dataframe: feature, dz0, dz1, …"""
    return parse_sindy_coefficients_csv(sindy_coefficients_csv(session))
