"""The reader must accept every common container and layout."""
import json
import os
import zipfile

import numpy as np
import pandas as pd
import pytest

from barista import io

V = np.geomspace(0.01, 1, 12)


def _I(T, d):
    return 1e-9 * V * np.exp(-d / 3) * (T / 300)


def check(ds, nT=None, nd=None, npts=None):
    if nT is not None:
        assert len(ds.temperatures) == nT, ds.temperatures
    if nd is not None:
        assert len(ds.thicknesses) == nd, ds.thicknesses
    if npts is not None:
        assert len(ds.df) == npts
    assert (ds.df.I.abs() > 0).all()


def test_long_csv(tmp_path):
    rows = [(v, i, T, d) for T in (100, 300) for d in (2, 4) for v, i in zip(V, _I(T, d))]
    p = tmp_path / "long.csv"
    pd.DataFrame(rows, columns=["Voltage (V)", "Current (nA)", "Temperature (K)", "Thickness (nm)"]).assign(
        **{"Current (nA)": lambda df: df["Current (nA)"] * 1e9}).to_csv(p, index=False)
    ds = io.load(str(p))
    check(ds, 2, 2, 48)
    assert np.isclose(ds.df.I.max(), _I(300, 2).max())


def test_keithley_header_and_semicolon_decimal_comma(tmp_path):
    p = tmp_path / "sample_A.txt"
    lines = ["Keithley 2400 sweep", "Sample: A", "Temperature: 77 K", "Thickness = 5 nm", "",
             "Voltage;Current"]
    lines += [f"{v:.6f};{i:.6e}".replace(".", ",") for v, i in zip(V, _I(77, 5))]
    p.write_text("\n".join(lines))
    ds = io.load(str(p))
    check(ds, 1, 1, 12)
    assert ds.temperatures[0] == 77 and ds.thicknesses[0] == 5


def test_filename_metadata_and_folder(tmp_path):
    for T in (10, 150, 300):
        np.savetxt(tmp_path / f"IV_{T}K_3nm.dat", np.c_[V, _I(T, 3)], header="V I")
    ds = io.load(str(tmp_path))
    check(ds, 3, 1, 36)


def test_wide_format_and_excel_sheets(tmp_path):
    p = tmp_path / "wide.xlsx"
    with pd.ExcelWriter(p) as w:
        for d in (2, 6):
            df = pd.DataFrame({"V (V)": V, **{f"I_{T}K (A)": _I(T, d) for T in (77, 200, 300)}})
            df.to_excel(w, sheet_name=f"d={d}nm", index=False)
    ds = io.load(str(p))
    check(ds, 3, 2, 72)


def test_paired_columns(tmp_path):
    p = tmp_path / "pairs.csv"
    pd.DataFrame({"V1": V, "I_100K": _I(100, 4), "V2": V, "I_300K": _I(300, 4)}).to_csv(p, index=False)
    ds = io.load(str(p), io.DeviceInfo(thickness_nm=4))
    check(ds, 2, None, 24)


def test_json_nested(tmp_path):
    obj = {"sample": "x", "area_um2": 100,
           "curves": [{"T": T, "d": 3, "V": V.tolist(), "I": _I(T, 3).tolist()} for T in (50, 300)]}
    p = tmp_path / "data.json"
    p.write_text(json.dumps(obj))
    check(io.load(str(p)), 2, 1, 24)


def test_hdf5(tmp_path):
    h5py = pytest.importorskip("h5py")
    p = tmp_path / "data.h5"
    with h5py.File(p, "w") as f:
        for T in (4, 300):
            g = f.create_group(f"T_{T}K")
            g.attrs["thickness_nm"] = 2.0
            g["V"] = V
            g["I"] = _I(T, 2)
    check(io.load(str(p)), 2, 1, 24)


def test_mat_npz_parquet_zip(tmp_path):
    from scipy.io import savemat
    savemat(tmp_path / "a_100K_2nm.mat", {"V": V, "I": _I(100, 2)})
    np.savez(tmp_path / "b_200K_2nm.npz", V=V, I=_I(200, 2))
    try:
        pd.DataFrame({"V": V, "I": _I(300, 2), "T": 300, "d": 2}).to_parquet(tmp_path / "c.parquet")
        n = 3
    except Exception:
        n = 2
    z = tmp_path / "all.zip"
    with zipfile.ZipFile(z, "w") as zz:
        for f in os.listdir(tmp_path):
            if f != "all.zip":
                zz.write(tmp_path / f, f)
    check(io.load(str(z)), n, 1, 12 * n)


def test_current_density_units(tmp_path):
    p = tmp_path / "j.csv"
    pd.DataFrame({"V (mV)": V * 1e3, "J (A/cm^2)": _I(300, 2), "T (K)": 300, "d (A)": 20}).to_csv(p, index=False)
    ds = io.load(str(p))
    assert ds.density and np.isclose(ds.df.V.max(), 1.0) and ds.thicknesses[0] == 2.0
    assert np.isclose(ds.df.I.max(), _I(300, 2).max() * 1e4)
