"""Reading I(V, T, d) data from (almost) any file.

Supported containers
--------------------
* delimited text of any flavour: .csv .tsv .txt .dat .prn .asc .iv .log .out and unknown
  extensions — comma, tab, semicolon, pipe or whitespace separated, decimal commas,
  instrument headers (Keithley, Keysight/Agilent B15xx, Lakeshore, LabVIEW, Origin ASCII
  exports) and several data blocks per file (one per sweep / temperature);
* spreadsheets: .xlsx .xlsm .xls .ods (every sheet; header row found automatically);
* structured data: .json, .h5/.hdf5/.nxs, .mat (MATLAB), .npy/.npz, .parquet,
  .feather, .pkl;
* archives (.zip, .tar, .tar.gz, .tgz) and folders, read recursively.

Supported layouts
-----------------
* long format: columns V, I (or J), T, d (any order, any recognised alias);
* wide format: one bias column and many current columns (``I_300K``, ``J 5nm``, ``77 K``);
* paired format: V1, I1, V2, I2, ...;
* one curve per file/sheet/block/group with T and d in a column, the header text,
  a metadata line (``Temperature: 77 K``, ``# thickness = 5 nm``), the sheet/group
  name, the file name (``IV_300K_12nm.csv``) or the device information.

Units in headers are honoured: ``I (nA)``, ``J [A/cm^2]``, ``V (mV)``, ``d (Å)``,
``T (°C)``, ``area (um2)``.
"""
from __future__ import annotations

import csv
import io as _io
import json
import os
import re
import shutil
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .device import DeviceInfo

TEXT_EXT = {".csv", ".tsv", ".txt", ".dat", ".prn", ".asc", ".iv", ".log", ".out", ".data", ".xy", ""}
EXCEL_EXT = {".xlsx", ".xlsm", ".xls", ".ods"}
ARCHIVE_EXT = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2")
SKIP_EXT = {".png", ".jpg", ".jpeg", ".pdf", ".gif", ".svg", ".py", ".md", ".html", ".yaml", ".yml", ".ipynb",
            ".pptx", ".docx", ".tif", ".tiff", ".bmp"}

ROLE_ALIASES = {
    "V": ["v", "bias", "voltage", "vbias", "v_bias", "u", "vsd", "vds", "v_sd", "v_ds", "v_applied",
          "applied_voltage", "vapp", "vsource", "vs", "source_voltage", "smu1_v", "va", "v1"],
    "I": ["i", "current", "id", "isd", "ids", "i_sd", "i_ds", "i_meas", "measured_current", "imeas",
          "smu1_i", "ia", "i1", "drain_current", "i_d"],
    "J": ["j", "current_density", "jdens", "density", "jd"],
    "T": ["t", "temp", "temperature", "tk", "t_k", "tsample", "t_sample", "sample_temperature"],
    "d": ["d", "thickness", "thick", "t_nm", "d_nm", "barrier_thickness", "layer_thickness", "tox", "t_ox"],
    "N": ["n_layers", "layers", "nlayers", "layer_number", "n_l"],
    "area": ["area", "a", "junction_area", "device_area"],
    "sweep": ["sweep", "direction", "sweep_dir"],
    "rate": ["rate", "sweep_rate", "ramp_rate"],
    "H": ["h", "b", "field", "magnetic_field", "bfield"],
    "gate": ["vg", "v_g", "gate", "vgate", "gate_voltage", "vgs"],
    "time": ["time", "t_s", "timestamp", "elapsed", "elapsed_time", "seconds"],
    "index": ["index", "idx", "point", "no", "n", "#", "num", "number", "unnamed_0", "point_number"],
}
PREFIX_ROLES = [("curr", "I"), ("volt", "V"), ("bias", "V"), ("temp", "T"), ("thick", "d"),
                ("dens", "J"), ("time", "time"), ("gate", "gate")]
_PREFIX = {"f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6, "µ": 1e-6, "μ": 1e-6, "m": 1e-3, "": 1.0, "k": 1e3}

_T_TOKEN = re.compile(r"(?<![A-Za-z0-9.])(\d+(?:[.p]\d+)?)\s*(K|°C|degC)(?![A-Za-z])")
_D_TOKEN = re.compile(r"(?<![A-Za-z0-9.])(\d+(?:[.p]\d+)?)\s*(nm|Å|A|Ang|ang|um|µm)(?![A-Za-z0-9/])")
_N_TOKEN = re.compile(r"(?<![A-Za-z0-9.])(\d+)\s*(?:L|layers?|ML)(?![A-Za-z])")
_META_T = re.compile(r"(?:temperature|temp|\bT(?:_?sample)?)\s*[:=]\s*([-+]?\d+(?:\.\d+)?)\s*(K|°C|C)?", re.I)
_META_D = re.compile(r"(?:thickness|thick|\bd|\bt_?ox)\s*[:=]\s*(\d+(?:\.\d+)?)\s*(nm|Å|A|um|µm)?", re.I)
_META_A = re.compile(r"(?:area)\s*[:=]\s*(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*(um2|µm2|um\^2|µm\^2|m2|cm2|mm2|nm2)?", re.I)


# ---------------------------------------------------------------------------
# small parsers
# ---------------------------------------------------------------------------
def _num(s):
    try:
        return float(str(s).replace("p", ".")) if s is not None else None
    except ValueError:
        return None


def tokens_from_text(text: str) -> dict:
    """T / d / N tokens in free text such as a header, a sheet name or a file name."""
    out = {}
    t = str(text)
    m = _T_TOKEN.search(t)
    if m:
        v = _num(m.group(1))
        out["T"] = v + 273.15 if m.group(2) in ("°C", "degC") else v
    m = _D_TOKEN.search(t)
    if m:
        v = _num(m.group(1))
        unit = m.group(2).lower()
        out["d"] = v * (0.1 if unit in ("å", "a", "ang") else 1e3 if unit in ("um", "µm") else 1.0)
    m = _N_TOKEN.search(t)
    if m and "d" not in out:
        out["N"] = int(m.group(1))
    return out


def parse_filename(path: str) -> dict:
    return tokens_from_text(os.path.splitext(os.path.basename(str(path)))[0].replace("_", " "))


def meta_from_lines(lines) -> dict:
    """Key-value metadata from instrument header lines."""
    out = {}
    for ln in lines:
        s = str(ln)
        m = _META_T.search(s)
        if m and "T" not in out:
            v = float(m.group(1))
            out["T"] = v + 273.15 if (m.group(2) or "").upper().endswith("C") else v
        m = _META_D.search(s)
        if m and "d" not in out:
            v = float(m.group(1))
            u = (m.group(2) or "nm").lower()
            out["d"] = v * (0.1 if u in ("å", "a") else 1e3 if u in ("um", "µm") else 1.0)
        m = _META_A.search(s)
        if m and "area_m2" not in out:
            v = float(m.group(1))
            u = (m.group(2) or "um2").lower().replace("^", "")
            out["area_m2"] = v * {"um2": 1e-12, "µm2": 1e-12, "m2": 1.0, "cm2": 1e-4, "mm2": 1e-6, "nm2": 1e-18}[u]
    return out


def _split_header(col):
    c = str(col).strip()
    m = re.match(r"^(.*?)[\s_]*[\(\[]\s*([^\)\]]*)\s*[\)\]](.*)$", c)
    if m:
        name, unit, rest = m.group(1), m.group(2), m.group(3)
        name = (name + " " + rest).strip()
    else:
        name, unit = c, ""
    return name, unit.strip()


def _clean(name: str) -> str:
    n = name.lower()
    n = re.sub(r"(?<![a-z])\d+(?:\.\d+)?\s*(k|nm|å|ang|um|µm|°c|degc|l|ml|layers?)(?![a-z])", " ", n)
    n = re.sub(r"[@:=]", " ", n)
    n = re.sub(r"[\s\-\.]+", "_", n.strip()).strip("_")
    return n


def column_role(col) -> tuple:
    """Return (role, unit, header-tokens) for a column header."""
    name, unit = _split_header(col)
    toks = tokens_from_text(str(col))
    base = _clean(name)
    u = unit.lower()
    candidates = [base, re.sub(r"_?\d+$", "", base), base.split("_")[0]]
    for cand in candidates:
        if not cand:
            continue
        for role, al in ROLE_ALIASES.items():
            if cand in al:
                if role == "T" and u in ("nm", "a", "å", "um"):
                    return "d", unit, toks
                if role == "T" and u in ("s", "sec", "ms", "min"):
                    return "time", unit, toks
                if role == "index" and cand == "n" and u:
                    continue
                return role, unit, toks
        for pre, role in PREFIX_ROLES:
            if cand.startswith(pre):
                return role, unit, toks
    # unit-only clues
    if re.fullmatch(r"[fpnuµμmk]?a", u):
        return "I", unit, toks
    if re.fullmatch(r"[fpnuµμmk]?a/(cm2|m2|mm2|um2|cm\^2|m\^2)", u.replace("²", "2")):
        return "J", unit, toks
    if re.fullmatch(r"[munµk]?v", u):
        return "V", unit, toks
    if u == "k" or u in ("°c", "degc"):
        return "T", unit, toks
    if u in ("nm", "å"):
        return "d", unit, toks
    # bare numbers or tokens as headers -> a curve label
    if toks or _num(name) is not None:
        return "label", unit, toks
    return "other", unit, toks


def _unit_factor(kind: str, unit: str) -> float:
    u = unit.replace(" ", "").replace("^", "").replace("²", "2")
    if kind == "V":
        m = re.fullmatch(r"([fpnuµμmk]?)V", u, re.I)
        return _PREFIX[m.group(1).lower() if m.group(1) not in ("µ", "μ") else "u"] if m else 1.0
    if kind == "I":
        m = re.fullmatch(r"([fpnuµμmk]?)A", u)
        return _PREFIX[m.group(1)] if m else 1.0
    if kind == "J":
        m = re.fullmatch(r"([fpnuµμmk]?)A/(cm2|m2|mm2|um2|µm2|nm2)", u)
        if not m:
            return 1e4 if "cm" in u else 1.0
        area = {"cm2": 1e-4, "m2": 1.0, "mm2": 1e-6, "um2": 1e-12, "µm2": 1e-12, "nm2": 1e-18}[m.group(2)]
        return _PREFIX[m.group(1)] / area
    if kind == "d":
        return {"": 1.0, "nm": 1.0, "a": 0.1, "å": 0.1, "ang": 0.1, "angstrom": 0.1, "um": 1e3, "µm": 1e3,
                "m": 1e9, "cm": 1e7}.get(u.lower(), 1.0)
    if kind == "area":
        return {"": 1e-12, "um2": 1e-12, "µm2": 1e-12, "m2": 1.0, "cm2": 1e-4, "mm2": 1e-6,
                "nm2": 1e-18}.get(u.lower(), 1e-12)
    return 1.0


# ---------------------------------------------------------------------------
# grid -> tables (handles instrument headers and multiple data blocks)
# ---------------------------------------------------------------------------
_DEC_COMMA = re.compile(r"^[-+]?\d+,\d+(?:[eE][-+]?\d+)?$")


def _to_float(x):
    if x is None:
        return None
    if isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, (bool, np.bool_)):
        try:
            xf = float(x)
        except (TypeError, ValueError, OverflowError):
            return None
        return xf if np.isfinite(xf) else None
    s = str(x).strip().strip('"').strip("'")
    if not s:
        return None
    if _DEC_COMMA.match(s):
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _grid_tables(grid, src: str, meta: dict):
    """Split a 2-D cell grid into one or more numeric tables with headers and metadata."""
    rows = [[c for c in r] for r in grid]
    width = max((len(r) for r in rows), default=0)
    rows = [r + [None] * (width - len(r)) for r in rows]
    isnum = [sum(_to_float(c) is not None for c in r) >= 2 for r in rows]
    tables = []
    i = 0
    text_above = []
    while i < len(rows):
        if not isnum[i]:
            text_above.append(rows[i])
            i += 1
            continue
        j = i
        while j < len(rows) and isnum[j]:
            j += 1
        if j - i >= 3:
            block = rows[i:j]
            # header: last text row above with >= 2 non-empty cells
            header = None
            hdr_idx = None
            for k in range(len(text_above) - 1, -1, -1):
                cells = [c for c in text_above[k] if c not in (None, "") and str(c).strip()]
                if len(cells) >= 2:
                    header, hdr_idx = text_above[k], k
                    break
            meta_lines = [" ".join(str(c) for c in r if c not in (None, "")) for r in text_above]
            bmeta = dict(meta)
            bmeta.update(meta_from_lines(meta_lines))
            for ln in meta_lines[:hdr_idx] if hdr_idx is not None else meta_lines:
                for k2, v2 in tokens_from_text(ln).items():
                    bmeta.setdefault(k2, v2)
            ncol = max(sum(c is not None and str(c).strip() != "" for c in r) for r in block)
            ncol = max(ncol, 2)
            if header is not None:
                # optional units row directly below the header (e.g. Origin long name / units)
                names = [str(c).strip() if c not in (None, "") else f"col{k}" for k, c in enumerate(header)]
            else:
                default = ["V", "I", "T", "d"]
                names = [default[k] if k < 4 else f"col{k}" for k in range(width)]
            data = [[_to_float(c) for c in r] for r in block]
            df = pd.DataFrame(data, columns=names[:width] if len(names) >= width else
                              names + [f"col{k}" for k in range(len(names), width)])
            df = df.loc[:, df.notna().any()]
            if df.shape[1] >= 2:
                tables.append((df, bmeta, src + (f"#block{len(tables) + 1}" if tables else "")))
            text_above = []
        else:
            text_above.extend(rows[i:j])
        i = j
    # label multi-block sources consistently
    if len(tables) > 1:
        tables = [(t, m, f"{src}#block{k + 1}") for k, (t, m, _) in enumerate(tables)]
    return tables


def _sniff_split(lines):
    num_lines = [ln for ln in lines if re.search(r"\d", ln)][:200]
    best, best_score = None, -1
    for delim in [",", "\t", ";", "|", None]:
        counts = []
        for ln in num_lines:
            parts = ln.split() if delim is None else next(csv.reader([ln], delimiter=delim))
            counts.append(sum(_to_float(p) is not None for p in parts))
        if not counts:
            continue
        good = [c for c in counts if c >= 2]
        if not good:
            continue
        mode = max(set(good), key=good.count)
        score = good.count(mode) * mode
        if delim == "," and any(_DEC_COMMA.match(p.strip()) for ln in num_lines[:20] for p in ln.split()):
            score *= 0.5        # decimal commas: prefer whitespace/semicolon
        if score > best_score:
            best, best_score = delim, score
    return best


def _read_text(path, src, meta):
    with open(path, "r", errors="replace", encoding="utf-8-sig") as fh:
        raw = fh.read()
    if "\x00" in raw[:2000]:
        with open(path, "r", errors="replace", encoding="utf-16") as fh:
            raw = fh.read()
    lines = [ln.rstrip("\r\n") for ln in raw.splitlines()]
    lines = [ln for ln in lines if ln.strip()]
    delim = _sniff_split(lines)
    grid = []
    for ln in lines:
        s = ln.lstrip("#%;/! ").rstrip() if not re.match(r"^\s*[-+]?\d", ln) else ln
        if delim is None:
            grid.append(s.split())
        else:
            try:
                grid.append(next(csv.reader([s], delimiter=delim)))
            except StopIteration:
                grid.append([])
        # header lines may be space separated even if data are tab separated
        if delim is not None and len(grid[-1]) == 1 and len(s.split()) > 1 and not re.match(r"^\s*[-+]?\d", s):
            grid[-1] = s.split()
    return _grid_tables(grid, src, meta)


def _read_excel(path, src, meta):
    engine = "odf" if path.lower().endswith(".ods") else None
    sheets = pd.read_excel(path, sheet_name=None, header=None, engine=engine)
    out = []
    for name, g in sheets.items():
        m = dict(meta)
        m.update({k: v for k, v in tokens_from_text(name).items() if k not in m})
        grid = g.where(pd.notna(g), None).values.tolist()
        out.extend(_grid_tables(grid, f"{src}[{name}]", m))
    return out


# ---- structured containers ------------------------------------------------
def _scalar_meta(key: str, val) -> dict:
    k = str(key).lower()
    try:
        v = float(np.asarray(val).ravel()[0])
    except Exception:
        return tokens_from_text(str(val)) if isinstance(val, str) else {}
    role, unit, _ = column_role(key)
    if role == "T":
        return {"T": v + 273.15 if unit.lower().endswith("c") else v}
    if role == "d":
        return {"d": v * _unit_factor("d", unit)}
    if role == "area":
        return {"area_m2": v * _unit_factor("area", unit)}
    return {}


def _walk(obj, path, meta, src, out):
    """Find tables inside nested dict / list structures (JSON, HDF5, MAT, NPZ)."""
    if isinstance(obj, pd.DataFrame):
        out.append((obj, meta, src + (":" + path if path else "")))
        return
    if isinstance(obj, np.ndarray) and obj.ndim == 2 and obj.dtype.kind in "fiu":
        a = obj if obj.shape[0] >= obj.shape[1] else obj.T
        if a.shape[1] >= 2 and a.shape[0] >= 3:
            names = ["V", "I", "T", "d"][: a.shape[1]] + [f"col{k}" for k in range(4, a.shape[1])]
            out.append((pd.DataFrame(a, columns=names), meta, src + (":" + path if path else "")))
        return
    if isinstance(obj, (list, tuple)):
        if obj and all(isinstance(x, dict) for x in obj):
            flat = all(not isinstance(v, (list, dict, np.ndarray)) for x in obj for v in x.values())
            if flat:
                out.append((pd.DataFrame(obj), meta, src + (":" + path if path else "")))
                return
            for k, x in enumerate(obj):
                _walk(x, f"{path}[{k}]", meta, src, out)
        elif obj and all(isinstance(x, (list, tuple)) for x in obj):
            _walk(np.asarray(obj, float), path, meta, src, out)
        return
    if isinstance(obj, dict):
        m = dict(meta)
        m.update({k: v for k, v in tokens_from_text(path.split("/")[-1] if path else "").items() if k not in m})
        arrays, children = {}, {}
        for k, v in obj.items():
            if isinstance(v, (dict, pd.DataFrame)) or (isinstance(v, (list, tuple)) and v and isinstance(v[0], (dict, list, tuple))):
                children[k] = v
            elif isinstance(v, (list, tuple, np.ndarray)) and np.asarray(v).ndim == 1 and len(v) >= 3 \
                    and np.asarray(v).dtype.kind in "fiu":
                arrays[k] = np.asarray(v, float)
            elif isinstance(v, np.ndarray) and v.ndim == 2 and v.dtype.kind in "fiu":
                children[k] = v
            else:
                m.update(_scalar_meta(k, v))
        if len(arrays) >= 2:
            n = max(len(a) for a in arrays.values())
            cols = {k: a for k, a in arrays.items() if len(a) == n}
            if len(cols) >= 2:
                out.append((pd.DataFrame(cols), m, src + (":" + path if path else "")))
        for k, v in children.items():
            _walk(v, f"{path}/{k}" if path else str(k), m, src, out)


def _read_json(path, src, meta):
    with open(path) as fh:
        obj = json.load(fh)
    out = []
    _walk(obj, "", meta, src, out)
    return out


def _h5_to_dict(g):
    d = {}
    for k, v in g.attrs.items():
        d[k] = v.decode() if isinstance(v, bytes) else v
    for k, v in g.items():
        if hasattr(v, "keys"):
            d[k] = _h5_to_dict(v)
        else:
            arr = v[()]
            d[k] = arr
            for ak, av in v.attrs.items():
                d.setdefault(f"{k}_{ak}", av)
    return d


def _read_hdf5(path, src, meta):
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("Reading HDF5 needs h5py:  pip install h5py") from exc
    with h5py.File(path, "r") as f:
        obj = _h5_to_dict(f)
    out = []
    _walk(obj, "", meta, src, out)
    return out


def _mat_to_py(x):
    from scipy.io.matlab import mat_struct
    if isinstance(x, mat_struct):
        return {k: _mat_to_py(getattr(x, k)) for k in x._fieldnames}
    if isinstance(x, np.ndarray) and x.dtype == object:
        return [_mat_to_py(v) for v in x.ravel()]
    return x


def _read_mat(path, src, meta):
    from scipy.io import loadmat
    try:
        raw = loadmat(path, squeeze_me=True, struct_as_record=False)
    except NotImplementedError:          # v7.3 files are HDF5
        return _read_hdf5(path, src, meta)
    obj = {k: _mat_to_py(v) for k, v in raw.items() if not k.startswith("__")}
    out = []
    _walk(obj, "", meta, src, out)
    return out


def _read_numpy(path, src, meta):
    if path.lower().endswith(".npz"):
        with np.load(path, allow_pickle=False) as z:
            obj = {k: z[k] for k in z.files}
    else:
        obj = {"array": np.load(path, allow_pickle=False)}
    out = []
    _walk(obj, "", meta, src, out)
    return out


def _read_pandas(path, src, meta):
    p = path.lower()
    if p.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif p.endswith(".feather"):
        df = pd.read_feather(path)
    else:
        df = pd.read_pickle(path)
    return [(df, meta, src)]


def read_tables(path: str, _tmpdirs=None):
    """Read any supported file/folder/archive into a list of (table, meta, source)."""
    path = str(path)
    if os.path.isdir(path):
        out = []
        for root, _, files in os.walk(path):
            for f in sorted(files):
                if f.startswith(".") or os.path.splitext(f)[1].lower() in SKIP_EXT:
                    continue
                full = os.path.join(root, f)
                try:
                    out.extend(read_tables(full, _tmpdirs))
                except Exception as exc:     # keep going on unreadable side files
                    out.append((None, {"error": f"{f}: {exc}"}, f))
        return out
    low = path.lower()
    src = os.path.basename(path)
    meta = parse_filename(path)
    if low.endswith(ARCHIVE_EXT):
        tmp = tempfile.mkdtemp(prefix="barista_")
        if _tmpdirs is not None:
            _tmpdirs.append(tmp)
        if low.endswith(".zip"):
            with zipfile.ZipFile(path) as z:
                z.extractall(tmp)
        else:
            with tarfile.open(path) as t:
                t.extractall(tmp, filter="data")
        return read_tables(tmp, _tmpdirs)
    ext = os.path.splitext(low)[1]
    if ext in EXCEL_EXT:
        return _read_excel(path, src, meta)
    if ext == ".json":
        return _read_json(path, src, meta)
    if ext in (".h5", ".hdf5", ".hdf", ".nxs", ".he5"):
        return _read_hdf5(path, src, meta)
    if ext == ".mat":
        return _read_mat(path, src, meta)
    if ext in (".npy", ".npz"):
        return _read_numpy(path, src, meta)
    if ext in (".parquet", ".feather", ".pkl", ".pickle"):
        return _read_pandas(path, src, meta)
    return _read_text(path, src, meta)


# ---------------------------------------------------------------------------
# tables -> standard long format
# ---------------------------------------------------------------------------
def _meta_val(meta, key, dev):
    return meta.get(key)


def _table_to_long(df: pd.DataFrame, meta: dict, src: str, dev: DeviceInfo, notes: list):
    """Turn one table (long, wide or paired) into standard long-format frames."""
    roles = {c: column_role(c) for c in df.columns}
    vcols = [c for c in df.columns if roles[c][0] == "V"]
    icols = [c for c in df.columns if roles[c][0] in ("I", "J")]
    labcols = [c for c in df.columns if roles[c][0] == "label"]
    other = [c for c in df.columns if roles[c][0] == "other"]
    frames = []

    def build(vc, ic, extra_meta, tag):
        f = pd.DataFrame(index=df.index)
        f["V"] = pd.to_numeric(df[vc], errors="coerce") * _unit_factor("V", roles[vc][1])
        r, unit, toks = roles[ic]
        dens = r == "J"
        f["I"] = pd.to_numeric(df[ic], errors="coerce") * _unit_factor("J" if dens else "I", unit)
        m = dict(meta)
        m.update({k: v for k, v in toks.items()})
        m.update(extra_meta)
        f["_density"] = dens
        for role, key in (("T", "T"), ("d", "d"), ("N", "N"), ("area", "area"), ("sweep", "sweep"),
                          ("rate", "rate"), ("H", "H"), ("gate", "gate")):
            cs = [c for c in df.columns if roles[c][0] == role]
            if cs:
                c = cs[0]
                if role in ("sweep",):
                    f[key] = df[c].astype(str)
                else:
                    val = pd.to_numeric(df[c], errors="coerce")
                    u = roles[c][1]
                    if role == "T" and u.lower() in ("c", "°c", "degc"):
                        val = val + 273.15
                    if role == "d":
                        val = val * _unit_factor("d", u)
                    if role == "area":
                        val = val * _unit_factor("area", u)
                    f[key] = val
        f["_meta"] = [m] * len(f)
        f["src"] = src + tag
        return f

    if not vcols:
        # no recognised bias column: first column is V
        vcols = [df.columns[0]]
        roles[vcols[0]] = ("V", roles[vcols[0]][1], roles[vcols[0]][2])
        notes.append(f"{src}: bias column not recognised, using '{df.columns[0]}'.")
        icols = [c for c in icols if c != vcols[0]]
        labcols = [c for c in labcols if c != vcols[0]]
        other = [c for c in other if c != vcols[0]]
    if len(vcols) >= 2:
        # paired format V1 I1 V2 I2 ...
        cols = list(df.columns)
        for k, vc in enumerate(vcols):
            start = cols.index(vc)
            stop = cols.index(vcols[k + 1]) if k + 1 < len(vcols) else len(cols)
            cand = [c for c in cols[start + 1:stop] if roles[c][0] in ("I", "J", "label", "other")]
            if cand:
                ic = cand[0]
                if roles[ic][0] in ("label", "other"):
                    roles[ic] = ("I", roles[ic][1], roles[ic][2])
                frames.append(build(vc, ic, tokens_from_text(vc), f"#{ic}"))
        return frames
    vc = vcols[0]
    if not icols:
        icols = labcols + other
        icols = [c for c in icols if c != vc]
        for c in icols:
            roles[c] = ("I", roles[c][1], roles[c][2])
        if not icols:
            return frames
        if len(icols) > 1:
            plain = [c for c in icols if not roles[c][2] and _num(_split_header(c)[0]) is not None]
            if plain:
                notes.append(f"{src}: numeric column headers {plain[:4]}… interpreted as temperatures in K.")
                for c in plain:
                    roles[c] = ("I", roles[c][1], {"T": _num(_split_header(c)[0])})
    if len(icols) == 1:
        frames.append(build(vc, icols[0], {}, ""))
    else:
        for ic in icols:
            frames.append(build(vc, ic, {}, f"#{ic}"))
        if not any(roles[c][2] for c in icols):
            notes.append(f"{src}: several current columns without T/d labels; treated as separate curves "
                         "(label headers like 'I_300K' or 'I_5nm').")
    return frames


def _apply_meta(f: pd.DataFrame, dev: DeviceInfo, notes: list) -> pd.DataFrame:
    m = f["_meta"].iloc[0] if len(f) else {}
    src = f["src"].iloc[0] if len(f) else ""
    if "T" not in f.columns or f["T"].isna().all():
        if "T" in m:
            f["T"] = m["T"]
        elif dev.temperature_K is not None:
            f["T"] = dev.temperature_K
        else:
            f["T"] = 300.0
            notes.append(f"{src}: no temperature found; assumed 300 K.")
    if "d" not in f.columns or f["d"].isna().all():
        if "N" in f.columns and dev.layer_spacing_nm:
            f["d"] = f["N"] * dev.layer_spacing_nm
        elif "d" in m:
            f["d"] = m["d"]
        elif "N" in m and dev.layer_spacing_nm:
            f["d"] = m["N"] * dev.layer_spacing_nm
        elif dev.thickness_nm is not None:
            f["d"] = dev.thickness_nm
        else:
            f["d"] = 10.0
            notes.append(f"{src}: no thickness found; assumed 10 nm (thickness checks disabled).")
    if "area" not in f.columns or f["area"].isna().all():
        if f["_density"].any():
            f["area"] = 1.0
        elif "area_m2" in m:
            f["area"] = m["area_m2"]
        elif dev.area_m2 is not None:
            f["area"] = dev.area_m2
        else:
            f["area"] = np.nan
    elif f["_density"].any():
        f["area"] = 1.0
    return f.drop(columns=["_meta"])


@dataclass
class Dataset:
    """Long-format I(V,T,d) table.

    Columns: V [V], I [A], T [K], d [nm], area [m^2], curve (int), sweep ('up'/'down'), src.
    If the input was a current density, ``density=True`` and the area column is 1 m^2,
    so that ``I`` is J in A/m^2.
    """
    df: pd.DataFrame
    density: bool = False
    area_known: bool = False
    notes: list = field(default_factory=list)
    sources: list = field(default_factory=list)

    @property
    def temperatures(self):
        return np.sort(self.df["T"].unique())

    @property
    def thicknesses(self):
        return np.sort(self.df["d"].unique())

    def curves(self, positive=True):
        for cid, g in self.df.groupby("curve"):
            if positive:
                g = g[(g.V > 0) & (g.I > 0)]
            g = g.sort_values("V")
            if len(g) >= 3:
                yield cid, g

    def positive(self) -> pd.DataFrame:
        return self.df[(self.df.V > 0) & (self.df.I > 0)].copy()

    def coverage(self) -> dict:
        T = self.temperatures
        d = self.thicknesses
        Vpos = self.positive().V
        return {
            "n_points": int(len(self.df)),
            "n_curves": int(self.df.curve.nunique()),
            "n_T": int(len(T)),
            "T_range": [float(T.min()), float(T.max())],
            "n_d": int(len(d)),
            "d_range_nm": [float(d.min()), float(d.max())],
            "V_range": [float(Vpos.min()) if len(Vpos) else 0.0, float(Vpos.max()) if len(Vpos) else 0.0],
            "V_decades": float(np.log10(Vpos.max() / Vpos.min())) if len(Vpos) > 1 else 0.0,
            "has_T_axis": bool(len(T) >= 3 and T.max() / T.min() >= 1.3),
            "has_d_axis": bool(len(d) >= 2),
            "has_d_axis_3": bool(len(d) >= 3),
            "has_negative_bias": bool((self.df.V < 0).any()),
            "density": self.density,
            "area_known": self.area_known,
        }


def _assign_curves(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_ak"] = df["area"].fillna(-1.0).round(20)
    extra = [c for c in ("H", "gate") if c in df.columns]
    for c in extra:
        df[c] = df[c].fillna(0.0)
    df["curve"] = df.groupby(["src", "T", "d", "_ak"] + extra, sort=False).ngroup()
    if "sweep" not in df.columns or df["sweep"].isna().all():
        df["sweep"] = _sweep_labels(df)
    return df.drop(columns=["_ak"])


def _sweep_labels(df):
    lab = pd.Series("up", index=df.index)
    for _, g in df.groupby("curve", sort=False):
        v = g["V"].values
        dv = np.sign(np.diff(v, prepend=v[0]))
        if len(dv) > 1:
            dv[0] = dv[1]
        lab.loc[g.index] = np.where(dv * np.sign(v) >= 0, "up", "down")
    return lab


def _finalise(frames, dev: DeviceInfo, notes: list, sources) -> Dataset:
    if not frames:
        raise ValueError("No I(V) data found in the input. " + " ".join(notes))
    df = pd.concat(frames, ignore_index=True)
    density = bool(df["_density"].any())
    df = df.drop(columns=["_density"])
    for c in ("V", "I", "T", "d", "area"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["V", "I", "T", "d"])
    area_known = density or bool(df["area"].notna().all())
    if not area_known:
        df["area"] = df["area"].fillna(1e-12)
        notes.append("Junction area unknown: a nominal 1 µm² is used, so absolute-magnitude tests "
                     "(protocol step 2) are relative to that area. Supply the area for a real check.")
    df = df[np.isfinite(df.V) & np.isfinite(df.I) & (df["T"] > 0) & (df.d > 0)]
    df = _assign_curves(df)
    return Dataset(df=df.reset_index(drop=True), density=density, area_known=area_known,
                   notes=notes, sources=sources)


def load(paths, dev: Optional[DeviceInfo] = None) -> Dataset:
    """Load files, folders or archives (any supported format) into a :class:`Dataset`."""
    dev = dev or DeviceInfo()
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    notes, frames, sources, tmpdirs = [], [], [], []
    try:
        for p in paths:
            for tbl, meta, src in read_tables(str(p), tmpdirs):
                if tbl is None:
                    notes.append(f"skipped {meta.get('error')}")
                    continue
                sources.append(src)
                for f in _table_to_long(tbl, meta, src, dev, notes):
                    frames.append(_apply_meta(f, dev, notes))
    finally:
        for t in tmpdirs:
            shutil.rmtree(t, ignore_errors=True)
    return _finalise(frames, dev, notes, sources)


def from_dataframe(df: pd.DataFrame, dev: Optional[DeviceInfo] = None, src: str = "dataframe") -> Dataset:
    dev = dev or DeviceInfo()
    notes = []
    frames = [_apply_meta(f, dev, notes) for f in _table_to_long(df, {}, src, dev, notes)]
    return _finalise(frames, dev, notes, [src])


def from_bytes(name: str, data: bytes, dev: Optional[DeviceInfo] = None) -> Dataset:
    """Load an uploaded file (used by the GUI)."""
    tmp = tempfile.mkdtemp(prefix="barista_up_")
    try:
        path = os.path.join(tmp, os.path.basename(name))
        with open(path, "wb") as fh:
            fh.write(data)
        return load(path, dev)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
