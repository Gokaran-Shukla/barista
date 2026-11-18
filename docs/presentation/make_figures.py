"""Regenerate every figure used in the BARISTA presentation.

All figures are produced by BARISTA itself, typeset in Inter (the slide typeface) at slide scale.
Run from the repository root:   python docs/presentation/make_figures.py
"""
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import matplotlib.font_manager as fm                                       # noqa: E402
from fonts_ttf import FAMILY, build as build_fonts                         # noqa: E402

for _f in build_fonts(HERE / "fonts"):                                     # Inter as TrueType for clean embedding
    fm.fontManager.addfont(str(_f))

from barista import io, engine, synth, spin, uncertainty                    # noqa: E402
from barista.device import DeviceInfo                                      # noqa: E402
from barista.plots import Style, make_all, make_certainty_figs, fig_param_corner   # noqa: E402
from barista.spin_report import spin_figures                                # noqa: E402

FIGS = HERE / "figs"
TMP = HERE / "_build_figs"
STYLE = Style(journal="aps", font=FAMILY, fontsize=10, formats=("pdf",))
WORKERS = os.cpu_count()


def collect(src: Path, prefix: str):
    for f in src.glob("*.pdf"):
        shutil.copy(f, FIGS / f"{prefix}_{f.name}")


def mixed_channels():
    dev = DeviceInfo(area_um2=100)
    ds = io.load(ROOT / "examples/mixed_channels/AlOx_junctions.xlsx", dev)
    res = engine.analyze(ds, dev, workers=WORKERS)
    out = TMP / "mixed"
    make_all(res, out, STYLE)
    collect(out, "mixed")
    print("mixed:", res.headline)


def sclc_with_certainty():
    pr = synth.PRESETS["sclc_trap"]
    dev = DeviceInfo.from_dict({**pr["dev"], "eps_r": 5.0})
    df = synth.generate("sclc_trap", **{k: v for k, v in pr.items() if k in ("params", "V", "T", "d", "dev")})
    res = engine.analyze(io.from_dataframe(df, dev), dev, workers=WORKERS)
    res.certainty = uncertainty.assess(res, B=60, workers=WORKERS)
    out = TMP / "sclc"
    make_all(res, out, STYLE)
    make_certainty_figs(res, res.certainty, out, STYLE)
    compact = Style(journal="aps", font=FAMILY, fontsize=10, formats=("pdf",))
    compact.double = 4.6                       # corner plot sized for a slide, so its labels stay legible
    fig_param_corner(res, res.certainty, compact, out)
    collect(out, "sclc")
    print("sclc:", res.headline, "|", res.traps.statement)


def spin_crsb():
    df, truth = spin.synth_proximity(N_list=(4, 5, 6, 8), T_list=(10, 100, 200, 300))
    tab = spin.mr_table(df, layer_spacing_nm=0.2735)
    sr = spin.analyse_spin(tab, B=100)
    out = TMP / "spin"
    spin_figures(sr, out, STYLE)
    collect(out, "spin")
    print("spin:", sr.statement[:160])


if __name__ == "__main__":
    FIGS.mkdir(exist_ok=True)
    for job in (spin_crsb, mixed_channels, sclc_with_certainty):
        job()
    shutil.rmtree(TMP, ignore_errors=True)
