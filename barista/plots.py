"""Publication-ready figures.

Every figure is sized to an exact journal column width, uses embedded TrueType
fonts (pdf.fonttype 42), inward ticks with minor ticks on all four sides, panel
labels (a), (b), ... and no titles, so it can be placed in a manuscript as is.

Colour follows the physical quantity (the review's convention): temperature
series on a single-hue purple ramp, thickness series on an orange ramp, measured
data as open markers and model curves as solid lines.
"""
from __future__ import annotations

import os
from contextlib import contextmanager

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter

from .constants import kB_eV
from .diagnostics import clean_curve
from .models import REGISTRY, SHORT
from .models.base import LN10

JOURNALS = {           # (single column, double column) widths in inches
    "aps": (3.375, 7.0),
    "nature": (3.504, 7.205),
    "acs": (3.25, 7.0),
    "ieee": (3.5, 7.16),
    "elsevier": (3.543, 7.48),
    "rsc": (3.27, 6.69),
}
SANS = ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"]
SERIF = ["Times New Roman", "Times", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"]

INK = "#1a1a1a"
INK2 = "#52514e"
ACCENT = "#2a78d6"     # best model
OTHER = "#b9b8b2"      # other models
C_FIT2 = "#eb6834"
C_FIT3 = "#1baf7a"


def _ramp(name, n, lo=0.38, hi=0.95):
    cmap = matplotlib.colormaps[name]
    if n == 1:
        return [cmap(0.8)]
    return [cmap(x) for x in np.linspace(lo, hi, n)]


class Style:
    def __init__(self, journal="aps", font="sans", fontsize=8, formats=("pdf", "png", "svg")):
        self.journal = journal if journal in JOURNALS else "aps"
        self.single, self.double = JOURNALS[self.journal]
        self.font = font
        self.fs = fontsize
        self.formats = tuple(formats)

    def rc(self):
        fam = "sans-serif" if self.font == "sans" else "serif"
        return {
            "font.family": fam,
            "font.sans-serif": SANS,
            "font.serif": SERIF,
            "mathtext.fontset": "dejavusans" if self.font == "sans" else "dejavuserif",
            "mathtext.default": "it",
            "font.size": self.fs,
            "axes.labelsize": self.fs,
            "axes.titlesize": self.fs,
            "xtick.labelsize": self.fs - 1,
            "ytick.labelsize": self.fs - 1,
            "legend.fontsize": self.fs - 1.5,
            "axes.linewidth": 0.6,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.minor.size": 1.6,
            "ytick.minor.size": 1.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.45,
            "ytick.minor.width": 0.45,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "lines.linewidth": 1.0,
            "lines.markersize": 3.2,
            "legend.frameon": False,
            "legend.handlelength": 1.6,
            "legend.borderaxespad": 0.3,
            "figure.dpi": 150,
            "figure.constrained_layout.use": True,
            "figure.constrained_layout.h_pad": 0.02,
            "figure.constrained_layout.w_pad": 0.02,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": True,
            "axes.formatter.use_mathtext": True,
        }

    @contextmanager
    def ctx(self):
        with matplotlib.rc_context(self.rc()):
            yield


def nice_log(ax):
    """Readable log axes: no overlapping minor labels; plain numbers when the span is short."""
    from matplotlib.ticker import FixedLocator, FuncFormatter
    for which in ("x", "y"):
        axis = ax.xaxis if which == "x" else ax.yaxis
        if axis.get_scale() != "log":
            continue
        lo, hi = (ax.get_xlim() if which == "x" else ax.get_ylim())
        if lo <= 0 or hi <= 0:
            continue
        axis.set_minor_formatter(NullFormatter())
        span = np.log10(hi / lo)
        if span < 1.6:
            cands = []
            for k in range(int(np.floor(np.log10(lo))) - 1, int(np.ceil(np.log10(hi))) + 1):
                for m in ((1, 2, 4) if span > 1.1 else (1, 2, 3, 4, 5, 7) if span > 0.5 else (1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8)):
                    v = m * 10.0**k
                    if lo <= v <= hi:
                        cands.append(v)
            if len(cands) >= 2:
                axis.set_major_locator(FixedLocator(cands))
                axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def finish(fig):
    for ax in fig.axes:
        try:
            nice_log(ax)
        except Exception:
            pass


def panel_label(ax, k, style):
    ax.text(-0.01, 1.015, f"({'abcdefghijklmnop'[k]})", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=style.fs, fontweight="bold", color=INK)


def _save(fig, outdir, name, style):
    finish(fig)
    paths = []
    os.makedirs(outdir, exist_ok=True)
    for fmt in style.formats:
        p = os.path.join(outdir, f"{name}.{fmt}")
        kw = {"dpi": 600} if fmt in ("png", "tif", "tiff") else {}
        if fmt in ("tif", "tiff"):
            kw["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(p, **kw)
        paths.append(p)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# model evaluation helpers
# ---------------------------------------------------------------------------
def model_for(res, key=None):
    """(mechanism, fit) for a single mechanism key, the mixed model or the series model."""
    if key is None:
        key = getattr(res, "assigned_key", "") or res.best.key
    if key == res.best.key:
        return REGISTRY[res.best.key], res.best.fit
    if res.mixed and key == res.mixed["key"]:
        return res.mixed["mech"], res.mixed["fit"]
    if res.series and key == res.series["mech"].key:
        return res.series["mech"], res.series["fit"]
    for a in res.assessments:
        if a.key == key:
            return REGISTRY[key], a.fit
    raise KeyError(key)


def predict_I(mech, fit, V, T, d_nm, area, dev):
    V = np.asarray(V, float)
    T = np.broadcast_to(np.asarray(T, float), V.shape)
    d = np.broadcast_to(np.asarray(d_nm, float) * 1e-9, V.shape)
    pref = 0.0
    for p in mech.params:
        if p.profile:
            pref = fit.params.get(p.name, 0.0)
    return np.exp(mech.lnJ(V, T, d, fit.params, dev) + LN10 * pref) * area


def _ylabel_current(ds):
    if ds.density or ds.area_known:
        return r"$J$ (A cm$^{-2}$)", 1e-4, True
    return r"$I$ (A)", 1.0, False


def _scale(c, ds):
    """Convert measured current to the plotted quantity."""
    _, f, dens = _ylabel_current(ds)
    return c.I.values / c.area.values * f if dens else c.I.values


def _curves(ds):
    out = []
    for cid, g in ds.curves():
        c = clean_curve(g)
        if len(c) >= 3:
            out.append(c)
    return out


def _tcolor(Ts):
    Ts = sorted(set(Ts))
    cols = _ramp("Purples", len(Ts))
    return {T: c for T, c in zip(Ts, cols)}, Ts


def _dcolor(ds_):
    ds_ = sorted(set(ds_))
    cols = _ramp("Oranges", len(ds_), 0.45, 0.95)
    return {d: c for d, c in zip(ds_, cols)}, ds_


def _colorbar(fig, axes, Ts, label, cmap_name="Purples"):
    if len(Ts) < 2:
        return
    cmap = mcolors.LinearSegmentedColormap.from_list("ramp", _ramp(cmap_name, 64))
    norm = mcolors.Normalize(vmin=min(Ts), vmax=max(Ts))
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=axes, fraction=0.04, pad=0.02, aspect=25)
    cb.set_label(label)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(direction="in", width=0.5, length=2)
    cb.ax.minorticks_off()


def _param_box(ax, lines, loc="lower right", style=None):
    txt = "\n".join(lines)
    ha = "right" if "right" in loc else "left"
    va = "bottom" if "lower" in loc else "top"
    x = 0.97 if ha == "right" else 0.04
    y = 0.04 if va == "bottom" else 0.96
    ax.text(x, y, txt, transform=ax.transAxes, ha=ha, va=va, fontsize=(style.fs - 1.5) if style else 6.5,
            color=INK, linespacing=1.25,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d0cfca", lw=0.4, alpha=0.92))


def param_lines(res, max_lines=6, key=None):
    """Short typeset parameter list of the best model, with uncertainties."""
    from .report_text import param_table
    rows = param_table(res, key)
    from .report_text import _fmt
    out = []
    for r in rows:
        if r["role"] in ("fixed", "prefactor"):
            continue
        val = _fmt(r["value"], r["error"]).replace("±", r"$\pm$")
        if "×10^" in val:
            mant, ex = val.split("×10^")
            val = mant + rf"$\times10^{{{ex}}}$"
        unit = r["latex_unit"]
        unit = unit if "$" in unit else unit
        ch = f" ({r['channel']})" if r.get("channel") else ""
        out.append(rf"${r['latex']}${ch} = {val} {unit}".rstrip())
    return out[:max_lines]


# ---------------------------------------------------------------------------
# Figure 1: data + global fit
# ---------------------------------------------------------------------------
def fig_iv_fit(res, style, outdir, key=None, name="fig1_iv_fit"):
    ds, dev = res.dataset, res.device
    mech, fit = model_for(res, key)
    curves = _curves(ds)
    ylab, f, dens = _ylabel_current(ds)
    thick = sorted({float(c["d"].iloc[0]) for c in curves})
    n = len(thick)
    ncol = min(n, 3)
    nrow = int(np.ceil(n / ncol))
    width = style.single if n == 1 else style.double
    h = (2.55 if n == 1 else 2.25 * nrow)
    with style.ctx():
        fig, axes = plt.subplots(nrow, ncol, figsize=(width, h), squeeze=False, sharey=True)
        tcol, Ts = _tcolor([float(c["T"].iloc[0]) for c in curves])
        for k, d in enumerate(thick):
            ax = axes.flat[k]
            for c in curves:
                if float(c["d"].iloc[0]) != d:
                    continue
                T = float(c["T"].iloc[0])
                y = _scale(c, ds)
                ax.plot(c.V, y, "o", mfc="none", mec=tcol[T], mew=0.6, ms=2.8, zorder=2)
                Vd = np.geomspace(c.V.min(), c.V.max(), 200)
                yi = predict_I(mech, fit, Vd, T, d, float(c.area.iloc[0]), dev)
                yi = yi / float(c.area.iloc[0]) * f if dens else yi
                ax.plot(Vd, yi, "-", color=tcol[T], lw=1.0, zorder=3)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel(r"$V$ (V)")
            if k % ncol == 0:
                ax.set_ylabel(ylab)
            if n > 1:
                ax.text(0.97, 0.04, rf"$d$ = {d:g} nm", transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=style.fs - 1, color=INK)
            panel_label(ax, k, style) if n > 1 else None
        for k in range(n, nrow * ncol):
            axes.flat[k].set_visible(False)
        # legend (upper left of first panel) and parameter box (upper left of last panel)
        ax0 = axes.flat[0]
        ax0.plot([], [], "o", mfc="none", mec=INK2, mew=0.6, ms=2.8, label="data")
        ax0.plot([], [], "-", color=INK2, lw=1.0, label="global fit")
        if n == 1:
            ax0.text(0.97, 0.04, rf"$d$ = {thick[0]:g} nm", transform=ax0.transAxes, ha="right", va="bottom",
                     fontsize=style.fs - 1, color=INK)
            ax0.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0))
            _param_box(ax0, param_lines(res), "lower right", style) if False else None
            leg_h = 0.84
            ax0.text(0.04, leg_h, "\n".join(param_lines(res, key=key)), transform=ax0.transAxes, ha="left", va="top",
                     fontsize=style.fs - 1.5, color=INK, linespacing=1.3)
        else:
            ax0.legend(loc="upper left")
            _param_box(axes.flat[n - 1], param_lines(res, key=key), "upper left", style)
        if len(Ts) > 1:
            _colorbar(fig, axes.ravel().tolist(), Ts, r"$T$ (K)")
        return _save(fig, outdir, name, style)


# ---------------------------------------------------------------------------
# Figure 2: the mechanism's own linearising plot
# ---------------------------------------------------------------------------
def _common_V(curves, nmax=4):
    vlo = max(c.V.min() for c in curves); vhi = min(c.V.max() for c in curves)
    if vhi <= vlo:
        return []
    return list(np.geomspace(vlo, vhi, nmax))


def fig_linearised(res, style, outdir, name="fig2_linearised"):
    ds, dev = res.dataset, res.device
    mech, fit = model_for(res)
    key = res.best.key
    curves = _curves(ds)
    ylab, f, dens = _ylabel_current(ds)
    unitJ = "J" if dens else "I"
    with style.ctx():
        fig, axes = plt.subplots(1, 2, figsize=(style.double, 2.5))
        tcol, Ts = _tcolor([float(c["T"].iloc[0]) for c in curves])
        dmin = min(float(c["d"].iloc[0]) for c in curves)
        sel = [c for c in curves if float(c["d"].iloc[0]) == dmin]
        area = float(sel[0].area.iloc[0])
        conv = (lambda I: I / area * f) if dens else (lambda I: I)

        # (a) field / bias linearisation
        ax = axes[0]
        if key in ("poole_frenkel", "gdm"):
            xl, yl = r"$F^{1/2}$ (V$^{1/2}$ cm$^{-1/2}$)", rf"ln(${unitJ}/F$)"
            X = lambda V, d: np.sqrt(V / (d * 1e-7)); Y = lambda I, V, d: np.log(conv(I) / (V / (d * 1e-7)))
        elif key == "schottky":
            xl, yl = r"$V^{1/2}$ (V$^{1/2}$)", rf"ln ${unitJ}$"
            X = lambda V, d: np.sqrt(V); Y = lambda I, V, d: np.log(conv(I))
        elif key in ("coherent", "band_to_band", "tat"):
            xl, yl = r"1/$V$ (V$^{-1}$)", rf"ln(${unitJ}/V^2$)"
            X = lambda V, d: 1 / V; Y = lambda I, V, d: np.log(conv(I) / V**2)
        elif key in ("nn_hopping", "polaron", "vrh"):
            xl, yl = r"$F$ (MV cm$^{-1}$)", rf"ln(${unitJ}/F$)"
            X = lambda V, d: V / (d * 1e-7) / 1e6; Y = lambda I, V, d: np.log(conv(I) / (V / (d * 1e-7)))
        else:
            xl, yl = r"$V$ (V)", ylab
            X = lambda V, d: V; Y = None
        for c in sel:
            T = float(c["T"].iloc[0]); d = float(c["d"].iloc[0])
            Vd = np.geomspace(c.V.min(), c.V.max(), 200)
            Im = predict_I(mech, fit, Vd, T, d, area, dev)
            if Y is None:
                ax.loglog(c.V, conv(c.I.values), "o", mfc="none", mec=tcol[T], mew=0.6, ms=2.8)
                ax.loglog(Vd, conv(Im), "-", color=tcol[T])
            else:
                ax.plot(X(c.V.values, d), Y(c.I.values, c.V.values, d), "o", mfc="none", mec=tcol[T], mew=0.6, ms=2.8)
                ax.plot(X(Vd, d), Y(Im, Vd, d), "-", color=tcol[T])
        if Y is None and key in ("sclc_trap", "sclc_exp"):
            V0 = np.geomspace(min(c.V.min() for c in sel), max(c.V.max() for c in sel), 10)
            yref = conv(sel[-1].I.values).min()
            for s, lab in [(1, "1"), (2, "2")]:
                ax.loglog(V0, yref * (V0 / V0[0]) ** s, ":", color=INK2, lw=0.6)
                ax.text(V0[-1], yref * (V0[-1] / V0[0]) ** s, f" ∝ $V^{lab}$", fontsize=style.fs - 2, color=INK2,
                        va="center")
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.text(0.97, 0.04, rf"$d$ = {dmin:g} nm", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=style.fs - 1, color=INK)
        panel_label(ax, 0, style)

        # (b) temperature / thickness linearisation
        ax = axes[1]
        Vs = _common_V(sel, 4)
        vcols = _ramp("Greens", max(len(Vs), 1), 0.45, 0.95)
        if key in ("sclc_trap", "sclc_exp") and len({float(c['d'].iloc[0]) for c in curves}) >= 2:
            dcol, dl = _dcolor([float(c["d"].iloc[0]) for c in curves])
            Tsel = max(Ts)
            for c in curves:
                if float(c["T"].iloc[0]) != Tsel:
                    continue
                d = float(c["d"].iloc[0]); a = float(c.area.iloc[0])
                J = c.I.values / a * f if dens else c.I.values
                ax.loglog(c.V / d**2, J / d, "o", mfc="none", mec=dcol[d], mew=0.6, ms=2.8, label=f"{d:g} nm")
                Vd = np.geomspace(c.V.min(), c.V.max(), 200)
                Im = predict_I(mech, fit, Vd, Tsel, d, a, dev)
                ax.loglog(Vd / d**2, (Im / a * f if dens else Im) / d, "-", color=dcol[d])
            ax.set_xlabel(r"$V/d^{2}$ (V nm$^{-2}$)"); ax.set_ylabel(rf"${unitJ}/d$ (arb. units)")
            ax.legend(title=rf"$T$ = {Tsel:g} K", loc="upper left", bbox_to_anchor=(0.08, 1))
        elif key == "coherent" and len({float(c['d'].iloc[0]) for c in curves}) >= 2:
            dcurves = [c for c in curves if float(c["T"].iloc[0]) == min(Ts)]
            Vs = _common_V(dcurves, 3)
            for V, col in zip(Vs, _ramp("Greens", len(Vs), 0.5, 0.95)):
                pts = [(float(c["d"].iloc[0]), np.exp(np.interp(np.log(V), np.log(c.V), c.lnI)) / float(c.area.iloc[0]) * f
                        if dens else np.exp(np.interp(np.log(V), np.log(c.V), c.lnI))) for c in dcurves]
                pts.sort()
                dd = np.array([p[0] for p in pts]); yy = np.array([p[1] for p in pts])
                ax.semilogy(dd, yy, "o", mfc="none", mec=col, mew=0.7, ms=3.2, label=f"{V:.2g} V")
                dg = np.linspace(dd.min() * 0.9, dd.max() * 1.1, 60)
                a = float(dcurves[0].area.iloc[0])
                Im = predict_I(mech, fit, np.full_like(dg, V), min(Ts), dg, a, dev)
                ax.semilogy(dg, Im / a * f if dens else Im, "-", color=col)
            ax.set_xlabel(r"$d$ (nm)"); ax.set_ylabel(ylab)
            ax.legend(title=rf"$T$ = {min(Ts):g} K", loc="upper right")
        elif len(Ts) >= 2:
            tvar = {"vrh": "vrh", "gdm": "gdm", "polaron": "polaron"}.get(key, "arr")
            p_vrh = fit.params.get("p", 0.25)
            for V, col in zip(Vs, vcols):
                pts = []
                for c in sel:
                    li = np.interp(np.log(V), np.log(c.V), c.lnI)
                    pts.append((float(c["T"].iloc[0]), np.exp(li)))
                pts.sort()
                T = np.array([p[0] for p in pts]); I = np.array([p[1] for p in pts])
                Tg = np.linspace(T.min() * 0.95, T.max() * 1.05, 80)
                Im = predict_I(mech, fit, np.full_like(Tg, V), Tg, dmin, area, dev)
                if tvar == "vrh":
                    xs, xg = T ** (-p_vrh), Tg ** (-p_vrh); xl2 = rf"$T^{{-{p_vrh:.2f}}}$ (K$^{{-{p_vrh:.2f}}}$)"
                elif tvar == "gdm":
                    xs, xg = 1e5 / T**2, 1e5 / Tg**2; xl2 = r"$10^{5}/T^{2}$ (K$^{-2}$)"
                else:
                    xs, xg = 1000 / T, 1000 / Tg; xl2 = r"1000/$T$ (K$^{-1}$)"
                if key == "schottky":
                    ys, yg = np.log(conv(I) / T**2), np.log(conv(Im) / Tg**2); yl2 = rf"ln(${unitJ}/T^2$)"
                elif key == "polaron":
                    ys, yg = np.log(conv(I) * T**1.5), np.log(conv(Im) * Tg**1.5); yl2 = rf"ln(${unitJ}\,T^{{3/2}}$)"
                else:
                    ys, yg = np.log(conv(I)), np.log(conv(Im)); yl2 = rf"ln ${unitJ}$"
                ax.plot(xs, ys, "o", mfc="none", mec=col, mew=0.7, ms=3.2, label=f"{V:.2g} V")
                ax.plot(xg, yg, "-", color=col)
            ax.set_xlabel(xl2); ax.set_ylabel(yl2)
            ax.legend(loc="best", ncol=1, title=r"$V$")
        else:
            ax.set_visible(False)
        panel_label(ax, 1, style)
        if len(Ts) > 1 and key not in ("sclc_trap", "sclc_exp", "coherent"):
            _colorbar(fig, [axes[0]], Ts, r"$T$ (K)")
        elif len(Ts) > 1 and key in ("sclc_trap", "sclc_exp", "coherent"):
            _colorbar(fig, [axes[0]], Ts, r"$T$ (K)")
        return _save(fig, outdir, name, style)


# ---------------------------------------------------------------------------
# Figure 3: model-free fingerprints
# ---------------------------------------------------------------------------
def fig_fingerprints(res, style, outdir, name="fig3_fingerprints"):
    ds, dg = res.dataset, res.diag
    mech, fit = model_for(res)
    with style.ctx():
        fig, axes = plt.subplots(2, 2, figsize=(style.double, 4.6))
        ax = axes[0, 0]
        tcol, Ts = _tcolor([c["T"] for c in dg["gamma"]] or [300])
        dmin = min(ds.thicknesses)
        for c in dg["gamma"]:
            if c["d"] != dmin:
                continue
            ax.plot(c["V"], c["gamma"], "o", mfc="none", mec=tcol[c["T"]], mew=0.6, ms=2.6)
            Vd = np.geomspace(min(c["V"]), max(c["V"]), 120)
            Im = predict_I(mech, fit, Vd, c["T"], c["d"], 1.0, res.device)
            gm = np.gradient(np.log(Im), np.log(Vd))
            ax.plot(Vd, gm, "-", color=tcol[c["T"]])
        ax.axhline(1, color=INK2, ls=":", lw=0.6); ax.axhline(2, color=INK2, ls=":", lw=0.6)
        ax.set_xscale("log"); ax.set_xlabel(r"$V$ (V)"); ax.set_ylabel(r"$\gamma = d\,\ln I/d\,\ln V$")
        ax.text(0.97, 0.96, rf"$d$ = {dmin:g} nm", transform=ax.transAxes, ha="right", va="top", fontsize=style.fs - 1)
        panel_label(ax, 0, style)

        # (b) Ea(V) vs sqrt(V)
        ax = axes[0, 1]
        if dg["arrhenius"]:
            dcol, dl = _dcolor([a["d"] for a in dg["arrhenius"]])
            for a in dg["arrhenius"]:
                t = a["table"]
                ax.errorbar(np.sqrt(t["V"]), t["Ea"], yerr=t["Ea_err"], fmt="o", mfc="none", mec=dcol[a["d"]],
                            ecolor=dcol[a["d"]], mew=0.7, ms=3.0, elinewidth=0.5, capsize=0,
                            label=f"{a['d']:g} nm")
                if "sqrtV_slope" in a:
                    xx = np.linspace(0, np.sqrt(max(t["V"])) * 1.05, 20)
                    ax.plot(xx, a["Ea0"] + a["sqrtV_slope"] * xx, "-", color=dcol[a["d"]], lw=0.8)
            ax.set_xlabel(r"$V^{1/2}$ (V$^{1/2}$)"); ax.set_ylabel(r"$E_a$ (eV)")
            ax.legend(loc="best", title=r"$d$")
        else:
            ax.text(0.5, 0.5, "Arrhenius analysis needs ≥ 3 temperatures", transform=ax.transAxes,
                    ha="center", va="center", color=INK2, fontsize=style.fs - 1)
            ax.set_xticks([]); ax.set_yticks([]); ax.minorticks_off()
        panel_label(ax, 1, style)

        # (c) Zabrodskii
        ax = axes[1, 0]
        zb = [z for z in dg["zabrodskii"] if np.isfinite(np.array(z["w"], float)).sum() >= 2]
        if zb:
            dcol, _ = _dcolor([z["d"] for z in zb])
            for z in zb:
                T = np.array(z["T"]); w = np.array(z["w"], float)
                m = np.isfinite(w) & (w > 0)
                ax.loglog(T[m], w[m], "o-", mfc="none", mec=dcol[z["d"]], color=dcol[z["d"]], lw=0.6, mew=0.7,
                          ms=3.2, label=f"{z['d']:g} nm")
            T0 = np.array([min(min(z["T"]) for z in zb), max(max(z["T"]) for z in zb)])
            wref = np.nanmax([np.nanmax(np.array(z["w"], float)) for z in zb])
            for p, ls in [(1, "--"), (0.5, "-."), (0.25, ":")]:
                ax.loglog(T0, wref * (T0 / T0[0]) ** (-p), ls, color=INK2, lw=0.6)
                ax.text(T0[1], wref * (T0[1] / T0[0]) ** (-p), f"$p$ = {p:g} ", fontsize=style.fs - 2.5,
                        color=INK2, va="bottom", ha="right")
            ax.set_xlabel(r"$T$ (K)"); ax.set_ylabel(r"$w = d\,\ln G/d\,\ln T$")
            ax.legend(loc="lower left", title=r"$d$")
        else:
            ax.text(0.5, 0.5, "w(T) needs ≥ 4 temperatures", transform=ax.transAxes, ha="center", va="center",
                    color=INK2, fontsize=style.fs - 1)
            ax.set_xticks([]); ax.set_yticks([]); ax.minorticks_off()
        panel_label(ax, 2, style)

        # (d) thickness scaling or Rowell plot
        ax = axes[1, 1]
        th = [t for t in dg["thickness"] if len(t["d"]) >= 2]
        if th:
            t = th[0]
            scores = {k: v for k, v in t["collapse"].items() if v is not None}
            names = list(scores)
            vals = [scores[k] for k in names]
            best = int(np.argmin(vals))
            colors = [ACCENT if i == best else OTHER for i in range(len(vals))]
            y = np.arange(len(names))
            ax.barh(y, vals, color=colors, height=0.6)
            ax.set_yticks(y)
            ax.set_yticklabels([n.split("  ")[0] + "\n" + n.split("  ")[-1] for n in names], fontsize=style.fs - 2)
            ax.set_xscale("log")
            ax.set_xlabel(r"collapse spread  $\langle\sigma_{\ln J}\rangle$")
            ax.tick_params(axis="y", which="minor", left=False, right=False)
            ax.invert_yaxis()
            ax.text(0.97, 0.97, f"{len(t['d'])} thicknesses, {t['T']:g} K", transform=ax.transAxes, ha="right",
                    va="top", fontsize=style.fs - 2, color=INK2)
        elif dg["rowell"]:
            Gd = dg["lowbias_G"]
            dcol, _ = _dcolor([float(k) for k in Gd])
            for k, r in Gd.items():
                T = np.array(r["T"]); G = np.array(r["G"])
                ax.plot(T, G / G[np.argmax(T)], "o-", color=dcol[float(k)], mfc="none", mew=0.7, ms=3.2, lw=0.6,
                        label=f"{float(k):g} nm")
            ax.set_yscale("log"); ax.set_xlabel(r"$T$ (K)"); ax.set_ylabel(r"$G(T)/G(T_{\max})$")
            ax.legend(loc="lower right")
        else:
            ax.set_visible(False)
        panel_label(ax, 3, style)
        return _save(fig, outdir, name, style)


# ---------------------------------------------------------------------------
# Figure 4: model comparison and residuals
# ---------------------------------------------------------------------------
def fig_models(res, style, outdir, name="fig4_model_comparison"):
    ranked = res.ranked()
    with style.ctx():
        fig, axes = plt.subplots(1, 2, figsize=(style.double, 2.9), gridspec_kw={"width_ratios": [1.15, 1]})
        ax = axes[0]
        labels = [SHORT.get(a.key, a.key) for a in ranked]
        vals = [max(a.d_aic, 0.0) + 1 for a in ranked]
        cols = [ACCENT if a.key == res.best.key else ("#e34948" if a.verdict == "EXCLUDED" else OTHER) for a in ranked]
        y = np.arange(len(ranked))
        ax.barh(y, vals, color=cols, height=0.62)
        ax.set_xscale("log")
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=style.fs - 2)
        ax.tick_params(axis="y", which="both", left=False, right=False)
        ax.invert_yaxis()
        ax.axvline(11, color=INK2, ls=":", lw=0.6)
        ax.set_xlabel(r"$\Delta$AIC + 1")
        for yi, a in zip(y, ranked):
            ax.text(vals[yi] * 1.15, yi, f"{100 * a.prob:.0f} %" if a.prob >= 0.005 else "", va="center",
                    fontsize=style.fs - 2.5, color=INK2)
        panel_label(ax, 0, style)
        # residuals of best model
        ax = axes[1]
        data = res.data
        r = data.lnI - res.best.fit.lnI_fit
        tcol, Ts = _tcolor(list(np.unique(data.T)))
        for T in Ts:
            m = data.T == T
            ax.plot(data.V[m], 100 * (np.exp(r[m]) - 1), "o", mfc="none", mec=tcol[T], mew=0.5, ms=2.4)
        ax.axhline(0, color=INK, lw=0.6)
        ax.set_xscale("log")
        lim = np.nanpercentile(np.abs(100 * (np.exp(r) - 1)), 99) * 1.3 + 1
        ax.set_ylim(-lim, lim)
        ax.set_xlabel(r"$V$ (V)"); ax.set_ylabel(r"$(I_{\rm exp} - I_{\rm fit})/I_{\rm fit}$ (%)")
        panel_label(ax, 1, style)
        if len(Ts) > 1:
            _colorbar(fig, [ax], Ts, r"$T$ (K)")
        return _save(fig, outdir, name, style)


# ---------------------------------------------------------------------------
# Figure 5: regime / channel map and extrapolation
# ---------------------------------------------------------------------------
def fig_regimes(res, style, outdir, name="fig5_regime_map"):
    """Channel fraction (two-channel model) or sub-regime of the assigned model over (V, T), per thickness."""
    from matplotlib.patches import Patch
    ds, dev = res.dataset, res.device
    curves = _curves(ds)
    Ts = sorted({float(c["T"].iloc[0]) for c in curves})
    thick = list(ds.thicknesses)
    if len(thick) > 3:
        thick = [thick[0], thick[len(thick) // 2], thick[-1]]
    Vlo = min(c.V.min() for c in curves); Vhi = max(c.V.max() for c in curves)
    Vg = np.geomspace(Vlo, Vhi, 160)
    Tg = np.linspace(min(Ts), max(Ts), 120) if len(Ts) > 1 else np.array([Ts[0]])
    VV, TT = np.meshgrid(Vg, Tg)
    use_mixed = bool(res.mixed) and getattr(res, "assigned_key", "") == res.mixed["key"]
    n = len(thick)
    width = style.single if n == 1 else style.double
    with style.ctx():
        fig, axes = plt.subplots(1, n, figsize=(width, 2.6 if n == 1 else 2.4), squeeze=False, sharey=True)
        axes = axes[0]
        handles = None
        pc = None
        for k, d in enumerate(thick):
            ax = axes[k]
            dd = np.full(VV.size, d * 1e-9)
            if use_mixed:
                pm, fit = res.mixed["mech"], res.mixed["fit"]
                la, lb = pm.channels(VV.ravel(), TT.ravel(), dd, fit.params, dev)
                frac = (1 / (1 + np.exp(np.clip(lb - la, -700, 700)))).reshape(VV.shape)
                cmap = mcolors.LinearSegmentedColormap.from_list("div", ["#eb6834", "#f2f1ee", "#2a78d6"])
                if len(Ts) > 1:
                    pc = ax.pcolormesh(VV, TT, frac, cmap=cmap, vmin=0, vmax=1, shading="auto", rasterized=True)
                    if 0 < frac.min() < 0.5 < frac.max():
                        ax.contour(VV, TT, frac, levels=[0.5], colors=INK, linewidths=0.7)
                else:
                    ax.plot(Vg, frac[0], color=ACCENT)
            else:
                mech, fit = model_for(res)
                labels = mech.regime(VV.ravel(), TT.ravel(), dd, fit.params, dev)
                if labels is None:
                    plt.close(fig)
                    return []
                allv = list(dict.fromkeys(labels))
                if handles is None:
                    uniq = allv
                else:
                    uniq = uniq + [u for u in allv if u not in uniq]
                pal = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]
                idx = np.array([uniq.index(l) for l in labels]).reshape(VV.shape)
                cmap = mcolors.ListedColormap(pal[: len(uniq)])
                if len(Ts) > 1:
                    ax.pcolormesh(VV, TT, idx, cmap=cmap, vmin=-0.5, vmax=len(uniq) - 0.5, shading="auto",
                                  rasterized=True, alpha=0.85)
                else:
                    ax.scatter(Vg, np.zeros_like(Vg), c=idx[0], cmap=cmap, s=6)
                    ax.set_yticks([])
                handles = [Patch(color=pal[i], label=u) for i, u in enumerate(uniq)]
            for c in curves:
                if float(c["d"].iloc[0]) == d and len(Ts) > 1:
                    ax.plot(c.V, np.full(len(c), float(c["T"].iloc[0])), "|", color=INK, ms=2, mew=0.4)
            ax.set_xscale("log"); ax.set_xlabel(r"$V$ (V)")
            if k == 0:
                ax.set_ylabel(r"$T$ (K)" if len(Ts) > 1 else "fraction, channel A")
            ax.text(0.97, 0.97, rf"$d$ = {d:g} nm", transform=ax.transAxes, ha="right", va="top",
                    fontsize=style.fs - 1, color=INK,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
            if n > 1:
                panel_label(ax, k, style)
        if use_mixed:
            if pc is not None:
                cb = fig.colorbar(pc, ax=list(axes), fraction=0.05 if n == 1 else 0.025, pad=0.02)
                cb.set_label("current fraction, channel A")
                cb.ax.tick_params(direction="in", width=0.5, length=2); cb.outline.set_linewidth(0.5)
                cb.ax.minorticks_off()
            pm = res.mixed["mech"]
            fig.legend(handles=[Patch(color="#2a78d6", label=f"A: {SHORT.get(pm.a.key, pm.a.key)}"),
                                Patch(color="#eb6834", label=f"B: {SHORT.get(pm.b.key, pm.b.key)}")],
                       loc="outside upper center", ncol=2, fontsize=style.fs - 1.5)
        elif handles:
            fig.legend(handles=handles, loc="outside upper center", ncol=min(len(handles), 3),
                       fontsize=style.fs - 1.5)
        return _save(fig, outdir, name, style)


def fig_extrapolation(res, style, outdir, name="fig6_prediction"):
    """Model prediction outside the measured window (dashed) — a falsifiable forecast."""
    ds, dev = res.dataset, res.device
    mech, fit = model_for(res)
    curves = _curves(ds)
    ylab, f, dens = _ylabel_current(ds)
    dmin = min(ds.thicknesses)
    sel = [c for c in curves if float(c["d"].iloc[0]) == dmin]
    Ts = sorted({float(c["T"].iloc[0]) for c in sel})
    Tex = sorted(set(Ts + [round(min(Ts) * 0.8, 0), round(max(Ts) * 1.15, 0)]))
    tcol, _ = _tcolor(Tex)
    Vlo = min(c.V.min() for c in sel); Vhi = max(c.V.max() for c in sel)
    Vg = np.geomspace(Vlo / 5, Vhi * 2, 240)
    area = float(sel[0].area.iloc[0])
    with style.ctx():
        fig, ax = plt.subplots(figsize=(style.single, 2.6))
        for T in Tex:
            Im = predict_I(mech, fit, Vg, T, dmin, area, dev)
            y = Im / area * f if dens else Im
            measured = T in Ts
            inside = (Vg >= Vlo) & (Vg <= Vhi)
            ax.loglog(Vg[inside], y[inside], "-" if measured else "--", color=tcol[T], lw=1.0 if measured else 0.8)
            ax.loglog(Vg[~inside & (Vg < Vlo)], y[~inside & (Vg < Vlo)], "--", color=tcol[T], lw=0.8)
            ax.loglog(Vg[~inside & (Vg > Vhi)], y[~inside & (Vg > Vhi)], "--", color=tcol[T], lw=0.8)
        for c in sel:
            T = float(c["T"].iloc[0])
            ax.loglog(c.V, _scale(c, ds), "o", mfc="none", mec=tcol[T], mew=0.5, ms=2.4)
        ax.axvspan(Vlo, Vhi, color="#f2f1ee", zorder=0, lw=0)
        ymin_data = min(_scale(c, ds).min() for c in sel)
        ymax_data = max(_scale(c, ds).max() for c in sel)
        lo_, hi_ = ax.get_ylim()
        ax.set_ylim(max(lo_, ymin_data / 1e3), min(hi_, ymax_data * 1e3))
        ax.set_xlabel(r"$V$ (V)"); ax.set_ylabel(ylab)
        ax.plot([], [], "-", color=INK2, label="fit (measured window)")
        ax.plot([], [], "--", color=INK2, label="prediction")
        ax.legend(loc="upper left")
        _colorbar(fig, [ax], Tex, r"$T$ (K)")
        return _save(fig, outdir, name, style)


def make_all(res, outdir, style: Style = None):
    style = style or Style()
    out = {}
    out["fig1_iv_fit"] = fig_iv_fit(res, style, outdir)
    out["fig2_linearised"] = fig_linearised(res, style, outdir)
    out["fig3_fingerprints"] = fig_fingerprints(res, style, outdir)
    out["fig4_model_comparison"] = fig_models(res, style, outdir)
    try:
        r = fig_regimes(res, style, outdir)
        if r:
            out["fig5_regime_map"] = r
    except Exception as exc:     # regime map is optional
        res.warnings.append(f"regime map skipped: {exc}")
    out["fig6_prediction"] = fig_extrapolation(res, style, outdir)
    if res.mixed and res.assigned_key != res.mixed["key"]:
        out["fig7_two_channel_fit"] = fig_iv_fit(res, style, outdir, key=res.mixed["key"], name="fig7_two_channel_fit")
    elif res.mixed:
        out["fig7_best_single_fit"] = fig_iv_fit(res, style, outdir, key=res.best.key, name="fig7_best_single_fit")
    try:
        r = fig_traps(res, style, outdir)
        if r:
            out["fig15_trap_density"] = r
    except Exception as exc:
        res.warnings.append(f"trap-density figure skipped: {exc}")
    return out


def fig_traps(res, style, outdir, name="fig15_trap_density"):
    """Forest plot of every trap-density route: point estimates with 95 % intervals, bounds as arrows."""
    tr = res.traps
    top = {a.key for a in res.ranked()[:3]}
    rows = [t for t in tr.estimates if (t.weight >= 0.01 or t.mechanism in top) and np.isfinite(t.Nt_cm3)
            and 1e8 < t.Nt_cm3 < 1e25]
    if not rows:
        return None
    assigned = set((res.assigned_key or res.best.key).split("+"))
    rows.sort(key=lambda t: (t.mechanism not in assigned, -t.weight))
    with style.ctx():
        fig, ax = plt.subplots(figsize=(style.double, 1.0 + 0.26 * len(rows)))
        y = np.arange(len(rows))[::-1]
        c = tr.consensus
        if np.isfinite(c.get("Nt_cm3", np.nan)):
            lo, hi = c.get("lo_cm3", np.nan), c.get("hi_cm3", np.nan)
            if np.isfinite(lo) and np.isfinite(hi):
                ax.axvspan(lo, hi, color=ACCENT, alpha=0.10, lw=0)
            ax.axvline(c["Nt_cm3"], color=ACCENT, lw=1.0, ls="-", alpha=0.7)
        if res.device.trap_density_cm3:
            ax.axvline(res.device.trap_density_cm3, color=INK, lw=0.9, ls="--")
            ax.text(res.device.trap_density_cm3, y.max() + 0.6, " supplied", fontsize=style.fs - 1, color=INK2,
                    va="bottom")
        xs = []
        for yi, t in zip(y, rows):
            col = ACCENT if t.mechanism in assigned else OTHER
            if t.kind == "upper bound":
                ax.annotate("", xy=(t.Nt_cm3 / 30, yi), xytext=(t.Nt_cm3, yi),
                            arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2))
                ax.plot([t.Nt_cm3], [yi], "|", ms=8, mew=1.5, color=col)
                xs += [t.Nt_cm3, t.Nt_cm3 / 30]
            elif t.kind == "lower bound":
                ax.annotate("", xy=(t.Nt_cm3 * 30, yi), xytext=(t.Nt_cm3, yi),
                            arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2))
                xs += [t.Nt_cm3, t.Nt_cm3 * 30]
            else:
                lo = t.lo_cm3 if np.isfinite(t.lo_cm3) and t.lo_cm3 > 0 else t.Nt_cm3
                hi = t.hi_cm3 if np.isfinite(t.hi_cm3) else t.Nt_cm3
                lo, hi = max(lo, t.Nt_cm3 / 1e4), min(hi, t.Nt_cm3 * 1e4)
                ax.plot([lo, hi], [yi, yi], "-", color=col, lw=1.6, solid_capstyle="round")
                ax.plot([t.Nt_cm3], [yi], "o", ms=5, color=col, mec="white", mew=0.8, zorder=3)
                xs += [lo, hi]
        ax.set_xscale("log")
        xs = np.array(xs)
        ax.set_xlim(10 ** np.floor(np.log10(xs.min()) - 0.3), 10 ** np.ceil(np.log10(xs.max()) + 0.3))
        labels = []
        for t in rows:
            en = f"; {t.energy_tex} = {t.energy_eV:.2f} eV" if np.isfinite(t.energy_eV) and t.energy_tex else ""
            labels.append(f"{SHORT.get(t.mechanism, t.mechanism)}: {t.short}{en}")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=style.fs - 1)
        ax.set_ylim(-0.7, len(rows) - 0.3)
        ax.set_xlabel(r"trap / localised-state density $N_t$ (cm$^{-3}$)")
        ax.tick_params(axis="y", which="both", length=0)
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.grid(axis="x", which="major", color="#e4e3de", lw=0.5)
        ax.set_axisbelow(True)
        return _save(fig, outdir, name, style)


# ---------------------------------------------------------------------------
# Certainty figures (uncertainty analytics)
# ---------------------------------------------------------------------------
def _pretty(pname):
    from .report_text import SYMBOLS
    base = pname.split(".")[-1]
    tex = SYMBOLS.get(base, (base, base.replace("_", r"\_"), "", "", None))[1]
    unit = SYMBOLS.get(base, (None, None, "", "", None))[3]
    ch = f" ({pname[0].upper()})" if pname.startswith(("a.", "b.")) else ""
    return rf"${tex}${ch}" + (f" ({unit})" if unit else "")


def fig_certainty(res, cert, style, outdir, name="fig8_model_certainty"):
    keys = sorted(cert.candidates, key=lambda k: -cert.selection[k]["freq"])
    lab = [SHORT.get(k, " ∥ ".join(SHORT.get(x, x) for x in k.split("+"))) for k in keys]
    with style.ctx():
        fig, axes = plt.subplots(1, 2, figsize=(style.double, 2.6))
        ax = axes[0]
        f = np.array([cert.selection[k]["freq"] for k in keys])
        lo = np.array([cert.selection[k]["ci_lo"] for k in keys]); hi = np.array([cert.selection[k]["ci_hi"] for k in keys])
        y = np.arange(len(keys))
        cols = [ACCENT if k == cert.assigned_key else OTHER for k in keys]
        ax.barh(y, 100 * f, color=cols, height=0.6)
        ax.errorbar(100 * f, y, xerr=[np.clip(100 * (f - lo), 0, None), np.clip(100 * (hi - f), 0, None)], fmt="none", ecolor=INK, elinewidth=0.6,
                    capsize=1.5)
        ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=style.fs - 2)
        ax.tick_params(axis="y", which="both", left=False, right=False)
        ax.invert_yaxis(); ax.set_xlim(0, 105)
        ax.set_xlabel(f"selected in bootstrap (%)   [B = {cert.B}]")
        panel_label(ax, 0, style)
        ax = axes[1]
        data = []
        for k in keys:
            ak = cert.akaike[k]
            data.append((ak["boot_lo"], ak["boot_mean"], ak["boot_hi"], ak["weight"]))
        for i, (l, m, h, w) in enumerate(data):
            ax.plot([l, h], [i, i], "-", color=INK2, lw=0.8)
            ax.plot(m, i, "o", color=cols[i], ms=4)
            ax.plot(w, i, "x", color=INK, ms=4, mew=0.8)
        ax.set_yticks(y); ax.set_yticklabels([])
        ax.tick_params(axis="y", which="both", left=False, right=False)
        ax.invert_yaxis(); ax.set_xlim(-0.03, 1.03)
        ax.set_xlabel("Akaike weight")
        ax.plot([], [], "o", color=INK2, label="bootstrap mean (95 % range)")
        ax.plot([], [], "x", color=INK, label="full data")
        ax.legend(loc="lower right")
        panel_label(ax, 1, style)
        return _save(fig, outdir, name, style)


def fig_param_corner(res, cert, style, outdir, name="fig9_parameter_uncertainty"):
    smp = cert.boot_samples.get(cert.assigned_key, {})
    pn = [p for p in cert.params if p in smp and len(smp[p]) >= 10][:4]
    if not pn:
        return []
    k = len(pn)
    w = style.single if k <= 2 else style.double * 0.8
    with style.ctx():
        fig, axes = plt.subplots(k, k, figsize=(w, w), squeeze=False)
        for i in range(k):
            for j in range(k):
                ax = axes[i, j]
                if j > i:
                    ax.set_visible(False); continue
                xi = smp[pn[j]]
                if i == j:
                    ax.hist(xi, bins=24, color="#c9dcf3", edgecolor=ACCENT, linewidth=0.5)
                    row = cert.params[pn[i]]
                    ax.axvline(row["estimate"], color=INK, lw=0.8)
                    if row.get("boot95"):
                        for v in row["boot95"]:
                            ax.axvline(v, color=ACCENT, ls="--", lw=0.7)
                    if row.get("profile95"):
                        for v in row["profile95"][:2]:
                            ax.axvline(v, color=C_FIT2, ls=":", lw=0.8)
                    ax.set_yticks([])
                    ax.tick_params(axis="y", which="both", left=False, right=False)
                else:
                    yi = smp[pn[i]]
                    ax.plot(xi, yi, ".", color=ACCENT, ms=1.5, alpha=0.6)
                if i == k - 1:
                    ax.set_xlabel(_pretty(pn[j]))
                else:
                    ax.set_xticklabels([])
                if j == 0 and i > 0:
                    ax.set_ylabel(_pretty(pn[i]))
                elif i != j:
                    ax.set_yticklabels([])
                ax.tick_params(labelsize=style.fs - 2.5)
                ax.xaxis.get_offset_text().set_fontsize(style.fs - 3)
        fig.legend(handles=[plt.Line2D([], [], color=INK, lw=0.8, label="estimate"),
                            plt.Line2D([], [], color=ACCENT, ls="--", lw=0.7, label="bootstrap 95 %"),
                            plt.Line2D([], [], color=C_FIT2, ls=":", lw=0.8, label="profile 95 %")],
                   loc="outside upper center", ncol=3, fontsize=style.fs - 2)
        return _save(fig, outdir, name, style)


def fig_profiles(res, cert, style, outdir, name="fig10_profile_likelihood"):
    pn = list(cert.profile)[:4]
    if not pn:
        return []
    n = len(pn)
    with style.ctx():
        fig, axes = plt.subplots(1, n, figsize=(style.single if n == 1 else style.double, 2.2), squeeze=False)
        for i, p in enumerate(pn):
            ax = axes[0, i]
            pr = cert.profile[p]
            g = np.array(pr["grid"]); dc = np.array(pr["dchi2"])
            ax.plot(g, dc, "o-", color=ACCENT, ms=2.5, lw=0.8)
            ax.axhline(1.0, color=INK2, ls=":", lw=0.6); ax.axhline(3.84, color=INK2, ls="--", lw=0.6)
            ax.text(g[-1], 1.0, " 68 %", va="bottom", ha="right", fontsize=style.fs - 2.5, color=INK2)
            ax.text(g[-1], 3.84, " 95 %", va="bottom", ha="right", fontsize=style.fs - 2.5, color=INK2)
            ax.set_ylim(0, max(6, min(np.nanmax(dc) * 1.05, 30)))
            ax.set_xlabel(_pretty(p))
            if i == 0:
                ax.set_ylabel(r"$\Delta\chi^2$")
            ax.xaxis.get_offset_text().set_fontsize(style.fs - 3)
            panel_label(ax, i, style)
        return _save(fig, outdir, name, style)


def fig_stability(res, cert, style, outdir, name="fig11_stability_residuals"):
    pn = list(cert.params)[:4]
    with style.ctx():
        fig, axes = plt.subplots(1, 2, figsize=(style.double, 2.7), gridspec_kw={"width_ratios": [1.4, 1]})
        ax = axes[0]
        W = cert.windows
        y = np.arange(len(W))
        pal = [ACCENT, C_FIT2, C_FIT3, "#6250d6"]
        for j, p in enumerate(pn):
            v0 = cert.params[p]["estimate"]; se0 = cert.params[p]["se_asymptotic"]
            z, ze = [], []
            for w in W:
                e = w.get(p + "_err", np.nan)
                tot = np.sqrt((se0 if np.isfinite(se0) else 0) ** 2 + (e if np.isfinite(e) else 0) ** 2)
                z.append((w.get(p, np.nan) - v0) / tot if tot > 0 else np.nan)
            ax.plot(z, y + (j - (len(pn) - 1) / 2) * 0.15, "o", color=pal[j % 4], ms=3.2, mfc="none", mew=0.8,
                    label=_pretty(p))
        ax.axvspan(-1, 1, color="#eef3fb", zorder=0); ax.axvline(0, color=INK, lw=0.6)
        for xv in (-3, 3):
            ax.axvline(xv, color=INK2, ls=":", lw=0.6)
        ax.set_yticks(y); ax.set_yticklabels([w["subset"] for w in W], fontsize=style.fs - 2.5)
        ax.tick_params(axis="y", which="both", left=False, right=False)
        ax.invert_yaxis()
        ax.set_xlabel(r"shift from full-data estimate ($\sigma$)")
        ax.legend(loc="lower right", fontsize=style.fs - 2.5)
        panel_label(ax, 0, style)
        ax = axes[1]
        r = np.array(cert.residuals["residuals"])
        r = r[np.isfinite(r)]
        osm, osr = stats_probplot(r)
        ax.plot(osm, osr, ".", color=ACCENT, ms=2)
        lim = [min(osm.min(), osr.min()), max(osm.max(), osr.max())]
        ax.plot(lim, lim, "-", color=INK, lw=0.6)
        ax.set_xlabel("normal quantile"); ax.set_ylabel(r"standardised residual of ln$I$")
        ax.text(0.04, 0.96, f"Shapiro–Wilk p = {cert.residuals['shapiro_p']:.2g}\n"
                            f"Durbin–Watson = {cert.residuals['durbin_watson_median']:.2f}",
                transform=ax.transAxes, va="top", fontsize=style.fs - 2.5, color=INK)
        panel_label(ax, 1, style)
        return _save(fig, outdir, name, style)


def stats_probplot(r):
    from scipy import stats as _st
    z = (r - np.mean(r)) / max(np.std(r), 1e-300)
    (osm, osr), _ = _st.probplot(z, dist="norm")
    return osm, osr


def make_certainty_figs(res, cert, outdir, style):
    out = {}
    out["fig8_model_certainty"] = fig_certainty(res, cert, style, outdir)
    p = fig_param_corner(res, cert, style, outdir)
    if p:
        out["fig9_parameter_uncertainty"] = p
    p = fig_profiles(res, cert, style, outdir)
    if p:
        out["fig10_profile_likelihood"] = p
    out["fig11_stability_residuals"] = fig_stability(res, cert, style, outdir)
    return out
