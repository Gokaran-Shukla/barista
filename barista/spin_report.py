"""Figures and tables for the spin-dependent analysis."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plots import Style, _ramp, _colorbar, panel_label, _save, INK, INK2, ACCENT, OTHER
from .spin import MODELS, interface_polarisation, magnetisation


def spin_tables(sr):
    out = {}
    tab = sr.table.copy()
    best = sr.fits[sr.best]
    tab["MR_fit"] = best.pred
    out["Spin MR data and fit"] = tab
    rows = []
    for k in sr.ranking:
        f = sr.fits[k]
        rows.append({"model": MODELS[k].name, "key": k, "support": sr.probs[k], "AIC": f.aic,
                     "delta_AIC": f.aic - min(x.aic for x in sr.fits.values()), "k": f.k, "rss": f.rss})
    out["Spin model comparison"] = pd.DataFrame(rows)
    prow = []
    for k in sr.ranking[:2]:
        f = sr.fits[k]
        for q, (name, lo, hi, init, unit, desc) in zip(f.params, MODELS[k].params):
            v = f.params[q]
            at_bound = abs(v - lo) < 1e-6 * max(abs(hi - lo), 1) or abs(v - hi) < 1e-6 * max(abs(hi - lo), 1)
            ci = f.ci95.get(q, (np.nan, np.nan))
            prow.append({"model": k, "parameter": q, "unit": unit, "value": v, "SE": f.errors.get(q),
                         "boot95_lo": ci[0], "boot95_hi": ci[1], "at_bound_(undetermined)": at_bound, "meaning": desc})
    out["Spin parameters"] = pd.DataFrame(prow)
    out["Spin fingerprints"] = pd.DataFrame([{"item": k, "value": str(v)} for k, v in sr.fingerprints.items()])
    return out


def spin_figures(sr, outdir, style: Style = None):
    style = style or Style()
    tab = sr.table
    best = sr.fits[sr.best]
    pred = best.pred
    Ns = sorted(np.unique(tab.N)); Ts = sorted(np.unique(tab["T"]))
    paths = {}
    with style.ctx():
        n = min(len(Ns), 3)
        sel = Ns if len(Ns) <= 3 else [Ns[0], Ns[len(Ns) // 2], Ns[-1]]
        fig, axes = plt.subplots(1, n, figsize=(style.single if n == 1 else style.double, 2.5), squeeze=False,
                                 sharey=True)
        tcol = dict(zip(Ts, _ramp("Purples", len(Ts))))
        for i, N in enumerate(sel):
            ax = axes[0, i]
            for T in Ts:
                m = (tab.N == N) & (tab["T"] == T)
                if not m.any():
                    continue
                o = np.argsort(tab.V[m].values)
                ax.plot(tab.V[m].values[o], 100 * tab.MR[m].values[o], "o", mfc="none", mec=tcol[T], ms=2.8, mew=0.6)
                ax.plot(tab.V[m].values[o], 100 * pred[m.values][o], "-", color=tcol[T], lw=1.0)
            ax.axhline(0, color=INK2, lw=0.5)
            ax.set_xlabel(r"$V$ (V)")
            if i == 0:
                ax.set_ylabel("MR (%)")
            ax.text(0.97, 0.95, rf"$N$ = {int(N)}", transform=ax.transAxes, ha="right", va="top", fontsize=style.fs - 1)
            if n > 1:
                panel_label(ax, i, style)
        if len(Ts) > 1:
            _colorbar(fig, list(axes[0]), Ts, r"$T$ (K)")
        paths["fig12_spin_MR_bias"] = _save(fig, outdir, "fig12_spin_MR_bias", style)

        fig, axes = plt.subplots(1, 2, figsize=(style.double, 2.5))
        ax = axes[0]
        low = tab[tab.V <= tab.V.quantile(0.25)]
        g = low.groupby(["N", "T"]).MR.median().reset_index()
        gp = pd.Series(pred[(tab.V <= tab.V.quantile(0.25)).values]).groupby(
            [low.N.values, low["T"].values]).median().reset_index()
        for T in Ts:
            q = g[g["T"] == T]; qp = gp[gp["level_1"] == T]
            ax.plot(q.N, 100 * q.MR, "o", mfc="none", mec=tcol[T], ms=3.2, mew=0.7)
            ax.plot(qp["level_0"], 100 * qp[0], "-", color=tcol[T], lw=0.9)
        ax.set_xlabel(r"layer number $N$"); ax.set_ylabel("MR at low bias (%)")
        panel_label(ax, 0, style)
        ax = axes[1]
        if sr.best == "proximity":
            p = best.params
            E = np.linspace(-80, 80, 400)
            for T, col in zip([Ts[0], Ts[-1]], [ACCENT, "#eb6834"]):
                D = p["Delta"] * magnetisation(T, p["TN"])
                ax.plot(E, interface_polarisation(E, D, p["mu"], p["Gamma"]), "-", color=col, label=f"{T:g} K")
            ax.axvline(0, color=INK2, lw=0.5, ls=":"); ax.axhline(0, color=INK2, lw=0.5)
            ax.set_xlabel(r"$E - E_F$ (meV)"); ax.set_ylabel(r"interface polarisation $P(E)$")
            ax.legend(loc="best")
        else:
            lowT = g.groupby("T").MR.median()
            ax.plot(lowT.index, 100 * lowT.values, "o-", color=ACCENT, mfc="none", ms=3.2, lw=0.7)
            ax.set_xlabel(r"$T$ (K)"); ax.set_ylabel("MR at low bias (%)")
        panel_label(ax, 1, style)
        paths["fig13_spin_scaling"] = _save(fig, outdir, "fig13_spin_scaling", style)

        fig, ax = plt.subplots(figsize=(style.single, 2.2))
        keys = sr.ranking
        y = np.arange(len(keys))
        ax.barh(y, [100 * sr.probs[k] for k in keys], color=[ACCENT if k == sr.best else OTHER for k in keys], height=0.6)
        ax.set_yticks(y); ax.set_yticklabels([MODELS[k].name.split("(")[0].strip()[:34] for k in keys],
                                             fontsize=style.fs - 2.5)
        ax.tick_params(axis="y", which="both", left=False, right=False); ax.invert_yaxis()
        ax.set_xlabel("support (%)"); ax.set_xlim(0, 105)
        paths["fig14_spin_models"] = _save(fig, outdir, "fig14_spin_models", style)
    return paths


def export_spin(sr, outdir, style=None):
    os.makedirs(outdir, exist_ok=True)
    figs = spin_figures(sr, os.path.join(outdir, "figures"), style)
    tabs = spin_tables(sr)
    tdir = os.path.join(outdir, "tables"); os.makedirs(tdir, exist_ok=True)
    for k, df in tabs.items():
        df.to_csv(os.path.join(tdir, k.lower().replace(" ", "_") + ".csv"), index=False)
    with pd.ExcelWriter(os.path.join(outdir, "spin_results.xlsx"), engine="openpyxl") as w:
        for k, df in tabs.items():
            df.to_excel(w, sheet_name=k[:31], index=False)
    with open(os.path.join(outdir, "spin_summary.txt"), "w") as fh:
        fh.write(sr.statement + "\n")
    return {"figures": figs, "tables": tabs}
