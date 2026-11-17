"""Model-free fingerprints (review Secs. XVII-XX).

All quantities here are computed without assuming a mechanism:
logarithmic derivative γ = d ln I / d ln V, the Rowell ratio, Arrhenius
activation energy Ea(V) and its √V dependence, Zabrodskii's w(T), thickness
scaling and scaling-collapse tests, power-law exponents vs T, absolute
magnitude in units of G0 per channel, hysteresis, NDR and zero-bias gaps.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import e, eps0, G0, kB_eV


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def clean_curve(g: pd.DataFrame) -> pd.DataFrame:
    """Positive branch, duplicates (up/down sweeps) averaged in ln I, sorted by V."""
    g = g[(g.V > 0) & (g.I > 0)].copy()
    if g.empty:
        return g
    g["lnI"] = np.log(g.I)
    g["Vr"] = g.V.round(9)
    out = g.groupby("Vr", as_index=False).agg(V=("V", "mean"), lnI=("lnI", "mean"), T=("T", "first"),
                                              d=("d", "first"), area=("area", "first"))
    out["I"] = np.exp(out.lnI)
    return out.sort_values("V").reset_index(drop=True)


def local_slope(x, y, half=2):
    """dy/dx by local quadratic fits (robust on non-uniform grids)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x)
    out = np.full(n, np.nan)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        edge = False
        if hi - lo < 3:
            if n < 3:
                continue
            lo, hi = (0, 3) if i < n / 2 else (n - 3, n)
            edge = True
        xx = x[lo:hi] - x[i]
        deg = 2 if (hi - lo >= 4 or edge) else 1
        try:
            c = np.polyfit(xx, y[lo:hi], deg)
            out[i] = c[-2]
        except Exception:
            pass
    return out


def _linfit(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3 or np.ptp(x) == 0:
        return None
    A = np.vstack([x, np.ones_like(x)]).T
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ coef
    ss = np.sum((y - y.mean()) ** 2)
    r2 = 1 - np.sum((y - yhat) ** 2) / ss if ss > 0 else 1.0
    n = len(x)
    s2 = np.sum((y - yhat) ** 2) / max(n - 2, 1)
    cov = s2 * np.linalg.pinv(A.T @ A)
    return {"slope": float(coef[0]), "intercept": float(coef[1]), "r2": float(r2),
            "slope_err": float(np.sqrt(max(cov[0, 0], 0))), "n": n}


def _interp_lnI(c: pd.DataFrame, V):
    return np.interp(np.log(V), np.log(c.V.values), c.lnI.values, left=np.nan, right=np.nan)


def _groups_by(ds, key):
    """{value: [clean curves]} grouped by 'T' or 'd'."""
    out = {}
    for _, g in ds.curves():
        c = clean_curve(g)
        if len(c) >= 3:
            out.setdefault(float(g[key].iloc[0]), []).append(c)
    return out


# ---------------------------------------------------------------------------
# individual fingerprints
# ---------------------------------------------------------------------------
def gamma_analysis(ds):
    curves = []
    for cid, g in ds.curves():
        c = clean_curve(g)
        if len(c) < 5:
            continue
        lv, li = np.log(c.V.values), c.lnI.values
        gam = local_slope(lv, li, half=2)
        n = len(gam)
        lowb = float(np.nanmedian(gam[: max(2, n // 6)]))
        hib = float(np.nanmedian(gam[-max(2, n // 4):]))
        imax = int(np.nanargmax(gam))
        gmax = float(gam[imax])
        after = gam[imax:]
        peak = bool(gmax > 2.4 and imax < n - 2 and np.nanmin(after) < 0.8 * gmax and imax > 1)
        upper = gam[n // 2:]
        flat_upper = bool(len(upper) >= 3 and np.nanstd(upper) / max(abs(np.nanmean(upper)), 1e-9) < 0.12)
        mid = gam[n // 4: 3 * n // 4]
        plateau2 = bool(np.nanmedian(np.abs(mid - 2.0)) < 0.25) if len(mid) >= 3 else False
        # local scatter (noise) estimate: residual of ln I around a 5-point quadratic
        resid = []
        for i in range(2, n - 2):
            cc = np.polyfit(lv[i - 2:i + 3], li[i - 2:i + 3], 2)
            resid.append(li[i] - np.polyval(cc, lv[i]))
        noise = float(np.std(resid) * 1.4) if len(resid) >= 3 else np.nan
        curves.append({"curve": int(cid), "T": float(c["T"].iloc[0]), "d": float(c["d"].iloc[0]),
                       "V": c.V.values, "gamma": gam, "gamma_low": lowb, "gamma_high": hib,
                       "gamma_max": gmax, "V_gamma_max": float(c.V.values[imax]), "peak": peak,
                       "plateau2": plateau2, "flat_upper": flat_upper, "noise_lnI": noise})
    return curves


def gamma_peak_scaling(gcurves):
    """V of the γ peak vs thickness at fixed T: V_TFL ∝ d² (SCLC), ∝ d (field), ∝ d⁰ (interface)."""
    rows = [c for c in gcurves if c["peak"]]
    if not rows:
        return None
    df = pd.DataFrame([{"T": r["T"], "d": r["d"], "Vp": r["V_gamma_max"]} for r in rows])
    best = None
    for T, g in df.groupby("T"):
        g = g.groupby("d", as_index=False).Vp.mean()
        if len(g) >= 2:
            f = _linfit(np.log(g.d), np.log(g.Vp)) if len(g) >= 3 else {
                "slope": float(np.diff(np.log(g.Vp))[0] / np.diff(np.log(g.d))[0]), "r2": 1.0, "n": 2}
            if best is None or f["n"] > best["n"]:
                best = {"T": float(T), "exponent": f["slope"], "r2": f["r2"], "n": f["n"],
                        "d": g.d.tolist(), "Vpeak": g.Vp.tolist()}
    return best or {"exponent": None, "n_peaks": len(rows)}


def linearisations(ds):
    """R² of the standard linearising plots on the upper half of each curve (informative only)."""
    out = []
    for cid, g in ds.curves():
        c = clean_curve(g)
        if len(c) < 6:
            continue
        h = c.iloc[len(c) // 2:]
        V, I = h.V.values, h.I.values
        row = {"curve": int(cid), "T": float(c["T"].iloc[0]), "d": float(c["d"].iloc[0])}
        for name, x, y in [("Fowler–Nordheim  ln(I/V²) vs 1/V", 1 / V, np.log(I / V**2)),
                           ("Schottky  ln I vs √V", np.sqrt(V), np.log(I)),
                           ("Poole–Frenkel  ln(I/V) vs √V", np.sqrt(V), np.log(I / V)),
                           ("power law  ln I vs ln V", np.log(V), np.log(I)),
                           ("hopping  ln I vs V", V, np.log(I))]:
            f = _linfit(x, y)
            row[name] = f["r2"] if f else np.nan
        out.append(row)
    return out


def fn_minimum(ds):
    """Transition voltage V_T (minimum of ln(I/V²) vs 1/V, i.e. γ = 2) per curve and its d scaling."""
    rows = []
    for cid, g in ds.curves():
        c = clean_curve(g)
        if len(c) < 6:
            continue
        y = c.lnI.values - 2 * np.log(c.V.values)
        i = int(np.argmin(y))
        if 0 < i < len(y) - 1:
            rows.append({"T": float(c["T"].iloc[0]), "d": float(c["d"].iloc[0]), "VT": float(c.V.values[i])})
    res = {"points": rows, "exponent": None}
    if rows:
        df = pd.DataFrame(rows)
        for T, gg in df.groupby("T"):
            gg = gg.groupby("d", as_index=False).VT.mean()
            if len(gg) >= 3:
                f = _linfit(np.log(gg.d), np.log(gg.VT))
                res.update({"exponent": f["slope"], "r2": f["r2"], "T": float(T)})
                break
    return res


def _lowbias_G(ds):
    """Low-bias conductance G(T) for each thickness, at a common reference bias."""
    out = {}
    for d, curves in _groups_by(ds, "d").items():
        Vref = max(c.V.min() for c in curves)
        Vref = max(Vref, min(c.V.max() for c in curves) * 0.02)
        rows = []
        for c in curves:
            lnI = _interp_lnI(c, Vref)
            if np.isfinite(lnI):
                rows.append((float(c["T"].iloc[0]), float(np.exp(lnI) / Vref), float(c.area.iloc[0])))
        if len(rows) >= 2:
            df = pd.DataFrame(rows, columns=["T", "G", "area"]).groupby("T", as_index=False).mean()
            out[d] = {"Vref": float(Vref), "T": df["T"].values, "G": df["G"].values, "area": df["area"].values}
    return out


def rowell(Gd):
    res = []
    for d, r in Gd.items():
        T, G = r["T"], r["G"]
        if len(T) < 2:
            continue
        ratio = float(G[np.argmax(T)] / G[np.argmin(T)])
        Tlo, Thi = float(T.min()), float(T.max())
        if ratio < 0.95:
            cls = "metallic (G falls on heating) → pinholes / filaments"
        elif ratio <= 1.5:
            cls = "weak, insulating-like → coherent tunnelling, TAT, cotunnelling, band-to-band"
        elif ratio < 5:
            cls = "moderate → weakly activated or mixed channels"
        else:
            cls = "strong → activation: emission, hopping, SCLC with traps, ionic"
        res.append({"d": d, "Vref": r["Vref"], "T_low": Tlo, "T_high": Thi, "ratio": ratio, "class": cls,
                    "window_ok": bool(Tlo <= 100 and Thi >= 200)})
    return res


def zabrodskii(Gd):
    """w(T) = d ln G / d ln T at low bias; log-log slope = -p."""
    res = []
    for d, r in Gd.items():
        T, G = r["T"], r["G"]
        if len(T) < 4:
            continue
        lnT, lnG = np.log(T), np.log(G)
        w = local_slope(lnT, lnG, half=1)
        entry = {"d": d, "T": T.tolist(), "w": w.tolist(), "p": None}
        pos = np.isfinite(w) & (w > 0)
        if np.all(w[np.isfinite(w)] < 0):
            entry["class"] = "w < 0: metallic"
        elif pos.sum() >= 3:
            f = _linfit(lnT[pos], np.log(w[pos]))
            entry["p"] = -f["slope"]
            entry["p_err"] = f["slope_err"]
            entry["r2"] = f["r2"]
            entry["w_decades"] = float(np.log10(w[pos].max() / w[pos].min()))
            p = entry["p"]
            if p < -0.2:
                entry["class"] = "w rises with T → tunnelling-like (thermal smearing)"
            elif abs(p - 1) < 0.2:
                entry["class"] = "p ≈ 1 → simple activation (NN hopping / emission / polaron)"
            elif abs(p - 0.5) < 0.12:
                entry["class"] = "p ≈ 1/2 → Efros–Shklovskii VRH (or GDM-like curvature)"
            elif abs(p - 1 / 3) < 0.06:
                entry["class"] = "p ≈ 1/3 → 2D Mott VRH"
            elif abs(p - 0.25) < 0.08:
                entry["class"] = "p ≈ 1/4 → 3D Mott VRH"
            elif abs(p - 2) < 0.4:
                entry["class"] = "p ≈ 2 → Gaussian disorder (ln G ∝ −1/T²)"
            else:
                entry["class"] = f"p = {p:.2f}: no canonical law"
        else:
            entry["class"] = "insufficient positive w"
        res.append(entry)
    return res


def arrhenius(ds, dev):
    """Ea(V) per thickness; tests the √V lowering of Schottky / Poole–Frenkel."""
    res = []
    for d, curves in _groups_by(ds, "d").items():
        Ts = sorted({float(c["T"].iloc[0]) for c in curves})
        if len(Ts) < 3:
            continue
        vlo = max(c.V.min() for c in curves)
        vhi = min(c.V.max() for c in curves)
        if vhi <= vlo * 1.05:
            continue
        Vg = np.geomspace(vlo, vhi, 14)
        rows = []
        for V in Vg:
            pts = [(float(c["T"].iloc[0]), _interp_lnI(c, V)) for c in curves]
            pts = [(T, l) for T, l in pts if np.isfinite(l)]
            if len(pts) < 3:
                continue
            T = np.array([p[0] for p in pts]); l = np.array([p[1] for p in pts])
            f = _linfit(1 / T, l)
            f2 = _linfit(1 / T, l - 2 * np.log(T))
            curv = np.polyfit(1 / T, l, 2)[0] if len(T) >= 4 else 0.0
            rows.append({"V": float(V), "Ea": -f["slope"] * kB_eV, "Ea_err": f["slope_err"] * kB_eV,
                         "r2": f["r2"], "Ea_T2": -f2["slope"] * kB_eV, "curvature": float(curv)})
        if len(rows) < 3:
            continue
        df = pd.DataFrame(rows)
        entry = {"d": d, "table": df.to_dict("list"), "T_used": Ts}
        f = _linfit(np.sqrt(df.V), df.Ea)
        dm = d * 1e-9
        if f is not None:
            beta = -f["slope"] * np.sqrt(dm)          # eV per sqrt(V/m)
            entry.update({"sqrtV_slope": f["slope"], "sqrtV_r2": f["r2"], "Ea0": f["intercept"],
                          "beta": beta})
            if beta > 0:
                entry["eps_schottky"] = float(e / (4 * np.pi * eps0 * beta**2))
                entry["eps_pf"] = float(e / (np.pi * eps0 * beta**2))
        Ea = df.Ea.values
        entry["Ea_range"] = [float(Ea.min()), float(Ea.max())]
        entry["Ea_lowV"] = float(Ea[0])
        entry["Ea_decreases"] = bool(Ea[-1] < Ea[0] - max(0.1 * abs(Ea[0]), 0.005))
        entry["Ea_flat"] = bool(np.ptp(Ea) < max(0.1 * abs(np.mean(Ea)), 0.005))
        entry["arrhenius_r2_median"] = float(np.median(df.r2))
        res.append(entry)
    return res


def exponent_vs_T(gcurves):
    """High-bias power-law exponent vs temperature (Mark–Helfrich: l+1 = Tc/T + 1)."""
    if not gcurves:
        return None
    flat = [c for c in gcurves if c["flat_upper"]]
    if len(flat) < max(3, 0.6 * len(gcurves)):
        return {"powerlaw": False, "note": "high-bias γ is not constant: no single power law per curve",
                "slope_vs_invT": None, "significant": None}
    df = pd.DataFrame([{"T": c["T"], "d": c["d"], "n": c["gamma_high"]} for c in flat])
    best = None
    for d, g in df.groupby("d"):
        g = g.groupby("T", as_index=False).n.mean()
        if len(g) >= 3:
            f = _linfit(1 / g["T"], g["n"])
            sig = bool(np.ptp(g["n"]) > 0.4 and f["r2"] > 0.6)
            entry = {"powerlaw": True, "d": float(d), "T": g["T"].tolist(), "n": g["n"].tolist(), "slope_vs_invT": f["slope"],
                     "r2": f["r2"], "significant": sig, "Tc_est": f["slope"] if f["slope"] > 0 else None}
            if best is None or len(g) > len(best["T"]):
                best = entry
    return best


def thickness_scaling(ds):
    """Thickness dependence at fixed V and scaling-collapse tests."""
    out = []
    byT = {}
    for _, g in ds.curves():
        c = clean_curve(g)
        if len(c) >= 3:
            c["J"] = c.I / c.area
            c["lnJ"] = np.log(c.J)
            byT.setdefault(float(g["T"].iloc[0]), []).append(c)
    for T, curves in byT.items():
        ds_ = sorted({float(c["d"].iloc[0]) for c in curves})
        if len(ds_) < 2:
            continue
        # merge duplicates per thickness
        perd = {}
        for c in curves:
            perd.setdefault(float(c["d"].iloc[0]), c)
        curves = [perd[d] for d in ds_]
        vlo = max(c.V.min() for c in curves)
        vhi = min(c.V.max() for c in curves)
        entry = {"T": T, "d": ds_}
        if vhi > vlo * 1.05:
            Vg = np.geomspace(vlo, vhi, 8)
            exp_fits, pow_fits = [], []
            for V in Vg:
                lnJ = np.array([np.interp(np.log(V), np.log(c.V), c.lnJ) for c in curves])
                dd = np.array(ds_)
                if len(dd) >= 3:
                    fe = _linfit(dd, lnJ); fp = _linfit(np.log(dd), lnJ)
                else:
                    sl = (lnJ[1] - lnJ[0]); fe = {"slope": sl / (dd[1] - dd[0]), "r2": np.nan}
                    fp = {"slope": sl / np.log(dd[1] / dd[0]), "r2": np.nan}
                exp_fits.append(fe); pow_fits.append(fp)
            entry["kappa_nm"] = float(-np.median([f["slope"] for f in exp_fits]) / 2)
            entry["exp_r2"] = float(np.nanmedian([f["r2"] for f in exp_fits]))
            entry["power_exponent"] = float(np.median([f["slope"] for f in pow_fits]))
            entry["power_r2"] = float(np.nanmedian([f["r2"] for f in pow_fits]))
            entry["kappa_bias_drift"] = float(np.ptp([f["slope"] for f in exp_fits]) / 2)
        # collapse tests
        hyp = {
            "field-driven  J = f(V/d)": (lambda c: c.V / c["d"], lambda c: c.lnJ),
            "SCLC  J/d = f(V/d²)": (lambda c: c.V / c["d"] ** 2, lambda c: c.lnJ - np.log(c["d"])),
            "contact-limited  J = f(V)": (lambda c: c.V, lambda c: c.lnJ),
            "ohmic bulk  J·d = f(V)": (lambda c: c.V, lambda c: c.lnJ + np.log(c["d"])),
        }
        scores = {}
        for name, (fx, fy) in hyp.items():
            xs = [np.log(fx(c).values) for c in curves]
            ys = [fy(c).values for c in curves]
            lo = max(x.min() for x in xs); hi = min(x.max() for x in xs)
            if hi <= lo:
                scores[name] = None
                continue
            xg = np.linspace(lo, hi, 20)
            Y = np.array([np.interp(xg, x, y) for x, y in zip(xs, ys)])
            scores[name] = float(np.mean(np.std(Y, axis=0)))
        entry["collapse"] = scores
        valid = {k: v for k, v in scores.items() if v is not None}
        if valid:
            bestk = min(valid, key=valid.get)
            entry["best_collapse"] = bestk
            entry["collapse_contrast"] = float(sorted(valid.values())[1] / max(valid[bestk], 1e-3)) \
                if len(valid) > 1 else None
        out.append(entry)
    return out


def absolute_magnitude(ds, dev, Gd):
    """Low-bias conductance per transverse channel, T̄ = G / (N_ch G0)."""
    res = []
    kF = dev.kF_per_A * 1e10
    for d, r in Gd.items():
        G = float(np.max(r["G"]))
        A = float(r["area"][0])
        entry = {"d": d, "G_max_S": G, "G_over_G0": G / G0}
        if ds.density:
            # G is a conductance per m^2
            Nch_per_m2 = kF**2 / (4 * np.pi)
            entry["T_per_channel"] = G / (Nch_per_m2 * G0)
        elif ds.area_known:
            Nch = A * kF**2 / (4 * np.pi)
            entry["N_channels"] = Nch
            entry["T_per_channel"] = G / (Nch * G0)
        res.append(entry)
    return res


def hysteresis(ds):
    res = []
    for cid, g in ds.df.groupby("curve"):
        g = g[(g.V > 0) & (g.I > 0)]
        up, dn = g[g.sweep == "up"], g[g.sweep == "down"]
        if len(up) < 3 or len(dn) < 3:
            continue
        up = up.sort_values("V"); dn = dn.sort_values("V")
        lo, hi = max(up.V.min(), dn.V.min()), min(up.V.max(), dn.V.max())
        if hi <= lo:
            continue
        Vg = np.geomspace(lo, hi, 15)
        a = np.interp(np.log(Vg), np.log(up.V), np.log(up.I))
        b = np.interp(np.log(Vg), np.log(dn.V), np.log(dn.I))
        res.append({"curve": int(cid), "T": float(g["T"].iloc[0]), "d": float(g["d"].iloc[0]),
                    "max_ratio": float(np.exp(np.max(np.abs(a - b))))})
    return res


def ndr(ds, noise=0.05):
    res = []
    for cid, g in ds.curves():
        c = clean_curve(g)
        I = c.I.values
        drops = np.where(np.diff(I) < -noise * I[:-1] * 3)[0]
        if len(drops):
            res.append({"curve": int(cid), "T": float(c["T"].iloc[0]), "d": float(c["d"].iloc[0]),
                        "V_peak": float(c.V.values[drops[0]]),
                        "PVR": float(I[drops[0]] / I[drops[0] + 1:].min())})
    return res


def zero_bias_gap(gcurves):
    if not gcurves:
        return None
    df = pd.DataFrame([{k: c[k] for k in ("T", "d", "gamma_max", "V_gamma_max")} |
                       {"Vmax": float(np.max(c["V"])), "gh": c["gamma_high"]} for c in gcurves])
    hits = []
    for d, g in df.groupby("d"):
        lo = g.loc[g["T"].idxmin()]
        hi = g.loc[g["T"].idxmax()]
        if lo.gamma_max > 4 and lo.V_gamma_max < 0.3 * lo.Vmax and lo.gh < 0.6 * lo.gamma_max:
            closes = bool(len(g) > 1 and hi.gamma_max < 0.6 * lo.gamma_max)
            hits.append({"d": float(d), "T": float(lo["T"]), "V_threshold": float(lo.V_gamma_max),
                         "closes_on_heating": closes})
    return hits


def asymmetry(ds):
    res = []
    for cid, g in ds.df.groupby("curve"):
        pos = g[g.V > 0]; neg = g[g.V < 0]
        if len(pos) < 3 or len(neg) < 3:
            continue
        Vm = min(pos.V.max(), -neg.V.min())
        ip = np.interp(Vm, pos.V.sort_values(), pos.sort_values("V").I.abs())
        ineg = np.interp(-Vm, neg.V.sort_values(), neg.sort_values("V").I.abs())
        if ineg > 0:
            res.append({"curve": int(cid), "T": float(g["T"].iloc[0]), "d": float(g["d"].iloc[0]),
                        "V": float(Vm), "rectification": float(ip / ineg)})
    return res


# ---------------------------------------------------------------------------
def run(ds, dev) -> dict:
    gcurves = gamma_analysis(ds)
    Gd = _lowbias_G(ds)
    diag = {
        "coverage": ds.coverage(),
        "gamma": gcurves,
        "gamma_peak_scaling": gamma_peak_scaling(gcurves),
        "linearisations": linearisations(ds),
        "fn_minimum": fn_minimum(ds),
        "lowbias_G": {str(k): {kk: (vv.tolist() if hasattr(vv, "tolist") else vv) for kk, vv in v.items()}
                      for k, v in Gd.items()},
        "rowell": rowell(Gd),
        "zabrodskii": zabrodskii(Gd),
        "arrhenius": arrhenius(ds, dev),
        "exponent_vs_T": exponent_vs_T(gcurves),
        "thickness": thickness_scaling(ds),
        "absolute": absolute_magnitude(ds, dev, Gd),
        "hysteresis": hysteresis(ds),
        "ndr": ndr(ds),
        "gap": zero_bias_gap(gcurves),
        "asymmetry": asymmetry(ds),
    }
    nz = [c["noise_lnI"] for c in gcurves if np.isfinite(c["noise_lnI"])]
    diag["noise_lnI"] = float(np.median(nz)) if nz else None
    return diag
