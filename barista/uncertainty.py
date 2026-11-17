"""How certain is the mechanism assignment?  Full numerical/statistical analysis.

Everything here quantifies uncertainty *after* the global fits of :mod:`engine`:

1. **Stratified bootstrap** (pairs resampled within every I(V) curve, B replicates,
   run in parallel): every competitive mechanism is refitted on every replicate, giving
   - the probability that each mechanism is selected (with a Wilson 95 % interval),
   - bootstrap distributions of all parameters → percentile confidence intervals
     (68 % and 95 %), bias and bootstrap standard errors,
   - the distribution of Akaike weights.
2. **Profile likelihood** intervals for every physical parameter of the assigned model
   (Δχ² = 1 and 3.84 for 68 % and 95 %), which do not assume a parabolic χ² surface.
3. **Asymptotic** (covariance-matrix) intervals with Student-t quantiles and the
   parameter correlation matrix.
4. **Fit-window stability** (review Sec. XVIII.E iii): refits on the lower/upper bias halves
   and with each temperature / thickness held out; drifts are expressed in units of σ.
5. **Residual diagnostics**: reduced χ² against the measured scatter, Shapiro–Wilk
   normality, Durbin–Watson and Wald–Wolfowitz runs tests per curve (systematic misfit).
6. **Model comparison statistics**: Akaike weights, BIC posterior probabilities,
   evidence ratios, Vuong's non-nested likelihood-ratio test (point-wise and
   curve-blocked) and a likelihood-ratio test for the two-channel extension.

The result is a single ``certainty`` grade with every number behind it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from .fitting import FitData, fit, Parallel
from .models import REGISTRY, short_name
from .models.base import LN10
from .parallel import Pool, cost_of


# ---------------------------------------------------------------------------
# worker tasks
# ---------------------------------------------------------------------------
def _mech(key, extra):
    if "+" in key and extra is not None:
        ka, kb, pa, pb = extra
        return Parallel(REGISTRY[ka], REGISTRY[kb], pa, pb)
    return REGISTRY[key]


def task_boot(rep, idx, data, dev, cands, seed):
    """Refit every candidate on one bootstrap replicate; returns {key: (rss, n, params)}."""
    sub = data.subset(idx)
    out = {}
    for key, warm, keff, extra in cands:
        m = _mech(key, extra)
        try:
            r = fit(m, sub, dev, hints=warm, n_starts=0, seed=seed + rep, uncertainties=False)
            out[key] = (r.rss, sub.n, {k: float(v) for k, v in r.params.items()})
        except Exception:
            out[key] = (np.nan, sub.n, {})
    return rep, out


def task_profile(key, extra, pname, value, data, dev, warm):
    m = _mech(key, extra)
    r = fit(m, data, dev, hints=warm, n_starts=0, fixed_extra={pname: value}, uncertainties=False)
    return pname, value, r.rss


def task_window(label, mask, key, extra, data, dev, warm):
    m = _mech(key, extra)
    r = fit(m, data.subset(mask), dev, hints=warm, n_starts=1, seed=7)
    return label, r.params, r.errors, int(mask.sum())


# ---------------------------------------------------------------------------
def _wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def _stratified_indices(curve, rng):
    idx = []
    for c in np.unique(curve):
        pos = np.where(curve == c)[0]
        idx.append(rng.choice(pos, size=len(pos), replace=True))
    return np.sort(np.concatenate(idx))


def _durbin_watson(r):
    r = np.asarray(r, float)
    return float(np.sum(np.diff(r) ** 2) / max(np.sum(r**2), 1e-300)) if len(r) > 2 else np.nan


def _runs_z(r):
    s = np.sign(r)
    s = s[s != 0]
    n1, n2 = np.sum(s > 0), np.sum(s < 0)
    if n1 < 2 or n2 < 2:
        return np.nan
    runs = 1 + np.sum(s[1:] != s[:-1])
    mu = 2 * n1 * n2 / (n1 + n2) + 1
    var = 2 * n1 * n2 * (2 * n1 * n2 - n1 - n2) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
    return float((runs - mu) / np.sqrt(var)) if var > 0 else np.nan


def _physical_params(mech, fit_res):
    names = []
    for p in mech.params:
        if p.name in fit_res.fixed or p.profile:
            continue
        base = p.name.split(".")[-1]
        if base in ("rel",):
            continue
        names.append(p.name)
    return names


@dataclass
class Certainty:
    grade: str
    summary: str
    candidates: list
    selection: dict                 # key -> {freq, ci_lo, ci_hi, wins}
    akaike: dict                    # key -> {weight, boot_mean, boot_lo, boot_hi}
    bic_posterior: dict
    evidence_ratio: dict
    params: dict                    # param -> table row (estimate, asymptotic, bootstrap, profile)
    profile: dict                   # param -> {grid, dchi2, ci68, ci95}
    windows: list
    residuals: dict
    vuong: dict
    lrt: dict
    boot_samples: dict              # key -> {param: array}
    corr: dict
    B: int
    workers: int
    timing: float
    assigned_key: str
    notes: list = field(default_factory=list)

    def as_dict(self):
        d = dict(self.__dict__)
        d["boot_samples"] = {k: {p: list(map(float, v)) for p, v in s.items()} for k, s in self.boot_samples.items()}
        return d


def assess(res, B=200, workers=None, top_k=4, seed=0, profile_points=13, progress=None) -> Certainty:
    say = progress or (lambda m: None)
    t0 = time.time()
    data: FitData = res.data
    dev = res.device
    ranked = res.ranked()
    notes = []

    # ---- candidates ------------------------------------------------------------
    cands = []
    seen = set()
    for a in ranked[:top_k] + sorted(res.assessments, key=lambda a: a.d_aic)[:2]:
        if a.key in seen:
            continue
        seen.add(a.key)
        cands.append((a.key, dict(a.fit.params), max(a.fit.k_eff, 1), None, a.fit))
    mixed_assigned = bool(res.mixed) and res.assigned_key == res.mixed["key"]
    if res.mixed:
        mx = res.mixed
        extra = (mx["a"], mx["b"], *mx["mech"]._split(mx["fit"].params))
        cands.append((mx["key"], dict(mx["fit"].params), max(mx["fit"].k_eff, 1), extra, mx["fit"]))
    keys = [c[0] for c in cands]
    assigned = res.assigned_key or res.best.key
    a_c = next(c for c in cands if c[0] == assigned)
    a_mech = _mech(assigned, a_c[3])
    a_fit = a_c[4]
    n = data.n

    with Pool(workers) as P:
        # ---- 1. bootstrap ------------------------------------------------------
        rng = np.random.default_rng(seed)
        idxs = [_stratified_indices(data.curve, rng) for _ in range(B)]
        task_c = [(k, w, ke, ex) for k, w, ke, ex, _ in cands]
        say(f"Bootstrap: {B} replicates × {len(cands)} models on {P.workers} worker(s)")
        boot = P.map(task_boot, [(i, idxs[i], data, dev, task_c, seed) for i in range(B)],
                     cost=lambda t: 1.0, progress=None)
        kmap = {k: ke for k, _, ke, _, _ in cands}
        aic = np.full((B, len(keys)), np.nan)
        samples = {k: {} for k in keys}
        for rep, out in boot:
            for j, k in enumerate(keys):
                rss, nn, prm = out.get(k, (np.nan, n, {}))
                if np.isfinite(rss) and rss > 0:
                    aic[rep, j] = nn * np.log(rss / nn) + 2 * kmap[k]
                for pn, pv in prm.items():
                    samples[k].setdefault(pn, []).append(pv)
        samples = {k: {p: np.array(v) for p, v in d.items()} for k, d in samples.items()}
        valid = np.all(np.isfinite(aic), axis=1)
        aicv = aic[valid]
        winners = np.argmin(aicv, axis=1) if len(aicv) else np.array([], int)
        selection = {}
        for j, k in enumerate(keys):
            w = int(np.sum(winners == j))
            lo, hi = _wilson(w, len(winners))
            selection[k] = {"wins": w, "freq": w / max(len(winners), 1), "ci_lo": lo, "ci_hi": hi}
        dA = aicv - aicv.min(axis=1, keepdims=True)
        wts = np.exp(-0.5 * dA); wts /= wts.sum(axis=1, keepdims=True)
        # ---- information criteria on the full data ------------------------------
        full_aic = np.array([c[4].aic for c in cands]); full_bic = np.array([c[4].bic for c in cands])
        aw = np.exp(-0.5 * (full_aic - full_aic.min())); aw /= aw.sum()
        bw = np.exp(-0.5 * (full_bic - full_bic.min())); bw /= bw.sum()
        akaike, bicp, er = {}, {}, {}
        ja = keys.index(assigned)
        for j, k in enumerate(keys):
            akaike[k] = {"weight": float(aw[j]), "boot_mean": float(np.mean(wts[:, j])) if len(wts) else np.nan,
                         "boot_lo": float(np.percentile(wts[:, j], 2.5)) if len(wts) else np.nan,
                         "boot_hi": float(np.percentile(wts[:, j], 97.5)) if len(wts) else np.nan}
            bicp[k] = float(bw[j])
            er[k] = float(np.exp(min(0.5 * (full_aic[j] - full_aic[ja]), 700)))

        # ---- 2. profile likelihood for the assigned model ------------------------
        pnames = _physical_params(a_mech, a_fit)
        s2 = a_fit.rss / max(n - a_fit.k_eff, 1)
        ptasks = []
        for pn in pnames:
            v0 = a_fit.params[pn]
            se = a_fit.errors.get(pn, np.nan)
            par = next(p for p in a_mech.params if p.name == pn)
            span = 5 * se if np.isfinite(se) and se > 0 else 0.2 * max(abs(v0), 0.05)
            span = max(span, 1e-4 * max(abs(v0), 1e-3))
            grid = np.linspace(max(par.lo, v0 - span), min(par.hi, v0 + span), profile_points)
            for g in grid:
                ptasks.append((assigned, a_c[3], pn, float(g), data, dev, dict(a_fit.params)))
        say(f"Profile likelihood: {len(pnames)} parameter(s) × {profile_points} points")
        pout = P.map(task_profile, ptasks, cost=lambda t: cost_of(t[0]))
        profile = {}
        for pn in pnames:
            pts = sorted([(v, r) for name, v, r in pout if name == pn])
            g = np.array([p[0] for p in pts]); rr = np.array([p[1] for p in pts])
            rmin = min(rr.min(), a_fit.rss)
            dchi = (rr - rmin) / s2
            entry = {"grid": g.tolist(), "dchi2": dchi.tolist()}
            for lab, thr in (("ci68", 1.0), ("ci95", 3.84)):
                inside = g[dchi <= thr]
                if len(inside):
                    lo_open = dchi[0] <= thr; hi_open = dchi[-1] <= thr
                    entry[lab] = [float(inside.min()), float(inside.max()), bool(lo_open), bool(hi_open)]
                else:
                    entry[lab] = None
            profile[pn] = entry

        # ---- 4. fit-window stability --------------------------------------------
        wtasks = []
        lowhalf = np.zeros(n, bool)
        for c in np.unique(data.curve):
            pos = np.where(data.curve == c)[0]
            med = np.median(data.V[pos])
            lowhalf[pos[data.V[pos] <= med]] = True
        wtasks.append(("lower half of each bias sweep", lowhalf))
        wtasks.append(("upper half of each bias sweep", ~lowhalf))
        Ts, ds_ = np.unique(data.T), np.unique(data.d)
        if len(Ts) >= 3:
            for T in Ts:
                wtasks.append((f"without T = {T:g} K", data.T != T))
        if len(ds_) >= 3:
            for d in ds_:
                wtasks.append((f"without d = {d * 1e9:g} nm", data.d != d))
        say(f"Fit-window stability: {len(wtasks)} refits")
        wout = P.map(task_window, [(lab, m, assigned, a_c[3], data, dev, dict(a_fit.params)) for lab, m in wtasks],
                     cost=lambda t: cost_of(t[2]))

    # ---- parameter table --------------------------------------------------------
    dof = max(n - a_fit.k_eff, 1)
    tq68, tq95 = stats.t.ppf(0.84, dof), stats.t.ppf(0.975, dof)
    params = {}
    smp = samples.get(assigned, {})
    for pn in pnames:
        v = a_fit.params[pn]; se = a_fit.errors.get(pn, np.nan)
        b = smp.get(pn, np.array([]))
        b = b[np.isfinite(b)]
        row = {"estimate": v, "se_asymptotic": se,
               "asym95": [v - tq95 * se, v + tq95 * se] if np.isfinite(se) else None,
               "asym68": [v - tq68 * se, v + tq68 * se] if np.isfinite(se) else None}
        if len(b) >= 10:
            row.update({"boot_median": float(np.median(b)), "boot_se": float(np.std(b, ddof=1)),
                        "boot68": [float(np.percentile(b, 16)), float(np.percentile(b, 84))],
                        "boot95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                        "boot_bias": float(np.mean(b) - v)})
        pr = profile.get(pn, {})
        row["profile68"] = pr.get("ci68"); row["profile95"] = pr.get("ci95")
        params[pn] = row

    # ---- window stability table ---------------------------------------------------
    windows = []
    for lab, prm, err, npts in wout:
        row = {"subset": lab, "n": npts}
        worst = 0.0
        for pn in pnames:
            v0, se0 = a_fit.params[pn], a_fit.errors.get(pn, np.nan)
            v = prm.get(pn, np.nan); e = err.get(pn, np.nan)
            row[pn] = v; row[pn + "_err"] = e
            tot = np.sqrt(np.nansum([se0**2 if np.isfinite(se0) else 0, e**2 if np.isfinite(e) else 0]))
            if tot > 0 and np.isfinite(v):
                z = abs(v - v0) / tot
                worst = max(worst, z)
            rel = abs(v - v0) / max(abs(v0), 1e-12)
            row[pn + "_rel_change"] = rel
        row["max_shift_sigma"] = worst
        windows.append(row)
    max_rel = max([max([w.get(pn + "_rel_change", 0) for pn in pnames] or [0]) for w in windows] or [0])

    # ---- 5. residual diagnostics ---------------------------------------------------
    r = data.lnI - a_fit.lnI_fit
    noise = (res.diag.get("noise_lnI") or np.nan)
    red_chi2 = float(np.mean(r**2) / noise**2) if np.isfinite(noise) and noise > 0 else np.nan
    sw = stats.shapiro(r[:5000]) if len(r) >= 8 else None
    dws, rz = [], []
    for c in np.unique(data.curve):
        pos = np.where(data.curve == c)[0]
        pos = pos[np.argsort(data.V[pos])]
        dws.append(_durbin_watson(r[pos])); rz.append(_runs_z(r[pos]))
    dws = np.array(dws); rz = np.array(rz)
    frac_sys = float(np.nanmean(np.abs(rz) > 2.58)) if len(rz) else np.nan
    residuals = {"rms_lnI": float(np.sqrt(np.mean(r**2))), "noise_lnI": noise, "reduced_chi2_vs_noise": red_chi2,
                 "shapiro_W": float(sw.statistic) if sw else np.nan, "shapiro_p": float(sw.pvalue) if sw else np.nan,
                 "durbin_watson_median": float(np.nanmedian(dws)), "runs_z_median": float(np.nanmedian(rz)),
                 "frac_curves_systematic": frac_sys, "residuals": r.tolist()}

    # ---- 6. pairwise tests -------------------------------------------------------------
    def pointwise_ll(fr):
        rr = data.lnI - fr.lnI_fit
        s2_ = np.mean(rr**2)
        return -0.5 * np.log(2 * np.pi * s2_) - rr**2 / (2 * s2_)

    vuong = {}
    l_a = pointwise_ll(a_fit)
    for k, w, ke, ex, fr in cands:
        if k == assigned:
            continue
        m_i = l_a - pointwise_ll(fr)
        corr_k = (a_fit.k_eff - fr.k_eff) * np.log(n) / 2      # Schwarz correction
        sd = np.std(m_i, ddof=1)
        z = (np.sum(m_i) - corr_k) / (np.sqrt(n) * sd) if sd > 0 else np.inf
        blk = np.array([np.sum(m_i[data.curve == c]) for c in np.unique(data.curve)])
        zb = (np.sum(blk) - corr_k) / (np.sqrt(len(blk)) * np.std(blk, ddof=1)) if len(blk) > 2 and np.std(blk) > 0 else np.nan
        vuong[k] = {"z": float(z), "p_one_sided": float(stats.norm.sf(z)),
                    "z_curve_blocked": float(zb) if np.isfinite(zb) else None,
                    "p_curve_blocked": float(stats.norm.sf(zb)) if np.isfinite(zb) else None,
                    "mean_loglik_gain_per_point": float(np.mean(m_i))}
    lrt = {}
    if res.mixed:
        f1 = res.best.fit; f2 = res.mixed["fit"]
        lr = n * np.log(max(f1.rss, 1e-300) / max(f2.rss, 1e-300))
        dk = max(f2.k_eff - f1.k_eff, 1)
        lrt = {"single": res.best.key, "two_channel": res.mixed["key"], "LR": float(lr), "df": int(dk),
               "p_value": float(stats.chi2.sf(lr, dk)),
               "note": "boundary (one channel → 0) makes this p-value approximate; AIC is the primary criterion"}

    corr = {}
    if a_fit.corr is not None:
        corr = {"names": a_fit.free, "matrix": np.nan_to_num(a_fit.corr).tolist()}

    # ---- grade ---------------------------------------------------------------------------
    sel = selection.get(assigned, {})
    p_sel, lo_sel = sel.get("freq", 0), sel.get("ci_lo", 0)
    runner = max((k for k in keys if k != assigned), key=lambda k: selection[k]["freq"], default=None)
    vu = vuong.get(runner, {}) if runner else {}
    pv = vu.get("p_curve_blocked") if vu.get("p_curve_blocked") is not None else vu.get("p_one_sided", 1.0)
    n_axes = 1 + int(res.diag["coverage"]["has_T_axis"]) + int(res.diag["coverage"]["has_d_axis"])
    stable = all(w["max_shift_sigma"] < 3 for w in windows) or max_rel < 0.1
    white = (frac_sys < 0.34) if np.isfinite(frac_sys) else True
    fit_ok = (red_chi2 < 4) if np.isfinite(red_chi2) else True
    score = 0
    score += 2 if lo_sel >= 0.9 else (1 if p_sel >= 0.7 else 0)
    score += 1 if (pv is not None and pv < 0.01) else 0
    score += 1 if stable else 0
    score += 1 if white and fit_ok else 0
    score += 1 if n_axes >= 2 else 0
    grade = {6: "VERY HIGH", 5: "HIGH", 4: "MODERATE"}.get(score, "LOW")
    if n_axes < 2 and grade in ("VERY HIGH", "HIGH"):
        grade = "MODERATE"
        notes.append("Data span only the bias axis: certainty is capped (review Sec. XVIII).")
    if not white:
        notes.append(f"{100 * frac_sys:.0f} % of curves show systematic residual runs (|z| > 2.58): "
                     "the model misses part of the shape.")
    if not stable:
        notes.append("Parameters shift by more than 3σ between fit windows: possible model misspecification "
                     "(review Sec. XVIII.E iii).")
    name = (f"{short_name(res.mixed['a'])} ∥ {short_name(res.mixed['b'])}" if mixed_assigned else short_name(assigned))
    summary = (f"Certainty of the assignment ({name}): {grade}. The mechanism is selected in "
               f"{100 * p_sel:.1f} % of {len(winners)} bootstrap replicates (95 % CI {100 * sel.get('ci_lo', 0):.1f}–"
               f"{100 * sel.get('ci_hi', 0):.1f} %); Akaike weight {aw[ja]:.3f}, BIC posterior {bw[ja]:.3f}.")
    if runner:
        summary += (f" Against the strongest alternative ({short_name(runner) if '+' not in runner else runner}) the "
                    f"evidence ratio is {er[runner]:.3g} and Vuong's test gives p = {pv:.2g}.")
    summary += (f" Residuals: rms {100 * (np.exp(residuals['rms_lnI']) - 1):.1f} % "
                f"(reduced χ² vs measured scatter {red_chi2:.2f}); parameters "
                f"{'stable' if stable else 'NOT stable'} across fit windows (largest shift "
                f"{max([w['max_shift_sigma'] for w in windows] or [0]):.1f}σ).")
    return Certainty(grade=grade, summary=summary, candidates=keys, selection=selection, akaike=akaike,
                     bic_posterior=bicp, evidence_ratio=er, params=params, profile=profile, windows=windows,
                     residuals=residuals, vuong=vuong, lrt=lrt, boot_samples=samples, corr=corr, B=B,
                     workers=P.workers, timing=time.time() - t0, assigned_key=assigned, notes=notes)
