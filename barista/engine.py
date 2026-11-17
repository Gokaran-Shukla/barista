"""Decision engine: from fingerprints + global fits to a mechanism assignment.

Implements the diagnostic protocol of Sec. XX of the review:

1. estimate the scales and decide which mechanisms are admissible (Fig. 10, user inputs);
2. absolute-magnitude checks (per-channel transmission, WKB / Richardson prefactors);
3. Rowell test;  4. temperature exponent (Zabrodskii);  5. thickness series and scaling collapse;
6. contacts vs bulk (series-resistance test, rectification);  7. hysteresis / NDR / gaps;
8. one model, all data: global fits ranked by AIC with cross-validation and physical-plausibility
   checks of every fitted parameter (Tables 6, 9, 18).

The final score of mechanism m is

    S_m = -½ min(ΔAIC_m, 80) + Σ evidence_m + Σ plausibility_m + CV_m (+ material prior),

and the relative support is softmax(S).  Necessary signatures that are violated give large
negative evidence; a mechanism whose parameters are unphysical is marked as excluded.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from . import diagnostics
from .device import DeviceInfo, MATERIAL_PRIORS
from .fitting import FitData, fit, cross_validate, Parallel, WithSeriesR, FitResult, _prof_name
from .models import REGISTRY, short_name
from .parallel import Pool, task_fit, task_cv, task_combo, task_series, cost_of
from .models.base import LN10

ACTIVATED = ["schottky", "poole_frenkel", "nn_hopping", "vrh", "polaron", "gdm", "sclc_trap", "sclc_exp"]
FIELD_DRIVEN = ["poole_frenkel", "nn_hopping", "polaron", "gdm", "vrh", "band_to_band"]
SCLC = ["sclc_trap", "sclc_exp"]

DISCRIMINATORS = {
    frozenset(["emission", "space-charge-limited"]):
        "a thickness series: SCLC collapses on J/d vs V/d², field-driven emission on J vs V/d (Fig. 56)",
    frozenset(["emission", "hopping"]):
        "Ea(V) and its √V slope compared with n² (refractive index), w(T) over a decade, and a thickness series",
    frozenset(["coherent tunnelling", "emission"]):
        "the temperature dependence at fixed bias down to ≤ 77 K: weak for tunnelling, activated for emission",
    frozenset(["coherent tunnelling", "defect-mediated tunnelling"]):
        "low-frequency noise (RTN), device-to-device statistics and the absolute current vs the WKB estimate",
    frozenset(["coherent tunnelling", "metallic paths"]):
        "the Rowell test (G(T) at low bias down to 4 K) and area scaling",
    frozenset(["hopping", "space-charge-limited"]):
        "a thickness series (I ∝ 1/d vs d⁻³; V_TFL ∝ d²) and pulsed I(V)",
    frozenset(["hopping"]):
        "the reduced activation energy w(T) over at least a decade (Zabrodskii) at several biases",
    frozenset(["emission"]):
        "ε_r from the √F slope compared with n², changing the electrode metal, and a global fit across T (T² vs F prefactor)",
    frozenset(["coherent tunnelling", "hopping"]):
        "a thickness series (ln I ∝ −d for tunnelling, I ∝ 1/d for hopping) and the Rowell ratio",
    frozenset(["coherent tunnelling", "space-charge-limited"]):
        "a thickness series (exponential vs V/d² scaling) and T at fixed V",
    frozenset(["defect-mediated tunnelling", "hopping"]):
        "a thickness series (dominant chain length N grows with d) and log G vs log T at low bias",
    frozenset(["space-charge-limited"]):
        "the exponent's temperature dependence (Mark–Helfrich) and V_TFL ∝ d²",
    frozenset(["defect-mediated tunnelling", "emission"]):
        "Arrhenius analysis at several biases (TAT is weakly activated) and noise spectroscopy",
    frozenset(["metallic paths", "hopping"]):
        "the sign of dG/dT and area scaling",
}


@dataclass
class Assessment:
    key: str
    name: str
    family: str
    fit: FitResult
    evidence: list = field(default_factory=list)     # (weight, text)
    checks: list = field(default_factory=list)       # Check
    score: float = 0.0
    prob: float = 0.0
    verdict: str = ""
    d_aic: float = 0.0

    def as_dict(self):
        return {"key": self.key, "name": self.name, "family": self.family, "score": self.score,
                "relative_support": self.prob, "verdict": self.verdict, "delta_aic": self.d_aic,
                "evidence": [{"weight": w, "text": t} for w, t in self.evidence],
                "checks": [{"level": c.level, "message": c.message, "weight": c.weight} for c in self.checks],
                "fit": self.fit.as_dict(), "summary": REGISTRY[self.key].summary(self.fit.params, self.fit.errors)}


@dataclass
class Result:
    dataset: object
    device: DeviceInfo
    diag: dict
    data: FitData
    assessments: list
    combos: list
    series: Optional[dict]
    best: Assessment
    confidence: str
    headline: str
    statement: str
    recommendations: list
    warnings: list
    regimes: Optional[dict]
    timing: dict
    mixed: Optional[dict] = None
    assigned_key: str = ""
    certainty: Optional[object] = None

    def ranked(self):
        return sorted(self.assessments, key=lambda a: -a.score)

    @property
    def traps(self):
        """Trap density / trap energy from the I–V data (recomputed once bootstrap samples exist)."""
        key = id(self.certainty)
        if getattr(self, "_traps_key", None) != key:
            from .traps import analyse
            self._traps = analyse(self)
            self._traps_key = key
        return self._traps


# ---------------------------------------------------------------------------
def _add(ev, keys, w, text):
    for k in ([keys] if isinstance(keys, str) else keys):
        if k in ev:
            ev[k].append((w, text))


def evidence(diag, ds, dev, keys):
    ev = {k: [] for k in keys}
    cov = diag["coverage"]

    # --- Rowell test --------------------------------------------------------
    rw = [r for r in diag["rowell"] if r["T_high"] / max(r["T_low"], 1e-9) >= 1.4]
    if rw:
        r = max(rw, key=lambda r: r["T_high"] / r["T_low"])
        R = r["ratio"]
        span = f"G({r['T_high']:.0f} K)/G({r['T_low']:.0f} K) = {R:.3g}"
        if R < 0.95:
            _add(ev, "ohmic_metallic", 3, f"Rowell: {span} < 1, conductance falls on heating (metallic)")
            _add(ev, "coherent", -4, f"Rowell: {span} < 1 falsifies tunnelling (G must rise on heating)")
            _add(ev, ACTIVATED + ["tat"], -2.5, f"Rowell: {span} < 1 is not activated")
        elif R <= 1.5:
            w = 3 if r["window_ok"] else 2
            _add(ev, ["coherent", "tat", "band_to_band"], w, f"Rowell: weak, insulating-like T dependence ({span})")
            _add(ev, ACTIVATED, -3, f"Rowell: {span} is far too weak for an activated mechanism")
            _add(ev, "ohmic_metallic", -2, f"Rowell: {span} > 1, not metallic")
        elif R < 5:
            _add(ev, "coherent", -1.5, f"Rowell: {span} stronger than the ≈10–30 % thermal smearing of tunnelling")
            _add(ev, "glazman_matveev", 1, f"Rowell: moderate {span}, compatible with power-law chains")
            _add(ev, "ohmic_metallic", -3, f"Rowell: {span} > 1, not metallic")
        else:
            _add(ev, ACTIVATED, 1.5, f"Rowell: strong activation, {span}")
            _add(ev, "coherent", -5, f"Rowell: {span} ≫ 1 violates the weak T dependence necessary for coherent tunnelling")
            _add(ev, "tat", -3, f"Rowell: {span} — strong Arrhenius behaviour falsifies elastic TAT")
            _add(ev, "band_to_band", -4, f"Rowell: {span} — band-to-band tunnelling is nearly T-independent")
            _add(ev, "ohmic_metallic", -5, f"Rowell: {span} — not metallic")
            _add(ev, "glazman_matveev", -1, f"Rowell: {span} larger than typical power-law chains")

    # --- Zabrodskii ----------------------------------------------------------
    zb = [z for z in diag["zabrodskii"] if z.get("p") is not None]
    if zb:
        z = max(zb, key=lambda z: len(z["T"]))
        p = z["p"]; dec = z.get("w_decades", 0)
        tag = f"Zabrodskii: w(T) ∝ T^−p with p = {p:.2f} ({dec:.2f} decade(s) of w)"
        if p < -0.2:
            _add(ev, ["coherent", "tat"], 2, tag + " — w rises with T (tunnelling-like)")
            _add(ev, ACTIVATED, -2, tag + " — not activated")
        elif abs(p - 1) < 0.2:
            _add(ev, "nn_hopping", 2, tag + " — single activation energy")
            _add(ev, ["polaron", "poole_frenkel", "schottky"], 1, tag + " — activated")
            _add(ev, "vrh", -2 if dec >= 0.5 else -1, tag + " — not a stretched exponential")
        elif 0.18 <= p <= 0.6:
            _add(ev, "vrh", 2 + (1 if dec >= 1 else 0), tag + " — variable-range hopping")
            _add(ev, ["nn_hopping", "polaron"], -2, tag + " — not a single activation energy")
        elif abs(p - 2) < 0.4:
            _add(ev, "gdm", 3, tag + " — ln G ∝ −1/T² (Gaussian disorder)")
        if dec < 1 and 0.18 <= p <= 0.6:
            _add(ev, "vrh", -0.5, "w spans less than one decade: p is only indicative (Sec. XVIII.A)")

    # --- Arrhenius / sqrt(V) lowering ---------------------------------------
    arr = diag["arrhenius"]
    if arr:
        a = max(arr, key=lambda a: len(a["T_used"]))
        epsS, epsPF = a.get("eps_schottky"), a.get("eps_pf")
        if a["Ea_decreases"] and a.get("sqrtV_r2", 0) > 0.95:
            _add(ev, ["schottky", "poole_frenkel"], 1.5,
                 f"Arrhenius: Ea falls linearly with √V (R² = {a['sqrtV_r2']:.3f}), Ea(0) = {a['Ea0']:.3f} eV")
            ref = dev.eps_optical
            if ref is not None and epsS and epsPF:
                for key, val, lab in [("schottky", epsS, "Schottky"), ("poole_frenkel", epsPF, "Poole–Frenkel")]:
                    rr = val / ref
                    if 0.6 <= rr <= 1.6:
                        _add(ev, key, 3, f"√V slope gives ε_r({lab}) = {val:.2f} ≈ n² = {ref:.2f}")
                    elif rr < 0.3 or rr > 3:
                        _add(ev, key, -3, f"√V slope gives ε_r({lab}) = {val:.2f}, inconsistent with n² = {ref:.2f}")
            elif epsS and epsPF:
                if epsS < 1.0:
                    _add(ev, "schottky", -2, f"√V slope implies ε_r(Schottky) = {epsS:.2f} < 1 — unphysical (Table 8)")
                if 1.5 <= epsPF <= 30:
                    _add(ev, "poole_frenkel", 1, f"√V slope implies ε_r(PF) = {epsPF:.2f} (plausible; give n_opt to test)")
        elif a["Ea_flat"]:
            _add(ev, ["schottky", "poole_frenkel"], -2.5, "Arrhenius: Ea independent of V — falsifies field-lowered emission")
            _add(ev, ["nn_hopping", "polaron", "sclc_trap"], 1, "Arrhenius: bias-independent activation energy (ohmic regime)")
        if a["arrhenius_r2_median"] < 0.97 and len(a["T_used"]) >= 4:
            _add(ev, ["vrh", "gdm"], 1, f"Arrhenius plots curved (median R² = {a['arrhenius_r2_median']:.3f})")
            _add(ev, ["nn_hopping", "schottky", "poole_frenkel"], -1, "Arrhenius plots are curved")

    # --- thickness series -----------------------------------------------------
    th = [t for t in diag["thickness"] if len(t["d"]) >= 2]
    if th:
        n3 = [t for t in th if len(t["d"]) >= 3]
        t = n3[0] if n3 else th[0]
        mult = 1.0 if len(t["d"]) >= 3 else 0.6
        best, contrast = t.get("best_collapse"), t.get("collapse_contrast") or 1.0
        if best and contrast >= 2.5:
            txt = f"thickness series ({len(t['d'])} d at {t['T']:.0f} K): best collapse on {best} (×{contrast:.0f} tighter)"
            if best.startswith("field"):
                _add(ev, FIELD_DRIVEN, 2.5 * mult, txt)
                _add(ev, "schottky", 1 * mult, txt + " — compatible (Schottky lowering depends on F)")
                _add(ev, SCLC, -3 * mult, txt + " — SCLC requires J/d = f(V/d²)")
                _add(ev, "ohmic_metallic", -2 * mult, txt)
            elif best.startswith("SCLC"):
                _add(ev, SCLC, 3.5 * mult, txt + " — necessary SCLC scaling satisfied")
                _add(ev, FIELD_DRIVEN + ["schottky"], -2.5 * mult, txt)
                _add(ev, "coherent", -3 * mult, txt)
            elif best.startswith("contact"):
                _add(ev, "ohmic_metallic", 2 * mult, txt + " — thickness independent")
                _add(ev, "schottky", 1 * mult, txt + " — interface limited")
                _add(ev, FIELD_DRIVEN + SCLC, -1.5 * mult, txt)
            elif best.startswith("ohmic"):
                _add(ev, ["nn_hopping", "vrh", "polaron"], 1.5 * mult, txt + " — I ∝ 1/d (bulk ohmic)")
                _add(ev, SCLC, -1 * mult, txt)
        if "kappa_nm" in t and len(t["d"]) >= 3:
            kap, r2 = t["kappa_nm"], t["exp_r2"]
            pe = t["power_exponent"]
            if r2 > 0.98 and 1.5 <= kap <= 15 and t.get("power_r2", 1) <= r2 + 1e-3:
                _add(ev, "coherent", 3, f"ln J ∝ −2κd with κ = {kap:.2f} nm⁻¹ (R² = {r2:.3f}) — necessary tunnelling signature")
                _add(ev, ["tat", "glazman_matveev"], 1, f"exponential thickness dependence, κ = {kap:.2f} nm⁻¹")
                _add(ev, ["nn_hopping", "vrh", "polaron", "gdm"], -2, "ln I ∝ −d falsifies bulk hopping")
            elif kap < 1.0:
                _add(ev, "coherent", -3, f"thickness dependence far too weak for tunnelling (κ = {kap:.2f} nm⁻¹)")
            if t.get("power_r2", 0) > 0.98:
                if abs(pe + 3) < 0.6:
                    _add(ev, "sclc_trap", 1.5, f"J ∝ d^{pe:.2f} ≈ d⁻³ (Mott–Gurney)")
                elif abs(pe + 1) < 0.35:
                    _add(ev, ["nn_hopping", "vrh", "polaron"], 1.5, f"J ∝ d^{pe:.2f} ≈ 1/d (ohmic bulk)")
    elif not cov["has_d_axis"]:
        pass

    # --- logarithmic derivative ------------------------------------------------
    g = diag["gamma"]
    if g:
        frac2 = np.mean([c["plateau2"] for c in g])
        if frac2 > 0.5:
            _add(ev, SCLC, 1.5, f"γ = d lnI/d lnV ≈ 2 over the mid-bias range in {100 * frac2:.0f} % of curves")
        flat = np.mean([c["flat_upper"] for c in g])
        if flat < 0.3:
            _add(ev, ["sclc_exp", "glazman_matveev"], -1, "high-bias γ keeps changing: not a single power law")
    ex = diag.get("exponent_vs_T")
    if ex and ex.get("powerlaw") and ex.get("significant"):
        _add(ev, "glazman_matveev", 0.5, "power-law exponent depends on T")

    fnm = diag.get("fn_minimum") or {}
    if fnm.get("exponent") is not None and abs(fnm["exponent"] - 1) < 0.3:
        _add(ev, ["coherent", "band_to_band"], 1, f"FN transition voltage scales as d^{fnm['exponent']:.2f} ≈ d")

    # --- absolute magnitude ------------------------------------------------------
    for a in diag["absolute"]:
        tpc = a.get("T_per_channel")
        if tpc is not None:
            if tpc > 0.1:
                _add(ev, "ohmic_metallic", 4, f"low-bias conductance = {tpc:.2g} G₀ per channel → metallic paths")
                _add(ev, "coherent", -4, f"T̄ = {tpc:.2g} per channel is incompatible with tunnelling (T̄ ≪ 1)")
            break

    # --- device-level features -----------------------------------------------------
    hy = [h for h in diag["hysteresis"] if h["max_ratio"] > 1.3]
    if hy:
        mr = max(h["max_ratio"] for h in hy)
        _add(ev, "nn_hopping", 1, f"I(V) hysteresis (up/down ratio {mr:.2f}): ionic motion is plausible")
        _add(ev, ["sclc_trap", "ohmic_metallic"], 0.5, f"I(V) hysteresis (×{mr:.2f}): trap charging / filaments")
        _add(ev, "coherent", -1, "hysteresis is not expected for coherent tunnelling")
    asy = diag["asymmetry"]
    if asy:
        rr = np.median([a["rectification"] for a in asy])
        if rr > 10 or rr < 0.1:
            _add(ev, "schottky", 1.5, f"strong rectification (I+/I− = {rr:.3g}): contact-limited injection")
            _add(ev, FIELD_DRIVEN + SCLC, -0.5, f"rectification {rr:.3g} points to the contacts")

    # --- admissibility from trap density and barrier thickness (Fig. 10) -----------------
    if dev.trap_density_cm3:
        dmin, dmax = float(ds.thicknesses.min()), float(ds.thicknesses.max())
        # traps met along a straight tunnelling path of cross-section ~1 nm²
        n_path = dev.trap_density_cm3 * 1e-21 * dmax
        r_t = (dev.trap_density_cm3 * 1e-21) ** (-1 / 3)          # mean trap spacing (nm)
        tag = f"N_t = {dev.trap_density_cm3:.2g} cm⁻³: ≈{n_path:.2g} traps per path, spacing {r_t:.2g} nm (Fig. 10)"
        if n_path < 0.1 and dmax < 4:
            _add(ev, ["coherent", "band_to_band"], 1.0, tag + " — clean gap, coherent tunnelling admissible")
            _add(ev, ["nn_hopping", "vrh", "polaron", "gdm"], -1.0, tag + " — too few states for hopping")
        elif n_path < 3:
            _add(ev, ["tat", "glazman_matveev"], 1.0, tag + " — a few localised states: TAT / chains admissible")
        else:
            _add(ev, ["nn_hopping", "vrh", "polaron", "sclc_trap", "sclc_exp", "poole_frenkel"], 1.0,
                 tag + " — dense localised states: hopping / trap-limited conduction admissible")
            if dmin > 4:
                _add(ev, "coherent", -1.0, tag + " — coherent tunnelling negligible")

    # --- material prior --------------------------------------------------------------
    if dev.material_class in MATERIAL_PRIORS:
        _add(ev, MATERIAL_PRIORS[dev.material_class], 0.5, f"commonly reported for {dev.material_class} barriers (Table 10)")
    return ev


def _verdict(a, best_key, top_prob, n_axes, rank):
    fails = [c for c in a.checks if c.level == "fail"]
    vetoes = [w for w, _ in a.evidence if w <= -4]
    if a.key == best_key:
        if n_axes < 2:
            return "LEADING CANDIDATE — not identifiable from bias alone"
        if top_prob >= 0.85:
            return "IDENTIFIED"
        if top_prob >= 0.5:
            return "MOST LIKELY"
        return "LEADING, BUT NOT UNIQUE"
    if fails or vetoes or a.d_aic > 50:
        return "EXCLUDED"
    if a.prob >= 0.08:
        return "CANNOT BE EXCLUDED"
    return "DISFAVOURED"


def analyze(ds, dev: Optional[DeviceInfo] = None, mechanisms=None, quick=False,
            progress: Optional[Callable] = None, combos=True, series=True, workers=None) -> Result:
    """Run the full protocol.  ``workers``: processes on this node (default: all CPUs available)."""
    with Pool(workers) as P:
        return _analyze(ds, dev, mechanisms, quick, progress, combos, series, P)


def _analyze(ds, dev, mechanisms, quick, progress, combos, series, P) -> Result:
    dev = dev or DeviceInfo()
    tm = {}
    say = progress or (lambda msg: None)
    t0 = time.time()
    say("Computing model-free fingerprints")
    diag = diagnostics.run(ds, dev)
    tm["diagnostics"] = time.time() - t0
    cov = diag["coverage"]
    data = FitData.from_frame(ds.df)
    keys = list(mechanisms or REGISTRY.keys())

    # hints from the fingerprints
    hints = {}
    if diag["arrhenius"]:
        a0 = diag["arrhenius"][0]
        Ea = max(a0["Ea_lowV"], 0.01)
        hints = {"Ea": Ea, "phit": a0.get("Ea0", Ea), "phi": a0.get("Ea0", Ea), "lam": 4 * Ea}
        if a0.get("eps_pf"):
            hints["eps_r"] = float(np.clip(a0["eps_pf"], 1.0, 50))

    t1 = time.time()
    tasks = []
    for k in keys:
        h = dict(hints)
        if k == "schottky" and diag["arrhenius"] and diag["arrhenius"][0].get("eps_schottky"):
            h["eps_r"] = float(np.clip(diag["arrhenius"][0]["eps_schottky"], 0.5, 50))
        tasks.append((k, data, dev, h, (1 if quick else None)))
    say(f"Global fits of {len(keys)} mechanisms on {P.workers} worker(s)")
    out = P.map(task_fit, tasks, cost=lambda t: cost_of(t[0]), progress=say,
                label=lambda t: f"global fit: {short_name(t[0])}")
    fits = dict(zip(keys, out))
    tm["fits"] = time.time() - t1
    tm["workers"] = P.workers

    aic_min = min(f.aic for f in fits.values())
    ev = evidence(diag, ds, dev, keys)
    assessments = []
    for k in keys:
        m = REGISTRY[k]
        f = fits[k]
        a = Assessment(key=k, name=m.name, family=m.family, fit=f, evidence=ev.get(k, []))
        a.checks = m.checks(f.params, f.errors, ds, dev, diag)
        a.d_aic = f.aic - aic_min
        assessments.append(a)

    # cross-validation of the statistically competitive models
    t2 = time.time()
    cand = sorted(assessments, key=lambda a: a.d_aic)[: (3 if quick else 5)]
    cand = [a for a in cand if a.d_aic < 200]
    cvout = P.map(task_cv, [(a.key, data, dev, a.fit, cov) for a in cand], cost=lambda t: cost_of(t[0]),
                  progress=say, label=lambda t: f"cross-validation: {short_name(t[0])}")
    for a, (cv, mode) in zip(cand, cvout):
        a.fit.cv_rms, a.fit.cv_mode = cv, mode
    tm["cross_validation"] = time.time() - t2
    cvs = [a.fit.cv_rms for a in assessments if a.fit.cv_rms is not None]
    cv_best = min(cvs) if cvs else None

    noise = diag.get("noise_lnI") or 0.05
    for a in assessments:
        s = -0.5 * min(a.d_aic, 80.0)
        s += sum(w for w, _ in a.evidence)
        s += sum(c.weight for c in a.checks)
        if a.fit.cv_rms is not None and cv_best:
            ratio = a.fit.cv_rms / max(cv_best, noise)
            s += -min(2.5 * np.log2(max(ratio, 1.0)), 10.0)
        elif cv_best is not None:
            s -= 5.0
        if not a.fit.success:
            s -= 20
        a.score = float(s)
    sc = np.array([a.score for a in assessments])
    pr = np.exp(sc - sc.max()); pr /= pr.sum()
    for a, p in zip(assessments, pr):
        a.prob = float(p)
    ranked = sorted(assessments, key=lambda a: -a.score)
    best = ranked[0]
    n_axes = 1 + int(cov["has_T_axis"]) + int(cov["has_d_axis"])
    for i, a in enumerate(ranked):
        a.verdict = _verdict(a, best.key, best.prob, n_axes, i)

    warnings = list(ds.notes)
    if best.fit.rms > max(4 * noise, 0.15) and best.verdict == "IDENTIFIED":
        best.verdict = "LEADING, BUT NOT UNIQUE"
    # goodness of fit relative to noise
    if best.fit.rms > max(4 * noise, 0.15):
        warnings.append(f"Even the best single mechanism leaves residuals of {100 * (np.exp(best.fit.rms) - 1):.0f} % "
                        f"(noise ≈ {100 * noise:.0f} %): several channels probably act together — see the mixed-channel analysis.")

    # ---- parallel channels ----------------------------------------------------
    combo_out, mixed = [], None
    t3 = time.time()
    poor = best.fit.rms > max(4 * noise, 0.15)
    if combos and not quick:
        by_aic = sorted(assessments, key=lambda a: a.d_aic)
        pool = []
        for a in ranked[:4] + by_aic[:4] + (by_aic[:9] if poor else []):
            if a not in pool:
                pool.append(a)
        pairs = []
        for i in range(len(pool)):
            for j in range(i + 1, len(pool)):
                A, B = pool[i], pool[j]
                if REGISTRY[A.key].expensive and REGISTRY[B.key].expensive:
                    continue
                if A.family == B.family == "metallic paths":
                    continue
                # put the expensive (or better) model first so its prefactor is profiled
                if REGISTRY[B.key].expensive or B.d_aic < A.d_aic:
                    A, B = B, A
                pairs.append((A, B))
        exp_ = lambda A, B: REGISTRY[A.key].expensive or REGISTRY[B.key].expensive
        say(f"Screening {len(pairs)} two-channel models on {P.workers} worker(s)")
        scr = P.map(task_combo, [(A.key, B.key, A.fit.params, B.fit.params, data, dev, True, None,
                                  1 if exp_(A, B) else 3, 2) for A, B in pairs],
                    cost=lambda t: cost_of(t[0] + "+" + t[1]), progress=say,
                    label=lambda t: f"screen {t[0]} + {t[1]}")
        screened = sorted([(r.aic, A, B, r) for (A, B), r in zip(pairs, scr)], key=lambda t: t[0])
        top2 = screened[:2]
        ref = P.map(task_combo, [(A.key, B.key, A.fit.params, B.fit.params, data, dev, False, r0.params,
                                  1 if exp_(A, B) else 4, 3) for _, A, B, r0 in top2],
                    cost=lambda t: cost_of(t[0] + "+" + t[1]), progress=say,
                    label=lambda t: f"refine {t[0]} + {t[1]}")
        for k, (_, A, B, r0) in enumerate(screened):
            pm = Parallel(REGISTRY[A.key], REGISTRY[B.key], A.fit.params, B.fit.params)
            r = ref[k] if k < len(ref) else r0
            la, lb = pm.channels(data.V, data.T, data.d, r.params, dev)
            fa = 1.0 / (1.0 + np.exp(lb - la))
            share_b = float(np.mean(fa < 0.9)); share_a = float(np.mean(fa > 0.1))
            combo_out.append({"key": pm.key, "a": A.key, "b": B.key, "name": pm.name, "fit": r,
                              "d_aic_vs_best": r.aic - best.fit.aic, "frac_a": fa,
                              "share_a": share_a, "share_b": share_b, "mech": pm})
        combo_out.sort(key=lambda c: c["fit"].aic)
        good = [c for c in combo_out if c["d_aic_vs_best"] < -10 and min(c["share_a"], c["share_b"]) > 0.1]
        if good:
            mixed = good[0]
            # physical plausibility of each channel's parameters
            pa, pb = mixed["mech"]._split(mixed["fit"].params)
            mixed["checks_a"] = REGISTRY[mixed["a"]].checks(pa, {}, ds, dev, diag)
            mixed["checks_b"] = REGISTRY[mixed["b"]].checks(pb, {}, ds, dev, diag)
    tm["combos"] = time.time() - t3

    # ---- series resistance ------------------------------------------------------
    series_out = None
    if series and not quick and best.key not in ("coherent", "tat"):
        say("Testing a series resistance (contact / lead) on the best model")
        area_typ = float(np.median(data.area))
        sm = WithSeriesR(REGISTRY[best.key], best.fit.params, area_typ)
        r = P.map(task_series, [(best.key, best.fit.params, area_typ, data, dev)])[0]
        Vj = sm.junction_voltage(data.V, data.T, data.d, r.params, dev)
        drop = float(np.max(1 - Vj / data.V))
        series_out = {"fit": r, "d_aic": r.aic - best.fit.aic, "Rs_ohm": 10 ** r.params["log10_Rs"],
                      "max_drop_fraction": drop, "significant": bool(r.aic - best.fit.aic < -10 and drop > 0.05),
                      "mech": sm}

    # ---- regimes of the best model ------------------------------------------------
    regimes = None
    reg = REGISTRY[best.key].regime(data.V, data.T, data.d, best.fit.params, dev)
    if reg is not None:
        labels, counts = np.unique(reg, return_counts=True)
        regimes = {"labels": reg, "summary": {str(l): int(c) for l, c in zip(labels, counts)}}

    # ---- recommendations ---------------------------------------------------------------
    rec = []
    runner = ranked[1] if len(ranked) > 1 else None
    if runner is not None and runner.prob > 0.05:
        key = frozenset([best.family, runner.family]) if best.family != runner.family else frozenset([best.family])
        disc = DISCRIMINATORS.get(key)
        if disc:
            rec.append(f"To separate {short_name(best.key)} from {short_name(runner.key)}: {disc}.")
    if not cov["has_d_axis_3"]:
        rec.append("Measure at least three barrier thicknesses: d scaling is often the most informative control "
                   "(tunnelling ln I ∝ −d, hopping I ∝ 1/d, SCLC J/d = f(V/d²), FN/PF via F = V/d).")
    if not cov["has_T_axis"]:
        rec.append("Measure I(V) at ≥ 4 temperatures spanning at least a factor of 3 (ideally 4–300 K) for the "
                   "Rowell test, Ea(V) and w(T).")
    elif cov["T_range"][0] > 100:
        rec.append("Extend the temperature range below 100 K: the Rowell ratio and w(T) are only decisive over a wide window.")
    if not ds.area_known:
        rec.append("Supply the junction area so that absolute magnitudes (WKB, Richardson, per-channel transmission) can be checked.")
    if best.key in ("schottky", "poole_frenkel") and dev.n_opt is None:
        rec.append("Supply the refractive index n: ε_r from the √F slope must match n² (decisive Schottky/PF test).")
    if best.key in SCLC:
        rec.append("Confirm with pulsed I(V) or impedance spectroscopy (transit time) and a contact-metal variation.")
    if best.key == "nn_hopping" and not diag["hysteresis"]:
        rec.append("Sweep-rate dependence / forward–backward sweeps distinguish electronic hopping from ionic conduction.")
    if diag["ndr"]:
        rec.append("NDR detected: compare voltage- vs current-sourced and pulsed measurements (resonant tunnelling, "
                   "Wannier–Stark or self-heating).")
    if diag["gap"]:
        rec.append("A zero-bias suppression is present at low T: gate maps and its T/H dependence test Coulomb blockade "
                   "or a zero-bias anomaly.")
    if best.key == "coherent":
        rec.append("Rowell's third criterion is the only discriminating one: G(300 K)/G(4 K) should be 1–1.3.")

    # ---- statements ---------------------------------------------------------------------
    m = REGISTRY[best.key]
    conf = best.verdict
    sub = ""
    if regimes:
        dom = max(regimes["summary"], key=regimes["summary"].get)
        sub = f" (dominant regime in the measured window: {dom})"
    headline = f"{m.name}{sub}"
    runner_txt = ""
    if runner is not None:
        runner_txt = (f" The runner-up, {short_name(runner.key)}, has ΔAIC = {runner.d_aic:.0f} "
                      f"and relative support {100 * runner.prob:.1f} %.")
    pos_ev = [t for w, t in best.evidence if w > 0][:4]
    statement = (f"Transport is assigned to {short_name(best.key)}{sub} — verdict: {conf} "
                 f"(relative support {100 * best.prob:.1f} %). Extracted parameters: "
                 f"{m.summary(best.fit.params, best.fit.errors)}. Global fit to {best.fit.n} points across "
                 f"{cov['n_T']} temperature(s) and {cov['n_d']} thickness(es): rms deviation "
                 f"{100 * (np.exp(best.fit.rms) - 1):.1f} % in current." + runner_txt)
    if pos_ev:
        statement += " Supporting fingerprints: " + "; ".join(pos_ev) + "."
    if mixed:
        statement += (f" A two-channel model ({mixed['name']}) improves the fit by ΔAIC = {-mixed['d_aic_vs_best']:.0f}; "
                      "both channels contribute in different (V, T) regions.")
    if series_out and series_out["significant"]:
        statement += (f" A series resistance R_s ≈ {series_out['Rs_ohm']:.3g} Ω absorbs up to "
                      f"{100 * series_out['max_drop_fraction']:.0f} % of the applied bias (Sec. XIX).")
    if mixed and (best.fit.rms > max(4 * noise, 0.15)) and mixed["fit"].rms < 0.6 * best.fit.rms:
        A, B = REGISTRY[mixed["a"]], REGISTRY[mixed["b"]]
        pa, pb = mixed["mech"]._split(mixed["fit"].params)
        ea = {k[2:]: v for k, v in mixed["fit"].errors.items() if k.startswith("a.")}
        eb = {k[2:]: v for k, v in mixed["fit"].errors.items() if k.startswith("b.")}
        fails = [c for c in mixed.get("checks_a", []) + mixed.get("checks_b", []) if c.level == "fail"]
        conf = "IDENTIFIED (two parallel channels)" if (n_axes >= 2 and not fails and
                                                        mixed["d_aic_vs_best"] < -50) else "MOST LIKELY (two parallel channels)"
        mixed["verdict"] = conf
        headline = f"{short_name(A.key)}  ∥  {short_name(B.key)}"
        fa = mixed["frac_a"]
        def where(mask):
            if not mask.any():
                return "nowhere in the measured window"
            Tm, dm, Vm = data.T[mask], data.d[mask] * 1e9, data.V[mask]
            rng = lambda a, u, f="g": (f"{a.min():{f}} {u}" if np.isclose(a.min(), a.max())
                                       else f"{a.min():{f}}–{a.max():{f}} {u}")
            return (f"T = {rng(Tm, 'K')}, d = {rng(dm, 'nm')}, V = {rng(Vm, 'V', '.2g')} "
                    f"({100 * mask.mean():.0f} % of the points)")
        dom_txt = (f"{short_name(A.key)} carries most of the current at {where(fa >= 0.5)}; "
                   f"{short_name(B.key)} at {where(fa < 0.5)}.")
        statement = (f"No single mechanism describes the data (best single model: {short_name(best.key)}, "
                     f"rms {100 * (np.exp(best.fit.rms) - 1):.0f} %). Two channels in parallel are required — verdict: "
                     f"{conf}; ΔAIC = {-mixed['d_aic_vs_best']:.0f} relative to the best single mechanism, rms "
                     f"{100 * (np.exp(mixed['fit'].rms) - 1):.1f} %. Channel A — {short_name(A.key)}: "
                     f"{A.summary(pa, ea)}. Channel B — {short_name(B.key)}: {B.summary(pb, eb)}. "
                     f"{dom_txt} See the channel map.")
        for c in fails:
            statement += f" Caution: {c.message}."
        conf_out = conf
        assigned = mixed["key"]
    else:
        conf_out = conf
        assigned = best.key
    conf = conf_out
    tm["total"] = time.time() - t0
    return Result(dataset=ds, device=dev, diag=diag, data=data, assessments=assessments, combos=combo_out,
                  series=series_out, best=best, confidence=conf, headline=headline, statement=statement,
                  recommendations=rec, warnings=warnings, regimes=regimes, timing=tm, mixed=mixed,
                  assigned_key=assigned)
