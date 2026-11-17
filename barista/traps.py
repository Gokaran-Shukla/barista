"""Trap density (and trap energy) from I–V data — whatever the transport mechanism.

Every route the data allow is used, and each estimate carries its assumptions and a 95 % interval
(bootstrap samples when the certainty analysis has run, otherwise the covariance errors):

=====================  =========================================================================
route                  estimate
=====================  =========================================================================
SCLC, discrete level   N_t and E_c − E_t from the exact drift–Poisson fit (review Eqs. 38–39)
SCLC, trap-filled lim. N_t = 2ε V_TFL / (e d²) from the γ-peak of each curve, model-free (Eq. 39),
                       with the V_TFL ∝ d² test across thicknesses
SCLC, exponential      total N_t and characteristic energy k T_c (Mark–Helfrich, Eq. 40)
trap-assisted tunnel.  N_t = C / (e ν₀) from the TAT prefactor, ν₀ = 10¹²–10¹⁴ s⁻¹; trap level Φ_t
NN / ionic hopping     site density N_t ≈ 3 / (4π r³) from the fitted hop length (sinh field law)
variable-range hopping density of localised states g(E_F) = β / (k_B T₀ ξ³) (β = 21.2, Mott 3D) and
                       the states inside the optimal hopping band, N ≈ g·ΔE_h(T)
polaron hopping        site density ≈ 3 / (4π a³)
Poole–Frenkel          trap depth Φ_t; bounds on N_t: traps must be isolated on the scale of the
                       field-lowering distance x_m (upper bound), and must supply the emitted carriers
                       (lower bound, needs μ)
coherent tunnelling /  no bulk traps needed; 95 % upper bound on N_t from a tunnelling ∥ TAT fit
Schottky / metallic
=====================  =========================================================================
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .constants import e, eps0, kB_eV
from .models import REGISTRY, short_name
from .models.base import LN10

NU0 = (1e12, 1e13, 1e14)          # attempt frequency range for trap-assisted tunnelling (s⁻¹)
TRAP_FREE = {"coherent", "schottky", "ohmic_metallic", "band_to_band"}


SHORT_ROUTE = [("SCLC fit", "drift–Poisson fit (single level)"),
               ("trap-filled limit", r"trap-filled limit $V_{TFL}$ (γ peak)"),
               ("SCLC, exponential", "Mark–Helfrich exponential tail"),
               ("trap-assisted", r"TAT prefactor $C/(e\nu_0)$"),
               ("site density from the hop length r", r"site density from hop length $r$"),
               ("site density from the hop length a", r"site density from hop length $a$"),
               ("variable-range", r"$g(E_F)\,\Delta E_h$ (Mott)"),
               ("Poole–Frenkel: traps isolated", "isolated-centre bound"),
               ("Poole–Frenkel prefactor", r"PF prefactor $C/(e\mu)$"),
               ("upper bound: TAT", "bound from a parallel TAT channel")]
ENERGY_TEX = {"E_c − E_t": r"$E_c-E_t$", "k_B T_c": r"$k_BT_c$", "Φ_t": r"$\Phi_t$", "trap depth": r"$\Phi_t$",
              "activation": r"$E_a$", "hop energy": r"$\Delta E_h$"}


@dataclass
class TrapEstimate:
    method: str
    mechanism: str
    Nt_cm3: float
    lo_cm3: float
    hi_cm3: float
    kind: str                  # 'estimate' | 'upper bound' | 'lower bound' | 'density of states'
    energy_eV: float = np.nan
    energy_label: str = ""
    weight: float = 0.0        # support of the mechanism this route belongs to
    assumptions: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def short(self):
        m = self.method
        for k, v in SHORT_ROUTE:
            if m.startswith(k):
                return v
        return m[:30]

    @property
    def energy_tex(self):
        for k, v in ENERGY_TEX.items():
            if self.energy_label.startswith(k):
                return v
        return ""


@dataclass
class TrapResult:
    estimates: list
    consensus: dict
    statement: str

    def table(self):
        return pd.DataFrame([{"method": t.method, "mechanism": short_name(t.mechanism), "type": t.kind,
                              "Nt_cm-3": t.Nt_cm3, "95%_low": t.lo_cm3, "95%_high": t.hi_cm3,
                              "trap_energy_eV": t.energy_eV, "energy_meaning": t.energy_label,
                              "mechanism_support": t.weight, "assumptions": t.assumptions,
                              **{f"x:{k}": v for k, v in t.extra.items()}} for t in self.estimates])


def _samples(res, key, pname, n=400):
    """Bootstrap samples of a parameter if available, else Gaussian from the covariance error."""
    cert = getattr(res, "certainty", None)
    if cert is not None and key in cert.boot_samples and pname in cert.boot_samples[key]:
        s = np.asarray(cert.boot_samples[key][pname], float)
        s = s[np.isfinite(s)]
        if len(s) >= 10:
            return s
    a = next((x for x in res.assessments if x.key == key), None)
    if a is None or pname not in a.fit.params:
        return None
    v = a.fit.params[pname]; se = a.fit.errors.get(pname, np.nan)
    if not np.isfinite(se) or se <= 0:
        se = 0.0
    return v + se * np.random.default_rng(0).standard_normal(n)


def _ci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x) & (x > 0)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    return float(np.median(x)), float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def analyse(res, nu0=None) -> TrapResult:
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return _analyse(res, nu0)


def _analyse(res, nu0=None) -> TrapResult:
    dev = res.device
    if nu0 is None:
        f = getattr(dev, "attempt_freq_Hz", None)
        nu0 = (f, f, f) if f else NU0
    ds = res.dataset
    eps = eps0 * dev.eps_static
    eps_note = "" if (dev.eps_r or dev.n_opt) else " (ε_r not supplied: 5 assumed — N_t scales with ε_r)"
    support = {a.key: a.prob for a in res.assessments}
    if res.mixed and res.assigned_key == res.mixed["key"]:
        for k in (res.mixed["a"], res.mixed["b"]):
            support[k] = max(support.get(k, 0), 0.5)
    fits = {a.key: a.fit for a in res.assessments}
    out = []

    # ---- SCLC with a discrete trap level --------------------------------------------
    if "sclc_trap" in fits:
        s = _samples(res, "sclc_trap", "log10_Nt"); se = _samples(res, "sclc_trap", "Et")
        if s is not None:
            m, lo, hi = _ci(10 ** s / 1e6)
            Et = float(np.median(se)) if se is not None else np.nan
            out.append(TrapEstimate("SCLC fit (exact drift–Poisson, single level)", "sclc_trap", m, lo, hi, "estimate",
                                    Et, "E_c − E_t", support.get("sclc_trap", 0),
                                    "ohmic injecting contact, single carrier, one trap level" + eps_note))
    # ---- model-free trap-filled limit from the γ peak --------------------------------------------
    peaks = [c for c in res.diag["gamma"] if c["peak"]]
    if peaks:
        vals = []
        for c in peaks:
            dm = c["d"] * 1e-9
            vals.append(2 * eps * c["V_gamma_max"] / (e * dm**2) / 1e6)
        m, lo, hi = _ci(vals)
        gp = res.diag.get("gamma_peak_scaling") or {}
        scal = gp.get("exponent")
        ok = scal is None or abs(scal - 2) < 0.5
        note = (f"V_TFL scales as d^{scal:.2f}" + (" ≈ d² ✓" if ok else " (not d²: peak may be a series element "
                                                     "or defect level, Sec. XIX)")) if scal is not None else \
            "single thickness: V_TFL ∝ d² not tested"
        w = support.get("sclc_trap", 0) + support.get("sclc_exp", 0)
        out.append(TrapEstimate("trap-filled limit from the γ = d lnI/d lnV peak (model-free, Eq. 39)", "sclc_trap",
                                m, min(vals), max(vals), "estimate", np.nan, "", w if ok else 0.3 * w,
                                note + eps_note, {"n_curves_with_peak": len(peaks)}))
    # ---- exponential trap distribution ---------------------------------------------------------
    if "sclc_exp" in fits:
        s = _samples(res, "sclc_exp", "log10_Nt"); tc = _samples(res, "sclc_exp", "Tc")
        if s is not None:
            m, lo, hi = _ci(10 ** s / 1e6)
            out.append(TrapEstimate("SCLC, exponential trap distribution (Mark–Helfrich)", "sclc_exp", m, lo, hi,
                                    "estimate", float(kB_eV * np.median(tc)) if tc is not None else np.nan,
                                    "k_B T_c (width of the trap tail)", support.get("sclc_exp", 0),
                                    "total density of an exponential tail below E_c" + eps_note))
    # ---- trap-assisted tunnelling -------------------------------------------------------------------
    if "tat" in fits:
        s = _samples(res, "tat", "log10_C")
        phit = fits["tat"].params.get("phit", np.nan)
        if s is None:
            s = np.array([fits["tat"].params["log10_C"]])
        Cs = 10 ** np.asarray(s, float)
        mid = Cs / (e * nu0[1]) / 1e6
        m, _, _ = _ci(mid)
        lo = float(np.percentile(Cs / (e * nu0[2]) / 1e6, 2.5)); hi = float(np.percentile(Cs / (e * nu0[0]) / 1e6, 97.5))
        out.append(TrapEstimate("trap-assisted tunnelling prefactor, N_t = C/(e ν₀)", "tat", m, lo, hi, "estimate",
                                phit, "Φ_t below the barrier band edge", support.get("tat", 0),
                                (f"ν₀ = {nu0[1]:.0e} s⁻¹ supplied" if nu0[0] == nu0[2] else
                                 "ν₀ = 10¹³ s⁻¹ (interval spans 10¹²–10¹⁴ s⁻¹)") + ", uniform traps, elastic two-step"
                                + ("" if ds.area_known else "; area unknown — N_t scales with the true area")))
    # ---- nearest-neighbour hopping / polaron: site density from the hop length ------------------
    for key, pn, lab in (("nn_hopping", "r", "hop length r"), ("polaron", "a", "hop length a")):
        if key in fits:
            s = _samples(res, key, pn)
            if s is not None:
                Ns = 3 / (4 * np.pi * (np.asarray(s) * 1e-7) ** 3)       # nm -> cm
                m, lo, hi = _ci(Ns)
                Ea = fits[key].params.get("Ea", fits[key].params.get("lam", np.nan) / 4 if key == "polaron" else np.nan)
                out.append(TrapEstimate(f"site density from the {lab} (sinh field law)", key, m, lo, hi, "estimate",
                                        Ea, "activation energy E_a", support.get(key, 0),
                                        "hops between nearest localised sites, N_t ≈ 3/(4π r³)"))
    # ---- variable-range hopping: density of states --------------------------------------------------
    if "vrh" in fits:
        p = fits["vrh"].params
        sT0 = _samples(res, "vrh", "log10_T0"); sxi = _samples(res, "vrh", "xi")
        if sT0 is not None and sxi is not None:
            L = min(len(sT0), len(sxi))
            T0 = 10 ** np.asarray(sT0[:L]); xi = np.asarray(sxi[:L]) * 1e-7        # cm
            g = 21.2 / (kB_eV * T0 * xi**3)                                         # cm⁻³ eV⁻¹
            Tm = float(np.median(ds.temperatures))
            dEh = p["p"] * kB_eV * Tm * (10 ** p["log10_T0"] / Tm) ** p["p"]
            m, lo, hi = _ci(g * dEh)
            gm, glo, ghi = _ci(g)
            out.append(TrapEstimate(f"variable-range hopping: states in the optimal band at {Tm:g} K", "vrh", m, lo, hi,
                                    "estimate", dEh, "hop energy ΔE_h", support.get("vrh", 0),
                                    "Mott 3D, g(E_F) = 21.2/(k_B T₀ ξ³); ξ from the field dependence",
                                    {"g(E_F)_cm-3eV-1": gm, "g_95%_low": glo, "g_95%_high": ghi}))
    # ---- Poole–Frenkel: depth and bounds ------------------------------------------------------------------
    if "poole_frenkel" in fits:
        p = fits["poole_frenkel"].params
        Fmax = float(np.max(res.data.V / res.data.d))
        xm = np.sqrt(e / (16 * np.pi * eps0 * p["eps_r"] * Fmax))            # distance of the lowered maximum
        Nmax = 1 / (8 * (xm * 1e2) ** 3)
        out.append(TrapEstimate("Poole–Frenkel: traps isolated on the field-lowering scale (upper bound)",
                                "poole_frenkel", Nmax, np.nan, Nmax, "upper bound", p["phit"], "trap depth Φ_t",
                                support.get("poole_frenkel", 0),
                                f"N_t < 1/(2x_m)³ with x_m = {xm * 1e9:.2f} nm at the highest field"))
        if dev.mobility_cm2Vs:
            sC = _samples(res, "poole_frenkel", "log10_C")
            Neff = 10 ** np.asarray(sC) / (e * dev.mobility_cm2Vs * 1e-4) / 1e6
            m, lo, hi = _ci(Neff)
            out.append(TrapEstimate("Poole–Frenkel prefactor: emitting-centre density C/(e μ)", "poole_frenkel",
                                    m, lo / 10, hi * 10, "estimate", p["phit"], "trap depth Φ_t",
                                    support.get("poole_frenkel", 0),
                                    f"J = e μ N_t F e^(−(Φ_t − β√F)/kT), occupied centres; μ = {dev.mobility_cm2Vs} "
                                    "cm²/Vs supplied; ±1 decade for the occupancy model"))
    # ---- trap-free mechanisms: upper bound from tunnelling ∥ TAT ----------------------------------------
    assigned = (res.assigned_key or res.best.key).split("+")
    if set(assigned) & TRAP_FREE and "tat" not in assigned:
        ub = _tat_upper_bound(res)
        if ub is not None:
            out.append(TrapEstimate("upper bound: TAT channel added to the assigned trap-free model", assigned[0],
                                    ub, np.nan, ub, "upper bound", np.nan, "", support.get(assigned[0], 0),
                                    "95 % bound on a parallel trap-assisted channel, ν₀ = 10¹⁴ s⁻¹ (conservative)"))
    cons, st = _consensus(out, assigned, res, dev)
    if dev.trap_density_cm3:
        u = dev.trap_density_cm3
        lo, hi = cons.get("lo_cm3", np.nan), cons.get("hi_cm3", np.nan)
        if np.isfinite(cons.get("Nt_cm3", np.nan)):
            ok = (not np.isfinite(lo) or u >= lo / 3) and (not np.isfinite(hi) or u <= hi * 3)
            st += (f" The supplied N_t = {u:.2g} cm⁻³ is {'consistent' if ok else 'NOT consistent'} with this.")
            cons["user_Nt_consistent"] = bool(ok)
        elif "upper_bound_cm3" in cons:
            ok = u <= cons["upper_bound_cm3"]
            st += (f" The supplied N_t = {u:.2g} cm⁻³ is {'below' if ok else 'ABOVE'} this bound"
                   + ("." if ok else " — those traps would have produced a measurable TAT current."))
            cons["user_Nt_consistent"] = bool(ok)
    return TrapResult(out, cons, st)


def _tat_upper_bound(res):
    """95 % upper bound on a trap-assisted channel added in parallel to the assigned model.

    The assigned fit is kept, a TAT channel (shape from the TAT fit, or Φ_t = Φ/2 when TAT was not fitted)
    is added with prefactor C, the overall prefactor is re-profiled, and C is raised until the likelihood
    ratio exceeds χ²₁(95 %) = 3.84.  N_t < C_max/(e ν₀) with the largest ν₀ (most conservative)."""
    from .fitting import predict_lnI
    key = (res.assigned_key or res.best.key).split("+")[0]
    if key not in REGISTRY or "tat" not in REGISTRY:
        return None
    a = next((x for x in res.assessments if x.key == key), None)
    if a is None:
        return None
    D = res.data
    tat = next((x for x in res.assessments if x.key == "tat"), None)
    pt = dict(tat.fit.params) if tat is not None else {p.name: p.init for p in REGISTRY["tat"].params}
    try:
        lnA = predict_lnI(REGISTRY[key], a.fit, D.V, D.T, D.d, D.area, res.device)
        pt["log10_C"] = 0.0
        lnB = REGISTRY["tat"].lnJ(D.V, D.T, D.d, pt, res.device) + np.log(D.area)    # C = 1 A m⁻²
    except Exception:
        return None
    w = D.w
    def rss(lnmod):
        r = D.lnI - lnmod
        c = np.sum(w**2 * r) / np.sum(w**2)          # re-profile the common prefactor
        return float(np.sum((w * (r - c))**2))
    n = len(D.lnI)
    r0 = rss(lnA)
    best = None
    for lc in np.arange(-40, 40.01, 0.25):
        r1 = rss(np.logaddexp(lnA, lnB + lc * LN10))
        if n * np.log(max(r1, 1e-300) / max(r0, 1e-300)) < 3.84:
            best = lc
        elif best is not None:
            break
    if best is None:
        return None
    return float(10 ** best / (e * NU0[2]) / 1e6)


def _consensus(est, assigned, res, dev):
    rel = [t for t in est if t.kind in ("estimate",) and t.mechanism in assigned and np.isfinite(t.Nt_cm3)]
    if not rel:
        rel = [t for t in est if t.kind == "estimate" and t.weight > 0.05 and np.isfinite(t.Nt_cm3)]
    ub = [t for t in est if t.kind == "upper bound" and t.mechanism in assigned]
    if rel:
        w = np.array([max(t.weight, 1e-3) for t in rel])
        lg = np.log10([t.Nt_cm3 for t in rel])
        mean = float(np.sum(w * lg) / np.sum(w))
        los = [t.lo_cm3 for t in rel if np.isfinite(t.lo_cm3)]
        his = [t.hi_cm3 for t in rel if np.isfinite(t.hi_cm3)]
        lo = min(los) if los else np.nan; hi = max(his) if his else np.nan
        prim = max(rel, key=lambda t: t.weight)
        en = f" at {prim.energy_label} = {prim.energy_eV:.3g} eV" if np.isfinite(prim.energy_eV) else ""
        spread = float(np.ptp(lg)) if len(lg) > 1 else 0.0
        st = (f"Trap density N_t ≈ {10 ** mean:.2g} cm⁻³ (95 % range {lo:.2g}–{hi:.2g} cm⁻³){en}, from "
              f"{len(rel)} route(s) consistent with {short_name(assigned[0])}"
              + (f"; the routes agree within {spread:.1f} decade(s)" if len(rel) > 1 else "")
              + f". Primary route: {prim.method}.")
        if not (dev.eps_r or dev.n_opt) and any("SCLC" in t.method or "γ" in t.method for t in rel):
            st += " Supply ε_r for an absolute value (SCLC routes scale with ε_r)."
        return {"Nt_cm3": 10 ** mean, "lo_cm3": lo, "hi_cm3": hi, "routes": len(rel), "spread_decades": spread,
                "primary": prim.method, "trap_energy_eV": prim.energy_eV, "energy_meaning": prim.energy_label}, st
    if ub and assigned[0] == "poole_frenkel":
        u = min(t.Nt_cm3 for t in ub)
        pf = ub[0]
        return {"Nt_cm3": np.nan, "upper_bound_cm3": u, "trap_energy_eV": pf.energy_eV,
                "energy_meaning": pf.energy_label}, (
            f"Poole–Frenkel emission fixes the trap depth Φ_t = {pf.energy_eV:.3g} eV; the trap density itself is not "
            f"identifiable from I–V alone without the carrier mobility. Bound: N_t < {u:.2g} cm⁻³ (Coulomb centres "
            "must be isolated on the field-lowering scale). Supply the mobility (--mobility) for an absolute N_t.")
    if ub:
        u = min(t.Nt_cm3 for t in ub)
        return {"Nt_cm3": np.nan, "upper_bound_cm3": u}, (
            f"The assigned mechanism ({short_name(assigned[0])}) needs no bulk traps; the data bound the density of "
            f"electrically active traps to N_t < {u:.2g} cm⁻³ (95 %).")
    return {"Nt_cm3": np.nan}, (f"The assigned mechanism ({short_name(assigned[0])}) does not involve bulk traps and "
                                "the data contain no trap-related signature; no trap density can be extracted.")
