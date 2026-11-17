"""Spin-dependent transport (review Sec. XV) and interface (proximity) polarisation.

Input: I(V, T, d|N) measured in two magnetic configurations (P/AP, FM/AFM, Néel±, ...),
or a magnetoresistance table MR(V, T, N, d, θ).  MR = G_P/G_AP − 1 at equal bias.

Models (all fitted globally to MR(V, T, N) and ranked by AIC, with bootstrap intervals):

* ``julliere``     — Jullière/Slonczewski spin valve: MR = 2P²/(1−P²), magnon softening
                      P(T) = P0(1 − αT^{3/2}), bias roll-off P² → P²/(1+(V/V½)²); N-independent.
* ``spin_filter``  — coherent spin filtering by a layered (A-type / altermagnetic) barrier:
                      1+MR = cosh(N aΔκ) (even N), cosh(N aΔκ)/cosh(aΔκ) (odd N), review Eq. (44);
                      aΔκ(T) = aΔκ0 m(T), m = (1−(T/T_N)²)^{0.35}; bias roll-off V½.
* ``spin_hopping`` — spin-dependent hopping (thick CrSBr): MR = exp(ΔEa/kT) − 1, ΔEa(V) = ΔEa0/(1+(V/V½)²).
* ``proximity``    — interface polarisation from the exchange field induced in Dirac electrodes
                      (graphene/CrSb/graphene).  Each graphene electrode has spin-split Dirac
                      cones, E_D,σ = −μ − σΔ(T)/2 with broadening Γ; the tunnelling current per spin
                      is I_σ ∝ e^{−σ N aΔκ} ∫ D^b_σ(E−eV/2) D^t_σ(E+eV/2)[f(E−eV/2)−f(E+eV/2)] dE
                      (k∥ not conserved).  P: Δ_b = Δ_t;  AP: Δ_b = −Δ_t.  Signatures: MR that changes
                      sign with bias near the Dirac point, follows m_int(T), contact term ∝ P_t P_b
                      independent of N, plus a bulk filter term ∝ N.
* ``valet_fert``   — Valet–Fert / Fert–Jaffrès spin drift–diffusion for CPP F/I/N/I/F stacks, solved
                      exactly (spin accumulation, spin-current continuity, interface spin asymmetry γ,
                      bulk asymmetry β, area resistance r_b*, spin-diffusion length l_sf^N): MR vs
                      spacer thickness, including the conductivity-mismatch window.

Fingerprints: N-scaling (exponential / parity / N-independent / decaying with d), sign change
of MR with bias, temperature law (Bloch T^{3/2}, Arrhenius, vanishing at T_N), and the charge
mechanism of the P state (tunnelling vs activated).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import expit

from .constants import kB_eV

P_NAMES = {"p", "parallel", "fm", "ferro", "ferromagnetic", "sat", "saturated", "high", "highfield", "on", "+",
           "aligned", "up"}
AP_NAMES = {"ap", "antiparallel", "anti-parallel", "afm", "antiferro", "antiferromagnetic", "zero", "zerofield",
            "off", "-", "down"}


# ---------------------------------------------------------------------------
# data handling
# ---------------------------------------------------------------------------
def _norm_state(s):
    t = str(s).strip().lower().replace(" ", "").replace("_", "")
    if t in P_NAMES:
        return "P"
    if t in AP_NAMES:
        return "AP"
    return str(s)


def mr_table(df, p_state=None, ap_state=None, layer_spacing_nm=None, n_layers=None):
    """Long MR table from I(V) in two configurations, or from an MR column.

    df columns: V, I or J, T, d (nm) and/or N, state | MR.
    Returns DataFrame with V, T, N, d, MR (fraction)."""
    df = df.copy()
    cols = {c.lower(): c for c in df.columns}
    if "mr" in cols or "tmr" in cols:
        c = cols.get("mr", cols.get("tmr"))
        mr = pd.to_numeric(df[c], errors="coerce")
        if np.nanmax(np.abs(mr)) > 5 or "%" in c:
            mr = mr / 100.0
        out = pd.DataFrame({"V": df.get(cols.get("v", "V"), 0.0), "T": df.get(cols.get("t", "T"), 300.0), "MR": mr})
    else:
        st = df["state"].map(_norm_state) if "state" in df else None
        if st is None:
            raise ValueError("spin analysis needs a 'state' column (P/AP, FM/AFM, …) or an MR column")
        df["state"] = st
        states = list(dict.fromkeys(st))
        P = p_state or ("P" if "P" in states else states[0])
        AP = ap_state or ("AP" if "AP" in states else [s for s in states if s != P][0])
        rows = []
        keys = [k for k in ("T", "d", "N", "gate", "angle") if k in df.columns]
        for kv, g in df.groupby(keys):
            kv = kv if isinstance(kv, tuple) else (kv,)
            gp = g[(g.state == P) & (g.V > 0) & (g.I > 0)].sort_values("V")
            ga = g[(g.state == AP) & (g.V > 0) & (g.I > 0)].sort_values("V")
            if len(gp) < 2 or len(ga) < 2:
                continue
            lo, hi = max(gp.V.min(), ga.V.min()), min(gp.V.max(), ga.V.max())
            Vg = np.unique(np.concatenate([gp.V[(gp.V >= lo) & (gp.V <= hi)], ga.V[(ga.V >= lo) & (ga.V <= hi)]]))
            Ip = np.exp(np.interp(np.log(Vg), np.log(gp.V), np.log(gp.I)))
            Ia = np.exp(np.interp(np.log(Vg), np.log(ga.V), np.log(ga.I)))
            r = dict(zip(keys, kv))
            for v, a, b in zip(Vg, Ip, Ia):
                rows.append({**r, "V": v, "MR": a / b - 1.0, "I_P": a, "I_AP": b})
        out = pd.DataFrame(rows)
        if out.empty:
            raise ValueError("no overlapping P/AP curves")
    if "N" not in out.columns:
        if "d" in out.columns and layer_spacing_nm:
            out["N"] = np.round(out["d"] / layer_spacing_nm).astype(int)
        elif n_layers:
            out["N"] = int(n_layers)
        else:
            out["N"] = 1
    if "d" not in out.columns:
        out["d"] = out["N"] * (layer_spacing_nm or 1.0)
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# physics
# ---------------------------------------------------------------------------
def magnetisation(T, TN, beta=0.35):
    x = np.clip(np.asarray(T, float) / max(TN, 1e-9), 0, 1)
    return np.power(np.clip(1 - x**2, 0, 1), beta)


def filter_mr(N, adk):
    N = np.asarray(N, float)
    adk = np.asarray(adk, float)
    even = (np.round(N) % 2 == 0)
    val = np.where(even, np.cosh(N * adk), np.cosh(N * adk) / np.cosh(adk))
    return val - 1.0


def proximity_currents(V, T, N, p, n_e=601):
    """Spin-resolved tunnelling between two proximitised graphene electrodes.
    p: Delta (meV), mu (meV), Gamma (meV), adk (per-layer spin asymmetry of the barrier), TN (K)."""
    V = np.atleast_1d(np.asarray(V, float)); T = np.broadcast_to(np.asarray(T, float), V.shape)
    N = np.broadcast_to(np.asarray(N, float), V.shape)
    D0 = p["Delta"] * 1e-3 * magnetisation(T, p["TN"])
    mu, G = p["mu"] * 1e-3, max(p["Gamma"], 0.05) * 1e-3
    out = {}
    kt = np.maximum(kB_eV * T, 1e-5)[:, None]
    w = (np.abs(V) / 2)[:, None] + 12 * kt
    u = np.linspace(-1, 1, n_e)[None, :]
    E = u * w
    v = V[:, None]; dd = D0[:, None]
    fw = expit(-(E - v / 2) / kt) - expit(-(E + v / 2) / kt)
    for conf, sb in (("P", 1.0), ("AP", -1.0)):
        I = np.zeros_like(V)
        for sig in (1.0, -1.0):
            Dt = np.sqrt((E + v / 2 + mu + sig * dd / 2) ** 2 + G**2)
            Db = np.sqrt((E - v / 2 + mu + sig * sb * dd / 2) ** 2 + G**2)
            Ib = np.trapezoid(Dt * Db * fw, E, axis=1)
            I += Ib * np.exp(-sig * N * p["adk"])
        out[conf] = I
    return out["P"], out["AP"]


def interface_polarisation(E_meV, Delta_meV, mu_meV, Gamma_meV):
    E = np.asarray(E_meV, float)
    Du = np.sqrt((E + mu_meV + Delta_meV / 2) ** 2 + Gamma_meV**2)
    Dd = np.sqrt((E + mu_meV - Delta_meV / 2) ** 2 + Gamma_meV**2)
    return (Du - Dd) / (Du + Dd)


def valet_fert(layers, interfaces, j=1.0):
    """Exact Valet–Fert solution for a CPP stack between two reservoirs (Δμ = 0 at both ends).

    layers: list of dicts {t (m), rho (Ω m, ρ*), beta, lsf (m)}
    interfaces: list (len(layers)-1) of dicts {rb (Ω m², r_b*), gamma}
    Returns the area resistance A·R (Ω m²)."""
    n = len(layers)
    # unknowns: A_i, B_i (Δμ = A e^{x/l} + B e^{-x/l}, x local from 0..t)
    M = np.zeros((2 * n, 2 * n)); rhs = np.zeros(2 * n)
    row = 0
    # left reservoir: Δμ(0) = 0 in layer 0
    M[row, 0] = 1; M[row, 1] = 1; row += 1
    for k in range(n - 1):
        L, R = layers[k], layers[k + 1]
        tl, ll = L["t"], L["lsf"]
        eL, emL = np.exp(tl / ll), np.exp(-tl / ll)
        # js = beta j - (1/(2 rho)) dΔμ/dx
        # continuity of js: betaL j - (A eL - B emL)/(2 rhoL lL) = betaR j - (A' - B')/(2 rhoR lR)
        M[row, 2 * k] = -eL / (2 * L["rho"] * ll); M[row, 2 * k + 1] = emL / (2 * L["rho"] * ll)
        M[row, 2 * k + 2] = 1 / (2 * R["rho"] * R["lsf"]); M[row, 2 * k + 3] = -1 / (2 * R["rho"] * R["lsf"])
        rhs[row] = (R["beta"] - L["beta"]) * j
        row += 1
        # jump: Δμ_R(0) − Δμ_L(t) = 2 rb γ j − 2 rb js
        rb, g = interfaces[k]["rb"], interfaces[k]["gamma"]
        # js at interface (use right side expression): betaR j − (A' − B')/(2 rhoR lR)
        M[row, 2 * k + 2] = 1 - 2 * rb / (2 * R["rho"] * R["lsf"])
        M[row, 2 * k + 3] = 1 + 2 * rb / (2 * R["rho"] * R["lsf"])
        M[row, 2 * k] = -eL; M[row, 2 * k + 1] = -emL
        rhs[row] = 2 * rb * g * j - 2 * rb * R["beta"] * j
        row += 1
    last = layers[-1]
    M[row, 2 * n - 2] = np.exp(last["t"] / last["lsf"]); M[row, 2 * n - 1] = np.exp(-last["t"] / last["lsf"])
    sol = np.linalg.solve(M, rhs)
    # V = ∫ ρ↑ j↑ dx + Σ r↑ j↑ ,  ρ↑ = 2ρ(1−β), j↑ = (j + js)/2,  r↑ = 2 rb (1−γ)
    V = 0.0
    for k, Lr in enumerate(layers):
        A, B = sol[2 * k], sol[2 * k + 1]
        t, l, rho, b = Lr["t"], Lr["lsf"], Lr["rho"], Lr["beta"]
        # ∫ js dx = b j t − (Δμ(t) − Δμ(0))/(2ρ)
        dmu = (A * np.exp(t / l) + B * np.exp(-t / l)) - (A + B)
        int_js = b * j * t - dmu / (2 * rho)
        V += 2 * rho * (1 - b) * 0.5 * (j * t + int_js)
    for k, itf in enumerate(interfaces):
        R = layers[k + 1]
        A, B = sol[2 * k + 2], sol[2 * k + 3]
        js = R["beta"] * j - (A - B) / (2 * R["rho"] * R["lsf"])
        V += 2 * itf["rb"] * (1 - itf["gamma"]) * 0.5 * (j + js)
    return V / j


def fert_mr(dN_nm, p, fixed):
    """MR of F/I/N/I/F vs spacer thickness (Fert–Jaffrès geometry, exact Valet–Fert)."""
    out = []
    for dn in np.atleast_1d(dN_nm):
        def stack(sign):
            F = dict(t=5 * fixed["lsfF"] * 1e-9, rho=fixed["rhoF"] * 1e-8, beta=p["beta"], lsf=fixed["lsfF"] * 1e-9)
            F2 = dict(F, beta=sign * p["beta"])
            Nl = dict(t=dn * 1e-9, rho=fixed["rhoN"] * 1e-8, beta=0.0, lsf=p["lsfN"] * 1e-9)
            rb = 10 ** p["log10_rb"] * 1e-12
            return [F, Nl, F2], [dict(rb=rb, gamma=p["gamma"]), dict(rb=rb, gamma=sign * p["gamma"])]
        Rp = valet_fert(*stack(1.0)); Rap = valet_fert(*stack(-1.0))
        out.append(Rap / Rp - 1.0)
    return np.array(out)


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
@dataclass
class SpinModel:
    key: str
    name: str
    params: list      # (name, lo, hi, init, unit, desc)

    def predict(self, D, p, fixed):
        raise NotImplementedError


class Julliere(SpinModel):
    def __init__(self):
        super().__init__("julliere", "Jullière/Slonczewski spin valve (magnon P(T), bias roll-off)",
                         [("P0", 0.0, 0.999, 0.4, "", "spin polarisation at T = 0"),
                          ("alpha", 0.0, 2e-3, 2e-5, "K^-3/2", "Bloch coefficient"),
                          ("Vh", 0.01, 10.0, 0.5, "V", "bias for half the zero-bias TMR")])

    def predict(self, D, p, fixed):
        P = p["P0"] * np.clip(1 - p["alpha"] * D["T"] ** 1.5, 0, 1)
        P2 = P**2 / (1 + (D["V"] / p["Vh"]) ** 2)
        return 2 * P2 / (1 - P2)


class SpinFilter(SpinModel):
    def __init__(self):
        super().__init__("spin_filter", "Coherent spin filter, layered barrier (1+MR = cosh(N aΔκ), parity law)",
                         [("adk0", 0.0, 4.0, 0.5, "", "per-layer spin asymmetry aΔκ at T = 0"),
                          ("TN", 5.0, 1500.0, 300.0, "K", "ordering temperature"),
                          ("Vh", 0.01, 10.0, 1.0, "V", "bias roll-off")])

    def predict(self, D, p, fixed):
        adk = p["adk0"] * magnetisation(D["T"], p["TN"]) / (1 + (D["V"] / p["Vh"]) ** 2)
        return filter_mr(D["N"], adk)


class SpinHopping(SpinModel):
    def __init__(self):
        super().__init__("spin_hopping", "Spin-dependent hopping (MR = e^{ΔEa/kT} − 1)",
                         [("dEa", -0.2, 0.2, 0.01, "eV", "Ea(AP) − Ea(P)"),
                          ("Vh", 0.01, 20.0, 2.0, "V", "bias roll-off of ΔEa")])

    def predict(self, D, p, fixed):
        dE = p["dEa"] / (1 + (D["V"] / p["Vh"]) ** 2)
        return np.expm1(dE / (kB_eV * D["T"]))


class Proximity(SpinModel):
    def __init__(self):
        super().__init__("proximity", "Interface (proximity-exchange) polarisation of Dirac electrodes + barrier filter",
                         [("Delta", 0.1, 200.0, 10.0, "meV", "proximity exchange splitting Δ(0)"),
                          ("mu", -300.0, 300.0, 30.0, "meV", "electrode Fermi level above the Dirac point"),
                          ("Gamma", 0.1, 100.0, 5.0, "meV", "disorder broadening"),
                          ("adk", -2.0, 2.0, 0.05, "", "barrier spin asymmetry per layer (altermagnet, k-selected)"),
                          ("TN", 5.0, 1500.0, 700.0, "K", "ordering temperature of the interface layer")])

    def predict(self, D, p, fixed):
        Ip, Ia = proximity_currents(D["V"], D["T"], D["N"], p, n_e=301)
        return Ip / np.maximum(Ia, 1e-300) - 1.0


class ValetFert(SpinModel):
    def __init__(self):
        super().__init__("valet_fert", "Valet–Fert / Fert–Jaffrès spin drift–diffusion (F/I/N/I/F)",
                         [("gamma", -0.99, 0.99, 0.5, "", "interface spin asymmetry γ"),
                          ("beta", -0.99, 0.99, 0.4, "", "bulk spin asymmetry β of F"),
                          ("log10_rb", -6.0, 6.0, -1.0, "Ω µm²", "interface area resistance r_b* (log10)"),
                          ("lsfN", 0.5, 1e5, 100.0, "nm", "spin-diffusion length of the spacer")])

    def predict(self, D, p, fixed):
        return fert_mr(D["d"], p, fixed)


MODELS = {m.key: m for m in (Julliere(), SpinFilter(), SpinHopping(), Proximity(), ValetFert())}


@dataclass
class SpinFit:
    key: str
    name: str
    params: dict
    errors: dict
    ci95: dict
    rss: float
    aic: float
    n: int
    k: int
    pred: np.ndarray


@dataclass
class SpinResult:
    table: pd.DataFrame
    fits: dict
    ranking: list
    best: str
    probs: dict
    fingerprints: dict
    statement: str
    fixed: dict = field(default_factory=dict)


def _fit_model(m, D, y, w, fixed, inits=None, n_starts=6, seed=0):
    names = [q[0] for q in m.params]
    lo = np.array([q[1] for q in m.params]); hi = np.array([q[2] for q in m.params])
    rng = np.random.default_rng(seed)
    starts = [np.array([(inits or {}).get(q[0], q[3]) for q in m.params])]
    for _ in range(n_starts):
        starts.append(rng.uniform(lo, hi))

    def res(x):
        p = dict(zip(names, x))
        r = (m.predict(D, p, fixed) - y) * w
        return np.where(np.isfinite(r), r, 1e3)
    best = None
    for s0 in starts:
        try:
            sol = least_squares(res, np.clip(s0, lo, hi), bounds=(lo, hi), x_scale="jac", max_nfev=400)
        except Exception:
            continue
        if best is None or sol.cost < best.cost:
            best = sol
    p = dict(zip(names, best.x))
    r = res(best.x)
    n = len(y); k = len(names)
    rss = float(np.sum(r**2))
    J = best.jac
    try:
        cov = np.linalg.pinv(J.T @ J) * rss / max(n - k, 1)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except Exception:
        se = np.full(k, np.nan)
    aic = n * np.log(max(rss, 1e-300) / n) + 2 * k
    return SpinFit(m.key, m.name, p, dict(zip(names, se)), {}, rss, float(aic), n, k, m.predict(D, p, fixed))


def fingerprints(tab):
    fp = {}
    Ns = np.unique(tab.N)
    if len(Ns) >= 2:
        g = tab.groupby("N").apply(lambda q: np.median(np.log1p(np.clip(q.MR, -0.99, None))), include_groups=False)
        lnm = g.values
        if len(Ns) >= 3:
            sl = np.polyfit(Ns, lnm, 1)[0]
            resid = lnm - np.polyval(np.polyfit(Ns, lnm, 1), Ns)
            parity = np.std([r for n_, r in zip(Ns, resid) if n_ % 2 == 0] or [0]) < np.std(resid) and \
                np.sign(np.mean([r for n_, r in zip(Ns, resid) if n_ % 2 == 1] or [0])) != \
                np.sign(np.mean([r for n_, r in zip(Ns, resid) if n_ % 2 == 0] or [0]))
        else:
            sl = (lnm[1] - lnm[0]) / (Ns[1] - Ns[0]); parity = False
        rel = np.ptp(lnm) / max(np.mean(np.abs(lnm)), 1e-9)
        fp["N_scaling"] = {"N": Ns.tolist(), "ln(1+MR)": lnm.tolist(), "slope_per_layer": float(sl),
                           "class": ("grows with N (bulk spin filter)" if sl > 0.05 and rel > 0.3 else
                                     "decays with thickness (spin diffusion / Valet–Fert)" if sl < -0.02 and rel > 0.3 else
                                     "≈ independent of N (interface/contact term)"),
                           "parity_effect": bool(parity)}
    signs = []
    for _, q in tab.groupby(["T", "N"]):
        q = q.sort_values("V")
        if len(q) > 3:
            s = np.sign(q.MR.values)
            signs.append(bool(np.any(s[1:] * s[:-1] < 0)))
    fp["sign_change_with_bias"] = bool(np.mean(signs) > 0.3) if signs else False
    Ts = np.unique(tab["T"])
    if len(Ts) >= 3:
        lo = tab[tab.V <= tab.V.quantile(0.3)].groupby("T").MR.median()
        x = 1 / lo.index.values; y = np.log1p(np.clip(lo.values, 1e-9, None))
        ok = lo.values > 0
        if ok.sum() >= 3:
            c = np.polyfit(x[ok], np.log(y[ok]), 1) if np.all(y[ok] > 0) else [np.nan]
            fp["T_dependence"] = {"T": lo.index.tolist(), "MR_lowbias": lo.values.tolist(),
                                  "MR_increases_on_cooling": bool(lo.values[0] > lo.values[-1])}
    return fp


def analyse_spin(table, fixed=None, models=None, B=100, seed=0, charge_result=None):
    """Fit all spin models to an MR table and rank them."""
    fixed = {"rhoN": 10.0, "rhoF": 20.0, "lsfF": 10.0, **(fixed or {})}   # µΩ cm, nm
    tab = table.dropna(subset=["MR"]).copy()
    D = {"V": tab.V.values.astype(float), "T": tab["T"].values.astype(float), "N": tab.N.values.astype(float),
         "d": tab.d.values.astype(float)}
    y = tab.MR.values.astype(float)
    sig = 0.03 * np.abs(y) + 0.01 * np.max(np.abs(y)) + 1e-4
    w = 1 / sig
    keys = models or list(MODELS)
    if len(np.unique(D["d"])) < 2 and "valet_fert" in keys:
        keys = [k for k in keys if k != "valet_fert"]       # needs a spacer-thickness series
    fits = {k: _fit_model(MODELS[k], D, y, w, fixed) for k in keys}
    # bootstrap (residual resampling around each model's fit) for 95 % intervals of the best models
    aics = {k: f.aic for k, f in fits.items()}
    amin = min(aics.values())
    wts = {k: np.exp(-0.5 * (a - amin)) for k, a in aics.items()}
    tot = sum(wts.values())
    probs = {k: v / tot for k, v in wts.items()}
    fp = fingerprints(tab)
    # fingerprint adjustments
    score = {k: np.log(max(probs[k], 1e-300)) for k in keys}
    ns = fp.get("N_scaling", {}).get("class", "")
    if ns.startswith("grows"):
        for k in ("spin_filter", "proximity"):
            if k in score:
                score[k] += 2
        for k in ("julliere", "spin_hopping"):
            if k in score:
                score[k] -= 2
    elif ns.startswith("decays") and "valet_fert" in score:
        score["valet_fert"] += 3
    elif ns.startswith("≈"):
        for k in ("julliere", "proximity", "spin_hopping"):
            if k in score:
                score[k] += 1
        if "spin_filter" in score:
            score["spin_filter"] -= 2
    if fp.get("sign_change_with_bias") and "proximity" in score:
        score["proximity"] += 3
    if charge_result is not None:
        fam = charge_result.best.family
        if fam in ("hopping", "emission", "space-charge-limited") and "spin_hopping" in score:
            score["spin_hopping"] += 2
        if fam in ("coherent tunnelling",):
            for k in ("spin_filter", "julliere", "proximity"):
                if k in score:
                    score[k] += 1
    mx = max(score.values())
    post = {k: np.exp(v - mx) for k, v in score.items()}
    t = sum(post.values()); post = {k: v / t for k, v in post.items()}
    ranking = sorted(keys, key=lambda k: -post[k])
    best = ranking[0]
    # bootstrap CIs for the best two models
    rng = np.random.default_rng(seed)
    for k in ranking[:2]:
        f = fits[k]
        resid = (y - f.pred)
        samples = {q: [] for q in f.params}
        for b in range(B):
            yb = f.pred + rng.choice(resid, size=len(resid), replace=True)
            fb = _fit_model(MODELS[k], D, yb, w, fixed, inits=f.params, n_starts=0, seed=b)
            for q, v in fb.params.items():
                samples[q].append(v)
        f.ci95 = {q: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for q, v in samples.items() if v}
    m = MODELS[best]
    par = ", ".join(f"{q} = {fits[best].params[q]:.4g}" + (f" [{fits[best].ci95[q][0]:.3g}, {fits[best].ci95[q][1]:.3g}]"
                                                           if q in fits[best].ci95 else "") for q in fits[best].params)
    statement = (f"Spin-dependent transport: {m.name} (support {100 * post[best]:.1f} %; Akaike weight "
                 f"{probs[best]:.3f}). Parameters (95 % bootstrap CI): {par}.")
    if fp.get("N_scaling"):
        statement += f" MR vs layer number: {fp['N_scaling']['class']}" + \
            (" with an odd/even parity effect" if fp["N_scaling"]["parity_effect"] else "") + "."
    if fp.get("sign_change_with_bias"):
        statement += " The MR changes sign with bias — a fingerprint of energy-dependent interface polarisation."
    return SpinResult(table=tab, fits=fits, ranking=ranking, best=best, probs=post, fingerprints=fp,
                      statement=statement, fixed=fixed)


# ---------------------------------------------------------------------------
# synthetic data for tests / tutorials
# ---------------------------------------------------------------------------
def synth_proximity(N_list=(4, 5, 6, 8), T_list=(10, 100, 200, 300), V=(0.005, 0.3), p=None, noise=0.03, seed=0):
    p = p or {"Delta": 20.0, "mu": 15.0, "Gamma": 3.0, "adk": 0.08, "TN": 700.0}
    rng = np.random.default_rng(seed)
    rows = []
    Vg = np.geomspace(*V, 25)
    for N in N_list:
        for T in T_list:
            Ip, Ia = proximity_currents(Vg, np.full_like(Vg, T), np.full_like(Vg, N), p)
            for v, a, b in zip(Vg, Ip, Ia):
                rows.append({"V": v, "T": T, "N": N, "d": N * 0.2735, "state": "P", "I": a * 1e-9 * np.exp(noise * rng.standard_normal())})
                rows.append({"V": v, "T": T, "N": N, "d": N * 0.2735, "state": "AP", "I": b * 1e-9 * np.exp(noise * rng.standard_normal())})
    return pd.DataFrame(rows), p
