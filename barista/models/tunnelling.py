"""Coherent and defect-mediated tunnelling (review Secs. VI and IX)."""
from __future__ import annotations

import numpy as np
from scipy.special import expit, logsumexp

from ..constants import e, h, hbar, me, NM
from .base import Mechanism, Param, Check, LN_FLOOR, kT, fmt, groups

PREF_TE = 4 * np.pi * me * e**3 / h**3          # A m^-2 eV^-2 (Tsu-Esaki supply prefactor)


def _wkb_c0(mstar):
    """2*sqrt(2 m* e)/hbar : multiplies an integral of sqrt(eV) over metres."""
    return 2.0 * np.sqrt(2.0 * mstar * me * e) / hbar


def _trapezoid_integral(a, V, d):
    """int_0^d sqrt(max(a - V x/d, 0)) dx  for a barrier tilted by V (eV).  a, V broadcast."""
    a = np.asarray(a, float)
    V = np.asarray(V, float)
    pos = a > 0
    ap = np.where(pos, a, 0.0)
    Vs = np.maximum(V, 1e-12)
    tilt = d * (2.0 / (3.0 * Vs)) * (ap**1.5 - np.maximum(ap - Vs, 0.0) ** 1.5)
    flat = d * np.sqrt(ap)
    small = Vs < 1e-6 * np.maximum(ap, 1e-12)
    return np.where(pos, np.where(small, flat, tilt), 0.0)


class Coherent(Mechanism):
    key = "coherent"
    name = "Coherent tunnelling (Tsu–Esaki/WKB: direct ↔ Fowler–Nordheim ↔ thermionic–field)"
    family = "coherent tunnelling"
    section = "VI, VII.A, App. A"
    short = ("Tsu–Esaki integral with WKB transmission through a trapezoidal barrier; one formula "
             "covers direct tunnelling, Fowler–Nordheim, thermionic–field and thermionic emission.")
    linearising_plot = "ln G vs d;  ln(J/V²) vs 1/V (FN);  G(V)/G(0) parabolic (BDR)"
    expensive = True
    params = [
        Param("phi", "eV", 0.05, 4.0, 1.0, "barrier height Φ from E_F", sanity=(0.1, 3.5),
              device_key="barrier_eV"),
        Param("mstar", "m_e", 0.05, 3.0, 0.5, "tunnelling effective mass m*", sanity=(0.05, 2.0),
              device_key="mass_ratio"),
        Param("log10_s", "", -12, 6, 0.0, "effective-area factor s (1 = whole area tunnels)",
              sanity=(-3, 1), profile=True),
    ]
    T_dependence = "weak, ∝ πckT/sin(πckT) (Stratton)"
    V_dependence = "ohmic → V+γV³; ln(I/V²) ∝ −bd/V above Φ/e"
    d_scaling = "ln I ∝ −2κd"
    key_condition = "d < ℓφ, ℓin;  no states in the gap"
    necessary = "ln G ∝ −2κd; weak, insulating-like T dependence (Rowell ratio ≈ 1–1.3)"
    falsified_by = "G rising on cooling (metallic); I ∝ 1/d; current far above the WKB estimate; strong Arrhenius"
    alternatives = "pinholes; trap-assisted tunnelling; thermionic–field emission"
    minimum_measurements = "thickness series; Rowell ratio; absolute magnitude"

    def _integrate(self, V, T, d, phi, mstar, EF, return_mean=False):
        Vmax = float(np.max(V))
        kt = float(kT(T))
        emin = max(-EF, -(Vmax + 1.5))
        emax = phi + 40.0 * kt + 0.02
        step = min(kt / 2.0, 0.003)
        n = int(np.clip((emax - emin) / step, 600, 3000))
        E = np.linspace(emin, emax, n)[:, None]               # eV, relative to E_F(left)
        Vr = np.asarray(V, float)[None, :]
        supply = kt * (np.logaddexp(0.0, -E / kt) - np.logaddexp(0.0, -(E + Vr) / kt))
        integ = _trapezoid_integral(phi - E, Vr, d)
        lnT = -_wkb_c0(mstar) * integ
        with np.errstate(divide="ignore"):
            lnint = lnT + np.log(np.maximum(supply, 1e-300))
        dE = E[1, 0] - E[0, 0]
        lnJ = np.log(PREF_TE) + logsumexp(lnint, axis=0) + np.log(dE)
        if return_mean:
            w = np.exp(lnint - lnint.max(axis=0, keepdims=True))
            Emean = (w * E).sum(axis=0) / np.maximum(w.sum(axis=0), 1e-300)
            return lnJ, Emean
        return lnJ

    def lnJ(self, V, T, d, p, dev):
        V = np.asarray(V, float); T = np.asarray(T, float); d = np.asarray(d, float)
        out = np.empty_like(V)
        for m in groups(T, d):
            out[m] = self._integrate(V[m], T[m][0], d[m][0], p["phi"], p["mstar"], dev.fermi_energy_eV)
        return np.maximum(out, LN_FLOOR)

    def regime(self, V, T, d, p, dev):
        V = np.asarray(V, float); T = np.asarray(T, float); d = np.asarray(d, float)
        labels = np.empty(V.shape, dtype=object)
        phi = p["phi"]
        for m in groups(T, d):
            _, Em = self._integrate(V[m], T[m][0], d[m][0], phi, p["mstar"], dev.fermi_energy_eV, True)
            lab = np.where(Em >= 0.8 * phi, "thermionic emission",
                  np.where(Em >= 0.15 * phi, "thermionic–field emission",
                  np.where(V[m] > phi, "Fowler–Nordheim", "direct tunnelling")))
            labels[m] = lab
        return labels

    def checks(self, p, perr, ds, dev, diag):
        out = []
        kappa = np.sqrt(2 * p["mstar"] * me * p["phi"] * e) / hbar / 1e9   # nm^-1
        lvl = "ok" if 2.0 <= kappa <= 12.0 else "warn"
        out.append(Check(lvl, f"decay constant κ = {kappa:.2f} nm⁻¹ (sanity 3–10 nm⁻¹)", 0 if lvl == "ok" else -1))
        s = p["log10_s"]
        if ds.area_known:
            if s > 1.5:
                out.append(Check("fail", f"measured current is 10^{s:.1f}× the WKB estimate for the full area → "
                                 "current cannot flow by coherent tunnelling through this barrier "
                                 "(pinholes, TAT or hopping)", -6))
            elif s < -4:
                out.append(Check("warn", f"only 10^{s:.1f} of the junction area would need to conduct — "
                                 "hot spots / thinner regions, or a different mechanism", -2))
            else:
                out.append(Check("ok", f"absolute magnitude consistent with WKB (area factor 10^{s:.1f})", +1))
        dmin = float(ds.thicknesses.min())
        if dmin > 8.0:
            out.append(Check("warn", f"barrier ≥ {dmin:.0f} nm: coherent tunnelling is normally negligible "
                             "unless Φ is very small (Sec. IV)", -2))
        if dev.barrier_eV is not None and abs(p["phi"] - dev.barrier_eV) > 0.3:
            out.append(Check("warn", f"fitted Φ = {p['phi']:.2f} eV differs from the supplied {dev.barrier_eV:.2f} eV", -1))
        if p["phi"] < 0.12:
            out.append(Check("warn", "Φ < 0.12 eV: transport is effectively over-barrier; "
                             "check the regime label", 0))
        return out

    def summary(self, p, perr):
        return (f"Φ = {fmt(p['phi'], perr.get('phi'), 'eV')}, m* = {fmt(p['mstar'], perr.get('mstar'), 'mₑ')}, "
                f"κ(E_F) = {np.sqrt(2 * p['mstar'] * me * p['phi'] * e) / hbar / 1e9:.2f} nm⁻¹")


class TrapAssisted(Mechanism):
    key = "tat"
    name = "Trap-assisted tunnelling (sequential, elastic)"
    family = "defect-mediated tunnelling"
    section = "IX.A"
    short = ("Two-step tunnelling via traps of depth Φt below the barrier band edge, uniformly "
             "distributed in the layer: J ∝ Nt ∫ T1T2/(T1+T2) over trap positions in the bias window.")
    linearising_plot = "dI/dV shoulders; ln(J) vs 1/F at high field"
    params = [
        Param("phi", "eV", 0.1, 4.0, 1.5, "barrier height Φ", sanity=(0.2, 3.5), device_key="barrier_eV"),
        Param("phit", "eV", 0.02, 3.5, 0.8, "trap depth Φt below the band edge", sanity=(0.05, 3.0),
              device_key="trap_depth_eV"),
        Param("mstar", "m_e", 0.05, 3.0, 0.5, "tunnelling mass", device_key="mass_ratio", fixed_default=True),
        Param("log10_C", "", -60, 60, 10.0, "ν·Nt prefactor (A m⁻²)", profile=True),
    ]
    T_dependence = "weak (elastic); activated if multiphonon"
    V_dependence = "shoulders/peaks when traps align; FN-like with Φt at high field"
    d_scaling = "∝ trap number; ~e^{−κd} (not e^{−2κd})"
    key_condition = "a few traps per path; Nt, Et, xt"
    necessary = "current ≫ direct-tunnelling estimate; weak T dependence"
    falsified_by = "strong Arrhenius behaviour at all biases"
    alternatives = "multiphonon TAT; pinholes"
    minimum_measurements = "noise (RTN); device statistics; thickness series"

    def lnJ(self, V, T, d, p, dev):
        V = np.asarray(V, float); T = np.asarray(T, float); d = np.asarray(d, float)
        nx = 160
        u = (np.arange(nx) + 0.5) / nx                              # fractional position
        x = u[:, None] * d[None, :]
        Fp = np.maximum(V, 1e-9)[None, :] / d[None, :]               # eV per metre
        phit, phi = p["phit"], p["phi"]
        c0 = _wkb_c0(p["mstar"])
        # left electrode -> trap at x : barrier (Phit + F(x - x')) for x' in [0, x]
        a1 = (2.0 / (3.0 * Fp)) * ((phit + Fp * x) ** 1.5 - phit**1.5)
        a1 = np.where(Fp * x < 1e-6 * phit, np.sqrt(phit) * x, a1)
        # trap -> right electrode : barrier (Phit - F(x'-x)) for x' in [x, d], clipped
        rem = d[None, :] - x
        a2 = (2.0 / (3.0 * Fp)) * (phit**1.5 - np.maximum(phit - Fp * rem, 0.0) ** 1.5)
        a2 = np.where(Fp * rem < 1e-6 * phit, np.sqrt(phit) * rem, a2)
        lnT1, lnT2 = -c0 * a1, -c0 * a2
        lnrate = -np.logaddexp(-lnT1, -lnT2)
        Etrap = phi - phit - V[None, :] * u[:, None]                # eV relative to E_F(left)
        kt = kT(T)[None, :]
        window = expit(-Etrap / kt) - expit(-(Etrap + V[None, :]) / kt)
        with np.errstate(divide="ignore"):
            lnw = np.log(np.maximum(window, 1e-300))
        lnJ = logsumexp(lnrate + lnw, axis=0) + np.log(d / nx)
        return np.maximum(lnJ, LN_FLOOR)

    def checks(self, p, perr, ds, dev, diag):
        out = []
        if p["phit"] >= p["phi"] + 0.5:
            out.append(Check("warn", "trap level lies far below the electrode Fermi level; "
                             "traps permanently filled", -1))
        if dev.trap_depth_eV is not None and abs(dev.trap_depth_eV - p["phit"]) > 0.3:
            out.append(Check("warn", f"fitted Φt = {p['phit']:.2f} eV vs supplied {dev.trap_depth_eV:.2f} eV", -1))
        return out

    def summary(self, p, perr):
        return f"Φ = {fmt(p['phi'], perr.get('phi'), 'eV')}, trap depth Φt = {fmt(p['phit'], perr.get('phit'), 'eV')}"


class GlazmanMatveev(Mechanism):
    key = "glazman_matveev"
    name = "Glazman–Matveev chains (inelastic hopping through N = 1, 2, 3 localised states)"
    family = "defect-mediated tunnelling"
    section = "IX.C"
    short = ("G = Σ_N c_N E^{p_N} e^{−2d/((N+1)ξ)}, p_N = N − 2/(N+1) = 0, 4/3, 5/2, "
             "E = √((kT)² + (eV)²): successive power laws in T and V.")
    linearising_plot = "log G vs log T (slopes 0, 4/3, 5/2)"
    params = [
        Param("x2", "", -14, 8, -2.0, "log10(c2/c1)"),
        Param("x3", "", -18, 8, -5.0, "log10(c3/c1)"),
        Param("xi", "nm", 0.2, 30.0, 2.0, "localisation length ξ (needs a thickness series)", sanity=(0.2, 20)),
        Param("log10_c1", "", -40, 40, 0.0, "elastic (N=1) conductance prefactor", profile=True),
    ]
    T_dependence = "G ∝ T^{N−2/(N+1)}: T⁰, T^{4/3}, T^{5/2}"
    V_dependence = "I ∝ V^{1+p_N}"
    d_scaling = "dominant N grows with d"
    key_condition = "r ∼ d/N; few-site chains"
    necessary = "power-law exponents 4/3, 5/2 in both T and V"
    falsified_by = "single activation energy; exponential d dependence with one κ"
    alternatives = "VRH; NN hopping"
    minimum_measurements = "log G vs log T at low bias; several thicknesses"

    def lnJ(self, V, T, d, p, dev):
        V = np.asarray(V, float); T = np.asarray(T, float); d = np.asarray(d, float)
        Emev = 1e3 * np.sqrt(kT(T) ** 2 + V**2)
        xi = p["xi"] * NM
        lnE = np.log(np.maximum(Emev, 1e-6))
        terms = [
            0.0 - 2 * d / (2 * xi),
            p["x2"] * np.log(10) + (4 / 3) * lnE - 2 * d / (3 * xi),
            p["x3"] * np.log(10) + (5 / 2) * lnE - 2 * d / (4 * xi),
        ]
        lnG = np.logaddexp(np.logaddexp(terms[0], terms[1]), terms[2])
        return np.log(V) + lnG

    def summary(self, p, perr):
        return f"ξ = {fmt(p['xi'], perr.get('xi'), 'nm')}, chain weights 10^{p['x2']:.1f} (N=2), 10^{p['x3']:.1f} (N=3)"


class BandToBand(Mechanism):
    key = "band_to_band"
    name = "Band-to-band (Zener/Kane) tunnelling"
    family = "coherent tunnelling"
    section = "VI.G"
    short = "T ≃ exp[−π √m* Eg^{3/2} / (2√2 eħF)];  J ∝ F² T"
    linearising_plot = "ln(J/F²) vs 1/F"
    params = [
        Param("Eg", "eV", 0.05, 5.0, 1.0, "band gap Eg", sanity=(0.1, 4.0), device_key="band_gap_eV"),
        Param("mstar", "m_e", 0.02, 2.0, 0.2, "reduced tunnelling mass", device_key="mass_ratio", fixed_default=True),
        Param("log10_C", "", -40, 40, 0.0, "prefactor", profile=True),
    ]
    T_dependence = "very weak, via Eg(T)"
    V_dependence = "ln(J/F²) ∝ −Eg^{3/2}/F"
    d_scaling = "via F only"
    key_condition = "band overlap under field"
    necessary = "very weak T dependence; FN-like plot with Eg"
    falsified_by = "activated current; Eg inconsistent with optics"
    alternatives = "Fowler–Nordheim; TAT"
    minimum_measurements = "T at fixed V; doping series; Eg from optics"

    def lnJ(self, V, T, d, p, dev):
        F = np.maximum(np.asarray(V, float), 1e-9) / np.asarray(d, float)
        b = np.pi * np.sqrt(p["mstar"] * me) * (p["Eg"] * e) ** 1.5 / (2 * np.sqrt(2) * e * hbar)
        return np.maximum(2 * np.log(F) - b / F, LN_FLOOR)

    def checks(self, p, perr, ds, dev, diag):
        if dev.band_gap_eV is not None and abs(p["Eg"] - dev.band_gap_eV) > 0.25 * dev.band_gap_eV:
            return [Check("warn", f"fitted Eg = {p['Eg']:.2f} eV inconsistent with optical {dev.band_gap_eV:.2f} eV", -2)]
        return []

    def summary(self, p, perr):
        return f"Eg = {fmt(p['Eg'], perr.get('Eg'), 'eV')} (m* = {p['mstar']:.2f} mₑ)"
