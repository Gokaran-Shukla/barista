"""Hopping through localised states (review Sec. X) and ionic hopping (Sec. XIV)."""
from __future__ import annotations

import numpy as np

from ..constants import NM
from .base import Mechanism, Param, Check, LN_FLOOR, kT, ln_sinhc, fmt


def _F(V, d):
    return np.maximum(np.asarray(V, float), 1e-9) / np.asarray(d, float)


class NNHopping(Mechanism):
    key = "nn_hopping"
    name = "Nearest-neighbour (activated) hopping  ≡  ionic hopping in steady state"
    family = "hopping"
    section = "X.B, X.D, XIV"
    short = "J = σ0 e^{−Ea/kT} F · sinh(eFr/2kT)/(eFr/2kT): single activation energy, sinh field law."
    linearising_plot = "ln σ vs 1/T;  ln J vs F"
    params = [
        Param("Ea", "eV", 0.002, 1.6, 0.2, "activation energy Ea", sanity=(0.005, 1.2)),
        Param("r", "nm", 0.05, 40.0, 2.0, "hop (or ionic jump) length r", sanity=(0.1, 20.0), log=True),
        Param("log10_s0", "", -40, 40, 0.0, "σ0 (S/m)", profile=True),
    ]
    T_dependence = "e^{−Ea/kT}  (w ∝ T⁻¹)"
    V_dependence = "ohmic → sinh(eFr/2kT)"
    d_scaling = "I ∝ 1/d (ohmic regime)"
    key_condition = "ξ < r < d"
    necessary = "single activation energy over a decade of w; I ∝ 1/d"
    falsified_by = "w(T) slope ≠ −1; ln I ∝ −d"
    alternatives = "polaron hopping; ionic conduction; SCLC with traps"
    minimum_measurements = "w(T); thickness series; sweep-rate dependence (ionic)"

    def lnJ(self, V, T, d, p, dev):
        F = _F(V, d)
        kt = kT(T)
        x = F * p["r"] * NM / (2 * kt)
        return np.maximum(np.log(F) - p["Ea"] / kt + ln_sinhc(x), LN_FLOOR)

    def checks(self, p, perr, ds, dev, diag):
        out = []
        dmin = float(ds.thicknesses.min())
        if p["r"] >= dmin:
            out.append(Check("fail", f"hop length r = {p['r']:.2f} nm ≥ thickness {dmin:.1f} nm: "
                             "not a bulk hopping process", -4))
        elif p["r"] > dmin / 3:
            out.append(Check("warn", f"hop length r = {p['r']:.2f} nm is a sizable fraction of d = {dmin:.1f} nm", -1))
        else:
            out.append(Check("ok", f"hop length r = {p['r']:.2f} nm < d", 0))
        if dev.layer_spacing_nm and p["r"] < 0.5 * dev.layer_spacing_nm:
            out.append(Check("warn", f"r smaller than the interlayer spacing a = {dev.layer_spacing_nm} nm", -1))
        return out

    def summary(self, p, perr):
        return f"Ea = {fmt(p['Ea'], perr.get('Ea'), 'eV')}, hop length r = {fmt(p['r'], perr.get('r'), 'nm')}"


class VRH(Mechanism):
    key = "vrh"
    name = "Variable-range hopping (Mott p = 1/4, 1/3 · Efros–Shklovskii p = 1/2)"
    family = "hopping"
    section = "X.C, X.D"
    short = "σ = σ0 exp[−(T0/T)^p]; field enhancement exp(eFL/kT), L ≈ 0.17 r_h, r_h ≈ 0.3 ξ (T0/T)^p."
    linearising_plot = "log w vs log T (Zabrodskii)"
    params = [
        Param("log10_T0", "K", 0.0, 10.0, 5.0, "log10 T0"),
        Param("p", "", 0.2, 0.75, 0.25, "exponent p", sanity=(0.24, 0.55)),
        Param("xi", "nm", 0.1, 60.0, 2.0, "localisation length ξ", sanity=(0.2, 30), log=True),
        Param("log10_s0", "", -40, 40, 0.0, "σ0 (S/m)", profile=True),
    ]
    T_dependence = "e^{−(T0/T)^p}  (w ∝ T^{−p})"
    V_dependence = "ohmic → e^{eFr_h/kT}"
    d_scaling = "I ∝ 1/d if r_h ≪ d"
    key_condition = "r_h ≪ d"
    necessary = "p from w(T) over a decade; r_h ≪ d"
    falsified_by = "r_h ≳ d; p drifting with T"
    alternatives = "Glazman–Matveev chains; NN hopping"
    minimum_measurements = "w(T) at several V; thickness series"

    @staticmethod
    def hop_length(p, T):
        return 0.3 * p["xi"] * (10.0 ** p["log10_T0"] / np.asarray(T, float)) ** p["p"]

    def lnJ(self, V, T, d, p, dev):
        F = _F(V, d)
        T = np.asarray(T, float)
        kt = kT(T)
        T0 = 10.0 ** p["log10_T0"]
        L = 0.17 * self.hop_length(p, T) * NM
        return np.maximum(np.log(F) - (T0 / T) ** p["p"] + ln_sinhc(F * L / kt), LN_FLOOR)

    def checks(self, p, perr, ds, dev, diag):
        out = []
        Tmin = float(ds.temperatures.min())
        rh = float(self.hop_length(p, Tmin))
        dmin = float(ds.thicknesses.min())
        if rh >= dmin:
            out.append(Check("fail", f"hop length r_h({Tmin:.0f} K) = {rh:.1f} nm ≥ d = {dmin:.1f} nm: "
                             "VRH is a bulk law and does not apply (Sec. X.C)", -5))
        elif rh > dmin / 3:
            out.append(Check("warn", f"r_h({Tmin:.0f} K) = {rh:.1f} nm is not ≪ d = {dmin:.1f} nm", -2))
        else:
            out.append(Check("ok", f"r_h({Tmin:.0f} K) = {rh:.1f} nm ≪ d", +0.5))
        pv = p["p"]
        canon = min([(abs(pv - c), n) for c, n in [(0.25, "3D Mott"), (1 / 3, "2D Mott"), (0.5, "Efros–Shklovskii")]])
        out.append(Check("ok" if canon[0] < 0.06 else "warn",
                         f"p = {pv:.2f} → closest to {canon[1]}", 0 if canon[0] < 0.06 else -1))
        cov = ds.coverage()
        if cov["T_range"][1] / max(cov["T_range"][0], 1e-9) < 3:
            out.append(Check("warn", "temperature window < factor 3: p cannot be pinned down (needs a decade in w)", -1))
        return out

    def summary(self, p, perr):
        return (f"T0 = 10^{p['log10_T0']:.2f} K, p = {fmt(p['p'], perr.get('p'))}, "
                f"ξ = {fmt(p['xi'], perr.get('xi'), 'nm')}")


class Polaron(Mechanism):
    key = "polaron"
    name = "Small-polaron (Marcus) hopping"
    family = "hopping"
    section = "X.E"
    short = "σ ∝ T^{−3/2} exp(−λ/4kT) (non-adiabatic Marcus, ΔG = 0); sinh field law with hop length a."
    linearising_plot = "ln(σT^{3/2}) vs 1/T"
    params = [
        Param("lam", "eV", 0.02, 3.0, 0.4, "reorganisation energy λ", sanity=(0.1, 1.0)),
        Param("a", "nm", 0.1, 10.0, 0.8, "hop length a", sanity=(0.2, 5.0), log=True),
        Param("log10_C", "", -40, 40, 0.0, "prefactor", profile=True),
    ]
    T_dependence = "T^{−3/2} e^{−λ/4kT}"
    V_dependence = "ohmic → field-assisted"
    d_scaling = "I ∝ 1/d"
    key_condition = "λ ≫ kT"
    necessary = "activation energy independent of carrier density; λ ≈ 4Ea in 0.1–1 eV"
    falsified_by = "VRH regime at low T; λ outside 0.1–1 eV"
    alternatives = "NN hopping; ionic"
    minimum_measurements = "ln(σT^{3/2}) vs 1/T; density (gate) dependence"

    def lnJ(self, V, T, d, p, dev):
        F = _F(V, d)
        T = np.asarray(T, float)
        kt = kT(T)
        x = F * p["a"] * NM / (2 * kt)
        return np.maximum(np.log(F) - 1.5 * np.log(T) - p["lam"] / (4 * kt) + ln_sinhc(x), LN_FLOOR)

    def checks(self, p, perr, ds, dev, diag):
        out = []
        lo, hi = 0.1, 1.0
        if lo <= p["lam"] <= hi:
            out.append(Check("ok", f"λ = {p['lam']:.2f} eV within 0.1–1 eV", 0))
        else:
            out.append(Check("warn", f"λ = {p['lam']:.2f} eV outside 0.1–1 eV", -1.5))
        Tmax = float(ds.temperatures.max())
        if p["lam"] < 4 * kT(Tmax) * 3:
            out.append(Check("warn", "λ not ≫ kT: Marcus limit questionable", -1))
        return out

    def summary(self, p, perr):
        return f"λ = {fmt(p['lam'], perr.get('lam'), 'eV')} (Ea ≈ λ/4 = {p['lam'] / 4:.3f} eV), a = {fmt(p['a'], perr.get('a'), 'nm')}"


class GDM(Mechanism):
    key = "gdm"
    name = "Gaussian-disorder hopping (Bässler)"
    family = "hopping"
    section = "X.F"
    short = "μ = μ∞ exp[−(2σ/3kT)²] exp{C[(σ/kT)² − Σ²]√F}, J = e n μ F."
    linearising_plot = "ln μ vs 1/T²;  ln μ vs √F"
    params = [
        Param("sigma", "eV", 0.02, 0.3, 0.1, "energetic disorder σ", sanity=(0.05, 0.15)),
        Param("Sigma", "", 0.0, 8.0, 2.0, "positional disorder Σ"),
        Param("log10_C", "", -40, 40, 0.0, "e n μ∞ (S/m)", profile=True),
    ]
    CB = 2.9e-5   # (m/V)^1/2   (= 2.9e-4 (cm/V)^1/2)
    T_dependence = "e^{−(2σ/3kT)²}"
    V_dependence = "ln μ ∝ √F"
    d_scaling = "I ∝ 1/d"
    key_condition = "disordered molecular solids"
    necessary = "ln μ linear in 1/T² rather than 1/T"
    falsified_by = "clean Arrhenius over a wide T range"
    alternatives = "Poole–Frenkel; polaron hopping"
    minimum_measurements = "wide T range; ln J vs 1/T²"

    def lnJ(self, V, T, d, p, dev):
        F = _F(V, d)
        kt = kT(T)
        s = p["sigma"] / kt
        return np.maximum(np.log(F) - (2 * s / 3) ** 2 + self.CB * (s**2 - p["Sigma"] ** 2) * np.sqrt(F), LN_FLOOR)

    def checks(self, p, perr, ds, dev, diag):
        lo, hi = 0.05, 0.15
        if lo <= p["sigma"] <= hi:
            return [Check("ok", f"σ = {1e3 * p['sigma']:.0f} meV within the organic range 50–150 meV", 0)]
        return [Check("warn", f"σ = {1e3 * p['sigma']:.0f} meV outside 50–150 meV", -1.5)]

    def summary(self, p, perr):
        return f"σ = {fmt(1e3 * p['sigma'], 1e3 * perr.get('sigma', np.nan), 'meV')}, Σ = {fmt(p['Sigma'], perr.get('Sigma'))}"
