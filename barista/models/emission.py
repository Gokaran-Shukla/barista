"""Emission over the barrier (review Sec. VII)."""
from __future__ import annotations

import numpy as np

from ..constants import e, eps0, A_RICHARDSON
from .base import Mechanism, Param, Check, LN_FLOOR, kT, softplus, fmt


def _beta_schottky(eps_r):   # eV per sqrt(V/m)
    return np.sqrt(e / (4 * np.pi * eps0 * eps_r))


def _beta_pf(eps_r):
    return np.sqrt(e / (np.pi * eps0 * eps_r))


def _eps_check(eps_fit, dev, label):
    out = []
    ref = dev.eps_optical
    if ref is None:
        if eps_fit < 1.0 or eps_fit > 30:
            out.append(Check("warn", f"{label}: ε_r = {eps_fit:.2f} outside the range of optical permittivities (1–30); "
                             "give n_opt for a decisive test", -2))
        else:
            out.append(Check("ok", f"{label}: ε_r = {eps_fit:.2f} (supply n_opt to compare with n²)", 0))
        return out
    ratio = eps_fit / ref
    if 0.6 <= ratio <= 1.6:
        out.append(Check("ok", f"{label}: ε_r = {eps_fit:.2f} ≈ n² = {ref:.2f} ✓ (Table 18)", +3))
    elif 0.2 <= ratio <= 3.0:
        out.append(Check("warn", f"{label}: ε_r = {eps_fit:.2f} vs n² = {ref:.2f} (ratio {ratio:.2f})", -1.5))
    else:
        out.append(Check("fail", f"{label}: ε_r = {eps_fit:.2f} inconsistent with n² = {ref:.2f} "
                         f"(ratio {ratio:.2f}); the √F slope belongs to a different mechanism", -5))
    return out


class Schottky(Mechanism):
    key = "schottky"
    name = "Richardson–Schottky (thermionic) emission at the contact"
    family = "emission"
    section = "VII.A"
    short = "J = s A* T² exp[−(Φ_B − √(e³F/4πε))/kT], interface-limited injection."
    linearising_plot = "ln(J/T²) vs 1/T;  ln J vs √V"
    params = [
        Param("phi", "eV", 0.02, 2.5, 0.6, "Schottky barrier Φ_B", sanity=(0.03, 2.0), device_key="barrier_eV"),
        Param("eps_r", "", 0.3, 100.0, 5.0, "dynamic permittivity from the √F slope", sanity=(1.0, 30.0),
              device_key="n_opt", device_map=lambda n: n * n, log=True),
        Param("log10_s", "", -25, 15, 0.0, "A*_eff / 120 A cm⁻²K⁻² (Table 18: within 10^±2)", sanity=(-2, 2),
              profile=True),
    ]
    T_dependence = "T² e^{−(Φ−ΔΦ_S)/kT}"
    V_dependence = "ln I ∝ √V"
    d_scaling = "weak (contact-limited); only via F"
    key_condition = "kT vs Φ; contact-limited"
    necessary = "activation energy decreasing as √V; Richardson constant of the correct order"
    falsified_by = "A* off by many orders; Ea independent of V; bulk d scaling"
    alternatives = "Poole–Frenkel; hopping; thermionic–field emission"
    minimum_measurements = "Ea(V); A* from the intercept; thickness series"

    def lnJ(self, V, T, d, p, dev):
        F = np.maximum(np.asarray(V, float), 1e-9) / np.asarray(d, float)
        kt = kT(T)
        barrier = kt * softplus((p["phi"] - _beta_schottky(p["eps_r"]) * np.sqrt(F)) / kt)
        alpha = 1.5 if dev.contact_2d else 2.0
        return np.maximum(np.log(A_RICHARDSON) + alpha * np.log(T) - barrier / kt, LN_FLOOR)

    def checks(self, p, perr, ds, dev, diag):
        out = _eps_check(p["eps_r"], dev, "Schottky")
        s = p["log10_s"]
        if ds.area_known and not dev.contact_2d:
            if abs(s) <= 2:
                out.append(Check("ok", f"effective Richardson constant 10^{s:.1f} × 120 A cm⁻²K⁻² (within 10^±2)", +1.5))
            elif abs(s) <= 4:
                out.append(Check("warn", f"effective Richardson constant 10^{s:.1f} × 120 A cm⁻²K⁻²", -2))
            else:
                out.append(Check("fail", f"Richardson constant off by 10^{s:.1f}: simple thermionic emission "
                                 "cannot supply this current (Sec. VII absolute-magnitude check)", -6))
        return out

    def summary(self, p, perr):
        return (f"Φ_B = {fmt(p['phi'], perr.get('phi'), 'eV')}, ε_r = {fmt(p['eps_r'], perr.get('eps_r'))}, "
                f"A*_eff = 10^{p['log10_s']:.1f} × 120 A cm⁻²K⁻²")


class PooleFrenkel(Mechanism):
    key = "poole_frenkel"
    name = "Poole–Frenkel emission from bulk Coulombic traps"
    family = "emission"
    section = "VII.C"
    short = "J = C F exp[−(Φt − √(e³F/πε))/kT], bulk-limited."
    linearising_plot = "ln(J/F) vs √F"
    params = [
        Param("phit", "eV", 0.02, 2.5, 0.5, "trap depth Φt", sanity=(0.05, 2.0), device_key="trap_depth_eV"),
        Param("eps_r", "", 0.3, 100.0, 5.0, "dynamic permittivity (expect ≈ n²)", sanity=(1.0, 30.0),
              device_key="n_opt", device_map=lambda n: n * n, log=True),
        Param("log10_C", "", -40, 40, 0.0, "e n μ prefactor (S/m)", profile=True),
    ]
    T_dependence = "e^{−(Φt − β√F)/kT}"
    V_dependence = "ln(I/V) ∝ √V;  ε_r ≈ n²"
    d_scaling = "via F = V/d"
    key_condition = "bulk Coulombic traps"
    necessary = "ε_r from the slope ≈ n²; bulk-limited (d enters through F)"
    falsified_by = "ε_r inconsistent with optics; contact-dependent current"
    alternatives = "Schottky; Gaussian-disorder hopping; SCLC"
    minimum_measurements = "slope vs T; thickness series; contact variation"

    def lnJ(self, V, T, d, p, dev):
        F = np.maximum(np.asarray(V, float), 1e-9) / np.asarray(d, float)
        kt = kT(T)
        barrier = kt * softplus((p["phit"] - _beta_pf(p["eps_r"]) * np.sqrt(F)) / kt)
        return np.maximum(np.log(F) - barrier / kt, LN_FLOOR)

    def checks(self, p, perr, ds, dev, diag):
        return _eps_check(p["eps_r"], dev, "Poole–Frenkel")

    def summary(self, p, perr):
        return f"Φt = {fmt(p['phit'], perr.get('phit'), 'eV')}, ε_r = {fmt(p['eps_r'], perr.get('eps_r'))}"
