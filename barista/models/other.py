"""Metallic paths: pinholes, filaments, point contacts (review Secs. XIV, XIX.C)."""
from __future__ import annotations

import numpy as np

from .base import Mechanism, Param, Check, fmt


class OhmicMetallic(Mechanism):
    key = "ohmic_metallic"
    name = "Ohmic metallic path (pinhole / filament / point contact)"
    family = "metallic paths"
    section = "XIV, XIX.C"
    short = "I = V / [R₀(1 + α(T − 300 K))], thickness- and area-independent; α > 0 for a metal."
    linearising_plot = "G histogram; G vs T"
    params = [
        Param("alpha", "1/K", -0.004, 0.01, 0.002, "temperature coefficient of resistance α"),
        Param("log10_G", "", -40, 40, 0.0, "conductance per area", profile=True),
    ]
    T_dependence = "metallic (dR/dT > 0)"
    V_dependence = "linear; G ≈ N G₀ for point contacts; abrupt switching"
    d_scaling = "area- and thickness-independent"
    key_condition = "formed filament / pinhole"
    necessary = "conductance decreasing on heating; area-independent resistance"
    falsified_by = "activated or tunnelling-like T dependence; exponential d scaling"
    alternatives = "degenerate semiconductor; ohmic hopping"
    minimum_measurements = "area scaling; G in units of G₀; device statistics"

    def lnJ(self, V, T, d, p, dev):
        V = np.maximum(np.asarray(V, float), 1e-12)
        R = 1.0 + p["alpha"] * (np.asarray(T, float) - 300.0)
        return np.log(V) - np.log(np.maximum(R, 1e-3))

    def checks(self, p, perr, ds, dev, diag):
        if p["alpha"] > 2e-4:
            return [Check("ok", f"α = {p['alpha']:.2e} K⁻¹ > 0: metallic temperature coefficient", +1)]
        return [Check("warn", f"α = {p['alpha']:.2e} K⁻¹ is not metallic (expected > 0)", -3)]

    def summary(self, p, perr):
        return f"α = {fmt(p['alpha'], perr.get('alpha'), 'K⁻¹')}"
