"""Space-charge-limited conduction (review Sec. XI, App. D)."""
from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid

from ..constants import e, eps0, kB_eV
from .base import Mechanism, Param, Check, LN_FLOOR, kT, fmt, groups

def _sgrid(ln_n0):
    """Grid in s = -ln(1-u), u = n0/n, reaching injected densities up to ~1e31 m^-3."""
    lg_umin = min(-12.0, (ln_n0 - np.log(1e31)) / np.log(10))
    S = np.concatenate([[0.0], np.logspace(lg_umin, np.log10(90.0), 3200)])
    return S, -np.expm1(-S), np.exp(-S)


def _nc(dev, T):
    return dev.Nc_cm3 * 1e6 * (np.asarray(T, float) / 300.0) ** 1.5


class SCLCTrap(Mechanism):
    key = "sclc_trap"
    name = "Space-charge-limited current with a discrete trap level (exact drift–Poisson; Mott–Gurney/Lampert)"
    family = "space-charge-limited"
    section = "XI, App. D"
    short = ("Exact single-carrier drift–Poisson solution with ohmic injection, equilibrium density n0 and one "
             "trap level (Nt, Ec−Et): ohmic → θ·(9/8)εμV²/d³ → trap-filled limit → Mott–Gurney.")
    linearising_plot = "log J vs log V (Lampert triangle);  J/d vs V/d²"
    expensive = True
    params = [
        Param("Ef0", "eV", 0.05, 1.6, 0.5, "Ec − E_F (sets equilibrium n0(T))", sanity=(0.05, 1.5)),
        Param("log10_Nt", "m⁻³", 17.0, 27.5, 23.0, "trap density Nt (log10 m⁻³)", sanity=(18, 27),
              device_key="trap_density_cm3", device_map=lambda n: np.log10(n * 1e6)),
        Param("Et", "eV", 0.02, 1.4, 0.4, "trap depth Ec − Et", sanity=(0.03, 1.3), device_key="trap_depth_eV"),
        Param("log10_mu", "m²/Vs", -14, 4, -5.0, "mobility μ (log10 m²/Vs)", sanity=(-10, -1), profile=True,
              device_key="mobility_cm2Vs", device_map=lambda m: np.log10(m * 1e-4)),
    ]
    T_dependence = "activated via θt (Ec − Et)"
    V_dependence = "I ∝ Vⁿ, n ≥ 2; peak of γ at V_TFL"
    d_scaling = "J/d = F(V/d²): J ∝ d⁻³ (trap-free), V_TFL ∝ d²"
    key_condition = "ohmic (injecting) contact, single carrier"
    necessary = "J/d = F(V/d²) collapse; V_TFL ∝ d²"
    falsified_by = "no collapse; contact-dependent current"
    alternatives = "series element; Poole–Frenkel; injection-limited current"
    minimum_measurements = "≥ 3 thicknesses; pulsed I(V); impedance (transit time)"

    def _group(self, V, T, d, p, dev):
        eps = eps0 * dev.eps_static
        kt = float(kT(T))
        Nc = float(_nc(dev, T))
        ln_n0 = np.log(Nc) - p["Ef0"] / kt
        ln_n1 = np.log(Nc) - p["Et"] / kt
        if ln_n0 < np.log(1e6):
            return self._insulator(V, d, eps, ln_n0, ln_n1, p)
        return self._with_background(V, d, eps, ln_n0, ln_n1, p)

    @staticmethod
    def _invert(V, lnVg, lnj):
        lnVg = np.maximum.accumulate(lnVg)
        return np.interp(np.log(np.maximum(V, 1e-9)), lnVg, lnj)

    def _insulator(self, V, d, eps, ln_n0, ln_n1, p):
        """n0 -> 0 limit: rho = e[n + Nt(n/(n+n1) - f0)], exact drift-Poisson by quadrature in ln n."""
        Nt = 10.0 ** p["log10_Nt"]
        f0 = 1.0 / (1.0 + np.exp(np.clip(ln_n1 - ln_n0, -700, 700)))
        lnn = np.linspace(np.log(1e-8), np.log(1e33), 4000)
        n = np.exp(lnn)
        n1 = np.exp(max(ln_n1, -700.0))
        extra = Nt * np.maximum(1.0 / (n + n1) - f0 / n, 0.0)
        lnR = np.log1p(extra)
        dl = lnn[1] - lnn[0]
        la = -2 * lnn - lnR + np.log(dl)
        lb = -3 * lnn - lnR + np.log(dl)
        lnA = np.logaddexp.accumulate(la[::-1])[::-1]          # int_n^inf dn/(n^3 R)
        lnB = np.logaddexp.accumulate(lb[::-1])[::-1]          # int_n^inf dn/(n^4 R)
        Vmax = float(np.max(V))
        hi = 9.0 / 8.0 * eps * Vmax**2 / d**3 * 1e3
        lnj = np.linspace(np.log(hi) - 600.0, np.log(hi), 900)
        ln_tA = 2 * np.log(e) + np.log(d) - np.log(eps) - lnj
        lnna = np.interp(-ln_tA, -lnA, lnn)                    # lnA decreasing in n
        lnBa = np.interp(lnna, lnn, lnB)
        lnVg = np.log(eps) + 2 * lnj + lnBa - 3 * np.log(e)
        return self._invert(V, lnVg, lnj)

    def _with_background(self, V, d, eps, ln_n0, ln_n1, p):
        """Finite equilibrium density n0: exact solution in u = n0/n with an ohmic tail."""
        Nt = 10.0 ** p["log10_Nt"]
        _S, _U, _ONE_MINUS_U = _sgrid(ln_n0)
        ln_r1 = ln_n1 - ln_n0
        ln_rt = np.log(Nt) - ln_n0
        with np.errstate(divide="ignore"):
            lnu = np.log(np.where(_U > 0, _U, 1e-300))
        ln_trap = ln_rt + ln_r1 - np.logaddexp(0.0, ln_r1 + lnu) - np.logaddexp(0.0, ln_r1)
        invu = np.where(_U > 0, 1.0 / np.where(_U > 0, _U, 1.0), np.inf)
        B = invu + np.exp(np.minimum(ln_trap, 700.0))
        invB = np.where(np.isfinite(B), 1.0 / B, 0.0)
        g = cumulative_trapezoid(invB, _S, initial=0.0)
        hh = cumulative_trapezoid(_U * invB, _S, initial=0.0)
        q = cumulative_trapezoid(_ONE_MINUS_U * invB, _S, initial=0.0)
        n0 = np.exp(ln_n0)
        Vmin, Vmax = max(float(np.min(V)), 1e-6), float(np.max(V))
        theta = 1.0 / (1.0 + Nt / max(np.exp(ln_n1), 1e-300))
        ohm = lambda v: e * n0 * v / d
        mg = lambda v: 9.0 / 8.0 * eps * v * v / d**3
        hi = max(ohm(Vmax), mg(Vmax)) * 1e3
        lo = max(min(ohm(Vmin), mg(Vmin) * max(theta, 1e-300)) * 1e-3, hi * 1e-250)
        lnj = np.linspace(np.log(lo), np.log(hi), 900)
        ln_gt = 2 * np.log(e) + np.log(d) + 2 * ln_n0 - np.log(eps) - lnj
        inside = ln_gt <= np.log(g[-1])
        lng = np.log(np.maximum(g[1:], 1e-300))
        lnh = np.log(np.maximum(hh[1:], 1e-300))
        lnh_a = np.interp(ln_gt, lng, lnh)
        lnV_in = np.log(eps) + 2 * lnj + lnh_a - 3 * np.log(e) - 3 * ln_n0
        rho = np.clip(q[-1] / np.exp(np.minimum(ln_gt, 700.0)), 0.0, 1 - 1e-15)
        lnV_out = lnj + np.log(d) - np.log(e) - ln_n0 + np.log1p(-rho)
        return self._invert(V, np.where(inside, lnV_in, lnV_out), lnj)

    def lnJ(self, V, T, d, p, dev):
        V = np.asarray(V, float); T = np.asarray(T, float); d = np.asarray(d, float)
        out = np.empty_like(V)
        for m in groups(T, d):
            out[m] = self._group(V[m], T[m][0], d[m][0], p, dev)
        return np.maximum(out, LN_FLOOR)

    @staticmethod
    def vtfl(p, d_m, dev):
        return e * 10.0 ** p["log10_Nt"] * d_m**2 / (2 * eps0 * dev.eps_static)

    def regime(self, V, T, d, p, dev):
        V = np.asarray(V, float); T = np.asarray(T, float); d = np.asarray(d, float)
        kt = kT(T)
        Nc = _nc(dev, T)
        n0 = np.maximum(Nc * np.exp(-p["Ef0"] / kt), 1e-6)
        n1 = Nc * np.exp(-p["Et"] / kt)
        Nt = 10.0 ** p["log10_Nt"]
        theta = n1 / (n1 + Nt)
        eps = eps0 * dev.eps_static
        VOm = 8 * e * n0 * d**2 / (9 * eps * np.maximum(theta, 1e-300))
        Vtfl = self.vtfl(p, d, dev)
        return np.where(V < VOm, "ohmic (equilibrium carriers)",
               np.where(V < 0.8 * Vtfl, "trap-limited SCLC (∝ θV²)",
               np.where(V < 1.5 * Vtfl, "trap-filled limit", "trap-free SCLC (Mott–Gurney)")))

    def checks(self, p, perr, ds, dev, diag):
        out = []
        dnm = ds.thicknesses
        Vmin, Vmax = ds.coverage()["V_range"]
        vt = [self.vtfl(p, dd * 1e-9, dev) for dd in dnm]
        txt = ", ".join(f"{v:.3g} V (d={dd:g} nm)" for v, dd in zip(vt, dnm))
        inside = any(Vmin <= v <= Vmax for v in vt)
        out.append(Check("ok", f"V_TFL = eNt d²/2ε = {txt}" + (" — inside the measured window" if inside else ""), 0))
        Nt_cm3 = 10.0 ** p["log10_Nt"] / 1e6
        if dev.trap_density_cm3 is not None:
            r = Nt_cm3 / dev.trap_density_cm3
            lvl = "ok" if 0.1 <= r <= 10 else "warn"
            out.append(Check(lvl, f"fitted Nt = {Nt_cm3:.2e} cm⁻³ vs supplied {dev.trap_density_cm3:.2e} cm⁻³",
                             +1 if lvl == "ok" else -1.5))
        if ds.area_known:
            mu = 10.0 ** p["log10_mu"] * 1e4
            if 1e-7 <= mu <= 1e4:
                out.append(Check("ok", f"μ = {mu:.3g} cm²/Vs (plausible)", 0))
            else:
                out.append(Check("warn", f"μ = {mu:.3g} cm²/Vs is implausible", -2))
        gp = (diag or {}).get("gamma_peak_scaling")
        if gp and gp.get("exponent") is not None:
            ex = gp["exponent"]
            if abs(ex - 2) < 0.5:
                out.append(Check("ok", f"γ-peak voltage scales as d^{ex:.2f} ≈ d² (V_TFL ∝ d²) ✓", +3))
            else:
                out.append(Check("warn", f"γ-peak voltage scales as d^{ex:.2f}, not d²", -2))
        if dev.eps_r is None and dev.n_opt is None:
            out.append(Check("warn", "ε_r not supplied; assumed 5 (Nt and μ scale with it)", 0))
        return out

    def summary(self, p, perr):
        Nt = 10.0 ** p["log10_Nt"] / 1e6
        return (f"Nt = {Nt:.2e} cm⁻³, Ec−Et = {fmt(p['Et'], perr.get('Et'), 'eV')}, "
                f"Ec−E_F = {fmt(p['Ef0'], perr.get('Ef0'), 'eV')}, μ = {10.0 ** p['log10_mu'] * 1e4:.3g} cm²/Vs")


class SCLCExp(Mechanism):
    key = "sclc_exp"
    name = "SCLC with an exponential trap distribution (Mark–Helfrich)"
    family = "space-charge-limited"
    section = "XI.F"
    short = "J ∝ μNc (ε/Nt)^l V^{l+1}/d^{2l+1}, l = Tc/T: power law steepening on cooling."
    linearising_plot = "log J vs log V; exponent l+1 vs 1/T"
    params = [
        Param("Tc", "K", 30.0, 6000.0, 800.0, "characteristic trap temperature Tc", sanity=(100, 3000), log=True),
        Param("log10_Nt", "m⁻³", 17.0, 27.5, 23.5, "total trap density (log10 m⁻³)", sanity=(18, 27),
              device_key="trap_density_cm3", device_map=lambda n: np.log10(n * 1e6)),
        Param("Ef0", "eV", 0.05, 1.6, 0.6, "Ec − E_F (ohmic background)"),
        Param("log10_mu", "m²/Vs", -14, 4, -5.0, "mobility (log10 m²/Vs)", profile=True,
              device_key="mobility_cm2Vs", device_map=lambda m: np.log10(m * 1e-4)),
    ]
    T_dependence = "exponent l+1 = Tc/T + 1 grows on cooling"
    V_dependence = "J ∝ V^{l+1}"
    d_scaling = "J ∝ d^{−(2l+1)}"
    key_condition = "injecting contact; distributed traps"
    necessary = "T-dependent power-law exponent; J/d = F(V/d²)"
    falsified_by = "T-independent exponent; no thickness collapse"
    alternatives = "Glazman–Matveev chains; series elements"
    minimum_measurements = "exponent vs T; thickness series"

    def lnJ(self, V, T, d, p, dev):
        V = np.maximum(np.asarray(V, float), 1e-9); T = np.asarray(T, float); d = np.asarray(d, float)
        eps = eps0 * dev.eps_static
        Nc = _nc(dev, T)
        Nt = 10.0 ** p["log10_Nt"]
        l = np.maximum(p["Tc"] / T, 1.001)
        lnMH = ((1 - l) * np.log(e) + np.log(Nc) + (l + 1) * np.log((2 * l + 1) / (l + 1))
                + l * np.log(l * eps / ((l + 1) * Nt)) + (l + 1) * np.log(V) - (2 * l + 1) * np.log(d))
        n0 = np.maximum(Nc * np.exp(-p["Ef0"] / kT(T)), 1e-6)
        lnOhm = np.log(e * n0 * V / d)
        return np.maximum(np.logaddexp(lnMH, lnOhm), LN_FLOOR)

    def checks(self, p, perr, ds, dev, diag):
        out = []
        Tmax = float(ds.temperatures.max())
        if p["Tc"] <= Tmax:
            out.append(Check("warn", f"Tc = {p['Tc']:.0f} K ≤ T_max: l < 1, Mark–Helfrich law not applicable", -2))
        ex = (diag or {}).get("exponent_vs_T")
        if ex and ex.get("slope_vs_invT") is not None:
            if ex["slope_vs_invT"] > 0 and ex.get("significant"):
                out.append(Check("ok", "power-law exponent increases on cooling (l = Tc/T) ✓", +2.5))
            elif ex.get("significant") is False:
                out.append(Check("warn", "power-law exponent does not grow on cooling", -2))
        return out

    def summary(self, p, perr):
        return (f"Tc = {fmt(p['Tc'], perr.get('Tc'), 'K')}, Nt = {10.0 ** p['log10_Nt'] / 1e6:.2e} cm⁻³, "
                f"μ = {10.0 ** p['log10_mu'] * 1e4:.3g} cm²/Vs")
