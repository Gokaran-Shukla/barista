"""Human-readable parameter tables and a ready-to-paste Methods paragraph."""
from __future__ import annotations

import numpy as np

from .models import REGISTRY, short_name

LN10 = np.log(10)

# name -> (unicode symbol, LaTeX symbol, unit, LaTeX unit, transform)
_ID = lambda v, e: (v, e)
_POW10 = lambda v, e: (10 ** v, 10 ** v * LN10 * e if e is not None else None)
_M3_TO_CM3 = lambda v, e: (10 ** v / 1e6, 10 ** v / 1e6 * LN10 * e if e is not None else None)
_M2_TO_CM2 = lambda v, e: (10 ** v * 1e4, 10 ** v * 1e4 * LN10 * e if e is not None else None)
_EV_TO_MEV = lambda v, e: (1e3 * v, 1e3 * e if e is not None else None)

SYMBOLS = {
    "phi": ("Φ", r"\Phi", "eV", "eV", _ID),
    "phit": ("Φt", r"\Phi_t", "eV", "eV", _ID),
    "mstar": ("m*/mₑ", r"m^*/m_e", "", "", _ID),
    "eps_r": ("εr", r"\varepsilon_r", "", "", _ID),
    "Ea": ("Ea", r"E_a", "eV", "eV", _ID),
    "r": ("r", r"r", "nm", "nm", _ID),
    "a": ("a", r"a", "nm", "nm", _ID),
    "lam": ("λ", r"\lambda", "eV", "eV", _ID),
    "sigma": ("σ", r"\sigma", "meV", "meV", _EV_TO_MEV),
    "Sigma": ("Σ", r"\Sigma", "", "", _ID),
    "log10_T0": ("T0", r"T_0", "K", "K", _POW10),
    "p": ("p", r"p", "", "", _ID),
    "xi": ("ξ", r"\xi", "nm", "nm", _ID),
    "Ef0": ("Ec−EF", r"E_c-E_F", "eV", "eV", _ID),
    "Et": ("Ec−Et", r"E_c-E_t", "eV", "eV", _ID),
    "log10_Nt": ("Nt", r"N_t", "cm⁻³", r"cm$^{-3}$", _M3_TO_CM3),
    "log10_mu": ("μ", r"\mu", "cm²/Vs", r"cm$^2$V$^{-1}$s$^{-1}$", _M2_TO_CM2),
    "Tc": ("Tc", r"T_c", "K", "K", _ID),
    "Eg": ("Eg", r"E_g", "eV", "eV", _ID),
    "alpha": ("α", r"\alpha", "K⁻¹", r"K$^{-1}$", _ID),
    "xi_gm": ("ξ", r"\xi", "nm", "nm", _ID),
    "x2": ("log c₂/c₁", r"\log_{10}(c_2/c_1)", "", "", _ID),
    "x3": ("log c₃/c₁", r"\log_{10}(c_3/c_1)", "", "", _ID),
    "log10_s": ("s (area factor)", r"s", "", "", _POW10),
    "log10_Rs": ("Rs", r"R_s", "Ω", r"$\Omega$", _POW10),
}


def _fmt(v, e):
    if v is None or not np.isfinite(v):
        return "n/a"
    if e is None or not np.isfinite(e) or e <= 0:
        return f"{v:.4g}"
    # round error to 2 significant figures and the value to the same decimal
    exp = int(np.floor(np.log10(abs(e)))) - 1
    if abs(v) >= 1e4 or abs(v) < 1e-3:
        m = int(np.floor(np.log10(abs(v)))) if v != 0 else 0
        return f"({v / 10**m:.{max(0, m - exp)}f} ± {e / 10**m:.{max(0, m - exp)}f})×10^{m}"
    dec = max(0, -exp)
    return f"{v:.{dec}f} ± {e:.{dec}f}"


def param_table(res, key=None, fit=None, mech=None):
    """Rows: symbol, value, error, unit, role (free/fixed/prefactor), description."""
    if fit is None:
        from .plots import model_for
        mech, fit = model_for(res, key)
    if mech is None:
        mech = REGISTRY[key]
    rows = []
    for p in mech.params:
        base = p.name.split(".")[-1]
        sym, tex, unit, texunit, tr = SYMBOLS.get(base, (base, base.replace("_", r"\_"), p.unit, p.unit, _ID))
        if base.startswith("log10_") and base not in SYMBOLS:
            sym, tex, unit, texunit, tr = (f"prefactor ({base[6:]})", r"C", p.unit, p.unit, _POW10)
        channel = ""
        if p.name.startswith(("a.", "b.")):
            channel = p.name[0].upper()
            sym = f"{sym} [{channel}]"
        v = fit.params.get(p.name)
        if v is None:
            continue
        e = fit.errors.get(p.name)
        vv, ee = tr(v, e)
        if p.name in fit.fixed:
            role = "fixed"
        elif p.profile or base in ("log10_C", "log10_s0", "log10_G", "log10_c1", "rel"):
            role = "prefactor"
        else:
            role = "free"
        degenerate = any(p.name in (dg[0], dg[1]) for dg in fit.degenerate)
        rows.append({"param": p.name, "symbol": sym, "latex": tex, "channel": channel, "value": vv, "error": ee,
                     "unit": unit, "latex_unit": texunit, "role": role, "description": p.desc,
                     "value_str": f"{_fmt(vv, ee)} {unit}".strip() + (" (fixed)" if role == "fixed" else ""),
                     "degenerate": degenerate,
                     "sanity": p.sanity})
    return rows


def latex_table(rows, caption, label="tab:barista"):
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\begin{tabular}{l l l}", r"\toprule", r"Parameter & Value & Unit \\", r"\midrule"]
    for r in rows:
        if r["role"] == "prefactor":
            continue
        val = _fmt(r["value"], r["error"])
        if "×10^" in val:
            mant, ex = val.split("×10^")
            val = "$" + mant.replace("±", r"\pm") + r"\times 10^{" + ex + "}$"
        else:
            val = "$" + val.replace("±", r"\pm") + "$"
        fixed = r" (fixed)" if r["role"] == "fixed" else ""
        ch = f" ({r['channel']})" if r.get("channel") else ""
        lines.append(rf"${r['latex']}${ch} & {val}{fixed} & {r['latex_unit']} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


REFS = {
    "coherent": "Simmons, J. Appl. Phys. 34, 1793 (1963); Fowler & Nordheim, Proc. R. Soc. A 119, 173 (1928); Stratton, J. Phys. Chem. Solids 23, 1177 (1962)",
    "tat": "Lifshitz & Kirpichenkov, Sov. Phys. JETP 50, 499 (1979); Chiu, Adv. Mater. Sci. Eng. 2014, 578168",
    "glazman_matveev": "Glazman & Matveev, Sov. Phys. JETP 67, 1276 (1988)",
    "band_to_band": "Kane, J. Appl. Phys. 32, 83 (1961)",
    "schottky": "Sze & Ng, Physics of Semiconductor Devices (Wiley, 2007); Tung, Appl. Phys. Rev. 1, 011304 (2014)",
    "poole_frenkel": "Frenkel, Phys. Rev. 54, 647 (1938); Chiu, Adv. Mater. Sci. Eng. 2014, 578168",
    "nn_hopping": "Miller & Abrahams, Phys. Rev. 120, 745 (1960); Shklovskii & Efros, Electronic Properties of Doped Semiconductors (Springer, 1984)",
    "vrh": "Mott, Philos. Mag. 19, 835 (1969); Efros & Shklovskii, J. Phys. C 8, L49 (1975); Zabrodskii & Zinov'eva, Sov. Phys. JETP 59, 425 (1984)",
    "polaron": "Marcus, J. Chem. Phys. 24, 966 (1956); Austin & Mott, Adv. Phys. 18, 41 (1969)",
    "gdm": "Bässler, Phys. Status Solidi B 175, 15 (1993)",
    "sclc_trap": "Mott & Gurney, Electronic Processes in Ionic Crystals (1940); Lampert, Phys. Rev. 103, 1648 (1956)",
    "sclc_exp": "Mark & Helfrich, J. Appl. Phys. 33, 205 (1962)",
    "ohmic_metallic": "Jönsson-Åkerman et al., Appl. Phys. Lett. 77, 1870 (2000)",
}


def methods_paragraph(res):
    cov = res.diag["coverage"]
    m = REGISTRY[res.best.key]
    rows = param_table(res)
    par = "; ".join(f"{r['symbol']} = {r['value_str']}" for r in rows if r["role"] != "prefactor")
    alt = [a for a in res.ranked()[1:4]]
    alt_txt = ", ".join(f"{short_name(a.key)} (ΔAIC = {a.d_aic:.0f})" for a in alt)
    cvm = (res.best.fit.cv_mode or "") + " cross-validation" if res.best.fit.cv_mode else "cross-validation"
    rms = res.best.fit.rms
    desc = f"{short_name(m.key)} [{REFS.get(res.best.key, '')}]"
    if res.mixed and res.assigned_key == res.mixed["key"]:
        A, B = REGISTRY[res.mixed["a"]], REGISTRY[res.mixed["b"]]
        desc = (f"two channels acting in parallel, {short_name(A.key)} [{REFS.get(A.key, '')}] and "
                f"{short_name(B.key)} [{REFS.get(B.key, '')}] (ΔAIC = {-res.mixed['d_aic_vs_best']:.0f} "
                f"relative to the best single mechanism)")
        rms = res.mixed["fit"].rms
    txt = (
        f"The transport mechanism was identified with BARISTA, which implements the diagnostic protocol of "
        f"Shukla (2026). The current–voltage characteristics, measured at {cov['n_T']} temperature(s) "
        f"({cov['T_range'][0]:g}–{cov['T_range'][1]:g} K) and {cov['n_d']} barrier thickness(es) "
        f"({cov['d_range_nm'][0]:g}–{cov['d_range_nm'][1]:g} nm), were first analysed without assuming a mechanism "
        f"(logarithmic derivative γ = d ln I/d ln V, Rowell ratio, Arrhenius activation energy Ea(V), reduced "
        f"activation energy w(T) and thickness-scaling collapse). Thirteen candidate mechanisms were then each fitted "
        f"globally to all {res.best.fit.n} data points with a single parameter set, minimising residuals in ln I. "
        f"Models were compared by the Akaike information criterion, by {cvm}, and by the physical plausibility of every "
        f"fitted parameter. The data are best described by {desc}, "
        f"with {par}; the rms deviation is {100 * (np.exp(rms) - 1):.1f} % in current. "
        f"Competing descriptions — {alt_txt} — are disfavoured. Uncertainties are one standard deviation from the "
        f"covariance matrix of the global fit."
    )
    return txt
