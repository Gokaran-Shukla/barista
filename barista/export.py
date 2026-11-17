"""Export every result: figures, tables (CSV / Excel / LaTeX), JSON and an HTML report."""
from __future__ import annotations

import base64
import datetime as _dt
import html
import json
import os

import numpy as np
import pandas as pd

from . import __version__
from .models import REGISTRY
from .plots import Style, make_all, model_for, predict_I
from .report_text import param_table, latex_table, methods_paragraph, REFS


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, float):
        return None if not np.isfinite(o) else o
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def tables(res):
    """All result tables as DataFrames (shared by CSV, Excel and the HTML report)."""
    ds, dev, data = res.dataset, res.device, res.data
    out = {}
    out["Summary"] = pd.DataFrame([
        ("Mechanism", res.headline),
        ("Verdict", res.confidence),
        ("Relative support", f"{100 * res.best.prob:.1f} %"),
        ("Statement", res.statement),
        ("Trap density", res.traps.statement),
        ("rms deviation in current", f"{100 * (np.exp(res.best.fit.rms) - 1):.2f} %"),
        ("Data points / curves", f"{res.best.fit.n} / {ds.coverage()['n_curves']}"),
        ("Temperatures (K)", ", ".join(f"{t:g}" for t in ds.temperatures)),
        ("Thicknesses (nm)", ", ".join(f"{t:g}" for t in ds.thicknesses)),
        ("Software", f"BARISTA {__version__}"),
        ("Date", _dt.datetime.now().strftime("%Y-%m-%d %H:%M")),
    ], columns=["item", "value"])
    rows = param_table(res)
    out["Parameters"] = pd.DataFrame([{k: r[k] for k in ("symbol", "value", "error", "unit", "role", "description",
                                                          "degenerate", "param")} for r in rows])
    allp = []
    for a in res.ranked():
        for r in param_table(res, a.key):
            allp.append({"mechanism": a.key, **{k: r[k] for k in ("symbol", "value", "error", "unit", "role", "param")}})
    out["Parameters (all models)"] = pd.DataFrame(allp)
    mrows = []
    for i, a in enumerate(res.ranked()):
        f = a.fit
        mrows.append({"rank": i + 1, "key": a.key, "mechanism": a.name, "family": a.family, "verdict": a.verdict,
                      "relative_support_%": round(100 * a.prob, 3), "score": round(a.score, 2),
                      "delta_AIC": round(a.d_aic, 2), "AIC": round(f.aic, 2), "AICc": round(f.aicc, 2),
                      "BIC": round(f.bic, 2), "k_identifiable": f.k_eff, "n": f.n,
                      "rms_%": round(100 * (np.exp(f.rms) - 1), 3),
                      "cv_rms_%": None if f.cv_rms is None else round(100 * (np.exp(f.cv_rms) - 1), 3),
                      "plausibility_fails": "; ".join(c.message for c in a.checks if c.level == "fail"),
                      "summary": REGISTRY[a.key].summary(f.params, f.errors)})
    out["Model comparison"] = pd.DataFrame(mrows)
    ev = []
    for a in res.ranked():
        for w, t in a.evidence:
            ev.append({"mechanism": a.key, "weight": w, "kind": "fingerprint", "text": t})
        for c in a.checks:
            ev.append({"mechanism": a.key, "weight": c.weight, "kind": f"plausibility ({c.level})", "text": c.message})
    out["Evidence"] = pd.DataFrame(ev)
    # fitted points
    mech, fit = model_for(res)
    pts = pd.DataFrame({"V_V": data.V, "T_K": data.T, "d_nm": data.d * 1e9, "area_m2": data.area,
                        "I_exp_A": np.exp(data.lnI), "I_fit_A": np.exp(fit.lnI_fit),
                        "residual_%": 100 * (np.exp(data.lnI - fit.lnI_fit) - 1)})
    if ds.density or ds.area_known:
        pts["J_exp_Acm2"] = pts.I_exp_A / pts.area_m2 * 1e-4
        pts["J_fit_Acm2"] = pts.I_fit_A / pts.area_m2 * 1e-4
    if res.regimes is not None:
        pts["regime"] = res.regimes["labels"]
    if res.mixed:
        pts["fraction_channel_A"] = res.mixed["frac_a"]
    out["Fitted points"] = pts
    # dense model curves incl. extrapolation
    cur = []
    for (T, d, a), g in pts.groupby(["T_K", "d_nm", "area_m2"]):
        Vg = np.geomspace(g.V_V.min() / 5, g.V_V.max() * 2, 200)
        I = predict_I(mech, fit, Vg, T, d, a, dev)
        inside = (Vg >= g.V_V.min()) & (Vg <= g.V_V.max())
        cur.append(pd.DataFrame({"T_K": T, "d_nm": d, "V_V": Vg, "I_model_A": I,
                                 "J_model_Acm2": I / a * 1e-4, "within_measured_window": inside}))
    out["Model curves"] = pd.concat(cur, ignore_index=True) if cur else pd.DataFrame()
    # fingerprints
    dg = res.diag
    g = []
    for c in dg["gamma"]:
        for V, gm in zip(c["V"], c["gamma"]):
            g.append({"T_K": c["T"], "d_nm": c["d"], "V_V": V, "gamma": gm})
    out["Gamma"] = pd.DataFrame(g)
    ar = []
    for a in dg["arrhenius"]:
        t = a["table"]
        for i in range(len(t["V"])):
            ar.append({"d_nm": a["d"], "V_V": t["V"][i], "Ea_eV": t["Ea"][i], "Ea_err_eV": t["Ea_err"][i],
                       "R2": t["r2"][i], "Ea_from_ln(I/T2)_eV": t["Ea_T2"][i]})
    out["Arrhenius Ea(V)"] = pd.DataFrame(ar)
    out["Arrhenius summary"] = pd.DataFrame([{k: v for k, v in a.items() if k != "table"} for a in dg["arrhenius"]])
    zb = []
    for z in dg["zabrodskii"]:
        for T, w in zip(z["T"], z["w"]):
            zb.append({"d_nm": z["d"], "T_K": T, "w": w, "p": z.get("p"), "class": z.get("class")})
    out["Zabrodskii w(T)"] = pd.DataFrame(zb)
    out["Rowell"] = pd.DataFrame(dg["rowell"])
    th = []
    for t in dg["thickness"]:
        row = {k: v for k, v in t.items() if k not in ("collapse",)}
        row["d"] = ", ".join(f"{x:g}" for x in t["d"])
        for k, v in (t.get("collapse") or {}).items():
            row[f"collapse: {k}"] = v
        th.append(row)
    out["Thickness scaling"] = pd.DataFrame(th)
    out["Linearisations R2"] = pd.DataFrame(dg["linearisations"])
    if res.combos:
        out["Two-channel models"] = pd.DataFrame([{
            "model": c["name"], "delta_AIC_vs_best_single": round(c["d_aic_vs_best"], 2),
            "rms_%": round(100 * (np.exp(c["fit"].rms) - 1), 3),
            "points_where_A_matters_%": round(100 * c["share_a"], 1),
            "points_where_B_matters_%": round(100 * c["share_b"], 1),
            **{f"p:{k}": v for k, v in c["fit"].params.items()}} for c in res.combos])
    tr = res.traps
    out["Trap density"] = tr.table()
    out["Trap summary"] = pd.DataFrame([("statement", tr.statement)] +
                                       [(k, v) for k, v in tr.consensus.items()], columns=["item", "value"])
    cert = getattr(res, "certainty", None)
    if cert is not None:
        out["Certainty summary"] = pd.DataFrame([("grade", cert.grade), ("summary", cert.summary),
                                                 ("bootstrap replicates", cert.B), ("workers", cert.workers),
                                                 ("runtime (s)", round(cert.timing, 1))] +
                                                [("note", n) for n in cert.notes], columns=["item", "value"])
        out["Selection probability"] = pd.DataFrame([{
            "model": k, "bootstrap_wins": v["wins"], "P_selected": v["freq"], "P_95CI_low": v["ci_lo"],
            "P_95CI_high": v["ci_hi"], "akaike_weight_full": cert.akaike[k]["weight"],
            "akaike_weight_boot_mean": cert.akaike[k]["boot_mean"], "akaike_weight_boot_2.5%": cert.akaike[k]["boot_lo"],
            "akaike_weight_boot_97.5%": cert.akaike[k]["boot_hi"], "BIC_posterior": cert.bic_posterior[k],
            "evidence_ratio_assigned_vs_model": cert.evidence_ratio[k]} for k, v in cert.selection.items()])
        rows = []
        for pn, r in cert.params.items():
            g = lambda key, i: (r.get(key) or [np.nan, np.nan])[i]
            rows.append({"parameter": pn, "estimate": r["estimate"], "SE_asymptotic": r["se_asymptotic"],
                         "asym_68_lo": g("asym68", 0), "asym_68_hi": g("asym68", 1),
                         "asym_95_lo": g("asym95", 0), "asym_95_hi": g("asym95", 1),
                         "boot_median": r.get("boot_median"), "SE_bootstrap": r.get("boot_se"),
                         "boot_68_lo": g("boot68", 0), "boot_68_hi": g("boot68", 1),
                         "boot_95_lo": g("boot95", 0), "boot_95_hi": g("boot95", 1), "boot_bias": r.get("boot_bias"),
                         "profile_68_lo": g("profile68", 0), "profile_68_hi": g("profile68", 1),
                         "profile_95_lo": g("profile95", 0), "profile_95_hi": g("profile95", 1)})
        out["Parameter confidence"] = pd.DataFrame(rows)
        pl = []
        for pn, pr in cert.profile.items():
            for gv, dc in zip(pr["grid"], pr["dchi2"]):
                pl.append({"parameter": pn, "value": gv, "delta_chi2": dc})
        out["Profile likelihood"] = pd.DataFrame(pl)
        out["Window stability"] = pd.DataFrame(cert.windows)
        out["Vuong tests"] = pd.DataFrame([{"alternative": k, **v} for k, v in cert.vuong.items()])
        out["Residual diagnostics"] = pd.DataFrame([{k: v for k, v in cert.residuals.items() if k != "residuals"}])
        if cert.lrt:
            out["Two-channel LRT"] = pd.DataFrame([cert.lrt])
        bs = cert.boot_samples.get(cert.assigned_key, {})
        if bs:
            L = min(len(v) for v in bs.values())
            out["Bootstrap samples"] = pd.DataFrame({k: v[:L] for k, v in bs.items()})
    out["Recommendations"] = pd.DataFrame({"next measurement": res.recommendations})
    out["Warnings"] = pd.DataFrame({"warning": res.warnings})
    out["Device inputs"] = pd.DataFrame(list(dev.provided().items()), columns=["input", "value"])
    return out


def _img(path):
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode()


def html_report(res, figs: dict, tabs: dict, path: str):
    m = REGISTRY[res.best.key]
    cov = res.diag["coverage"]
    rows = param_table(res)
    esc = html.escape

    def table(df, maxrows=60):
        return df.head(maxrows).to_html(index=False, border=0, classes="t", float_format=lambda x: f"{x:.4g}",
                                        na_rep="")

    figs_html = ""
    captions = {
        "fig1_iv_fit": "Measured I(V) (markers) and the global fit of the identified mechanism (lines) for all temperatures and thicknesses.",
        "fig2_linearised": "The mechanism's own linearising plots, data and fit.",
        "fig3_fingerprints": "Model-free fingerprints: (a) logarithmic derivative γ(V); (b) Arrhenius activation energy vs √V; (c) Zabrodskii w(T); (d) thickness-scaling collapse or Rowell plot.",
        "fig4_model_comparison": "(a) ΔAIC of every candidate mechanism (blue: assigned; red: excluded) with relative support; (b) residuals of the assigned model.",
        "fig5_regime_map": "Sub-regime (or channel fraction) of the assigned model across bias and temperature.",
        "fig6_prediction": "Model prediction outside the measured window (dashed): a falsifiable forecast.",
        "fig7_mixed_fit": "Best two-channel description.",
        "fig15_trap_density": "Trap (localised-state) density from every route the data allow: points with 95 % intervals, arrows are bounds; blue routes belong to the assigned mechanism, the shaded band is the consensus.",
        "fig8_model_certainty": "(a) Probability that each mechanism is selected in the stratified bootstrap (Wilson 95 % intervals); (b) Akaike weights.",
        "fig9_parameter_uncertainty": "Bootstrap distributions and correlations of the extracted parameters with bootstrap and profile-likelihood 95 % intervals.",
        "fig10_profile_likelihood": "Profile likelihood of each parameter; Δχ² = 1 and 3.84 give the 68 % and 95 % intervals.",
        "fig11_stability_residuals": "(a) Parameter shifts when the fit window is changed, in units of σ; (b) normal quantile plot of the residuals.",
    }
    for k, paths in figs.items():
        png = [p for p in paths if p.endswith(".png")]
        if png:
            figs_html += (f'<figure><img src="data:image/png;base64,{_img(png[0])}" alt="{esc(k)}"/>'
                          f"<figcaption><b>{esc(k)}</b> — {esc(captions.get(k, ''))}</figcaption></figure>")
    ev_rows = "".join(f"<li class='{'pos' if w > 0 else 'neg'}'><span>{w:+.1f}</span> {esc(t)}</li>"
                      for w, t in res.best.evidence)
    ck_rows = "".join(f"<li class='{c.level}'><span>{c.level}</span> {esc(c.message)}</li>" for c in res.best.checks)
    par_rows = "".join(f"<tr><td>{esc(r['symbol'])}</td><td>{esc(r['value_str'])}</td><td>{esc(r['description'])}</td>"
                       f"<td>{'⚠ correlated' if r['degenerate'] else ''}</td></tr>" for r in rows)
    rec = "".join(f"<li>{esc(r)}</li>" for r in res.recommendations)
    cert = getattr(res, "certainty", None)
    cert_html = ""
    if cert is not None:
        cert_html = (f"<h2>How certain is this? — {esc(cert.grade)}</h2><div class='card'>{esc(cert.summary)}"
                     + "".join(f"<br/>• {esc(n)}" for n in cert.notes) + "</div>"
                     + "<h3>Probability that each mechanism is selected (stratified bootstrap)</h3><div class='scroll'>"
                     + table(tabs["Selection probability"]) + "</div>"
                     + "<h3>Confidence intervals of the extracted parameters</h3><div class='scroll'>"
                     + table(tabs["Parameter confidence"]) + "</div>"
                     + "<h3>Fit-window stability</h3><div class='scroll'>" + table(tabs["Window stability"]) + "</div>"
                     + "<h3>Non-nested model tests (Vuong)</h3><div class='scroll'>" + table(tabs["Vuong tests"]) + "</div>"
                     + "<h3>Residual diagnostics</h3><div class='scroll'>" + table(tabs["Residual diagnostics"]) + "</div>")
    warn = "".join(f"<li>{esc(w)}</li>" for w in res.warnings)
    badge = {"IDENTIFIED": "#1baf7a", "MOST LIKELY": "#2a78d6"}.get(res.confidence.split(" (")[0], "#eda100")
    doc = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>BARISTA report — {esc(res.headline)}</title>
<style>
:root{{--ink:#1a1a1a;--ink2:#52514e;--line:#e4e3de;--bg:#fcfcfb;--acc:#2a78d6}}
body{{font:15px/1.55 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;color:var(--ink);background:var(--bg);margin:0}}
main{{max-width:1080px;margin:0 auto;padding:32px 24px 80px}}
h1{{font-size:26px;margin:0 0 4px}} h2{{font-size:19px;margin:36px 0 10px;border-bottom:1px solid var(--line);padding-bottom:6px}}
.sub{{color:var(--ink2)}} .badge{{display:inline-block;padding:2px 10px;border-radius:12px;color:#fff;background:{badge};font-weight:600;font-size:13px}}
.card{{background:#fff;border:1px solid var(--line);border-radius:10px;padding:16px 20px;margin:14px 0}}
table.t{{border-collapse:collapse;font-size:13px;width:100%}} .t th,.t td{{text-align:left;padding:4px 8px;border-bottom:1px solid var(--line)}}
.t th{{background:#f4f3ef}} figure{{margin:18px 0}} figure img{{max-width:100%;border:1px solid var(--line);border-radius:6px;background:#fff}}
figcaption{{font-size:13px;color:var(--ink2);margin-top:4px}} ul.ev{{list-style:none;padding:0}} ul.ev li{{padding:3px 0;font-size:14px}}
ul.ev span{{display:inline-block;width:52px;font-variant-numeric:tabular-nums;font-weight:600}} li.pos span,li.ok span{{color:#008300}}
li.neg span,li.fail span{{color:#c0392b}} li.warn span{{color:#b07800}} .scroll{{overflow-x:auto}}
code{{background:#f4f3ef;padding:1px 4px;border-radius:4px}}
</style></head><body><main>
<h1>{esc(res.headline)}</h1>
<div class="sub"><span class="badge">{esc(res.confidence)}</span> &nbsp; relative support {100 * res.best.prob:.1f} % ·
rms {100 * (np.exp(res.best.fit.rms) - 1):.1f} % · {res.best.fit.n} points · {cov['n_T']} T × {cov['n_d']} d ·
BARISTA {__version__} · {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
<div class="card">{esc(res.statement)}</div>
<h2>Extracted parameters</h2>
<table class="t"><tr><th>Parameter</th><th>Value (±1σ)</th><th>Meaning</th><th></th></tr>{par_rows}</table>
<p class="sub">Reference: {esc(REFS.get(res.best.key, ''))}. Diagnostic protocol: G. Shukla, “Charge transport mechanisms across thin barriers: theory and experimental fingerprints” (2026).</p>
<h2>Trap density</h2><div class="card">{esc(res.traps.statement)}</div>
<div class="scroll">{table(tabs['Trap density'][['method','mechanism','type','Nt_cm-3','95%_low','95%_high','trap_energy_eV','energy_meaning','mechanism_support','assumptions']].sort_values('mechanism_support', ascending=False))}</div>
<h2>Why this mechanism</h2>
<div class="card"><b>Fingerprints</b><ul class="ev">{ev_rows or '<li>none</li>'}</ul>
<b>Physical plausibility of the fitted parameters</b><ul class="ev">{ck_rows or '<li>none</li>'}</ul></div>
<h2>Figures</h2>{figs_html}
{cert_html}
<h2>Model comparison</h2><div class="scroll">{table(tabs['Model comparison'][['rank','mechanism','verdict','relative_support_%','delta_AIC','k_identifiable','rms_%','cv_rms_%','plausibility_fails']])}</div>
{('<h2>Two-channel models</h2><div class="scroll">' + table(tabs['Two-channel models'].iloc[:, :5]) + '</div>') if 'Two-channel models' in tabs else ''}
<h2>Recommended next measurements</h2><ul>{rec or '<li>none</li>'}</ul>
<h2>Warnings and notes</h2><ul>{warn or '<li>none</li>'}</ul>
<h2>Methods paragraph</h2><div class="card">{esc(methods_paragraph(res))}</div>
<h2>Fingerprint tables</h2>
<h3>Rowell test</h3><div class="scroll">{table(tabs['Rowell'])}</div>
<h3>Arrhenius summary</h3><div class="scroll">{table(tabs['Arrhenius summary'])}</div>
<h3>Thickness scaling</h3><div class="scroll">{table(tabs['Thickness scaling'])}</div>
</main></body></html>"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(doc)
    return path


def export_all(res, outdir, style: Style = None, formats=None):
    os.makedirs(outdir, exist_ok=True)
    style = style or Style(formats=formats or ("pdf", "png", "svg"))
    paths = {}
    figdir = os.path.join(outdir, "figures")
    if os.path.isdir(figdir):
        for f in os.listdir(figdir):
            if f.startswith("fig"):
                os.remove(os.path.join(figdir, f))
    figs = make_all(res, figdir, style)
    if getattr(res, "certainty", None) is not None:
        from .plots import make_certainty_figs
        figs.update(make_certainty_figs(res, res.certainty, figdir, style))
    paths["figures"] = figs
    tabs = tables(res)
    tdir = os.path.join(outdir, "tables")
    os.makedirs(tdir, exist_ok=True)
    for name, df in tabs.items():
        fn = os.path.join(tdir, name.lower().replace(" ", "_").replace("(", "").replace(")", "")
                          .replace("/", "_") + ".csv")
        df.to_csv(fn, index=False)
    xl = os.path.join(outdir, "barista_results.xlsx")
    with pd.ExcelWriter(xl, engine="openpyxl") as w:
        for name, df in tabs.items():
            df.to_excel(w, sheet_name=name[:31], index=False)
        ws = w.book["Summary"]
        ws.column_dimensions["A"].width = 26
        ws.column_dimensions["B"].width = 120
    paths["excel"] = xl
    tex = os.path.join(outdir, "parameters.tex")
    with open(tex, "w") as fh:
        fh.write("% requires \\usepackage{booktabs}\n")
        fh.write(latex_table(param_table(res), f"Parameters of the assigned transport model ({res.headline}) "
                             f"extracted by a global fit (±1$\\sigma$).", "tab:transport"))
    paths["latex"] = tex
    with open(os.path.join(outdir, "methods.txt"), "w") as fh:
        fh.write(methods_paragraph(res) + "\n")
    js = {
        "software": f"BARISTA {__version__}", "mechanism": res.assigned_key, "name": res.headline,
        "verdict": res.confidence, "relative_support": res.best.prob, "statement": res.statement,
        "parameters": param_table(res), "ranking": [a.as_dict() for a in res.ranked()],
        "two_channel": [{"model": c["name"], "delta_aic": c["d_aic_vs_best"], "params": c["fit"].params}
                        for c in res.combos],
        "series_resistance": None if not res.series else {k: v for k, v in res.series.items() if k not in ("fit", "mech")},
        "regimes": None if not res.regimes else res.regimes["summary"],
        "recommendations": res.recommendations, "warnings": res.warnings, "device": res.device.provided(),
        "coverage": res.diag["coverage"],
        "fingerprints": {k: v for k, v in res.diag.items() if k not in ("gamma", "linearisations")},
        "timing_s": res.timing,
        "traps": {"statement": res.traps.statement, "consensus": res.traps.consensus,
                  "routes": res.traps.table().to_dict(orient="records")},
        "certainty": None if getattr(res, "certainty", None) is None else
        {k: v for k, v in res.certainty.as_dict().items() if k not in ("boot_samples",)},
    }
    jp = os.path.join(outdir, "results.json")
    with open(jp, "w") as fh:
        json.dump(_clean(js), fh, indent=2, ensure_ascii=False)
    paths["json"] = jp
    paths["html"] = html_report(res, figs, tabs, os.path.join(outdir, "report.html"))
    with open(os.path.join(outdir, "summary.txt"), "w") as fh:
        fh.write(f"{res.headline}\nVerdict: {res.confidence}\n\n{res.statement}\n\nTraps: {res.traps.statement}\n\n"
                 + "\n".join(f"- {r}" for r in res.recommendations) + "\n")
    return paths
