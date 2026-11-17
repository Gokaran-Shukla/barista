"""Paper → data → mechanism: read a published paper and fingerprint its transport mechanism.

    from barista.paper.pipeline import analyse_paper
    pa = analyse_paper("paper.pdf", outdir="paper_out")

Steps
-----
1. Text: title, DOI, abstract, the mechanism the authors claim (keyword scoring with the
   abstract and conclusions weighted ×3) and the parameters they report (barrier height,
   trap depth, ε_r, n, N_t, m*, thickness, area, temperatures).
2. Figures: every vector plot is digitised exactly; embedded bitmaps (or a PNG/JPG you pass
   instead of a PDF) are digitised by OCR + colour segmentation.
3. Every plot whose axes are current (or current density) vs bias (or field) becomes an
   I(V, T, d) dataset; temperature / thickness come from the legend.  Experimental markers
   are preferred over lines (lines are usually the authors' fits).
4. BARISTA's full protocol (+ optional bootstrap certainty) runs on each dataset.
5. The verdict is compared with the authors' claim and their reported parameters.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import io as bio, engine
from ..device import DeviceInfo
from ..io import tokens_from_text
from ..models import REGISTRY, short_name
from . import text as ptext
from .vector import digitise_pdf
from .raster import digitise_raster_pdf, digitise_image_file

FAMILY = {k: m.family for k, m in REGISTRY.items()}


def _unit_factor(label, kind):
    l = label.replace(" ", "").replace("−", "-").replace("^", "").lower()
    if kind == "V":
        if "(mv)" in l:
            return 1e-3
        if "(µv)" in l or "(uv)" in l:
            return 1e-6
        return 1.0
    if kind == "I":
        for pre, f in (("(pa)", 1e-12), ("(na)", 1e-9), ("(µa)", 1e-6), ("(ua)", 1e-6), ("(ma)", 1e-3), ("(fa)", 1e-15)):
            if pre in l:
                return f
        return 1.0
    if kind == "J":       # -> A/m^2
        f = 1.0
        if re.search(r"ma/?cm", l) or "macm" in l:
            return 10.0
        if re.search(r"µa/?cm|ua/?cm", l):
            return 1e-2
        if re.search(r"na/?cm", l):
            return 1e-5
        if "cm" in l:
            return 1e4
        return f
    if kind == "F":       # -> V/m
        if "mv/cm" in l or "mvcm" in l:
            return 1e8
        if "kv/cm" in l:
            return 1e5
        if "v/cm" in l or "vcm" in l:
            return 1e2
        if "v/nm" in l:
            return 1e9
        if "mv/m" in l:
            return 1e6
        return 1.0
    return 1.0


def classify_axis(label, axis):
    l = label.lower()
    if axis == "x":
        if re.search(r"1000\s*/\s*t|1/t|t\s*-1|t\^?-1", l):
            return "invT"
        if re.search(r"\bfield\b|\be\s*\(|\bf\s*\(|mv/cm|v/cm|v/nm", l) and "bias" not in l:
            return "F"
        if re.search(r"\bv\b|bias|voltage|\(v\)|\(mv\)", l) and not re.search(r"1\s*/\s*v|v\s*\^?\s*1/2|v1/2|√|sqrt", l):
            return "V"
        if re.search(r"\bt\b.*\(k\)|temperature", l):
            return "T"
        return None
    if re.search(r"ln|log|/v\^?2|/t\^?2|/f\b|g\s*\(|g/|residual|γ|gamma|mr|tmr", l):
        return None
    if re.search(r"density|\bj\b|a\s*/?\s*cm|a\s*m-2|am-2|a/m", l):
        return "J"
    if re.search(r"current|\bi\b|\(a\)|\(na\)|\(pa\)|\(ma\)|\(µa\)|\(ua\)", l):
        return "I"
    return None


@dataclass
class PaperDataset:
    plot_desc: str
    page: int
    panel: str
    source: str
    df: pd.DataFrame
    labelled: bool
    notes: list = field(default_factory=list)
    result: object = None
    agreement: str = ""
    caption: str = ""
    caption_claim: object = None
    claim_used: object = None


@dataclass
class PaperAnalysis:
    path: str
    title: str
    doi: str
    abstract: str
    claimed: list
    claim_hits: dict
    reported: dict
    device: DeviceInfo
    plots: list
    datasets: list
    summary: str = ""
    benchmark: dict = field(default_factory=dict)


def _series_to_rows(p, s, xq, yq, d_default, T_default, xl, yl):
    lab = s.label or ""
    tok = tokens_from_text(lab)
    T = tok.get("T", T_default)
    d = tok.get("d", d_default)
    x = np.asarray(s.x, float); y = np.asarray(s.y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if xq == "V":
        V = x * _unit_factor(xl, "V")
    elif xq == "F":
        if d is None:
            return None, "field axis but no thickness known"
        V = x * _unit_factor(xl, "F") * d * 1e-9
    else:
        return None, "x axis is not bias"
    if yq == "I":
        I = y * _unit_factor(yl, "I"); col = "I (A)"
    else:
        I = y * _unit_factor(yl, "J"); col = "J (A/m^2)"
    if s.kind == "line" and len(V) > 60:
        idx = np.unique(np.linspace(0, len(V) - 1, 50).astype(int))
        V, I = V[idx], I[idx]
    df = pd.DataFrame({"V (V)": V, col: I, "T (K)": T, "d (nm)": d if d is not None else np.nan,
                       "series": lab or f"series@{np.round(s.color, 2).tolist()}"})
    return df, None


def _captions(path, pages_needed):
    """{page: [(top, x0, text)]} figure captions (handles one- and two-column layouts)."""
    import pdfplumber
    out = {}
    with pdfplumber.open(path) as pdf:
        for i in pages_needed:
            pg = pdf.pages[i]
            caps = []
            try:
                hits = pg.search(r"(?:Figure|FIG\.|Fig\.)\s*\d+[.:|]", regex=True)
            except Exception:
                hits = []
            for h in hits:
                left = h["x0"] < pg.width * 0.45
                wide = h["x0"] < pg.width * 0.2
                x0 = 0 if left else pg.width * 0.48
                x1 = pg.width if wide or not left else pg.width * 0.52
                try:
                    t = pg.crop((x0, h["top"], x1, min(h["top"] + 140, pg.height))).extract_text() or ""
                except Exception:
                    t = ""
                t = re.sub(r"\s+", " ", t)
                m = re.search(r"\.\s+[A-Z][a-z]+\s.*$", t[400:]) if len(t) > 400 else None
                caps.append((h["top"], h["x0"], t[:700]))
            out[i] = caps
    return out


def _caption_for(plot, caps):
    if not caps:
        return ""
    x0, top, x1, bot = plot.bbox
    below = [c for c in caps if c[0] >= bot - 5 and (abs(c[1] - x0) < 120 or c[1] < 100)]
    if below:
        return min(below, key=lambda c: c[0])[2]
    return min(caps, key=lambda c: abs(c[0] - bot))[2]


def analyse_paper(path, outdir=None, pages=None, analyse=True, bootstrap=0, workers=None, quick=False,
                  progress=None, dev_override=None):
    say = progress or (lambda m: None)
    is_pdf = path.lower().endswith(".pdf")
    if is_pdf:
        say("Reading the text")
        txt, page_txt, title = ptext.full_text(path)
    else:
        txt, page_txt, title = "", [], os.path.basename(path)
    claimed, hits = ptext.claimed_mechanism(txt) if txt else ([], {})
    rep = ptext.reported_parameters(txt) if txt else {}
    best = {k: ptext.best_guess(v) for k, v in rep.items() if k != "temperatures_K"}
    devd = {}
    for k in ("area_um2", "barrier_eV", "trap_depth_eV", "eps_r", "n_opt", "trap_density_cm3", "mass_ratio",
              "mobility_cm2Vs"):
        if best.get(k) is not None:
            devd[k] = best[k]
    if best.get("thickness_nm") is not None:
        devd["thickness_nm"] = best["thickness_nm"]
    devd.update(dev_override or {})
    dev = DeviceInfo.from_dict(devd)
    T_default = float(np.median(rep.get("temperatures_K") or [300])) if rep.get("temperatures_K") else 300.0
    if rep.get("temperatures_K") and 300.0 in rep["temperatures_K"]:
        T_default = 300.0

    say("Digitising figures")
    plots = digitise_pdf(path, pages) if is_pdf else []
    vec_pages = {p.page for p in plots}
    if is_pdf:
        try:
            plots += [p for p in digitise_raster_pdf(path, pages) if p.page not in vec_pages]
        except Exception as exc:
            say(f"raster digitisation skipped: {exc}")
    else:
        plots += digitise_image_file(path)

    caps = _captions(path, sorted({p.page for p in plots})) if is_pdf else {}
    for p in plots:
        p.caption = _caption_for(p, caps.get(p.page, []))
    datasets = []
    for p in plots:
        ctok = tokens_from_text(p.caption) if p.caption else {}
        cd = re.search(r"d\s*=\s*([0-9.]+)\s*nm", p.caption or "")
        cT = re.search(r"(?:T\s*=\s*|at\s+)([0-9.]+)\s*K", p.caption or "")
        d_plot = float(cd.group(1)) if cd else (dev.thickness_nm if dev.thickness_nm is not None else ctok.get("d"))
        T_plot = float(cT.group(1)) if cT else T_default
        xq = classify_axis(p.xaxis.label, "x") if p.xaxis.label else None
        yq = classify_axis(p.yaxis.label, "y") if p.yaxis.label else None
        notes = []
        if p.source != "vector" and (xq is None or yq is None):
            if p.xaxis.label == "" and p.yaxis.label == "":
                xq, yq = "V", "I"
                notes.append("raster plot without readable axis titles: assumed current vs bias")
        if xq not in ("V", "F") or yq not in ("I", "J"):
            continue
        ser = [s for s in p.series if s.kind == "markers" and len(s.x) >= 5]
        if any(tokens_from_text(s.label) for s in ser):
            dropped = [s for s in ser if not tokens_from_text(s.label)]
            ser = [s for s in ser if tokens_from_text(s.label)]
            if dropped:
                notes.append(f"{len(dropped)} unlabelled marker group(s) ignored (legend/text artefacts)")
        if not ser:
            ser = [s for s in p.series if s.kind == "line" and len(s.x) >= 5]
            if ser:
                notes.append("no markers: digitised lines were used as data")
        frames = []
        for s in ser:
            df, err = _series_to_rows(p, s, xq, yq, d_plot, T_plot, p.xaxis.label, p.yaxis.label)
            if df is not None:
                frames.append(df)
            elif err:
                notes.append(err)
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        df = df[(df["V (V)"] > 0)]
        labelled = all(bool(tokens_from_text(s.label)) for s in ser)
        if not labelled and len(ser) > 1:
            notes.append("series without T/d labels: each curve is analysed with the same nominal T and d")
        ds_ = PaperDataset(plot_desc=p.describe(), page=p.page, panel=p.panel, source=p.source,
                           df=df, labelled=labelled, notes=notes)
        cc, _ = ptext.claimed_mechanism(p.caption) if p.caption else ([], {})
        ds_.caption_claim = cc[0][0] if cc else None
        ds_.caption = p.caption
        datasets.append(ds_)

    claimed_key = claimed[0][0] if claimed else None
    if analyse:
        for k, dset in enumerate(datasets):
            say(f"Analysing dataset {k + 1}/{len(datasets)} ({dset.plot_desc})")
            raw = dset.df.drop(columns=["series"])
            if raw["d (nm)"].isna().all():
                raw = raw.drop(columns=["d (nm)"])
            try:
                ds = bio.from_dataframe(raw, dev, src=f"page{dset.page + 1}{dset.panel}")
                res = engine.analyze(ds, dev, quick=quick, workers=workers)
                if bootstrap:
                    from .. import uncertainty
                    res.certainty = uncertainty.assess(res, B=bootstrap, workers=workers)
                dset.result = res
                ck = getattr(dset, "caption_claim", None) or claimed_key
                dset.claim_used = ck
                if ck:
                    claimed_key_local = ck
                    got = res.assigned_key or res.best.key
                    parts = got.split("+")
                    if res.regimes and "coherent" in parts:
                        dom = max(res.regimes["summary"], key=res.regimes["summary"].get)
                        parts += {"thermionic emission": ["schottky"], "thermionic–field emission": ["schottky"],
                                  "Fowler–Nordheim": ["coherent"]}.get(dom, [])
                    a = next((x for x in res.assessments if x.key == claimed_key_local), None)
                    if "not identifiable" in res.confidence:
                        dset.agreement = ("CONSISTENT, BUT DATA CANNOT DECIDE" if claimed_key_local in parts or
                                          (a is not None and a.verdict != "EXCLUDED") else "DATA CANNOT DECIDE")
                    elif claimed_key_local in parts:
                        dset.agreement = "AGREES"
                    elif any(FAMILY.get(claimed_key_local) == FAMILY.get(g) for g in parts):
                        dset.agreement = "SAME FAMILY"
                    else:
                        dset.agreement = ("CLAIM NOT EXCLUDED" if a is not None and a.verdict == "CANNOT BE EXCLUDED"
                                          else "DISAGREES")
            except Exception as exc:
                dset.notes.append(f"analysis failed: {exc}")

    pa = PaperAnalysis(path=path, title=title, doi=ptext.doi(txt) if txt else None,
                       abstract=ptext.abstract(txt) if txt else "", claimed=claimed, claim_hits=hits, reported=rep,
                       device=dev, plots=plots, datasets=datasets)
    pa.benchmark = benchmark_summary(pa)
    pa.summary = _summary(pa)
    if outdir:
        export_paper(pa, outdir)
    return pa


def _param_comparison(pa, res):
    rows = []
    if res is None:
        return rows
    p = res.best.fit.params; e = res.best.fit.errors
    rep = {k: ptext.best_guess(v) for k, v in pa.reported.items() if k != "temperatures_K"}
    pairs = [("barrier_eV", "phi"), ("trap_depth_eV", "phit"), ("trap_depth_eV", "Ea"), ("eps_r", "eps_r"),
             ("mass_ratio", "mstar"), ("trap_depth_eV", "Et")]
    for rk, pk in pairs:
        if rep.get(rk) is not None and pk in p:
            v = p[pk]; err = e.get(pk, np.nan)
            ref = rep[rk]
            z = abs(v - ref) / err if np.isfinite(err) and err > 0 else np.nan
            rows.append({"quantity": rk, "reported": ref, "BARISTA": v, "BARISTA_err": err,
                         "relative_difference": (v - ref) / ref if ref else np.nan, "difference_sigma": z})
    if rep.get("trap_density_cm3") is not None and "log10_Nt" in p:
        v = 10 ** p["log10_Nt"] / 1e6
        rows.append({"quantity": "trap_density_cm3", "reported": rep["trap_density_cm3"], "BARISTA": v,
                     "BARISTA_err": np.nan, "relative_difference": (v - rep["trap_density_cm3"]) / rep["trap_density_cm3"],
                     "difference_sigma": np.nan})
    return rows


def benchmark_summary(pa):
    out = {"claimed": pa.claimed[0][0] if pa.claimed else None, "datasets": []}
    for d in pa.datasets:
        r = d.result
        out["datasets"].append({
            "plot": d.plot_desc, "source": d.source, "n_points": int(len(d.df)), "labelled": d.labelled,
            "barista": None if r is None else (r.assigned_key or r.best.key),
            "verdict": None if r is None else r.confidence,
            "certainty": None if r is None or getattr(r, "certainty", None) is None else r.certainty.grade,
            "claim_compared": d.claim_used, "caption": d.caption[:300],
            "agreement": d.agreement, "parameters": _param_comparison(pa, r), "notes": d.notes,
            "traps": None if r is None else _trap_comparison(pa, r)})
    return out


def _trap_comparison(pa, r):
    """BARISTA's trap density vs the value quoted in the paper (if any)."""
    tr = r.traps
    out = {"statement": tr.statement, **{k: v for k, v in tr.consensus.items() if not isinstance(v, str) or len(v) < 80}}
    paper = (pa.reported or {}).get("trap_density_cm3")
    if paper:
        out["paper_Nt_cm3"] = paper
        Nt = tr.consensus.get("Nt_cm3", np.nan)
        if np.isfinite(Nt):
            out["log10_ratio_barista_over_paper"] = float(np.log10(Nt / paper))
            lo, hi = tr.consensus.get("lo_cm3", np.nan), tr.consensus.get("hi_cm3", np.nan)
            out["paper_within_95%"] = bool((not np.isfinite(lo) or paper >= lo) and (not np.isfinite(hi) or paper <= hi))
        elif "upper_bound_cm3" in tr.consensus:
            out["paper_below_bound"] = bool(paper <= tr.consensus["upper_bound_cm3"])
    return out


def _summary(pa):
    lines = [f"{pa.title or os.path.basename(pa.path)}" + (f"  (doi:{pa.doi})" if pa.doi else "")]
    if pa.claimed:
        lines.append("Authors' mechanism (text analysis): " + ", ".join(f"{short_name(k)} [{s}]" for k, s in pa.claimed[:3]))
    lines.append(f"Figures digitised: {len(pa.plots)}; I–V datasets found: {len(pa.datasets)}")
    for d in pa.datasets:
        r = d.result
        if r is None:
            lines.append(f"  • {d.plot_desc}: not analysed ({'; '.join(d.notes)})")
            continue
        cert = getattr(r, "certainty", None)
        lines.append(f"  • {d.plot_desc} [{d.source}]: BARISTA → {r.headline} — {r.confidence}"
                     + (f"; certainty {cert.grade}" if cert else "") + (f"; vs authors ({short_name(d.claim_used)}): {d.agreement}" if d.agreement else ""))
        lines.append(f"      traps: {r.traps.statement}")
    return "\n".join(lines)


def export_paper(pa, outdir):
    from ..export import export_all
    from ..plots import Style
    os.makedirs(outdir, exist_ok=True)
    for k, d in enumerate(pa.datasets):
        sub = os.path.join(outdir, f"dataset{k + 1}_p{d.page + 1}{d.panel}")
        os.makedirs(sub, exist_ok=True)
        d.df.to_csv(os.path.join(sub, "digitised_data.csv"), index=False)
        if d.result is not None:
            export_all(d.result, sub, Style(formats=("pdf", "png", "svg")))
    _digitisation_figure(pa, outdir)
    with open(os.path.join(outdir, "paper_benchmark.json"), "w") as fh:
        json.dump({"title": pa.title, "doi": pa.doi, "claimed": pa.claimed, "claim_evidence": pa.claim_hits,
                   "reported_parameters": {k: (v if k == "temperatures_K" else [(a, b) for a, b in v])
                                           for k, v in pa.reported.items()},
                   "device_used": pa.device.provided(), "benchmark": pa.benchmark, "summary": pa.summary},
                  fh, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o))
    with open(os.path.join(outdir, "paper_summary.txt"), "w") as fh:
        fh.write(pa.summary + "\n")


def _digitisation_figure(pa, outdir):
    """Re-plot every digitised panel (data as markers, lines as lines) for visual verification."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not pa.plots:
        return
    n = len(pa.plots)
    ncol = min(n, 3); nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.7 * nrow), squeeze=False)
    for ax, p in zip(axes.flat, pa.plots):
        for s in p.series:
            c = tuple(min(max(v, 0), 1) for v in s.color)
            if s.kind == "markers":
                ax.plot(s.x, s.y, "o", ms=2.5, mfc="none", color=c, label=s.label or None)
            else:
                ax.plot(s.x, s.y, "-", lw=0.8, color=c, label=s.label or None)
        ax.set_xscale(p.xaxis.scale); ax.set_yscale(p.yaxis.scale)
        ax.set_title(f"p.{p.page + 1} {p.panel} ({p.source})", fontsize=8)
        ax.set_xlabel(p.xaxis.label or "x", fontsize=7); ax.set_ylabel(p.yaxis.label or "y", fontsize=7)
        ax.tick_params(labelsize=6)
        if any(s.label for s in p.series):
            ax.legend(fontsize=5, frameon=False)
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "digitised_figures.png"), dpi=200)
    fig.savefig(os.path.join(outdir, "digitised_figures.pdf"))
    plt.close(fig)
