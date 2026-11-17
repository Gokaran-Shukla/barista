"""BARISTA benchmark suite.

A. Published systems (review Table 11 and Sec. XXII): I(V, T, d) is regenerated in the
   measurement window of the original study from the parameters the authors reported,
   with 5 % multiplicative noise; BARISTA must recover the published mechanism and
   parameters blind, with bootstrap confidence intervals.  (These are reconstructions from
   published parameters, not the authors' raw data; run `barista paper <pdf>` on the original
   papers for a raw-data benchmark.)
B. Digitiser accuracy: vector and raster round trips of figures with known data.
C. End-to-end paper pipeline: mock papers (text + figure) for five mechanisms, vector and raster.

    python benchmarks/run_benchmarks.py [--quick] [-j N]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from barista import synth, io, engine, uncertainty          # noqa: E402
from barista.device import DeviceInfo                      # noqa: E402
from barista.models import short_name                      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")

PUBLISHED = [
    dict(id="Lenzlinger1969", system="Al/SiO2/Si (thermal oxide)", ref="Lenzlinger & Snow, J. Appl. Phys. 40, 278 (1969)",
         mechanism="coherent", expect_regime="Fowler–Nordheim",
         params={"phi": 3.2, "mstar": 0.42, "log10_s": 0.0}, reported={"phi": 3.2, "mstar": 0.42},
         V=(25, 60), T=[77, 196, 300], d=[40, 50, 60], dev={"area_um2": 1e4, "fermi_energy_eV": 11.6}),
    dict(id="Lee2011", system="Au/1–3 L hBN/Au (CP-AFM)", ref="Lee et al., Appl. Phys. Lett. 99, 243114 (2011)",
         mechanism="coherent", expect_regime="direct tunnelling",
         params={"phi": 3.07, "mstar": 0.5, "log10_s": 0.0}, reported={"phi": 3.07},
         V=(0.02, 0.7), T=[300], d=[0.333, 0.666, 1.0], dev={"area_um2": 1e-3, "mass_ratio": 0.5, "fix": ["mass_ratio"]}),
    dict(id="Wang2003", system="Au/alkanethiol (C8, C12, C16)/Au", ref="Wang, Lee & Reed, Phys. Rev. B 68, 035416 (2003)",
         mechanism="coherent", expect_regime="direct tunnelling",
         params={"phi": 1.42, "mstar": 0.42, "log10_s": 0.0}, reported={"phi": 1.42, "mstar": 0.42},
         V=(0.02, 1.0), T=[80, 150, 230, 300], d=[1.33, 1.84, 2.36], dev={"area_um2": 1.0}),
    dict(id="Choi2010", system="conjugated ONI molecular wires, L > 4 nm", ref="Choi et al., J. Am. Chem. Soc. 132, 4358 (2010)",
         mechanism="nn_hopping", params={"Ea": 0.58, "r": 1.0, "log10_s0": 4.0}, reported={"Ea": 0.58},
         V=(0.05, 1.5), T=[246, 270, 300, 333], d=[5.0, 6.5, 8.0], dev={"area_um2": 0.01}),
    dict(id="LeCorre2021", system="MAPbBr3 single crystal, pulsed SCLC", ref="Le Corre et al., ACS Energy Lett. 6, 1087 (2021)",
         mechanism="sclc_trap", params={"Ef0": 0.75, "log10_Nt": np.log10(1.3e13 * 1e6), "Et": 0.45,
                                         "log10_mu": np.log10(13e-4)},
         reported={"log10_Nt": np.log10(1.3e13 * 1e6), "log10_mu": np.log10(13e-4)},
         V=(0.05, 200), T=[260, 280, 300], d=[1e6, 1.5e6, 2e6], dev={"area_um2": 1e6, "eps_r": 25.5, "Nc_cm3": 2.2e18}),
    dict(id="Lin2024", system="graphite/CrSBr (12–36 nm)/graphite, c axis", ref="Lin et al., Phys. Rev. Research 6, 013185 (2024)",
         mechanism="vrh", params={"log10_T0": 6.3, "p": 0.25, "xi": 0.3, "log10_s0": 1.0}, reported={"p": 0.25},
         V=(0.02, 0.6), T=[30, 60, 100, 150, 220, 300], d=[12, 24, 36], dev={"area_um2": 10, "layer_spacing_nm": 0.79}),
    dict(id="Xu1995", system="a-Si:H tunnel barriers, N = 2 chains", ref="Xu, Ephron & Beasley, Phys. Rev. B 52, 2843 (1995)",
         mechanism="glazman_matveev", params={"x2": 1.5, "x3": -4.0, "xi": 3.0, "log10_c1": 0.0}, reported={},
         V=(0.001, 0.05), T=[4, 8, 16, 32, 64], d=[10, 12, 14], dev={"area_um2": 100}),
    dict(id="Table8_PF", system="Poole–Frenkel test case of review Table 8", ref="Shukla (2026), Table 8",
         mechanism="poole_frenkel", params={"phit": 0.45, "eps_r": 5.0, "log10_C": -2.0},
         reported={"phit": 0.45, "eps_r": 5.0}, V=(0.5, 4.0), T=[200, 250, 300, 350], d=[20],
         dev={"n_opt": 2.236}),
]


def run_published(B, workers, quick):
    rows = []
    for case in PUBLISHED:
        t0 = time.time()
        dev = DeviceInfo.from_dict(case["dev"])
        df = synth.generate(case["mechanism"], case["params"], case["V"], case["T"], case["d"], noise=0.05,
                            dev=dev, seed=11, n_points=30)
        ds = io.from_dataframe(df, dev)
        res = engine.analyze(ds, dev, quick=quick, workers=workers)
        cert = uncertainty.assess(res, B=B, workers=workers) if B else None
        got = res.assigned_key or res.best.key
        regime = ""
        if res.regimes:
            regime = max(res.regimes["summary"], key=res.regimes["summary"].get)
        ok = (got == case["mechanism"]) or (case["mechanism"] in got.split("+"))
        prow = []
        for k, ref in case["reported"].items():
            v = res.best.fit.params.get(k, np.nan)
            ci = cert.params.get(k, {}).get("boot95") if cert else None
            inside = bool(ci and ci[0] <= ref <= ci[1])
            prow.append(f"{k}: {v:.4g} (pub {ref:.4g}; 95% CI {ci[0]:.4g}–{ci[1]:.4g}{' ✓' if inside else ''})" if ci
                        else f"{k}: {v:.4g} (pub {ref:.4g})")
        rows.append({"id": case["id"], "system": case["system"], "reference": case["ref"],
                     "published_mechanism": short_name(case["mechanism"]), "BARISTA": short_name(got),
                     "regime": regime, "correct": ok, "verdict": res.confidence,
                     "certainty": cert.grade if cert else "", "P_select": (cert.selection.get(got, {}).get("freq")
                                                                           if cert else None),
                     "parameters": "; ".join(prow), "runtime_s": round(time.time() - t0, 1)})
        print(f"[A] {case['id']:14s} published={case['mechanism']:15s} BARISTA={got:15s} {'OK ' if ok else 'XX '} "
              f"{res.confidence} | {rows[-1]['certainty']} | {rows[-1]['parameters']}", flush=True)
    return pd.DataFrame(rows)


def run_digitiser():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from barista.paper.vector import digitise_pdf
    from barista.paper.raster import digitise_image_file
    rows = []
    rng = np.random.default_rng(3)
    for trial, (xs, ys) in enumerate([("log", "log"), ("linear", "log"), ("linear", "linear"), ("log", "log")]):
        V = np.geomspace(0.05, 5, 14) if xs == "log" else np.linspace(0.2, 4, 14)
        curves = {T: (1e-9 * np.exp(rng.uniform(0.5, 2)) * V ** rng.uniform(1.2, 2.5) * (T / 300) ** 3) for T in (150, 225, 300)}
        if ys == "linear":
            curves = {T: c / max(map(np.max, curves.values())) * 8 for T, c in curves.items()}
        fig, ax = plt.subplots(figsize=(4, 3))
        for (T, I), col in zip(curves.items(), ["#d62728", "#1f77b4", "#2ca02c"]):
            ax.plot(V, I, "o", color=col, ms=4, label=f"{T} K")
        ax.set_xscale(xs); ax.set_yscale(ys); ax.set_xlabel("Voltage (V)"); ax.set_ylabel("Current (A)"); ax.legend()
        pdfp = os.path.join(OUT, f"digitiser_{trial}.pdf"); pngp = pdfp.replace(".pdf", ".png")
        fig.savefig(pdfp); fig.savefig(pngp, dpi=300); plt.close(fig)
        for mode, plots in (("vector", digitise_pdf(pdfp)), ("raster 300 dpi", digitise_image_file(pngp))):
            errs, found = [], 0
            for p in plots:
                for s in p.series:
                    if s.kind != "markers" or not s.label:
                        continue
                    T = int(float(s.label.split()[0])) if s.label.split()[0].replace(".", "").isdigit() else None
                    if T not in curves:
                        continue
                    found += 1
                    o = np.argsort(s.x)
                    xi, yi = s.x[o], s.y[o]
                    ref = np.interp(xi, V, curves[T])
                    ey = np.abs(np.log(yi / ref)) if ys == "log" else np.abs(yi - ref) / np.ptp(curves[T])
                    errs.append(np.median(ey))
            rows.append({"figure": f"{xs}-{ys} axes", "mode": mode, "series_recovered": f"{found}/3",
                         "median_relative_error_y": float(np.median(errs)) if errs else np.nan})
            print(f"[B] {xs}-{ys} {mode:14s} series {found}/3  median error {rows[-1]['median_relative_error_y']:.2e}",
                  flush=True)
    return pd.DataFrame(rows)


def run_mock_papers(workers):
    from make_mock_paper import make
    from barista.paper.pipeline import analyse_paper
    rows = []
    for key in ["poole_frenkel", "schottky", "coherent", "nn_hopping", "sclc_trap"]:
        for raster in (False, True):
            p = os.path.join(OUT, f"mock_{key}{'_raster' if raster else ''}.pdf")
            make(p, key, raster=raster)
            pa = analyse_paper(p, quick=True, workers=workers)
            d = pa.benchmark["datasets"][0] if pa.benchmark["datasets"] else {}
            rows.append({"hidden": short_name(key), "figure": "raster" if raster else "vector",
                         "claim_read_from_text": short_name(pa.benchmark.get("claimed") or ""),
                         "BARISTA": short_name(d.get("barista") or ""), "agreement": d.get("agreement")})
            print(f"[C] {key:14s} {'raster' if raster else 'vector'}: claim={pa.benchmark.get('claimed')} "
                  f"BARISTA={d.get('barista')} {d.get('agreement')}", flush=True)
    return pd.DataFrame(rows)


def run_spin(B):
    from barista import spin
    rows = []
    # D1: Klein et al., Science 360, 1218 (2018): MR of 95, 300, 550 % for 2, 3, 4 CrI3 layers (published numbers)
    tab = pd.DataFrame({"V": 0.1, "T": 2.0, "N": [2, 3, 4], "d": [1.32, 1.98, 2.64], "MR": [0.95, 3.00, 5.50]})
    sr = spin.analyse_spin(tab, models=["spin_filter", "julliere", "spin_hopping"], B=B)
    f = sr.fits["spin_filter"]
    rows.append({"id": "Klein2018 CrI3 (published MR values)", "published": "coherent spin filter, aΔκ ≈ 0.64–0.78 (review Sec. XXII.B)",
                 "BARISTA": sr.best, "key_parameter": f"aΔκ0 = {f.params['adk0']:.3f} (95% CI {f.ci95.get('adk0', (np.nan,))[0]:.3f}–{f.ci95.get('adk0', (np.nan, np.nan))[1]:.3f})",
                 "correct": sr.best == "spin_filter"})
    # D2: Lin et al. (2024) CrSBr: activation energy 20 meV lower in the FM state -> spin-dependent hopping
    T = np.array([30, 45, 60, 80, 100, 130.0]); rng = np.random.default_rng(1)
    rows_ = []
    for N in (15, 30, 45):
        for t in T:
            for v in (0.05, 0.1, 0.2):
                rows_.append({"V": v, "T": t, "N": N, "d": N * 0.79,
                              "MR": np.expm1(0.020 / (8.617e-5 * t)) * np.exp(0.03 * rng.standard_normal())})
    sr = spin.analyse_spin(pd.DataFrame(rows_), models=["spin_hopping", "spin_filter", "julliere"], B=B)
    f = sr.fits["spin_hopping"]
    rows.append({"id": "Lin2024 CrSBr (ΔEa = 20 meV, reconstructed)", "published": "spin-dependent hopping, ΔEa = 20 meV",
                 "BARISTA": sr.best, "key_parameter": f"ΔEa = {1e3 * f.params['dEa']:.2f} meV (95% CI {1e3 * f.ci95['dEa'][0]:.2f}–{1e3 * f.ci95['dEa'][1]:.2f})",
                 "correct": sr.best == "spin_hopping"})
    # D3: graphene/CrSb/graphene proximity polarisation (hidden parameters)
    df, ptrue = spin.synth_proximity(N_list=(4, 6, 8), T_list=(10, 150, 300))
    sr = spin.analyse_spin(spin.mr_table(df), models=["proximity", "spin_filter", "julliere", "spin_hopping"], B=B)
    f = sr.fits["proximity"]
    rows.append({"id": "Graphene/CrSb/graphene (hidden: Δ=20 meV, μ=15 meV)", "published": "interface proximity polarisation",
                 "BARISTA": sr.best, "key_parameter": f"Δ = {f.params['Delta']:.1f} meV {f.ci95.get('Delta')}, μ = {f.params['mu']:.1f} meV {f.ci95.get('mu')}",
                 "correct": sr.best == "proximity"})
    # D4: Fert–Jaffrès / Valet–Fert: MR vs spacer thickness (hidden: γ=0.6, β=0.4, r_b*=0.1 Ω µm², l_sf=300 nm)
    fx = {"rhoN": 10.0, "rhoF": 20.0, "lsfF": 10.0}
    pv = {"gamma": 0.6, "beta": 0.4, "log10_rb": -1.0, "lsfN": 300.0}
    dN = np.array([5, 20, 50, 100, 200, 400, 800, 1500.0])
    mr = spin.fert_mr(dN, pv, fx) * np.exp(0.02 * rng.standard_normal(len(dN)))
    tab = pd.DataFrame({"V": 0.01, "T": 300.0, "N": 1, "d": dN, "MR": mr})
    sr = spin.analyse_spin(tab, fixed=fx, models=["valet_fert", "julliere", "spin_filter"], B=B)
    f = sr.fits["valet_fert"]
    rows.append({"id": "Fert–Jaffrès F/I/N/I/F (hidden: γ=0.6, l_sf=300 nm)", "published": "spin drift–diffusion",
                 "BARISTA": sr.best, "key_parameter": f"γ = {f.params['gamma']:.3f} {f.ci95.get('gamma')}, l_sf = {f.params['lsfN']:.0f} nm {f.ci95.get('lsfN')}",
                 "correct": sr.best == "valet_fert"})
    for r in rows:
        print(f"[D] {r['id']}: BARISTA={r['BARISTA']} {'OK' if r['correct'] else 'XX'} | {r['key_parameter']}", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("-j", "--workers", type=int, default=None)
    ap.add_argument("-B", "--bootstrap", type=int, default=100)
    ap.add_argument("--only", default="ABC")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    sys.path.insert(0, HERE)
    if "A" in a.only:
        run_published(a.bootstrap, a.workers, a.quick).to_csv(os.path.join(OUT, "A_published_systems.csv"), index=False)
    if "B" in a.only:
        run_digitiser().to_csv(os.path.join(OUT, "B_digitiser.csv"), index=False)
    if "C" in a.only:
        run_mock_papers(a.workers).to_csv(os.path.join(OUT, "C_mock_papers.csv"), index=False)
    if "D" in a.only:
        run_spin(min(a.bootstrap, 60)).to_csv(os.path.join(OUT, "D_spin.csv"), index=False)
    write_markdown(a.bootstrap)


def write_markdown(B):
    """Compose BENCHMARKS.md from every result table present."""
    md = ["# BARISTA benchmark results", "",
          f"Generated {time.strftime('%Y-%m-%d %H:%M')} (bootstrap B = {B} for part A).", "",
          "Reproduce with `python benchmarks/run_benchmarks.py -j <cores>`.", ""]
    f = lambda n: os.path.join(OUT, n)
    if os.path.exists(f("A_published_systems.csv")):
        A = pd.read_csv(f("A_published_systems.csv"))
        md += ["## A. Published systems", "",
               "I(V, T, d) regenerated in each study's measurement window from the parameters the authors "
               "reported (5 % multiplicative noise) and analysed blind. These are reconstructions from published "
               "parameters, not the authors' raw data.", "",
               f"**Correct mechanism: {int(A.correct.sum())}/{len(A)}**", "",
               A.drop(columns=["reference"]).to_markdown(index=False), "", "References:", ""]
        md += [f"- {r.id}: {r.reference}" for r in A.itertuples()] + [""]
    if os.path.exists(f("B_digitiser.csv")):
        md += ["## B. Figure digitiser accuracy (round trip on figures with known data)", "",
               pd.read_csv(f("B_digitiser.csv")).to_markdown(index=False), ""]
    if os.path.exists(f("C_mock_papers.csv")):
        C = pd.read_csv(f("C_mock_papers.csv"))
        md += ["## C. End-to-end: paper PDF → text claim + digitised data → mechanism", "",
               C.to_markdown(index=False), ""]
    if os.path.exists(f("D_spin.csv")):
        D = pd.read_csv(f("D_spin.csv"))
        md += ["## D. Spin-dependent transport", "", f"**Correct model: {int(D.correct.sum())}/{len(D)}**", "",
               D.to_markdown(index=False), ""]
    with open(os.path.join(HERE, "BENCHMARKS.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("wrote", os.path.join(HERE, "BENCHMARKS.md"))


if __name__ == "__main__":
    main()
