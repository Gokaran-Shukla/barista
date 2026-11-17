"""Command-line interface:  barista analyze | synth | mechanisms | gui"""
from __future__ import annotations

import argparse
import os
import sys
import time

from . import __version__


def _device_args(p):
    g = p.add_argument_group("optional device information (all optional; used as guesses, checks or --fix constraints)")
    g.add_argument("--device-file", help="YAML/JSON file with any of the fields below")
    g.add_argument("--area", type=float, dest="area_um2", help="junction area in µm²")
    g.add_argument("--thickness", type=float, dest="thickness_nm", help="thickness (nm) if not in the data")
    g.add_argument("--temperature", type=float, dest="temperature_K", help="temperature (K) if not in the data")
    g.add_argument("--layer-spacing", type=float, dest="layer_spacing_nm", help="interlayer spacing a (nm)")
    g.add_argument("--eps-r", type=float, dest="eps_r", help="static relative permittivity")
    g.add_argument("--n-opt", type=float, dest="n_opt", help="refractive index n (optical ε = n²)")
    g.add_argument("--barrier", type=float, dest="barrier_eV", help="barrier height Φ (eV)")
    g.add_argument("--mass", type=float, dest="mass_ratio", help="tunnelling effective mass m*/mₑ")
    g.add_argument("--band-gap", type=float, dest="band_gap_eV", help="band gap Eg (eV)")
    g.add_argument("--trap-density", type=float, dest="trap_density_cm3", help="trap density Nt (cm⁻³)")
    g.add_argument("--trap-depth", type=float, dest="trap_depth_eV", help="trap depth (eV)")
    g.add_argument("--Nc", type=float, dest="Nc_cm3", help="effective density of states (cm⁻³)")
    g.add_argument("--mobility", type=float, dest="mobility_cm2Vs", help="mobility (cm²/Vs)")
    g.add_argument("--fermi-energy", type=float, dest="fermi_energy_eV", help="electrode Fermi energy (eV)")
    g.add_argument("--material", dest="material_class",
                   help="amorphous_oxide | crystalline_oxide | high_k_oxide | layered_vdw | organic | molecular | "
                        "perovskite | resistive_memory | mtj | nitride | semiconductor")
    g.add_argument("--contact-2d", action="store_true", dest="contact_2d", help="2D channel contact (T^3/2 Richardson)")
    g.add_argument("--layers", type=int, dest="n_layers", help="number of barrier layers N (e.g. CrSb layers); d = N·a")
    g.add_argument("--exchange", type=float, dest="exchange_meV", help="proximity exchange splitting guess (meV)")
    g.add_argument("--ordering-T", type=float, dest="ordering_T_K", help="Néel / Curie temperature (K)")
    g.add_argument("--system", help="preset: graphene/CrSb/graphene, graphite/CrI3/graphite, graphite/CrSBr/graphite")
    g.add_argument("--fix", help="comma-separated inputs to impose as hard constraints, e.g. "
                                 "barrier_eV,trap_density_cm3,mass_ratio")


def _make_device(a):
    from .device import DeviceInfo
    base = {}
    if a.device_file:
        base = DeviceInfo.from_file(a.device_file).__dict__.copy()
    for k in ("area_um2", "thickness_nm", "temperature_K", "layer_spacing_nm", "eps_r", "n_opt", "barrier_eV",
              "mass_ratio", "band_gap_eV", "trap_density_cm3", "trap_depth_eV", "Nc_cm3", "mobility_cm2Vs",
              "fermi_energy_eV", "material_class", "n_layers", "exchange_meV", "ordering_T_K"):
        v = getattr(a, k, None)
        if v is not None:
            base[k] = v
    if a.contact_2d:
        base["contact_2d"] = True
    sysname = getattr(a, "system", None)
    if sysname:
        from .device import SYSTEM_PRESETS
        pre = SYSTEM_PRESETS.get(sysname.lower().replace(" ", ""))
        if pre is None:
            raise SystemExit(f"unknown --system {sysname}; known: {', '.join(SYSTEM_PRESETS)}")
        for k, v in pre.items():
            base.setdefault(k, v)
    if a.fix:
        base["fix"] = [s.strip() for s in a.fix.split(",") if s.strip()]
    return DeviceInfo.from_dict(base)


def cmd_analyze(a):
    from . import io, engine
    from .export import export_all
    from .plots import Style
    dev = _make_device(a)
    t0 = time.time()
    ds = io.load(a.inputs, dev)
    cov = ds.coverage()
    print(f"BARISTA {__version__} — read {cov['n_points']} points in {cov['n_curves']} curves "
          f"({cov['n_T']} T: {cov['T_range'][0]:g}–{cov['T_range'][1]:g} K; {cov['n_d']} d: "
          f"{cov['d_range_nm'][0]:g}–{cov['d_range_nm'][1]:g} nm)")
    for n in ds.notes:
        print("  note:", n)
    mechs = [m.strip() for m in a.mechanisms.split(",")] if a.mechanisms else None
    full_ds = ds
    if "state" in ds.df.columns and ds.df["state"].nunique() >= 2:
        from . import spin as _spin
        st = ds.df["state"].map(_spin._norm_state)
        keep = "P" if (st == "P").any() else st.iloc[0]
        print(f"  magnetic configurations found {sorted(ds.df['state'].unique())}: charge analysis on '{keep}', "
              "spin analysis on the MR")
        ds = io.Dataset(df=ds.df[st == keep].reset_index(drop=True), density=ds.density, area_known=ds.area_known,
                        notes=ds.notes)
    res = engine.analyze(ds, dev, mechanisms=mechs, quick=a.quick, workers=a.workers,
                         progress=(lambda m: print("  ·", m)) if a.verbose else None)
    if not a.no_certainty:
        from . import uncertainty
        print(f"  · uncertainty analysis: {a.bootstrap} bootstrap replicates, profile likelihood, window stability")
        res.certainty = uncertainty.assess(res, B=a.bootstrap, workers=a.workers,
                                           progress=(lambda m: print("  ·", m)) if a.verbose else None)
    out = a.output or "barista_output"
    style = Style(journal=a.style, font=a.font, fontsize=a.fontsize,
                  formats=tuple(f.strip() for f in a.formats.split(",")))
    paths = export_all(res, out, style)
    res.spin = _spin_after_charge(full_ds, dev, res, a, out)
    print("\n" + "=" * 78)
    print(f"MECHANISM : {res.headline}")
    print(f"VERDICT   : {res.confidence}")
    print("=" * 78)
    print(res.statement)
    print("\nRanking:")
    for i, x in enumerate(res.ranked()[:6]):
        print(f"  {i + 1}. {x.key:16s} support {100 * x.prob:6.2f} %  ΔAIC {x.d_aic:9.1f}  {x.verdict}")
    if res.certainty is not None:
        print("\nCERTAINTY :", res.certainty.grade)
        print(res.certainty.summary)
        for pn, r in res.certainty.params.items():
            b = r.get("boot95"); pr = r.get("profile95")
            print(f"   {pn:14s} {r['estimate']:.5g}  boot95 {b and [round(x, 5) for x in b]}  "
                  f"profile95 {pr and [round(x, 5) for x in pr[:2]]}")
    if res.recommendations:
        print("\nNext measurements:")
        for r in res.recommendations:
            print("  -", r)
    print(f"\nResults written to {os.path.abspath(out)}  (report.html, barista_results.xlsx, figures/, tables/)"
          f"  [{time.time() - t0:.1f} s on {res.timing.get('workers', 1)} worker(s)]")


def _spin_after_charge(ds, dev, res, a, out):
    """If the data contain two magnetic configurations, run the spin analysis as well."""
    if "state" not in ds.df.columns or ds.df["state"].nunique() < 2:
        return None
    from . import spin
    from .spin_report import export_spin
    tab = spin.mr_table(ds.df, p_state=getattr(a, "p_state", None), ap_state=getattr(a, "ap_state", None),
                        layer_spacing_nm=dev.layer_spacing_nm, n_layers=dev.n_layers)
    sr = spin.analyse_spin(tab, fixed=_fert_fixed(a), B=getattr(a, "spin_bootstrap", 100), charge_result=res)
    export_spin(sr, os.path.join(out, "spin"))
    print("\nSPIN      :", sr.statement)
    return sr


def _fert_fixed(a):
    fx = {}
    for k in ("rhoN", "rhoF", "lsfF"):
        v = getattr(a, k, None)
        if v is not None:
            fx[k] = v
    return fx


def cmd_spin(a):
    from . import io, spin, engine
    from .spin_report import export_spin
    dev = _make_device(a)
    ds = io.load(a.inputs, dev)
    out = a.output or "barista_spin"
    res = None
    if not a.no_charge and "state" in ds.df.columns:
        # charge mechanism of the parallel / FM state (the review's protocol for spin claims)
        pstate = ds.df["state"].map(spin._norm_state)
        sub = io.Dataset(df=ds.df[pstate == (a.p_state or "P")].copy(), density=ds.density,
                         area_known=ds.area_known, notes=ds.notes)
        if len(sub.df) > 10:
            res = engine.analyze(sub, dev, quick=True, workers=a.workers)
            print(f"charge mechanism of the P state: {res.headline} — {res.confidence}")
    tab = spin.mr_table(ds.df, p_state=a.p_state, ap_state=a.ap_state, layer_spacing_nm=dev.layer_spacing_nm,
                        n_layers=dev.n_layers)
    models = [m.strip() for m in a.models.split(",")] if a.models else None
    sr = spin.analyse_spin(tab, fixed=_fert_fixed(a), models=models, B=a.bootstrap, charge_result=res)
    export_spin(sr, out)
    print(sr.statement)
    print("ranking:", ", ".join(f"{k} {100 * sr.probs[k]:.1f}%" for k in sr.ranking))
    print(f"results in {os.path.abspath(out)}")


def cmd_paper(a):
    from .paper.pipeline import analyse_paper
    dev_over = {}
    for k in ("area_um2", "thickness_nm", "n_opt", "eps_r", "trap_density_cm3", "barrier_eV"):
        v = getattr(a, k, None)
        if v is not None:
            dev_over[k] = v
    pages = [int(x) - 1 for x in a.pages.split(",")] if a.pages else None
    pa = analyse_paper(a.pdf, outdir=a.output or "barista_paper", pages=pages, bootstrap=a.bootstrap,
                       workers=a.workers, quick=a.quick, progress=print if a.verbose else None,
                       dev_override=dev_over)
    print(pa.summary)
    print(f"digitised data, figures and per-dataset reports in {os.path.abspath(a.output or 'barista_paper')}")


def cmd_synth(a):
    from . import synth
    if a.preset:
        df, dev, truth = synth.preset(a.mechanism, noise=a.noise, seed=a.seed)
    else:
        import json
        params = json.loads(a.params) if a.params else None
        T = [float(x) for x in a.T.split(",")]
        d = [float(x) for x in a.d.split(",")]
        V = tuple(float(x) for x in a.V.split(","))
        df = synth.generate(a.mechanism, params, V, T, d, noise=a.noise, seed=a.seed,
                            dev={"area_um2": a.area} if a.area else None)
    df.to_csv(a.output, index=False)
    print(f"wrote {len(df)} rows to {a.output}")


def cmd_mechanisms(a):
    from .models import ALL
    for m in ALL:
        print(f"{m.key:16s} {m.name}\n{'':16s} section {m.section}; fit parameters: "
              + ", ".join(f"{p.name} [{p.unit}]" for p in m.params) + "\n")


def cmd_gui(a):
    import subprocess
    app = os.path.join(os.path.dirname(__file__), "gui_app.py")
    sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", app]))


def main(argv=None):
    p = argparse.ArgumentParser(prog="barista", description="BARISTA — identify the charge-transport mechanism "
                                "across thin barriers from I(V, T, d) data.")
    p.add_argument("--version", action="version", version=f"BARISTA {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    pa = sub.add_parser("analyze", help="identify the mechanism and export results")
    pa.add_argument("inputs", nargs="+", help="files, folders or archives (csv, txt, dat, xlsx, ods, json, h5, mat, npz, zip …)")
    pa.add_argument("-o", "--output", help="output folder (default: barista_output)")
    pa.add_argument("--style", default="aps", help="figure widths: aps | nature | acs | ieee | elsevier | rsc")
    pa.add_argument("--font", default="sans", choices=["sans", "serif"])
    pa.add_argument("--fontsize", type=float, default=8)
    pa.add_argument("--formats", default="pdf,png,svg", help="figure formats, e.g. pdf,png,svg,eps,tiff")
    pa.add_argument("--mechanisms", help="restrict to these mechanism keys (see `barista mechanisms`)")
    pa.add_argument("--quick", action="store_true", help="fewer starts, no two-channel / series tests")
    pa.add_argument("-j", "--workers", type=int, default=None,
                    help="processes on this node (default: all CPUs available to the job; 1 = serial)")
    pa.add_argument("--bootstrap", type=int, default=200, help="bootstrap replicates for the certainty analysis")
    pa.add_argument("--no-certainty", action="store_true", help="skip bootstrap / profile-likelihood analysis")
    pa.add_argument("-v", "--verbose", action="store_true")
    _device_args(pa)
    pa.set_defaults(func=cmd_analyze)
    ps = sub.add_parser("synth", help="write synthetic data with a hidden mechanism")
    ps.add_argument("mechanism")
    ps.add_argument("-o", "--output", default="synthetic.csv")
    ps.add_argument("--preset", action="store_true", help="use the built-in preset (review values)")
    ps.add_argument("--params", help='JSON dict of model parameters')
    ps.add_argument("--V", default="0.01,2", help="Vmin,Vmax")
    ps.add_argument("--T", default="300", help="comma-separated temperatures")
    ps.add_argument("--d", default="10", help="comma-separated thicknesses (nm)")
    ps.add_argument("--area", type=float, help="area µm²")
    ps.add_argument("--noise", type=float, default=0.03)
    ps.add_argument("--seed", type=int, default=0)
    ps.set_defaults(func=cmd_synth)
    pspin = sub.add_parser("spin", help="spin-dependent transport: MR(V,T,N) from P/AP (FM/AFM) data or an MR table")
    pspin.add_argument("inputs", nargs="+")
    pspin.add_argument("-o", "--output")
    pspin.add_argument("--p-state", dest="p_state", help="label of the parallel / FM configuration (default: P)")
    pspin.add_argument("--ap-state", dest="ap_state", help="label of the antiparallel / AFM configuration (default: AP)")
    pspin.add_argument("--models", help="julliere,spin_filter,spin_hopping,proximity,valet_fert")
    pspin.add_argument("--bootstrap", type=int, default=100)
    pspin.add_argument("--no-charge", action="store_true", help="skip the charge-mechanism analysis of the P state")
    pspin.add_argument("--rhoN", type=float, help="spacer resistivity ρ*_N (µΩ cm) for Valet–Fert")
    pspin.add_argument("--rhoF", type=float, help="ferromagnet resistivity ρ*_F (µΩ cm)")
    pspin.add_argument("--lsfF", type=float, help="spin-diffusion length of F (nm)")
    pspin.add_argument("-j", "--workers", type=int, default=None)
    _device_args(pspin)
    pspin.set_defaults(func=cmd_spin)
    pp = sub.add_parser("paper", help="read a published paper (PDF or figure image), digitise its I–V data and "
                                      "fingerprint the mechanism")
    pp.add_argument("pdf")
    pp.add_argument("-o", "--output")
    pp.add_argument("--pages", help="comma-separated page numbers (1-based) to digitise")
    pp.add_argument("--bootstrap", type=int, default=0, help="bootstrap replicates for certainty (0 = skip)")
    pp.add_argument("--quick", action="store_true")
    pp.add_argument("-j", "--workers", type=int, default=None)
    pp.add_argument("-v", "--verbose", action="store_true")
    pp.add_argument("--area", type=float, dest="area_um2")
    pp.add_argument("--thickness", type=float, dest="thickness_nm")
    pp.add_argument("--n-opt", type=float, dest="n_opt")
    pp.add_argument("--eps-r", type=float, dest="eps_r")
    pp.add_argument("--trap-density", type=float, dest="trap_density_cm3")
    pp.add_argument("--barrier", type=float, dest="barrier_eV")
    pp.set_defaults(func=cmd_paper)
    pm = sub.add_parser("mechanisms", help="list the mechanisms and their parameters")
    pm.set_defaults(func=cmd_mechanisms)
    pg = sub.add_parser("gui", help="launch the graphical interface (needs streamlit)")
    pg.set_defaults(func=cmd_gui)
    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
