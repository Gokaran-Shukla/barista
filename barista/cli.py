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
    g.add_argument("--fix", help="comma-separated inputs to impose as hard constraints, e.g. "
                                 "barrier_eV,trap_density_cm3,mass_ratio")


def _make_device(a):
    from .device import DeviceInfo
    base = {}
    if a.device_file:
        base = DeviceInfo.from_file(a.device_file).__dict__.copy()
    for k in ("area_um2", "thickness_nm", "temperature_K", "layer_spacing_nm", "eps_r", "n_opt", "barrier_eV",
              "mass_ratio", "band_gap_eV", "trap_density_cm3", "trap_depth_eV", "Nc_cm3", "mobility_cm2Vs",
              "fermi_energy_eV", "material_class"):
        v = getattr(a, k, None)
        if v is not None:
            base[k] = v
    if a.contact_2d:
        base["contact_2d"] = True
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
    res = engine.analyze(ds, dev, mechanisms=mechs, quick=a.quick, workers=a.workers,
                         progress=(lambda m: print("  ·", m)) if a.verbose else None)
    out = a.output or "barista_output"
    style = Style(journal=a.style, font=a.font, fontsize=a.fontsize,
                  formats=tuple(f.strip() for f in a.formats.split(",")))
    paths = export_all(res, out, style)
    print("\n" + "=" * 78)
    print(f"MECHANISM : {res.headline}")
    print(f"VERDICT   : {res.confidence}")
    print("=" * 78)
    print(res.statement)
    print("\nRanking:")
    for i, x in enumerate(res.ranked()[:6]):
        print(f"  {i + 1}. {x.key:16s} support {100 * x.prob:6.2f} %  ΔAIC {x.d_aic:9.1f}  {x.verdict}")
    if res.recommendations:
        print("\nNext measurements:")
        for r in res.recommendations:
            print("  -", r)
    print(f"\nResults written to {os.path.abspath(out)}  (report.html, barista_results.xlsx, figures/, tables/)"
          f"  [{time.time() - t0:.1f} s on {res.timing.get('workers', 1)} worker(s)]")


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
    pm = sub.add_parser("mechanisms", help="list the mechanisms and their parameters")
    pm.set_defaults(func=cmd_mechanisms)
    pg = sub.add_parser("gui", help="launch the graphical interface (needs streamlit)")
    pg.set_defaults(func=cmd_gui)
    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
