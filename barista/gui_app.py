"""BARISTA graphical interface (Streamlit).   Run with:  barista gui"""
import io as _io
import os
import shutil
import tempfile
import zipfile

import numpy as np
import pandas as pd
import streamlit as st

from barista import __version__, io as bio, engine
from barista.device import DeviceInfo, MATERIAL_PRIORS
from barista.export import export_all, tables
from barista.plots import Style
from barista.models import REGISTRY

st.set_page_config(page_title="BARISTA — transport mechanism identification", layout="wide")
st.title("BARISTA")
st.caption(f"BARrier Inference of Sequential, Tunnelling & Activated transport · v{__version__} · "
           "identify the charge-transport mechanism across thin barriers from I(V, T, d) data")

with st.sidebar:
    mode = st.radio("What do you want to analyse?", ["My I(V) data", "A published paper (PDF or figure image)"])
    if mode.startswith("A published"):
        paper = st.file_uploader("Paper PDF or figure image", type=["pdf", "png", "jpg", "jpeg", "tif", "tiff"])
        pages = st.text_input("Pages to digitise (e.g. 3,4; blank = all)")
        pboot = st.slider("Bootstrap replicates for certainty", 0, 500, 100, 50)
        pgo = st.button("Read paper and fingerprint the mechanism", type="primary", use_container_width=True)
        if paper is None:
            st.info("Upload a paper. Vector figures are recovered exactly; bitmap figures by OCR + colour segmentation.")
            st.stop()
        if pgo:
            from barista.paper.pipeline import analyse_paper
            tmpd = tempfile.mkdtemp(prefix="barista_paper_")
            pth = os.path.join(tmpd, paper.name)
            with open(pth, "wb") as fh:
                fh.write(paper.getbuffer())
            pg = [int(x) - 1 for x in pages.split(",") if x.strip()] if pages.strip() else None
            with st.status("Reading the paper…", expanded=True) as stt:
                pa = analyse_paper(pth, outdir=os.path.join(tmpd, "out"), pages=pg, bootstrap=pboot,
                                   progress=lambda m: stt.write(m))
            st.session_state["paper"] = (pa, os.path.join(tmpd, "out"))
        if "paper" in st.session_state:
            pa, pout = st.session_state["paper"]
            st.subheader(pa.title or "Paper")
            st.text(pa.summary)
            if os.path.exists(os.path.join(pout, "digitised_figures.png")):
                st.image(os.path.join(pout, "digitised_figures.png"), caption="Digitised figures (check against the paper)")
            for d in pa.datasets:
                if d.result is not None:
                    st.markdown(f"**{d.plot_desc}** → {d.result.headline} — *{d.result.confidence}* — vs authors: {d.agreement}")
                    if getattr(d.result, "certainty", None) is not None:
                        st.caption(d.result.certainty.summary)
            buf = _io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                for root, _, fs in os.walk(pout):
                    for f in fs:
                        z.write(os.path.join(root, f), os.path.relpath(os.path.join(root, f), pout))
            st.download_button("Download digitised data + full reports", buf.getvalue(), file_name="barista_paper.zip",
                               type="primary", use_container_width=True)
        st.stop()
    st.header("1 · Data")
    files = st.file_uploader("I(V) files — any format (csv, txt, dat, xlsx, ods, json, h5, mat, npz, zip…)",
                             accept_multiple_files=True)
    st.header("2 · Optional device information")
    st.caption("Leave blank if unknown. Values are used as starting guesses and consistency checks; "
               "tick 'fix' to impose them.")

    def num(label, key, fmt="%g"):
        c1, c2 = st.columns([3, 1])
        v = c1.text_input(label, key=key)
        f = c2.checkbox("fix", key=key + "_fix")
        try:
            return (float(v) if v.strip() else None), f
        except ValueError:
            st.warning(f"'{v}' is not a number")
            return None, False

    fields = {
        "area_um2": "Junction area (µm²)", "thickness_nm": "Thickness d (nm), if not in the data",
        "temperature_K": "Temperature (K), if not in the data", "barrier_eV": "Barrier height Φ (eV)",
        "mass_ratio": "Effective mass m*/mₑ", "eps_r": "Static permittivity εr", "n_opt": "Refractive index n",
        "trap_density_cm3": "Trap density Nt (cm⁻³)", "trap_depth_eV": "Trap depth (eV)",
        "band_gap_eV": "Band gap Eg (eV)", "mobility_cm2Vs": "Mobility μ (cm²/Vs)",
        "layer_spacing_nm": "Interlayer spacing a (nm)",
    }
    devd, fix = {}, []
    for k, lab in fields.items():
        v, f = num(lab, k)
        if v is not None:
            devd[k] = v
            if f:
                fix.append(k)
    mat = st.selectbox("Material class", ["(unknown)"] + list(MATERIAL_PRIORS))
    if mat != "(unknown)":
        devd["material_class"] = mat
    devd["contact_2d"] = st.checkbox("2D channel contact")
    devd["fix"] = fix
    st.header("3 · Figures")
    journal = st.selectbox("Journal column widths", ["aps", "nature", "acs", "ieee", "elsevier", "rsc"])
    font = st.radio("Font", ["sans", "serif"], horizontal=True)
    formats = st.multiselect("Formats", ["pdf", "png", "svg", "eps", "tiff"], default=["pdf", "png", "svg"])
    quick = st.checkbox("Quick mode (skip two-channel and series tests)")
    boot = st.slider("Bootstrap replicates (certainty analysis; 0 = skip)", 0, 1000, 200, 50)
    nl = st.number_input("Number of barrier layers N (e.g. CrSb layers; 0 = not used)", 0, 500, 0)
    if nl:
        devd["n_layers"] = int(nl)
    sysp = st.selectbox("System preset (spin analysis)", ["(none)", "graphene/crsb/graphene", "graphite/cri3/graphite",
                                                           "graphite/crsbr/graphite", "cofeb/mgo/cofeb"])
    if sysp != "(none)":
        from barista.device import SYSTEM_PRESETS
        for k, v in SYSTEM_PRESETS[sysp].items():
            devd.setdefault(k, v)
    mechs = st.multiselect("Restrict to mechanisms (optional)", list(REGISTRY))
    go = st.button("Identify mechanism", type="primary", use_container_width=True)

if not files:
    st.info("Upload one or more data files to begin. A long table (V, I, T, d), a wide table (V, I_300K, I_77K…), "
            "one file per curve (T and d in the file name, a column or a header line) and archives are all accepted.")
    st.stop()

dev = DeviceInfo.from_dict(devd)
tmp = tempfile.mkdtemp(prefix="barista_gui_")
for f in files:
    with open(os.path.join(tmp, f.name), "wb") as fh:
        fh.write(f.getbuffer())
try:
    ds = bio.load(tmp, dev)
except Exception as exc:
    st.error(f"Could not read the data: {exc}")
    st.stop()

cov = ds.coverage()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Points / curves", f"{cov['n_points']} / {cov['n_curves']}")
c2.metric("Temperatures", f"{cov['n_T']}  ({cov['T_range'][0]:g}–{cov['T_range'][1]:g} K)")
c3.metric("Thicknesses", f"{cov['n_d']}  ({cov['d_range_nm'][0]:g}–{cov['d_range_nm'][1]:g} nm)")
c4.metric("Bias range", f"{cov['V_range'][0]:.3g}–{cov['V_range'][1]:.3g} V")
for n in ds.notes:
    st.caption("ℹ " + n)
with st.expander("Parsed data"):
    st.dataframe(ds.df.drop(columns=["curve"]).head(500), use_container_width=True)

if go:
    prog = st.status("Analysing…", expanded=True)
    full = ds
    if "state" in ds.df.columns and ds.df["state"].nunique() >= 2:
        from barista import spin as _spin
        stt_ = ds.df["state"].map(_spin._norm_state)
        ds = bio.Dataset(df=ds.df[stt_ == ("P" if (stt_ == "P").any() else stt_.iloc[0])].reset_index(drop=True),
                         density=ds.density, area_known=ds.area_known, notes=ds.notes)
    res = engine.analyze(ds, dev, mechanisms=mechs or None, quick=quick, progress=lambda m: prog.write(m))
    if boot:
        from barista import uncertainty
        prog.write(f"Certainty analysis ({boot} bootstrap replicates)")
        res.certainty = uncertainty.assess(res, B=boot)
    res.spin = None
    if "state" in full.df.columns and full.df["state"].nunique() >= 2:
        from barista import spin as _spin
        from barista.spin_report import export_spin
        prog.write("Spin-dependent analysis")
        tab = _spin.mr_table(full.df, layer_spacing_nm=dev.layer_spacing_nm, n_layers=dev.n_layers)
        res.spin = _spin.analyse_spin(tab, B=60, charge_result=res)
    out = os.path.join(tmp, "results")
    paths = export_all(res, out, Style(journal=journal, font=font, formats=tuple(formats) or ("png",)))
    if res.spin is not None:
        sp = export_spin(res.spin, os.path.join(out, "spin"))
        paths["figures"].update(sp["figures"])
    prog.update(label="Done", state="complete", expanded=False)
    st.session_state["res"] = (res, out, paths)

if "res" in st.session_state:
    res, out, paths = st.session_state["res"]
    b = res.best
    st.subheader(res.headline)
    st.markdown(f"**{res.confidence}** · relative support **{100 * b.prob:.1f} %** · rms "
                f"{100 * (np.exp(b.fit.rms) - 1):.1f} %")
    st.write(res.statement)
    if getattr(res, "certainty", None) is not None:
        st.success(f"Certainty: {res.certainty.grade}")
        st.write(res.certainty.summary)
    if getattr(res, "spin", None) is not None:
        st.info(res.spin.statement)
    tabs_ = tables(res)
    st.dataframe(tabs_["Parameters"][["symbol", "value", "error", "unit", "role", "description"]],
                 use_container_width=True, hide_index=True)
    for k, ps in paths["figures"].items():
        png = [p for p in ps if p.endswith(".png")]
        if png:
            st.image(png[0], caption=k, use_container_width=True)
    st.markdown("#### Model comparison")
    st.dataframe(tabs_["Model comparison"][["rank", "mechanism", "verdict", "relative_support_%", "delta_AIC",
                                             "rms_%", "cv_rms_%", "plausibility_fails"]],
                 use_container_width=True, hide_index=True)
    if res.recommendations:
        st.markdown("#### Recommended next measurements")
        for r in res.recommendations:
            st.markdown(f"- {r}")
    buf = _io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, fs in os.walk(out):
            for f in fs:
                full = os.path.join(root, f)
                z.write(full, os.path.relpath(full, out))
    st.download_button("Download everything (figures, tables, Excel, LaTeX, JSON, HTML report)", buf.getvalue(),
                       file_name="barista_results.zip", type="primary", use_container_width=True)
    with open(paths["html"], "rb") as fh:
        st.download_button("Download HTML report", fh.read(), file_name="barista_report.html")
