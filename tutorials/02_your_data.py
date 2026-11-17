"""Tutorial 2 — your own data.

Point `load` at files, folders or archives in any format: csv/txt/dat (any delimiter,
instrument headers), xlsx/ods (all sheets), json, h5, mat, npz, parquet, zip.
Temperature and thickness are read from columns, headers, metadata lines, sheet names
or file names (e.g. IV_300K_5nm.csv).
"""
from barista import io, engine
from barista.device import DeviceInfo
from barista.export import export_all
from barista.plots import Style

dev = DeviceInfo(
    area_um2=100,             # junction area: enables the absolute-magnitude tests
    n_opt=1.65,               # refractive index: decisive Poole–Frenkel / Schottky check
    material_class="amorphous_oxide",
    # trap_density_cm3=1e18, trap_depth_eV=0.8, barrier_eV=1.2, mass_ratio=0.4,
    # fix=["trap_density_cm3"],   # impose an input as a hard constraint
)
ds = io.load(["examples/mixed_channels/AlOx_junctions.xlsx"], dev)
print(ds.coverage())
res = engine.analyze(ds, dev, workers=None)        # None = every core available to this job
export_all(res, "tutorial_output/02_your_data",
           Style(journal="nature", font="sans", formats=("pdf", "png", "svg", "eps")))
print(res.statement)
