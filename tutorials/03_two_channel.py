"""Tutorial 3 — two channels in parallel.

At low temperature (and in the thinnest barrier) coherent tunnelling dominates; in the
thickest barrier at room temperature, activated hopping takes over. No single law fits.
BARISTA screens every pair of candidate channels and returns both parameter sets, and the
channel map (fig5) shows where each channel carries the current.
"""
import numpy as np
from barista import synth, io, engine
from barista.export import export_all

T, d = [10, 77, 150, 225, 300], [2.0, 2.5, 3.0]
a = synth.generate("coherent", {"phi": 1.2, "mstar": 0.4, "log10_s": 0.0}, (0.02, 1.0), T, d,
                   noise=0, dev={"area_um2": 100})
b = synth.generate("nn_hopping", {"Ea": 0.2, "r": 0.3, "log10_s0": 0.3}, (0.02, 1.0), T, d,
                   noise=0, dev={"area_um2": 100})
a["I (A)"] = (a["I (A)"] + b["I (A)"]) * np.exp(0.04 * np.random.default_rng(1).standard_normal(len(a)))
ds = io.from_dataframe(a, io.DeviceInfo(area_um2=100))
res = engine.analyze(ds, io.DeviceInfo(area_um2=100))
print(res.headline, "→", res.confidence)
print(res.statement)
export_all(res, "tutorial_output/03_two_channel")
