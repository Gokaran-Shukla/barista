# © 2026 Gokaran Shukla · github.com/Gokaran-Shukla/barista · MIT License  (gs-watermark)
"""Tutorial 1 — blind test: can BARISTA find a hidden mechanism?

Synthetic data are generated from a Poole–Frenkel law with the parameters of Table 8 of the review
(Φt = 0.45 eV, εr = 5, d = 20–40 nm, 5 % noise, 0.5–4 V, 200–350 K). In a narrow bias window at one
temperature, Schottky emission fits just as well (with an unphysical εr ≈ 1). Only the global fit
across temperature and thickness, plus the εr ≈ n² check, separates them.
"""
from barista import synth, io, engine
from barista.export import export_all

df, dev, truth = synth.preset("poole_frenkel", noise=0.05)
print("hidden truth:", truth)
ds = io.from_dataframe(df, dev)
res = engine.analyze(ds, dev)               # uses all cores of this machine
print(res.headline, "→", res.confidence)
print(res.statement)
export_all(res, "tutorial_output/01_blind_test")
