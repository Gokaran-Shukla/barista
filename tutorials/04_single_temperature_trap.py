"""Tutorial 4 — why one curve is not enough (review Sec. XVIII, Fig. 7).

The same Poole–Frenkel device is analysed (i) from a single I(V) curve over a narrow bias
window and (ii) from the full temperature and thickness series. With bias alone,
BARISTA refuses to claim an identification and lists the measurements that would decide.
"""
from barista import synth, io, engine

narrow = synth.generate("poole_frenkel", {"phit": 0.45, "eps_r": 5, "log10_C": -2}, (1.5, 2.5), [300], [20],
                        noise=0.05)
r1 = engine.analyze(io.from_dataframe(narrow), quick=True)
print("one curve      :", r1.headline, "→", r1.confidence)
for rec in r1.recommendations:
    print("   next:", rec)

df, dev, _ = synth.preset("poole_frenkel")
r2 = engine.analyze(io.from_dataframe(df, dev), dev, quick=True)
print("full V,T,d set :", r2.headline, "→", r2.confidence)
