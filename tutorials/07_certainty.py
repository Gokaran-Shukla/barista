"""Tutorial 7 — how certain is the mechanism? Bootstrap, profile likelihood, stability and residual tests."""
from barista import synth, io, engine, uncertainty
from barista.export import export_all

df, dev, truth = synth.preset("sclc_trap")
res = engine.analyze(io.from_dataframe(df, dev), dev)
res.certainty = uncertainty.assess(res, B=300)       # every core on this node is used
print(res.certainty.summary)
for p, row in res.certainty.params.items():
    print(p, row["estimate"], "boot95", row.get("boot95"), "profile95", row.get("profile95"))
export_all(res, "tutorial_output/07_certainty")
