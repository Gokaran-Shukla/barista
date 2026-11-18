# BARISTA — feature presentation

`barista_slides.pdf` is a 36-slide, 16:9 LaTeX Beamer deck covering every BARISTA feature:
data input, the 13 mechanisms and fingerprints, global fitting and verdicts, certainty analytics,
trap-density prediction, spin-dependent transport (graphene/CrSb/graphene, Valet–Fert), the paper
reader, benchmarks, manuscript-ready outputs, one-node parallel runs on Ibex/Shaheen and usage.

Design: custom Beamer theme, Inter typeface for text, math and every figure; equations colour-coded
by physical quantity (bias V green, temperature T purple, thickness d orange, current blue,
densities red, energies brown, spin magenta, operators teal).

## Rebuild

```bash
python docs/presentation/make_figures.py   # regenerate every figure with BARISTA (≈ 5 min)
docs/presentation/build.sh                 # compile with XeLaTeX
```

Requirements: TeX Live with XeLaTeX, beamer, tcolorbox, unicode-math; the Inter font
(`fonts-inter` on Debian/Ubuntu). `fonts_ttf.py` converts Inter to TrueType so matplotlib embeds it cleanly.
