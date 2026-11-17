# BARISTA — feature presentation

`barista_slides.pdf` is a 16:9 LaTeX Beamer deck (metropolis theme) covering every BARISTA feature:
input formats, the 13 mechanisms, fingerprints, global fitting and verdicts, certainty analytics,
trap-density prediction, spin-dependent transport (graphene/CrSb/graphene, Valet–Fert), the paper reader,
benchmarks, manuscript-ready outputs, one-node parallel runs on Ibex/Shaheen and usage.

Every figure in `figs/` was produced by BARISTA itself. The equations use one colour per physical quantity:
bias V green, temperature T purple, thickness d orange, current I/J blue, densities N red, energies brown,
spin quantities magenta, and operators teal.

Rebuild after editing: `./build.sh` (or `latexmk -pdf barista_slides.tex`).
