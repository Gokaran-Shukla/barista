# © 2026 Gokaran Shukla · github.com/Gokaran-Shukla/barista · MIT License  (gs-watermark)
"""Tutorial 5 — read a published paper.

Pass any paper PDF (or a PNG/JPG of a figure). BARISTA extracts the text claims and reported parameters,
digitises every current–voltage figure, runs the full protocol with bootstrap certainty, and compares
its verdict with the authors'. Here we build a mock paper first so the tutorial runs offline.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
from make_mock_paper import make
from barista.paper.pipeline import analyse_paper

make("tutorial_output/mock_paper.pdf", "poole_frenkel")
pa = analyse_paper("tutorial_output/mock_paper.pdf", outdir="tutorial_output/05_paper", bootstrap=100)
print(pa.summary)
