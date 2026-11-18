#!/bin/bash
# Build the BARISTA feature presentation (needs TeX Live (XeLaTeX, beamer, tcolorbox) and the Inter font).
set -euo pipefail
cd "$(dirname "$0")"
latexmk -xelatex -interaction=nonstopmode barista_slides.tex
latexmk -c barista_slides.tex
