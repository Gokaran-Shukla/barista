#!/bin/bash
# Build the BARISTA feature presentation (needs TeX Live with beamer + metropolis).
set -euo pipefail
cd "$(dirname "$0")"
latexmk -pdf -interaction=nonstopmode barista_slides.tex
latexmk -c barista_slides.tex
