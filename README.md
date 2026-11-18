# BARISTA

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24334682.svg)](https://doi.org/10.5281/zenodo.24334682)

**BARrier Inference of Sequential, Tunnelling & Activated transport**

BARISTA works out *which* charge-transport mechanism carries the current across a thin insulating,
semiconducting, molecular or magnetic layer. It reads your measured I(V) data at one or more
temperatures and thicknesses, fits every candidate mechanism to all of it at once, and tells you
which one wins and why. It reports the extracted parameters with uncertainties, says which next
measurement would settle any remaining ambiguity, and exports manuscript-ready figures and tables.

It implements the theory, fingerprints and diagnostic protocol of

> G. Shukla, *Charge transport mechanisms across thin barriers: theory and experimental fingerprints* (2026).

The guiding principle of that review is built into the code: **a good fit is not proof.** Over one
or two decades of bias, exponentials, power laws and sinh functions all look alike. A mechanism is
therefore assigned only when three things agree:

- the fingerprints (the thickness, temperature and bias dependences);
- a *global* fit across all the data;
- the physical plausibility of every fitted parameter.

---

## What it does

| Step (review Sec. XX) | Implementation |
|---|---|
| 1. Scales and admissibility | data-coverage audit (V, T, d axes); optional device inputs; material-class priors (Table 10) |
| 2. Absolute magnitude | conductance per channel T̄ = G/(N_ch G₀); WKB area factor; Richardson constant within 10^±2 |
| 3. Rowell test | G(T_high)/G(T_low) at low bias: metallic, tunnelling-like or activated |
| 4. Temperature exponent | Zabrodskii w(T) = d ln G/d ln T, exponent p and its decade span |
| 5. Thickness series | ln J vs d (κ), J ∝ dⁿ, scaling collapse: J = f(V/d), J/d = f(V/d²), J = f(V), J·d = f(V) |
| 6. Contacts vs bulk | series-resistance test, rectification, γ-peak scaling (V_TFL ∝ d²) |
| 7. Spectroscopic features | logarithmic derivative γ(V), FN transition voltage, hysteresis, NDR, zero-bias gaps |
| 8. One model, all data | global fits of 13 mechanisms; AIC/BIC with identifiable-parameter counting; cross-validation that extrapolates to held-out T, d or bias; plausibility of every parameter (Table 18); two-channel parallel models |

### Mechanisms

| key | mechanism | parameters |
|---|---|---|
| `coherent` | Tsu–Esaki/WKB through a trapezoidal barrier. One formula covers **direct ↔ Fowler–Nordheim ↔ thermionic–field ↔ thermionic**, and every point gets a sub-regime label | Φ, m*, area factor |
| `tat` | trap-assisted (sequential) tunnelling | Φ, Φt |
| `glazman_matveev` | inelastic chains, N = 1, 2, 3 | ξ, chain weights |
| `band_to_band` | Zener/Kane tunnelling | Eg |
| `schottky` | Richardson–Schottky emission (3D or 2D contact) | Φ_B, ε_r, A* |
| `poole_frenkel` | Poole–Frenkel emission | Φt, ε_r |
| `nn_hopping` | nearest-neighbour / ionic hopping (sinh field law) | Ea, r |
| `vrh` | Mott / Efros–Shklovskii variable-range hopping | T0, p, ξ |
| `polaron` | small-polaron (Marcus) hopping | λ, a |
| `gdm` | Gaussian-disorder (Bässler) hopping | σ, Σ |
| `sclc_trap` | **exact** drift–Poisson SCLC with a discrete trap level (ohmic → θ-limited → TFL → Mott–Gurney) | Nt, Ec−Et, Ec−E_F, μ |
| `sclc_exp` | Mark–Helfrich SCLC with exponential traps | Tc, Nt, μ |
| `ohmic_metallic` | pinholes / filaments | α |

On blind synthetic tests with 5 % noise, every mechanism is recovered, along with its parameters.
This includes the review's Table 8 case: Poole–Frenkel data that Schottky fits equally well at one
temperature, which BARISTA rejects through ε_r ≈ 1 and the global fit. A coherent-tunnelling plus
hopping mixture is also resolved into its two channels.

---

## Benchmarks

See [`benchmarks/BENCHMARKS.md`](benchmarks/BENCHMARKS.md) (reproduce with `python benchmarks/run_benchmarks.py -j <cores>`):

| part | what is tested | result |
|---|---|---|
| A | 8 published systems (review Table 11 / Sec. XXII): Lenzlinger–Snow SiO₂ FN, Lee hBN, Wang alkanethiols, Choi molecular wires, Le Corre MAPbBr₃ SCLC, Lin CrSBr VRH, Xu a-Si GM chains, Table 8 PF. Data regenerated from the reported parameters in the original measurement windows | **8/8 mechanisms**; published parameters inside the bootstrap 95 % CI |
| B | figure digitiser round trip | vector: 10⁻⁵–10⁻⁴ relative error; raster at 300 dpi: 0.6–2 % |
| C | paper PDF → text claim + digitised figure → mechanism, five mechanisms, vector and raster | vector 5/5 agree; raster 4/5 agree or consistent, 1/5 disagrees (overlapping markers) |
| D | spin: Klein 2018 CrI₃ (published MR 95/300/550 %), Lin 2024 CrSBr ΔEa, graphene/CrSb/graphene proximity, Fert–Jaffrès | **4/4**; Klein aΔκ = 0.66 (0.62–0.71), the review's 0.64–0.78 |

Part A uses parameters the authors published, not their raw data (the raw I–V files are not public). To benchmark
against original data, run `barista paper <pdf>` on the papers themselves.

## How certain is the assignment?

`barista analyze` runs a full uncertainty analysis by default (`--bootstrap 200`; `--no-certainty` skips it), in
parallel over all cores:

* **stratified bootstrap** (resampling within each I(V) curve) → probability that each mechanism is selected,
  with a Wilson 95 % interval; distributions of the Akaike weights; bootstrap 68 %/95 % intervals, standard errors
  and bias of every parameter;
* **profile likelihood** intervals (Δχ² = 1 and 3.84), which do not assume a parabolic χ² surface;
* asymptotic (covariance) intervals with Student-t quantiles, and the full correlation matrix;
* **fit-window stability**: refits on each half of the bias range and with each temperature or thickness held
  out, with shifts expressed in σ (review Sec. XVIII.E iii);
* **residual diagnostics**: reduced χ² against the measured scatter, Shapiro–Wilk normality, Durbin–Watson and
  runs tests per curve;
* **model tests**: Akaike weights, BIC posterior probabilities, evidence ratios, Vuong's non-nested test (point-wise
  and curve-blocked), and a likelihood-ratio test for the two-channel model.

These combine into a certainty grade: LOW, MODERATE, HIGH or VERY HIGH. Figures 8–11 and the Excel sheets
*Selection probability*, *Parameter confidence*, *Profile likelihood*, *Window stability*, *Vuong tests* and
*Bootstrap samples* report every number behind the grade.

## Reading a published paper

```bash
barista paper paper.pdf -o paper_out --bootstrap 200      # or a PNG/JPG/TIFF of a figure
```

1. **Text.** BARISTA reads the title, DOI and abstract. It works out which mechanism the authors claim, scoring
   keywords in the whole text with the abstract, conclusions and each figure's caption weighted more heavily. It
   also extracts the parameters the authors report: barrier height, trap depth, ε_r, n, N_t, m*, thickness, area
   and temperatures.
2. **Figures.**
   - Vector plots are recovered exactly: axes are calibrated from the tick labels (including 10ⁿ superscripts),
     labels are snapped to the major ticks, and series are grouped by colour and labelled from the legend.
   - Bitmap figures are read with OCR (including a glyph-level reader for 10ⁿ labels), colour clustering, and
     marker or line extraction.
3. **Analysis.** Every current-vs-bias (or current-vs-field) panel becomes an I(V, T, d) dataset. Temperature and
   thickness come from the legend or the caption, and the authors' markers are preferred over their fit lines.
   The full protocol and certainty analysis then run on each dataset.
4. **Verdict against the authors.** Each dataset is graded AGREES, SAME FAMILY, CLAIM NOT EXCLUDED, DISAGREES or
   DATA CANNOT DECIDE, with a side-by-side comparison of the reported and fitted parameters. The digitised data
   (`digitised_data.csv`) and a re-plot of every digitised panel (`digitised_figures.png`) are exported so you
   can check them against the paper.

## Spin-dependent transport

```bash
barista spin data/ --system graphene/CrSb/graphene --layers 6 --trap-density 1e18
barista analyze data/        # runs automatically when a 'state' column (P/AP, FM/AFM, Néel±) is present
```

MR = G_P/G_AP − 1 is built from the two configurations, or read from an `MR` column. Five models are fitted
globally to MR(V, T, N, d) and ranked by AIC together with their fingerprints:

| model | physics | signature |
|---|---|---|
| `julliere` | Jullière/Slonczewski, P(T) = P0(1−αT^{3/2}), bias roll-off | N-independent, Bloch softening |
| `spin_filter` | coherent filtering by a layered barrier, 1+MR = cosh(N aΔκ) (even N) / cosh(N aΔκ)/cosh(aΔκ) (odd N) | exponential in N, odd/even parity, vanishes at T_N |
| `spin_hopping` | spin-dependent hopping, MR = e^{ΔEa/kT} − 1 | Arrhenius MR, activated P-state current |
| `proximity` | **interface polarisation from the exchange field induced in Dirac electrodes** (graphene/CrSb/graphene): spin-split Dirac cones Δ(T), Fermi level μ, broadening Γ, in series with an altermagnetic barrier whose spin-dependent decay depends on momentum, e^{−σN aΔκ} | MR changes sign with bias near the Dirac point; contact term independent of N plus a bulk term growing with N; follows m_int(T) |
| `valet_fert` | **Valet–Fert / Fert–Jaffrès spin drift–diffusion**, solved exactly for F/I/N/I/F (γ, β, r_b*, l_sf^N; conductivity-mismatch window) | MR decays with spacer thickness |

User inputs that pin the regime:

- `--layers N` sets the number of CrSb (or other) layers, so that d = N·a.
- `--trap-density` estimates the traps met along a tunnelling path and the mean trap spacing, which tells
  coherent, few-trap (TAT / chains) and hopping regimes apart (review Fig. 10).
- `--system` presets supply the interlayer spacing and ordering temperature for graphene/CrSb/graphene,
  graphite/CrI₃/graphite, graphite/CrSBr/graphite and CoFeB/MgO/CoFeB.

## Presentation

A Beamer deck covering every feature, built from BARISTA's own figures: [`docs/presentation/barista_slides.pdf`](docs/presentation/barista_slides.pdf) (source and `build.sh` alongside).

## Install

```bash
git clone https://github.com/Gokaran-Shukla/barista.git
cd barista
pip install -e ".[all]"        # core + GUI + extra file formats + paper reader
```

Reading bitmap figures also needs the Tesseract OCR binary (`apt install tesseract-ocr`, `brew install tesseract`
or `conda install -c conda-forge tesseract`). Vector figures need nothing else.

## Use

### Command line

```bash
barista analyze my_data/ --area 100 --n-opt 2.1 -o results
barista analyze iv.xlsx --style nature --font serif --formats pdf,png,svg,eps,tiff
barista analyze run1.csv run2.csv --trap-density 1e18 --eps-r 7 --fix trap_density_cm3
barista paper article.pdf -o paper_out               # read a published paper and fingerprint it
barista spin cri3_series/ --system graphite/CrI3/graphite
barista mechanisms                          # list mechanisms and their parameters
barista synth poole_frenkel --preset -o pf.csv   # synthetic test data with a hidden answer
```

### Graphical interface

```bash
barista gui
```

### Python

```python
from barista import io, engine
from barista.device import DeviceInfo
from barista.export import export_all

dev = DeviceInfo(area_um2=100, n_opt=2.1)           # every field is optional
ds  = io.load(["data/"], dev)
res = engine.analyze(ds, dev)
print(res.headline, res.confidence)
print(res.statement)
export_all(res, "results")
```

## Input data — any format

* **Containers:**
  - delimited text (`.csv .tsv .txt .dat .prn .asc .iv .log`) with any delimiter, decimal commas, instrument headers (Keithley, Keysight, LabVIEW, Origin) and several sweeps per file
  - spreadsheets (`.xlsx .xlsm .xls .ods`, every sheet)
  - structured data (`.json`, `.h5/.hdf5/.nxs`, MATLAB `.mat`, `.npy/.npz`, `.parquet`, `.feather`, `.pkl`)
  - archives (`.zip`, `.tar.gz`) and folders
* **Layouts:**
  - long tables (V, I, T, d)
  - wide tables (`V`, `I_300K`, `I_77K`, …)
  - paired columns (`V1 I1 V2 I2`)
  - one curve per file, sheet, block or HDF5 group
* **Metadata:** T and d are taken from a column, a column header, an instrument header line
  (`Temperature: 77 K`), a sheet or group name, the file name (`IV_300K_5nm.csv`) or the device
  inputs.
* **Units in headers are honoured:** `I (nA)`, `J (A/cm^2)`, `V (mV)`, `d (Å)`, `T (°C)`, `area (um2)`.

## Trap density from the I–V data

Once the mechanism is assigned, BARISTA predicts the **trap (localised-state) density N_t and the trap energy** from the
I–V data alone. Every route the data allow is used. Each route gives a 95 % interval (bootstrap samples when the
certainty analysis has run) and its assumptions. A support-weighted consensus is reported in the CLI, the
Excel/CSV files (sheets *Trap density* and *Trap summary*), `report.html`, `results.json` and **Fig. 15**.

| Mechanism | Route | Energy |
|---|---|---|
| SCLC, single level | N_t from the exact drift–Poisson fit | E_c − E_t |
| SCLC, any | model-free trap-filled limit N_t = 2ε V_TFL/(e d²) from the γ peak, with the V_TFL ∝ d² test | — |
| SCLC, exponential tail | total N_t (Mark–Helfrich) | k_B T_c |
| Trap-assisted tunnelling | N_t = C/(e ν₀), ν₀ = 10¹²–10¹⁴ s⁻¹ (or `--attempt-freq`) | Φ_t |
| NN / ionic / polaron hopping | site density 3/(4π r³) from the hop length | E_a |
| Variable-range hopping | g(E_F) = 21.2/(k_B T₀ ξ³) and the states in the optimal hopping band | ΔE_h |
| Poole–Frenkel | trap depth; N_t bound (isolated centres), absolute N_t with `--mobility` | Φ_t |
| Coherent tunnelling, Schottky, metallic | no traps needed; 95 % **upper bound** from a parallel TAT channel | — |

A supplied `--trap-density` is checked against the prediction. For papers, the predicted value is compared with the
value the authors quote.

Validation on synthetic data with known traps (`tests/test_traps.py`):

| Mechanism | True N_t (cm⁻³) | Recovered (cm⁻³) | True energy | Recovered energy |
|---|---|---|---|---|
| SCLC, single level | 1.0e17 | 1.0e17 (fit), 9.5e16 (γ peak, model-free) | 0.35 eV | 0.350 eV |
| SCLC, exponential tail | 3.2e17 | 3.2e17 | 77.6 meV | 77.5 meV |
| TAT | 6.2e11 | 6.2e11 | 1.0 eV | 1.0 eV |
| NN hopping | 7.1e19 | 7.1e19 | 0.12 eV | 0.12 eV |
| VRH, g(E_F) | 2.5e20 eV⁻¹ | 2.5e20 eV⁻¹ | — | — |

## Optional device information

All inputs are optional. By default they are used as **starting guesses and consistency checks**.
For example, the Poole–Frenkel ε_r is compared with n², and the SCLC Nt with your trap density.
They become **hard constraints** only when listed in `--fix`.

`--area --thickness --temperature --layer-spacing --eps-r --n-opt --barrier --mass --band-gap
--trap-density --trap-depth --Nc --mobility --attempt-freq --fermi-energy --material --contact-2d --fix`
(or `--device-file device.yaml`).

## Outputs

```
results/
├── report.html               self-contained report (verdict, evidence, figures, tables)
├── summary.txt               verdict + statement + next measurements
├── methods.txt               ready-to-paste Methods paragraph with references
├── parameters.tex            booktabs LaTeX table of the extracted parameters (±1σ)
├── barista_results.xlsx      every table in one workbook
├── results.json              everything, machine readable
├── tables/*.csv              parameters, model comparison, fitted points, model curves,
│                             γ(V), Ea(V), w(T), Rowell, thickness scaling, evidence …
└── figures/                  PDF (embedded TrueType) · PNG/TIFF (600 dpi) · SVG · EPS
    ├── fig1_iv_fit           data + global fit, all T and d
    ├── fig2_linearised       the mechanism's own linearising plots (FN, PF, Arrhenius, collapse …)
    ├── fig3_fingerprints     γ(V), Ea vs √V, Zabrodskii w(T), thickness collapse / Rowell
    ├── fig4_model_comparison ΔAIC of all mechanisms + residuals
    ├── fig5_regime_map       sub-regime or two-channel fraction over (V, T)
    └── fig6_prediction       falsifiable prediction outside the measured window
```

The figures are sized to exact journal column widths (`--style aps|nature|acs|ieee|elsevier|rsc`).
They use 8 pt fonts, inward ticks with minor ticks, (a)/(b) panel labels and no titles. Colours
follow the physical quantity: temperature is a purple ramp, thickness orange and bias green. Every
figure is colour-blind safe and can go straight into a manuscript.

## How the verdict is reached

```
S_m = −½ min(ΔAIC_m, 80) + Σ fingerprint evidence + Σ parameter plausibility + cross-validation
```

The relative support is softmax(S). A violated *necessary* signature (Table 9) or an unphysical
parameter marks a mechanism EXCLUDED. "IDENTIFIED" requires data along at least two independent
axes (bias plus temperature and/or thickness) and a fit at the noise level. With bias-only data,
BARISTA will not claim more than "MOST LIKELY", and it tells you which measurement is missing.

## Citing BARISTA

If BARISTA helps your work, please cite the software:

> G. Shukla, *BARISTA: BARrier Inference of Sequential, Tunnelling and Activated transport*, version 0.3.0 (2026). Zenodo. https://doi.org/10.5281/zenodo.24334682

```bibtex
@software{shukla_barista_2026,
  author    = {Shukla, Gokaran},
  title     = {BARISTA: BARrier Inference of Sequential, Tunnelling and Activated transport},
  version   = {0.3.0},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.24334682},
  url       = {https://github.com/Gokaran-Shukla/barista}
}
```

GitHub's **"Cite this repository"** button (from `CITATION.cff`) gives the same reference in APA and BibTeX.

## Licence

MIT © Gokaran Shukla
