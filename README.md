# BARISTA

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

## Install

```bash
git clone https://github.com/Gokaran-Shukla/barista.git
cd barista
pip install -e ".[all]"        # core + GUI + extra file formats
```

## Use

### Command line

```bash
barista analyze my_data/ --area 100 --n-opt 2.1 -o results
barista analyze iv.xlsx --style nature --font serif --formats pdf,png,svg,eps,tiff
barista analyze run1.csv run2.csv --trap-density 1e18 --eps-r 7 --fix trap_density_cm3
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

## Optional device information

All inputs are optional. By default they are used as **starting guesses and consistency checks**.
For example, the Poole–Frenkel ε_r is compared with n², and the SCLC Nt with your trap density.
They become **hard constraints** only when listed in `--fix`.

`--area --thickness --temperature --layer-spacing --eps-r --n-opt --barrier --mass --band-gap
--trap-density --trap-depth --Nc --mobility --fermi-energy --material --contact-2d --fix`
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

## Licence

MIT © Gokaran Shukla
