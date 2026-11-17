# BARISTA benchmark results

Generated 2026-11-17 20:26 (bootstrap B = 100 for part A).

Reproduce with `python benchmarks/run_benchmarks.py -j <cores>`.

## A. Published systems

I(V, T, d) regenerated in each study's measurement window from the parameters the authors reported (5 % multiplicative noise) and analysed blind. These are reconstructions from published parameters, not the authors' raw data.

**Correct mechanism: 8/8**

| id             | system                                     | published_mechanism               | BARISTA                           | regime                    | correct   | verdict    | certainty   |   P_select | parameters                                                                                    |   runtime_s |
|:---------------|:-------------------------------------------|:----------------------------------|:----------------------------------|:--------------------------|:----------|:-----------|:------------|-----------:|:----------------------------------------------------------------------------------------------|------------:|
| Lenzlinger1969 | Al/SiO2/Si (thermal oxide)                 | Coherent tunnelling (direct/FN)   | Coherent tunnelling (direct/FN)   | Fowler–Nordheim           | True      | IDENTIFIED | VERY HIGH   |       1    | phi: 3.216 (pub 3.2; 95% CI 3.172–3.254 ✓); mstar: 0.4131 (pub 0.42; 95% CI 0.3983–0.4309 ✓)  |       187.9 |
| Lee2011        | Au/1–3 L hBN/Au (CP-AFM)                   | Coherent tunnelling (direct/FN)   | Coherent tunnelling (direct/FN)   | direct tunnelling         | True      | IDENTIFIED | VERY HIGH   |       1    | phi: 3.059 (pub 3.07; 95% CI 3.04–3.098 ✓)                                                    |        31.5 |
| Wang2003       | Au/alkanethiol (C8, C12, C16)/Au           | Coherent tunnelling (direct/FN)   | Coherent tunnelling (direct/FN)   | direct tunnelling         | True      | IDENTIFIED | VERY HIGH   |       1    | phi: 1.426 (pub 1.42; 95% CI 1.414–1.439 ✓); mstar: 0.4166 (pub 0.42; 95% CI 0.4129–0.4205 ✓) |       153.3 |
| Choi2010       | conjugated ONI molecular wires, L > 4 nm   | Nearest-neighbour / ionic hopping | Nearest-neighbour / ionic hopping | nan                       | True      | IDENTIFIED | MODERATE    |       0.92 | Ea: 0.58 (pub 0.58; 95% CI 0.5789–0.5813 ✓)                                                   |         8.8 |
| LeCorre2021    | MAPbBr3 single crystal, pulsed SCLC        | SCLC with a discrete trap level   | SCLC with a discrete trap level   | trap-limited SCLC (∝ θV²) | True      | IDENTIFIED | VERY HIGH   |       1    | log10_Nt: 19.1 (pub 19.11; 95% CI 18.92–19.23 ✓); log10_mu: -2.906 (pub -2.886)               |        74.2 |
| Lin2024        | graphite/CrSBr (12–36 nm)/graphite, c axis | Variable-range hopping            | Variable-range hopping            | nan                       | True      | IDENTIFIED | VERY HIGH   |       1    | p: 0.2499 (pub 0.25; 95% CI 0.2447–0.2541 ✓)                                                  |        99.8 |
| Xu1995         | a-Si:H tunnel barriers, N = 2 chains       | Glazman–Matveev chains            | Glazman–Matveev chains            | nan                       | True      | IDENTIFIED | HIGH        |       1    | nan                                                                                           |        38.6 |
| Table8_PF      | Poole–Frenkel test case of review Table 8  | Poole–Frenkel emission            | Poole–Frenkel emission            | nan                       | True      | IDENTIFIED | VERY HIGH   |       1    | phit: 0.4495 (pub 0.45; 95% CI 0.4487–0.4505 ✓); eps_r: 5.024 (pub 5; 95% CI 4.997–5.047 ✓)   |        24.4 |

References:

- Lenzlinger1969: Lenzlinger & Snow, J. Appl. Phys. 40, 278 (1969)
- Lee2011: Lee et al., Appl. Phys. Lett. 99, 243114 (2011)
- Wang2003: Wang, Lee & Reed, Phys. Rev. B 68, 035416 (2003)
- Choi2010: Choi et al., J. Am. Chem. Soc. 132, 4358 (2010)
- LeCorre2021: Le Corre et al., ACS Energy Lett. 6, 1087 (2021)
- Lin2024: Lin et al., Phys. Rev. Research 6, 013185 (2024)
- Xu1995: Xu, Ephron & Beasley, Phys. Rev. B 52, 2843 (1995)
- Table8_PF: Shukla (2026), Table 8

## B. Figure digitiser accuracy (round trip on figures with known data)

| figure             | mode           | series_recovered   |   median_relative_error_y |
|:-------------------|:---------------|:-------------------|--------------------------:|
| log-log axes       | vector         | 3/3                |               2.17074e-05 |
| log-log axes       | raster 300 dpi | 3/3                |               0.00592679  |
| linear-log axes    | vector         | 3/3                |               0.000139599 |
| linear-log axes    | raster 300 dpi | 3/3                |               0.00742005  |
| linear-linear axes | vector         | 3/3                |               1.05953e-05 |
| linear-linear axes | raster 300 dpi | 3/3                |               0.011195    |
| log-log axes       | vector         | 3/3                |               0.000117627 |
| log-log axes       | raster 300 dpi | 3/3                |               0.0192224   |

## C. End-to-end: paper PDF → text claim + digitised data → mechanism

| hidden                            | figure   | claim_read_from_text              | BARISTA                           | agreement                          |
|:----------------------------------|:---------|:----------------------------------|:----------------------------------|:-----------------------------------|
| Poole–Frenkel emission            | vector   | Poole–Frenkel emission            | Poole–Frenkel emission            | AGREES                             |
| Poole–Frenkel emission            | raster   | Poole–Frenkel emission            | SCLC with a discrete trap level   | DISAGREES                          |
| Schottky emission                 | vector   | Schottky emission                 | Schottky emission                 | AGREES                             |
| Schottky emission                 | raster   | Schottky emission                 | Schottky emission                 | AGREES                             |
| Coherent tunnelling (direct/FN)   | vector   | Coherent tunnelling (direct/FN)   | Coherent tunnelling (direct/FN)   | AGREES                             |
| Coherent tunnelling (direct/FN)   | raster   | Coherent tunnelling (direct/FN)   | Coherent tunnelling (direct/FN)   | CONSISTENT, BUT DATA CANNOT DECIDE |
| Nearest-neighbour / ionic hopping | vector   | Nearest-neighbour / ionic hopping | Nearest-neighbour / ionic hopping | AGREES                             |
| Nearest-neighbour / ionic hopping | raster   | Nearest-neighbour / ionic hopping | Nearest-neighbour / ionic hopping | AGREES                             |
| SCLC with a discrete trap level   | vector   | SCLC with a discrete trap level   | SCLC with a discrete trap level   | AGREES                             |
| SCLC with a discrete trap level   | raster   | SCLC with a discrete trap level   | SCLC with a discrete trap level   | CONSISTENT, BUT DATA CANNOT DECIDE |

## D. Spin-dependent transport

**Correct model: 4/4**

| id                                                  | published                                                  | BARISTA      | key_parameter                                                                                             | correct   |
|:----------------------------------------------------|:-----------------------------------------------------------|:-------------|:----------------------------------------------------------------------------------------------------------|:----------|
| Klein2018 CrI3 (published MR values)                | coherent spin filter, aΔκ ≈ 0.64–0.78 (review Sec. XXII.B) | spin_filter  | aΔκ0 = 0.660 (95% CI 0.619–0.713)                                                                         | True      |
| Lin2024 CrSBr (ΔEa = 20 meV, reconstructed)         | spin-dependent hopping, ΔEa = 20 meV                       | spin_hopping | ΔEa = 19.99 meV (95% CI 19.97–20.01)                                                                      | True      |
| Graphene/CrSb/graphene (hidden: Δ=20 meV, μ=15 meV) | interface proximity polarisation                           | proximity    | Δ = 19.3 meV (17.3863701068275, 20.65250313820905), μ = 14.7 meV (13.756082228329298, 15.956420022047226) | True      |
| Fert–Jaffrès F/I/N/I/F (hidden: γ=0.6, l_sf=300 nm) | spin drift–diffusion                                       | valet_fert   | γ = 0.753 (0.5390326320663041, 0.7720574085899419), l_sf = 365 nm (284.0919955750844, 520.8984396097377)  | True      |

