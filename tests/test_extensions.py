"""Certainty analytics, spin-dependent transport and the paper reader."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

from barista import synth, io, engine, uncertainty, spin

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "benchmarks"))


def test_certainty_contains_truth():
    df, dev, truth = synth.preset("poole_frenkel")
    res = engine.analyze(io.from_dataframe(df, dev), dev, combos=False, series=False)
    c = uncertainty.assess(res, B=30)
    assert c.selection["poole_frenkel"]["freq"] > 0.9
    for k in ("phit", "eps_r"):
        lo, hi = c.params[k]["profile95"][:2]
        assert lo - 0.01 * abs(truth[k]) <= truth[k] <= hi + 0.01 * abs(truth[k])
    assert c.grade in ("HIGH", "VERY HIGH")


def test_valet_fert_limits():
    # no spin asymmetry -> plain series resistance
    R = spin.valet_fert([dict(t=1e-9, rho=1e-7, beta=0, lsf=1e-8)] * 2, [dict(rb=1e-15, gamma=0)])
    assert np.isclose(R, 2e-16 + 1e-15)
    # MR decays with spacer thickness beyond l_sf
    mr = spin.fert_mr([10, 1000, 5000], {"gamma": 0.5, "beta": 0.5, "log10_rb": -1, "lsfN": 200},
                      {"rhoN": 10, "rhoF": 20, "lsfF": 10})
    assert mr[0] > mr[1] > mr[2] >= 0


def test_spin_filter_klein_numbers():
    tab = pd.DataFrame({"V": 0.1, "T": 2.0, "N": [2, 3, 4], "d": [1.3, 2.0, 2.6], "MR": [0.95, 3.0, 5.5]})
    sr = spin.analyse_spin(tab, models=["spin_filter", "julliere"], B=10)
    assert sr.best == "spin_filter"
    assert 0.6 < sr.fits["spin_filter"].params["adk0"] < 0.75


def test_proximity_sign_change_detected():
    df, p = spin.synth_proximity(N_list=(4, 6), T_list=(10, 300), noise=0.01)
    tab = spin.mr_table(df)
    fp = spin.fingerprints(tab)
    assert fp["sign_change_with_bias"]


def test_vector_digitiser_exact(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from barista.paper.vector import digitise_pdf
    V = np.geomspace(0.1, 3, 12); I = 1e-9 * V**2
    fig, ax = plt.subplots()
    ax.plot(V, I, "o", label="300 K"); ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Voltage (V)"); ax.set_ylabel("Current (A)"); ax.legend()
    p = tmp_path / "f.pdf"; fig.savefig(p); plt.close(fig)
    plots = digitise_pdf(str(p))
    s = [s for pl in plots for s in pl.series if s.kind == "markers" and s.label][0]
    o = np.argsort(s.x)
    assert np.max(np.abs(np.log(s.y[o] / I))) < 2e-3


def test_mock_paper_end_to_end(tmp_path):
    from make_mock_paper import make
    from barista.paper.pipeline import analyse_paper
    p = str(tmp_path / "mock.pdf")
    make(p, "poole_frenkel")
    pa = analyse_paper(p, quick=True)
    assert pa.claimed[0][0] == "poole_frenkel"
    assert pa.datasets and pa.datasets[0].agreement == "AGREES"
