"""Blind identification: synthetic data from a hidden mechanism must be recovered."""
import numpy as np
import pytest

from barista import synth, io, engine

CASES = ["poole_frenkel", "schottky", "coherent", "nn_hopping", "vrh", "sclc_trap", "ohmic_metallic", "polaron"]


@pytest.mark.parametrize("key", CASES)
def test_blind(key):
    df, dev, truth = synth.preset(key, noise=0.05, seed=3)
    ds = io.from_dataframe(df, dev)
    res = engine.analyze(ds, dev, quick=True)
    assert res.best.key == key, [(a.key, round(a.score, 1)) for a in res.ranked()[:4]]
    for k, v in truth.items():
        if k in res.best.fit.params and not k.startswith("log10"):
            assert np.isclose(res.best.fit.params[k], v, rtol=0.1, atol=0.02), (k, res.best.fit.params[k], v)


def test_bias_only_data_is_not_overclaimed():
    df = synth.generate("poole_frenkel", {"phit": 0.45, "eps_r": 5, "log10_C": -2}, (1.5, 2.5), [300], [20],
                        noise=0.05)
    ds = io.from_dataframe(df)
    res = engine.analyze(ds, quick=True)
    assert "IDENTIFIED" not in res.best.verdict
    assert any("thickness" in r for r in res.recommendations)
