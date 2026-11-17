"""Trap density predicted from I–V data, for every trap-related mechanism, against known synthetic truth."""
import numpy as np
import pytest

from barista import synth, io, engine
from barista.device import DeviceInfo


def _run(key, **dev_extra):
    df, dev, truth = synth.preset(key)
    for k, v in dev_extra.items():
        setattr(dev, k, v)
    res = engine.analyze(io.from_dataframe(df, dev), dev, combos=False, series=False)
    return res, truth


@pytest.mark.parametrize("key,expected,energy", [
    ("sclc_trap", 1e17, 0.35),                      # log10_Nt = 23 m⁻³, Ec − Et = 0.35 eV
    ("sclc_exp", 10 ** 17.5, 900 * 8.617e-5),       # total tail density, k_B T_c
    ("nn_hopping", 3 / (4 * np.pi * (1.5e-7) ** 3), 0.12),   # r = 1.5 nm
    ("tat", 1e12 / (1.602e-19 * 1e13) / 1e6, 1.0),  # C = 1e12 A m⁻², ν₀ = 1e13 s⁻¹
])
def test_trap_density_recovered(key, expected, energy):
    res, _ = _run(key)
    c = res.traps.consensus
    assert abs(np.log10(c["Nt_cm3"] / expected)) < 0.15, res.traps.statement
    assert abs(c["trap_energy_eV"] - energy) < 0.05 * energy + 0.01
    assert c["lo_cm3"] <= expected * 1.4 and c["hi_cm3"] >= expected / 1.4


def test_vrh_density_of_states():
    res, _ = _run("vrh")
    g = [t for t in res.traps.estimates if t.mechanism == "vrh"][0].extra["g(E_F)_cm-3eV-1"]
    assert abs(np.log10(g / (21.2 / (8.617e-5 * 1e6 * 1e-21)))) < 0.3


def test_trap_free_gives_bound_and_flags_user_value():
    res, _ = _run("coherent", trap_density_cm3=1e22)
    c = res.traps.consensus
    assert np.isnan(c["Nt_cm3"]) and c["upper_bound_cm3"] < 1e22
    assert c["user_Nt_consistent"] is False
    assert "needs no bulk traps" in res.traps.statement


def test_poole_frenkel_depth_and_mobility_route():
    res, truth = _run("poole_frenkel")
    assert "not identifiable" in res.traps.statement and abs(res.traps.consensus["trap_energy_eV"] - 0.45) < 0.03
    res.device.mobility_cm2Vs = 1e-2
    res._traps_key = None
    est = [t for t in res.traps.estimates if t.method.startswith("Poole–Frenkel prefactor")]
    assert est and np.isfinite(est[0].Nt_cm3)
