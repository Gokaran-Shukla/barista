"""Base classes for transport mechanisms.

Every mechanism returns ``ln J`` (J in A m^-2) for arrays of bias V [V],
temperature T [K] and thickness d [m]; the current is I = J * area.
Working in log space keeps exponentially small or large currents finite.

One parameter of each model may be flagged ``profile=True``: it is a pure
log10 prefactor and is optimised analytically (least squares in ln I),
which makes the non-linear fits faster and more robust.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..constants import kB_eV

LN10 = np.log(10.0)
LN_FLOOR = -700.0     # ln of the smallest current density we keep


@dataclass
class Param:
    name: str
    unit: str
    lo: float
    hi: float
    init: float
    desc: str = ""
    sanity: Optional[tuple] = None        # physically reasonable range (Table 18)
    profile: bool = False                 # log10 prefactor, solved analytically
    device_key: Optional[str] = None      # DeviceInfo attribute used as guess / constraint
    device_map: Optional[Callable] = None # converts the device value to parameter units
    fixed_default: bool = False           # not identifiable from I(V,T,d) alone: fixed unless freed
    log: bool = False                     # sample starts in log space


@dataclass
class Check:
    level: str          # 'ok' | 'warn' | 'fail'
    message: str
    weight: float = 0.0  # contribution to the decision score (negative = penalty)


class Mechanism:
    key = "base"
    name = "base"
    family = ""
    section = ""
    short = ""
    linearising_plot = ""
    params: list = []
    expensive = False
    # fingerprints from Table 6 (used verbatim in the report)
    T_dependence = ""
    V_dependence = ""
    d_scaling = ""
    key_condition = ""
    # necessary signatures / falsifiers (Table 9)
    necessary = ""
    falsified_by = ""
    alternatives = ""
    minimum_measurements = ""

    # ------------------------------------------------------------------
    def lnJ(self, V, T, d, p: dict, dev) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def checks(self, p: dict, perr: dict, ds, dev, diag) -> list:
        return []

    def regime(self, V, T, d, p: dict, dev) -> Optional[np.ndarray]:
        """Optional per-point sub-regime label (e.g. direct vs Fowler-Nordheim)."""
        return None

    def summary(self, p: dict, perr: dict) -> str:
        return ""

    # helpers -----------------------------------------------------------
    def param(self, name) -> Param:
        for q in self.params:
            if q.name == name:
                return q
        raise KeyError(name)

    def __repr__(self):
        return f"<Mechanism {self.key}>"


# ---------------------------------------------------------------------------
# numerical helpers shared by the models
# ---------------------------------------------------------------------------
def kT(T):
    return kB_eV * np.asarray(T, float)


def ln_sinhc(x):
    """ln(sinh(x)/x), stable for all x >= 0."""
    x = np.abs(np.asarray(x, float))
    small = x < 1e-3
    xs = np.where(small, 1.0, x)
    big = xs + np.log1p(-np.exp(-2 * xs)) - np.log(2 * xs)
    return np.where(small, x * x / 6.0, big)


def softplus(x):
    return np.logaddexp(0.0, x)


def fmt(v, err=None, unit="", digits=3):
    if v is None or not np.isfinite(v):
        return "n/a"
    if err is None or not np.isfinite(err) or err <= 0:
        return f"{v:.{digits}g} {unit}".strip()
    return f"{v:.{digits}g} ± {err:.2g} {unit}".strip()


def groups(T, d):
    """Yield boolean masks, one per unique (T, d) pair."""
    pairs = np.stack([np.asarray(T, float), np.asarray(d, float)], axis=1)
    _, inv = np.unique(pairs, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    for k in range(inv.max() + 1 if inv.size else 0):
        yield inv == k
