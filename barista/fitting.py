"""Global fits of I(V, T, d) and information-criterion model selection (review Sec. XVIII.E).

* One parameter set per mechanism describes *all* curves simultaneously.
* Residuals are taken in ln I, so every decade of current carries equal weight;
  each curve receives the same total weight whatever its number of points.
* The log10 prefactor is profiled out analytically; the remaining parameters are
  optimised with bounded trust-region least squares from several starts.
* Uncertainties and correlations come from the Jacobian at the optimum.
  The number of *identifiable* parameters (numerical rank of the Jacobian) is
  used in AIC/BIC, so parameters that the data cannot constrain (e.g. an
  activation energy when only one temperature was measured) are not counted.
"""
from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import least_squares

from .models.base import Mechanism, Param, LN10

warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class FitData:
    V: np.ndarray
    T: np.ndarray
    d: np.ndarray          # metres
    area: np.ndarray       # m^2
    lnI: np.ndarray
    curve: np.ndarray
    w: np.ndarray          # residual weights

    @classmethod
    def from_frame(cls, df):
        df = df[(df.V > 0) & (df.I > 0)]
        curve = df["curve"].values
        counts = df.groupby("curve")["curve"].transform("count").values
        w = np.sqrt(counts.mean() / counts)
        return cls(V=df.V.values.astype(float), T=df["T"].values.astype(float), d=df["d"].values * 1e-9,
                   area=df["area"].values.astype(float), lnI=np.log(df.I.values), curve=curve, w=w)

    def subset(self, mask):
        return FitData(*(getattr(self, k)[mask] for k in ("V", "T", "d", "area", "lnI", "curve", "w")))

    @property
    def n(self):
        return len(self.V)


@dataclass
class FitResult:
    key: str
    name: str
    params: dict
    errors: dict
    free: list
    fixed: dict
    n: int
    k: int
    k_eff: int
    rss: float
    rms: float
    aic: float
    aicc: float
    bic: float
    lnI_fit: np.ndarray
    corr: Optional[np.ndarray] = None
    degenerate: list = field(default_factory=list)
    success: bool = True
    message: str = ""
    cv_rms: Optional[float] = None
    cv_mode: Optional[str] = None
    extras: dict = field(default_factory=dict)

    def as_dict(self):
        return {
            "key": self.key, "name": self.name, "params": self.params, "errors": self.errors,
            "free": self.free, "fixed": self.fixed, "n": self.n, "k": self.k, "k_eff": self.k_eff,
            "rss": self.rss, "rms_lnI": self.rms, "factor_rms": float(np.exp(self.rms)), "aic": self.aic,
            "aicc": self.aicc, "bic": self.bic, "degenerate": self.degenerate, "success": self.success,
            "cv_rms_lnI": self.cv_rms, "cv_mode": self.cv_mode, "extras": self.extras,
        }


# ---------------------------------------------------------------------------
def _device_value(p: Param, dev):
    if p.device_key is None:
        return None
    v = getattr(dev, p.device_key, None)
    if v is None:
        return None
    try:
        v = p.device_map(v) if p.device_map else float(v)
    except Exception:
        return None
    return float(np.clip(v, p.lo, p.hi))


def setup(mech: Mechanism, dev, hints=None, free_override=(), fixed_extra=None):
    """Split parameters into free / fixed / profiled and choose starting values."""
    hints = hints or {}
    fixed_extra = fixed_extra or {}
    fixed, free, init = {}, [], {}
    prof = None
    for p in mech.params:
        if p.name in fixed_extra:
            fixed[p.name] = float(fixed_extra[p.name])
            continue
        dv = _device_value(p, dev)
        start = dv if dv is not None else hints.get(p.name, p.init)
        start = float(np.clip(start, p.lo, p.hi))
        if p.profile:
            prof = p
            if p.device_key and dev.is_fixed(p.device_key) and dv is not None:
                fixed[p.name] = dv
                prof = None
            init[p.name] = start
            continue
        if (p.device_key and dev.is_fixed(p.device_key) and dv is not None):
            fixed[p.name] = dv
        elif p.fixed_default and p.name not in free_override:
            fixed[p.name] = dv if dv is not None else p.init
        else:
            free.append(p)
            init[p.name] = start
    return free, fixed, prof, init


def _model_lnI(mech, data, dev, pdict):
    return mech.lnJ(data.V, data.T, data.d, pdict, dev) + np.log(data.area)


def _random_starts(free, n, rng):
    starts = []
    for _ in range(n):
        x = []
        for p in free:
            if p.log and p.lo > 0:
                x.append(np.exp(rng.uniform(np.log(p.lo), np.log(p.hi))))
            else:
                lo, hi = p.lo, p.hi
                if p.sanity is not None:     # sample inside the physically sensible range
                    lo, hi = max(lo, p.sanity[0]), min(hi, p.sanity[1])
                x.append(rng.uniform(lo, hi))
        starts.append(np.array(x))
    return starts


def fit(mech: Mechanism, data: FitData, dev, hints=None, n_starts=None, seed=0, extra_starts=(),
        free_override=(), screen=False, fixed_extra=None, uncertainties=True) -> FitResult:
    free, fixed, prof, init = setup(mech, dev, hints, free_override, fixed_extra)
    names = [p.name for p in free]
    lo = np.array([p.lo for p in free]); hi = np.array([p.hi for p in free])

    def pdict(x, pref=None):
        d = dict(fixed)
        d.update({n: float(v) for n, v in zip(names, x)})
        if prof is not None:
            d[prof.name] = 0.0 if pref is None else pref
        return d

    def profiled(x):
        lnI0 = _model_lnI(mech, data, dev, pdict(x))
        if prof is None:
            return lnI0, None
        w2 = data.w**2
        pref = np.sum(w2 * (data.lnI - lnI0)) / np.sum(w2) / LN10
        pref = float(np.clip(pref, prof.lo, prof.hi))
        return lnI0 + LN10 * pref, pref

    def resid(x):
        lnI, _ = profiled(x)
        r = (lnI - data.lnI) * data.w
        return np.where(np.isfinite(r), r, 1e3)

    rng = np.random.default_rng(seed)
    if n_starts is None:
        n_starts = 3 if mech.expensive else 8
    x0 = np.array([init[n] for n in names])
    starts = [x0] + [np.clip(np.asarray(s, float), lo, hi) for s in extra_starts] + \
        _random_starts(free, n_starts, rng)
    best = None
    if not names:
        x_best = np.array([])
    else:
        # quick screening of starts, then polish the best few
        scored = []
        for s in starts:
            s = np.clip(s, lo + 1e-12 * (hi - lo), hi - 1e-12 * (hi - lo))
            r = resid(s)
            scored.append((float(np.sum(r**2)), s))
        scored.sort(key=lambda t: t[0])
        n_polish = 1 if screen else (2 if mech.expensive else 4)
        nf1 = 25 if screen else (150 if mech.expensive else 400)
        nf2 = 15 if screen else (80 if mech.expensive else 300)
        for _, s in scored[:n_polish]:
            try:
                sol = least_squares(resid, s, bounds=(lo, hi), loss="soft_l1", f_scale=0.3,
                                    x_scale="jac", max_nfev=nf1)
                sol = least_squares(resid, sol.x, bounds=(lo, hi), loss="linear", x_scale="jac",
                                    max_nfev=nf2)
            except Exception as exc:     # pragma: no cover
                continue
            if best is None or sol.cost < best.cost:
                best = sol
        x_best = best.x if best is not None else x0
    lnI_fit, pref = profiled(x_best)
    params = pdict(x_best, pref)
    r = (lnI_fit - data.lnI)
    rw = r * data.w
    rss = float(np.sum(rw**2))
    n = data.n

    # ---- uncertainties: Jacobian over all free params incl. the prefactor ----
    all_names = names + ([prof.name] if prof is not None else [])
    xa = np.array([params[k] for k in all_names])

    def resid_all(xx):
        d = dict(fixed)
        d.update({k: float(v) for k, v in zip(all_names, xx)})
        extra = LN10 * d[prof.name] if prof is not None else 0.0
        return (_model_lnI(mech, data, dev, d) + extra - data.lnI) * data.w

    errors, corr, k_eff, degenerate = {}, None, len(all_names), []
    if len(all_names) and not screen and uncertainties:
        J = np.empty((n, len(all_names)))
        r0 = resid_all(xa)
        for j in range(len(all_names)):
            h = 1e-5 * max(abs(xa[j]), 1e-2)
            xp = xa.copy(); xp[j] += h
            J[:, j] = (resid_all(xp) - r0) / h
        J = np.where(np.isfinite(J), J, 0.0)
        norms = np.linalg.norm(J, axis=0)
        norms[norms == 0] = 1.0
        Js = J / norms
        sv = np.linalg.svd(Js, compute_uv=False)
        k_eff = int(np.sum(sv > 1e-6 * sv.max())) if sv.size else 0
        dof = max(n - k_eff, 1)
        s2 = rss / dof
        cov_s = np.linalg.pinv(Js.T @ Js, rcond=1e-12) * s2
        cov = cov_s / np.outer(norms, norms)
        sd = np.sqrt(np.clip(np.diag(cov), 0, None))
        errors = {k: float(v) for k, v in zip(all_names, sd)}
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = cov / np.outer(sd, sd)
        for i in range(len(all_names)):
            for j in range(i + 1, len(all_names)):
                if np.isfinite(corr[i, j]) and abs(corr[i, j]) > 0.995:
                    degenerate.append((all_names[i], all_names[j], float(corr[i, j])))
        # parameters with no leverage at all
        for j, nm in enumerate(all_names):
            if np.linalg.norm(J[:, j]) < 1e-8 * max(np.linalg.norm(J), 1e-30):
                degenerate.append((nm, "(no effect on the data)", 0.0))
    k = len(all_names)
    ke = max(k_eff, 1)
    aic = n * np.log(max(rss, 1e-300) / n) + 2 * ke
    aicc = aic + (2 * ke * (ke + 1)) / max(n - ke - 1, 1)
    bic = n * np.log(max(rss, 1e-300) / n) + ke * np.log(n)
    return FitResult(key=mech.key, name=mech.name, params=params, errors=errors, free=all_names,
                     fixed=fixed, n=n, k=k, k_eff=k_eff, rss=rss, rms=float(np.sqrt(np.mean(r**2))),
                     aic=float(aic), aicc=float(aicc), bic=float(bic), lnI_fit=lnI_fit, corr=corr,
                     degenerate=degenerate, success=best is not None or not names)


def predict_lnI(mech, res: FitResult, V, T, d_m, area, dev):
    return mech.lnJ(np.asarray(V, float), np.asarray(T, float), np.asarray(d_m, float), res.params, dev) \
        + np.log(area) + LN10 * res.params.get(_prof_name(mech), 0.0)


def _prof_name(mech):
    for p in mech.params:
        if p.profile:
            return p.name
    return "__none__"


# ---------------------------------------------------------------------------
# cross-validation: extrapolate to held-out temperatures / thicknesses / biases
# ---------------------------------------------------------------------------
def cross_validate(mech, data: FitData, dev, res: FitResult, ds_cov) -> tuple:
    folds = []
    if ds_cov["n_T"] >= 3:
        mode = "leave-one-temperature-out"
        Ts = np.unique(data.T)
        # hold out the extreme temperatures: the most demanding extrapolation
        for T in [Ts.min(), Ts.max()]:
            folds.append(data.T == T)
    elif ds_cov["n_d"] >= 3:
        mode = "leave-one-thickness-out"
        dd = np.unique(data.d)
        for d in [dd.min(), dd.max()]:
            folds.append(data.d == d)
    else:
        mode = "bias extrapolation (fit lower 65 %, predict upper 35 % of each curve)"
        m = np.zeros(data.n, bool)
        for c in np.unique(data.curve):
            idx = np.where(data.curve == c)[0]
            Vc = data.V[idx]
            cut = np.quantile(Vc, 0.65)
            m[idx[Vc > cut]] = True
        folds.append(m)
    errs = []
    start = {k: res.params[k] for k in res.params}
    for test in folds:
        train = ~test
        if train.sum() < 6 or test.sum() < 2:
            continue
        sub = data.subset(train)
        r = fit(mech, sub, dev, hints=start, n_starts=1 if mech.expensive else 2, seed=1)
        pred = mech.lnJ(data.V[test], data.T[test], data.d[test], r.params, dev) + np.log(data.area[test]) \
            + LN10 * r.params.get(_prof_name(mech), 0.0)
        errs.append(np.sqrt(np.mean((pred - data.lnI[test]) ** 2)))
    if not errs:
        return None, mode
    return float(np.sqrt(np.mean(np.square(errs)))), mode


# ---------------------------------------------------------------------------
# composite models: two channels in parallel, or a model plus a series resistance
# ---------------------------------------------------------------------------
class Parallel(Mechanism):
    def __init__(self, a: Mechanism, b: Mechanism, pa: dict, pb: dict):
        self.a, self.b = a, b
        self.key = f"{a.key}+{b.key}"
        self.name = f"{a.name.split('(')[0].strip()}  ∥  {b.name.split('(')[0].strip()}"
        self.family = "parallel channels"
        self.expensive = a.expensive or b.expensive
        self.params = []
        for p in a.params:
            q = copy.copy(p); q.name = "a." + p.name; q.init = pa.get(p.name, p.init)
            self.params.append(q)
        pb_prof = _prof_name(b)
        for p in b.params:
            q = copy.copy(p); q.name = "b." + p.name; q.init = pb.get(p.name, p.init)
            if p.profile:
                q.profile = False
                q.name = "b.rel"
                q.lo, q.hi = -60, 60
                q.init = pb.get(pb_prof, 0.0) - pa.get(_prof_name(a), 0.0)
                q.desc = "log10 of channel-B prefactor relative to channel A"
            q.device_key = None if p.profile else p.device_key
            self.params.append(q)

    def _split(self, p):
        pa = {k[2:]: v for k, v in p.items() if k.startswith("a.")}
        pb = {k[2:]: v for k, v in p.items() if k.startswith("b.") and k != "b.rel"}
        return pa, pb

    def channels(self, V, T, d, p, dev):
        pa, pb = self._split(p)
        la = self.a.lnJ(V, T, d, pa, dev)
        lb = self.b.lnJ(V, T, d, pb, dev) + LN10 * p.get("b.rel", 0.0)
        return la, lb

    def lnJ(self, V, T, d, p, dev):
        la, lb = self.channels(V, T, d, p, dev)
        return np.logaddexp(la, lb)


class WithSeriesR(Mechanism):
    """Model + series resistance: V = V_j + I R_s (review Sec. XIX)."""

    def __init__(self, m: Mechanism, pm: dict, area_typ: float):
        self.m = m
        self.key = f"{m.key}+Rs"
        self.name = f"{m.name.split('(')[0].strip()} + series resistance"
        self.family = m.family
        self.expensive = m.expensive
        self.params = []
        for p in m.params:
            q = copy.copy(p); q.init = pm.get(p.name, p.init); q.profile = False
            self.params.append(q)
        self.params.append(Param("log10_Rs", "Ω", -3, 14, 3.0, "series resistance R_s (log10 Ω)"))
        self.area_typ = area_typ

    def lnJ(self, V, T, d, p, dev, area=None):
        V = np.asarray(V, float)
        area = self.area_typ if area is None else area
        Rs = 10.0 ** p["log10_Rs"]
        pref = LN10 * p.get(_prof_name(self.m), 0.0)
        lo = np.zeros_like(V) + 1e-12
        hi = V.copy()
        for _ in range(32):
            mid = 0.5 * (lo + hi)
            I = np.exp(self.m.lnJ(mid, T, d, p, dev) + pref) * area
            too_big = mid + I * Rs > V
            hi = np.where(too_big, mid, hi)
            lo = np.where(too_big, lo, mid)
        Vj = 0.5 * (lo + hi)
        return self.m.lnJ(Vj, T, d, p, dev) + pref

    def junction_voltage(self, V, T, d, p, dev):
        V = np.asarray(V, float)
        I = np.exp(self.lnJ(V, T, d, p, dev)) * self.area_typ
        return V - I * 10.0 ** p["log10_Rs"]
