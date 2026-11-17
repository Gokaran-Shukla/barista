"""Optional user knowledge about the device.

Every field is optional.  BARISTA uses what is supplied in three ways
(Sec. XVIII of the review: parameters are best *checked* against
independent knowledge rather than imposed):

1. as starting guesses for the global fits,
2. as consistency checks on fitted parameters (e.g. Poole-Frenkel eps_r vs n^2),
3. as hard constraints, but only for the names listed in ``fix``.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields
from typing import Optional
import json

import numpy as np

# Material classes and the mechanisms the review lists as typical (Tables 2, 10).
MATERIAL_PRIORS = {
    "amorphous_oxide": ["coherent", "tat", "poole_frenkel", "schottky", "fowler_nordheim"],
    "crystalline_oxide": ["coherent", "tat", "schottky"],
    "high_k_oxide": ["tat", "poole_frenkel", "schottky", "coherent"],
    "layered_vdw": ["nn_hopping", "vrh", "tat", "sclc_trap", "schottky", "glazman_matveev"],
    "organic": ["sclc_trap", "sclc_exp", "gdm", "polaron", "schottky"],
    "molecular": ["coherent", "polaron", "nn_hopping"],
    "perovskite": ["sclc_trap", "sclc_exp", "nn_hopping", "ionic"],
    "resistive_memory": ["ohmic_metallic", "sclc_trap", "poole_frenkel", "schottky", "nn_hopping"],
    "mtj": ["coherent", "tat", "glazman_matveev"],
    "nitride": ["poole_frenkel", "tat"],
    "semiconductor": ["schottky", "sclc_trap", "band_to_band", "vrh"],
}


# Layered-magnet barriers: interlayer spacing (nm), typical ordering temperature, electrodes.
SYSTEM_PRESETS = {
    "graphene/crsb/graphene": {"layer_spacing_nm": 0.2735, "ordering_T_K": 700.0, "electrode": "graphene",
                               "system": "graphene/CrSb/graphene", "material_class": "layered_vdw"},
    "graphite/cri3/graphite": {"layer_spacing_nm": 0.66, "ordering_T_K": 45.0, "electrode": "graphene",
                               "system": "graphite/CrI3/graphite", "material_class": "mtj"},
    "graphite/crsbr/graphite": {"layer_spacing_nm": 0.79, "ordering_T_K": 132.0, "electrode": "graphene",
                                "system": "graphite/CrSBr/graphite", "material_class": "layered_vdw"},
    "cofeb/mgo/cofeb": {"ordering_T_K": 1000.0, "electrode": "ferromagnet", "system": "CoFeB/MgO/CoFeB",
                        "material_class": "crystalline_oxide"},
}


@dataclass
class DeviceInfo:
    # geometry
    area_um2: Optional[float] = None          # junction area (um^2)
    thickness_nm: Optional[float] = None      # used only if the data has no thickness column
    layer_spacing_nm: Optional[float] = None  # a, interlayer spacing (layered barriers)
    # dielectric
    eps_r: Optional[float] = None             # static relative permittivity (SCLC, VTFL)
    n_opt: Optional[float] = None             # refractive index -> optical eps = n^2 (PF / Schottky check)
    # barrier / band structure
    barrier_eV: Optional[float] = None        # Phi, barrier height from E_F
    mass_ratio: Optional[float] = None        # tunnelling effective mass m*/me
    band_gap_eV: Optional[float] = None       # Eg (band-to-band)
    fermi_energy_eV: float = 5.0              # electrode Fermi energy (Tsu-Esaki supply)
    # traps / carriers
    trap_density_cm3: Optional[float] = None  # N_t
    trap_depth_eV: Optional[float] = None     # E_c - E_t (SCLC) or Phi_t (PF/TAT)
    Nc_cm3: float = 2.5e19                    # effective DOS of the transport band at 300 K
    mobility_cm2Vs: Optional[float] = None
    # magnetic / spin
    n_layers: Optional[int] = None            # number of barrier layers (e.g. CrSb layers); d = N·a
    exchange_meV: Optional[float] = None      # proximity exchange splitting guess (spin module)
    ordering_T_K: Optional[float] = None      # Néel / Curie temperature
    electrode: Optional[str] = None           # 'graphene' | 'metal' | 'ferromagnet'
    system: Optional[str] = None              # e.g. 'graphene/CrSb/graphene'
    # misc
    temperature_K: Optional[float] = None     # used only if the data has no temperature column
    material_class: Optional[str] = None      # key of MATERIAL_PRIORS
    contact_2d: bool = False                  # 2D channel contact -> T^{3/2} Richardson prefactor
    kF_per_A: float = 1.0                     # electrode Fermi wavevector (channel counting)
    noise_rel: Optional[float] = None         # known relative current noise (e.g. 0.05)
    fix: list = field(default_factory=list)   # names of inputs to impose as hard constraints

    # ---- derived helpers -------------------------------------------------
    @property
    def area_m2(self) -> Optional[float]:
        return None if self.area_um2 is None else self.area_um2 * 1e-12

    @property
    def eps_static(self) -> float:
        if self.eps_r is not None:
            return self.eps_r
        if self.n_opt is not None:
            return max(self.n_opt**2, 1.0)
        return 5.0

    @property
    def eps_optical(self) -> Optional[float]:
        return None if self.n_opt is None else self.n_opt**2

    @property
    def mstar(self) -> float:
        return self.mass_ratio if self.mass_ratio is not None else 0.5

    def is_fixed(self, name: str) -> bool:
        return name in (self.fix or [])

    def provided(self) -> dict:
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            default = f.default if f.default is not None else None
            if f.name == "fix":
                if v:
                    out[f.name] = v
                continue
            if v is not None and v != default:
                out[f.name] = v
        return out

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "DeviceInfo":
        names = {f.name for f in fields(cls)}
        clean = {k: v for k, v in (d or {}).items() if k in names and v is not None}
        if isinstance(clean.get("fix"), str):
            clean["fix"] = [s.strip() for s in clean["fix"].split(",") if s.strip()]
        return cls(**clean)

    @classmethod
    def from_file(cls, path: str) -> "DeviceInfo":
        with open(path) as fh:
            if path.endswith((".yml", ".yaml")):
                import yaml
                d = yaml.safe_load(fh)
            else:
                d = json.load(fh)
        return cls.from_dict(d)
