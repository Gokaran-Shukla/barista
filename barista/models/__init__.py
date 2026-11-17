"""Registry of transport mechanisms."""
from .base import Mechanism, Param, Check
from .tunnelling import Coherent, TrapAssisted, GlazmanMatveev, BandToBand
from .emission import Schottky, PooleFrenkel
from .hopping import NNHopping, VRH, Polaron, GDM
from .sclc import SCLCTrap, SCLCExp
from .other import OhmicMetallic

ALL = [Coherent(), TrapAssisted(), GlazmanMatveev(), BandToBand(), Schottky(), PooleFrenkel(),
       NNHopping(), VRH(), Polaron(), GDM(), SCLCTrap(), SCLCExp(), OhmicMetallic()]
REGISTRY = {m.key: m for m in ALL}


SHORT = {
    "coherent": "Coherent tunnelling (direct/FN)", "tat": "Trap-assisted tunnelling",
    "glazman_matveev": "Glazman–Matveev chains", "band_to_band": "Band-to-band (Zener) tunnelling",
    "schottky": "Schottky emission", "poole_frenkel": "Poole–Frenkel emission",
    "nn_hopping": "Nearest-neighbour / ionic hopping", "vrh": "Variable-range hopping",
    "polaron": "Polaron (Marcus) hopping", "gdm": "Gaussian-disorder hopping",
    "sclc_trap": "SCLC with a discrete trap level", "sclc_exp": "SCLC with exponential traps",
    "ohmic_metallic": "Metallic path (pinhole/filament)",
}


def short_name(key: str) -> str:
    return SHORT.get(key, key)


def get(key: str) -> Mechanism:
    return REGISTRY[key]


__all__ = ["Mechanism", "Param", "Check", "ALL", "REGISTRY", "get", "SHORT", "short_name"]
