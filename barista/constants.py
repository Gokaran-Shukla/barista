"""Physical constants (SI) used throughout BARISTA."""
import numpy as np

e = 1.602176634e-19        # C
h = 6.62607015e-34         # J s
hbar = h / (2 * np.pi)     # J s
me = 9.1093837015e-31      # kg
kB = 1.380649e-23          # J/K
kB_eV = kB / e             # eV/K  (8.617e-5)
eps0 = 8.8541878128e-12    # F/m
G0 = 2 * e**2 / h          # conductance quantum, 77.5 uS
A_RICHARDSON = 4 * np.pi * e * me * kB**2 / h**3   # 1.2017e6 A m^-2 K^-2 (=120 A cm^-2 K^-2)

NM = 1e-9
CM3 = 1e6                  # 1 cm^-3 = 1e6 m^-3
