# © 2026 Gokaran Shukla · github.com/Gokaran-Shukla/barista · MIT License  (gs-watermark)
"""Tutorial 6 — spin-dependent transport in graphene/CrSb/graphene.

The two graphene electrodes acquire a proximity exchange splitting from CrSb, so their interface spin
polarisation depends on energy and changes sign near the Dirac point. The CrSb barrier adds an altermagnetic
spin filter whose strength grows with the layer number N. BARISTA separates the two from MR(V, T, N).
"""
from barista import spin
from barista.spin_report import export_spin

df, truth = spin.synth_proximity(N_list=(4, 5, 6, 8), T_list=(10, 100, 200, 300))
tab = spin.mr_table(df, layer_spacing_nm=0.2735)
sr = spin.analyse_spin(tab, B=100)
print("hidden:", truth)
print(sr.statement)
export_spin(sr, "tutorial_output/06_spin")
