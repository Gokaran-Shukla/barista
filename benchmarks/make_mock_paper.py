"""Build a mock 'published paper' PDF (text + vector figure) from a hidden mechanism.
Used to test the paper → data → mechanism pipeline end-to-end."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from barista import synth

def make(path, key="poole_frenkel", raster=False):
    pr = synth.PRESETS[key]
    df = synth.generate(key, pr["params"], pr["V"], pr["T"], pr["d"][:1], noise=0.04, dev=pr["dev"], seed=5,
                        n_points=18)
    d = pr["d"][0]
    claims = {"poole_frenkel": "Poole-Frenkel emission from bulk traps",
              "schottky": "Schottky emission over the interface barrier",
              "coherent": "direct tunneling and Fowler-Nordheim tunneling",
              "nn_hopping": "nearest-neighbour hopping conduction",
              "sclc_trap": "space-charge-limited conduction with a trap-filled limit"}
    text = (f"Leakage current mechanisms in {d:g} nm thick dielectric films\n\n"
            f"Abstract. We measured the current-voltage characteristics of {d:g} nm thick films between "
            f"{min(pr['T'])} K and {max(pr['T'])} K. The conduction is dominated by {claims.get(key, key)}. "
            f"The refractive index of the film is n = {pr['dev'].get('n_opt', 2.0)} and the junction area of "
            f"{pr['dev'].get('area_um2', 100)} um2 was defined by lithography.\n\n"
            f"Conclusions\nThe data are consistent with {claims.get(key, key)}.")
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.08, 0.95, text.split("\n")[0], fontsize=16, va="top")
        import textwrap
        fig.text(0.08, 0.9, "\n".join(textwrap.wrap(text.split("\n\n", 1)[1], 95)), fontsize=9, va="top")
        pdf.savefig(fig); plt.close(fig)
        fig, ax = plt.subplots(figsize=(4, 3))
        cols = plt.cm.viridis(np.linspace(0, 0.9, len(pr["T"])))
        for T, c in zip(pr["T"], cols):
            g = df[df["T (K)"] == T]
            ax.plot(g["V (V)"], g["I (A)"], "o", color=c, ms=3, label=f"{T} K")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Voltage (V)"); ax.set_ylabel("Current (A)")
        ax.legend(fontsize=6)
        if raster:
            fig.savefig(path.replace(".pdf", "_fig.png"), dpi=300)
            img = plt.imread(path.replace(".pdf", "_fig.png"))
            plt.close(fig)
            fig = plt.figure(figsize=(8.27, 11.69)); ax2 = fig.add_axes([0.1, 0.5, 0.8, 0.4]); ax2.imshow(img); ax2.axis("off")
        pdf.savefig(fig); plt.close(fig)
    return df

if __name__ == "__main__":
    make(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "poole_frenkel", raster=len(sys.argv) > 3)
