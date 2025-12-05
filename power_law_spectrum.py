# power_law_spectrum.py
# ---------------------
# Simple power-law φ0(E) ∝ E^{-s}.
# Useful as a toy comparator. Not meant to be physical at very low E;
# we clamp below Emin_physical for numerical stability.
#
# Normalization: ∫_{30 MeV/n}^{∞} φ(E) dE = 1 (handled by SpectrumBase).

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from spectrum import Spectrum, _ensure_array


class PowerLawSpectrum(Spectrum):
    """
    Pure power-law spectrum:
        φ0(E) ∝ E^{-s}

    Caveats:
    --------
    - At E → 0, a pure power-law diverges for s>0; we therefore return 0 for E < Emin_physical.
    - This is *not* meant to be a literal acceleration model—just a control shape.

    Parameters
    ----------
    s : float
        Power-law index (typical comparison values: 2–5).
    """

    def __init__(self,
                 s: float = 4.0,
                 Emin_physical: float = 1.0,
                 Emin_norm: float = 30.0,
                 Emin_plot: float = 1.0,
                 Emax_plot: float = 1000.0):
        self.s = float(s)
        super().__init__(Emin_physical=Emin_physical,
                         Emin_norm=Emin_norm,
                         Emin_plot=Emin_plot,
                         Emax_plot=Emax_plot)


    def unnormalized(self, E_k: float | np.ndarray) -> float | np.ndarray:
        E_k = _ensure_array(E_k)
        y = np.zeros_like(E_k)
        m = (E_k >= self.Emin_physical)
        if np.any(m):
            y[m] = np.power(E_k[m], -self.s)
        return y

    def recompute(self, Z_new, A_new):
        pass

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Plot a normalized power-law spectrum.")
    ap.add_argument("--s", type=float, default=4.0)
    ap.add_argument("--emin-phys", type=float, default=1.0)
    ap.add_argument("--emin-norm", type=float, default=30.0)
    ap.add_argument("--emin-plot", type=float, default=1.0)
    ap.add_argument("--emax-plot", type=float, default=1000.0)
    args = ap.parse_args()

    spec = PowerLawSpectrum(s=args.s,
                            Emin_physical=args.emin_phys,
                            Emin_norm=args.emin_norm,
                            Emin_plot=args.emin_plot,
                            Emax_plot=args.emax_plot)

    E = np.logspace(np.log10(spec.Emin_plot), np.log10(spec.Emax_plot), 500)
    plt.figure(figsize=(7.5,5.5))
    plt.loglog(E, spec.normalized_N_of_E(E), label=f"Power-law (s={spec.s:g})")
    plt.xlabel("Energy per nucleon [MeV/n]")
    plt.ylabel("φ(E) [normalized]")
    plt.title("Power-law spectrum")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()