import numpy as np
from scipy.special import kv  # Modified Bessel function of the 2nd kind
from spectrum import Spectrum, _ensure_array


class StochasticSpectrum(Spectrum):
    """
    Stochastic (Fokker–Planck) accelerated particle spectrum.

    Physical Model
    ---------------
    Derived from the steady–state solution to the Fokker–Planck equation
    for stochastic acceleration (Murphy, Dermer, & Ramaty 1987).

    The particle differential flux is proportional to a modified Bessel function
    of the second kind, K₂(x), where:

        φ₀(E) ∝ N_j· K₂(x)

    with

        x(E) = 2 √[ 3·p(E) / (m_pc·aT)]

    Here:
      - N_j is proportional to thw accelerated particle (taken as 1 for protons)
      - E : kinetic energy per nucleon [MeV/n]
      - p(E) : momentum per particle (MeV/c)
      - aT : dimensionless temperature parameter

    """

    def __init__(self, aT: float = 0.018,
                 Emin_physical: float = 1.0,
                 Emin_norm: float = 30.0,
                 Emin_plot: float = 1.0,
                 Emax_plot: float = 1000.0):
        """
        Parameters
        ----------
        aT : float
            Dimensionless parameter controlling the spectral shape.
            Physically related to plasma temperature and turbulence intensity.
        Emin_physical : float
            Minimum energy [MeV/n] below which the spectrum is clamped to 0.
            This is for computational convenience, not a physical cutoff.
        """

        self.aT = aT
        super().__init__(Emin_physical=Emin_physical,
                         Emin_norm=Emin_norm,
                         Emin_plot=Emin_plot,
                         Emax_plot=Emax_plot)


    # ------------------------------------------------------------------
    # --- Relativistic helper functions -------------------------------
    # ------------------------------------------------------------------

    def unnormalized(self, E_k: float | np.ndarray) -> float | np.ndarray:
        """
        Evaluate the unnormalized stochastic spectrum.

        Formula:
            φ₀(E) ∝ K₂(x)

        Returns
        -------
        np.ndarray or float: unnormalized φ₀(E) values.
        """
        E_k = _ensure_array(E_k)
        y = np.zeros_like(E_k)

        mask = E_k > self.Emin_physical
        if not np.any(mask):
            return y

        p = np.sqrt(E_k * (E_k + 2.0 * self.E_0p_rest))  # MeV
        arg = 2.0 * np.sqrt((3.0 * p) / (self.E_0p_rest * self.aT))
        return kv(2.0, arg)


    # ------------------------------------------------------------------
    # --- Main entry point for normalized spectrum --------------------
    # ------------------------------------------------------------------

    def normalized_N_of_E(self, E_k: float | np.ndarray) -> float | np.ndarray:
        """
        Compute normalized spectrum φ(E) such that:

            ∫ φ(E) dE = 1   (from Emin → ∞)

        Uses the abstract normalization method from SpectrumBase.
        """
        return super().normalized_N_of_E(E_k)

    def recompute(self, Z_new, A_new):
        pass

    # ------------------------------------------------------------------
    # --- Convenience plotting (standalone main) ----------------------
    # ------------------------------------------------------------------

    @staticmethod
    def main():
        """
        Example usage and diagnostic plot.
        Compares stochastic spectra for proton, ³He, and ⁴He.
        """
        import matplotlib.pyplot as plt

        E = np.logspace(0, 3, 500)  # 1 → 1000 MeV/n

        species = [
            ("p", StochasticSpectrum(aT=0.018)),
            ("³He", StochasticSpectrum(aT=0.018)),
            ("⁴He", StochasticSpectrum(aT=0.018)),
        ]

        plt.figure(figsize=(7, 5))
        for label, spec in species:
            phi = spec.normalized_N_of_E(E)
            plt.loglog(E, phi, label=f"{label} (aT={spec.aT})")

        plt.xlabel("Energy per nucleon [MeV/n]")
        plt.ylabel("Normalized differential flux φ(E)")
        plt.title("Stochastic Acceleration Spectra (K₂-type)")
        plt.grid(True, which="both", ls="--", alpha=0.6)
        plt.xlim(1, 1000)
        plt.legend()
        plt.tight_layout()
        plt.show()


# ----------------------------------------------------------------------
# --- Entry point ------------------------------------------------------
# ----------------------------------------------------------------------

if __name__ == "__main__":
    StochasticSpectrum.main()