import numpy as np
from scipy.integrate import quad


def _ensure_array(E_k):
    """Return E_k as a 1-D NumPy float array, even if nested or scalar."""
    if np.isscalar(E_k):
        return np.array([E_k], dtype=float)
    return np.atleast_1d(np.asarray(E_k, dtype=float)).ravel()


class Spectrum:
    """
    Abstract base class for all particle energy spectra.

    Each subclass (e.g. stochastic, shock, or power-law) must implement:
        - `unnormalized(E)`: returns the raw (unnormalized) φ₀(E)

    This base class provides:
        - normalization handling (∫φ₀(E)dE)
        - clamping logic for plotting and integration bounds
        - diagnostics for integral normalization
    """
    E_0p_rest = 938.272
    c = 29979245800.0 # cm/s
    e = 1.60217663 * (10 ** -19)

    def __init__(
        self,
        Emin_physical: float = 1.0,
        Emin_norm: float = 30.0,
        Emin_plot: float = 1.0,
        Emax_plot: float = 1000.0,
    ):
        """
        Parameters
        ----------
        Emin_physical : float
            Minimum physically meaningful energy [MeV/n].
            Below this, φ(E) is treated as 0 (e.g., for plotting or integration cutoff).

        Emin_norm : float
            Lower integration bound [MeV/n] for normalization.
            This defines ∫_{Emin_norm}^{∞} φ₀(E)dE = 1.
            For solar-flare conditions, typically 30 MeV/n.

        Emin_plot : float
            Lower energy bound for plotting [MeV/n].
            Used to define the x-range in visualization utilities.

        Emax_plot : float
            Upper energy bound for plotting [MeV/n].
            Defines the visualization range (usually up to 1 GeV/n)
        """
        self.Emin_physical = Emin_physical
        self.Emin_norm = Emin_norm
        self.Emin_plot = Emin_plot
        self.Emax_plot = Emax_plot
        # rest energy per nucleon [MeV]
        self.N = self.compute_normalization_constant(self.Emin_norm)

    # ------------------------------------------------------------------
    # --- Abstract method ----------------------------------------------
    # ------------------------------------------------------------------

    def unnormalized(self, E_k: float | np.ndarray) -> float | np.ndarray:
        """
        Must be implemented by subclasses.

        Returns
        -------
        np.ndarray : Unnormalized spectral shape φ₀(E).
        """
        raise NotImplementedError("Subclasses must implement `unnormalized(E)`")

    def recompute(self, Z_new, A_new):
        raise NotImplementedError("Subclasses must implement `recompute`")

    # ------------------------------------------------------------------
    # --- Normalization ------------------------------------------------
    # ------------------------------------------------------------------

    def compute_normalization_constant(self, norm_emin) -> float:
        """
        Numerically compute normalization constant:
            N = ∫_{Emin_norm}^{∞} φ₀(E) dE

        Returns
        -------
        float : normalization constant N
        """
        return self.N_integrand(norm_emin)

    def N_integrand(self, E_0_integrand):
        def integrand(E_k):
            return float(self.unnormalized(E_k))

        # noinspection PyTupleAssignmentBalance
        I, abserr = quad(
            integrand,
            E_0_integrand,
            np.inf,
            limit=200,
            epsabs=1e-10,
            epsrel=1e-6,
        )

        if not np.isfinite(I) or I <= 0.0:
            raise RuntimeError(
                f"Normalization integral invalid: I={I}, err={abserr}"
            )
        return I
    # ------------------------------------------------------------------
    # --- Normalized Spectrum ------------------------------------------
    # ------------------------------------------------------------------

    def normalized_N_of_E(self, E_k: float | np.ndarray) -> float | np.ndarray:
        """
        Return normalized spectrum φ(E) = φ₀(E) / N
        such that ∫ φ(E)dE = 1 over [Emin_norm, ∞).

        Or, return the normalized value at E
        """
        return self.unnormalized(E_k) / self.N


if __name__ == "__main__":
    from stochastic_spectrum import StochasticSpectrum

    spec = StochasticSpectrum(aT=0.018)
    E = np.logspace(0, 3, 300)
    phi = spec.normalized_N_of_E(E)

    print(f"Normalization constant = {spec.compute_normalization_constant():.3e}")