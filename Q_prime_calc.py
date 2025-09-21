import numpy as np

# ---------- Recommended Q′ formulations ----------

@staticmethod
def _avg_over_spectrum(E: np.ndarray, f: np.ndarray, N: np.ndarray) -> float:
    w = N
    denom = np.trapz(w, E)
    if denom <= 0.0:
        return 0.0
    return float(np.trapz(f * w, E) / denom)


def qprime_constant_grammage(self,
                             E: np.ndarray,
                             sigma: np.ndarray,
                             N: np.ndarray,
                             Xesc_const: float | None = None) -> float:
    """
    Q′_c ≈ X_esc * <σ>_N  (constant grammage).
    """
    if Xesc_const is None:
        Xesc_const = self.X0
    sigma_avg = self._avg_over_spectrum(E, sigma, N)
    return float(Xesc_const * sigma_avg)


def qprime_energy_dep_grammage(self,
                               E: np.ndarray,
                               sigma: np.ndarray,
                               N: np.ndarray,
                               Z: int, A: int) -> float:
    """
    Q′_c ≈ <X_esc>_N * <σ>_N  (energy-dependent grammage, factorized).
    """
    Xesc_E = self.Xesc_of_E(E, Z, A)
    Xavg = self._avg_over_spectrum(E, Xesc_E, N)
    sigma_avg = self._avg_over_spectrum(E, sigma, N)
    return float(Xavg * sigma_avg)


# ---------- Compatibility helper (if your code expects (dE/dx)^(-1)) ----------
