"""
stopping_power_leaky_box.py
----------------------------------------------------------------------
Leaky-Box Escape Model
----------------------------------------------------------------------
This module implements a classical *leaky-box* formulation — a simple
model for the survival of cosmic-ray particles subject to escape or
catastrophic losses, but **not** continuous energy loss.

Physics:
--------
The fundamental equation is:
    dN/dX = -N / Λ(E)
which integrates to:
    P_surv(E, X) = exp[-X / Λ(E)]

where:
    Λ(E) = λ₀ * (E / E_ref)^α  [g cm⁻²]
is the *escape (or attenuation) path length*, typically increasing with
energy (since high-energy particles are harder to confine).

This module does **not** define a stopping power dE/dx; for that,
see Bethe–Bloch or Neutral Medium models.  A small "toy" dE/dx can be
enabled for demonstration purposes only.
----------------------------------------------------------------------
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class StoppingPowerLeakyBox:
    """
    Leaky-box escape/survival model.

    Parameters
    ----------
    lambda0 : float
        Normalization of escape path length Λ(E) [g cm⁻²].
    alpha : float
        Power-law index describing energy dependence of Λ(E).
        Typical values: 0.3–0.6 for cosmic rays.
    E_ref : float
        Reference energy [MeV/n] for normalization (default 100).
    kappa : float
        Optional dimensionless coefficient for a "toy" dE/dx coupling:
        dE/dX = κ * E / Λ(E).  Default 0.0 → no coupling.
    """

    lambda0: float = 10.0
    alpha: float = 0.5
    E_ref: float = 100.0
    kappa: float = 0.0

    # ------------------------------------------------------------------
    def lambda_E(self, E: float | np.ndarray) -> np.ndarray:
        """
        Escape (attenuation) path length Λ(E) [g cm⁻²].

        Λ(E) = λ₀ * (E / E_ref)^α
        """
        E = np.asarray(E, dtype=float)
        return self.lambda0 * (E / self.E_ref) ** self.alpha

    # ------------------------------------------------------------------
    def survival_probability(self, E: float | np.ndarray, X: float) -> np.ndarray:
        """
        Survival probability of a particle after traversing column depth X.

        P_surv(E, X) = exp[-X / Λ(E)]
        """
        Λ = self.lambda_E(E)
        return np.exp(-X / np.maximum(Λ, 1e-300))

    # ------------------------------------------------------------------
    def dEdx_toy(self, E: float | np.ndarray) -> np.ndarray:
        """
        Toy energy-loss law (for demonstration only).

        If kappa ≠ 0, defines:
            dE/dX = κ * E / Λ(E)
        Otherwise returns zeros.

        NOTE:
            This is *not* a physical leaky-box energy-loss term.
            It is provided only for plotting compatibility.
        """
        E = np.asarray(E, dtype=float)
        if self.kappa == 0.0:
            return np.zeros_like(E)
        Λ = self.lambda_E(E)
        return self.kappa * E / np.maximum(Λ, 1e-300)