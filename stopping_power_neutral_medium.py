"""
stopping_power_neutral_medium.py
----------------------------------------------------------------------
Neutral Medium Energy Loss Model
----------------------------------------------------------------------

Implements the empirical stopping power law for ions traversing a
neutral medium, typically used for *stellar chromosphere* or
*photospheric* transport, where Coulomb collisions with neutral
atoms dominate rather than ionized plasma effects.

The functional form is based on an empirical Bethe–Bloch–type scaling
in neutral gas (see Ramaty et al. 1979; Tatischeff 2006):

    dE/dx = 630 * (Z_eff² / A) * E^(-0.8)

where:
    - E is the kinetic energy per nucleon [MeV/n],
    - Z is the projectile charge number (input),
    - A is the projectile mass number,
    - Z_eff is the *effective* charge, accounting for partial
      electron screening in a neutral medium.

Z_eff is modeled as:
    Z_eff = Z * [ 1 - exp(- (137 / Z^(2/3)) * sqrt(2E / E₀p)) ]

with E₀p = 938 MeV (proton rest energy).

All outputs are in units of [MeV g⁻¹ cm²].
----------------------------------------------------------------------
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class NeutralMediumStoppingPower:
    """
    Stopping power for ions traversing a neutral medium.

    Parameters
    ----------
    Z : int
        Projectile nuclear charge number (e.g., 1 for proton, 2 for alpha).
    A : int
        Projectile mass number.
    E_0p_rest : float, optional
        Proton rest energy [MeV], defaults to 938.272 MeV.
    coeff : float, optional
        Prefactor [MeV g⁻¹ cm²]; empirical normalization, default 630.
    exp_power : float, optional
        Energy exponent (empirical), default -0.8.
    """

    Z: int = 1
    A: int = 1
    E_0p_rest: float = 938.272
    coeff: float = 630.0
    exp_power: float = -0.8

    # ------------------------------------------------------------------
    def Z_eff(self, E: float | np.ndarray) -> np.ndarray:
        """
        Compute effective charge Z_eff(E) as a function of kinetic energy per nucleon.

        Formula:
            Z_eff = Z * [1 - exp(- (137 / Z^(2/3)) * sqrt(2E / E₀p))]

        Returns:
            Z_eff (dimensionless, same shape as E)
        """
        E = np.asarray(E, dtype=float)
        # Avoid overflow/underflow for small E
        factor = (137.0 / np.power(self.Z, 2 / 3)) * np.sqrt(2.0 * E / self.E_0p_rest)
        return self.Z * (1.0 - np.exp(-factor))

    # ------------------------------------------------------------------
    def dEdx(self, E: float | np.ndarray) -> np.ndarray:
        """
        Compute energy loss rate (stopping power) [MeV g⁻¹ cm²].

        Formula:
            dE/dx = 630 * (Z_eff² / A) * E^(-0.8)
        """
        E = np.asarray(E, dtype=float)
        Z_eff_val = self.Z_eff(E)
        return self.coeff * (Z_eff_val ** 2 / self.A) * np.power(E, self.exp_power)

    # ------------------------------------------------------------------
    def summary(self, E: float = 100.0) -> str:
        """
        Print a quick diagnostic for the effective charge and stopping power
        at a given reference energy (default 100 MeV/n).

        Returns a formatted string.
        """
        z_eff = self.Z_eff(E)
        dedx = self.dEdx(E)
        return (
            f"Neutral Medium Stopping Power Summary\n"
            f"-------------------------------------\n"
            f"Projectile: Z={self.Z}, A={self.A}\n"
            f"E = {E:.1f} MeV/n\n"
            f"Z_eff(E) = {z_eff:.3f}\n"
            f"dE/dx(E) = {dedx:.3e} MeV g⁻¹ cm²\n"
        )


# ----------------------------------------------------------------------
# Example Usage / Plot Driver
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Example for p, 3He, 4He
    species = [
        NeutralMediumStoppingPower(Z=1, A=1),
        NeutralMediumStoppingPower(Z=2, A=3),
        NeutralMediumStoppingPower(Z=2, A=4),
    ]
    labels = ["p", "³He", "⁴He"]

    Egrid = np.logspace(0, 3, 300)  # 1–1000 MeV/n

    plt.figure(figsize=(8, 5))
    for s, lbl in zip(species, labels):
        dEdx_vals = s.dEdx(Egrid)
        plt.loglog(Egrid, dEdx_vals, lw=2, label=f"{lbl}: Z={s.Z}, A={s.A}")

    plt.xlabel("Energy per nucleon [MeV/n]")
    plt.ylabel(r"$|dE/dx|$ [MeV g$^{-1}$ cm$^{2}$]")
    plt.title("Neutral Medium Stopping Power for p, ³He, ⁴He")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.show()