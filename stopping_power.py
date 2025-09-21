# stopping_power.py
from dataclasses import dataclass
import numpy as np
from typing import Callable

@dataclass
class NeutralMediumStoppingPower:
    """
    Energy loss in a neutral medium (e.g., photosphere/chromosphere).
    """
    Z: int                     # projectile charge
    A: int                     # projectile mass number
    E_op: float = 938.0        # proton rest energy [MeV]
    coeff: float = 630.0       # prefactor [MeV/nucleon * g/cm^2]

    def Zeff(self, E: float) -> float:
        beta = np.sqrt(2.0 * E / self.E_op)
        return self.Z * (1.0 - np.exp(-137.0 / (self.Z ** (2.0/3.0)) * beta))

    def dEdx(self, E: float) -> float:
        Zeff_val = self.Zeff(E)
        return self.coeff * (Zeff_val ** 2) / self.A * (E ** -0.8)

    # Factories
    @classmethod
    def proton(cls) -> "NeutralMediumStoppingPower":
        return cls(Z=1, A=1)

    @classmethod
    def helium3(cls) -> "NeutralMediumStoppingPower":
        return cls(Z=2, A=3)

    @classmethod
    def helium4(cls) -> "NeutralMediumStoppingPower":
        return cls(Z=2, A=4)

@dataclass
class LeakyBoxStoppingPower:
    """
    Leaky-box stopping power and survival probability for cosmic-ray propagation.
    Includes an approximate total inelastic cross section σ_tot(E) with mild
    energy dependence (Bradt–Peters style).
    """
    grammage: float = 7.0  # g/cm^2, typical escape path length
    projectile_A: int = 1  # projectile mass number (1=proton, 4=alpha, etc.)
    target_A: int = 14     # target mass number (default N)

    def sigma_tot(self, E: float) -> float:
        """
        Approximate total inelastic cross section σ_tot(E) in mb.
        E is kinetic energy per nucleon in MeV.
        """
        r0 = 1.2  # fm
        delta = 1.0
        # Geometric overlap term
        sigma_geom = np.pi * (r0**2) * (self.projectile_A**(1/3) + self.target_A**(1/3) - delta)**2
        sigma_mb = sigma_geom * 10.0  # fm^2 → mb (1 fm^2 ≈ 10 mb)

        # Mild energy dependence: σ slightly larger at low E
        correction = 1.0 - 0.05 * np.log10(max(E, 10.0) / 100.0)
        return sigma_mb * correction

    def dedx(self, E: float) -> float:
        """
        Energy loss per unit grammage (MeV / (g/cm^2)).
        In the leaky-box picture, energy loss is small compared to escape,
        so this can be approximated as constant or parameterized separately.
        Here we just return a dummy weak energy dependence.
        """
        return 1e-2 * (E / 100.0)**-0.2

    def survival_probability(self, E: float) -> float:
        sigma = self.sigma_tot(E)  # mb
        sigma_cm2 = sigma * 1e-27
        mp = 1.67e-24  # g
        A_target = self.target_A
        Lambda = (A_target * mp) / sigma_cm2  # g/cm^2
        return np.exp(-self.grammage / Lambda)