import numpy as np
from dataclasses import dataclass

@dataclass
class SigmaTotalInelastic:
    """
    Total inelastic (mass-changing) cross section for accelerated CNO projectiles
    on ambient H and He, anchored to Webber et al. (1990) values at 600 and 1500 MeV/n.
    Energies are in MeV/nucleon, cross sections in mb.
    """
    proj: str
    sigma_H_600: float
    sigma_H_1500: float
    sigma_He_600: float
    sigma_He_1500: float
    nH: float = 1.0      # ambient hydrogen abundance (relative)
    nHe: float = 0.1     # ambient helium abundance (relative)
    include_he: bool = True

    def __post_init__(self):
        # Precompute slopes (alpha exponents) for scaling
        self.alpha_H = np.log(self.sigma_H_1500 / self.sigma_H_600) / np.log(1500/600)
        self.alpha_He = np.log(self.sigma_He_1500 / self.sigma_He_600) / np.log(1500/600)

    def sigma_H(self, E: float) -> float:
        """Return sigma for H target at energy E (MeV/n)."""
        return self.sigma_H_600 * (E/600.0)**self.alpha_H

    def sigma_He(self, E: float) -> float:
        """Return sigma for He target at energy E (MeV/n)."""
        return self.sigma_He_600 * (E/600.0)**self.alpha_He

    def sigma_tot(self, E: float) -> float:
        """Weighted total sigma (H + He if enabled)."""
        total = self.nH * self.sigma_H(E)
        if self.include_he:
            total += self.nHe * self.sigma_He(E)
        return total


# ------------------------------------------------------------------
# Factory functions: values digitized from Webber et al. (1990)
# Table: total inelastic cross sections (mass changing), mb
# Anchored at 600 and 1500 MeV/n

def sigma_C_projectile() -> SigmaTotalInelastic:
    return SigmaTotalInelastic(
        proj="C",
        sigma_H_600=273.0, sigma_H_1500=251.0,
        sigma_He_600=433.0, sigma_He_1500=407.0
    )

def sigma_N_projectile() -> SigmaTotalInelastic:
    return SigmaTotalInelastic(
        proj="N",
        sigma_H_600=310.0, sigma_H_1500=288.0,
        sigma_He_600=485.0, sigma_He_1500=460.0
    )

def sigma_O_projectile() -> SigmaTotalInelastic:
    return SigmaTotalInelastic(
        proj="O",
        sigma_H_600=347.0, sigma_H_1500=323.0,
        sigma_He_600=542.0, sigma_He_1500=515.0
    )

# ------------------------------------------------------------------
# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    models = [sigma_C_projectile(), sigma_N_projectile(), sigma_O_projectile()]
    Egrid = np.logspace(2, 3.3, 200)  # 100 to 2000 MeV/n

    plt.figure(figsize=(7,5))
    for model in models:
        y = [model.sigma_tot(E) for E in Egrid]
        plt.loglog(Egrid, y, label=f"{model.proj} proj on H+He")

    plt.xlabel("Energy per nucleon (MeV/n)")
    plt.ylabel("Total mass-changing σ (mb)")
    plt.title("Webber+1990 Anchored CNO Projectiles on H, He")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.show()