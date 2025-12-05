# total_inelastic_lambda.py
# ---------------------------------------------------------------------
# Projectile total inelastic (mass-changing) cross sections on H and He,
# anchored at 600 & 1500 MeV/n, with a weak power-law slope between anchors.
# Converted to catastrophic interaction length Λ(E) in column-depth units
# [g cm^-2] for an H/He mixture.
#
# Use in CSDA transport: survival over slab ΔX is exp( -ΔX / Λ(E) ).
#
# Units:
#   E:   MeV per nucleon (MeV/n)
#   σ:   cm^2   (anchors given in mb; convert via 1 mb = 1e-27 cm^2)
#   Λ:   g cm^-2
# ---------------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
import math

MB_TO_CM2 = 1e-27
M_P_G     = 1.67262192369e-24  # proton mass [g]

@dataclass
class TotalInelasticCrossSection:
    """
    σ_tot,inel for a projectile on H and He, anchored at 600 & 1500 MeV/n.
    For each target t ∈ {H, He}:
        σ_t(E) = σ_t(600) * (E/600)^{α_t},  with α_t set by σ_t(1500).
    Anchors supplied in mb; evaluation returns cm^2.
    """
    proj: str
    sigma_H_600_mb: float
    sigma_H_1500_mb: float
    sigma_He_600_mb: float
    sigma_He_1500_mb: float

    _alpha_H: float = None
    _alpha_He: float = None

    def __post_init__(self):
        den = math.log(1500.0 / 600.0)  # ln(2.5)
        for name, val in [
            ("sigma_H_600_mb", self.sigma_H_600_mb),
            ("sigma_H_1500_mb", self.sigma_H_1500_mb),
            ("sigma_He_600_mb", self.sigma_He_600_mb),
            ("sigma_He_1500_mb", self.sigma_He_1500_mb),
        ]:
            if val <= 0:
                raise ValueError(f"{name} must be positive, got {val}")
        self._alpha_H  = math.log(self.sigma_H_1500_mb  / self.sigma_H_600_mb)  / den
        self._alpha_He = math.log(self.sigma_He_1500_mb / self.sigma_He_600_mb) / den

    def sigma_H(self, E_MeV_per_n: float) -> float:
        E = max(E_MeV_per_n, 1e-12)
        σ_mb = self.sigma_H_600_mb * (E / 600.0) ** self._alpha_H
        return σ_mb * MB_TO_CM2

    def sigma_He(self, E_MeV_per_n: float) -> float:
        E = max(E_MeV_per_n, 1e-12)
        σ_mb = self.sigma_He_600_mb * (E / 600.0) ** self._alpha_He
        return σ_mb * MB_TO_CM2

    @property
    def alpha_H(self) -> float:  return self._alpha_H
    @property
    def alpha_He(self) -> float: return self._alpha_He

    def __repr__(self) -> str:
        return (f"<σ_inel {self.proj}: α_H={self.alpha_H:.3f}, α_He={self.alpha_He:.3f}; "
                f"H(600,1500)={self.sigma_H_600_mb},{self.sigma_H_1500_mb} mb; "
                f"He(600,1500)={self.sigma_He_600_mb},{self.sigma_He_1500_mb} mb>")

@dataclass
class LambdaMix:
    """
    Interaction length Λ(E) [g cm^-2] for an H/He ambient (mass fractions X_H, X_He).

    1/Λ(E) = (n_H/ρ) σ_H(E) + (n_He/ρ) σ_He(E)
           = (X_H/m_p) σ_H(E) + (X_He/(4 m_p)) σ_He(E).
    """
    X_H: float = 0.74
    X_He: float = 0.25

    def inv_lambda(self, csec: TotalInelasticCrossSection, E: float) -> float:
        return (self.X_H / M_P_G) * csec.sigma_H(E) + (self.X_He / (4.0 * M_P_G)) * csec.sigma_He(E)

    def lambda_g_cm2(self, csec: TotalInelasticCrossSection, E: float) -> float:
        invL = self.inv_lambda(csec, E)
        if invL <= 0.0 or not math.isfinite(invL):
            raise ValueError("Invalid 1/Λ (non-positive or non-finite).")
        return 1.0 / invL

# Convenience factories (your anchors)
def sigma_C_projectile() -> TotalInelasticCrossSection:
    return TotalInelasticCrossSection("C", 273.0, 251.0, 433.0, 407.0)

def sigma_N_projectile() -> TotalInelasticCrossSection:
    return TotalInelasticCrossSection("N", 310.0, 288.0, 485.0, 460.0)

def sigma_O_projectile() -> TotalInelasticCrossSection:
    return TotalInelasticCrossSection("O", 347.0, 323.0, 542.0, 515.0)

# ------------------------------ sanity main ------------------------------
if __name__ == "__main__":
    mix = LambdaMix(X_H=0.74, X_He=0.25)
    for csec in (sigma_C_projectile(), sigma_N_projectile(), sigma_O_projectile()):
        # slopes should be weakly negative
        assert csec.alpha_H  < 0.0 and csec.alpha_He < 0.0
        # σ(1500) < σ(600)
        assert csec.sigma_H(1500.0)  < csec.sigma_H(600.0)
        assert csec.sigma_He(1500.0) < csec.sigma_He(600.0)
        # Λ ~ tens–hundreds g/cm^2 in this band
        for E in (600.0, 1000.0, 1500.0):
            L = mix.lambda_g_cm2(csec, E)
            assert 10.0 < L < 1e4
    print("total_inelastic_lambda: sanity checks passed ✓")