# stopping_power_bethe_bloch.py
# ---------------------------------------------------------------------
# Bethe–Bloch MASS stopping power (per grammage) for heavy projectiles:
#   S(E) = -(dE/dX)  [MeV cm^2 g^-1],   X = ρ x  [g cm^-2]
#
# We expose S(E) for kinetic energy per nucleon E [MeV/n].
#
# Minimal PDG-like form (no density-effect, no shell correction):
#   S(E) = K * z^2 * (Z/A)_mix * (1/β^2) *
#          [ ln( 2 m_e c^2 β^2 γ^2 / I_mix ) - β^2 ]
#
# With:  (βγ)^2 = E(E+2E0) / E0^2 and β^2 = E(E+2E0)/(E+E0)^2
#        E0 = m_p c^2 = 938.272 MeV (per nucleon)
#
# Units:
#   E [MeV/n], m_e c^2 = 0.511 MeV, m_p c^2 = 938.272 MeV
#   K = 0.307075 MeV cm^2 g^-1
#   I_mix in MeV (e.g., 19.31 eV = 19.31e-6 MeV)
# ---------------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class StoppingPowerBetheBloch:
    """
    Parameters
    ----------
    Z_over_A_mix : float
        Mixture Z/A by mass (height-independent for this non-depth-varying case).
        Example H/He photospheric-like: X_H=0.74, X_He=0.25 ⇒ (Z/A)_mix = 0.74*(1/1) + 0.25*(2/4) = 0.865
    I_mix_MeV : float
        Mean excitation energy of the mixture [MeV]. (19.31 eV = 19.31e-6 MeV)
    z_proj : int
        Projectile charge number (z). 1 for protons, 2 for alphas, etc.
    """
    Z_over_A_mix: float = 0.865
    I_mix_MeV: float = 19.31e-6
    z_proj: int = 1

    # Physical constants (PDG)
    K: float = 0.307075         # MeV cm^2 g^-1
    m_e_MeV: float = 0.51099895 # MeV
    m_p_MeV: float = 938.27208816 # MeV

    # --------------------- relativistic helpers ---------------------

    def beta2(self, E_MeV_per_n) -> np.ndarray:
        """
        β^2 = v^2/c^2 for kinetic energy per nucleon E.
        Identity: β^2 = E(E+2E0) / (E+E0)^2,  with E0 = m_p c^2.
        """
        E = np.asarray(E_MeV_per_n, dtype=float)
        E0 = self.m_p_MeV
        return np.clip(E*(E + 2.0*E0) / (E + E0)**2, 0.0, 1.0)

    def beta2gamma2(self, E_MeV_per_n) -> np.ndarray:
        """
        (βγ)^2 = E(E+2E0)/E0^2  (exact for per-nucleon quantities).
        """
        E = np.asarray(E_MeV_per_n, dtype=float)
        E0 = self.m_p_MeV
        return E*(E + 2.0*E0) / (E0*E0)

    # --------------------- main API: S(E) ---------------------------

    def dEdx(self, E_MeV_per_n) -> np.ndarray:
        """
        Mass stopping power S(E) = -(dE/dX)  [MeV cm^2 g^-1] at per-nucleon energy E.

        S(E) = K z^2 (Z/A)_mix (1/β^2) [ ln( 2 m_e c^2 β^2 γ^2 / I_mix ) - β^2 ]

        Numerical guards:
          - clamp β^2 and (βγ)^2 away from zero,
          - ensure log-argument > 1 to keep bracket positive at very low E.
        """
        E = np.asarray(E_MeV_per_n, dtype=float)
        z2 = float(self.z_proj)**2

        beta2 = np.clip(self.beta2(E), 1e-10, 1.0)
        bg2   = np.maximum(self.beta2gamma2(E), 1e-20)

        # Log argument: (2 m_e c^2 β^2 γ^2) / I
        arg = (2.0*self.m_e_MeV*bg2) / max(self.I_mix_MeV, 1e-30)
        arg = np.maximum(arg, 1.0000001)  # ensure >1

        log_term = np.log(arg)
        bracket  = log_term - beta2

        S = self.K * z2 * self.Z_over_A_mix * (1.0 / beta2) * bracket
        # S must be positive and finite
        S = np.where(np.isfinite(S) & (S > 1e-12), S, 1e-12)
        return S


# ----------------------------- sanity main ----------------------------------
if __name__ == "__main__":
    bb = StoppingPowerBetheBloch(Z_over_A_mix=0.865, I_mix_MeV=19.31e-6, z_proj=1)

    E = np.logspace(0, 3, 400)      # 1 MeV/n .. 1 GeV/n
    S = bb.dEdx(E)

    # Basic checks: positivity and reasonable variation
    assert np.all(np.isfinite(S)) and np.all(S > 0.0)
    # No wild collapse across 1–1000 MeV/n
    assert S[0] > S[-1]*0.2, "S shouldn’t drop by orders of magnitude here."

    # Identity checks at a point
    b2 = bb.beta2(100.0)
    bg2 = bb.beta2gamma2(100.0)
    assert 0 < b2 < 1 and bg2 > 0

    print("stopping_power_bethe_bloch: sanity checks passed ✓")