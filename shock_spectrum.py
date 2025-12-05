# shock_spectrum.py
# -----------------
# Diffusive shock acceleration (DSA) injection spectrum with species-dependent cutoff
# via the Murphy–Dermer–Ramaty condition v(E0i) R(E0i) = v(E0p) R(E0p).
#
# This restores the Z, A, and vR=const scaling you asked for. We keep your normalization
# policy (∫_{30 MeV/n}^{∞} φ(E) dE = 1) and clamp only below Emin_physical (default 1 MeV/n).
#
# For completeness, you can keep any simpler ShockSpectrum class you already have in
# this file; the class below is named ShockSpectrumRigidity and is self-contained.

from __future__ import annotations
import numpy as np
from scipy.optimize import brentq
from spectrum import Spectrum, _ensure_array


class ShockSpectrumRigidity(Spectrum):
    """
    Shock-accelerated injection spectrum with species-dependent cutoff.

    Physics
    -------
    We model the unnormalized spectrum as
        φ0(E) ∝ (1/b(E)) · p(E)^(-s) · exp[-E / E0i]                       (1)
    where:
        E     : kinetic energy per nucleon [MeV/n]
        s     : momentum-space power-law index (strong shocks: s ≈ 2.3–2.6)
        β     : v/c
        p(E)  : relativistic momentum per PARTICLE [MeV/c] (E is per nucleon)
        E0i   : *species cutoff* energy per nucleon for ion (Z,A) [MeV/n]

    The species cutoff follows Murphy–Dermer–Ramaty (1987):
        v(E0i; A) · R(E0i; Z,A) = v(E0p; A=1) · R(E0p; Z=1)             (2)
    i.e. all species share a common cutoff in vR (rigidity*speed) space, not in E.
    Here R = p c / (Z e). Numerically, we use: R[GV] = p[GeV/c] / Z.

    Practical notes:
    - We *normalize* from 30 MeV/n to ∞ (as in Ramaty fits).
    - We *plot* typically 1–1000 MeV/n; the >GeV tail is physically negligible for LiBeB yields.
    - We clamp only below Emin_physical (default 1 MeV/n) for numerical convenience.
    """

    def __init__(self,
                 s: float = 2.5,
                 E0p_turnover: float = 20.0,  # proton turnover [MeV/n]
                 Z: int = 1,
                 A: int = 1,
                 Emin_physical: float = 1.0,
                 Emin_norm: float = 30.0,
                 Emin_plot: float = 1.0,
                 Emax_plot: float = 1000.0,):
        self.s = float(s)
        self.Z = int(Z)
        self.A = int(A)
        self.E0p_turnover = E0p_turnover
        # Precompute the species cutoff E0i by solving vR condition
        self.E0i = self._compute_species_cutoff_vR()
        super().__init__(Emin_physical=Emin_physical,
                         Emin_norm=Emin_norm,
                         Emin_plot=Emin_plot,
                         Emax_plot=Emax_plot)


    # ---------- Relativistic kinematics (per species) ----------
    def recompute(self, Z_new, A_new):
        self.Z = Z_new
        self.A = A_new
        self.E0i = self._compute_species_cutoff_vR()
        N_new = self.compute_normalization_constant(self.Emin_norm)
        print(f"shock spectrum now approproate for Z={self.Z}, A={self.A}, and new cutoff is E_0i = {self.E0i}, "
              f"and recomputed normalization constant N={self.N} -> {N_new}")
        self.N = N_new

    def _momentum(self, E_per_n: float | np.ndarray) -> float | np.ndarray:
        E_k = _ensure_array(E_per_n)
        p = np.sqrt(E_k * (E_k + 2.0 * self.E_0p_rest)) / self.c
        return p

    def _beta(self, E_per_n: float | np.ndarray) -> float | np.ndarray:
        E_k = _ensure_array(E_per_n)
        beta = np.sqrt( (E_k * (E_k + 2 * self.E_0p_rest)) / ((E_k + self.E_0p_rest) **2) )
        return np.clip(beta, 1e-16, 1.0)

    def _rigidity(self, E_per_n: float | np.ndarray) -> float | np.ndarray:
        """
        Rigidity R dropping constants
        """
        E = _ensure_array(E_per_n)
        p_MeV = self._momentum(E)
        R = (self.A / self.Z) * p_MeV * self.c
        R /= 1000.0
        return R

    # ---------- Species cutoff via vR = const ----------

    def _compute_species_cutoff_vR(self) -> float:
        """
        Solve for E0i such that v(E0i;A)*R(E0i;Z,A) = v(E0p;1)*R(E0p;1,1).
        Robust 1D root-finding with brentq on E ∈ [1e-3, 1e5] MeV/n.
        """
        # Proton case: trivial
        if self.Z == 1 and self.A == 1:
            return self.E0p_turnover

        # Proton reference vR at the proton cutoff E0p
        target = self._beta(self.E0p_turnover) * self._momentum(self.E0p_turnover)
        f = lambda E: self._beta(E) * self._momentum(E) * self.A / self.Z - target
        return brentq(f, 1.0, 1e5)  # MeV/nuc

    # ---------- Spectral form (unnormalized) ----------

    def unnormalized(self, E_per_n: float | np.ndarray) -> float | np.ndarray:
        """
        φ0(E) ∝ (1/β) * p(E)^(-s) * exp[-E / E0i], evaluated for species (Z,A).

        - E is per nucleon [MeV/n].
        - p, β are computed per nucleon
        - Below Emin_physical, we return 0 (computational convenience).
        """
        E = _ensure_array(E_per_n)
        m = (E >= self.Emin_physical)
        p = np.clip(self._momentum(E[m]), 1e-30, None)
        beta = self._beta(E)
        pref = (1.0 / beta)
        powerlaw = p ** -self.s
        cutoff = np.exp(-E[m] / self.E0i)
        return pref * powerlaw * cutoff

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    E = np.logspace(0, 3, 200)
    plt.figure(figsize=(7, 5))
    for label, Z, A in [("p", 1, 1), ("3He", 2, 3), ("4He", 2, 4)]:
        spec = ShockSpectrumRigidity(s=3.3, E0p_turnover=30.0, Z=Z, A=A)
        R = spec._rigidity(E)
        plt.loglog(E, R, label=f"{label}")
    plt.xlabel("Energy per nucleon [MeV/n]")
    plt.ylabel("Rigidity [GV]")
    plt.legend()
    plt.grid(True, which="both", alpha=0.4)
    plt.title("Rigidity vs Energy (vR = const cutoff validation)")
    plt.show()