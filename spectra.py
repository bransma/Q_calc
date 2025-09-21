"""
spectra.py
==========

Heavily commented particle energy spectra for LiBeB spallation work.

This module defines FOUR spectrum families with a common, minimal API:

    - StochasticK2   : N(E) ∝ K2( 2 * sqrt( 3 p / (m_p c^2 * aT) ) )      (flare stochastic)
    - PowerLaw       : N(E) ∝ E^{-γ}                              (benchmark / CR)
    - ShockRigidity  : N(E) ∝ R(E)^{-s} * exp(-E/E0_species)      (DSA in rigidity)
    - CompositeSpectrum: linear mixture of any spectra above

All spectra are normalized on a *finite* energy-per-nucleon interval [Emin, Emax]
which you control at construction. The normalization constant C is computed
numerically and stored, so .N(E) returns a probability density function (pdf)
with unit area under the curve over [Emin, Emax].

Design notes
------------
1) We normalize numerically (trapz) for robustness (no closed forms needed).
2) Outside [Emin, Emax], N(E) = 0 by construction (clamps), to avoid unintended
   contributions if called out-of-bounds.
3) We keep "unnormalized(E)" separate so that you can examine shapes and/or
   form composites before normalization.
4) ShockRigidity uses rigidity R(E; Z, A). For different species, we provide
   a helper to derive E0_species from a proton cutoff using the vR-invariance
   used in Ramaty+87: v(E0_i) * R(E0_i) = v(E0_p) * R(E0_p).

Dependencies
------------
- numpy
- scipy.special (Kν Bessel) and scipy.optimize (brentq)

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.special import kv as Knu  # Modified Bessel Kν
from scipy.optimize import brentq


# ---------------------------------------------------------------------
# Helper kinematics (per nucleon)
# ---------------------------------------------------------------------

M_P_MEV = 938.2720813  # proton rest energy (MeV), used per nucleon

def beta_from_E(E_MeV_per_nuc: np.ndarray | float) -> np.ndarray:
    """
    β(E): speed / c from kinetic energy per nucleon (MeV/n).
    Uses relativistic γ = 1 + E / (m_p c^2) per nucleon.
    """
    E = np.asarray(E_MeV_per_nuc, dtype=float)
    gamma = 1.0 + E / M_P_MEV
    gamma = np.maximum(gamma, 1.0 + 1e-14)      # avoid γ=1 exactly (numeric noise)
    beta2 = 1.0 - 1.0 / (gamma * gamma)
    beta2 = np.clip(beta2, 0.0, 1.0)
    return np.sqrt(beta2)


def momentum_per_nucleon_MeV(E_MeV_per_nuc: np.ndarray | float) -> np.ndarray:
    """
    p(E) per nucleon in MeV/c from kinetic energy per nucleon.
    p = γ β m_p c  (with m_p c^2 in MeV)
    """
    E = np.asarray(E_MeV_per_nuc, dtype=float)
    gamma = 1.0 + E / M_P_MEV
    beta = beta_from_E(E)
    return gamma * beta * M_P_MEV  # MeV/c per nucleon

def _p_of_E_mev(E_mev_per_nuc: np.ndarray | float) -> float:
    """
    Relativistic 'pc' in MeV for a nucleon with kinetic energy per nucleon E (MeV/n):
        p(E) = sqrt( E * (E + 2 m_p c^2) )
    This matches the expression in the Ramaty caption you provided.
    """
    E = E_mev_per_nuc
    return np.sqrt(E * (E + 2.0 * M_P_MEV))


def rigidity_GV(E_MeV_per_nuc: np.ndarray | float, Z: int, A: int) -> np.ndarray:
    """
    Rigidity R (GV) from kinetic energy per nucleon, charge Z, mass A.

    R = (p c) / (Z e)
      = [ A * p_per_nucleon(MeV/c) ] / (Z * 1000)   in GV
    """
    p_per_nuc = momentum_per_nucleon_MeV(E_MeV_per_nuc)  # MeV/c per nucleon
    p_particle = p_per_nuc * float(A)                    # MeV/c per particle
    return (p_particle / 1000.0) / float(Z)              # convert MeV→GeV (1000), then divide by Z


# ---------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------

def _ensure_array(E: np.ndarray | float) -> np.ndarray:
    """Coerce to a 1D float64 numpy array (safe for scalar)."""
    arr = np.asarray(E, dtype=float)
    return np.atleast_1d(arr)


def _clamp_to_band(E: np.ndarray, Emin: float, Emax: float) -> np.ndarray:
    """Boolean mask for E within [Emin, Emax]."""
    return (E >= Emin) & (E <= Emax)


def _trapz_safe(y: np.ndarray, x: np.ndarray) -> float:
    """Wrapper around numpy.trapz returning float, with guard for zero-length."""
    if y.size < 2:
        return 0.0
    return float(np.trapz(y, x))


# ---------------------------------------------------------------------
# Base “interface”
# ---------------------------------------------------------------------

class _SpectrumBase:
    """
    Minimal base class to make type hints clearer and factor common code.

    Concrete subclasses must implement:
      - unnormalized(E)
      - _compute_normalization_constant()

    They inherit:
      - N(E): normalized PDF on [Emin, Emax]
      - Emin, Emax, C
    """

    Emin: float
    Emax: float
    C: float  # normalization constant

    def unnormalized(self, E: np.ndarray | float) -> np.ndarray:
        raise NotImplementedError

    def _compute_normalization_constant(self) -> float:
        # Numeric normalization over [Emin, Emax] with a log grid
        E = np.logspace(np.log10(30), np.log10(self.Emax), 2000)
        y = self.unnormalized(E)
        area = _trapz_safe(y, E)
        if area <= 0.0:
            return 0.0
        return 1.0 / area

    def N(self, E: np.ndarray | float) -> np.ndarray:
        """
        Normalized spectrum (pdf) over [Emin, Emax].
        Outside the band, returns 0.0 by design.
        """
        arr = _ensure_array(E)
        y = np.zeros_like(arr, dtype=float)
        mask = _clamp_to_band(arr, self.Emin, self.Emax)
        if not np.any(mask) or self.C <= 0.0:
            return y if isinstance(E, np.ndarray) else float(y.item())
        y[mask] = self.C * self.unnormalized(arr[mask])
        return y if isinstance(E, np.ndarray) else float(y.item())


# ---------------------------------------------------------------------
# 1) Stochastic K2 spectrum (flare photosphere)
# ---------------------------------------------------------------------

@dataclass
class StochasticK2(_SpectrumBase):
    r"""
    Build N(E) ∝ K2( 2 * sqrt( 3 p / (m_p c^2 * aT) ) )

    Parameters
    ----------
    aT : float
        The dimensionless spectral parameter in Ramaty (0.014–0.020 typical).
    Emax : float
        Upper cutoff for normalization/integration [MeV/n].

    Returns
    -------
    N : Callable[[float], float]
        Normalized spectrum N(E) [1/MeV per nucleon], such that ∫_{E0}^{Emax} N(E)dE ≈ 1.

    Notes
    -----
    - Uses m_p c^2 in MeV in the K2 argument, exactly as in the caption you shared.
    - p(E) is taken as sqrt(E(E+2 m_p c^2)) in MeV (i.e., 'pc' with c=1 in units).
    - The species index j only enters via abundances elsewhere; the K2 shape here
      is the same functional form Ramaty used (Bj handled via normalization).
    """

    aT: float
    Emin: float
    Emax: float

    def __post_init__(self):
        # Guard rails to avoid singularities:
        self.Emin = max(float(self.Emin), 1e-6)
        self.Emax = max(float(self.Emax), self.Emin * 1.0001)
        self.aT = max(float(self.aT), 1e-6)
        self.C = self._compute_normalization_constant()
        print(f"norm constant for stochastic C={self.C}")

    def unnormalized(self, E: np.ndarray | float) -> np.ndarray:
        arr = _ensure_array(E)
        p = _p_of_E_mev(arr)  # MeV
        arg = 2.0 * np.sqrt( (3.0 * p) / (M_P_MEV * self.aT) )
        return Knu(2.0, arg)


        # x = np.sqrt(arr / self.aT)        # dimensionless argument
        # # Safe K2 for small x: kv is well-behaved; we guard x>0
        # x = np.maximum(x, 1e-12)
        # return np.exp(-arr / self.aT) * Knu(2, x)


# ---------------------------------------------------------------------
# 2) Power-law spectrum (benchmark / CR-like)
# ---------------------------------------------------------------------

@dataclass
class PowerLaw(_SpectrumBase):
    """
    Pure power-law in energy per nucleon:

        φ(E) ∝ E^{-γ}   for E in [Emin, Emax]

    Notes
    -----
    - We *always* normalize numerically; no special casing for γ=1, etc.
    - If you want a low-energy cutoff, set Emin accordingly (e.g., 1 MeV/n).

    Parameters
    ----------
    gamma : float
        Spectral index (γ > 0 typical).
    Emin, Emax : float
        Normalization band (MeV/n).
    """

    gamma: float
    Emin: float
    Emax: float

    def __post_init__(self):
        self.Emin = max(float(self.Emin), 1e-12)
        self.Emax = max(float(self.Emax), self.Emin * 1.0001)
        self.gamma = float(self.gamma)
        self.C = self._compute_normalization_constant()
        print(f"norm constant for powerlaw C={self.C}")

    def unnormalized(self, E: np.ndarray | float) -> np.ndarray:
        arr = _ensure_array(E)
        arr = np.maximum(arr, 1e-30)  # avoid E=0
        return np.power(arr, -self.gamma)


# ---------------------------------------------------------------------
# 3) Shock spectrum in rigidity with exponential E-cutoff
# ---------------------------------------------------------------------

@dataclass
class ShockRigidity(_SpectrumBase):
    """
    Diffusive-shock-like spectrum in *rigidity* with an energy cutoff:

        φ(E) ∝ R(E; Z, A)^{-s} * exp(-E / E0_species)

    where:
        - s is the shock slope (s = (r+2)/(r-1) for compression ratio r)
        - E0_species is the cutoff *in energy per nucleon* for this species
        - R(E; Z, A) is the particle rigidity computed from E, charge Z, mass A

    For alphas (or other nuclei), E0_species can be derived from a proton cutoff E0p
    by vR-invariance: v(E0_i) * R(E0_i) = v(E0_p) * R(E0_p). We provide a helper.

    Parameters
    ----------
    s : float
        Shock slope in rigidity-space.
    E0_species : float
        Cutoff energy-per-nucleon for THIS species (MeV/n).
    Z, A : int
        Charge and mass number of the accelerated species (projectile).
    Emin, Emax : float
        Normalization band (MeV/n).
    """

    s: float
    E0_species: float
    Z: int
    A: int
    Emin: float
    Emax: float

    def __post_init__(self):
        self.Emin = max(float(self.Emin), 1e-6)
        self.Emax = max(float(self.Emax), self.Emin * 1.0001)
        self.s = float(self.s)
        self.E0_species = max(float(self.E0_species), 1e-6)
        self.Z = int(self.Z)
        self.A = int(self.A)
        self.C = self._compute_normalization_constant()

    def unnormalized(self, E: np.ndarray | float) -> np.ndarray:
        arr = _ensure_array(E)
        R = rigidity_GV(arr, self.Z, self.A)           # GV
        R = np.maximum(R, 1e-30)
        return np.power(R, -self.s) * np.exp(-arr / self.E0_species)

    # ---------- Helpers for cross-species cutoff via vR-invariance ----------

    @staticmethod
    def _vR(E_MeV_per_nuc: float, Z: int, A: int) -> float:
        """Return v(E) * R(E) at a scalar E (MeV/n), for given (Z, A)."""
        b = float(beta_from_E(E_MeV_per_nuc))
        R = float(rigidity_GV(E_MeV_per_nuc, Z, A))
        return b * R

    @classmethod
    def E0_species_from_proton(cls, E0p: float, Z: int, A: int,
                               bracket: Tuple[float, float] = (1e-3, 5e3)) -> float:
        """
        Given a proton cutoff E0p (MeV/n), find E0_species (MeV/n) such that:
            v(E0_species; Z,A) * R(E0_species; Z,A)  =  v(E0p; 1,1) * R(E0p; 1,1)
        The root is found in 'bracket' (MeV/n). Choose a wide bracket to be safe.
        """
        target = cls._vR(E0p, 1, 1)

        def f(E):
            return cls._vR(E, Z, A) - target

        # Root find with brentq; if bracket does not work, expand it
        a, b = bracket
        fa, fb = f(a), f(b)
        # Expand bracket if signs are not opposite (crude but robust)
        expand = 0
        while fa * fb > 0 and expand < 20:
            a *= 0.5
            b *= 2.0
            fa, fb = f(a), f(b)
            expand += 1

        if fa * fb > 0:
            # Fallback: return a proportional scaling in non-relativistic limit:
            # vR ∝ sqrt(E) * (A/Z) → E ~ (Z/A)^2 * E0p as a crude estimate
            return (float(Z) / float(A))**2 * float(E0p)

        return float(brentq(f, a, b))


# ---------------------------------------------------------------------
# 4) Composite linear mixture
# ---------------------------------------------------------------------

class CompositeSpectrum(_SpectrumBase):
    """
    Linear combination of spectra:
        φ_comp(E) = Σ w_i * φ_i_norm(E)

    We normalize the *mixture* again on [Emin, Emax] so that .N(E) is a proper
    pdf independent of the sum of weights (weights just shape the mixture).

    Parameters
    ----------
    components : List[Tuple[float, _SpectrumBase]]
        A list of (weight, spectrum). Each component spectrum should itself
        have well-defined [Emin, Emax] covering the mixture band.
    Emin, Emax : float
        Normalization band.
    """

    def __init__(self,
                 components: List[Tuple[float, _SpectrumBase]],
                 Emin: float,
                 Emax: float):
        self.components = list(components)
        self.Emin = max(float(Emin), 1e-6)
        self.Emax = max(float(Emax), self.Emin * 1.0001)
        self.C = self._compute_normalization_constant()

    def unnormalized(self, E: np.ndarray | float) -> np.ndarray | float:
        arr = _ensure_array(E)
        y = np.zeros_like(arr, dtype=float)
        for w, spec in self.components:
            # Use each component's normalized pdf within the composite band.
            y += float(w) * spec.N(arr)
        return y


# ---------------------------------------------------------------------
# Tiny self-test (optional)
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Quick smoke test and doc-style usage
    import matplotlib.pyplot as plt

    E = np.logspace(0, 2, 600)  # 1 to 100 MeV/n

    aT = 0.014
    gamma = 4
    st = StochasticK2(aT=aT, Emin=1.0, Emax=100.0)
    pl = PowerLaw(gamma=4.0, Emin=1.0, Emax=10000000000.0)

    E0a = ShockRigidity.E0_species_from_proton(E0p=30.0, Z=2, A=4)
    sh = ShockRigidity(s=3.3, E0_species=E0a, Z=2, A=4, Emin=1.0, Emax=500.0)

    comp = CompositeSpectrum(components=[(0.7, st), (0.3, sh)], Emin=1.0, Emax=100.0)

    plt.figure(figsize=(7,5))
    plt.loglog(E, st.N(E), label=f"StochasticK2 aT={aT} (norm)")
    plt.loglog(E, pl.N(E), label=f"PowerLaw γ={gamma} (norm)")
    plt.loglog(E, sh.N(E), label="Shock α (norm)")
    plt.loglog(E, comp.N(E), label="Composite 0.7·stoch + 0.3·shock")
    plt.xlabel("E (MeV/n)")
    plt.ylabel("N(E) [normalized]")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()