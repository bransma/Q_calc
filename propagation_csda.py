# propagation_csda.py
# ---------------------------------------------------------------------
# CSDA transport in a non–depth-varying medium, WITHOUT catastrophic loss.
#
# PDE (steady-state, 1D column X, continuous slowing down):
#   ∂φ/∂X - ∂(S φ)/∂E = 0
#
# Characteristic solution:
#   φ(E, X) = φ0(E0) * S(E0)/S(E),
#   with  E0 = R^{-1}( R(E) + X ),   R(E) = ∫^E dE'/S(E').
#
# This file:
#   - provides a Propagator that uses an E0 (range-inversion) solver
#   - can import your E0_solver.E0Solver if available
#   - otherwise falls back to an internal E0 solver with same API
# ---------------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np

# --- Try to use user's E0_solver if available; otherwise define a compatible fallback
try:
    from E0_solver import E0Solver  # expected API: E0Solver(stopping_power, E_grid); .get_E0(E, X)
    _HAS_EXTERNAL_E0 = True
except Exception:
    _HAS_EXTERNAL_E0 = False
    from scipy.interpolate import PchipInterpolator

    class E0Solver:
        """
        Fallback E0 solver: builds R(E)=∫ dE'/S, its inverse E(R), and returns E0 = E(R(E)+X).
        Compatible with the expected external E0_solver API: .get_E0(E, X).
        """
        def __init__(self, stopping_power: Callable[[np.ndarray], np.ndarray], E_grid: np.ndarray):
            E = np.asarray(E_grid, dtype=float)
            if not np.all(np.diff(E) > 0):
                E, _ = np.unique(E, return_index=True)
            if not np.all(np.diff(E) > 0):
                raise ValueError("E_grid must be strictly increasing.")
            self.E = E
            S = np.asarray(stopping_power(E), dtype=float)
            if np.any(~np.isfinite(S)) or np.any(S <= 0):
                raise ValueError("Stopping power S(E) must be positive and finite on E_grid.")
            # Build cumulative range R(E) with trapezoidal accumulation
            dE = np.diff(E)
            invS = 1.0 / S
            R_vals = np.zeros_like(E)
            R_vals[1:] = np.cumsum(0.5 * (invS[:-1] + invS[1:]) * dE)

            # Monotonic maps
            self._R_of_E = PchipInterpolator(E, R_vals, extrapolate=True)
            self._E_of_R = PchipInterpolator(R_vals, E, extrapolate=True)
            self.Rmin, self.Rmax = float(R_vals[0]), float(R_vals[-1])
            # Store an interpolant for S so we can evaluate S(E0) robustly
            self._S = PchipInterpolator(E, S, extrapolate=True)

        def get_E0(self, E: np.ndarray, X: np.ndarray | float) -> np.ndarray:
            E = np.asarray(E, dtype=float)
            X = np.asarray(X, dtype=float)
            if X.size == 1:
                X = np.full_like(E, float(X))
            R_target = self._R_of_E(E) + X
            # We allow mild extrapolation; if you prefer strict clamping:
            # R_target = np.clip(R_target, self.Rmin, self.Rmax)
            return np.asarray(self._E_of_R(R_target), dtype=float)

        def S(self, E: np.ndarray) -> np.ndarray:
            return np.asarray(self._S(E), dtype=float)


@dataclass
class CSDAPropagator:
    """
    Pure-CSDA propagator (no catastrophic survival), using an E0 solver.

    Parameters
    ----------
    E_grid : array
        Strictly-increasing per-nucleon energy grid [MeV/n] to tabulate R(E).
    S_of_E : callable
        Mass stopping power S(E) = -(dE/dX) [MeV cm^2 g^-1].
    phi0_of_E : callable
        Injection spectrum φ0(E) at X=0 (often normalized above 30 MeV/n).

    Notes
    -----
    Solution used:
        φ(E,X) = φ0(E0) * S(E0)/S(E),  E0 = R^{-1}(R(E)+X).
    """
    E_grid: np.ndarray
    S_of_E: Callable[[np.ndarray], np.ndarray]
    phi0_of_E: Callable[[np.ndarray], np.ndarray]

    _e0_solver: Optional[E0Solver] = None

    def __post_init__(self):
        if self._e0_solver is None:
            self._e0_solver = E0Solver(self.S_of_E, self.E_grid)

    def phi(self, E: np.ndarray, X: np.ndarray | float) -> np.ndarray:
        """
        Evaluate φ(E,X) on arbitrary E at a single X or array of X.
        """
        E = np.asarray(E, dtype=float)
        E0 = self._e0_solver.get_E0(E, X)
        S_ratio = np.asarray(self._e0_solver.S(E0), dtype=float) / np.asarray(self._e0_solver.S(E), dtype=float)
        return np.asarray(self.phi0_of_E(E0), dtype=float) * S_ratio

    def phi_over_depths(self, E: np.ndarray, X_list: np.ndarray) -> np.ndarray:
        E = np.asarray(E, dtype=float)
        X_list = np.asarray(X_list, dtype=float)
        out = np.zeros((X_list.size, E.size), dtype=float)
        for k, X in enumerate(X_list):
            out[k, :] = self.phi(E, X)
        return out


# ----------------------------- sanity main ----------------------------------
if __name__ == "__main__":
    # Toy-but-physical S(E): simple Bethe-like shape (monotonic guard)
    def S_of_E(E):
        E = np.asarray(E, dtype=float)
        b2 = E * (E + 2*938.27208816) / (E + 938.27208816)**2
        b2 = np.clip(b2, 1e-6, 1.0)
        I = 19.31e-6
        arg = np.maximum((2*0.51099895) * (E*(E+2*938.27208816)/938.27208816**2) / I, 1.0000001)
        bracket = np.log(arg) - b2
        S = 0.307075 * 1.0**2 * 0.865 * (1.0/b2) * bracket
        return np.maximum(S, 1e-12)

    # φ0(E): normalized power law above 30 MeV/n (s=2.5)
    from scipy.integrate import quad
    Emin = 30.0
    s = 2.5
    norm = 1.0 / quad(lambda x: x**(-s), Emin, np.inf, limit=200)[0]
    phi0 = lambda E: norm * (np.where(E >= Emin, E**(-s), 0.0))

    Egrid = np.logspace(0, 3, 600)  # 1 .. 1000 MeV/n
    prop  = CSDAPropagator(E_grid=Egrid, S_of_E=S_of_E, phi0_of_E=phi0)

    E_eval = np.logspace(0, 3, 200)
    Xs     = np.array([0.0, 5.0, 10.0, 20.0])
    Phi    = prop.phi_over_depths(E_eval, Xs)

    # Sanity: φ decreases with X at fixed E
    assert np.all(Phi[1:, :] <= Phi[:-1, :] + 1e-12)
    print("propagation_csda: sanity checks passed ✓")