"""
propagation_csda_core.py

CSDA (continuous slowing-down approximation) transport in a non–depth-varying
medium, WITHOUT catastrophic losses, using an externally supplied E0 solver.

Physics model
-------------
We solve the steady-state 1D transport in column depth X [g cm^-2]:

    ∂φ/∂X - ∂(S(E) φ)/∂E = 0,             (no catastrophic term)

with boundary condition φ(E, X=0) = φ₀(E).

Along characteristics (constant range), the solution is:

    φ(E, X) = φ₀(E₀) * S(E₀) / S(E),      (1)

where:
    - S(E) = - dE/dX is the mass stopping power [MeV cm^2 g^-1].
    - E₀ = E₀(E, X) is the "entry" energy at X=0 of a particle
      observed with energy E at depth X. It is defined implicitly
      via the range function:

          R(E) = ∫^E dE' / S(E')          (2)

      and

          E₀(E, X) = R^{-1}( R(E) + X ).  (3)

Here we *do not* compute R(E) or R^{-1} ourselves; instead we require
an E0 solver object supplied by the caller. This keeps the propagation
code focused and lets you plug in whatever E0 machinery you prefer.

Reaction yield (Q)
------------------
Given a cross section σ(E) for a particular channel, the depth-local
production rate per unit column depth is

    q(X) = ∫ φ(E, X) σ(E) dE.             (4)

In discrete form on an energy grid {E_k}:

    q(X_m) ≈ Σ_k φ(E_k, X_m) σ(E_k) ΔE_k. (5)

You can then integrate q(X) over X to get a depth-integrated Q:

    Q_tot = ∫_0^{X_max} q(X) dX.          (6)

This module provides:
    - A CSDAPropagator class to compute φ(E, X) on a fixed energy grid.
    - A method to compute q(X) via (5) at a given depth.
    - A method to compute Q_tot and q(X) profile on a chosen X grid.

All the physics of S(E), φ₀(E), σ(E), and E₀(E,X) are supplied by you.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Protocol
import numpy as np


# -----------------------------------------------------------------------------
# Type protocol for the E0 solver
# -----------------------------------------------------------------------------

class E0SolverProtocol(Protocol):
    """
    Protocol (interface) for an E0 solver.

    Requirements
    ------------
    Any E0 solver you pass in must implement:

        get_E0(E: np.ndarray, X: float | np.ndarray) -> np.ndarray

    where:
        - E is an array of observed energies at depth X [MeV/n].
        - X is a scalar or array of the same shape (column depth [g cm^-2]).
        - The return value is an array of E₀(E, X) [MeV/n] of the same shape.

    That is exactly the "ramp inversion" map:

        E₀(E, X) = R^{-1}( R(E) + X )
    """
    def get_E0(self, E: np.ndarray, X: float | np.ndarray) -> np.ndarray:
        ...


# -----------------------------------------------------------------------------
# CSDA propagator core
# -----------------------------------------------------------------------------

@dataclass
class CSDAPropagator:
    """
    CSDA propagator in a fixed medium, without catastrophic loss.

    Parameters
    ----------
    E_grid : np.ndarray
        Strictly increasing array of kinetic energies per nucleon [MeV/n].
        This is the grid on which all φ(E, X) values are computed.
    stopping_power : Callable[[np.ndarray], np.ndarray]
        Function S(E) giving the mass stopping power [MeV cm^2 g^-1].
        Must accept an array of E and return array of same shape.
    phi0 : Callable[[np.ndarray], np.ndarray]
        Injection spectrum φ₀(E) at X=0, defined on [E_min, E_max].
        Units can be arbitrary as long as everything is consistent.
    sigma : Callable[[np.ndarray], np.ndarray]
        Cross section σ(E) for the reaction channel of interest.
        Typically in mb, but any units are fine for the core math.
    e0_solver : E0SolverProtocol
        Object implementing get_E0(E, X) → E₀(E, X) as described above.
    Emin : float
        Lower limit of energy integration [MeV/n].
    Emax : float
        Upper limit of energy integration [MeV/n].

    Notes
    -----
    - This class does NOT know about catastrophic losses (no Λ(E)).
      It is strictly the CSDA (stopping-power-only) transport.
    - It treats the integrals over E using simple trapezoidal quadrature
      on the provided E_grid, restricted to [Emin, Emax].
    - The role of E₀(E, X) is purely geometric: mapping a point (E, X)
      back along the "ramp" to find the entry energy at X=0.
    """

    E_grid: np.ndarray
    stopping_power: Callable[[np.ndarray], np.ndarray]
    phi0: Callable[[np.ndarray], np.ndarray]
    sigma: Callable[[np.ndarray], np.ndarray]
    e0_solver: E0SolverProtocol
    Emin: float
    Emax: float

    # internal masks and cached grid subsets
    _mask: np.ndarray | None = None
    _E_int: np.ndarray | None = None
    _dE: np.ndarray | None = None

    def __post_init__(self):
        """
        Basic validation and precomputation of integration mask and ΔE.
        """
        E = np.asarray(self.E_grid, dtype=float)
        if E.ndim != 1:
            raise ValueError("E_grid must be a 1D array.")
        if not np.all(np.diff(E) > 0):
            raise ValueError("E_grid must be strictly increasing.")

        # Restrict integration to [Emin, Emax]
        mask = (E >= self.Emin) & (E <= self.Emax)
        if not np.any(mask):
            raise ValueError("No E_grid points lie inside [Emin, Emax].")

        self._mask = mask
        self._E_int = E[mask]

        # Precompute ΔE_k for trapezoidal integration on the restricted grid.
        # If E = [E0, E1, ..., En], then
        #   ∫ f(E) dE ≈ Σ_k 0.5 * (f_k + f_{k+1}) * (E_{k+1} - E_k).
        dE_full = np.diff(E)
        # For convenience, we'll store an array of length len(E_int)-1
        # corresponding to the interior intervals.
        # We'll apply these to φ(E_k, X)*σ(E_k) restricted by the same mask.
        int_indices = np.nonzero(mask)[0]
        # consecutive indices in the full grid
        E_int_indices = int_indices
        if E_int_indices.size < 2:
            raise ValueError("Need at least 2 points in [Emin, Emax] to integrate.")

        # ΔE on the *restricted* grid
        dE_int = np.diff(E[E_int_indices])
        self._dE = dE_int

    # -------------------------------------------------------------------------
    # Core: compute φ(E, X) on the E_grid
    # -------------------------------------------------------------------------

    def phi_E_X(self, X: float | np.ndarray) -> np.ndarray:
        """
        Compute the transported spectrum φ(E, X) on the full E_grid.

        Parameters
        ----------
        X : float or np.ndarray
            Column depth [g cm^-2]. May be a scalar or an array.
            - If scalar, φ is returned as shape (N_E,).
            - If 1D array of length N_X, φ is returned as shape (N_X, N_E).

        Returns
        -------
        phi : np.ndarray
            Values of φ(E, X) on E_grid (and for each X if X is an array).

        Physics
        -------
        Uses the CSDA solution:

            φ(E, X) = φ₀(E₀) * S(E₀)/S(E),

        where E₀(E, X) is obtained from e0_solver.get_E0(E, X).
        """
        E = self.E_grid
        X_arr = np.asarray(X, dtype=float)

        if X_arr.ndim == 0:
            # Single depth X: produce shape (N_E,)
            E0 = self.e0_solver.get_E0(E, float(X_arr))
            S_E0 = self.stopping_power(E0)
            S_E  = self.stopping_power(E)
            # Avoid division by zero by assuming stopping_power is checked upstream.
            ratio = S_E0 / S_E
            return self.phi0(E0) * ratio

        elif X_arr.ndim == 1:
            # Multiple depths: produce shape (N_X, N_E)
            N_X = X_arr.size
            N_E = E.size
            phi = np.zeros((N_X, N_E), dtype=float)
            for i, Xi in enumerate(X_arr):
                E0 = self.e0_solver.get_E0(E, float(Xi))
                S_E0 = self.stopping_power(E0)
                S_E  = self.stopping_power(E)
                ratio = S_E0 / S_E
                phi[i, :] = self.phi0(E0) * ratio
            return phi

        else:
            raise ValueError("X must be scalar or 1D array.")

    # -------------------------------------------------------------------------
    # Isolated function: Q(Emin, Emax; X) = ∫ φ(E,X) σ(E) dE
    # -------------------------------------------------------------------------

    def energy_integral_at_depth(self, X: float) -> float:
        """
        Compute the depth-local reaction rate per unit column depth:

            q(X) = ∫_{Emin}^{Emax} φ(E, X) σ(E) dE         (Eq. 5)

        using trapezoidal quadrature on the restricted [Emin, Emax] grid.

        Parameters
        ----------
        X : float
            Column depth [g cm^-2] at which to evaluate q(X).

        Returns
        -------
        qX : float
            Approximate value of q(X) at this depth.
            Units: whatever φ * σ * dE gives you (e.g., reactions per proton per g cm^-2).
        """
        E_full = self.E_grid
        mask   = self._mask
        E_int  = self._E_int
        dE_int = self._dE

        # φ(E, X) on full grid, then restrict to [Emin, Emax]
        phi_full = self.phi_E_X(X)          # shape (N_E,)
        phi_int  = phi_full[mask]           # shape (N_int,)
        sigma_int = self.sigma(E_int)       # same shape

        # Integrand on restricted grid: f_k = φ(E_k, X) σ(E_k)
        f = phi_int * sigma_int             # shape (N_int,)

        # Trapezoidal integration on the restricted grid:
        # ∫ f(E) dE ≈ Σ 0.5 * (f_k + f_{k+1}) * ΔE_k
        f_left  = f[:-1]
        f_right = f[1:]
        integral = np.sum(0.5 * (f_left + f_right) * dE_int)

        return float(integral)

    # -------------------------------------------------------------------------
    # Integrate q(X) over depth X
    # -------------------------------------------------------------------------

    def depth_integrated_yield(self, X_grid: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Compute:
            - the depth-local rate q(X_m) at each X_m in X_grid
            - the total yield Q_tot = ∫ q(X) dX over [X_min, X_max]

        using trapezoidal integration over the supplied X_grid.

        Parameters
        ----------
        X_grid : np.ndarray
            1D array of depths [g cm^-2]. Must be strictly increasing.

        Returns
        -------
        q_of_X : np.ndarray
            Array of q(X_m) values for each depth in X_grid.
        Q_tot : float
            Approximate ∫ q(X) dX over the span of X_grid.

        Physics
        -------
        Each q(X_m) is the clipboard count for the slab around X_m:

            q(X_m) ≈ ∫ φ(E, X_m) σ(E) dE,

        and Q_tot is the integrated production per projectile over the
        entire column from X_min to X_max.
        """
        X = np.asarray(X_grid, dtype=float)
        if X.ndim != 1:
            raise ValueError("X_grid must be 1D.")
        if not np.all(np.diff(X) > 0):
            raise ValueError("X_grid must be strictly increasing.")

        # Compute q(X_m) for each depth
        q_of_X = np.array([self.energy_integral_at_depth(Xm) for Xm in X], dtype=float)

        # Integrate q(X) over X using trapezoidal rule
        dX = np.diff(X)
        q_left  = q_of_X[:-1]
        q_right = q_of_X[1:]
        Q_tot = np.sum(0.5 * (q_left + q_right) * dX)

        return q_of_X, float(Q_tot)

# -----------------------------------------------------------------------------
# Demo main: toy ramp model with S(E) = 1/E and φ0 ∝ E^-2
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Minimal demonstration of the CSDAPropagator using an analytic toy model:

        S(E) = 1/E                         (stopping power)
        φ0(E) ∝ E^-2                       (injection spectrum)
        σ(E) = const.                      (cross section)

    For S(E) = 1/E, the range is:

        R(E) = ∫_0^E E' dE' = E^2 / 2,

    so the exact E₀ inversion is analytic:

        E0(E, X) = sqrt( E^2 + 2X ).

    This "E0SolverToy" implements that relation, and we use it to build
    a CSDAPropagator instance and evaluate:

        q(X) = ∫ φ(E, X) σ(E) dE,

    and the depth-integrated Q_tot for a small X-grid.
    """

    # -------------------------
    # 1. Define E0 solver
    # -------------------------
    class E0SolverToy:
        """
        Analytic E0 solver for the toy stopping power S(E) = 1/E.

        Physics:
            R(E) = ∫ dE'/S(E') = ∫ E' dE' = E^2 / 2
            E0(E, X) satisfies: R(E0) = R(E) + X
            => E0^2 / 2 = E^2 / 2 + X
            => E0(E, X) = sqrt(E^2 + 2X)
        """

        def get_E0(self, E: np.ndarray, X: float | np.ndarray) -> np.ndarray:
            E = np.asarray(E, dtype=float)
            X_arr = np.asarray(X, dtype=float)
            # Broadcast X to match E if scalar
            if X_arr.ndim == 0:
                X_arr = np.full_like(E, float(X_arr))
            # E0(E, X) = sqrt(E^2 + 2X)
            E0 = np.sqrt(np.maximum(E**2 + 2.0 * X_arr, 0.0))
            return E0

    # -------------------------
    # 2. Define S(E), φ0(E), σ(E)
    # -------------------------

    def stopping_power_toy(E: np.ndarray) -> np.ndarray:
        """
        Toy stopping power:
            S(E) = 1/E

        This is purely for demonstration and has dimensions stripped out;
        it captures the qualitative behavior that higher-energy particles
        lose energy more slowly (per unit grammage) than low-energy ones.
        """
        E = np.asarray(E, dtype=float)
        # Avoid division by zero at very small energies
        return 1.0 / np.maximum(E, 1e-12)

    def phi0_powerlaw(E: np.ndarray, s: float = 2.0) -> np.ndarray:
        """
        Injection spectrum at X = 0:

            φ0(E) ∝ E^(-s), E >= Emin_norm

        For this demo we do not normalize explicitly; the relative shape
        is sufficient to illustrate how φ(E, X) and q(X) behave with depth.
        """
        E = np.asarray(E, dtype=float)
        phi = np.zeros_like(E)
        mask = E > 0.0
        phi[mask] = E[mask] ** (-s)
        return phi

    def sigma_constant(E: np.ndarray) -> np.ndarray:
        """
        Constant cross section σ(E) = 1 [arb. units].

        This makes q(X) ∝ ∫ φ(E, X) dE, so we are effectively probing how
        the spectrum φ(E, X) evolves with depth.
        """
        E = np.asarray(E, dtype=float)
        return np.ones_like(E)

    # -------------------------
    # 3. Build the propagator
    # -------------------------

    # Energy grid: from 1 to 100 in arbitrary units, log-spaced
    E_grid = np.logspace(0, 2, 200)   # 10^0 .. 10^2

    # Integration limits in energy
    Emin = 1.0   # lower bound
    Emax = 100.0 # upper bound (same as grid max here)

    # Instantiate the toy E0 solver
    e0_solver = E0SolverToy()

    # Build the propagator
    propagator = CSDAPropagator(
        E_grid=E_grid,
        stopping_power=stopping_power_toy,
        phi0=phi0_powerlaw,
        sigma=sigma_constant,
        e0_solver=e0_solver,
        Emin=Emin,
        Emax=Emax,
    )

    # -------------------------
    # 4. Define depth grid and compute q(X) and Q_tot
    # -------------------------

    # Depth grid in "ramp length" units (same X that appears in E0(E, X))
    X_grid = np.linspace(0.0, 10.0, 11)  # X = 0,1,2,...,10

    # Compute q(X) and depth-integrated Q_tot
    q_of_X, Q_tot = propagator.depth_integrated_yield(X_grid)

    # -------------------------
    # 5. Print results
    # -------------------------
    print("Toy CSDA propagation demo (S(E) = 1/E, φ0 ∝ E^-2, σ = 1)")
    print("--------------------------------------------------------")
    print(f"Energy grid: Emin = {Emin:.2f}, Emax = {Emax:.2f}, N_E = {E_grid.size}")
    print(f"Depth grid:  X_min = {X_grid[0]:.2f}, X_max = {X_grid[-1]:.2f}, N_X = {X_grid.size}")
    print()
    print("Depth X    q(X)  (arbitrary units)")
    for Xm, qXm in zip(X_grid, q_of_X):
        print(f"{Xm:7.2f}  {qXm:12.6e}")
    print()
    print(f"Depth-integrated yield Q_tot ≈ {Q_tot:.6e} (arb. units)")