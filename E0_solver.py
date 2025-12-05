from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
import numpy as np

class E0Solver:
    """
    Given S(E), build R(E)=∫ dE'/S(E') and its inverse. Then
        E0(E, X) = R^{-1}( R(E) + X ).
    """

    def __init__(self, stopping_power, E_grid: np.ndarray):
        self.S = stopping_power
        self.E_grid = np.asarray(E_grid, float)

        invS = 1.0 / self.S(self.E_grid)              # vectorized 1/S(E)
        R_vals = cumtrapz(invS, self.E_grid, initial=0.0)
        self.R_vals = R_vals
        self.R_min, self.R_max = float(R_vals[0]), float(R_vals[-1])

        # E → R(E)
        self.R_of_E = interp1d(self.E_grid, R_vals, bounds_error=False,
                               fill_value=(self.R_min, self.R_max))
        # R → E(R)
        self.E_of_R = interp1d(R_vals, self.E_grid, bounds_error=False,
                               fill_value=(self.E_grid[0], self.E_grid[-1]))

    def get_E0(self, E: float | np.ndarray, X: float | np.ndarray) -> np.ndarray:
        """
        Compute entry energy at the surface for a particle of energy E
        observed after traversing column depth X.

        Steps:
          1) R_target = R(E) + X
          2) E0 = E(R_target)
          3) enforce E0 ≥ E (numerical safety)
        """
        E = np.asarray(E, float)
        R_target = self.R_of_E(E) + X
        R_target = np.clip(R_target, self.R_min, self.R_max)
        E0 = self.E_of_R(R_target)
        return np.maximum(E0, E)