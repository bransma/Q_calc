import os
import numpy as np
from scipy.interpolate import interp1d, PchipInterpolator
from typing import Dict, Tuple, Callable, List

# Type aliases
CrossSectionData = Dict[str, Tuple[np.ndarray, np.ndarray]]
CrossSectionInterpolators = Dict[str, Callable[[float | np.ndarray], float]]

class CrossSectionReader:
    """
    Reads cross section CSV files from ./csv/.
    Interpolation method is configurable: "linear", "pchip", or "loglog".
    Ensures energies are strictly increasing (no sorting — fails if not).
    Clamps σ(E)=0 below first data point, and σ(E)=σ_last above last.
    """

    # --- All available keys ---
    all_keys: List[str] = [
        "3He4He_a6",
        "4He4He_a6","4He4He_a7",
        "4He12C_a6","4He12C_a7","4He12C_a9","4He12C_a10","4He12C_a11",
        "4He14N_a6","4He14N_a7","4He14N_a9","4He14N_a10","4He14N_a11",
        "4He16O_a6","4He16O_a7","4He16O_a9","4He16O_a10","4He16O_a11",
        "P12C_a6","P12C_a7","P12C_a9","P12C_a10","P12C_a11",
        "P13C_a6","P13C_a9","P13C_a10",
        "P14N_a6","P14N_a7","P14N_a9","P14N_a10","P14N_a11",
        "P16O_a6","P16O_a7","P16O_a9","P16O_a10","P16O_a11"
    ]

    # --- Projectile-target groups ---
    P12C = ["P12C_a6", "P12C_a7", "P12C_a9", "P12C_a10", "P12C_a11"]
    P13C = ["P13C_a6", "P13C_a9", "P13C_a10"]
    PN  = ["P14N_a6","P14N_a7","P14N_a9","P14N_a10","P14N_a11"]
    PO  = ["P16O_a6","P16O_a7","P16O_a9","P16O_a10","P16O_a11"]
    HeC = ["4He12C_a6","4He12C_a7","4He12C_a9","4He12C_a10","4He12C_a11"]
    HeN = ["4He14N_a6","4He14N_a7","4He14N_a9","4He14N_a10","4He14N_a11"]
    HeO = ["4He16O_a6","4He16O_a7","4He16O_a9","4He16O_a10","4He16O_a11"]
    HeHe = ["3He4He_a6","4He4He_a6","4He4He_a7"]

    # --- Product isotope groups ---
    A6  = ["3He4He_a6","4He4He_a6","4He12C_a6","4He14N_a6","4He16O_a6",
           "P12C_a6","P13C_a6","P14N_a6","P16O_a6"]
    A7  = ["4He4He_a7","4He12C_a7","4He14N_a7","4He16O_a7",
           "P12C_a7","P14N_a7","P16O_a7"]
    A9  = ["4He12C_a9","4He14N_a9","4He16O_a9",
           "P12C_a9","P13C_a9","P14N_a9","P16O_a9"]
    A10 = ["4He12C_a10","4He14N_a10","4He16O_a10",
           "P12C_a10","P13C_a10","P14N_a10","P16O_a10"]
    A11 = ["4He12C_a11","4He14N_a11","4He16O_a11",
           "P12C_a11","P14N_a11","P16O_a11"]

    def __init__(self, csv_dir: str = "./csv", interp_kind: str = "linear"):
        self.csv_dir = csv_dir
        self.interp_kind = interp_kind.lower()
        self.cross_section_data_mb: CrossSectionData = {}
        self.cross_section_interp: CrossSectionInterpolators = {}

    def load_csv(self, key: str) -> None:
        """Load csv/{key}.csv into memory and build interpolator."""
        # tested
        filename = os.path.join(self.csv_dir, f"{key}.csv")
        data = np.loadtxt(filename, delimiter=",")
        energies, sigma = data[:, 0], data[:, 1]

        # --- Strict monotonicity check (no sorting) ---
        diffs = np.diff(energies)
        if not np.all(diffs > 0):
            bad_idx = np.where(diffs <= 0)[0]
            raise ValueError(
                f"[ERROR] Non-increasing energies in {filename} at rows {bad_idx}:\n"
                f"{energies[bad_idx]} -> {energies[bad_idx+1]}"
            )

        # Deduplicate (keep first occurrence if duplicates exist)
        unique_E, idx = np.unique(energies, return_index=True)
        energies, sigma = unique_E, sigma[idx]

        # Save raw data
        self.cross_section_data_mb[key] = (energies, sigma)

        # --- Choose interpolation method ---
        if self.interp_kind == "linear":
            f_interp = interp1d(
                energies, sigma, kind="linear",
                bounds_error=False, fill_value=(0.0, sigma[-1])
            )
        elif self.interp_kind == "pchip":
            f_interp = PchipInterpolator(energies, sigma, extrapolate=True)
        elif self.interp_kind == "loglog":
            # log-log interpolation (assumes σ > 0)
            logE, logs = np.log10(energies[energies > 0]), np.log10(sigma[energies > 0])
            f_base = interp1d(logE, logs, kind="linear", bounds_error=False,
                              fill_value=(-np.inf, logs[-1]))
            def f_interp(x):
                x = np.asarray(x)
                y = np.full_like(x, 0.0, dtype=float)
                mask = x > 0
                y[mask] = 10**f_base(np.log10(x[mask]))
                return y
        else:
            raise ValueError(f"Unknown interp_kind: {self.interp_kind}")

        # Wrap with clamping
        def safe_interp(E):
            E = np.asarray(E)
            result = f_interp(E)
            result[E < energies[0]] = 0.0
            result[E > energies[-1]] = sigma[-1]
            return result

        self.cross_section_interp[key] = safe_interp

    def sigma(self, key: str, E: float | np.ndarray) -> float | np.ndarray:
        """Evaluate σ(E) for a given reaction channel."""
        return self.cross_section_interp[key](E)

    def read_all(self) -> None:
        """Load all cross sections listed in all_keys."""
        for key in self.all_keys:
            self.load_csv(key)