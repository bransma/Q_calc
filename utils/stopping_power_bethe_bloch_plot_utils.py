"""
stopping_power_bethe_bloch_plot_utils.py
----------------------------------------------------------------------
Visualization utility for Bethe–Bloch stopping power calculations.

This module provides a CLI to visualize |dE/dX| as a function of energy
for ions propagating through a partially ionized medium using the
Bethe–Bloch formalism

Red curve  : |dE/dX| [MeV g⁻¹ cm²]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from stopping_power_bethe_bloch import StoppingPowerBetheBloch

# ======================================================================
# Core plotting function
# ======================================================================
def plot_bethe_bloch_stopping(
    bb_model: StoppingPowerBetheBloch,
    Emin: float = 1.0,
    Emax: float = 1e3,
    npts: int = 400,
    figsize: tuple[float, float] = (9, 6),
) -> None:
    """
    Plot Bethe–Bloch stopping power (|dE/dx|) and Z_eff/Z.

    Parameters
    ----------
    bb_model : BetheBlochStoppingPower
        Configured stopping power model.
    Emin, Emax : float
        Energy range [MeV/n].
    npts : int
        Number of log-spaced energy samples.
    figsize : tuple
        Figure size in inches.
    """
    E = np.logspace(np.log10(Emin), np.log10(Emax), npts)

    # Compute stopping power
    dEdx_vals = np.array([bb_model.dEdx(Ei) for Ei in E])

    # Plot setup
    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.loglog(E, dEdx_vals, color="tab:red", lw=2, label="|dE/dx|")

    ax1.set_xlabel("Energy per nucleon [MeV/n]", fontsize=12)
    ax1.set_ylabel(r"|dE/dx| [MeV g$^{-1}$ cm$^{2}$]", color="tab:red", fontsize=12)
    ax1.tick_params(axis="y", colors="tab:red")

    ax1.grid(True, which="both", ls="--", alpha=0.4)
    ax1.set_xlim(Emin, Emax)
    ax1.set_ylim(0, 1e5)

    ax1.set_title(
        f"Bethe–Bloch Stopping Power",
        fontsize=14,
        pad=12,
    )

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    ax1.legend(lines_1, labels_1, loc="lower left", fontsize=9)

    plt.tight_layout()
    plt.show()


# ======================================================================
# Command Line Interface
# ======================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot Bethe–Bloch stopping power with optional variable depth."
    )
    parser.add_argument(
        "--variable-depth",
        action="store_true",
        help="Enable toy depth dependence (I_mix, Z_mix, A_mix vary with column depth).",
    )
    args = parser.parse_args()

    # Example species (p, 3He, 4He)
    models = [
        StoppingPowerBetheBloch(z_proj=1),
        StoppingPowerBetheBloch(z_proj=2),
        StoppingPowerBetheBloch(z_proj=2),
    ]

    for model in models:
        plot_bethe_bloch_stopping(model)