"""
neutral_medium_plot_utils.py
----------------------------------------------------------------------
Plotting Utility for Neutral Medium Stopping Power
----------------------------------------------------------------------

Visualizes the stopping power (energy loss rate) and effective charge
for ions traversing a *neutral* astrophysical medium (e.g., solar
chromosphere).  Implements twin–axis log plots with consistent scaling
and physical annotation, suitable for publication figures.

The left y–axis shows |dE/dx| [MeV g⁻¹ cm²].
The right y–axis shows Z_eff(E) / Z, the degree of ionization fraction.
----------------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
from stopping_power_neutral_medium import NeutralMediumStoppingPower
from stopping_power_bethe_bloch import StoppingPowerBetheBloch


# ======================================================================
# Utility Function
# ======================================================================
def plot_stopping_powers(
    species: list,
    labels: list[str] | None = None,
    Emin: float = 1.0,
    Emax: float = 1e3,
    npts: int = 400,
    figsize: tuple[float, float] = (9, 6),
) -> None:
    """
    Plot |dE/dx| for one or more species.

    Parameters
    ----------
    species_nm : list[NeutralMediumStoppingPower, StoppingPowerBetheBloch]
        List of initialized stopping power models.
    labels : list[str], optional
        Optional list of labels for the legend.
    Emin, Emax : float
        Energy range [MeV/n].
    npts : int
        Number of log-spaced energy samples.
    figsize : tuple
        Figure size in inches.
    """

    # Create energy grid
    E = np.logspace(np.log10(Emin), np.log10(Emax), npts)

    # ------------------------------------------------------------------
    # Create dual–axis figure: left = |dE/dx|, right = Z_eff/Z
    # ------------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=figsize)

    colors = ["tab:red", "tab:blue", "tab:green", "tab:purple", "tab:orange", "tab:cyan"]

    for i, (sp, lbl) in enumerate(zip(species, labels)):
        c = colors[i % len(colors)]

        # --- Stopping power ---
        dedx = sp.dEdx(E)
        ax1.loglog(
            E,
            dedx,
            color=c,
            lw=2,
            label=f"{lbl}  (|dE/dx|)",
        )

    # ------------------------------------------------------------------
    # Axis labels, titles, and formatting
    # ------------------------------------------------------------------
    ax1.set_xlabel("Energy per nucleon [MeV/n]", fontsize=12)
    ax1.set_ylabel(r"|dE/dx| [MeV g$^{-1}$ cm$^{2}$]", color="tab:red", fontsize=12)

    ax1.grid(True, which="both", ls="--", alpha=0.4)
    ax1.set_xlim(Emin, Emax)
    ax1.set_ylim(0, 5000)
    #ax1.set_ylim(0, 1000)

    ax1.tick_params(axis="y", colors="tab:red")

    ax1.set_title(
        "Stopping Power Compairison",
        fontsize=14,
        pad=12,
    )

    # Combined legend (merge both axes handles)
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    ax1.legend(lines_1, labels_1, loc="lower left", fontsize=9)

    plt.tight_layout()
    plt.show()


# ======================================================================
# Example Driver
# ======================================================================
if __name__ == "__main__":
    # Define projectiles of astrophysical interest
    species = [
        NeutralMediumStoppingPower(Z=1, A=1),  # proton
        NeutralMediumStoppingPower(Z=2, A=3),  # ³He
        NeutralMediumStoppingPower(Z=2, A=4),  # ⁴He
        StoppingPowerBetheBloch(z_proj=1),  # proton
        StoppingPowerBetheBloch(z_proj=2),  # ³He
        StoppingPowerBetheBloch(z_proj=2),  # ⁴He
    ]
    labels = ["p_nm", "³He_nm", "⁴He_nm", "p_bb", "³He_bb", "⁴He_bb"]

    # species = [
    #     #NeutralMediumStoppingPower(Z=1, A=1),  # proton
    #     #NeutralMediumStoppingPower(Z=2, A=3),  # ³He
    #     NeutralMediumStoppingPower(Z=2, A=4),  # ⁴He
    #     #StoppingPowerBetheBloch(z_proj=1),  # proton
    #     #StoppingPowerBetheBloch(z_proj=2),  # ³He
    #     StoppingPowerBetheBloch(z_proj=2),  # ⁴He
    # ]
    # labels = ["⁴He_nm", "⁴He_bb"]

    plot_stopping_powers(species, labels)