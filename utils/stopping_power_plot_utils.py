# utils/stopping_power_plot_utils.py

import numpy as np
import matplotlib.pyplot as plt

from stopping_power import NeutralMediumStoppingPower, LeakyBoxStoppingPower


class StoppingPowerPlotUtils:
    """
    Plot utilities for stopping power models.

    Supports:
    - NeutralMediumStoppingPower: plots dE/dx vs E.
    - LeakyBoxStoppingPower: plots dE/dx vs E, survival probability vs E,
      or both on a dual-axis plot.
    """

    def __init__(self, Emin: float = 0.5, Emax: float = 500.0, npoints: int = 500):
        # Energy grid [MeV/n]
        self.E = np.logspace(np.log10(Emin), np.log10(Emax), npoints)

    # -------- Neutral medium --------
    def plot_neutral_dedx(self, Z: int = 1, A: int = 1, label: str | None = None):
        """
        Plot dE/dx for a projectile in a neutral medium.
        """
        model = NeutralMediumStoppingPower(Z=Z, A=A)
        dEdx_vals = [model.dEdx(e) for e in self.E]
        plt.loglog(self.E, dEdx_vals, label=label or f"Neutral medium (Z={Z}, A={A})")

    # -------- Leaky box --------
    def plot_leakybox_dedx(self, grammage: float = 7.0, sigma_tot: float = 100.0,
                           A: int = 1, label: str | None = None):
        """
        Plot dE/dx for the leaky-box model.
        """
        model = LeakyBoxStoppingPower(grammage=grammage, sigma_tot=sigma_tot, A=A)
        dEdx_vals = [model.dEdx(e) for e in self.E]
        plt.loglog(self.E, dEdx_vals, label=label or f"Leaky box dE/dx (X={grammage} g/cm²)")

    def plot_leakybox_survival(self, grammage: float = 7.0, sigma_tot: float = 100.0,
                               A: int = 1, label: str | None = None):
        """
        Plot constant survival probability for leaky-box model.
        """
        model = LeakyBoxStoppingPower(grammage=grammage, sigma_tot=sigma_tot, A=A)
        P = model.survival_probability()
        y = np.full_like(self.E, P, dtype=float)
        plt.semilogx(self.E, y, linestyle="--", label=label or f"Survival (X={grammage} g/cm²)")

    def plot_neutral_different_species(self):
        # Proton (Z=1, A=1)
        proton = NeutralMediumStoppingPower(Z=1, A=1)
        # Alpha (Z=2, A=4)
        alpha = NeutralMediumStoppingPower(Z=2, A=4)

        # Energy grid
        E = np.logspace(-1, 2, 500)  # 0.1 to 100 MeV/nucleon

        # Compute stopping power
        dedx_p = [proton.dEdx(e) for e in self.E]
        dedx_a = [alpha.dEdx(e) for e in self.E]

        # Plot
        plt.figure(figsize=(7, 5))
        plt.loglog(E, dedx_p, label="Proton (Z=1, A=1)")
        plt.loglog(E, dedx_a, label="Alpha (Z=2, A=4)")
        plt.xlabel("Energy (MeV/nucleon)")
        plt.ylabel("Stopping power dE/dx")
        plt.title("Neutral Medium Stopping Power")
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.show()

    def plot_leakybox_dual(self, grammage: float = 7.0, sigma_tot: float = 100.0,
                           A: int = 1, title: str | None = None):
        """
        Dual plot: dE/dx (left axis) and survival probability (right axis).
        """
        model = LeakyBoxStoppingPower(grammage=grammage, sigma_tot=sigma_tot, A=A)

        fig, ax1 = plt.subplots()
        dEdx_vals = [model.dEdx(e) for e in self.E]

        # Left axis: dE/dx
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.plot(self.E, dEdx_vals, "b-", label="dE/dx")
        ax1.set_xlabel("Energy per nucleon [MeV/n]")
        ax1.set_ylabel("dE/dx [MeV/n / g cm$^{-2}$]", color="b")

        # Right axis: survival
        ax2 = ax1.twinx()
        P = model.survival_probability()
        ax2.semilogx(self.E, np.full_like(self.E, P, dtype=float), "r--", label="Survival")
        ax2.set_ylabel("Survival probability", color="r")

        # Legends
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")

        if title:
            plt.title(title)
        plt.grid(True, which="both", ls="--", lw=0.6, alpha=0.5)
        plt.show()

    # -------- Generic "show" --------
    def show(self, xlabel: str = "Energy per nucleon [MeV/n]",
             ylabel: str | None = None, title: str | None = None):
        """
        Finalize and show a matplotlib plot.
        """
        if title:
            plt.title(title)
        plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, which="both", ls="--", lw=0.6, alpha=0.5)
        plt.show()



# ---------------- Main (demo) ----------------
if __name__ == "__main__":
    # plotter = StoppingPowerPlotUtils(Emin=0.5, Emax=500)
    # plotter.plot_neutral_different_species()
    # Example: Neutral medium
    # plotter.plot_neutral_dedx(Z=1, A=1, label="Proton in neutral medium")
    # plotter.plot_neutral_dedx(Z=2, A=4, label="Alpha in neutral medium")
    # plotter.show(ylabel="dE/dx [MeV/n / g cm$^{-2}$]", title="Neutral medium stopping")

    # # Example: Leaky box dE/dx
    # plotter.plot_leakybox_dedx(grammage=7.0, sigma_tot=120.0, A=1, label="Leaky box (X=7 g/cm²)")
    # plotter.show(ylabel="dE/dx [MeV/n / g cm$^{-2}$]", title="Leaky box dE/dx")
    #
    # # Example: Leaky box dual
    box = LeakyBoxStoppingPower(grammage=7.0, projectile_A=1, target_A=14)
    E_vals = np.logspace(1, 3, 100)  # 10–1000 MeV/n
    survival = [box.survival_probability(E) for E in E_vals]

    import matplotlib.pyplot as plt

    plt.loglog(E_vals, survival)
    plt.xlabel("Energy (MeV/n)")
    plt.ylabel("Survival Probability")
    plt.title("Leaky-Box Survival Probability")
    plt.show()