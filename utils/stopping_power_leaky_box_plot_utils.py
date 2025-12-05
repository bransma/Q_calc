"""
stopping_power_leaky_box_plot_utils.py
----------------------------------------------------------------------
Dual-curve plot for Leaky-Box Escape Model.

Left axis (red):   dE/dx from supplied stopping model or toy law
Right axis (blue): Survival probability P_surv(E, X) = exp[-X / Λ(E)]

Use:
    python stopping_power_leaky_box_plot_utils.py --X 5 --use-neutral
----------------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
from stopping_power_leaky_box import StoppingPowerLeakyBox


def plot_leaky_box_survival(
    lb_model: StoppingPowerLeakyBox,
    X: float = 5.0,
    Emin: float = 1.0,
    Emax: float = 1000.0,
    n_points: int = 300,
    stopping_model=None,  # e.g. NeutralMediumStoppingPower or BetheBlochStoppingPower
    show: bool = True,
    save_path: str | None = None,
):
    """
    Plot survival probability (blue, right axis) and dE/dx (red, left axis).

    Parameters
    ----------
    lb_model : StoppingPowerLeakyBox
        Instance defining Λ(E) and P_surv(E, X).
    X : float
        Column depth [g cm⁻²] for survival probability.
    Emin, Emax : float
        Energy range [MeV/n].
    n_points : int
        Number of log-spaced energy points.
    stopping_model : object or None
        If provided, must expose `dEdx(E)` or similar callable.
        Otherwise falls back to the leaky-box toy dE/dX (κ term).
    show : bool
        Display the figure.
    save_path : str or None
        Optional file path to save the figure.
    """
    Egrid = np.logspace(np.log10(Emin), np.log10(Emax), n_points)

    # Compute survival probability
    P_surv = lb_model.survival_probability(Egrid, X)

    # Compute dE/dx (either from external stopping model or toy)
    if stopping_model is not None:
        dEdx_vals = np.array([stopping_model.dEdx(E) for E in Egrid])
        dEdx_label = r"$dE/dx$ (from stopping model)"
    else:
        dEdx_vals = lb_model.dEdx_toy(Egrid)
        dEdx_label = r"$dE/dx$ (toy, κ⋅E/Λ)"

    # Create figure
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # --- Left axis: dE/dx ---
    ax1.loglog(Egrid, np.abs(dEdx_vals), color='crimson', lw=2, label=dEdx_label)
    ax1.set_xlabel("Energy per nucleon [MeV n⁻¹]")
    ax1.set_ylabel(r"Stopping power $|dE/dx|$ [MeV g$^{-1}$ cm$^{2}$]", color='crimson')
    ax1.tick_params(axis='y', labelcolor='crimson')
    ax1.grid(True, which='both', ls='--', alpha=0.4)

    # --- Right axis: survival probability ---
    ax2 = ax1.twinx()
    ax2.semilogx(Egrid, P_surv, color='navy', lw=2, ls='--',
                 label=fr"$P_\mathrm{{surv}}(X={X:g}\,\mathrm{{g\,cm^{{-2}}}})$")
    ax2.set_ylabel("Survival probability", color='navy')
    ax2.tick_params(axis='y', labelcolor='navy')

    # --- Title & Legend ---
    plt.title("Leaky-Box Survival Probability (blue) and Stopping Power (red)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=9)

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"✅ Saved plot to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    from stopping_power_neutral_medium import NeutralMediumStoppingPower

    parser = argparse.ArgumentParser(
        description="Plot leaky-box survival (blue) + stopping power (red)."
    )
    parser.add_argument("--X", type=float, default=5.0, help="Column depth [g cm⁻²].")
    parser.add_argument("--lambda0", type=float, default=10.0, help="Escape length normalization.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Energy exponent for escape length.")
    parser.add_argument("--use-neutral", action="store_true",
                        help="Use NeutralMediumStoppingPower for red curve.")
    parser.add_argument("--kappa", type=float, default=0.0,
                        help="Toy energy-loss coupling (κ).")
    parser.add_argument("--save", type=str, default=None, help="Save path.")
    parser.add_argument("--no-show", action="store_true", help="Disable interactive display.")
    args = parser.parse_args()

    lb = StoppingPowerLeakyBox(lambda0=args.lambda0, alpha=args.alpha, kappa=args.kappa)
    stopping = NeutralMediumStoppingPower(Z=1, A=1) if args.use_neutral else None

    plot_leaky_box_survival(lb, X=args.X, stopping_model=stopping,
                            show=not args.no_show, save_path=args.save)