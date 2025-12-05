# ------------------------------------------------------------------
# --- Standalone visualization driver ------------------------------
# ------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot spectrum shape (unnormalized and normalized)."
    )
    parser.add_argument(
        "--type",
        choices=["stochastic", "shock", "powerlaw"],
        required=True,
        help="Spectrum type to plot."
    )
    parser.add_argument("--s", type=float, default=3.3, help="Power-law index (for shock or power-law).")
    parser.add_argument("--aT", type=float, default=0.018, help="Thermal αT parameter (for stochastic).")
    parser.add_argument("--E0p", type=float, default=30.0, help="Proton cutoff energy [MeV/n] for shock.")
    parser.add_argument("--Emin", type=float, default=1.0, help="Lower plot energy [MeV/n].")
    parser.add_argument("--Emax", type=float, default=1000.0, help="Upper plot energy [MeV/n].")
    parser.add_argument("--logx", action="store_true", help="Use log scale on X axis.")
    parser.add_argument("--logy", action="store_true", help="Use log scale on Y axis.")
    parser.add_argument("--no-normalize", action="store_true", help="Plot unnormalized spectrum only.")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # --- Select spectrum class ---------------------------------------
    # ------------------------------------------------------------------
    if args.type == "stochastic":
        from stochastic_spectrum import StochasticSpectrum
        spec = StochasticSpectrum(aT=args.aT)
        label = f"Stochastic (aT={args.aT})"

    elif args.type == "shock":
        from shock_spectrum import ShockSpectrumRigidity
        spec = ShockSpectrumRigidity(s=args.s, E0p_turnover=args.E0p)
        label = f"Shock (s={args.s}, E0p={args.E0p} MeV/n)"

    elif args.type == "powerlaw":
        from power_law_spectrum import PowerLawSpectrum
        spec = PowerLawSpectrum(s=args.s)
        label = f"Power law (s={args.s})"

    else:
        raise ValueError("Unknown spectrum type")

    # ------------------------------------------------------------------
    # --- Evaluate over chosen grid -----------------------------------
    # ------------------------------------------------------------------
    E = np.logspace(np.log10(args.Emin), np.log10(args.Emax), 400)
    phi_unnorm = spec.unnormalized(E)

    if not args.no_normalize:
        phi_norm = spec.normalized_N_of_E(E)
    else:
        phi_norm = None

    # ------------------------------------------------------------------
    # --- Plotting -----------------------------------------------------
    # ------------------------------------------------------------------
    plt.figure(figsize=(7, 5))

    plt.plot(E, phi_unnorm, label=f"{label} (unnormalized)", color="C0", lw=2)
    if phi_norm is not None:
        plt.plot(E, phi_norm, "--", label=f"{label} (normalized)", color="C3", lw=2)

    plt.xlabel("Energy per nucleon [MeV/n]", fontsize=12)
    plt.ylabel("Differential flux (arb. units)", fontsize=12)
    plt.title(f"{label} Spectrum", fontsize=13)
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.4)

    if args.logx:
        plt.xscale("log")
    if args.logy:
        plt.yscale("log")

    plt.tight_layout()
    plt.show()