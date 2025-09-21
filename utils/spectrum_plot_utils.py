"""
spectrum_plot_utils.py
======================

Plotting utilities for particle energy spectra defined in `spectra.py`.

Features
--------
- Plot any single spectrum (normalized or unnormalized).
- Sweep parameters:
    * StochasticK2: aT list / ranges
    * PowerLaw   : gamma list / ranges
    * ShockRigidity: s list / ranges (with E0p → E0_species conversion)
- Plot composite mixtures.
- Always log–log axes for clarity in flare/CR regimes.

Command-line examples
---------------------
# 1) Single stochastic, normalized, 1–100 MeV/n
python spectrum_plot_utils.py single --type stochastic --aT 0.025 --Emin 1 --Emax 100

# 2) Sweep aT normalized (0.01→0.04 step 0.005)
python spectrum_plot_utils.py sweep-stochastic --aT-start 0.01 --aT-stop 0.04 --aT-step 0.005 --Emin 1 --Emax 100

# 3) Sweep power-law gammas (2.0→6.0 step 0.5), unnormalized
python spectrum_plot_utils.py sweep-powerlaw --gamma-start 2 --gamma-stop 6 --gamma-step 0.5 --unnormalized --Emin 1 --Emax 100

# 4) Sweep shock slope s (2.0→4.0 step 0.2) for alphas with E0p=30 MeV, 1–500 MeV/n
python spectrum_plot_utils.py sweep-shock --s-start 2 --s-stop 4 --s-step 0.2 --E0p 30 --Z 2 --A 4 --Emin 1 --Emax 500

# 5) Composite (0.7 stochastic aT=0.025 + 0.3 power-law gamma=4)
python spectrum_plot_utils.py composite --aT 0.025 --w-stoch 0.7 --gamma 4 --w-pl 0.3 --Emin 1 --Emax 100
"""

from __future__ import annotations

import argparse
from typing import Iterable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Import the spectra implementations
from spectra import (
    StochasticK2,
    PowerLaw,
    ShockRigidity,
    CompositeSpectrum,
)


# ---------------------------
# Styling helpers
# ---------------------------

LINESTYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]  # default MPL palette


def _linestyle_for(i: int) -> str:
    return LINESTYLES[i % len(LINESTYLES)]


def _color_for(i: int) -> str:
    return COLORS[i % len(COLORS)]


# ---------------------------
# Energy grid helper
# ---------------------------

def make_energy_grid(Emin: float, Emax: float, n: int = 800) -> np.ndarray:
    """
    Create a log-spaced energy grid in MeV/nucleon for plotting and integration.

    Parameters
    ----------
    Emin, Emax : float
        Energy bounds (MeV/n). We enforce small padding to avoid degenerate ranges.
    n : int
        Number of log samples.

    Returns
    -------
    np.ndarray
        Log-spaced energies from Emin to Emax (inclusive).
    """
    Emin = max(float(Emin), 1e-6)
    Emax = max(float(Emax), Emin * 1.0001)
    return np.logspace(np.log10(Emin), np.log10(Emax), n)


# ---------------------------
# Plotting utilities class
# ---------------------------

class SpectrumPlotUtils:
    """
    A small plotting helper around spectra.py with consistent, log–log visuals.

    Methods
    -------
    plot_single(spec, normalized=True, label=None, title=None)
    sweep_stochastic(aT_values, Emin, Emax, normalized=True)
    sweep_powerlaw(gamma_values, Emin, Emax, normalized=True)
    sweep_shock(s_values, E0p, Z, A, Emin, Emax, normalized=True)
    plot_composite(components, Emin, Emax, normalized=True)

    Notes
    -----
    - All plots are log–log by design (flare/CR use-case).
    - The “normalized” flag toggles spec.N(E) vs spec.unnormalized(E).
    - For shock spectra, we convert from a proton cutoff E0p → species cutoff E0_species
      using the helper in ShockRigidity (vR-invariance).
    """

    def __init__(self, nE: int = 800):
        self.nE = int(nE)

    # ---------- core single plot ----------

    def plot_single(self,
                    spec,
                    normalized: bool = True,
                    label: str | None = None,
                    title: str | None = None) -> None:
        """
        Plot a single spectrum (normalized or unnormalized) on log–log axes.
        """
        E = make_energy_grid(spec.Emin, spec.Emax, self.nE)
        y = spec.N(E) if normalized else spec.unnormalized(E)

        plt.figure(figsize=(7.2, 5.0))
        plt.loglog(E, y, lw=2.0, label=(label or self._auto_label(spec, normalized)))
        plt.xlabel("Energy per nucleon E (MeV/n)")
        plt.ylabel("Normalized N(E)" if normalized else "Unnormalized φ(E)")
        if title:
            plt.title(title)
        plt.legend()
        # No grid for a clean scientific look; uncomment if you prefer:
        # plt.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.show()

    # ---------- sweeps ----------

    def sweep_stochastic(self,
                         aT_values: Iterable[float],
                         Emin: float,
                         Emax: float,
                         normalized: bool = True) -> None:
        """
        Sweep over aT values for StochasticK2 and plot on one figure.
        """
        E = make_energy_grid(Emin, Emax, self.nE)
        plt.figure(figsize=(7.6, 5.2))
        for i, aT in enumerate(aT_values):
            spec = StochasticK2(aT=float(aT), Emin=Emin, Emax=Emax)
            y = spec.N(E) if normalized else spec.unnormalized(E)
            plt.loglog(E, y, lw=2.0, ls=_linestyle_for(i), color=_color_for(i),
                       label=rf"Stoch $aT={aT:.3f}$")
        plt.xlabel("Energy per nucleon E (MeV/n)")
        plt.ylabel("Normalized N(E)" if normalized else "Unnormalized φ(E)")
        plt.title("Stochastic K2 sweep")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def sweep_powerlaw(self,
                       gamma_values: Iterable[float],
                       Emin: float,
                       Emax: float,
                       normalized: bool = True) -> None:
        """
        Sweep over power-law indices gamma and plot on one figure.
        """
        E = make_energy_grid(Emin, Emax, self.nE)
        plt.figure(figsize=(7.6, 5.2))
        for i, g in enumerate(gamma_values):
            spec = PowerLaw(gamma=float(g), Emin=Emin, Emax=Emax)
            y = spec.N(E) if normalized else spec.unnormalized(E)
            plt.loglog(E, y, lw=2.0, ls=_linestyle_for(i), color=_color_for(i),
                       label=rf"Power-law $\gamma={g:.2f}$")
        plt.xlabel("Energy per nucleon E (MeV/n)")
        plt.ylabel("Normalized N(E)" if normalized else "Unnormalized φ(E)")
        plt.title("Power-law sweep")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def sweep_shock(self,
                    s_values: Iterable[float],
                    E0p: float,
                    Z: int,
                    A: int,
                    Emin: float,
                    Emax: float,
                    normalized: bool = True) -> None:
        """
        Sweep over shock slope 's' for a given projectile (Z,A).
        E0p is the proton cutoff (MeV/n). We convert to species cutoff E0_species via vR-invariance.
        """
        E = make_energy_grid(Emin, Emax, self.nE)
        plt.figure(figsize=(7.6, 5.2))
        for i, s in enumerate(s_values):
            E0_species = ShockRigidity.E0_species_from_proton(E0p=float(E0p), Z=int(Z), A=int(A))
            spec = ShockRigidity(s=float(s), E0_species=E0_species, Z=int(Z), A=int(A),
                                 Emin=Emin, Emax=Emax)
            y = spec.N(E) if normalized else spec.unnormalized(E)
            plt.loglog(E, y, lw=2.0, ls=_linestyle_for(i), color=_color_for(i),
                       label=rf"Shock $s={s:.2f}$, $E_0^p={E0p:.1f}$ MeV")
        plt.xlabel("Energy per nucleon E (MeV/n)")
        plt.ylabel("Normalized N(E)" if normalized else "Unnormalized φ(E)")
        plt.title(rf"Shock (Z={Z}, A={A}) sweep")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # ---------- composite ----------

    def plot_composite(self,
                       components: List[Tuple[float, object]],
                       Emin: float,
                       Emax: float,
                       normalized: bool = True) -> None:
        """
        Plot a composite spectrum from (weight, spectrum) components.
        Each component spectrum should have its own [Emin,Emax] compatible with the composite band.
        """
        comp = CompositeSpectrum(components=components, Emin=Emin, Emax=Emax)
        E = make_energy_grid(Emin, Emax, self.nE)
        y = comp.N(E) if normalized else comp.unnormalized(E)

        plt.figure(figsize=(7.2, 5.0))
        plt.loglog(E, y, lw=2.2, label="Composite")
        # Overplot components (faint) for context
        for i, (w, sp) in enumerate(components):
            yc = sp.N(E) if normalized else sp.unnormalized(E)
            plt.loglog(E, yc, lw=1.2, ls=_linestyle_for(i), color=_color_for(i),
                       alpha=0.6, label=self._auto_label(sp, normalized, prefix=f"{w:.2f}× "))

        plt.xlabel("Energy per nucleon E (MeV/n)")
        plt.ylabel("Normalized N(E)" if normalized else "Unnormalized φ(E)")
        plt.title("Composite spectrum (components faint)")
        plt.legend(ncol=2, fontsize="small")
        plt.tight_layout()
        plt.show()

    # ---------- internal ----------

    @staticmethod
    def _auto_label(spec, normalized: bool, prefix: str = "") -> str:
        """
        Produce a compact, math-friendly label per spectrum type.
        """
        base = ""
        if isinstance(spec, StochasticK2):
            base = rf"Stoch $aT={spec.aT:.3f}$"
        elif isinstance(spec, PowerLaw):
            base = rf"PL $\gamma={spec.gamma:.2f}$"
        elif isinstance(spec, ShockRigidity):
            base = rf"Shock $s={spec.s:.2f}$, $E_0^{{\rm sp}}={spec.E0_species:.1f}$"
        elif isinstance(spec, CompositeSpectrum):
            base = "Composite"
        else:
            base = spec.__class__.__name__
        return (prefix + base + (" [N]" if normalized else " [φ]"))


# ---------------------------
# CLI
# ---------------------------

def _floats_from_range(start: float, stop: float, step: float) -> np.ndarray:
    """
    Inclusive float range helper (ensures the last bin is included within epsilon).
    """
    n = int(np.floor((stop - start) / step + 0.5)) + 1
    return np.array([start + i * step for i in range(max(n, 1))], dtype=float)


def main():
    parser = argparse.ArgumentParser(description="Spectrum plotting utilities (log–log always).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Single spectrum
    p_single = sub.add_parser("single", help="Plot a single spectrum.")
    p_single.add_argument("--type", choices=["stochastic", "powerlaw", "shock"], required=True)
    p_single.add_argument("--aT", type=float, help="Stochastic aT (MeV)")
    p_single.add_argument("--gamma", type=float, help="Power-law index gamma")
    p_single.add_argument("--s", type=float, help="Shock slope s")
    p_single.add_argument("--E0p", type=float, help="Shock proton cutoff E0p (MeV)")
    p_single.add_argument("--Z", type=int, default=1, help="Shock Z (projectile charge)")
    p_single.add_argument("--A", type=int, default=1, help="Shock A (projectile mass number)")
    p_single.add_argument("--Emin", type=float, required=True)
    p_single.add_argument("--Emax", type=float, required=True)
    p_single.add_argument("--unnormalized", action="store_true", help="Plot unnormalized φ(E)")

    # Sweep stochastic
    p_ss = sub.add_parser("sweep-stochastic", help="Sweep aT values.")
    p_ss.add_argument("--aT-start", type=float, required=True)
    p_ss.add_argument("--aT-stop", type=float, required=True)
    p_ss.add_argument("--aT-step", type=float, required=True)
    p_ss.add_argument("--Emin", type=float, required=True)
    p_ss.add_argument("--Emax", type=float, required=True)
    p_ss.add_argument("--unnormalized", action="store_true")

    # Sweep power-law
    p_pl = sub.add_parser("sweep-powerlaw", help="Sweep gamma values.")
    p_pl.add_argument("--gamma-start", type=float, required=True)
    p_pl.add_argument("--gamma-stop", type=float, required=True)
    p_pl.add_argument("--gamma-step", type=float, required=True)
    p_pl.add_argument("--Emin", type=float, required=True)
    p_pl.add_argument("--Emax", type=float, required=True)
    p_pl.add_argument("--unnormalized", action="store_true")

    # Sweep shock
    p_sh = sub.add_parser("sweep-shock", help="Sweep shock slope s values.")
    p_sh.add_argument("--s-start", type=float, required=True)
    p_sh.add_argument("--s-stop", type=float, required=True)
    p_sh.add_argument("--s-step", type=float, required=True)
    p_sh.add_argument("--E0p", type=float, required=True, help="Proton cutoff (MeV)")
    p_sh.add_argument("--Z", type=int, required=True)
    p_sh.add_argument("--A", type=int, required=True)
    p_sh.add_argument("--Emin", type=float, required=True)
    p_sh.add_argument("--Emax", type=float, required=True)
    p_sh.add_argument("--unnormalized", action="store_true")

    # Composite example
    p_cmp = sub.add_parser("composite", help="Composite = w_stoch*stoch + w_pl*powerlaw (simple demo).")
    p_cmp.add_argument("--aT", type=float, required=True)
    p_cmp.add_argument("--w-stoch", type=float, default=0.7)
    p_cmp.add_argument("--gamma", type=float, required=True)
    p_cmp.add_argument("--w-pl", type=float, default=0.3)
    p_cmp.add_argument("--Emin", type=float, required=True)
    p_cmp.add_argument("--Emax", type=float, required=True)
    p_cmp.add_argument("--unnormalized", action="store_true")

    args = parser.parse_args()
    utils = SpectrumPlotUtils()

    if args.cmd == "single":
        if args.type == "stochastic":
            spec = StochasticK2(aT=args.aT, Emin=args.Emin, Emax=args.Emax)
        elif args.type == "powerlaw":
            spec = PowerLaw(gamma=args.gamma, Emin=args.Emin, Emax=args.Emax)
        elif args.type == "shock":
            if args.E0p is None:
                raise SystemExit("For --type shock you must provide --E0p (MeV).")
            E0_species = ShockRigidity.E0_species_from_proton(args.E0p, Z=args.Z, A=args.A)
            spec = ShockRigidity(s=args.s, E0_species=E0_species, Z=args.Z, A=args.A,
                                 Emin=args.Emin, Emax=args.Emax)
        utils.plot_single(spec, normalized=(not args.unnormalized))

    elif args.cmd == "sweep-stochastic":
        aT_vals = _floats_from_range(args.aT_start, args.aT_stop, args.aT_step)
        utils.sweep_stochastic(aT_vals, Emin=args.Emin, Emax=args.Emax,
                               normalized=(not args.unnormalized))

    elif args.cmd == "sweep-powerlaw":
        g_vals = _floats_from_range(args.gamma_start, args.gamma_stop, args.gamma_step)
        utils.sweep_powerlaw(g_vals, Emin=args.Emin, Emax=args.Emax,
                             normalized=(not args.unnormalized))

    elif args.cmd == "sweep-shock":
        s_vals = _floats_from_range(args.s_start, args.s_stop, args.s_step)
        utils.sweep_shock(s_vals, E0p=args.E0p, Z=args.Z, A=args.A,
                          Emin=args.Emin, Emax=args.Emax,
                          normalized=(not args.unnormalized))

    elif args.cmd == "composite":
        st = StochasticK2(aT=args.aT, Emin=args.Emin, Emax=args.Emax)
        pl = PowerLaw(gamma=args.gamma, Emin=args.Emin, Emax=args.Emax)
        comps = [(args.w_stoch, st), (args.w_pl, pl)]
        utils.plot_composite(comps, Emin=args.Emin, Emax=args.Emax,
                             normalized=(not args.unnormalized))


if __name__ == "__main__":
    main()