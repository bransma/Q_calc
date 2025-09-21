import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from cross_section_channels import CrossSectionReader

class CrossSectionPlotUtils:
    """
    Utility class for plotting cross section channels using CrossSectionReader.
    - Always log-log plots
    - Black and white line styles (5 types)
    - Legends use LaTeX-style formatting (superscripts, Greek α)
    - Clean plots: no grid, bold ticks
    - Channel mosaic in 3x3 grid:
        Row 1: p+12C, p+14N, p+16O
        Row 2: α+12C, α+14N, α+16O
        Row 3: p+13C, He+He (combined: 3He4He + 4He4He)
    """

    def __init__(self, reader: CrossSectionReader, displaytype: str = "show", dpi: int = 300):
            self.reader = reader
            self.displaytype = displaytype.lower()
            self.dpi = dpi

            # Consistent line style per product A across ALL plots
            self.style_map = {
                "6": "-",
                "7": "--",
                "9": "-.",
                "10": ":",
                "11": (0, (3, 1, 1, 1)),
            }

            # Fallback cyclic styles (used only if some code still references line_styles)
            self.line_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

            if self.displaytype in ["png", "eps"]:
                os.makedirs("plots", exist_ok=True)

    def _finalize(self, fig, name: str):
        """Handle output depending on displaytype (show/png/eps)."""
        if self.displaytype == "show":
            plt.show()
        elif self.displaytype == "png":
            out = os.path.join("plots", f"{name}.png")
            fig.savefig(out, dpi=self.dpi, bbox_inches="tight")
            print(f"[INFO] Saved {out}")
        elif self.displaytype == "eps":
            out = os.path.join("plots", f"{name}.eps")
            fig.savefig(out, format="eps", dpi=self.dpi, bbox_inches="tight")
            print(f"[INFO] Saved {out}")
        else:
            raise ValueError(f"Unknown displaytype: {self.displaytype}")

    def _latex_from_key(self, key: str) -> str:
        """Build LaTeX legend title automatically from a reaction key."""
        base = key.split("_a")[0]

        if base.startswith("P"):  # Proton projectiles
            Z = base[1:len(base)-1]
            element = base[-1]
            return rf"$p + ^{{{Z}}}{element} \;\rightarrow$"
        if base.startswith("3He"):  # Mixed helium
            return r"$^{3}He + ^{4}He \;\rightarrow$"
        if base.startswith("4He4He"):  # Double-alpha
            return r"$\alpha + \alpha \;\rightarrow$"
        if base.startswith("4He"):  # Alpha + CNO
            Z = base[3:len(base)-1]
            element = base[-1]
            return rf"$\alpha + ^{{{Z}}}{element} \;\rightarrow$"

        return base

    def compute_bounds(self, keys):
        """Compute smart log-log axis bounds for a set of channels."""
        all_E, all_sigma = [], []
        for key in keys:
            E, sigma = self.reader.cross_section_data_mb[key]
            E = E[E > 0]
            sigma = sigma[sigma > 0]
            if len(E) > 0:
                all_E.append(E)
            if len(sigma) > 0:
                all_sigma.append(sigma)
        if not all_E or not all_sigma:
            return None
        all_E = np.concatenate(all_E)
        all_sigma = np.concatenate(all_sigma)
        return all_E.min()*0.8, all_E.max()*1.2, all_sigma.min()*0.8, all_sigma.max()*1.2

    def _style_axis(self, ax):
        """Apply clean axis style: no grid, bold ticks."""
        ax.tick_params(axis="both", which="both",
                       direction="in", length=6, width=1.5, labelsize=10)
        ax.grid(False)

    def plot_one(self, key: str) -> None:
        E, sigma = self.reader.cross_section_data_mb[key]
        E_dense = np.logspace(np.log10(E.min()), np.log10(E.max()), 500)
        sigma_interp = self.reader.sigma(key, E_dense)

        fig, ax = plt.subplots(figsize=(7,5))
        ax.loglog(E_dense, sigma_interp, "-", color="black", lw=1, label=key)

        bounds = self.compute_bounds([key])
        if bounds:
            ax.set_xlim(bounds[0], bounds[1])
            ax.set_ylim(bounds[2], bounds[3])

        ax.set_xlabel("Energy (MeV/nucleon)")
        ax.set_ylabel("Cross Section (mb)")
        ax.legend(loc="lower right")
        self._style_axis(ax)

        self._finalize(fig, key)

    def plot_group(self, keys, title: str) -> None:
        fig, ax = plt.subplots(figsize=(7,5))
        for i, key in enumerate(keys):
            E, sigma = self.reader.cross_section_data_mb[key]
            E_dense = np.logspace(np.log10(E.min()), np.log10(E.max()), 500)
            sigma_interp = self.reader.sigma(key, E_dense)
            style = self.line_styles[i % len(self.line_styles)]
            A = key.split("_a")[-1]
            ax.loglog(E_dense, sigma_interp, linestyle=style,
                      color="black", lw=1, label=f"A={A}")

        bounds = self.compute_bounds(keys)
        if bounds:
            ax.set_xlim(bounds[0], bounds[1])
            ax.set_ylim(bounds[2], bounds[3])

        ax.set_xlabel("Energy (MeV/nucleon)")
        ax.set_ylabel("Cross Section (mb)")
        ax.legend(title=self._latex_from_key(keys[0]), loc="lower right", fontsize="small")
        self._style_axis(ax)

        self._finalize(fig, title.replace(" ", "_"))

    def plot_mosaic_channels(self) -> None:
        groups = {
            "p+12C": self.reader.P12C,
            "p+14N": self.reader.PN,
            "p+16O": self.reader.PO,
            "α+12C": self.reader.HeC,
            "α+14N": self.reader.HeN,
            "α+16O": self.reader.HeO,
            "p+13C": self.reader.P13C,
            "He+He": ["3He4He_a6", "4He4He_a6", "4He4He_a7"],  # combined
        }

        # Hard-coded label placements (E, σ) in data coords
        label_positions = {
            "p+12C": (100, 0.85),
            "p+14N": (100, 1),
            "p+16O": (100, 1),
            "α+12C": (20, 3),
            "α+14N": (20, 3),
            "α+16O": (20, 3),
            "p+13C": (7, 10),
            # He+He handled separately
        }

        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 7.5))
        axes = axes.flatten()

        mapping = {
            0: ("p+12C", groups["p+12C"]),
            1: ("p+14N", groups["p+14N"]),
            2: ("p+16O", groups["p+16O"]),
            3: ("α+12C", groups["α+12C"]),
            4: ("α+14N", groups["α+14N"]),
            5: ("α+16O", groups["α+16O"]),
            6: ("p+13C", groups["p+13C"]),
            7: ("He+He", groups["He+He"]),
        }

        # Remove unused 9th subplot
        fig.delaxes(axes[8])

        for idx, (group_name, keys) in mapping.items():
            ax = axes[idx]

            if group_name == "He+He":  # Special case
                for key in keys:
                    E, sigma = self.reader.cross_section_data_mb[key]
                    E_dense = np.logspace(np.log10(E.min()), np.log10(E.max()), 500)
                    sigma_interp = self.reader.sigma(key, E_dense)

                    if key.startswith("4He4He") and key.endswith("a6"):
                        ax.loglog(E_dense, sigma_interp, "-", color="black", lw=1, label="A=6")
                        pos = int(0.7 * len(E_dense))
                        ax.text(26, 20,
                                r"$\alpha + \alpha$", fontsize=8, va="bottom", ha="left")
                    elif key.startswith("4He4He") and key.endswith("a7"):
                        ax.loglog(E_dense, sigma_interp, "--", color="black", lw=1, label="A=7")
                    elif key.startswith("3He4He"):
                        ax.loglog(E_dense, sigma_interp, "-", color="black", lw=1)
                        ax.text(2.6, 103,  # fixed position for clarity
                                r"$^{3}He + \alpha \;\rightarrow\; ^6Li$",
                                fontsize=8, va="bottom", ha="left")

                self._style_axis(ax)

            elif group_name == "p+12C":  # Special case with legend
                for key in keys:
                    E, sigma = self.reader.cross_section_data_mb[key]
                    E_dense = np.logspace(np.log10(E.min()), np.log10(E.max()), 500)
                    sigma_interp = self.reader.sigma(key, E_dense)
                    A = key.split("_a")[-1]
                    style = self.style_map.get(A, "-")
                    ax.loglog(E_dense, sigma_interp, linestyle=style,
                              color="black", lw=1, label=f"A={A}")

                    ax.set_ylabel("σ (mb)")
                    ax.set_xlabel("E (MeV/nucleon)")

                ax.legend(loc="lower right", fontsize="x-small", frameon=False)

                # Reaction label at fixed coords
                ax.text(*label_positions[group_name],
                        r"$p + ^{12}C$", fontsize=9, va="bottom", ha="left")
                self._style_axis(ax)

            else:  # All other panels
                for key in keys:
                    E, sigma = self.reader.cross_section_data_mb[key]
                    E_dense = np.logspace(np.log10(E.min()), np.log10(E.max()), 500)
                    sigma_interp = self.reader.sigma(key, E_dense)
                    A = key.split("_a")[-1]
                    style = self.style_map.get(A, "-")
                    ax.loglog(E_dense, sigma_interp, linestyle=style,
                              color="black", lw=1)

                bounds = self.compute_bounds(keys)
                if bounds:
                    ax.set_xlim(bounds[0], bounds[1])
                    ax.set_ylim(bounds[2], bounds[3])

                # Reaction label at hard-coded coords
                if group_name in label_positions:
                    reaction_label = self._latex_from_key(keys[0]).replace(r"\;\rightarrow", "")
                    ax.text(*label_positions[group_name], reaction_label,
                            fontsize=9, va="bottom", ha="left")

                self._style_axis(ax)

        plt.tight_layout()
        self._finalize(fig, "mosaic_channels")

    def plot_mosaic_A(self) -> None:
        groups = {
            "A=6": self.reader.A6,
            "A=7": self.reader.A7,
            "A=9": self.reader.A9,
            "A=10": self.reader.A10,
            "A=11": self.reader.A11,
        }

        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))
        axes = axes.flatten()

        mapping = {
            0: ("A=6", groups["A=6"]),
            1: ("A=7", groups["A=7"]),
            2: ("A=9", groups["A=9"]),
            3: ("A=10", groups["A=10"]),
            4: ("A=11", groups["A=11"]),
        }

        for idx, (title, keys) in mapping.items():
            ax = axes[idx]
            for i, key in enumerate(keys):
                E, sigma = self.reader.cross_section_data_mb[key]
                E_dense = np.logspace(np.log10(E.min()), np.log10(E.max()), 500)
                sigma_interp = self.reader.sigma(key, E_dense)

                # LaTeX reaction channel label
                reaction_label = self._latex_from_key(key).replace(r"\;\rightarrow", "")

                style = self.line_styles[i % len(self.line_styles)]
                ax.loglog(E_dense, sigma_interp, linestyle=style,
                          color="black", lw=1, label=reaction_label)

            # Uniform y-axis scaling
            ax.set_ylim(1e-1, 300)

            # Only label left column and bottom row
            row, col = divmod(idx, 3)
            if col == 0:
                ax.set_ylabel("σ (mb)")
            if row == 1:
                ax.set_xlabel("E (MeV/nucleon)")

            # Legend inside, bottom left
            ax.legend(loc="lower right", frameon=False, fontsize="x-small")

            # A=.. annotation in top right
            ax.text(0.95, 0.95, title, transform=ax.transAxes,
                    fontsize=9, va="top", ha="right")

            self._style_axis(ax)

        fig.delaxes(axes[5])  # drop extra panel
        plt.tight_layout()
        self._finalize(fig, "mosaic_A")

# -------------------------
# Main runner
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross Section Plot Utils")
    parser.add_argument("--type", type=str, required=True,
                        help=("Which plot to generate: "
                              "mosaic_channels, mosaic_A, "
                              "P12C, P13C, PN, PO, HeC, HeN, HeO, HeHe, "
                              "A6, A7, A9, A10, A11, "
                              "or a specific channel key like P12C_a6"))
    parser.add_argument("--displaytype", type=str, default="show",
                        help="Output type: show (default), png, eps")
    parser.add_argument("--dpi", type=int, default=300,
                        help="DPI for saved figures (png/eps)")

    args = parser.parse_args()

    reader = CrossSectionReader("./csv")
    reader.read_all()
    utils = CrossSectionPlotUtils(reader, displaytype=args.displaytype, dpi=args.dpi)

    if args.type == "mosaic_channels":
        utils.plot_mosaic_channels()
    elif args.type == "mosaic_A":
        utils.plot_mosaic_A()
    elif args.type in ["P12C","P13C","PN","PO","HeC","HeN","HeO","HeHe"]:
        keys = getattr(reader, args.type)
        utils.plot_group(keys, args.type)
    elif args.type in ["A6","A7","A9","A10","A11"]:
        keys = getattr(reader, args.type)
        utils.plot_group(keys, args.type)
    elif args.type in reader.all_keys:
        utils.plot_one(args.type)
    else:
        print(f"Unknown type: {args.type}")