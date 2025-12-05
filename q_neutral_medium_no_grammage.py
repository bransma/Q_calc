# q_integrator_refined_class.py
# ------------------------------------------------------------
# Class-based implementation of the Q integrator
# using CrossSectionChannels, NeutralMediumStoppingPower,
# and Spectrum subclasses (e.g., StochasticSpectrum).
# ------------------------------------------------------------

import numpy as np
import argparse
import decimal
from scipy.integrate import simpson
from typing import Callable, Tuple, List

from cross_section_channels import CrossSectionChannels
from spectrum import Spectrum
from stopping_power_neutral_medium import NeutralMediumStoppingPower
from stochastic_spectrum import StochasticSpectrum
from power_law_spectrum import PowerLawSpectrum
from shock_spectrum import ShockSpectrumRigidity


# --------------------------------------------------------
# --- helper: compute normalized function R(E) ---------------
# --------------------------------------------------------

def precompute_R(E: np.ndarray, N_func: Callable[[float], float]) -> np.ndarray:
    """
    (Pre)Compute inner integralR(E) = ∫_E^∞ N(E') dE' / ∫_{E_norm}^∞ N(E') dE'
    """
    # Evaluate spectrum pointwise (scalar-safe)
    N_vals = np.array([float(N_func(e)) for e in np.ravel(E)], dtype=float)
    return np.nan_to_num(N_vals, nan=0.0, posinf=0.0, neginf=0.0)

class QNeutralMediumNoGrammageSimpsonRule:
    def __init__(
        self,
        spectral_type: str = "k2",
        spectral_index: float = 0.020,
        Z: int = 1,
        A: int = 1,
        E_0 = 30.0, # N_p(E>E_0) = 1 normalization
        E_min: float = 1.0, # starting energy for integration
        E_max: float = 3000.0, # finite GeV-range limit for integration
        E_0p : float = 30.0, # proton turnover for shock spectrum
        csv_name : str = "",
        reader : CrossSectionChannels = None,
        keys : List[str] = None
    ):
        # --- cross sections ---
        self.reader = reader
        self.keys = keys
        # --- stopping power (neutral medium) ---
        self.Z = A
        self.A = A
        self.stopping_power = NeutralMediumStoppingPower(Z, A)
        self.spectral_index = 0.0
        self.E_0 = E_0
        self.E_0p = E_0p
        self.spectral_type = spectral_type
        self.spectrum = self.create_spectrum(spectral_index=spectral_index)
        self.E_min = E_min
        self.E_max = E_max

        # physical constants
        self.mass_proton = 1.6726e-24  # g
        self.mb_to_cm2 = 1.0e-27
        self.keys = keys

        # Choose the CSV file name based on spectral type
        self.csv_name = csv_name
        with open(self.csv_name, "w") as f:
            f.write("spectral_value," + ",".join(self.keys) + "\n")

    def create_spectrum(self, spectral_index: float=0.0014) -> Spectrum:
        # --- spectrum selection ---
        if spectral_type.lower() == "k2":
            spectrum = StochasticSpectrum(spectral_index, Emin_norm=self.E_0)
        elif spectral_type.lower() == "powerlaw":
            spectrum = PowerLawSpectrum(spectral_index, Emin_norm=self.E_0)
        elif spectral_type.lower() == "shock":
            spectrum = ShockSpectrumRigidity(spectral_index, Emin_norm=self.E_0, E0p_turnover=self.E_0p)
        else:
            raise ValueError(f"Unknown spectral type: {spectral_type}")
        self.spectral_index = spectral_index
        print(f"spectrum {spectral_type} created with index {spectral_index} and "
               f"computed normalization {spectrum.N}")
        return  spectrum

    # --------------------------------------------------------
    # --- main integration routine ---------------------------
    # --------------------------------------------------------
    def integrate_Q(
        self,
        sigma_func: Callable[[np.ndarray], np.ndarray],
        S_func: Callable[[np.ndarray], np.ndarray],
        N_func: Callable[[float], float],
        N_points: int = 40000,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Compute Q = ∫ σ(E) * S(E) * R(E) dE  (dimensionless per projectile)
        using Simpson integration on a log grid.

        P12C_a6: Q' = 1.0 to X (MeV/GeV)
        3.294358e-03 ; 4000 points stochastic to 3 GeV
        3.294328e-03; 40000 points stochastic to 3 GeV <-- safe enough, good default, both points and energy
        3.294334e-03; 40000 points stochastic to 1 GeV <-- the reported approx limit to ion acceleration
        3.294330e-03; 40000 points to 30 Gev (non-physical, probing tail contribution - very negligible)
        """
        E = np.geomspace(self.E_min, self.E_max, N_points)
        R = precompute_R(E, N_func)

        f = sigma_func(E) * (1.0 / S_func(E)) * R
        Q_integral = simpson(f, x=E)   # convert to per-proton rate
        return Q_integral, E, f

    # --------------------------------------------------------
    # --- compute Q for a given reaction channel -------------
    # --------------------------------------------------------
    def compute_channel_Q(self, key: str) -> float:
        """Compute Q for one specific cross-section channel."""
        sigma_func = lambda E_integration: self.reader.sigma(key, E_integration)  # [mb]
        S_func = self.stopping_power.dEdx
        N_func = self.spectrum.N_integrand

        Q_integral, E, f = self.integrate_Q(
            sigma_func=sigma_func,
            S_func=S_func,
            N_func=N_func
        )
        Q_prime_channel = ((1.0 / self.mass_proton) *
                           self.mb_to_cm2 *  (1.0 / self.spectrum.N) * Q_integral)
        return Q_prime_channel

    # --------------------------------------------------------
    # --- driver for all available channels ------------------
    # --------------------------------------------------------
    def compute_Q(self) -> None:
        """
        Loop over a list of reaction keys and compute Q for each.
        If no list is provided, compute for all in reader.all_keys.
        """
        results = {}
        for channel_key in self.keys:
            self.check_stopping_power(channel_key)
            try:
                Q = self.compute_channel_Q(channel_key)
                results[channel_key] = Q
                print(f"{self.spectral_index}; {channel_key:15s}  Q = {Q:.6e}")
            except Exception as e:
                print(f"{channel_key:15s}  [error] {e}")
        self.make_csv_row(self.spectral_index, results)

    def compute_Q_spect_range(self, spect_start: int = "0.014", spect_end: int = "0.040",
                       spect_step: int = "0.001"):
        index = decimal.Decimal(spect_start)
        step = decimal.Decimal(spect_step)
        end = decimal.Decimal(spect_end)
        while index <= end:
            self.spectrum = self.create_spectrum(spectral_index=float(index))
            self.compute_Q()
            index += step

    def check_stopping_power(self, key : str) -> None:
        if key.startswith("P"):
            if self.stopping_power.Z == 1 and self.stopping_power.A == 1:
                return
            else:
                print(f"changing stopping power to proton bullet Z={self.stopping_power.Z} -> Z=1 "
                      f" & A={self.stopping_power.A} -> A=1")
                self.stopping_power = NeutralMediumStoppingPower(Z=1, A=1)
                self.spectrum.recompute(Z_new=1, A_new=1)
        if key.startswith("4He"):
            if self.stopping_power.Z == 2 and self.stopping_power.A == 4:
                return
            else:
                print(f"chaning stopping power to 4He bullet Z={self.stopping_power.Z} -> Z=2 "
                      f" & A={self.stopping_power.A} -> A=4")
                self.stopping_power = NeutralMediumStoppingPower(Z=2, A=4)
                self.spectrum.recompute(Z_new=2, A_new=4)
        if key.startswith("3He"):
            if self.stopping_power.Z == 2 and self.stopping_power.A == 3:
                return
            else:
                print(f"chaning stopping power to 3He bullet Z={self.stopping_power.Z} -> Z=2 "
                      f" & A={self.stopping_power.A} -> A=3")
                self.stopping_power = NeutralMediumStoppingPower(Z=2, A=3)
                self.spectrum.recompute(Z_new=2, A_new=3)
        return

    def make_csv_row(self, spectral_value: float, qprime_dict: dict) -> None:
        """
        Construct a CSV data row using the same key ordering as the header.

        Parameters
        ----------
        spectral_value : float
            The spectral index (aT, s, E0p, etc.)
        qprime_dict : dict
            Mapping {reaction_key -> Q′} returned by compute_all_Q().
        """
        # Pull Q′ values in the same order as the header keys
        q_values = [qprime_dict.get(k, 0.0) for k in self.keys]

        # Convert to CSV text
        row = f"{spectral_value}," + ",".join(f"{v:.6e}" for v in q_values)
        with open(self.csv_name, "a") as f:
            f.write(row + "\n")


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Input into light element Q' calculations. ")
    parser.add_argument("--key", type=str, default="",
                        help="keys for single cross section channel. Superceeds keys if"
                             " both are passed")
    parser.add_argument("--keys", type=str, default="",
                        help="keys for multiple cross section channels")
    parser.add_argument("--specttype", type=str, default="K2",
                        help="spectrum type")
    parser.add_argument("--spectindex", type=float, default=0.014,
                        help="spectral index")
    parser.add_argument("--spectstep", type=float, default=0.001,
                        help="spectral step for multi-spectral value computations")
    parser.add_argument("--spectstart", type=float, default=0.014,
                        help="starting spectral index for computations "
                             " that compute a range of spectral indicies")
    parser.add_argument("--spectend", type=float, default=0.016,
                        help="ending spectral index for computations "
                             " that compute a range of spectral indicies")
    parser.add_argument("--Enorm", type=float, default=30,
                        help="normalization constant for spectrum")
    parser.add_argument("--Emin", type=float, default=1.0,
                        help="lower limit for integration")
    parser.add_argument("--Emax", type=float, default=3000.0,
                        help="upper limit for integration")
    parser.add_argument("--E0p", type=float, default=30.0,
                        help="proton turnover for shock rigidity")
    parser.add_argument("--Z", type=int, default=1,
                        help="atomic number of accelerated particle, required for stopping power")
    parser.add_argument("--A", type=int, default=1,
                        help="atomic mass of accelerated particle, required for stopping power")


    args = parser.parse_args()
    key = args.key
    keys = args.keys
    spectral_type = args.specttype
    spectral_index = args.spectindex
    spect_step = args.spectstep
    spect_start = args.spectstart
    spect_end = args.spectend
    E_norm = args.Enorm
    E_min = args.Emin
    E_max = args.Emax
    E_0p = args.E0p
    Z = args.Z
    A = args.A

    if key:
        file_name = key
        # encode as a list, as that is the data structure code is expecting
        keys = list(key)
    else:
        file_name = keys

# --------------------------------------------------------
    reader = CrossSectionChannels("./csv")
    reader.read_all()
    cross_sections_keys = reader.get_keys(keys)

    if spectral_type == "K2":
        csv_name = "K2"
    elif spectral_type == "powerlaw":
        csv_name = "power_law"
    elif spectral_type == "shock":
        # For shock, encode the E0p range into the file name
        csv_name = f"shock_E0p_{E_0p:g}"
    else:
        csv_name = "error"
    csv_name = "/Users/marcus/astrophysics/spallation_calc_output/" + csv_name + "_" + file_name + ".csv"

    '''
        def __init__(
            self,
            spectral_type: str = "k2",
            spectral_index: float = 0.020,
            Z: int = 1,
            A: int = 1,
            E_0 = 30.0, # N_p(E>E_0) = 1 normalization
            E_min: float = 1.0, # starting energy for integration
            E_max: float = 3000.0, # finite GeV-range limit for integration
            E_0p : float = 30.0, # proton turnover for shock spectrum
            csv_name : str = "",
            reader : CrossSectionChannels = None,
            keys : List[str] = None
        ):
    '''
    model = QNeutralMediumNoGrammageSimpsonRule(
        spectral_type=spectral_type,
        spectral_index=spectral_index,
        Z=Z,
        A=A,
        E_0=E_norm,
        E_min=E_min,
        E_max=E_max,
        E_0p=E_0p,
        csv_name=csv_name,
        reader=reader,
        keys=cross_sections_keys
    )

    model.compute_Q_spect_range(spect_start=spect_start,  spect_end=spect_end, spect_step=spect_step)