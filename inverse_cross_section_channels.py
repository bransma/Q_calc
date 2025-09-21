from dataclasses import dataclass
import numpy as np
from cross_section_channels import CrossSectionReader

# atomic masses in MeV/c^2
MASS = {
    "p": 938.272,       # proton
    "He3": 3 * 931.494,
    "He4": 4 * 931.494,
    "C": 12 * 931.494,
    "N": 14 * 931.494,
    "O": 16 * 931.494,
}

@dataclass
class InverseCrossSectionReader:
    """
    Wrapper around CrossSectionReader that provides inverse-kinematics
    cross sections by reusing direct measurements.

    Example:
        p + 12C → LiBeB  (direct, tabulated)
        12C + p → LiBeB  (inverse, computed by kinematic scaling)
    """
    direct_reader: CrossSectionReader

    def sigma_direct(self, key: str, E: float | np.ndarray) -> float | np.ndarray:
        """
        Get direct-kinematics cross section.
        """
        return self.direct_reader.cross_section_interp[key](E)

    def sigma_inverse(
        self,
        key: str,
        E: float | np.ndarray,
        direct_proj: str,
        direct_targ: str,
    ) -> float | np.ndarray:
        """
        Compute inverse-kinematics cross section.

        Parameters
        ----------
        key : str
            Reaction key in direct_reader, e.g. "P12C_a6".
        E : float or array
            Projectile kinetic energy per nucleon in lab frame (MeV/n).
        direct_proj : str
            Projectile in the *direct* reaction (e.g. "p").
        direct_targ : str
            Target in the *direct* reaction (e.g. "C").
        """
        m_proj = MASS[direct_proj]
        m_targ = MASS[direct_targ]

        # inverse projectile lab energy maps onto scaled direct projectile lab energy
        E_direct_lab = (m_proj / m_targ) * np.array(E, ndmin=1)

        return self.direct_reader.cross_section_interp[key](E_direct_lab)