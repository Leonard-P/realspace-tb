"""Unit helpers and default graphene material parameters for ``orbitronics_2d``.

The core solver uses mixed units, where ``e = hbar = m_e = 4\pi\varepsilon_0 = a_nn = 1``.
Proper value of t_hop is set in the Hamiltonian. However, the proper value of a_nn must be set when calculating induced fields.
The helpers in this module convert those dimensionless quantities back to
graphene-like SI scales for the observable and Biot-Savart utilities.
"""

from __future__ import annotations
from scipy.constants import e, hbar, h, m_e, value

ELEMENTARY_CHARGE_C = e
HBAR_J_S = hbar
PLANCK_J_S = h
ELECTRON_MASS_KG = m_e
HARTREE_ENERGY_J = value("Hartree energy")

DEFAULT_T_HOP_AU = 0.1
DEFAULT_A_NN_M = 0.142e-9


def effective_electron_mass(
    t_hop_au: float | None = None,
    a_nn_m: float | None = None,
) -> float:
    r"""Return the effective electron mass using t_hop in Hartree atomic units."""
    if t_hop_au is None:
        t_hop_au = DEFAULT_T_HOP_AU
    if a_nn_m is None:
        a_nn_m = DEFAULT_A_NN_M

    # Convert Hartree to Joules, rather than eV to Joules
    t_hop_j = t_hop_au * HARTREE_ENERGY_J

    return ELECTRON_MASS_KG * a_nn_m**2 * t_hop_j / HBAR_J_S**2


def current_unit_amperes() -> float:
    r"""Return the SI current corresponding to one Hartree atomic unit of current.
    Conversion is I_au = e * E_h / hbar.
    """
    return ELEMENTARY_CHARGE_C * HARTREE_ENERGY_J / HBAR_J_S


DEFAULT_EFFECTIVE_ELECTRON_MASS = effective_electron_mass()
DEFAULT_CURRENT_UNIT_A = current_unit_amperes()
