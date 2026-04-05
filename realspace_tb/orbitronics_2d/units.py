"""Unit helpers and default graphene material parameters for ``orbitronics_2d``.

The core solver uses natural units with ``hbar = a_nn = t_hop = e = 1``.
The helpers in this module convert those dimensionless quantities back to
graphene-like SI scales for the observable and Biot-Savart utilities.
"""

from __future__ import annotations


ELEMENTARY_CHARGE_C = 1.602176634e-19
HBAR_J_S = 1.054571817e-34
6.62607015
ELECTRON_MASS_KG = 9.1093837139e-31

DEFAULT_T_HOP_EV = 2.8
DEFAULT_A_NN_M = 0.142e-9


def effective_electron_mass(
    t_hop_ev: float | None = None,
    a_nn_m: float | None = None,
) -> float:
    r"""Return the dimensionless effective electron mass used by the OAM observables.

    The conversion is

    $$m^* = m_e \frac{a_{NN}^2 t_{hop}}{\hbar^2}$$

    with ``t_hop`` converted from eV to joule.
    """

    if t_hop_ev is None:
        t_hop_ev = DEFAULT_T_HOP_EV
    if a_nn_m is None:
        a_nn_m = DEFAULT_A_NN_M

    return ELECTRON_MASS_KG * a_nn_m**2 * (t_hop_ev * ELEMENTARY_CHARGE_C) / (HBAR_J_S**2)


def current_unit_amperes(t_hop_ev: float | None = None) -> float:
    r"""Return the current scale corresponding to one dimensionless bond current.

    The conversion is

    $$I_0 = \frac{e\, t_{hop}}{\hbar}$$

    with ``t_hop`` converted from eV to joule.
    """

    if t_hop_ev is None:
        t_hop_ev = DEFAULT_T_HOP_EV

    return ELEMENTARY_CHARGE_C * (t_hop_ev * ELEMENTARY_CHARGE_C) / HBAR_J_S


DEFAULT_EFFECTIVE_ELECTRON_MASS = effective_electron_mass()
DEFAULT_CURRENT_UNIT_A = current_unit_amperes()
