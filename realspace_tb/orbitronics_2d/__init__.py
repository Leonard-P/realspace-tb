"""
Graphene/orbitronics-specific components:
- Geometries
- Homogeneous-field Hamiltonians
- Observables
- OHC utilities
"""

from .honeycomb_geometry import HoneycombLatticeGeometry
from .lattice_2d_geometry import Lattice2DGeometry


from .homogeneous_field_hamiltonian import (
    RampedACFieldAmplitude,
    LinearFieldHamiltonian,
    LinearFieldHamiltonianPeierls,
    HomogeneousFieldAmplitude,
)

from . import observables as observables
from .plot_utils import show_simulation_frame, save_simulation_animation

from .ohc import ohc, fourier_at_omega


__all__ = [
    # modules
    "observables",
    # classes
    "HoneycombLatticeGeometry",
    "RampedACFieldAmplitude",
    "LinearFieldHamiltonian",
    "LinearFieldHamiltonianPeierls",
    "Lattice2DGeometry",
    "HomogeneousFieldAmplitude",
    # functions
    "ohc",
    "show_simulation_frame",
    "save_simulation_animation",
    "fourier_at_omega",
]