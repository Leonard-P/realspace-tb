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
from . import units as units
from .biot_savart import (
    net_current_vectors,
    calculate_biot_savart_vectorized,
    calculate_biot_savart_batch,
    biot_savart_on_plane,
)
from .plot_utils import show_simulation_frame, save_simulation_animation, PlotConfig, append_colorbar

from .ohc import ohc, fourier_at_omega


__all__ = [
    # modules
    "observables",
    "units",
    # classes
    "HoneycombLatticeGeometry",
    "RampedACFieldAmplitude",
    "LinearFieldHamiltonian",
    "LinearFieldHamiltonianPeierls",
    "Lattice2DGeometry",
    "HomogeneousFieldAmplitude",
    # functions
    "net_current_vectors",
    "calculate_biot_savart_vectorized",
    "calculate_biot_savart_batch",
    "biot_savart_on_plane",
    "ohc",
    "show_simulation_frame",
    "save_simulation_animation",
    "PlotConfig",
    "fourier_at_omega",
    "append_colorbar",
]