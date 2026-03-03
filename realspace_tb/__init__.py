"""
realspace_tb: time-domain real-space tight binding simulations, primarily for finite graphene-like systems.

Core API:
- backend: precision/backend selection
- RK4NeumannSolver: time integrator
- orbitronics_2d: graphene-specific Hamiltonians, observables for plaquette-wise intersite current OAM (submodule)
"""

# Public core
from . import backend as backend
from .rk4 import RK4NeumannSolver
from .hamiltonian import Hamiltonian
from .observable import Observable, MeasurementWindow

# Version
from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("realspace_tb")
except PackageNotFoundError:
    __version__ = "0+unknown"


# subpackage namespace
from . import orbitronics_2d as orbitronics_2d

__all__ = [
    "backend",
    "RK4NeumannSolver",
    "orbitronics_2d",
    "Hamiltonian",
    "Observable",
    "MeasurementWindow",
    "__version__",
]