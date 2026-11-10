"""
realspace_tb: time-domain real-space tight binding simulations, primarily for finite graphene-like systems.

Core API:
- backend: precision/backend selection
- RK4NeumannSolver: time integrator (real space)
- BlochHamiltonian, KMesh, BlochRK4NeumannSolver: k-space (Bloch) simulation of systems
  periodic along one axis, using the translation symmetry (see bloch.py)
- orbitronics_2d: graphene-specific Hamiltonians, observables for plaquette-wise intersite current OAM (submodule)
"""

# Public core
from . import backend as backend
from .lindbladsolver import RK4NeumannSolver
from .hamiltonian import Hamiltonian
from .observable import Observable, MeasurementWindow
from .hamiltonian import fermi_occupation
from .bloch import BlochHamiltonian, KMesh, RingHamiltonian
from .bloch_solver import BlochRK4NeumannSolver, evaluate_observable

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
    "BlochRK4NeumannSolver",
    "BlochHamiltonian",
    "KMesh",
    "RingHamiltonian",
    "evaluate_observable",
    "fermi_occupation",
    "orbitronics_2d",
    "Hamiltonian",
    "Observable",
    "MeasurementWindow",
    "__version__",
]
