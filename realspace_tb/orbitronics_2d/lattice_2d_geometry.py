from abc import ABC, abstractmethod
from .. import backend as B
import numpy as np
from numpy.typing import NDArray

class Lattice2DGeometry(ABC):
    def __init__(self) -> None:
        self._site_positions: "B.FCPUArray | None" = None

    @property
    def site_positions(self) -> B.FCPUArray:
        if self._site_positions is None:
            self._site_positions = np.array([
                self.index_to_position(i) for i in range(self.Lx * self.Ly)
            ], dtype=B.FCPUDTYPE)
        return self._site_positions

    @property
    def bond_vectors(self) -> NDArray[np.floating]:
        """Displacement vectors ``r_j - r_i`` for each nearest-neighbor pair ``[i, j]``.

        For open-boundary geometries the default implementation derives these
        from ``site_positions``.  Subclasses with periodic boundary conditions
        must override this so wrapped bonds return the *short* vector rather
        than the full lattice-traversal vector.
        """
        nn = self.nearest_neighbors
        pos = self.site_positions
        return pos[nn[:, 1]] - pos[nn[:, 0]]

    @abstractmethod
    def index_to_position(self, index: int) -> B.FCPUArray:
        """Convert site index to real space position"""
        ...

    @property
    @abstractmethod
    def nearest_neighbors(self) -> B.FCPUArray:
        """Array of nearest neighbor indices [[i, j], ...] = <i, j>"""
        ...

    @property
    @abstractmethod
    def bravais_site_indices(self) -> B.FCPUArray:
        """List of all indices that form the Bravais lattice."""
        ...

    @property
    def origin(self) -> B.FCPUArray:
        """Origin of the lattice as real space vector."""
        return np.array([0.0, 0.0], dtype=B.FCPUDTYPE)

    Lx: int
    Ly: int

    # [[i, j], ...] the integer offsets of the plaquette that need to be added to the bravais lattice index to traverse the ring of bonds i->j around the plaquette counter-clockwise (looking against z)
    plaquette_path_offsets_ccw: NDArray[np.int_]

    # real space area of a single plaquette, often the unit cell area
    plaquette_area: float

