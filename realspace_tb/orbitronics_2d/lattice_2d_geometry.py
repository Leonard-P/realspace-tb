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

    def prepare_3d_current_segments(
        self,
        currents: NDArray[np.floating],
        n_images: int = 0,
        *,
        current_threshold: float = 1e-10,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        """Prepare 3D current segments from 2D bond currents for field calculations (e.g. Biot-Savart).

        This method extracts a 1D array of observable bond currents, filters out
        negligible values below `current_threshold`, and pairs them with their
        corresponding 3D start and end points in real space (padding z=0).

        The default implementation supports finite systems only (`n_images=0`).
        Periodic subclasses override this to yield periodically replicated
        segments across `[-n_images, n_images]` lattice translations.

        Returns:
            Tuple of (r_start, r_end, J):
            - r_start: (N_segments, 3) start coordinates in 3D.
            - r_end: (N_segments, 3) end coordinates in 3D.
            - J: (N_segments,) filtered and optionally replicated current values.
        """
        if n_images < 0:
            raise ValueError("n_images must be >= 0")
        if n_images > 0:
            raise NotImplementedError(
                "Periodic replication is not implemented for this geometry. If you use open boundaries, set n_images=0."
            )

        nn = self.nearest_neighbors
        currents_arr = np.asarray(currents, dtype=float).reshape(-1)
        if currents_arr.shape[0] != nn.shape[0]:
            raise ValueError(
                "currents must have one entry per nearest-neighbor bond"
            )

        r_i_2d = np.asarray(self.site_positions[nn[:, 0]], dtype=float)
        r_k_2d = r_i_2d + np.asarray(self.bond_vectors, dtype=float)

        mask = np.abs(currents_arr) > current_threshold
        r_i_2d = r_i_2d[mask]
        r_k_2d = r_k_2d[mask]
        J = currents_arr[mask]

        r_i = np.pad(r_i_2d, ((0, 0), (0, 1)), mode="constant")
        r_k = np.pad(r_k_2d, ((0, 0), (0, 1)), mode="constant")
        return r_i, r_k, J

    def extent_along(self, direction: NDArray[np.floating]) -> float:
        """Return geometric extent of the lattice sites along a 2D direction.

        The extent is computed as ``max(r·n_hat) - min(r·n_hat)`` over all
        site positions ``r`` and unit direction ``n_hat``.

        Args:
            direction: 2D direction vector. It does not need to be normalized.

        Returns:
            Real-space length of the system along ``direction`` in units of
            ``a_NN``.
        """
        n = np.asarray(direction, dtype=float).reshape(-1)
        if n.size != 2:
            raise ValueError("direction must be a 2D vector")

        norm = np.linalg.norm(n)
        if norm <= 0.0:
            raise ValueError("direction must be non-zero")

        n_hat = n / norm
        projections = np.asarray(self.site_positions, dtype=float) @ n_hat
        return float(np.max(projections) - np.min(projections))

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

