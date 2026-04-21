from .lattice_2d_geometry import Lattice2DGeometry
from .. import backend as B
import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class HoneycombLatticeGeometry(Lattice2DGeometry):
    def __init__(self, Lx: int, Ly: int, pbc_x: bool = False, pbc_y: bool = False):
        super().__init__()

        self.Lx = Lx
        self.Ly = Ly
        self.pbc_x = pbc_x
        self.pbc_y = pbc_y

        self.plaquette_path_offsets_ccw = np.array(
            [(0, 1), (1, 2), (2, Lx + 2), (Lx + 2, Lx + 1), (Lx + 1, Lx), (Lx, 0)]
        )

        self._row_height = 1.5
        self._col_width = np.sqrt(3) / 2

        self.plaquette_area = np.sqrt(3) * 3 / 2

        self._origin = (
            np.array(
                [(self.Lx - 1) * self._col_width, (self.Ly - 1) * self._row_height]
            )
            / 2
        )
        self._nearest_neighbors: "B.FCPUArray | None" = None
        self._bond_vectors_cache: "NDArray[np.floating] | None" = None
        self._bravais_site_indices: "B.FCPUArray | None" = None
        self._plaquette_indices: "B.FCPUArray | None" = None
        self._plaquette_positions: "B.FCPUArray | None" = None
        self._plaquette_anchors_cpu: "B.FCPUArray | None" = None

        assert np.allclose(
            np.linalg.norm(self.bond_vectors, axis=1), 1.0
        )  # ensure that Lx, Ly are chosen for a correct supercell

    def _build_neighbors(self) -> None:
        """Populate ``_nearest_neighbors`` and ``_bond_vectors_cache`` together."""
        Lx, Ly = self.Lx, self.Ly
        neighbors: list[list[int]] = []
        bond_vecs: list[NDArray[np.floating]] = []

        for index in range(Lx * Ly):
            row = index // Lx
            col = index % Lx

            if (row + col) % 2 == 0:
                continue  # emit each (A, B) pair once

            for dr, dc in [(-1, 0), (0, -1), (0, 1)]:
                nrow, ncol = row + dr, col + dc

                # periodic wrap flags (±1 if we crossed a boundary, else 0)
                wrap_r = wrap_c = 0
                if not (0 <= nrow < Ly):
                    if not self.pbc_y:
                        continue
                    wrap_r = -1 if nrow < 0 else 1
                    nrow %= Ly
                if not (0 <= ncol < Lx):
                    if not self.pbc_x:
                        continue
                    wrap_c = -1 if ncol < 0 else 1
                    ncol %= Lx

                neighbor_index = nrow * Lx + ncol
                neighbors.append([index, neighbor_index])

                # Short bond vector: neighbour position in the periodic image
                r_i = self.index_to_position(index)
                r_j = self.index_to_position(neighbor_index)
                r_j_image = r_j + np.array(
                    [
                        wrap_c * Lx * self._col_width,
                        wrap_r * Ly * self._row_height,
                    ],
                    dtype=B.FCPUDTYPE,
                )
                bond_vecs.append(r_j_image - r_i)

        self._nearest_neighbors = np.array(neighbors, dtype=int)
        self._bond_vectors_cache = np.array(bond_vecs, dtype=B.FCPUDTYPE)

    def _list_plaquettes(self) -> None:
        """List all plaquettes as lists of site indices in CCW order."""
        Lx, Ly = self.Lx, self.Ly
        pbc_x, pbc_y = self.pbc_x, self.pbc_y
        # Create plaquette edge indices from anchor positions using relative coordinate offsets

        ax = self.bravais_site_indices % Lx
        ay = self.bravais_site_indices // Lx

        self._plaquette_anchors_cpu = self.bravais_site_indices

        path_offsets = self.plaquette_path_offsets_ccw.astype(np.int64)
        i_offsets = path_offsets[:, 0]
        j_offsets = path_offsets[:, 1]

        i_dr = i_offsets // Lx
        i_dc = i_offsets % Lx
        j_dr = j_offsets // Lx
        j_dc = j_offsets % Lx

        rows_r = ay[:, None] + i_dr[None, :]
        rows_c = ax[:, None] + i_dc[None, :]
        cols_r = ay[:, None] + j_dr[None, :]
        cols_c = ax[:, None] + j_dc[None, :]

        if pbc_x:
            rows_c = rows_c % Lx
            cols_c = cols_c % Lx
            x_in_bounds = np.ones_like(rows_c, dtype=bool)
        else:
            x_in_bounds = (rows_c >= 0) & (rows_c < Lx) & (cols_c >= 0) & (cols_c < Lx)

        if pbc_y:
            rows_r = rows_r % Ly
            cols_r = cols_r % Ly
            y_in_bounds = np.ones_like(rows_r, dtype=bool)
        else:
            y_in_bounds = (rows_r >= 0) & (rows_r < Ly) & (cols_r >= 0) & (cols_r < Ly)

        in_bounds = x_in_bounds & y_in_bounds

        if pbc_x or pbc_y:
            valid_edge = in_bounds
        else:
            dx_r = rows_r - ay[:, None]
            dy_r = rows_c - ax[:, None]
            dx_c = cols_r - ay[:, None]
            dy_c = cols_c - ax[:, None]

            flat = path_offsets.ravel()
            w = int((flat % Lx).max())
            h = int((flat // Lx).max())

            valid_rows = (dx_r >= 0) & (dy_r >= 0) & (dx_r <= w + 1) & (dy_r <= h + 1)
            valid_cols = (dx_c >= 0) & (dy_c >= 0) & (dx_c <= w + 1) & (dy_c <= h + 1)

            valid_edge = in_bounds & valid_rows & valid_cols

        rows_cpu = (rows_r * Lx + rows_c).astype(np.int64)
        cols_cpu = (cols_r * Lx + cols_c).astype(np.int64)

        # only keep plaquettes whose edges all lie within the system
        cell_mask = valid_edge.all(axis=1)
        self._plaquette_anchors_cpu = self._plaquette_anchors_cpu[cell_mask]
        self._n_cells = self._plaquette_anchors_cpu.shape[0]

        rows_cpu = rows_cpu[cell_mask]
        cols_cpu = cols_cpu[cell_mask]

        self._plaquette_indices = B.xp().array([rows_cpu, cols_cpu], dtype=B.xp().int64)

        # compute plaquette centers, handling PBC by unwrapping in periodic directions
        plaquette_positions = []
        for indices in rows_cpu:
            positions = np.array([self.index_to_position(int(idx)) for idx in indices])
            if self.pbc_y:
                lattice_y = self.Ly * self._row_height
                y_vals = positions[:, 1]
                if np.max(y_vals) - np.min(y_vals) > lattice_y / 2:
                    positions[:, 1] = np.where(
                        y_vals < lattice_y / 2, y_vals + lattice_y, y_vals
                    )
            if self.pbc_x:
                lattice_x = self.Lx * self._col_width
                x_vals = positions[:, 0]
                if np.max(x_vals) - np.min(x_vals) > lattice_x / 2:
                    positions[:, 0] = np.where(
                        x_vals < lattice_x / 2, x_vals + lattice_x, x_vals
                    )
            center = np.mean(positions, axis=0)
            plaquette_positions.append(center)
        self._plaquette_positions = B.xp().array(plaquette_positions, dtype=B.FCPUDTYPE)

    @property
    def plaquettes(self) -> Tuple[B.FCPUArray, B.FCPUArray]:
        """Tuple of (plaquette_indices, plaquette_positions).
        plaquette_indices: Array of site indices in CCW order for each plaquette [[i, j, k, ...], ...]
        plaquette_positions: Array of positions for each plaquette, center of boundary sites.
        """
        if self._plaquette_indices is None:
            self._list_plaquettes()
        return self._plaquette_indices, self._plaquette_positions

    @property
    def site_plaquette_count(self) -> B.FCPUArray:
        """Count of boundary plaquettes per site index."""
        # ensure plaquette data exists
        plaquette_indices, _ = self.plaquettes  # (2, n_cells, n_edges)

        # flatten cell edges
        all_sites = plaquette_indices[0].ravel()
        N = self.Lx * self.Ly

        # use np.bincount on CPU (fast indexed count)
        counts = np.bincount(all_sites.astype(np.int64), minlength=N)

        # cast to backend array
        return B.xp().array(counts, dtype=B.xp().int64)

    @property
    def nearest_neighbors(self) -> B.FCPUArray:
        """Array of nearest neighbor indices [[i, j], ...] = <i, j>"""
        if self._nearest_neighbors is None:
            self._build_neighbors()
        return self._nearest_neighbors  # type: ignore[return-value]

    @property
    def bond_vectors(self) -> NDArray[np.floating]:
        """Short bond displacement vectors ``r_j - r_i`` for each neighbor pair.

        For periodic bonds the vector points to the nearest periodic image,
        not across the full system.
        """
        if self._bond_vectors_cache is None:
            self._build_neighbors()
        return self._bond_vectors_cache  # type: ignore[return-value]

    @property
    def bravais_site_indices(self) -> B.FCPUArray:
        """List of all indices that form the Bravais lattice."""
        if self._bravais_site_indices is not None:
            return self._bravais_site_indices

        # Return indices where (i + j) % 2 == 0 (A sublattice)
        self._bravais_site_indices = np.array(
            [i for i in range(self.Lx * self.Ly) if sum(divmod(i, self.Lx)) % 2 == 0]
        )
        return self._bravais_site_indices

    @property
    def plaquette_anchors(self) -> B.FCPUArray:
        """Indices of the anchor sites for each plaquette."""
        if self._plaquette_anchors_cpu is None:
            self._list_plaquettes()
        return self._plaquette_anchors_cpu

    @property
    def origin(self) -> B.FCPUArray:
        """Origin of the lattice as real space vector."""
        return self._origin

    def index_to_position(self, index: int) -> B.FCPUArray:
        row = index // self.Lx
        col = index % self.Lx

        y_offset = 0.25 * (-1) ** ((col + row) % 2)

        x = self._col_width * (index % self.Lx)
        y = self._row_height * row + y_offset

        return np.array([x, y], dtype=B.FCPUDTYPE)

    def prepare_3d_current_segments(
        self,
        currents: NDArray[np.floating],
        n_images: int = 0,
        *,
        current_threshold: float = 1e-10,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        """Prepare 3D current segments, extending the system periodically if requested.

        This converts the internal 2D bond currents into explicit 3D segments
        (r_start, r_end, J) used for external sums like the Biot-Savart law.
        It filters out magnitudes below `current_threshold` for efficiency.

        If `n_images > 0`, it replicates the physical segments in periodic
        directions (where `pbc_x` or `pbc_y` are True) across the spatial range
        `[-n_images, n_images]`.
        """
        if n_images < 0:
            raise ValueError("n_images must be >= 0")

        nn = self.nearest_neighbors
        currents_arr = np.asarray(currents, dtype=float).reshape(-1)
        if currents_arr.shape[0] != nn.shape[0]:
            raise ValueError("currents must have one entry per nearest-neighbor bond")

        r_i_2d = np.asarray(self.site_positions[nn[:, 0]], dtype=float)
        r_k_2d = r_i_2d + np.asarray(self.bond_vectors, dtype=float)

        mask = np.abs(currents_arr) > current_threshold
        r_i_2d = r_i_2d[mask]
        r_k_2d = r_k_2d[mask]
        J_base = currents_arr[mask]

        if J_base.size == 0:
            return (
                np.zeros((0, 3), dtype=float),
                np.zeros((0, 3), dtype=float),
                np.zeros((0,), dtype=float),
            )

        x_range = range(-n_images, n_images + 1) if self.pbc_x else range(1)
        y_range = range(-n_images, n_images + 1) if self.pbc_y else range(1)

        cell_x = self.Lx * self._col_width
        cell_y = self.Ly * self._row_height
        shifts_2d = np.array(
            [[ix * cell_x, iy * cell_y] for ix in x_range for iy in y_range],
            dtype=float,
        )

        r_i_rep_2d = (r_i_2d[:, np.newaxis, :] + shifts_2d[np.newaxis, :, :]).reshape(
            -1, 2
        )
        r_k_rep_2d = (r_k_2d[:, np.newaxis, :] + shifts_2d[np.newaxis, :, :]).reshape(
            -1, 2
        )
        J_rep = np.repeat(J_base, shifts_2d.shape[0])

        r_i = np.pad(r_i_rep_2d, ((0, 0), (0, 1)), mode="constant")
        r_k = np.pad(r_k_rep_2d, ((0, 0), (0, 1)), mode="constant")
        return r_i, r_k, J_rep
