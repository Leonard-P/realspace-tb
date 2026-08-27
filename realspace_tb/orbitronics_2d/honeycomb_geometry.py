from .lattice_2d_geometry import Lattice2DGeometry
from .. import backend as B
import numpy as np
from numpy.typing import NDArray
from typing import Tuple


def get_valid_adjacent_pairs(lst):
    forward_pairs = []
    reverse_pairs = []
    n = len(lst)

    if n < 2:
        return [forward_pairs, reverse_pairs]

    for i in range(n):
        u = lst[i]
        v = lst[(i + 1) % n]

        if u != -1 and v != -1:
            forward_pairs.append((u, v))
            reverse_pairs.append((v, u))

    return [forward_pairs, reverse_pairs]


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
        # for sites
        self._nearest_neighbors: "B.FCPUArray | None" = None
        self._next_nearest_neighbors: "B.FCPUArray | None" = None
        self._nn_bond_vectors_cache: "NDArray[np.floating] | None" = None
        self._nnn_bond_vectors_cache: "NDArray[np.floating] | None" = None
        # ------------------------------------------------------------------------
        # for plaquettes
        self._nearest_plaquette_neighbors: "B.FCPUArray | None" = None
        self._boundary_nearest_plaquette_neighbors: "list[tuple] | None" = None
        self._nn_plaquette_vectors_cache: "NDArray[np.floating] | None" = None
        # ------------------------------------------------------------------------
        self._bravais_site_indices: "B.FCPUArray | None" = None
        self._plaquette_indices: "B.FCPUArray | None" = None
        self._plaquette_positions: "B.FCPUArray | None" = None
        self._plaq_xstep: np.floating | None = None
        self._plaq_ystep: np.floating | None = None

        assert np.allclose(
            np.linalg.norm(self.nn_bond_vectors, axis=1), 1.0
        )  # ensure that Lx, Ly are chosen for a correct supercell

    def _build_nearest_neighbors(self) -> None:
        """Populate ``_nearest_neighbors`` and ``_nn_bond_vectors_cache`` together."""
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
        self._nn_bond_vectors_cache = np.array(bond_vecs, dtype=B.FCPUDTYPE)

    def _build_next_nearest_neighbors(self) -> None:
        """Populate ``_next_nearest_neighbors`` and ``_nnn_bond_vectors_cache`` together."""
        Lx, Ly = self.Lx, self.Ly
        next_neighbors: list[list[int]] = []
        bond_vecs: list[NDArray[np.floating]] = []

        for index in range(Lx * Ly):
            row = index // Lx
            col = index % Lx

            for dr, dc in [(1, 1), (1, -1), (0, 2)]:
                nrow, ncol = row + dr, col + dc

                # periodic wrap flags (±1 if we crossed a boundary, else 0)
                wrap_r = wrap_c = 0
                # Handle periodic boundary conditions
                if not (0 <= nrow < Ly):
                    if not self.pbc_y:
                        continue
                    wrap_r = nrow // Ly
                    nrow %= Ly

                if not (0 <= ncol < Lx):
                    if not self.pbc_x:
                        continue
                    wrap_c = ncol // Lx
                    ncol %= Lx

                neighbor_index = nrow * Lx + ncol
                next_neighbors.append([index, neighbor_index])

                # Short bond vector: neighbour position in the periodic image
                r_i = self.index_to_position(index)
                r_j = self.index_to_position(neighbor_index)

                # Shift by the supercell dimensions to get the correct periodic image
                r_j_image = r_j + np.array(
                    [
                        wrap_c * Lx * self._col_width,
                        wrap_r * Ly * self._row_height,
                    ],
                    dtype=B.FCPUDTYPE,
                )
                bond_vecs.append(r_j_image - r_i)

        self._next_nearest_neighbors = np.array(next_neighbors, dtype=int)
        self._nnn_bond_vectors_cache = np.array(bond_vecs, dtype=B.FCPUDTYPE)

    def _build_nearest_plaquette_neighbors(self) -> None:
        """Populate ``_nearest_plaquette_neighbors`` and ``_nn_plaquette_vectors_cache`` together.
        Return bonds corresponding to vortex current between two neighboring plaquettes
        """
        neighbors: list[list[int]] = []
        plaq_boundaries: list[tuple] = []
        bond_vecs: list[NDArray[np.floating]] = []
        if self._plaquette_indices is None:
            self._list_plaquettes()

        list_of_plaquettes = self._plaquette_indices[0]
        # Extract the length of the plaquette-system along the x-direction
        plaq_Lx = (
            B.xp().max(self._plaquette_positions[:, 0])
            - B.xp().min(self._plaquette_positions[:, 0])
            + self._plaq_xstep
        )
        # Extract the length of the plaquette-system along the y-direction
        plaq_Ly = (
            B.xp().max(self._plaquette_positions[:, 1])
            - B.xp().min(self._plaquette_positions[:, 1])
            + self._plaq_ystep
        )
        # Construct x/y-distance and y-distance matrix
        x_coords = self._plaquette_positions[:, 0].reshape(-1, 1)
        y_coords = self._plaquette_positions[:, 1].reshape(-1, 1)
        # Subtract the x/y-coordinates as a (1, n_sites) row vector
        # array[i, j] = x/y_coords_row[j] - x/y_coords_col[i]
        x_dist_matrix = x_coords.T - x_coords
        y_dist_matrix = y_coords.T - y_coords
        if self.pbc_x:
            x_dist_matrix -= plaq_Lx * np.round(x_dist_matrix / plaq_Lx)
        if self.pbc_y:
            y_dist_matrix -= plaq_Ly * np.round(y_dist_matrix / plaq_Ly)
        for idx, plaq in enumerate(list_of_plaquettes[:-1], start=0):
            fwd_adj_plaq_pairs, back_adj_plaq_pairs = get_valid_adjacent_pairs(plaq)
            for idx2, plaq2 in enumerate(list_of_plaquettes[idx + 1 :], start=idx + 1):
                fwd_adj_plaq2_pairs, _ = get_valid_adjacent_pairs(plaq2)
                has_common = not set(back_adj_plaq_pairs).isdisjoint(
                    fwd_adj_plaq2_pairs
                )
                if has_common:
                    neighbors.append([idx, idx2])
                    intersecting_boundary = list(
                        set(back_adj_plaq_pairs).intersection(fwd_adj_plaq2_pairs)
                    )[0]
                    idx_intersection_boundary = back_adj_plaq_pairs.index(
                        intersecting_boundary
                    )
                    plaq_boundaries.append(
                        fwd_adj_plaq_pairs[idx_intersection_boundary]
                    )
                    bond_vecs.append(
                        [x_dist_matrix[idx, idx2], y_dist_matrix[idx, idx2]]
                    )

        self._nearest_plaquette_neighbors = np.array(neighbors, dtype=int)
        self._boundary_nearest_plaquette_neighbors = np.array(
            plaq_boundaries, dtype=int
        )
        self._nn_plaquette_vectors_cache = np.array(bond_vecs, dtype=B.FCPUDTYPE)

    def _list_plaquettes(self) -> None:
        """List all plaquettes as lists of site indices in CCW order.
        Includes boundary pseudo-plaquettes padded with -1 in their strict geometric slots
        to rigorously preserve physical edge orientations.
        """
        Lx, Ly = self.Lx, self.Ly
        pbc_x, pbc_y = self.pbc_x, self.pbc_y

        # Expand anchor boundaries slightly to catch partial faces on open edges
        ax_min = 0 if pbc_x else -2
        ax_max = Lx - 1
        ay_min = 0 if pbc_y else -1
        ay_max = Ly - 1

        ax_grid, ay_grid = np.meshgrid(
            np.arange(ax_min, ax_max + 1), np.arange(ay_min, ay_max + 1)
        )
        mask = (ax_grid + ay_grid) % 2 == 0
        ax = ax_grid[mask]
        ay = ay_grid[mask]

        path_offsets = self.plaquette_path_offsets_ccw.astype(np.int64)
        i_offsets = path_offsets[:, 0]
        i_dr = i_offsets // Lx
        i_dc = i_offsets % Lx

        rows_cpu_list = []
        cols_cpu_list = []
        # anchors_list = []
        positions_list = []

        for x, y in zip(ax, ay):
            nodes = []
            for n_idx in range(6):
                r = y + i_dr[n_idx]
                c = x + i_dc[n_idx]

                # Check bounds and apply PBC wrapping
                if pbc_x:
                    c = c % Lx
                elif not (0 <= c < Lx):
                    nodes.append(-1)
                    continue

                if pbc_y:
                    r = r % Ly
                elif not (0 <= r < Ly):
                    nodes.append(-1)
                    continue

                nodes.append(r * Lx + c)

            # A plaquette requires at least 2 valid boundary sites
            valid_count = sum(1 for n in nodes if n != -1)
            if valid_count >= 2:
                # Assign u and v by strictly retaining topological slots.
                # This guarantees edges automatically inherit the correct CCW sequence.
                edges_u = nodes
                edges_v = [nodes[(i + 1) % 6] for i in range(6)]

                rows_cpu_list.append(edges_u)
                cols_cpu_list.append(edges_v)
                # anchors_list.append(y * Lx + x)

                # Compute the ideal geometric center directly from the anchor coordinate
                center_x = float((x + 1) * self._col_width)
                center_y = float((y + 0.5) * self._row_height)
                positions_list.append([center_x, center_y])

        rows_cpu = np.array(rows_cpu_list, dtype=np.int64)
        cols_cpu = np.array(cols_cpu_list, dtype=np.int64)

        self._plaquette_indices = B.xp().array([rows_cpu, cols_cpu], dtype=B.xp().int64)
        self._plaquette_positions = B.xp().array(positions_list, dtype=B.FCPUDTYPE)

        unique_x = B.xp().unique(self._plaquette_positions[:, 0])
        self._plaq_xstep = float(B.xp().diff(unique_x)[0]) if len(unique_x) > 1 else 0.0

        unique_y = B.xp().unique(self._plaquette_positions[:, 1])
        self._plaq_ystep = float(B.xp().diff(unique_y)[0]) if len(unique_y) > 1 else 0.0

    @property
    def plaquettes(self) -> Tuple[B.FCPUArray, B.FCPUArray, float, float]:
        """Tuple of (plaquette_indices, plaquette_positions).
        plaquette_indices: Array of site indices in CCW order for each plaquette [[i, j, k, ...], ...].
        plaquette_positions: Array of positions for each plaquette, center of boundary sites.
        plaq_xstep: x-distance of two nearest neighboring plaquette centers.
        plaq_ystep: y-distance of two nearest neighboring plaquette centers.
        """
        if self._plaquette_indices is None:
            self._list_plaquettes()
        return (
            self._plaquette_indices,
            self._plaquette_positions,
            self._plaq_xstep,
            self._plaq_ystep,
        )

    @property
    def site_plaquette_count(self) -> B.FCPUArray:
        """Count of boundary plaquettes per site index."""
        plaquette_indices, _, _, _ = self.plaquettes

        all_sites = plaquette_indices[0].ravel()

        # Filter out the -1 padding dynamically generated for boundary plaquettes
        valid_sites = all_sites[all_sites >= 0]
        N = self.Lx * self.Ly

        counts = np.bincount(valid_sites.astype(np.int64), minlength=N)

        return B.xp().array(counts, dtype=B.xp().int64)

    @property
    def nearest_neighbors(self) -> B.FCPUArray:
        """Array of nearest neighbor indices [[i, j], ...] = <i, j>"""
        if self._nearest_neighbors is None:
            self._build_nearest_neighbors()
        return self._nearest_neighbors  # type: ignore[return-value]

    @property
    def next_nearest_neighbors(self) -> B.FCPUArray:
        """Array of next-nearest neighbor indices [[i, j], ...] = <i, j>"""
        if self._next_nearest_neighbors is None:
            self._build_next_nearest_neighbors()
        return self._next_nearest_neighbors  # type: ignore[return-value]

    @property
    def nn_bond_vectors(self) -> NDArray[np.floating]:
        """Short bond displacement vectors ``r_j - r_i`` for each neighbor pair.

        For periodic bonds the vector points to the nearest periodic image,
        not across the full system.
        """
        if self._nn_bond_vectors_cache is None:
            self._build_nearest_neighbors()
        return self._nn_bond_vectors_cache  # type: ignore[return-value]

    @property
    def nnn_bond_vectors(self) -> NDArray[np.floating]:
        """Short next-nearest neighbor bond displacement vectors ``r_j - r_i`` for each neighbor pair.

        For periodic bonds the vector points to the nearest periodic image,
        not across the full system.
        """
        if self._nnn_bond_vectors_cache is None:
            self._build_next_nearest_neighbors()
        return self._nnn_bond_vectors_cache  # type: ignore[return-value]

    @property
    def nearest_plaquette_neighbors(self) -> B.FCPUArray:
        """Array of nearest neighbors PLAQUETTES [[i, j], ...] = <i, j>"""
        if self._nearest_plaquette_neighbors is None:
            self._build_nearest_plaquette_neighbors()
        return self._nearest_plaquette_neighbors

    @property
    def boundary_nn_plaquette_vectors(self) -> NDArray[np.floating]:
        """Boundaries from intersection between two neighboring plaquettes"""
        if self._nn_bond_vectors_cache is None:
            self._build_nearest_plaquette_neighbors()
        return self._boundary_nearest_plaquette_neighbors  # type: ignore[return-value]

    @property
    def nn_plaquette_vectors(self) -> NDArray[np.floating]:
        """Short PLAQUETTE displacement vectors ``r_j - r_i`` for each neighbor plaquette pair.

        For periodic bonds the vector points to the nearest periodic image,
        not across the full system.
        """
        if self._nn_bond_vectors_cache is None:
            self._build_nearest_plaquette_neighbors()
        return self._nn_plaquette_vectors_cache  # type: ignore[return-value]

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
        r_k_2d = r_i_2d + np.asarray(self.nn_bond_vectors, dtype=float)

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
