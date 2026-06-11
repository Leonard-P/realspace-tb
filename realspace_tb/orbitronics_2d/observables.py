from ..observable import Observable, MeasurementWindow
from .lattice_2d_geometry import Lattice2DGeometry
from .honeycomb_geometry import HoneycombLatticeGeometry
from ..hamiltonian import Hamiltonian
from .. import backend as B
from .units import effective_electron_mass
from typing import cast, Tuple
import numpy as np
from collections import deque


class VorticityObservable(Observable):
    r"""Measures the vorticity at each smallest possible plaquette

    $$\sum_{(R,R')\in\partial P} \hat{J}_{R\rightarrow R'}$$
    """

    def __init__(
        self,
        geometry: HoneycombLatticeGeometry,
        window: MeasurementWindow | None = None,
        hamiltonian: Hamiltonian | None = None,
    ):
        super().__init__(window)

        if hamiltonian is None:
            print(
                f"No Hamiltonian passed to {self.__class__}. Assuming Onsite-Potential Hamiltonian with t_hop=1."
            )

        self._hamiltonian = hamiltonian
        # changing the prefactor to calculate vorticity instead
        self._c = 1.0  # -electron_mass * geometry.plaquette_area / 3
        plaquette_indices, _, _, _ = geometry.plaquettes
        self._rows = plaquette_indices[0]
        self._cols = plaquette_indices[1]

    def _compute_edge_currents(self, rho: B.Array, t: float) -> B.Array:
        r"""Compute bond currents along plaquette edges.

        Uses the gauge-invariant formula
        $I_{i<-j}(t) = 2\,\mathrm{Im}(H_{ij}(t)\,\rho_{ji}(t))$.
        When no Hamiltonian is stored (`t_{hop} = -1`), this reduces to
        $2\,\mathrm{Im}(\rho_{ij})$.
        """
        invalid_mask = (self._rows == -1) | (self._cols == -1)
        if self._hamiltonian is not None:
            H_t = self._hamiltonian.at_time(t)
            xp = B.xp()
            rows_flat = self._rows.ravel()
            cols_flat = self._cols.ravel()
            # Order reversed due to sign-change from -1.0 to +1.0 in self._c, which flips the current direction convention and thus the order of indices in the current formula. This is a bit subtle and could be made clearer by defining a helper function for the current that takes care of the index ordering and sign convention.
            # h_ij = xp.asarray(H_t[rows_flat, cols_flat]).reshape(self._rows.shape)
            h_ij = xp.asarray(H_t[cols_flat, rows_flat]).reshape(self._cols.shape)
            rho_values = rho[self._rows, self._cols]
            rho_values = B.xp().where(invalid_mask, 0.0, rho_values)
            # return 2.0 * xp.imag(h_ij * rho[self._cols, self._rows])
            return 2.0 * xp.imag(h_ij * rho_values)
        rho_values = rho[self._cols, self._rows]
        rho_values = B.xp().where(invalid_mask, 0.0, rho_values)
        # return 2.0 * xp.imag(h_ij * rho[self._rows, self._cols])
        return 2.0 * B.xp().imag(rho_values)

    def _compute_vortices(self, rho: B.Array, t: float) -> B.Array:
        I_edges = self._compute_edge_currents(rho, t)
        return B.xp().sum(I_edges, axis=1)  # (n_cells,)

    def _compute(self, rho: B.Array, t: float) -> B.Array:
        return self._c * self._compute_vortices(rho, t)


class VortFluxObservable(VorticityObservable):
    r"""Measures the unmodified vorticity flux

    $$\hat{\Tilde{j}}^{z}_{\omega,P_{i}\rightarrow P_{j}}
    =(e_{z}\cross n_{P_{i} -> P})\cdot\sum_{X}\langle\hat{\Pi}^{(b)}_{X}\rangle(t)$$
    This observable is ANTISYMMETRIC, since n_{P_{i} -> P} = -n_{P -> P_{i}}
    1. WARNING: Currently hard-coded for 2D lattices, whose plaquettes have all their normal
                vectors pointing along one (z-)direction.
    """

    def __init__(
        self,
        geometry: HoneycombLatticeGeometry,
        window: MeasurementWindow | None = None,
        hamiltonian: Hamiltonian | None = None,
    ):
        super().__init__(geometry, window, hamiltonian)
        self._nearest_plaquette_neighbors = geometry.nearest_plaquette_neighbors
        self._boundary_plaquette_vectors = geometry.boundary_nn_plaquette_vectors
        self._num_plaquettes, _ = self._rows.shape

    def _compute_div_edge_fluxes_per_bond(self, rho: B.Array, t: float) -> B.Array:
        r"""Measures divergent of the current flux tensor for each bond at
        the intersection of two neighboring plaquettes $$\sum_{X}\langle\hat{\Pi}^{R->R'}_{X}\rangle(t)$$
        """
        r_arr = B.xp().array(self._boundary_plaquette_vectors[:, 0], dtype=B.xp().int64)
        rp_arr = B.xp().array(
            self._boundary_plaquette_vectors[:, 1], dtype=B.xp().int64
        )
        assert len(r_arr) == len(
            rp_arr
        ), "Nearest neighbor list should have shape (n_edges, 2)."
        if self._hamiltonian is not None:
            xp = B.xp()
            H_t = self._hamiltonian.at_time(t)
            h_scalars = xp.asarray(H_t[rp_arr, r_arr]).ravel()
            H_rho = H_t.dot(rho)
            rho_H = rho @ H_t
            H_rho_vals = H_rho[r_arr, rp_arr]
            rho_H_vals = rho_H[r_arr, rp_arr]
            return 2.0 * xp.real(h_scalars * (H_rho_vals - rho_H_vals))
        return 2.0 * B.xp().real(rho_H_vals - H_rho_vals)

    def _compute_tilde_j_omega(self, rho: B.Array, t: float) -> B.Array:
        div_edge_fluxes_per_bond = self._compute_div_edge_fluxes_per_bond(rho, t)
        tilde_j_omega = B.xp().zeros(
            (self._num_plaquettes, self._num_plaquettes),
            dtype=div_edge_fluxes_per_bond.dtype,
        )
        i_indices = B.xp().array(
            self._nearest_plaquette_neighbors[:, 0], dtype=B.xp().int64
        )
        j_indices = B.xp().array(
            self._nearest_plaquette_neighbors[:, 1], dtype=B.xp().int64
        )

        tilde_j_omega[i_indices, j_indices] = div_edge_fluxes_per_bond
        tilde_j_omega[j_indices, i_indices] = -div_edge_fluxes_per_bond
        return tilde_j_omega

    def _compute(self, rho: B.Array, t: float) -> B.Array:
        return self._compute_tilde_j_omega(rho, t)


class VortFluxModObservable(VortFluxObservable):
    r"""Measures the modified vorticity flux

    $$\hat{j}^{z}_{\omega,P_{i}\rightarrow P_{j}}
    =(e_{z}\cross n_{P_{i} -> P})\cdot\biggl(\langle\sum_{X}\hat{\Pi}^{(b)}_{X}\rangle(t)
    -\langle\hat{f}^{(b)}\rangle(t)\biggr)$$
    This observable is ANTISYMMETRIC, since n_{P_{i} -> P} = -n_{P -> P_{i}}
    1. WARNING: Currently hard-coded for 2D lattices, whose plaquettes have all their normal
                vectors pointing along one (z-)direction.
    """

    def __init__(
        self,
        geometry: HoneycombLatticeGeometry,
        window: MeasurementWindow | None = None,
        hamiltonian: Hamiltonian | None = None,
    ):
        super().__init__(geometry, window, hamiltonian)
        self._nearest_plaquette_neighbors = geometry.nearest_plaquette_neighbors
        self._boundary_plaquette_vectors = geometry.boundary_nn_plaquette_vectors
        self._num_plaquettes, _ = self._rows.shape

    def _compute_edge_forces_per_bond(self, rho: B.Array, t: float) -> B.Array:
        r"""Measures forces for each bond at the intersection of two neighboring plaquettes
        $$\sum_{X}\langle\hat{\Pi}^{R->R'}_{X}\rangle(t)$$"""
        r_arr = B.xp().array(self._boundary_plaquette_vectors[:, 0], dtype=B.xp().int64)
        rp_arr = B.xp().array(
            self._boundary_plaquette_vectors[:, 1], dtype=B.xp().int64
        )
        assert len(r_arr) == len(
            rp_arr
        ), "Nearest neighbor list should have shape (n_edges, 2)."
        if self._hamiltonian is not None:
            xp = B.xp()
            dH_dt = self._hamiltonian.derivative_at_time(t)
            h_scalars = xp.asarray(dH_dt[rp_arr, r_arr]).ravel()
            # return 2.0 * xp.real(h_scalars * (H_rho_vals - rho_H_vals))
            return 2.0 * xp.imag(h_scalars * rho[r_arr, rp_arr])
        return 2.0 * B.xp().imag(rho[rp_arr, r_arr])

    def _compute_f_j_omega(self, rho: B.Array, t: float) -> B.Array:
        edge_forces_per_bond = self._compute_edge_forces_per_bond(rho, t)
        f_j_omega = B.xp().zeros(
            (self._num_plaquettes, self._num_plaquettes),
            dtype=edge_forces_per_bond.dtype,
        )
        i_indices = B.xp().array(
            self._nearest_plaquette_neighbors[:, 0], dtype=B.xp().int64
        )
        j_indices = B.xp().array(
            self._nearest_plaquette_neighbors[:, 1], dtype=B.xp().int64
        )

        f_j_omega[i_indices, j_indices] = edge_forces_per_bond
        f_j_omega[j_indices, i_indices] = -edge_forces_per_bond
        return -f_j_omega

    def _compute(self, rho: B.Array, t: float) -> B.Array:
        return self._compute_tilde_j_omega(rho, t) + self._compute_f_j_omega(rho, t)


class VortSourceObservable(VorticityObservable):
    r"""Measures the vorticity source

    $$\hat{\Omega}_{f,P_{i}}=\sum_{(\mathbf{R},\mathbf{R}')\in\partial P_{i}}\hat{f}_{\mathbf{R}\rightarrow\mathbf{R}'}$$
    """

    def __init__(
        self,
        geometry: HoneycombLatticeGeometry,
        window: MeasurementWindow | None = None,
        hamiltonian: Hamiltonian | None = None,
    ):
        super().__init__(geometry, window, hamiltonian)

        if hamiltonian is None:
            print(
                f"No Hamiltonian passed to {self.__class__}. Assuming Onsite-Potential Hamiltonian with t_hop=1."
            )

        self._hamiltonian = hamiltonian
        # changing the prefactor to calculate vorticity instead
        self._c = 1.0  # -electron_mass * geometry.plaquette_area / 3
        plaquette_indices, _, _, _ = geometry.plaquettes
        self._rows = plaquette_indices[0]
        self._cols = plaquette_indices[1]

    def _compute_edge_forces(self, rho: B.Array, t: float) -> B.Array:
        r"""Compute bond currents along plaquette edges.

        Uses the gauge-invariant formula
        $f_{i<-j}(t) = 2\,\mathrm{Im}(dH_{ij}(t)_dt\,\rho_{ji}(t))$.
        When no Hamiltonian is stored (`t_{hop} = -1`), this reduces to
        $2\,\mathrm{Im}(0.0*\rho_{ij})$.
        """
        invalid_mask = (self._rows == -1) | (self._cols == -1)
        if self._hamiltonian is not None:
            dH_dt = self._hamiltonian.derivative_at_time(t)
            xp = B.xp()
            rows_flat = self._rows.ravel()
            cols_flat = self._cols.ravel()
            # Order reversed due to sign-change from -1.0 to +1.0 in self._c, which flips the current direction convention and thus the order of indices in the current formula. This is a bit subtle and could be made clearer by defining a helper function for the current that takes care of the index ordering and sign convention.
            # h_ij = xp.asarray(H_t[rows_flat, cols_flat]).reshape(self._rows.shape)
            dh_ij_dt = xp.asarray(dH_dt[cols_flat, rows_flat]).reshape(self._cols.shape)
            rho_values = rho[self._rows, self._cols]
            rho_values = B.xp().where(invalid_mask, 0.0, rho_values)
            # return 2.0 * xp.imag(h_ij * rho[self._cols, self._rows])
            return 2.0 * xp.imag(dh_ij_dt * rho_values)
        rho_values = rho[self._cols, self._rows]
        rho_values = B.xp().where(invalid_mask, 0.0, rho_values)
        # return 2.0 * xp.imag(h_ij * rho[self._rows, self._cols])
        return 2.0 * B.xp().imag(0.0 * rho_values)

    def _compute(self, rho: B.Array, t: float) -> B.Array:
        F_edges = self._compute_edge_forces(rho, t)
        return self._c * B.xp().sum(F_edges, axis=1)


class VortPolObservable(VorticityObservable):
    r"""Measures the vortex polarization

    $$\mu^{i}_{\omega}=\sum_{k}\frac{\Omega_{k}}{A_{k}}\sum_{(R,R')\in\partial P_{k}}J_{R\rightarrow R'}(t)
    |R'-R|\hat{n}^{k}_{i}(R^{k}+\alpha^{k}-R^{ref})$$
    \alpha^{k} is chosen such that R^{k}+\alpha^{k} points to the center of the plaquette k.
    1. WARNING: Currently hard-coded for 2D lattices, whose plaquettes have all their normal
                vectors pointing along one (z-)direction and all equal surface area.
    2. WARNING: Observable is well-defined only along the direction of the OBC.!
    """

    def __init__(
        self,
        geometry: HoneycombLatticeGeometry,
        window: MeasurementWindow | None = None,
        hamiltonian: Hamiltonian | None = None,
    ):

        super().__init__(geometry, window, hamiltonian)
        if not isinstance(geometry, Lattice2DGeometry):
            raise TypeError(
                "Currently supports only Lattice2DGeometry with all plaquettespointing in the same direction."
            )
        if geometry.pbc_x and geometry.pbc_y:
            raise ValueError(
                "Observable is well-defined only along the direction of the OBC. Please ensure that at least one of the directions has OBC."
            )
        _, plaquette_positions, _, _ = geometry.plaquettes
        origin = B.xp().mean(plaquette_positions, axis=0)
        # Put the origin of CF in the centre of the current vortices
        self._plaq_pos = plaquette_positions - origin
        # Total area of plaquettes of the system
        self._A = plaquette_positions.shape[0] * geometry.plaquette_area

    def _compute(self, rho: B.Array, t: float) -> B.Array:
        # Calculate vorticity in each plaquette
        vortices = self._c * self._compute_vortices(rho, t)
        # Multiply vortices with plaquette positions
        vort_times_r = self._plaq_pos * vortices[:, np.newaxis]

        return B.xp().sum(vort_times_r, axis=0) / self._A


class VectorVortFluxObservable(VortFluxObservable):
    r"""Measures the vector-valued unmodified vortex flow

    $$\Tilde{\mathbf{j}}^{z}_{\omega,P_{i}\rightarrow P_{j}}
    =1/2\sum_{i,j}\Bigl(\hat{\Tilde{j}}^{z}_{\omega,P_{i}\rightarrow P_{j}}(R_{j}-R_{i})\Bigr)$$
    WARNING: Currently hard-coded for 2D lattices, whose plaquettes have all their normal
             vectors pointing along one (z-)direction and all equal surface area.
    """

    def __init__(
        self,
        geometry: HoneycombLatticeGeometry,
        window: MeasurementWindow | None = None,
        hamiltonian: Hamiltonian | None = None,
    ):

        super().__init__(geometry, window, hamiltonian)
        assert isinstance(
            geometry, Lattice2DGeometry
        ), "Currently supports only Lattice2DGeometry with all plaquettes pointing in the same direction."
        # Take the position of each plaquette, difference in x & y between two nearest neighboring plaquettes
        _, plaquette_positions, plaq_xstep, plaq_ystep = geometry.plaquettes
        # Extract the length of the plaquette-system along the x-direction
        self._plaq_Lx = (
            B.xp().max(plaquette_positions[:, 0])
            - B.xp().min(plaquette_positions[:, 0])
            + plaq_xstep
        )
        # Extract the length of the plaquette-system along the y-direction
        self._plaq_Ly = (
            B.xp().max(plaquette_positions[:, 1])
            - B.xp().min(plaquette_positions[:, 1])
            + plaq_ystep
        )
        # Total area of plaquettes of the system
        self._A = plaquette_positions.shape[0] * geometry.plaquette_area
        # Construct x/y-distance and y-distance matrix
        x_coords = plaquette_positions[:, 0].reshape(-1, 1)
        y_coords = plaquette_positions[:, 1].reshape(-1, 1)
        # Subtract the x/y-coordinates as a (1, n_sites) row vector
        # array[i, j] = x/y_coords_row[j] - x/y_coords_col[i]
        self._x_dist_matrix = x_coords.T - x_coords
        self._y_dist_matrix = y_coords.T - y_coords
        if geometry.pbc_x:
            self._x_dist_matrix -= self._plaq_Lx * np.round(
                self._x_dist_matrix / self._plaq_Lx
            )
        if geometry.pbc_y:
            self._y_dist_matrix -= self._plaq_Ly * np.round(
                self._y_dist_matrix / self._plaq_Ly
            )

    def _compute_vector_vort_flux_tensor(
        self, rho: B.Array, t: float
    ) -> Tuple[B.Array, B.Array]:
        tilde_j_omega = self._compute_tilde_j_omega(rho, t)
        # Omega = 0.5 * (Lambda - Lambda.T)
        tilde_j_omega_x = 0.5 * tilde_j_omega * self._x_dist_matrix
        tilde_j_omega_y = 0.5 * tilde_j_omega * self._y_dist_matrix
        return tilde_j_omega_x, tilde_j_omega_y

    def _compute(self, rho: B.Array, t: float) -> B.Array:
        tilde_j_omega_x, tilde_j_omega_y = self._compute_vector_vort_flux_tensor(rho, t)
        tilde_j_omega_x, tilde_j_omega_y = (
            tilde_j_omega_x.ravel(),
            tilde_j_omega_y.ravel(),
        )
        vec_tilde_j_omega = B.xp().column_stack((tilde_j_omega_x, tilde_j_omega_y))
        return B.xp().sum(vec_tilde_j_omega, axis=0) / self._A


class VortSourcePolObservable(VortSourceObservable):
    r"""Measures the vortex source polarization

    $$\mu^{i}_{\omega,f}=\sum_{k}\frac{\Omega_{k}}{A_{k}}\sum_{(R,R')\in\partial P_{k}}f_{R\rightarrow R'}(t)
    |R'-R|\hat{n}^{k}_{i}(R^{k}+\alpha^{k}-R^{ref})$$
    \alpha^{k} is chosen such that R^{k}+\alpha^{k} points to the center of the plaquette k.
    1. WARNING: Currently hard-coded for 2D lattices, whose plaquettes have all their normal
                vectors pointing along one (z-)direction and all equal surface area.
    2. WARNING: Observable is well-defined only along the direction of the OBC!
    """

    def __init__(
        self,
        geometry: HoneycombLatticeGeometry,
        window: MeasurementWindow | None = None,
        hamiltonian: Hamiltonian | None = None,
    ):

        super().__init__(geometry, window, hamiltonian)
        if not isinstance(geometry, Lattice2DGeometry):
            raise TypeError(
                "Currently supports only Lattice2DGeometry with all plaquettespointing in the same direction."
            )
        if geometry.pbc_x and geometry.pbc_y:
            raise ValueError(
                "Observable is well-defined only along the direction of the OBC. Please ensure that at least one of the directions has OBC."
            )
        _, plaquette_positions, _, _ = geometry.plaquettes
        origin = B.xp().mean(plaquette_positions, axis=0)
        # Put the origin of CF in the centre of the current vortices
        self._plaq_pos = plaquette_positions - origin
        # Total area of plaquettes of the system
        self._A = plaquette_positions.shape[0] * geometry.plaquette_area

    def _compute(self, rho: B.Array, t: float) -> B.Array:
        # Calculate vortex source in each plaquette
        F_edges = self._compute_edge_forces(rho, t)
        vort_sources = self._c * B.xp().sum(F_edges, axis=1)
        # Multiply vortex sources with plaquette positions
        vort_sources_times_r = self._plaq_pos * vort_sources[:, np.newaxis]

        return B.xp().sum(vort_sources_times_r, axis=0) / self._A


class VectorVortFluxModObservable(VortFluxModObservable):
    r"""Measures the vector-valued modified vortex flow

    $$\mathbf{j}^{z}_{\omega,P_{i}\rightarrow P_{j}}
    =1/2\sum_{i,j}\Bigl(\hat{j}^{z}_{\omega,P_{i}\rightarrow P_{j}}(R_{j}-R_{i})\Bigr)$$
    WARNING: Currently hard-coded for 2D lattices, whose plaquettes have all their normal
             vectors pointing along one (z-)direction and all equal surface area.
    """

    def __init__(
        self,
        geometry: HoneycombLatticeGeometry,
        window: MeasurementWindow | None = None,
        hamiltonian: Hamiltonian | None = None,
    ):

        super().__init__(geometry, window, hamiltonian)
        assert isinstance(
            geometry, Lattice2DGeometry
        ), "Currently supports only Lattice2DGeometry with all plaquettes pointing in the same direction."
        # Take the position of each plaquette, difference in x & y between two nearest neighboring plaquettes
        _, plaquette_positions, plaq_xstep, plaq_ystep = geometry.plaquettes
        # Extract the length of the plaquette-system along the x-direction
        self._plaq_Lx = (
            B.xp().max(plaquette_positions[:, 0])
            - B.xp().min(plaquette_positions[:, 0])
            + plaq_xstep
        )
        # Extract the length of the plaquette-system along the y-direction
        self._plaq_Ly = (
            B.xp().max(plaquette_positions[:, 1])
            - B.xp().min(plaquette_positions[:, 1])
            + plaq_ystep
        )
        # Total area of plaquettes of the system
        self._A = plaquette_positions.shape[0] * geometry.plaquette_area
        # Construct x/y-distance and y-distance matrix
        x_coords = plaquette_positions[:, 0].reshape(-1, 1)
        y_coords = plaquette_positions[:, 1].reshape(-1, 1)
        # Subtract the x/y-coordinates as a (1, n_sites) row vector
        # array[i, j] = x/y_coords_row[j] - x/y_coords_col[i]
        self._x_dist_matrix = x_coords.T - x_coords
        self._y_dist_matrix = y_coords.T - y_coords
        if geometry.pbc_x:
            self._x_dist_matrix -= self._plaq_Lx * np.round(
                self._x_dist_matrix / self._plaq_Lx
            )
        if geometry.pbc_y:
            self._y_dist_matrix -= self._plaq_Ly * np.round(
                self._y_dist_matrix / self._plaq_Ly
            )

    def _compute_vector_vort_flux_tensor(
        self, rho: B.Array, t: float
    ) -> Tuple[B.Array, B.Array]:
        tilde_j_omega = self._compute_tilde_j_omega(rho, t)
        f_j_omega = self._compute_f_j_omega(rho, t)
        j_omega = tilde_j_omega + f_j_omega
        # Omega = 0.5 * (Lambda - Lambda.T)
        j_omega_x = 0.5 * j_omega * self._x_dist_matrix
        j_omega_y = 0.5 * j_omega * self._y_dist_matrix
        return j_omega_x, j_omega_y

    def _compute(self, rho: B.Array, t: float) -> B.Array:
        j_omega_x, j_omega_y = self._compute_vector_vort_flux_tensor(rho, t)
        j_omega_x, j_omega_y = (
            j_omega_x.ravel(),
            j_omega_y.ravel(),
        )
        vec_j_omega = B.xp().column_stack((j_omega_x, j_omega_y))
        return B.xp().sum(vec_j_omega, axis=0) / self._A


class SiteDensityObservable(Observable):
    """Measures the site-resolved electron density $n_i = \\rho_{ii}$."""

    def _compute(self, rho: B.Array, t: float) -> B.Array:
        return B.xp().real(B.xp().diag(rho))


class BondCurrentObservable(Observable):
    r"""Measures the bond currents $I_{i<-j}(t) = 2\,\mathrm{Im}(H_{ij}(t)\,\rho_{ji}(t))$.

    When no *hamiltonian* is provided the hopping is assumed real with
    $t_{\text{hop}} = -1$, reducing the expression to
    $2\,\mathrm{Im}(\rho_{ij})$.
    """

    def __init__(
        self,
        geometry: HoneycombLatticeGeometry,
        window: MeasurementWindow | None = None,
        hamiltonian: Hamiltonian | None = None,
    ):
        super().__init__(window)

        if hamiltonian is None:
            print(
                f"No Hamiltonian passed to {self.__class__}. Assuming Onsite-Potential Hamiltonian with t_hop=1."
            )

        self._hamiltonian = hamiltonian

        nn = geometry.nearest_neighbors
        self._nn_rows = B.xp().array(nn[:, 0], dtype=B.xp().int64)
        self._nn_cols = B.xp().array(nn[:, 1], dtype=B.xp().int64)
        assert len(self._nn_rows) == len(
            self._nn_cols
        ), "Nearest neighbor list should have shape (n_edges, 2)."

    def _compute(self, rho: B.Array, t: float) -> B.Array:
        if self._hamiltonian is not None:
            xp = B.xp()
            H_t = self._hamiltonian.at_time(t)
            h_ij = xp.asarray(H_t[self._nn_rows, self._nn_cols]).ravel()
            return 2.0 * xp.imag(h_ij * rho[self._nn_cols, self._nn_rows])  # (E,)
        return 2.0 * B.xp().imag(rho[self._nn_rows, self._nn_cols])  # (E,)


class BondCurrentForceObservable(BondCurrentObservable):
    r"""Measures the bond force $f_{i<-j}(t) = 2\,\mathrm{Im}(dH_{ij}_dt\,\rho_{ji}(t))$.

    When no *hamiltonian* is provided the hopping is assumed real and time-independent
    $t_{\text{hop}} = -1$, reducing the expression to $0.0$.
    """

    def __init__(
        self,
        geometry: HoneycombLatticeGeometry,
        window: MeasurementWindow | None = None,
        hamiltonian: Hamiltonian | None = None,
    ):
        super().__init__(geometry, window, hamiltonian)

    def _compute(self, rho: B.Array, t: float) -> B.Array:
        if self._hamiltonian is not None:
            xp = B.xp()
            dH_dt = self._hamiltonian.derivative_at_time(t)
            dh_ij_dt = xp.asarray(dH_dt[self._nn_rows, self._nn_cols]).ravel()
            return 2.0 * xp.imag(dh_ij_dt * rho[self._nn_cols, self._nn_rows])  # (E,)
        return 2.0 * B.xp().imag(0.0 * rho[self._nn_rows, self._nn_cols])  # (E,)


class BondCurrentFluxObservable(BondCurrentObservable):
    r"""Measures the bond current flux
    $\Pi^{i<-j}_{k} = 2\,\mathrm{Re}(H_{ij}H_{jk}\,\rho_{ki}-H_{ij}H_{ki}\,\rho_{jk})$.

    When no *hamiltonian* is provided the hopping is assumed real with
    $t_{\text{hop}} = -1$, reducing the expression to
    $2\,\mathrm{Re}(\rho_{ki}-\rho_{jk})$.
    """

    def __init__(
        self,
        geometry: HoneycombLatticeGeometry,
        window: MeasurementWindow | None = None,
        hamiltonian: Hamiltonian | None = None,
    ):
        super().__init__(geometry, window, hamiltonian)
        size, len_rows = geometry.Lx * geometry.Ly, len(self._nn_rows)
        self._nn_rows_repeated = B.xp().repeat(self._nn_rows, size)
        self._nn_cols_repeated = B.xp().repeat(self._nn_cols, size)
        self._X = B.xp().array(list(range(size)) * len_rows, dtype=B.xp().int64)

    def _compute(self, rho: B.Array, t: float) -> B.Array:
        if self._hamiltonian is not None:
            xp = B.xp()
            H_t = self._hamiltonian.at_time(t)
            h_ij = xp.asarray(
                H_t[self._nn_rows_repeated, self._nn_cols_repeated]
            ).ravel()
            h_jk = xp.asarray(H_t[self._nn_cols_repeated, self._X]).ravel()
            h_ki = xp.asarray(H_t[self._X, self._nn_rows_repeated]).ravel()
            return 2.0 * xp.real(
                rho[self._X, self._nn_rows_repeated] * h_ij * h_jk
                - h_ki * h_ij * rho[self._nn_cols_repeated, self._X]
            )  # (E*V,)
        return 2.0 * B.xp().real(
            rho[self._nn_cols_repeated, self._X] - rho[self._X, self._nn_rows_repeated]
        )  # (E*V,)


class LatticeFrameObservable(Observable):
    """Composite observable that records site densities, bond currents, and
    curret vortices at each measurement step."""

    def __init__(
        self,
        geometry: HoneycombLatticeGeometry,
        window: MeasurementWindow | None = None,
        hamiltonian: Hamiltonian | None = None,
    ) -> None:
        super().__init__(window)

        self.geometry = geometry

        self.density_obs = SiteDensityObservable(window)
        self.current_obs = BondCurrentObservable(geometry, window, hamiltonian)
        self.current_vort_obs = VorticityObservable(geometry, window, hamiltonian)

        # plaquette_anchors no longer needed, since HoneycombLatticeGeometry is equipped with plaquette_positions

    def _compute(self, rho: B.Array, t: float) -> B.Array:
        # compute handled by sub-observables via measure()
        raise NotImplementedError

    # override the ABC measure to call the sub-observables' measure methods (that call their _compute methods)
    def measure(self, rho: B.Array, t: float, step_index: int) -> None:
        self.density_obs.measure(rho, t, step_index)
        self.current_obs.measure(rho, t, step_index)
        self.current_vort_obs.measure(rho, t, step_index)

    def finalize(self) -> None:
        self.density_obs.finalize()
        self.current_obs.finalize()
        self.current_vort_obs.finalize()

        self.values = {
            "densities": cast(B.FCPUArray, self.density_obs.values),
            "currents": cast(B.FCPUArray, self.current_obs.values),
            "current_vorts": cast(B.FCPUArray, self.current_vort_obs.values),
        }

        self.measurement_times = self.density_obs.measurement_times
