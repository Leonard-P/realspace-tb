from ..observable import Observable, MeasurementWindow
from .lattice_2d_geometry import Lattice2DGeometry
from .honeycomb_geometry import HoneycombLatticeGeometry
from ..hamiltonian import Hamiltonian
from .. import backend as B
from .units import effective_electron_mass
from typing import cast, Tuple
import numpy as np


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
        if self._hamiltonian is not None:
            H_t = self._hamiltonian.at_time(t)
            xp = B.xp()
            rows_flat = self._rows.ravel()
            cols_flat = self._cols.ravel()
            # Order reversed due to sign-change from -1.0 to +1.0 in self._c, which flips the current direction convention and thus the order of indices in the current formula. This is a bit subtle and could be made clearer by defining a helper function for the current that takes care of the index ordering and sign convention.
            # h_ij = xp.asarray(H_t[rows_flat, cols_flat]).reshape(self._rows.shape)
            h_ij = xp.asarray(H_t[cols_flat, rows_flat]).reshape(self._cols.shape)
            # return 2.0 * xp.imag(h_ij * rho[self._cols, self._rows])
            return 2.0 * xp.imag(h_ij * rho[self._rows, self._cols])
        # return 2.0 * xp.imag(h_ij * rho[self._rows, self._cols])
        return 2.0 * B.xp().imag(rho[self._cols, self._rows])

    def _compute_vortices(self, rho: B.Array, t: float) -> B.Array:
        I_edges = self._compute_edge_currents(rho, t)
        return B.xp().sum(I_edges, axis=1)  # (n_cells,)

    def _compute(self, rho: B.Array, t: float) -> B.Array:
        return self._c * self._compute_vortices(rho, t)


class VortPolObservable(VorticityObservable):
    r"""Measures the vortex polarization

    $$\mu^{i}_{\omega}=\sum_{k}\frac{\Omega_{k}}{A_{k}}\sum_{(R,R')\in\partial P_{k}}J_{R\rightarrow R'}(t)
    |R'-R|\hat{n}^{k}_{i}(R^{k}+\alpha^{k}-R^{ref})$$
    \alpha^{k} is chosen such that R^{k}+\alpha^{k} points to the center of the plaquette k.
    1. WARNING: Observable is well-defined only along the direction of the OBC!
    2. WARNING: Currently hard-coded for 2D lattices, whose plaquettes have all their normal
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


class VortFluxObservable(VorticityObservable):
    r"""Measures the unmodified vorticity flux

    $$\hat{\Omega}_{P_{i}\rightarrow P_{j}}
    =\frac{1}{2}\Bigl(\hat{\Lambda}_{P_{i}\rightarrow P_{j}}-\hat{\Lambda}_{P_{j}\rightarrow P_{i}}\Bigr)$$
    This observable is already ANTISYMMETRIC
    """

    def __init__(
        self,
        geometry: HoneycombLatticeGeometry,
        window: MeasurementWindow | None = None,
        hamiltonian: Hamiltonian | None = None,
    ):
        super().__init__(geometry, window, hamiltonian)
        xp = B.xp()
        self._num_plaquettes, self._num_sites_plaquette = self._rows.shape

        dX = geometry.site_plaquette_count
        self._x = xp.tile(
            self._rows, (1, self._num_sites_plaquette * self._num_plaquettes)
        ).ravel()
        self._invdX = 1.0 / dX[self._x]
        self._rows_repeated = xp.tile(
            xp.repeat(self._rows, self._num_sites_plaquette, axis=1),
            (self._num_plaquettes, 1),
        )
        self._cols_repeated = xp.tile(
            xp.repeat(self._cols, self._num_sites_plaquette, axis=1),
            (self._num_plaquettes, 1),
        )

    def _compute_edge_fluxes_per_site(self, rho: B.Array, t: float) -> B.Array:
        r"""Measures current flux tensor for each site X

        $$\frac{1}{d_{X}}\sum_{(R,R')\in\partial P_{i}}\Pi^{R \rightarrow R'}_{X}$$
        """
        rows_repeated_flat = self._rows_repeated.ravel()
        cols_repeated_flat = self._cols_repeated.ravel()
        xp = B.xp()
        rho_ki = xp.asarray(rho[self._x, cols_repeated_flat])
        rho_jk = xp.asarray(rho[rows_repeated_flat, self._x])
        if self._hamiltonian is not None:
            H_t = self._hamiltonian.at_time(t)
            # self._x has been vectorized
            h_ij = xp.asarray(H_t[cols_repeated_flat, rows_repeated_flat])
            h_jk = xp.asarray(H_t[rows_repeated_flat, self._x])
            h_ki = xp.asarray(H_t[self._x, cols_repeated_flat])
            Pi_sites = 2 * (
                self._invdX * xp.real(h_ij * (h_jk * rho_ki - rho_jk * h_ki))
            )
            return Pi_sites.reshape(
                (
                    self._num_plaquettes ** (2),
                    self._num_sites_plaquette,
                    self._num_sites_plaquette,
                )
            )
        Pi_sites = 2 * (self._invdX * xp.real(rho_jk - rho_ki))
        return Pi_sites.reshape(
            (
                self._num_plaquettes ** (2),
                self._num_sites_plaquette,
                self._num_sites_plaquette,
            )
        )

    def _compute_edge_fluxes_per_plaquette(self, rho: B.Array, t: float) -> B.Array:
        Pi_sites = self._compute_edge_fluxes_per_site(rho, t)
        return B.xp().sum(Pi_sites, axis=2)

    def _compute_vort_fluxes(self, rho: B.Array, t: float) -> B.Array:
        Pi = self._compute_edge_fluxes_per_plaquette(rho, t)
        return (
            self._c
            * (
                (B.xp().sum(Pi, axis=1)).reshape(
                    self._num_plaquettes, self._num_plaquettes
                )
            ).T
        )

    def _compute(self, rho: B.Array, t: float) -> B.Array:
        Lambda = self._compute_vort_fluxes(rho, t)
        return 0.5 * (Lambda - Lambda.T)


class VectorVortFluxObservable(VortFluxObservable):
    r"""Measures the vector-valued unmodified vorticity flux

    $$\hat{\Omega}_{P_{i}\rightarrow P_{j}}
    =\frac{1}{2}\Bigl(\hat{\Lambda}_{P_{i}\rightarrow P_{j}}
    -\hat{\Lambda}_{P_{j}\rightarrow P_{i}}\Bigr)(R_{j}-R_{i})$$
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
        # Extract the length of the plaquette-system along the x-direction
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
        Lambda = self._compute_vort_fluxes(rho, t)
        Omega = 0.5 * (Lambda - Lambda.T)
        Omega_x = 0.5 * Omega * self._x_dist_matrix
        Omega_y = 0.5 * Omega * self._y_dist_matrix
        return Omega_x, Omega_y

    def _compute(self, rho: B.Array, t: float) -> B.Array:
        Omega_x, Omega_y = self._compute_vector_vort_flux_tensor(rho, t)
        Omega_x, Omega_y = Omega_x.ravel(), Omega_y.ravel()
        Vec_Omega = B.xp().column_stack((Omega_x, Omega_y))
        return B.xp().sum(Vec_Omega, axis=0) / self._A


class VortSourceObservable(VortFluxObservable):
    r"""Measures the vorticity source

    $$\hat{\Omega}_{f,P_{i}}=-\frac{1}{2}\sum_{P}\Bigl(\hat{\Lambda}_{P_{i}\rightarrow P}+\hat{\Lambda}_{P\rightarrow P_{i}}\Bigr)
    +\sum_{(\mathbf{R},\mathbf{R}')\in\partial P_{i}}\hat{f}_{\mathbf{R}\rightarrow\mathbf{R}'}$$
    This observable already captures the SYMMETRIC part of the old vorticity flux
    """

    def _compute_edge_forces(self, rho: B.Array, t: float) -> B.Array:
        r"""Compute bond currents along plaquette edges.

        Uses the gauge-invariant formula
        $f_{i<-j}(t) = 2\,\mathrm{Im}(dH_{ij}(t)_dt\,\rho_{ji}(t))$.
        When no Hamiltonian is stored (`t_{hop} = -1`), this reduces to
        $2\,\mathrm{Im}(0.0*\rho_{ij})$.
        """
        if self._hamiltonian is not None:
            dH_dt = self._hamiltonian.derivative_at_time(t)
            xp = B.xp()
            rows_flat = self._rows.ravel()
            cols_flat = self._cols.ravel()
            # Order reversed due to sign-change from -1.0 to +1.0 in self._c, which flips the current direction convention and thus the order of indices in the current formula. This is a bit subtle and could be made clearer by defining a helper function for the current that takes care of the index ordering and sign convention.
            # h_ij = xp.asarray(H_t[rows_flat, cols_flat]).reshape(self._rows.shape)
            dh_ij_dt = xp.asarray(dH_dt[cols_flat, rows_flat]).reshape(self._cols.shape)
            # return 2.0 * xp.imag(h_ij * rho[self._cols, self._rows])
            return 2.0 * xp.imag(dh_ij_dt * rho[self._rows, self._cols])
        # return 2.0 * xp.imag(h_ij * rho[self._rows, self._cols])
        return 2.0 * B.xp().imag(0.0 * rho[self._cols, self._rows])

    def _compute(self, rho: B.Array, t: float) -> B.Array:
        Lambda = self._compute_vort_fluxes(rho, t)
        F_edges = self._compute_edge_forces(rho, t)
        return -0.5 * B.xp().sum(Lambda + Lambda.T, axis=1) + self._c * B.xp().sum(
            F_edges, axis=1
        )  # (n_cells,)


class VortFluxModObservable(VortSourceObservable):
    r"""Measures the modified vorticity flux

    $$\Omega_{P_{i}\rightarrow P}=\sum_{X\in\partial P}
    \frac{1}{d_{X}}\sum_{(R,R')\in\partial P_{i}}\hat{\Pi}^{R\rightarrow R'}_{X}
    -\delta_{P,P_{i}}\Omega_{f,P_{i}}$$
    This observable is NEITHER symmetric nor antisymmetric.
    """

    def _compute(self, rho: B.Array, t: float) -> B.Array:
        F_edges = self._compute_edge_forces(rho, t)
        Pi = self._compute_edge_fluxes_per_plaquette(rho, t)
        return (
            self._c
            * (
                (B.xp().sum(Pi, axis=1)).reshape(
                    self._num_plaquettes, self._num_plaquettes
                )
            ).T
        ) - B.xp().diag(self._c * B.xp().sum(F_edges, axis=1))


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
        return 2.0 * xp.real(
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

        self.plaquette_anchor_indices = geometry.plaquette_anchors

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
