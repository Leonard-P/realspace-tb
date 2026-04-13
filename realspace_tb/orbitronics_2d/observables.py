from ..observable import Observable, MeasurementWindow
from .honeycomb_geometry import HoneycombLatticeGeometry
from ..hamiltonian import Hamiltonian
from .. import backend as B
from .units import effective_electron_mass
from typing import cast
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
        plaquette_indices, _ = geometry.plaquettes
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

    def _compute(self, rho: B.Array, t: float) -> B.Array:
        I_edges = self._compute_edge_currents(rho, t)
        return self._c * B.xp().sum(I_edges, axis=1)  # (n_cells,)


class VortFluxObservable(VorticityObservable):
    r"""Measures the unmodified vorticity flux

    $$\hat{\Omega}_{P_{i}\rightarrow P}
    =\frac{1}{2}\Bigl(\hat{\Lambda}_{P_{i}\rightarrow P}-\hat{\Lambda}_{P\rightarrow P_{i}}\Bigr)$$
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


class OrbitalPolarizationObservable(VorticityObservable):
    r"""Measures the orbital polarization using loop currents around cells as

    $$\expval{P_{orb}} = -i\frac{m_e}{A_\mathrm{tot}} \sum_\alpha\sum_{(k,l)\in\circlearrowleft_{\vec R_\alpha}} (\sqrt 3\,\vec R_\alpha +\frac{5}{12}\begin{pmatrix}0&-1\\1&0\end{pmatrix} (\vec r_l - \vec r_k)) \Im \rho_{kl}$$
    """

    def __init__(
        self,
        geometry: HoneycombLatticeGeometry,
        electron_mass: float | None = None,
        window: MeasurementWindow | None = None,
        hamiltonian: Hamiltonian | None = None,
    ):
        if electron_mass is None:
            electron_mass = effective_electron_mass()

        super().__init__(geometry, window, hamiltonian)

        self._origin = B.xp().array(geometry.origin)
        self._m = electron_mass
        self._c1 = -self._m * (B.xp().sqrt(3.0) / 2.0)
        self._c2 = -self._m * (5.0 / 24.0)
        _, plaquette_positions = geometry.plaquettes
        self._A = plaquette_positions.shape[0] * geometry.plaquette_area

        site_positions = B.xp().array(
            [geometry.index_to_position(i) for i in range(geometry.Lx * geometry.Ly)],
            dtype=B.FDTYPE,
        )

        # Edge vectors r_l - r_k per (cell, edge)
        self._edge_vecs = (
            site_positions[self._cols] - site_positions[self._rows]
        )  # (n_cells, L, 2)
        # Their 90° rotation R @ (r_l - r_k) with R @ v = [-v_y, v_x]
        self._rot_edge_vecs = B.xp().stack(
            (-self._edge_vecs[..., 1], self._edge_vecs[..., 0]), axis=-1
        )  # (n_cells, L, 2)

        # per-plaquette positions
        self._plaquette_positions = plaquette_positions

    def _compute(self, rho: B.Array, t: float) -> B.Array:
        # Bond currents along the oriented loop edges for each cell
        I_edges = self._compute_edge_currents(rho, t)

        # R_alpha * sum_edges I_edge per cell
        curl_per_cell = B.xp().sum(I_edges, axis=1)  # (n_cells,)
        centered = self._plaquette_positions - self._origin  # (n_cells, 2)
        term1_vec = centered.T @ curl_per_cell  # (2,)

        # Term 2: sum_edges [ R @ (r_l - r_k) * I_edge ] over cells
        weighted_rot_edges = (I_edges[..., None] * self._rot_edge_vecs).sum(
            axis=1
        )  # (n_cells, 2)
        term2_vec = B.xp().sum(weighted_rot_edges, axis=0)  # (2,)

        return (self._c1 * term1_vec + self._c2 * term2_vec) / self._A


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
