"""The Bloch extension relies only on the Lattice2DGeometry interface, not on the honeycomb lattice.
A minimal square lattice (both ribbon orientations) is checked against its real-space ring, and a
Hamiltonian with beyond-nearest-neighbour hopping is rejected."""
import numpy as np
import pytest
import scipy.sparse as sp

import realspace_tb as tb
import realspace_tb.orbitronics_2d as orb
from realspace_tb.orbitronics_2d.lattice_2d_geometry import Lattice2DGeometry


class SquareLattice(Lattice2DGeometry):
    """Square lattice with spacing a and optional periodicity (short bond vectors on wrapped bonds)."""

    def __init__(self, Lx, Ly, a=1.0, pbc_x=False, pbc_y=False):
        super().__init__()
        self.Lx, self.Ly, self.a, self.pbc_x, self.pbc_y = Lx, Ly, a, pbc_x, pbc_y
        nn, bv = [], []
        for r in range(Ly):
            for c in range(Lx):
                i = r * Lx + c
                if c + 1 < Lx or pbc_x:
                    nn.append((i, r * Lx + (c + 1) % Lx)); bv.append((a, 0.0))
                if r + 1 < Ly or pbc_y:
                    nn.append((i, ((r + 1) % Ly) * Lx + c)); bv.append((0.0, a))
        self._nn, self._bv = np.array(nn), np.array(bv, dtype=float)

    def index_to_position(self, i):
        return np.array([(i % self.Lx) * self.a, (i // self.Lx) * self.a])

    @property
    def nearest_neighbors(self):
        return self._nn

    @property
    def nn_bond_vectors(self):
        return self._bv

    def bravais_site_indices(self):
        return np.arange(self.Lx * self.Ly)

    def replicated(self, n_x=1, n_y=1):
        return SquareLattice(self.Lx * n_x, self.Ly * n_y, self.a, self.pbc_x, self.pbc_y)


FIELD = orb.RampedACFieldAmplitude(E0=1e-3, omega=0.1, T_ramp=20, direction=np.array([1.0, 1.0]))


@pytest.mark.parametrize("cell, ring", [
    (dict(Lx=9, Ly=3, pbc_y=True), dict(Lx=9, Ly=15, pbc_y=True)),    # periodic in y
    (dict(Lx=3, Ly=9, pbc_x=True), dict(Lx=15, Ly=9, pbc_x=True)),    # periodic in x
])
def test_square_lattice_matches_real_space_ring(cell, ring):
    H_k = tb.BlochHamiltonian(
        orb.LinearFieldHamiltonianPeierls(SquareLattice(**cell), -1.0, [FIELD], B0=0.05),
        tb.KMesh(5, shift=0.0),
    )
    H_r = orb.LinearFieldHamiltonianPeierls(SquareLattice(**ring), -1.0, [FIELD], B0=0.05)
    rho_k, rho_r = H_k.thermal_state_density_matrix(), H_r.thermal_state_density_matrix()
    dt, T = 0.1, 20.0  # RK4 needs dt * ||H|| small enough (||H|| ~ 4 here)
    tb.BlochRK4NeumannSolver().evolve(rho_k, H_k, dt, T, progress=False)
    tb.RK4NeumannSolver().evolve(rho_r, H_r, dt, T, progress=False)
    t = int(T / dt) * dt
    J_r = orb.observables.BondCurrentObservable(H_r.geometry, hamiltonian=H_r)._compute(rho_r, t)
    assert np.allclose(H_k.real_space_density_matrix(rho_k), rho_r, atol=1e-12)
    assert np.allclose(H_k.real_space_bond_currents(rho_k, t), J_r, atol=1e-12)


def test_beyond_nearest_neighbour_hopping_is_rejected():
    geo = orb.HoneycombLatticeGeometry(Lx=17, Ly=2, a_nn=2.6834, pbc_y=True)
    base = orb.LinearFieldHamiltonianPeierls(geo, -0.1, [], B0=0.0)
    P = geo.site_positions
    d = np.linalg.norm(P[:, None] - P[None], axis=-1)
    nnn = sp.csr_matrix(0.01 * (np.abs(d - np.sqrt(3) * 2.6834) < 1e-6))

    class WithNNN(tb.Hamiltonian):
        geometry = geo
        def at_time(self, t):
            return base.at_time(t) + nnn
        def derivative_at_time(self, t):
            return base.derivative_at_time(t)

    with pytest.raises(ValueError, match="not nearest neighbours"):
        tb.BlochHamiltonian(WithNNN(), tb.KMesh(4)).thermal_state_density_matrix()
