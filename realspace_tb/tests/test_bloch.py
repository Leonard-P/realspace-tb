"""Consistency tests for the k-space (Bloch) implementation.

Run from the folder that contains `realspace_tb`:   python -m pytest realspace_tb/tests -q
"""
import copy
import numpy as np
import pytest

import realspace_tb as tb
import realspace_tb.orbitronics_2d as orb
from realspace_tb.bloch_solver import _MeasureContext

SCALE = 0.0992
A_NN = 2.6834
T_HOP = -SCALE
B0 = -5e-1 * SCALE


def field(direction=(1.0, 0.0), E0=1e-3 * SCALE):
    return orb.RampedACFieldAmplitude(E0=E0, omega=1.11 * SCALE, T_ramp=20 / SCALE,
                                      direction=np.array(direction))


def model(Ly, direction=(1.0, 0.0), gauge_origin=None, Lx=17, B=B0):
    geo = orb.HoneycombLatticeGeometry(Lx=Lx, Ly=Ly, a_nn=A_NN, pbc_x=False, pbc_y=True)
    return orb.LinearFieldHamiltonianPeierls(geo, T_HOP, [field(direction)], B0=B,
                                             gauge_origin=gauge_origin)


def bond_map(geo_small, geo_big):
    """Index of every bond of the small cell in the big (real-space) cell's bond list."""
    def key(geo, b):
        i, j = geo.nearest_neighbors[b]
        return tuple(np.round(np.r_[geo.site_positions[i], geo.nn_bond_vectors[b]], 6))
    big = {key(geo_big, b): b for b in range(len(geo_big.nearest_neighbors))}
    return np.array([big[key(geo_small, b)] for b in range(len(geo_small.nearest_neighbors))])


def edge_currents(geo, J):
    """Net y-current carried by the bonds of the leftmost / rightmost column."""
    nn, bv, x = geo.nearest_neighbors, geo.nn_bond_vectors, geo.site_positions[:, 0]
    vertical = np.abs(bv[:, 0]) < 1e-9
    xb = x[nn[:, 0]]
    left = vertical & np.isclose(xb, x.min())
    right = vertical & np.isclose(xb, x.max())
    s = np.sign(bv[:, 1])
    return np.sum((J * s)[left]), np.sum((J * s)[right])


def test_gamma_only_equals_real_space_ring():
    H = model(10)
    rho_rs = H.thermal_state_density_matrix(0.0, np.inf)
    Hb = tb.BlochHamiltonian(model(10), tb.KMesh(1, shift=0.0))
    rho_b = Hb.thermal_state_density_matrix(0.0, np.inf)
    assert np.allclose(rho_b[:, 0, :], rho_rs, atol=1e-12)


def test_primitive_cell_with_5_k_equals_ring_of_5_cells():
    Hbig = model(10)
    obs_big = orb.observables.BondCurrentObservable(Hbig.geometry, hamiltonian=Hbig)
    J_big = obs_big._compute(Hbig.thermal_state_density_matrix(0.0, np.inf), 0.0)
    Hb = tb.BlochHamiltonian(model(2), tb.KMesh(5, shift=0.0))
    obs = orb.observables.BondCurrentObservable(Hb.geometry, hamiltonian=Hb)
    J = _MeasureContext(Hb, Hb.thermal_state_density_matrix(0.0, np.inf), 0.0).value(obs)
    assert np.allclose(J, J_big[bond_map(Hb.geometry, Hbig.geometry)], atol=1e-12)


@pytest.mark.parametrize("direction", [(1.0, 0.0), (0.0, 1.0)])
def test_dynamics_match_real_space_ring(direction):
    """Field along x (open) and along y (periodic): primitive cell + 5 k == 10-row ring."""
    dt, T = 0.05 / SCALE, 40 / SCALE
    Hbig = model(10, direction)
    rho = Hbig.thermal_state_density_matrix(0.0, np.inf)
    ob = orb.observables.BondCurrentObservable(Hbig.geometry, hamiltonian=Hbig)
    tb.RK4NeumannSolver().evolve(rho, Hbig, dt, T, observables=[ob], progress=False)

    Hb = tb.BlochHamiltonian(model(2, direction), tb.KMesh(5, shift=0.0))
    rho_b = Hb.thermal_state_density_matrix(0.0, np.inf)
    o = orb.observables.BondCurrentObservable(Hb.geometry, hamiltonian=Hb)
    tb.BlochRK4NeumannSolver().evolve(rho_b, Hb, dt, T, observables=[o], progress=False)
    m = bond_map(Hb.geometry, Hbig.geometry)
    assert np.allclose(o.values, ob.values[:, m], atol=1e-10)


def test_pairwise_shortcut_equals_k_resolved_and_flux_observable():
    """Gamma-only Bloch on the full ring reproduces every real-space observable,
    including the k-resolved path (VortFlux uses H @ rho)."""
    dt, T = 0.05 / SCALE, 10 / SCALE
    Hrs = model(10)
    rho = Hrs.thermal_state_density_matrix(0.0, np.inf)
    mk = lambda H: [orb.observables.VorticityObservable(H.geometry, hamiltonian=H),
                    orb.observables.VortFluxObservable(H.geometry, hamiltonian=H),
                    orb.observables.LatticeFrameObservable(H.geometry, hamiltonian=H)]
    obs_rs = mk(Hrs)
    tb.RK4NeumannSolver().evolve(rho, Hrs, dt, T, observables=obs_rs, progress=False)
    Hb = tb.BlochHamiltonian(model(10), tb.KMesh(1, shift=0.0))
    rho_b = Hb.thermal_state_density_matrix(0.0, np.inf)
    obs_b = mk(Hb)
    tb.BlochRK4NeumannSolver().evolve(rho_b, Hb, dt, T, observables=obs_b, progress=False)
    for a, b in zip(obs_rs[:2], obs_b[:2]):
        assert np.allclose(a.values, b.values, atol=1e-10)
    for key in ("densities", "currents", "current_vorts"):
        assert np.allclose(obs_rs[2].values[key], obs_b[2].values[key], atol=1e-10)

    # pairwise shortcut == explicit k-by-k sum on a non-trivial mesh
    Hb = tb.BlochHamiltonian(model(2), tb.KMesh(7))
    rho_b = Hb.thermal_state_density_matrix(0.0, np.inf)
    o = orb.observables.VorticityObservable(Hb.geometry, hamiltonian=Hb)
    ctx = _MeasureContext(Hb, rho_b, 0.0)
    fast = ctx.value(o)
    type(o).bloch_pairwise = False
    try:
        slow = ctx.value(o)
    finally:
        type(o).bloch_pairwise = True
    assert np.allclose(fast, slow, atol=1e-13)


def test_parallel_equals_serial():
    dt, T = 0.05 / SCALE, 10 / SCALE
    Hb = tb.BlochHamiltonian(model(2), tb.KMesh(12))
    rho1 = Hb.thermal_state_density_matrix(0.0, np.inf)
    rho2 = rho1.copy()
    mk = lambda: [orb.observables.BondCurrentObservable(Hb.geometry, hamiltonian=Hb),
                  orb.observables.VortFluxObservable(Hb.geometry, hamiltonian=Hb.base)]
    o1, o2 = mk(), mk()
    tb.BlochRK4NeumannSolver().evolve(rho1, Hb, dt, T, observables=o1, progress=False)
    tb.BlochRK4NeumannSolver().evolve(rho2, Hb, dt, T, observables=o2, progress=False, n_workers=3)
    assert np.allclose(rho1, rho2, atol=1e-12)
    for a, b in zip(o1, o2):
        assert np.allclose(a.values, b.values, atol=1e-12)


def test_converged_edge_currents_symmetric_and_gauge_independent():
    """x0 at the centre: symmetric at every N_k. x0 at the left edge: symmetric only once
    converged (here T = 0 with near-zero modes, so convergence is power-law in N_k)."""
    res = {}
    for origin in ("zero", "center"):
        Hb = tb.BlochHamiltonian(model(2, gauge_origin=origin), tb.KMesh(1024))
        o = orb.observables.BondCurrentObservable(Hb.geometry, hamiltonian=Hb)
        J = _MeasureContext(Hb, Hb.thermal_state_density_matrix(0.0, np.inf), 0.0).value(o)
        res[origin] = J
        left, right = edge_currents(Hb.geometry, J)
        assert np.isclose(left, -right, rtol=1e-12 if origin == "center" else 1e-6)
    assert np.max(np.abs(res["zero"] - res["center"])) < 1e-5 * np.max(np.abs(res["center"]))


def test_ring_with_left_origin_is_asymmetric():
    """The original observation: a Gamma-only ring with x0 at the left edge."""
    H = model(10, gauge_origin="zero")
    o = orb.observables.BondCurrentObservable(H.geometry, hamiltonian=H)
    left, right = edge_currents(H.geometry, o._compute(H.thermal_state_density_matrix(0.0, np.inf), 0.0))
    assert not np.isclose(left, -right, rtol=1e-3)


def test_half_occupation_at_zero_modes():
    Hb = tb.BlochHamiltonian(model(2, B=0.0), tb.KMesh(4, shift=0.0))  # metallic width, k=0 included
    f = tb.fermi_occupation(Hb.eigenvalues, 0.0, np.inf, matrix_size=Hb.n_sites)
    zero = np.abs(Hb.eigenvalues) < 1e-9
    assert zero.any() and np.all(f[zero] == 0.5)
    rho = Hb.thermal_state_density_matrix(0.0, np.inf)
    assert np.allclose(np.einsum("aka->ka", rho).real, 0.5, atol=1e-12)  # half filling, every site


@pytest.mark.parametrize("pbc_y, Ly", [(False, 10), (True, 10), (True, 2)])
def test_uniform_flux_per_hexagon(pbc_y, Ly):
    """Peierls phases give the same flux B * A_hex through every complete hexagon, in both
    gauges (open flake: A = -B y e_x; ring: A = B x e_y), including across the boundary."""
    geo = orb.HoneycombLatticeGeometry(Lx=17, Ly=Ly, a_nn=A_NN, pbc_y=pbc_y)
    H = orb.LinearFieldHamiltonianPeierls(geo, T_HOP, [], B0=B0).at_time(0.0).toarray()
    rows, cols = geo.plaquettes[0]
    full = np.all(rows != -1, axis=1)
    loop = [np.angle(np.prod(H[c, r] / T_HOP)) for r, c in zip(rows[full], cols[full])]
    assert np.allclose(np.abs(loop), abs(B0 * geo.plaquette_area), atol=1e-12)
    assert np.allclose(loop, loop[0], atol=1e-12)


# ---------------------------------------------------------------- both ribbon orientations
ORIENTATIONS = {
    # armchair ribbon: open in x, periodic in y; cell = 2 rows, ring = 10 rows
    "armchair": (dict(Lx=17, Ly=2, pbc_x=False, pbc_y=True), dict(Lx=17, Ly=10, pbc_x=False, pbc_y=True)),
    # zigzag ribbon: periodic in x, open in y; cell = 4 columns (2 periods), ring = 20 columns
    "zigzag": (dict(Lx=4, Ly=10, pbc_x=True, pbc_y=False), dict(Lx=20, Ly=10, pbc_x=True, pbc_y=False)),
}


def peierls(geo_kwargs, direction):
    geo = orb.HoneycombLatticeGeometry(a_nn=A_NN, **geo_kwargs)
    return orb.LinearFieldHamiltonianPeierls(geo, T_HOP, [field(direction)], B0=B0)


@pytest.mark.parametrize("orientation", list(ORIENTATIONS))
@pytest.mark.parametrize("direction", [(1.0, 0.0), (0.0, 1.0)])
def test_cell_with_gamma_mesh_reproduces_ring_elementwise(orientation, direction):
    """Cell + KMesh(5, shift=0) == the user's own ring of 5 cells: geometry, every element of
    H(t), dH/dt and rho(t), and every bond current, before and after time evolution."""
    cell, ring = ORIENTATIONS[orientation]
    dt, T = 0.05 / SCALE, 20 / SCALE
    H_r = peierls(ring, direction)
    rho_r = H_r.thermal_state_density_matrix(0.0, np.inf)
    obs_r = orb.observables.BondCurrentObservable(H_r.geometry, hamiltonian=H_r)
    H_k = tb.BlochHamiltonian(peierls(cell, direction), tb.KMesh(5, shift=0.0))  # axis inferred
    rho_k = H_k.thermal_state_density_matrix(0.0, np.inf)
    geo = H_k.ring_geometry  # built from the mesh, no user input
    assert (geo.Lx, geo.Ly, geo.pbc_x, geo.pbc_y) == (ring["Lx"], ring["Ly"], ring.get("pbc_x", False), ring.get("pbc_y", False))
    assert np.allclose(geo.site_positions, H_r.geometry.site_positions)
    assert np.array_equal(geo.nearest_neighbors, H_r.geometry.nearest_neighbors)
    assert np.allclose(np.sort(H_k.eigenvalues.ravel()), np.sort(H_r.eigenvalues), atol=1e-13)
    H_ring = H_k.to_real_space()
    for t in (0.0, None):
        if t is None:
            tb.RK4NeumannSolver().evolve(rho_r, H_r, dt, T, progress=False)
            tb.BlochRK4NeumannSolver().evolve(rho_k, H_k, dt, T, progress=False)
            t = int(T / dt) * dt
        assert np.allclose(H_k.real_space_hamiltonian(t), H_r.at_time(t).toarray(), atol=1e-14)
        assert np.allclose(H_ring.at_time(t).toarray(), H_r.at_time(t).toarray(), atol=1e-14)
        assert np.allclose(H_ring.derivative_at_time(t).toarray(), H_r.derivative_at_time(t).toarray(), atol=1e-14)
        assert np.allclose(H_k.real_space_density_matrix(rho_k), rho_r, atol=1e-12)
        assert np.allclose(H_k.real_space_bond_currents(rho_k, t), obs_r._compute(rho_r, t), atol=1e-13)


@pytest.mark.parametrize("orientation", list(ORIENTATIONS))
@pytest.mark.parametrize("n_k, shift", [(1, 0.0), (2, 0.0), (3, 0.5), (4, 0.25)])
def test_any_uniform_mesh_is_a_twisted_ring(orientation, n_k, shift):
    """KMesh(N, shift) == ring of N cells with seam twist 2 pi shift: spectrum and dynamics
    agree with the generated real-space ring Hamiltonian (incl. N = 1, 2 where bonds merge)."""
    cell, _ = ORIENTATIONS[orientation]
    dt, T = 0.05 / SCALE, 10 / SCALE
    H_k = tb.BlochHamiltonian(peierls(cell, (1.0, 1.0)), tb.KMesh(n_k, shift=shift))
    H_ring = H_k.to_real_space()
    assert np.allclose(np.sort(H_k.eigenvalues.ravel()), np.sort(H_ring.eigenvalues), atol=1e-13)
    rho_k = H_k.thermal_state_density_matrix(0.0, np.inf)
    rho_r = H_ring.thermal_state_density_matrix(0.0, np.inf)
    assert np.allclose(H_k.real_space_density_matrix(rho_k), rho_r, atol=1e-12)
    tb.BlochRK4NeumannSolver().evolve(rho_k, H_k, dt, T, progress=False)
    tb.RK4NeumannSolver().evolve(rho_r, H_ring, dt, T, progress=False)
    assert np.allclose(H_k.real_space_density_matrix(rho_k), rho_r, atol=1e-11)


def test_explicit_mesh_has_no_ring():
    cell, _ = ORIENTATIONS["armchair"]
    H_k = tb.BlochHamiltonian(peierls(cell, (1.0, 0.0)), tb.KMesh(thetas=[0.1, 0.7], weights=[0.5, 0.5]))
    with pytest.raises(ValueError, match="explicit"):
        H_k.ring_geometry


def test_half_shifted_mesh_is_not_the_ring():
    cell, ring = ORIENTATIONS["armchair"]
    H_r = peierls(ring, (1.0, 0.0))
    H_k = tb.BlochHamiltonian(peierls(cell, (1.0, 0.0)), tb.KMesh(5))  # shift = 0.5
    assert not np.allclose(np.sort(H_k.eigenvalues.ravel()), np.sort(H_r.eigenvalues), atol=1e-6)


def test_zigzag_observables_gamma_only():
    """All observables (incl. the k-resolved VortFlux) on a zigzag ring, Gamma-only Bloch."""
    _, ring = ORIENTATIONS["zigzag"]
    dt, T = 0.05 / SCALE, 10 / SCALE
    H_r = peierls(ring, (0.0, 1.0))
    rho = H_r.thermal_state_density_matrix(0.0, np.inf)
    mk = lambda H: [orb.observables.VorticityObservable(H.geometry, hamiltonian=H),
                    orb.observables.VortFluxObservable(H.geometry, hamiltonian=H),
                    orb.observables.BondCurrentObservable(H.geometry, hamiltonian=H)]
    o_r = mk(H_r)
    tb.RK4NeumannSolver().evolve(rho, H_r, dt, T, observables=o_r, progress=False)
    H_b = tb.BlochHamiltonian(peierls(ring, (0.0, 1.0)), tb.KMesh(1, shift=0.0))
    rho_b = H_b.thermal_state_density_matrix(0.0, np.inf)
    o_b = mk(H_b)
    tb.BlochRK4NeumannSolver().evolve(rho_b, H_b, dt, T, observables=o_b, progress=False)
    for a, b in zip(o_r, o_b):
        assert np.allclose(a.values, b.values, atol=1e-10)


def test_zigzag_cell_too_short_is_rejected():
    geo = orb.HoneycombLatticeGeometry(Lx=2, Ly=10, a_nn=A_NN, pbc_x=True)
    with pytest.raises(ValueError, match="longer supercell"):
        tb.BlochHamiltonian(orb.LinearFieldHamiltonianPeierls(geo, T_HOP, [], B0=B0), 8)
