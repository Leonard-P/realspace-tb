# k-space (Bloch) simulations

For systems periodic along one axis (e.g. an armchair ribbon open in x, periodic in y),
the real-space supercell is a ring of finite circumference. `BlochHamiltonian` uses the
translation symmetry to simulate the infinite system: `N_k` independent small problems
instead of one ring of `N_k` cells. The real-space workflow is unchanged.

```python
geo = orb.HoneycombLatticeGeometry(Lx=17, Ly=2, a_nn=2.6834, pbc_x=False, pbc_y=True)
H_real = orb.LinearFieldHamiltonianPeierls(geo, t_hop, [field], B0=B0)
H = tb.BlochHamiltonian(H_real, kmesh=tb.KMesh(320))          # 320 k-points
rho = H.thermal_state_density_matrix(fermi_level=0.0, beta=float("inf"))   # (N_s, N_k, N_s)
obs = orb.observables.BondCurrentObservable(geo, window=..., hamiltonian=H)
tb.BlochRK4NeumannSolver().evolve(rho, H, dt, total_time, observables=[obs], n_workers=4)
```

## What is new

| Object | Purpose |
|---|---|
| `KMesh(n_k, shift=0.5)` | twist angles θ = kL and weights. `shift=0`: Γ-centred, identical to a ring of `n_k` cells. |
| `BlochHamiltonian(base, kmesh)` | wraps any real-space Hamiltonian of a cell periodic along y; hoppings across the boundary get `e^{±iθ}`. |
| `BlochRK4NeumannSolver` | RK4 for all k at once (batched sparse products); `n_workers` processes on CPU. |
| `evaluate_observable(obs, rho, H, t)` | k-averaged value of one observable without time evolution. |
| `fermi_occupation(E, μ, β, tol)` | shared occupation rule; at β = ∞, states within `tol` of μ get 1/2. |
| `LinearFieldHamiltonianPeierls(..., gauge_origin=...)` | `None`/`"zero"` (old behaviour), `"center"`, or a point. |

## Ribbon orientations

The Bloch axis is the single periodic direction of the geometry (or pass `axis=`).

| Ribbon | Geometry | Smallest cell |
|---|---|---|
| armchair (open x, periodic y) | `pbc_x=False, pbc_y=True` | `Ly=2` (one armchair period) |
| zigzag (periodic x, open y) | `pbc_x=True, pbc_y=False` | `Lx=4` (two zigzag periods; `Lx=2` would bond a pair of sites twice) |

## Back to real space: the mesh defines the ring

A uniform mesh `KMesh(N, shift)` is *exactly* a real-space ring of N cells whose seam bonds carry
the twist `exp(±2πi·shift)` (an Aharonov-Bohm flux of `shift` flux quanta). The ring is therefore
built from the mesh; there is no separate geometry to pass in (and none to get wrong):

| Method / property | Returns |
|---|---|
| `H_k.ring_geometry` | the cell repeated N times along the Bloch axis (armchair: `Ly*N` rows, zigzag: `Lx*N` columns) |
| `H_k.to_real_space()` | a real-space `Hamiltonian` on that ring, usable with `RK4NeumannSolver` and all observables |
| `H_k.real_space_hamiltonian(t)` | dense ring H(t) |
| `H_k.real_space_density_matrix(rho)` | dense ring ρ, convention `rho_ij = <c_j^† c_i>` |
| `H_k.real_space_bond_currents(rho, t, pairs=None)` | `2 Im(H_ij rho_ji)` for ring site pairs (default: all bonds, order of `ring_geometry.nearest_neighbors`); computes only the requested elements |
| `H_k.site_map()` | ring index -> (site in cell, cell index) |

With `shift=0` (Γ-centred) the ring is the ordinary periodic ring, so e.g. `Ly=2` + `KMesh(5, shift=0)`
reproduces your own `Ly=10` real-space ring element by element. The default `KMesh(5)` (shift 0.5) is
the same ring with half a flux quantum threaded through it. A mesh given as explicit `thetas/weights`
has no finite ring; the methods above then raise an error. `correlator`, `hopping` and
`bond_current` give infinite-system elements for sites labelled `(alpha, cell)`.

## Rules

* The fields must keep translation symmetry along y: Peierls (vector-potential) fields in any
  direction, B in the Landau gauge A = (0, B x) (used automatically when `pbc_y=True`).
  A scalar potential along y is rejected; along x it is allowed.
* The initial state must be translation-invariant too (any `f(H)` is).
* Observables are k-averages, i.e. values of the infinite ribbon, on the bonds of one cell.
  Bond-local observables (densities, bond currents, vorticities, polarizations, forces,
  `LatticeFrameObservable`) are evaluated once with a k-averaged matrix
  (`bloch_pairwise = True`). Observables with products such as `H @ rho` (`VortFlux*`,
  `BondCurrentFluxObservable`) and any observable without the flag are evaluated k-point by
  k-point, which is exact for anything linear in ρ.
* Converge in `n_k`: x0-independence and symmetric edge currents are reached only when
  converged (with x0 at the centre they are symmetric at every `n_k`, but not yet converged).
  At T = 0 with states near μ the convergence is power-law; a small finite temperature
  (`beta`) makes it exponential.

## Parallelism

* GPU: `tb.backend.set_backend(use_gpu=True)` runs all k-points in one batch (`n_workers=1`).
* CPU: `n_workers > 1` splits the k-mesh into chunks evolved in separate processes; results
  are merged with the chunk weights (exact, the blocks never couple). Install `threadpoolctl`
  to pin each worker to one BLAS thread. Custom classes (fields, observables) defined inside a
  notebook may not be picklable on macOS/Windows (spawn); define them in a module.

## Tests

`python -m pytest realspace_tb/tests -q` (run from the folder containing `realspace_tb`) checks:
Γ-only Bloch = real-space ring; primitive cell with 5 k = 10-row ring (state and dynamics,
fields along x and y); pairwise shortcut = k-by-k sum; parallel = serial; converged edge
currents symmetric and gauge-origin independent; 1/2 occupation of zero modes; uniform
flux B·A_hex per hexagon in both gauges.
