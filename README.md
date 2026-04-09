# realspace-tb

Real-space tight-binding simulations with time-dependent density matrix evolution, primarily for studying orbital magnetism in graphene-like 2D systems.

`realspace_tb` solves the von Neumann equation for a density matrix on a tight-binding lattice and measures observables at each time step. The core module provides generic infrastructure (time integrator, abstract Hamiltonian/Observable, CPU/GPU backend). The `orbitronics_2d` submodule supplies concrete components for honeycomb-lattice simulations of orbital angular momentum from loop currents.

**Documentation:** [API reference (pdoc)](https://leonard-p.github.io/realspace-tb/)

## Installation

```bash
pip install realspace-tb
```

For GPU acceleration (optional):

```bash
pip install "realspace-tb[gpu]"
```

## Quick start

```python
import numpy as np
import realspace_tb as tb
from realspace_tb import orbitronics_2d as o2d
from realspace_tb.orbitronics_2d import observables as obs

geom = o2d.HoneycombLatticeGeometry(Lx=20, Ly=10)

field = o2d.RampedACFieldAmplitude(
    E0=0.01, omega=3.0, T_ramp=5.0, direction=np.array([1.0, 0.0])
)

H = o2d.LinearFieldHamiltonianPeierls(geom, field)
rho = H.ground_state_density_matrix(fermi_level=0.0)

window = tb.MeasurementWindow(start_time=0.0, stride=10)
frame_obs = obs.LatticeFrameObservable(geom, hamiltonian=H, window=window)

solver = tb.RK4NeumannSolver()
solver.evolve(rho, H, dt=0.01, total_time=50.0, observables=[frame_obs])

o2d.save_simulation_animation(frame_obs, "result.gif", fps=15)
```

See [`examples/quickstart.ipynb`](examples/quickstart.ipynb) for a runnable version and [`examples/custom_observable.ipynb`](examples/custom_observable.ipynb) for how to define your own observables.

## Architecture

```
realspace_tb
  backend          -- NumPy/CuPy switch + precision control
  Hamiltonian      -- abstract base class: return sparse H(t)
  Observable       -- abstract base class: measure from rho(t)
  MeasurementWindow-- control when measurements happen
  RK4NeumannSolver -- RK4 time integrator (von Neumann + optional relaxation)

realspace_tb.orbitronics_2d
  HoneycombLatticeGeometry       -- honeycomb lattice with OBC/PBC
  LinearFieldHamiltonian         -- dipole-gauge E-field coupling
  LinearFieldHamiltonianPeierls  -- Peierls-gauge E-field coupling
  RampedACFieldAmplitude         -- ramped AC electric field E(t)
  observables.*                  -- OAM, density, currents, composite frame
  units                          -- default graphene material/unit conversion constants
  ohc, fourier_at_omega          -- orbital Hall conductivity analysis
  biot_savart.*                  -- magnetic-field reconstruction from bond currents
  save_simulation_animation      -- render animated GIF/MP4
  show_simulation_frame          -- render a single snapshot
```

## Core concepts

**Backend.** Call `tb.backend.set_backend(use_gpu=False, precision="double")` before constructing any objects. All Hamiltonians, observables, and arrays use the selected backend and precision. Do not change it mid-simulation.

**Hamiltonian.** Subclass `Hamiltonian` and implement `at_time(t)` returning a sparse matrix. Built-in implementations couple a honeycomb hopping matrix to a homogeneous electric field in either dipole gauge or Peierls gauge.

**Observable.** Subclass `Observable` and implement `_compute(rho, t)`. The base class handles windowed/strided measurement and result accumulation. After `evolve()`, access results via `obs.values` and `obs.measurement_times`.

**Units.** The core solver works in natural units with `hbar = a_nn = t_hop = e = 1`. That means the density matrix, Hamiltonian matrix elements, and bond currents returned by the built-in observables are dimensionless unless a helper explicitly converts them back to SI. For graphene-like defaults, the orbitronics helpers use the material constants from `realspace_tb.orbitronics_2d.units`:

- `DEFAULT_T_HOP_EV = 2.8`
- `DEFAULT_A_NN_M = 0.142e-9`
- `DEFAULT_EFFECTIVE_ELECTRON_MASS = m_e * a_nn^2 * t_hop / hbar^2 = 0.740937862...`
- `DEFAULT_CURRENT_UNIT_A = e * t_hop / hbar = 6.82e-4 A`

To simulate a different material, pass your own values at construction time or use the helper functions directly:

```python
from realspace_tb.orbitronics_2d import units

m_eff = units.effective_electron_mass(t_hop_ev=1.6, a_nn_m=0.25e-9)
I0 = units.current_unit_amperes(t_hop_ev=1.6)
```

The OAM observables default to `DEFAULT_EFFECTIVE_ELECTRON_MASS`, and the Biot-Savart helpers default to `DEFAULT_T_HOP_EV` and `DEFAULT_A_NN_M`.

**Solver.** `RK4NeumannSolver().evolve()` integrates the (optionally damped) von Neumann equation in-place. The density matrix `rho` is modified directly — copy it beforehand if needed.

## Provided observables

| Observable | Output shape | Description |
|---|---|---|
| `PlaquetteOAMObservable` | `(frames, plaquettes)` | Orbital angular momentum per hexagonal plaquette from loop currents |
| `OrbitalPolarizationObservable` | `(frames, 2)` | Macroscopic orbital polarization vector |
| `SiteDensityObservable` | `(frames, sites)` | Diagonal of the density matrix |
| `BondCurrentObservable` | `(frames, bonds)` | Gauge-invariant nearest-neighbor bond currents |
| `LatticeFrameObservable` | dict | Composite: density + currents + OAM (for animation) |

## Common pitfalls

- **Set the backend before constructing anything.** Objects read dtype/device settings at construction time.
- **Dipole gauge + PBC is unphysical.** Use `LinearFieldHamiltonianPeierls` for periodic systems.
- **Pass the Hamiltonian to current-based observables when using Peierls gauge.** Without it, the simple `2 Im(rho_ij)` formula is not gauge-invariant.
- **Biot-Savart helpers expect dimensionless bond currents.** If you want amperes, multiply the current observable by `units.current_unit_amperes(...)` first; the field reconstruction then converts that scale internally and returns Tesla.
- **`ground_state_density_matrix()` uses dense diagonalisation** — O(N^3), practical up to a few thousand sites.
- **`electron_mass` defaults to the graphene-derived effective mass.** Override it if your material has a different `t_hop` or `a_nn`.

## Extending

Custom Hamiltonians, observables, geometries, and field amplitudes can be defined by subclassing the corresponding abstract base classes. See [`examples/custom_observable.ipynb`](examples/custom_observable.ipynb) and the [API reference](https://leonard-p.github.io/realspace-tb/) for details.

## License

MIT