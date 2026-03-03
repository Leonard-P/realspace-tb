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
  ohc, fourier_at_omega          -- orbital Hall conductivity analysis
  save_simulation_animation      -- render animated GIF/MP4
  show_simulation_frame          -- render a single snapshot
```

## Core concepts

**Backend.** Call `tb.backend.set_backend(use_gpu=False, precision="double")` before constructing any objects. All Hamiltonians, observables, and arrays use the selected backend and precision. Do not change it mid-simulation.

**Hamiltonian.** Subclass `Hamiltonian` and implement `at_time(t)` returning a sparse matrix. Built-in implementations couple a honeycomb hopping matrix to a homogeneous electric field in either dipole gauge or Peierls gauge.

**Observable.** Subclass `Observable` and implement `_compute(rho, t)`. The base class handles windowed/strided measurement and result accumulation. After `evolve()`, access results via `obs.values` and `obs.measurement_times`.

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
- **`ground_state_density_matrix()` uses dense diagonalisation** — O(N^3), practical up to a few thousand sites.
- **`electron_mass=0.741`** is the default effective mass in natural units (`t_hop = 1`, `a = 1`). Adjust if using different parameters.

## Extending

Custom Hamiltonians, observables, geometries, and field amplitudes can be defined by subclassing the corresponding abstract base classes. See [`examples/custom_observable.ipynb`](examples/custom_observable.ipynb) and the [API reference](https://leonard-p.github.io/realspace-tb/) for details.

## License

MIT