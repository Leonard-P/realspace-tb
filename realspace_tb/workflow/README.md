# `realspace_tb.workflow`: configuration-driven simulations and sweeps

A simulation is described by a configuration (TOML or JSON file, or a Python dict). A
configuration can contain a sweep; every combination of the swept values is one run. The same
configuration drives local runs, notebooks and Slurm job arrays.

```bash
python -m realspace_tb.workflow list  sweep.toml          # runs, swept values, done/todo
python -m realspace_tb.workflow run   sweep.toml --index 3 # one run (Slurm: index from the array task)
python -m realspace_tb.workflow run   sweep.toml --all     # all runs, one after the other
python -m realspace_tb.workflow estimate sweep.toml --cpus 16   # memory and time per run
python -m realspace_tb.workflow slurm sweep.toml --account hpc-prf-metdyn --cpus 16 --mem auto --time auto --submit
```

```python
from realspace_tb.workflow import load_config, expand, run_simulation, load_results
runs = expand(load_config("sweep.toml"))
run_simulation(runs[0])
results = load_results("/scratch/.../results/<name>")   # list of {config, metadata, data, path}
```

## Configuration reference

Units: every number is in Hartree atomic units, as everywhere in the package (energies and
frequencies in Hartree, fields in E_h/(e a_0), times in hbar/E_h = 0.0242 fs, lengths in bohr,
`beta` in 1/E_h, `B0` as in `LinearFieldHamiltonianPeierls`). Nothing is converted.
Handy values: 2.7 eV = 0.0992 Ha, 3.0 eV = 0.1101 Ha, 1 fs = 41.34 a.u. of time,
5.1 MV/m = 9.92e-6 a.u. of field. Unknown keys are errors (typos are caught).

| Section | Key | Default | Meaning |
|---|---|---|---|
| top | `name`, `output_dir` | `"run"`, `"results"` | results go to `output_dir/name/<run_id>/` |
| `[system]` | `ribbon`, `width` | – | `"armchair"` (open x, periodic y; width = dimer lines) or `"zigzag"` (periodic x, open y; width = rows) |
| | `cells` | 1 | periodic cells in the supercell (armchair: 2 rows each, zigzag: 4 columns each) |
| | `periodic` | true | false: open ends, i.e. a finite flake (real space) |
| | `Lx`, `Ly`, `pbc_x`, `pbc_y` | – | explicit geometry instead of `ribbon`/`width` |
| | `t_hop`, `B0`, `a_nn` | -0.0992, 0, 2.6834 | hopping (Ha), magnetic field (a.u.), bond length (bohr) |
| | `gauge_origin` | `"center"` | `"center"`, `"zero"` or `[x0, y0]` |
| | `gauge` | `"velocity"` | `"velocity"` (Peierls; any fields, B) or `"length"` (one field, no B) |
| `[method]` | `mode` | `"kspace"` | `"kspace"` (Bloch, infinite ribbon) or `"realspace"` |
| | `n_k`, `k_shift` | 320, 0 | k-points, mesh shift (threaded flux) |
| | `n_workers` | `"auto"` | processes for the k-chunks; auto = `$SLURM_CPUS_PER_TASK` |
| | `precision`, `gpu` | `"double"`, false | backend |
| `[state]` | `fermi_level`, `beta`, `tol` | 0, inf, None | initial grand-canonical state |
| `[time]` | `dt`, `total_time`, `tau` | 0.1, required, inf | time step, length, relaxation time (hbar/E_h) |
| `[[fields]]` | `type` + parameters | – | any number; superimposed (see below) |
| `[[observables]]` | `name`, `stride`, `start`, `end` | –, 1, 0, inf | what to measure and when |
| `[output]` | `save_equilibrium`, `save_final_state`, `skip_existing` | true, false, true | |

Field types (`direction`: `"periodic"`, `"transverse"`, `"x"`, `"y"` or `[dx, dy]`):

| `type` | parameters (defaults) |
|---|---|
| `pulse` | `E0, t_c, sigma, omega, cep (0), direction ("periodic")` |
| `elliptical_pulse` | `E0, t_c, sigma, omega, cep (0), ellipticity (1), vartheta (0), varphi (0)` |
| `ramped_ac` | `E0, omega, T_ramp, direction ("periodic")` |
| `ramped_dc` | `E0, T_ramp, direction ("periodic")` |
| `delta_kick` | `E0 (field x time), direction ("periodic")` |

Observables: `bond_current`, `bond_current_flux`, `bond_current_force`, `lattice_frame`,
`site_density`, `vector_bond_current`, `vector_vort_flux`, `vector_vort_flux_mod`, `vort_flux`,
`vort_flux_mod`, `vort_pol`, `vort_pol_restricted`, `vort_source`, `vort_source_pol`,
`vort_source_pol_restricted`, `vorticity` (every `...Observable` class, found automatically).

## Sweeps

```toml
[sweep.product]                       # every combination
"system.B0" = [-0.0496, 0.0, 0.0496]
"fields.0.E0" = [9.92e-6, -9.92e-6]

[sweep.zip]                           # vary together (same length)
"system.ribbon" = ["armchair", "zigzag"]
"system.width" = [17, 10]
```

Paths address any key; list entries by index (`fields.1.t_c`). Runs = product x zip
(here 3 x 2 x 2 = 12). Each run gets a `run_id` = index + hash of its physical configuration.

## Output of each run

`config.json` (complete configuration of the run), `observables.npz` (for each observable
`<name>` and `<name>__times`, `<name>_equilibrium`; composite ones as `<name>__<key>`),
`metadata.json` (wall time, steps, residual integral of E, versions, git commit, Slurm ids),
optional `final_state.npy`. A run whose results exist for the same configuration is skipped,
so a sweep can be resubmitted after failures; `slurm --submit` only submits unfinished runs.

## Extending

New field type: add its parameter units to `config.FIELD_TYPES` and a builder with
`build.register_field("name", lambda run, f: [MyFieldAmplitude(...)])`. New observables in
`orbitronics_2d/observables.py` are available by their snake_case name automatically.

## Estimating memory and time

`estimate CONFIG --cpus N` builds every run, evaluates each observable once (exact size of one
measurement), counts the measurements, and benchmarks a few RK4 steps on one worker's share of
the k-points. It prints per run the state size, the stored measurement series, the total memory
of all processes (main + forkserver + N workers, each with its own copy of the series) and the
expected time loop, with recommended `--mem` / `-t` (30 % margin; time + 2 min for start-up).
`--details` lists the memory per observable. In tests the memory estimate was 20-40 % above the
measured peak. Timing depends on the machine: run `estimate` on a compute node
(`srun -A ... -p normal -c N -t 00:30:00 --pty bash`) for numbers that match the jobs.
`slurm ... --mem auto --time auto` uses the largest estimate of the sweep.
