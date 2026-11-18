"""
Estimate the memory (for #SBATCH --mem) and the run time (for #SBATCH -t) of a run before submitting.

How it works:
* Sizes are computed, not guessed: the run's geometry, Hamiltonian and observables are built, and
  every observable is evaluated once to get the exact size of one measurement.
* Memory = a per-process baseline (measured: Python + numpy/scipy/package) + the density-matrix
  arrays (state, RK4 work arrays, eigenvectors) + the stored measurement series, with the
  parallel layout taken into account (main process, forkserver, n_workers worker processes;
  every worker stores full-length measurement series, and the main process receives all of them).
* Time = a short benchmark of a few RK4 steps and measurements on ONE worker's share of the
  k-points, scaled to the full number of steps. It is only as representative as the machine it
  runs on: run `estimate` on a compute node (e.g. inside `srun --pty`) for numbers that match jobs.
"""

from __future__ import annotations

import copy
import math
import resource
import time

import numpy as np

from ..bloch import BlochHamiltonian
from ..bloch_solver import BlochRK4NeumannSolver, _MeasureContext, _measure
from ..lindbladsolver import RK4NeumannSolver
from . import build
from .runner import _reset_windows

# per-array overheads of the time loop, in units of the state size (measured, with margin):
WORKER_STATE_COPIES = 10      # rho, acc, tmp, k, X, d and short-lived temporaries in one RK4 stage
MAIN_STATE_COPIES = 4         # rho, eigenvectors, rho(k) during setup, returned chunks
SAFETY = 1.3                  # margin on top of the estimate


def _baseline_bytes() -> int:
    """Peak resident memory of this process so far: Python + numpy/scipy + the package (Linux: KiB)."""
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def _n_measurements(n_steps: int, dt: float, spec: dict) -> int:
    first = max(0, math.ceil(spec["start"] / dt - 1e-9))
    last = n_steps - 1 if math.isinf(spec["end"]) else min(n_steps - 1, math.floor(spec["end"] / dt + 1e-9))
    if last < first:
        return 0
    stride = int(spec["stride"])
    first = first + (-first) % stride
    return 0 if first > last else (last - first) // stride + 1


def _nbytes(value) -> int:
    if isinstance(value, dict):
        return sum(_nbytes(v) for v in value.values())
    return 0 if value is None else int(np.asarray(value).nbytes)


def estimate(run_cfg: dict, benchmark_steps: int = 10) -> dict:
    """Memory and time estimate for one expanded run configuration (see module docstring)."""
    base = _baseline_bytes()          # Python + imported packages, before any run-specific arrays
    build.set_backend(run_cfg)
    geometry = build.build_geometry(run_cfg)
    fields = build.build_fields(run_cfg)
    H_real, H = build.build_hamiltonian(run_cfg, geometry, fields)
    kspace = isinstance(H, BlochHamiltonian)
    n_workers = min(build.n_workers(run_cfg), H.n_k) if kspace else 1
    n_k = H.n_k if kspace else 1
    n_s = int(H_real.geometry.Lx * H_real.geometry.Ly)
    dt, total = run_cfg["time"]["dt"], run_cfg["time"]["total_time"]
    n_steps = int(round(total / dt))

    # one worker's share of the k-points: used to measure sizes and time
    chunk = math.ceil(n_k / n_workers)
    H_probe = H.subset(range(chunk))[0] if kspace else H
    rho = build.initial_state(run_cfg, H_probe)
    itemsize = rho.itemsize
    state_full = n_s * n_s * n_k * itemsize
    state_chunk = n_s * n_s * chunk * itemsize

    observables = build.build_observables(run_cfg, geometry, H_probe)
    series_bytes, per_obs = 0, {}
    t_meas = 0.0
    for spec, (name, obs) in zip(run_cfg["observables"], observables.items()):
        # probe copy that measures now, whatever the observable's window says
        obs = copy.deepcopy(obs, memo={id(H_probe): H_probe, id(getattr(H_probe, "base", None)): getattr(H_probe, "base", None)})
        _reset_windows(obs)
        t0 = time.perf_counter()
        if kspace:
            _measure(obs, _MeasureContext(H_probe, rho, 0.0), 0)
        else:
            obs.measure(rho, 0.0, 0)
        cost = time.perf_counter() - t0
        obs.finalize()
        one = _nbytes(obs.values) // max(1, len(obs.measurement_times))
        n_meas = _n_measurements(n_steps, dt, spec)
        per_obs[name] = {"bytes_per_measurement": one, "measurements": n_meas,
                         "series_MB": one * n_meas / 1e6}
        series_bytes += one * n_meas
        t_meas += cost * n_meas

    if kspace and n_workers > 1:
        worker = base + WORKER_STATE_COPIES * state_chunk + 2 * series_bytes
        main = base + MAIN_STATE_COPIES * state_full + (n_workers + 1) * series_bytes
        total_bytes = main + n_workers * worker + base          # + forkserver
    else:
        main = base + (WORKER_STATE_COPIES + MAIN_STATE_COPIES) * state_full + 2 * series_bytes
        worker, total_bytes = 0, main

    # time: a few RK4 steps on one worker's share
    probe = rho.copy()
    t0 = time.perf_counter()
    if kspace:
        solver = BlochRK4NeumannSolver()
        for i in range(benchmark_steps):
            solver._step(i * dt, probe, H_probe, dt)
    else:
        solver = RK4NeumannSolver()
        for i in range(benchmark_steps):
            solver._step(i * dt, probe, H_probe, dt)
    t_step = (time.perf_counter() - t0) / benchmark_steps
    t_loop = n_steps * t_step + t_meas

    return {
        "run_id": run_cfg["_run_id"], "mode": run_cfg["method"]["mode"],
        "n_sites": n_s, "n_k": n_k, "n_workers": n_workers, "k_per_worker": chunk, "n_steps": n_steps,
        "state_MB": state_full / 1e6, "measurement_series_MB": series_bytes / 1e6, "observables": per_obs,
        "memory_main_MB": main / 1e6, "memory_per_worker_MB": worker / 1e6,
        "memory_total_MB": total_bytes / 1e6,
        "recommended_mem_GB": max(1, math.ceil(SAFETY * total_bytes / 1e9)),
        "time_per_step_s": t_step, "measurement_time_s": t_meas,
        "time_loop_s": t_loop,
        "recommended_time": _format_time(SAFETY * t_loop + 120),   # + start-up / setup margin
    }


def _format_time(seconds: float) -> str:
    seconds = int(math.ceil(seconds / 60.0) * 60)
    h, rest = divmod(seconds, 3600)
    d, h = divmod(h, 24)
    m = rest // 60
    return f"{d}-{h:02d}:{m:02d}:00" if d else f"{h:02d}:{m:02d}:00"
