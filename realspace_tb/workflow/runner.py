"""
Run simulations from configurations and load their results.

    from realspace_tb.workflow import load_config, expand, run, load_results
    runs = expand(load_config("sweep.toml"))
    run_simulation(runs[3])                      # one run (e.g. one Slurm array task)
    results = load_results("results/my_sweep")  # all finished runs, with their configs

Each run writes to <output_dir>/<name>/<run_id>/:
    config.json      the complete configuration of this run (input units)
    observables.npz  times + one array per observable (+ "<name>_equilibrium")
    metadata.json    wall time, steps, versions, host, Slurm ids, residual vector potential
    final_state.npy  optional (output.save_final_state)
"""

from __future__ import annotations

import copy
import json
import os
import platform
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from .. import backend as B
from ..bloch import BlochHamiltonian
from ..bloch_solver import BlochRK4NeumannSolver, _MeasureContext, _measure
from ..lindbladsolver import RK4NeumannSolver
from ..observable import MeasurementWindow
from . import build
from .config import config_hash, public_config


def _to_numpy(a: Any) -> Any:
    if isinstance(a, dict):
        return {k: _to_numpy(v) for k, v in a.items()}
    return a.get() if hasattr(a, "get") else np.asarray(a)


def run_dir(run_cfg: dict) -> Path:
    return Path(run_cfg["output_dir"]) / run_cfg["name"] / run_cfg["_run_id"]


def is_done(run_cfg: dict) -> bool:
    d = run_dir(run_cfg)
    if not (d / "observables.npz").exists() or not (d / "config.json").exists():
        return False
    stored = json.loads((d / "config.json").read_text())
    return config_hash(stored) == config_hash(run_cfg)   # ignores n_workers, output settings


def equilibrium_values(observables: dict, rho, H) -> dict:
    """Every observable evaluated once on the initial state (works for composite observables too)."""
    out = {}
    for name, obs in observables.items():
        probe = copy.deepcopy(obs, memo={id(H): H, id(getattr(H, "base", None)): getattr(H, "base", None)})
        _reset_windows(probe)
        if isinstance(H, BlochHamiltonian):
            _measure(probe, _MeasureContext(H, rho, 0.0), 0)
        else:
            probe.measure(rho, 0.0, 0)
        probe.finalize()
        out[name] = _first(probe.values)
    return out


def _reset_windows(obs) -> None:
    obs.window = MeasurementWindow()
    obs._results, obs._times = [], []
    for v in vars(obs).values():
        if hasattr(v, "window") and hasattr(v, "_results"):
            _reset_windows(v)


def _first(values):
    if isinstance(values, dict):
        return {k: _first(v) for k, v in values.items()}
    return None if values is None else values[0]


def _flatten(prefix: str, value, out: dict) -> None:
    value = _to_numpy(value)
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten(f"{prefix}__{k}", v, out)
    elif value is not None:
        out[prefix] = value


def _git_commit() -> str | None:
    try:
        here = Path(__file__).resolve().parent
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True,
                              text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def run_simulation(run_cfg: dict, force: bool = False, progress: bool = False) -> Path:
    """Execute one expanded run configuration and save its results. Returns the run directory.

    With output.skip_existing (default) a run whose results exist for the identical
    configuration is skipped, so a sweep can simply be resubmitted after failures.
    """
    out_dir = run_dir(run_cfg)
    if not force and run_cfg["output"]["skip_existing"] and is_done(run_cfg):
        print(f"[{run_cfg['_run_id']}] results exist, skipping ({out_dir})")
        return out_dir

    build.set_backend(run_cfg)
    geometry = build.build_geometry(run_cfg)
    fields = build.build_fields(run_cfg)
    H_real, H = build.build_hamiltonian(run_cfg, geometry, fields)
    rho = build.initial_state(run_cfg, H)
    observables = build.build_observables(run_cfg, geometry, H)
    equilibrium = equilibrium_values(observables, rho, H) if run_cfg["output"]["save_equilibrium"] else {}

    tm = run_cfg["time"]
    n_steps = int(round(tm["total_time"] / tm["dt"]))
    total_time = (n_steps + 0.5) * tm["dt"]  # the solvers take int(total/dt) steps: exactly n_steps
    print(f"[{run_cfg['_run_id']}] {run_cfg['_swept'] or 'single run'}: {run_cfg['method']['mode']}, "
          f"{H_real.geometry.Lx}x{H_real.geometry.Ly} sites, {n_steps} steps", flush=True)

    t0 = time.time()
    kwargs = dict(dt=tm["dt"], total_time=total_time, tau=tm["tau"],
                  observables=list(observables.values()), progress=progress)
    if isinstance(H, BlochHamiltonian):
        BlochRK4NeumannSolver().evolve(rho, H, n_workers=build.n_workers(run_cfg), **kwargs)
    else:
        RK4NeumannSolver().evolve(rho, H, **kwargs)
    wall = time.time() - t0

    out_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {}
    for name, obs in observables.items():
        _flatten(name, obs.values, arrays)
        _flatten(name + "__times", obs.measurement_times, arrays)
        if name in equilibrium:
            _flatten(name + "_equilibrium", equilibrium[name], arrays)
    np.savez_compressed(out_dir / "observables.npz", **arrays)
    if run_cfg["output"]["save_final_state"]:
        np.save(out_dir / "final_state.npy", _to_numpy(rho))

    a_final = build.final_vector_potential(fields, n_steps * tm["dt"])
    pulse_area = sum(abs(f["E0"]) * f["sigma"] * np.sqrt(2 * np.pi)
                     for f in run_cfg["fields"]
                     if f["type"] in ("pulse", "elliptical_pulse"))
    only_pulses = all(f["type"] in ("pulse", "elliptical_pulse") for f in run_cfg["fields"])
    meta = {
        "run_id": run_cfg["_run_id"], "index": run_cfg["_index"], "swept": run_cfg["_swept"],
        "wall_time_s": wall, "n_steps": n_steps, "total_time_au": n_steps * tm["dt"], "dt_au": tm["dt"],
        "n_sites_cell": int(H_real.geometry.Lx * H_real.geometry.Ly),
        "n_workers": build.n_workers(run_cfg) if isinstance(H, BlochHamiltonian) else 1,
        "integral_E_dt_final_au": a_final.tolist(),
        "precision": run_cfg["method"]["precision"], "dtype": str(np.dtype(B.DTYPE)),
        "host": socket.gethostname(), "python": platform.python_version(),
        "numpy": np.__version__, "git_commit": _git_commit(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    # written last: its presence (with observables.npz) marks the run as complete
    (out_dir / "config.json").write_text(json.dumps(public_config(run_cfg), indent=2))
    print(f"[{run_cfg['_run_id']}] done in {wall:.1f} s -> {out_dir}; "
          f"integral of E dt at the end: {np.round(a_final, 12).tolist()} a.u.", flush=True)
    if only_pulses and pulse_area > 0 and np.linalg.norm(a_final) > 1e-6 * pulse_area:
        print(f"[{run_cfg['_run_id']}] WARNING: the pulses leave a residual vector potential "
              f"({np.linalg.norm(a_final) / pulse_area:.1e} of the pulse area); increase sigma*omega "
              f"or adjust the CEP.", flush=True)
    return out_dir


# ------------------------------------------------------------------------------- loading
def load_run(path: str | Path) -> dict:
    """{'config', 'metadata', 'data'} of one finished run; data[name] etc. as numpy arrays."""
    path = Path(path)
    with np.load(path / "observables.npz") as npz:
        data = {k: npz[k] for k in npz.files}
    return {
        "path": path,
        "config": json.loads((path / "config.json").read_text()),
        "metadata": json.loads((path / "metadata.json").read_text()),
        "data": data,
    }


def load_results(sweep_dir: str | Path) -> list[dict]:
    """All finished runs below <output_dir>/<name>, sorted by run index."""
    runs = [load_run(p) for p in sorted(Path(sweep_dir).iterdir())
            if (p / "config.json").exists() and (p / "observables.npz").exists()]
    return sorted(runs, key=lambda r: r["metadata"]["index"])
