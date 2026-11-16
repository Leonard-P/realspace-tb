"""
Simulation configurations: one plain dictionary per run, read from TOML/JSON files or built in Python.

A configuration has these sections (all optional except `time.total_time`; see DEFAULTS):

    name, output_dir
    [system]        lattice geometry, hopping, magnetic field, gauge
    [method]        "kspace" (Bloch) or "realspace", k-mesh, workers, precision, GPU
    [state]         initial grand-canonical state (Fermi level, inverse temperature)
    [time]          time step, total time, relaxation time
    [[fields]]      any number of field components; they are superimposed
    [[observables]] what to measure and when
    [output]        what to save
    [sweep]         parameters to vary: one run per combination

Units: every input is in Hartree atomic units (e = hbar = m_e = 4 pi eps_0 = 1), exactly as in the
rest of the package: energies and frequencies in Hartree, fields in E_h/(e a_0), B0 in the package's
atomic-unit convention, times in hbar/E_h, lengths in bohr, beta in 1/E_h. Nothing is converted.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

try:  # Python >= 3.11
    import tomllib as _toml
except ImportError:  # Python 3.10: optional "tomli" package
    try:
        import tomli as _toml  # type: ignore[no-redef]
    except ImportError:
        _toml = None

INF = float("inf")

# ------------------------------------------------------------------------------------- schema
DEFAULTS: dict[str, Any] = {
    "name": "run",
    "output_dir": "results",
    "system": {
        "lattice": "honeycomb",
        "ribbon": None,          # "armchair" (open x, periodic y) | "zigzag" (periodic x, open y) | None
        "width": None,           # armchair: dimer lines (Lx); zigzag: zigzag rows (Ly)
        "cells": 1,              # periodic cells in the simulated supercell (armchair 2 rows, zigzag 4 columns each)
        "periodic": True,        # False: open ends -> a finite flake (real space only)
        "Lx": None, "Ly": None, "pbc_x": None, "pbc_y": None,   # explicit geometry instead of `ribbon`
        "a_nn": 2.6834,          # bond length (bohr)
        "t_hop": -0.0992,        # hopping (Hartree; -0.0992 = -2.7 eV)
        "B0": 0.0,               # out-of-plane magnetic field (atomic units)
        "gauge_origin": "center",  # "center" | "zero" | [x0, y0] (bohr)
        "gauge": "velocity",     # "velocity" (Peierls, any fields) | "length" (one field, no B)
    },
    "method": {
        "mode": "kspace",        # "kspace" | "realspace"
        "n_k": 320,
        "k_shift": 0.0,          # mesh shift = threaded flux in flux quanta (0: Gamma-centred)
        "n_workers": "auto",     # "auto" = $SLURM_CPUS_PER_TASK (or 1); k-space only
        "precision": "double",   # "double" | "single"
        "gpu": False,
    },
    "state": {
        "fermi_level": 0.0,      # Hartree
        "beta": INF,             # inverse temperature (1/Hartree); inf = T -> 0 with f = 1/2 at mu
        "tol": None,             # degeneracy tolerance at T = 0 in Hartree (None: eigensolver error bound)
    },
    "time": {
        "dt": 0.1,               # hbar/E_h (0.1 a.u. = 2.4 as)
        "total_time": None,      # required (hbar/E_h)
        "tau": INF,              # relaxation time towards the initial state (inf = none)
    },
    "fields": [],
    "observables": [{"name": "vorticity"}, {"name": "bond_current"}],
    "output": {
        "save_equilibrium": True,  # observables evaluated on the initial state (to subtract)
        "save_final_state": False,  # final density matrix (e.g. to restart)
        "skip_existing": True,     # do not recompute a run whose results already exist
    },
    "sweep": {"product": {}, "zip": {}},
}

# field types: allowed parameters, required parameters, defaults (all in atomic units)
FIELD_TYPES: dict[str, dict[str, Any]] = {
    "pulse": {  # Gaussian pulse E0 exp(-(t-t_c)^2 / 2 sigma^2) cos(omega (t - t_c) + cep)
        "params": ["E0", "t_c", "sigma", "omega", "cep", "direction"],
        "required": ["E0", "t_c", "sigma", "omega"],
        "defaults": {"cep": 0.0, "direction": "periodic"},
    },
    "elliptical_pulse": {  # two orthogonal pulse components (EllipticalPulseFactory)
        "params": ["E0", "t_c", "sigma", "omega", "cep", "ellipticity", "vartheta", "varphi"],
        "required": ["E0", "t_c", "sigma", "omega"],
        "defaults": {"cep": 0.0, "ellipticity": 1.0, "vartheta": 0.0, "varphi": 0.0},
    },
    "ramped_ac": {  # E0 sin(omega t), switched on over T_ramp
        "params": ["E0", "omega", "T_ramp", "direction"],
        "required": ["E0", "omega", "T_ramp"],
        "defaults": {"direction": "periodic"},
    },
    "ramped_dc": {  # constant field after a ramp of length T_ramp
        "params": ["E0", "T_ramp", "direction"],
        "required": ["E0", "T_ramp"],
        "defaults": {"direction": "periodic"},
    },
    "delta_kick": {  # E0 delta(t)
        "params": ["E0", "direction"],
        "required": ["E0"],
        "defaults": {"direction": "periodic"},
    },
}

OBSERVABLE_KEYS = {"name": None, "stride": 1, "start": 0.0, "end": INF}


class ConfigError(ValueError):
    """Raised for invalid or incomplete configurations."""


# --------------------------------------------------------------------------------- loading
def load_config(source: str | Path | dict) -> dict:
    """Read a configuration from a TOML/JSON file (or take a dict) and fill in the defaults.

    The returned dict still contains the `sweep` section; use `expand` to get the individual runs.
    """
    if isinstance(source, dict):
        raw = copy.deepcopy(source)
    else:
        path = Path(source)
        text = path.read_text()
        if path.suffix == ".json":
            raw = json.loads(text)
        else:
            if _toml is None:
                raise ConfigError("Reading TOML needs Python >= 3.11 or the 'tomli' package; use JSON otherwise.")
            raw = _toml.loads(text)
    return _with_defaults(raw)


def _with_defaults(raw: dict) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if {"units", "scale"} & set(raw):
        raise ConfigError("'units'/'scale' no longer exist: all inputs are Hartree atomic units")
    unknown = set(raw) - set(DEFAULTS)
    if unknown:
        raise ConfigError(f"Unknown top-level key(s): {sorted(unknown)}")
    for key, value in raw.items():
        if isinstance(DEFAULTS[key], dict) and key != "sweep":
            if not isinstance(value, dict):
                raise ConfigError(f"[{key}] must be a table/dict")
            bad = set(value) - set(DEFAULTS[key])
            if bad:
                raise ConfigError(f"Unknown key(s) in [{key}]: {sorted(bad)}; allowed: {sorted(DEFAULTS[key])}")
            cfg[key].update(value)
        elif key == "sweep":
            cfg["sweep"] = _normalize_sweep(value)
        else:
            cfg[key] = copy.deepcopy(value)
    cfg["fields"] = [_field_with_defaults(f, i) for i, f in enumerate(cfg["fields"])]
    cfg["observables"] = [_observable_with_defaults(o, i) for i, o in enumerate(cfg["observables"])]
    return cfg


def _normalize_sweep(sweep: dict) -> dict:
    """Allow `[sweep]` with plain keys (= product) besides `[sweep.product]` / `[sweep.zip]`."""
    out = {"product": {}, "zip": {}}
    for key, value in sweep.items():
        if key in ("product", "zip"):
            out[key].update(value)
        else:
            out["product"][key] = value
    for kind, table in out.items():
        for path, values in table.items():
            if not isinstance(values, list) or not values:
                raise ConfigError(f"sweep.{kind}.'{path}' must be a non-empty list of values")
    lengths = {len(v) for v in out["zip"].values()}
    if len(lengths) > 1:
        raise ConfigError(f"All sweep.zip lists must have the same length, got {sorted(lengths)}")
    return out


def _field_with_defaults(field: dict, i: int) -> dict:
    ftype = field.get("type")
    if ftype not in FIELD_TYPES:
        raise ConfigError(f"fields[{i}]: unknown type {ftype!r}; available: {sorted(FIELD_TYPES)}")
    spec = FIELD_TYPES[ftype]
    bad = set(field) - set(spec["params"]) - {"type"}
    if bad:
        raise ConfigError(f"fields[{i}] ({ftype}): unknown key(s) {sorted(bad)}; allowed: {sorted(spec['params'])}")
    return {"type": ftype, **spec["defaults"], **field}


def _observable_with_defaults(obs: dict | str, i: int) -> dict:
    if isinstance(obs, str):
        obs = {"name": obs}
    bad = set(obs) - set(OBSERVABLE_KEYS)
    if bad:
        raise ConfigError(f"observables[{i}]: unknown key(s) {sorted(bad)}; allowed: {sorted(OBSERVABLE_KEYS)}")
    if "name" not in obs:
        raise ConfigError(f"observables[{i}] needs a 'name'")
    return {**OBSERVABLE_KEYS, **obs}


# ---------------------------------------------------------------------------------- sweeps
def set_by_path(cfg: dict, path: str, value: Any) -> None:
    """Set e.g. 'system.B0' or 'fields.0.E0' (list indices as integers) in a nested config."""
    keys = path.split(".")
    node: Any = cfg
    for key in keys[:-1]:
        node = node[int(key)] if isinstance(node, list) else node[key]
    last = keys[-1]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        if last not in node:
            raise ConfigError(f"sweep path '{path}': '{last}' is not a valid key")
        node[last] = value


def expand(cfg: dict) -> list[dict]:
    """All individual runs of a configuration: the Cartesian product of the `sweep.product` lists,
    combined with the `sweep.zip` lists, which vary together. Each run is validated."""
    product = cfg["sweep"]["product"]
    zipped = cfg["sweep"]["zip"]
    axes = [[(path, v) for v in values] for path, values in product.items()]
    if zipped:
        n = len(next(iter(zipped.values())))
        axes.append([tuple((path, zipped[path][i]) for path in zipped) for i in range(n)])
    runs = []
    for combo in itertools.product(*axes) if axes else [()]:
        run = copy.deepcopy(cfg)
        run["sweep"] = {"product": {}, "zip": {}}
        swept = {}
        for item in combo:
            pairs = item if isinstance(item[0], tuple) else (item,)
            for path, value in pairs:
                set_by_path(run, path, value)
                swept[path] = value
        run["fields"] = [_field_with_defaults(f, i) for i, f in enumerate(run["fields"])]
        validate(run)
        run["_swept"] = swept
        runs.append(run)
    for i, run in enumerate(runs):
        run["_index"] = i
        run["_run_id"] = f"{i:04d}_{config_hash(run)}"
    return runs


def config_hash(run: dict) -> str:
    """Short hash of everything that determines the physics/numerics of a run."""
    relevant = {k: v for k, v in run.items() if not k.startswith("_") and k not in ("output_dir", "output", "sweep")}
    relevant["method"] = {k: v for k, v in relevant["method"].items() if k != "n_workers"}
    blob = json.dumps(relevant, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:8]


# ------------------------------------------------------------------------------ validation
def validate(run: dict) -> None:
    """Check one (expanded) run for consistency; raises ConfigError with a clear message."""
    s, m, t = run["system"], run["method"], run["time"]
    if t["total_time"] is None:
        raise ConfigError("time.total_time is required")
    if m["mode"] not in ("kspace", "realspace"):
        raise ConfigError("method.mode must be 'kspace' or 'realspace'")
    if m["precision"] not in ("single", "double"):
        raise ConfigError("method.precision must be 'single' or 'double'")
    explicit = [k for k in ("Lx", "Ly", "pbc_x", "pbc_y") if s[k] is not None]
    if s["ribbon"] is not None:
        if s["ribbon"] not in ("armchair", "zigzag"):
            raise ConfigError("system.ribbon must be 'armchair' or 'zigzag' (or omit it and give Lx, Ly, pbc_x, pbc_y)")
        if s["width"] is None:
            raise ConfigError("system.width is required with system.ribbon")
        if explicit:
            raise ConfigError(f"Give either system.ribbon/width or the explicit geometry, not both: {explicit}")
    elif len(explicit) != 4:
        raise ConfigError("Without system.ribbon, give all of system.Lx, Ly, pbc_x, pbc_y")
    if m["mode"] == "kspace":
        pbc = resolved_pbc(s)
        if sum(pbc) != 1:
            raise ConfigError("k-space mode needs a geometry periodic along exactly one axis")
    if s["gauge"] == "length":
        if len(run["fields"]) != 1 or run["fields"][0]["type"] == "elliptical_pulse":
            raise ConfigError("gauge = 'length' supports exactly one (linearly polarized) field")
        if s["B0"] != 0:
            raise ConfigError("gauge = 'length' does not support a magnetic field; use 'velocity'")
    elif s["gauge"] != "velocity":
        raise ConfigError("system.gauge must be 'velocity' or 'length'")
    for i, f in enumerate(run["fields"]):
        missing = [p for p in FIELD_TYPES[f["type"]]["required"] if p not in f]
        if missing:
            raise ConfigError(f"fields[{i}] ({f['type']}): missing {missing}")


def resolved_pbc(system: dict) -> tuple[bool, bool]:
    if system["ribbon"] == "armchair":
        return (False, bool(system["periodic"]))
    if system["ribbon"] == "zigzag":
        return (bool(system["periodic"]), False)
    return (bool(system["pbc_x"]), bool(system["pbc_y"]))


def public_config(run: dict) -> dict:
    """The run's configuration as it should be stored next to the results (JSON-safe)."""
    out = {k: v for k, v in run.items() if not k.startswith("_") and k != "sweep"}
    return json.loads(json.dumps(out, default=str))
