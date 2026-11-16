"""
Builders: turn one validated run configuration (see config.py) into package objects.

Extending: add a field type with `register_field("name", builder)` plus an entry in
config.FIELD_TYPES, and observables are found automatically by name (see `observable_registry`).
"""

from __future__ import annotations

import inspect
import os
import re
from typing import Any, Callable

import numpy as np

from .. import backend as B
from ..bloch import BlochHamiltonian, KMesh
from ..observable import MeasurementWindow, Observable
from .. import orbitronics_2d as orb
from .config import ConfigError, resolved_pbc


# ----------------------------------------------------------------------------- geometry
def build_geometry(run: dict):
    s = run["system"]
    if s["lattice"] != "honeycomb":
        raise ConfigError("Only system.lattice = 'honeycomb' is available so far")
    pbc_x, pbc_y = resolved_pbc(s)
    if s["ribbon"] == "armchair":
        Lx, Ly = s["width"], 2 * s["cells"]
    elif s["ribbon"] == "zigzag":
        Lx, Ly = 4 * s["cells"], s["width"]
    else:
        Lx, Ly = s["Lx"], s["Ly"]
    return orb.HoneycombLatticeGeometry(Lx=Lx, Ly=Ly, a_nn=s["a_nn"], pbc_x=pbc_x, pbc_y=pbc_y)


def ribbon_axis(run: dict) -> int | None:
    """0 (x) or 1 (y): the axis along the ribbon; None if not defined."""
    s = run["system"]
    if s["ribbon"] == "armchair":
        return 1
    if s["ribbon"] == "zigzag":
        return 0
    pbc = resolved_pbc(s)
    return pbc.index(True) if sum(pbc) == 1 else None


def resolve_direction(run: dict, direction: Any) -> np.ndarray:
    """'periodic' / 'transverse' (relative to the ribbon), 'x' / 'y', or an explicit [dx, dy]."""
    if isinstance(direction, str):
        if direction in ("x", "y"):
            return np.array([1.0, 0.0]) if direction == "x" else np.array([0.0, 1.0])
        axis = ribbon_axis(run)
        if axis is None:
            raise ConfigError(f"direction '{direction}' needs a ribbon axis; give 'x', 'y' or [dx, dy]")
        along = np.eye(2)[axis]
        if direction == "periodic":
            return along
        if direction == "transverse":
            return along[::-1].copy()
        raise ConfigError(f"unknown direction {direction!r}")
    vec = np.asarray(direction, dtype=float)
    if vec.shape != (2,):
        raise ConfigError(f"direction must be a 2-vector, got {direction!r}")
    return vec


# ------------------------------------------------------------------------------- fields
FieldBuilder = Callable[[dict, dict], list]
_FIELD_BUILDERS: dict[str, FieldBuilder] = {}


def register_field(name: str, builder: FieldBuilder) -> None:
    """builder(run, field_dict) -> list of HomogeneousFieldAmplitude objects (atomic units)."""
    _FIELD_BUILDERS[name] = builder


register_field("pulse", lambda run, f: [orb.LightPulseComponent(
    E0=f["E0"], t_c=f["t_c"], sigma=f["sigma"], omega=f["omega"], phase_shift=f["cep"],
    direction=resolve_direction(run, f["direction"]))])
register_field("elliptical_pulse", lambda run, f: list(orb.EllipticalPulseFactory.create(
    E0=f["E0"], t_c=f["t_c"], sigma=f["sigma"], varepsilon=f["ellipticity"], omega=f["omega"],
    phase_shift=f["cep"], vartheta=f["vartheta"], varphi=f["varphi"])))
register_field("ramped_ac", lambda run, f: [orb.RampedACFieldAmplitude(
    E0=f["E0"], omega=f["omega"], T_ramp=f["T_ramp"], direction=resolve_direction(run, f["direction"]))])
register_field("ramped_dc", lambda run, f: [orb.RampedConstantFieldAmplitude(
    E0=f["E0"], T_ramp=f["T_ramp"], direction=resolve_direction(run, f["direction"]))])
register_field("delta_kick", lambda run, f: [orb.DeltaKickFieldAmplitude(
    E0=f["E0"], direction=resolve_direction(run, f["direction"]))])


def build_fields(run: dict) -> list:
    """All field components of the run; they are superimposed by the Hamiltonian."""
    components = []
    for f in run["fields"]:
        if f["type"] not in _FIELD_BUILDERS:
            raise ConfigError(f"no builder registered for field type {f['type']!r}")
        components += _FIELD_BUILDERS[f["type"]](run, f)
    return components


def final_vector_potential(fields: list, t: float) -> np.ndarray:
    """-A(t) = integral of E up to t (atomic units): nonzero after a pulse means a residual drift."""
    total = np.zeros(2)
    for comp in fields:
        total += float(comp.integrate_to_time(t)) * np.asarray(comp.direction, dtype=float)
    return total


# --------------------------------------------------------------------------- hamiltonian
def build_hamiltonian(run: dict, geometry, fields: list):
    """Real-space Hamiltonian, and its k-space version if method.mode == 'kspace'."""
    s = run["system"]
    origin = s["gauge_origin"]
    if isinstance(origin, list):
        origin = tuple(origin)
    if s["gauge"] == "length":
        H_real = orb.LinearFieldHamiltonian(geometry, s["t_hop"], fields[0])
    else:
        H_real = orb.LinearFieldHamiltonianPeierls(geometry, s["t_hop"], fields, B0=s["B0"], gauge_origin=origin)
    if run["method"]["mode"] == "realspace":
        return H_real, H_real
    m = run["method"]
    return H_real, BlochHamiltonian(H_real, KMesh(m["n_k"], shift=m["k_shift"]))


def initial_state(run: dict, H):
    st = run["state"]
    return H.thermal_state_density_matrix(fermi_level=st["fermi_level"], beta=st["beta"], tol=st["tol"])


def n_workers(run: dict) -> int:
    if run["method"]["gpu"]:          # on the GPU all k-points run in one batch
        return 1
    value = run["method"]["n_workers"]
    if value == "auto":
        return int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    return int(value)


# --------------------------------------------------------------------------- observables
def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def observable_registry() -> dict[str, type]:
    """All observable classes of orbitronics_2d.observables, by snake_case name without
    'Observable': VortFluxObservable -> 'vort_flux', SiteDensityObservable -> 'site_density'."""
    reg = {}
    for cname, cls in vars(orb.observables).items():
        if inspect.isclass(cls) and issubclass(cls, Observable) and cls is not Observable and cname.endswith("Observable"):
            reg[_snake(cname[: -len("Observable")])] = cls
    return reg


def build_observables(run: dict, geometry, H) -> dict[str, Observable]:
    reg = observable_registry()
    out = {}
    for spec in run["observables"]:
        name = spec["name"]
        if name not in reg:
            raise ConfigError(f"unknown observable {name!r}; available: {sorted(reg)}")
        if name in out:
            raise ConfigError(f"observable {name!r} listed twice")
        window = MeasurementWindow(start_time=spec["start"], end_time=spec["end"], stride=int(spec["stride"]))
        cls = reg[name]
        params = inspect.signature(cls.__init__).parameters
        kwargs: dict[str, Any] = {"window": window}
        if "geometry" in params:
            kwargs["geometry"] = geometry
        if "hamiltonian" in params:
            kwargs["hamiltonian"] = H
        out[name] = cls(**kwargs)
    return out


def set_backend(run: dict) -> None:
    m = run["method"]
    B.set_backend(use_gpu=bool(m["gpu"]), precision=m["precision"])
