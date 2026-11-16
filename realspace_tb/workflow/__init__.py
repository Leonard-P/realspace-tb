"""
Configuration-driven simulations and parameter sweeps.

Describe a simulation (or a whole sweep) in a TOML/JSON file or a dict, then

    from realspace_tb.workflow import load_config, expand, run_simulation, load_results
    runs = expand(load_config("examples/armchair_sweep.toml"))   # one dict per run
    run_simulation(runs[0])
    results = load_results("results/armchair_sweep")

or from the command line / Slurm (see __main__.py):

    python -m realspace_tb.workflow list  examples/armchair_sweep.toml
    python -m realspace_tb.workflow slurm examples/armchair_sweep.toml --account hpc-prf-metdyn --submit

Several field components in `[[fields]]` are superimposed. Field types and observables can be
extended (build.register_field, observables are discovered by name).
"""

from .config import ConfigError, DEFAULTS, FIELD_TYPES, expand, load_config, set_by_path
from .build import observable_registry, register_field
from .runner import load_results, load_run, run_dir, run_simulation

__all__ = [
    "ConfigError", "DEFAULTS", "FIELD_TYPES", "expand", "load_config", "set_by_path",
    "observable_registry", "register_field",
    "load_results", "load_run", "run_dir", "run_simulation",
]
