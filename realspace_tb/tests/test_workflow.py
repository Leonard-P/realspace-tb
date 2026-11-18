"""Tests for realspace_tb.workflow: configuration, sweeps, runs, results, command line."""
import json
from pathlib import Path

import numpy as np
import pytest

import realspace_tb as tb
import realspace_tb.orbitronics_2d as orb
from realspace_tb.workflow import ConfigError, expand, load_config, load_results, run_simulation
from realspace_tb.workflow.__main__ import main as cli

EXAMPLES = Path(__file__).resolve().parents[1] / "workflow" / "examples"


def small_config(tmp_path, **overrides):
    cfg = {
        "name": "t", "output_dir": str(tmp_path),
        "system": {"ribbon": "armchair", "width": 9, "B0": -0.05},          # atomic units throughout
        "method": {"n_k": 8},
        "time": {"dt": 0.5, "total_time": 40.0},
        "fields": [
            {"type": "pulse", "E0": 1e-4, "t_c": 20.0, "sigma": 5.0, "omega": 0.11},
            {"type": "ramped_dc", "E0": 1e-5, "T_ramp": 10.0, "direction": "transverse"},
        ],
        "observables": [{"name": "vorticity", "stride": 2}, {"name": "bond_current"}],
    }
    for k, v in overrides.items():
        cfg[k] = v
    return cfg


@pytest.mark.parametrize("example", sorted(EXAMPLES.glob("*.toml")))
def test_examples_load_and_expand(example):
    runs = expand(load_config(example))
    assert len(runs) >= 1
    assert len({r["_run_id"] for r in runs}) == len(runs)


def test_sweep_product_and_zip():
    runs = expand(load_config(EXAMPLES / "ribbons_pm_sweep.toml"))
    assert len(runs) == 8
    combos = {(r["system"]["ribbon"], r["system"]["width"], r["system"]["B0"], r["fields"][0]["E0"]) for r in runs}
    assert len(combos) == 8
    assert all((rb, w) in {("armchair", 17), ("zigzag", 10)} for rb, w, _, _ in combos)


@pytest.mark.parametrize("bad, match", [
    ({"system": {"widht": 17}}, "Unknown key"),
    ({"fields": [{"type": "pulse", "E0": 1, "t_c": 1, "sigma": 1, "omega": 1, "sigam": 2}]}, "unknown key"),
    ({"fields": [{"type": "laser"}]}, "unknown type"),
    ({"sweep": {"zip": {"system.B0": [1, 2], "system.width": [3]}}}, "same length"),
    ({"sweep": {"product": {"system.nonsense": [1]}}}, "not a valid key"),
    ({"observables": [{"name": "vortcity"}]}, "unknown observable"),
    ({"units": "scale"}, "atomic units"),
    ({"method": {"gpu": "False"}}, "true/false without quotes"),
    ({"time": {"dt": "0.1", "total_time": 4.0}}, "time.dt must be float"),
    ({"method": {"k_shift": 1.0}}, "k_shift"),
    ({"fields": {"type": "pulse", "E0": 1e-5, "t_c": 1.0, "sigma": 1.0, "omega": 0.1}}, "double brackets"),
])
def test_invalid_configs_are_rejected(tmp_path, bad, match):
    cfg = small_config(tmp_path)
    cfg.update(bad)
    with pytest.raises(ConfigError, match=match):
        expand(load_config(cfg))          # everything is checked before anything runs


def test_workflow_equals_manual_construction(tmp_path):
    """Superimposed fields in k-space, inputs in atomic units: identical to building everything by hand."""
    out = run_simulation(expand(load_config(small_config(tmp_path)))[0])
    data = np.load(out / "observables.npz")

    geo = orb.HoneycombLatticeGeometry(Lx=9, Ly=2, a_nn=2.6834, pbc_y=True)
    fields = [orb.LightPulseComponent(E0=1e-4, t_c=20.0, sigma=5.0, omega=0.11,
                                      phase_shift=0.0, direction=np.array([0.0, 1.0])),
              orb.RampedConstantFieldAmplitude(E0=1e-5, T_ramp=10.0, direction=np.array([1.0, 0.0]))]
    H = tb.BlochHamiltonian(orb.LinearFieldHamiltonianPeierls(geo, -0.0992, fields, B0=-0.05,
                                                              gauge_origin="center"), tb.KMesh(8))
    rho = H.thermal_state_density_matrix()
    v = orb.observables.VorticityObservable(geo, window=tb.MeasurementWindow(stride=2), hamiltonian=H)
    eq = tb.evaluate_observable(v, rho, H, 0.0)
    tb.BlochRK4NeumannSolver().evolve(rho, H, 0.5, 40.25, observables=[v], progress=False)
    assert np.allclose(data["vorticity"], v.values, atol=1e-14)
    assert np.allclose(data["vorticity_equilibrium"], eq, atol=1e-14)
    assert len(data["vorticity__times"]) == 40 and len(data["bond_current__times"]) == 80


def test_realspace_flake_with_composite_observable(tmp_path):
    cfg = small_config(tmp_path, observables=[{"name": "lattice_frame", "stride": 4}])
    cfg["system"] = {"Lx": 8, "Ly": 4, "pbc_x": False, "pbc_y": False, "B0": -0.05}
    cfg["method"] = {"mode": "realspace"}
    cfg["fields"] = [{"type": "pulse", "E0": 1e-4, "t_c": 20.0, "sigma": 5.0, "omega": 0.11, "direction": "x"}]
    out = run_simulation(expand(load_config(cfg))[0])
    data = np.load(out / "observables.npz")
    assert {"lattice_frame__densities", "lattice_frame__currents", "lattice_frame__current_vorts",
            "lattice_frame_equilibrium__densities"} <= set(data.files)


def test_skip_existing_and_load_results(tmp_path):
    runs = expand(load_config(small_config(tmp_path, sweep={"product": {"system.B0": [0.0, -0.05]}})))
    dirs = [run_simulation(r) for r in runs]
    stamp = (dirs[0] / "observables.npz").stat().st_mtime_ns
    run_simulation(runs[0])                                       # skipped: same config, results exist
    assert (dirs[0] / "observables.npz").stat().st_mtime_ns == stamp
    res = load_results(Path(tmp_path) / "t")
    assert [r["config"]["system"]["B0"] for r in res] == [0.0, -0.05]
    assert res[1]["metadata"]["swept"] == {"system.B0": -0.05}


def test_command_line(tmp_path, capsys):
    cfg_path = tmp_path / "c.json"
    cfg_path.write_text(json.dumps(small_config(tmp_path, sweep={"system.B0": [0.0, -0.05, 0.05]})))
    cli(["count", str(cfg_path)])
    assert capsys.readouterr().out.strip() == "3"
    cli(["run", str(cfg_path), "--index", "1"])
    cli(["list", str(cfg_path)])
    lines = capsys.readouterr().out.strip().splitlines()
    assert [ln.split()[2] for ln in lines[-3:]] == ["todo", "done", "todo"]
    script = tmp_path / "job.slurm"
    cli(["slurm", str(cfg_path), "--account", "hpc-prf-metdyn", "--cpus", "8", "--max-parallel", "2",
         "--env", str(tmp_path / "cluster_env.sh"), "--log-dir", str(tmp_path / "logs"), "--script", str(script)])
    text = script.read_text()
    assert "#SBATCH --array=0-2%2" in text and "#SBATCH -A hpc-prf-metdyn" in text
    assert 'srun "$PYTHON" -m realspace_tb.workflow run' in text


def test_estimate_counts_and_sizes(tmp_path, capsys):
    """The estimate predicts exactly the number and size of the stored measurements."""
    from realspace_tb.workflow.estimate import estimate
    cfg = small_config(tmp_path)
    cfg["observables"] = [{"name": "vorticity", "stride": 3, "start": 10.0}, {"name": "vort_flux", "stride": 2}]
    cfg["method"]["n_workers"] = 2
    run = expand(load_config(cfg))[0]
    est = estimate(dict(run))
    out = run_simulation(expand(load_config(cfg))[0])
    data = np.load(out / "observables.npz")
    for name in ("vorticity", "vort_flux"):
        assert est["observables"][name]["measurements"] == len(data[name + "__times"])
        assert est["observables"][name]["bytes_per_measurement"] == data[name][0].nbytes
    assert est["recommended_mem_GB"] >= 1 and est["time_loop_s"] > 0
    cli(["estimate", str(tmp_path / "c.json") if False else _write(tmp_path, cfg), "--cpus", "2"])
    assert "for all runs: --mem=" in capsys.readouterr().out


def _write(tmp_path, cfg):
    p = tmp_path / "e.json"
    p.write_text(json.dumps(cfg))
    return str(p)


def test_quoted_inf_is_accepted(tmp_path):
    runs = expand(load_config(small_config(tmp_path, sweep={"state.beta": ["inf", 2000.0]})))
    assert runs[0]["state"]["beta"] == float("inf") and runs[1]["state"]["beta"] == 2000.0


def test_every_observable_in_parallel_workers(tmp_path):
    """Every registered observable must work inside worker processes and agree with a serial run."""
    from realspace_tb.workflow import observable_registry
    names = sorted(observable_registry())
    results = {}
    for n_workers in (1, 2):
        cfg = small_config(tmp_path, name=f"all_{n_workers}",
                           observables=[{"name": n, "stride": 5} for n in names])
        cfg["method"]["n_workers"] = n_workers
        cfg["time"]["total_time"] = 10.0
        out = run_simulation(expand(load_config(cfg))[0])
        with np.load(out / "observables.npz") as d:
            results[n_workers] = {k: d[k] for k in d.files}
    assert set(results[1]) == set(results[2])
    for key in results[1]:
        assert np.allclose(results[1][key], results[2][key], atol=1e-12), key
