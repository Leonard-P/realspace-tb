r"""
Time evolution in k-space: :math:`i\,\dot\rho_\theta = [H(\theta,t), \rho_\theta]` for all k at once.

Parallelism
-----------
* All k-points are propagated together with batched sparse-dense products, so the
  GPU backend (``backend.set_backend(use_gpu=True)``) parallelizes over k automatically.
* On CPU, ``evolve(..., n_workers=n)`` splits the k-mesh into n chunks and evolves them
  in separate processes. k-blocks never couple, so this is exact; observables (which
  are k-averages, i.e. linear in rho) are merged with the chunk weights.
"""

from __future__ import annotations

import copy
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
from tqdm.auto import trange, tqdm

from . import backend as B
from .bloch import BlochHamiltonian
from .hamiltonian import Hamiltonian
from .observable import Observable


# ----------------------------------------------------------------------------- observables
def _child_observables(obs: Observable) -> list[tuple[str, Observable]]:
    return [(k, v) for k, v in vars(obs).items() if isinstance(v, Observable)]


def _is_composite(obs: Observable) -> bool:
    return type(obs).measure is not Observable.measure and bool(_child_observables(obs))


def _rebind_hamiltonian(obs: Observable, H: BlochHamiltonian) -> None:
    """Point observables that were built with the real-space base Hamiltonian at H,
    so the k-resolved path sees the twisted H(theta, t)."""
    if getattr(obs, "_hamiltonian", None) is H.base:
        obs._hamiltonian = H
    for _, child in _child_observables(obs):
        _rebind_hamiltonian(child, H)


class _MeasureContext:
    """Per-measurement cache of the k-averaged matrix (computed at most once per time)."""

    def __init__(self, H: BlochHamiltonian, rho: B.Array, t: float) -> None:
        self.H, self.rho, self.t = H, rho, t
        self._rho_bar = None

    @property
    def rho_bar(self) -> B.Array:
        if self._rho_bar is None:
            self._rho_bar = self.H.k_averaged_density_matrix(self.rho)
        return self._rho_bar

    def value(self, obs: Observable) -> B.Array:
        H, t = self.H, self.t
        if getattr(obs, "bloch_pairwise", False):
            return obs._compute(self.rho_bar, t)  # untwisted H(t) + k-averaged rho
        xp = B.xp()
        w = H.kmesh.weights
        total = None
        for k in range(H.n_k):  # exact for any observable that is linear in rho
            with H.at_k(k):
                v = w[k] * obs._compute(xp.ascontiguousarray(self.rho[:, k, :]), t)
            total = v if total is None else total + v
        return total


def evaluate_observable(obs: Observable, rho: B.Array, H: BlochHamiltonian, t: float = 0.0) -> B.Array:
    """k-averaged (infinite-system) value of a single observable for the Bloch state `rho`
    at time `t`, without time evolution (e.g. to check convergence of the initial state)."""
    _rebind_hamiltonian(obs, H)
    return _MeasureContext(H, rho, t).value(obs)


def _measure(obs: Observable, ctx: _MeasureContext, step: int) -> None:
    if _is_composite(obs):
        for _, child in _child_observables(obs):
            _measure(child, ctx, step)
        return
    if not obs.window.should_measure(ctx.t, step):
        return
    obs._results.append(B.xp().asarray(ctx.value(obs)).copy())
    obs._times.append(ctx.t)


def _collect(obs: Observable) -> dict:
    return {
        "values": obs.values,
        "times": obs.measurement_times,
        "children": {name: _collect(c) for name, c in _child_observables(obs)},
    }


def _weighted_sum(items: list[Any], weights: list[float]) -> Any:
    first = items[0]
    if isinstance(first, dict):
        return {k: _weighted_sum([it[k] for it in items], weights) for k in first}
    if first is None:
        return None
    return sum(w * np.asarray(it) for w, it in zip(weights, items))


def _merge(obs: Observable, results: list[dict], weights: list[float]) -> None:
    obs.values = _weighted_sum([r["values"] for r in results], weights)
    obs.measurement_times = results[0]["times"]
    for name, child in _child_observables(obs):
        _merge(child, [r["children"][name] for r in results], weights)


# ----------------------------------------------------------------------------- worker

def _own_arrays(obj, _seen: set | None = None):
    """Give every numpy array reachable from `obj` its own memory.

    Arrays received by a worker process are unpickled as views of an immutable `bytes` buffer.
    Some operations (e.g. scipy sparse fancy indexing, which re-broadcasts index arrays) fail on
    such views with "cannot set WRITEABLE flag to True of this array". Copying them avoids that.
    """
    _seen = set() if _seen is None else _seen
    if id(obj) in _seen:
        return obj
    _seen.add(id(obj))
    if isinstance(obj, np.ndarray):
        return obj.copy() if isinstance(obj.base, (bytes, memoryview)) or not obj.flags.writeable else obj
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            obj[i] = _own_arrays(v, _seen)
        return obj
    if isinstance(obj, tuple):
        return type(obj)(_own_arrays(v, _seen) for v in obj) if not hasattr(obj, "_fields") else obj
    if isinstance(obj, dict):
        for k in list(obj):
            obj[k] = _own_arrays(obj[k], _seen)
        return obj
    attrs = getattr(obj, "__dict__", None)
    if isinstance(attrs, dict) and not isinstance(obj, type):
        for k, v in list(attrs.items()):
            new = _own_arrays(v, _seen)
            if new is not v:
                try:
                    setattr(obj, k, new)
                except AttributeError:
                    attrs[k] = new
    return obj


def _evolve_chunk(args: tuple) -> tuple[list[dict], np.ndarray, dict]:
    t_begin = time.time()   # the worker has started and imported everything
    solver, rho, H, dt, total_time, rho_0, tau, observables, precision = args
    # worker processes start with a fresh import of the package, i.e. with the default
    # backend (CPU, double): restore the precision chosen in the main process
    B.set_backend(use_gpu=False, precision=precision)
    rho, H, rho_0, observables = _own_arrays([rho, H, rho_0, observables])
    try:  # avoid oversubscription: one BLAS thread per worker process
        from threadpoolctl import threadpool_limits

        limiter = threadpool_limits(1)
    except ImportError:
        limiter = None
    t_loop = time.time()
    solver._evolve_serial(rho, H, dt, total_time, rho_0, tau, observables, progress=False)
    t_end = time.time()
    del limiter
    timing = {"begin": t_begin, "loop_s": t_end - t_loop, "end": t_end, "n_k": H.n_k}
    return [_collect(o) for o in observables], rho, timing


# ----------------------------------------------------------------------------- solver
class BlochRK4NeumannSolver:
    r"""RK4 integration of :math:`i\dot\rho_\theta = [H(\theta,t),\rho_\theta] + i(\rho_{0,\theta}-\rho_\theta)/\tau`
    for all k-points of a :class:`BlochHamiltonian`.

    The state is a Bloch density matrix of shape ``(N_s, N_k, N_s)`` (see ``bloch.py``),
    e.g. from ``H.thermal_state_density_matrix(...)``. It is evolved in place.
    Observables are the existing real-space observables; they return k-averaged values,
    i.e. the values of the infinite system.
    """

    def _derivative(self, t, rho, H: BlochHamiltonian, rho_0, tau) -> B.Array:
        X = H.apply(rho, t)  # H_k rho_k for all k
        # -i[H, rho] = i(X^dagger - X) (rho Hermitian); per k, dagger swaps axes 0 and 2
        d = X.conj().transpose(2, 1, 0) - X
        d *= 1j
        if tau != float("inf") and rho_0 is not None:
            d += (rho_0 - rho) / tau
        return d

    def _step(self, t, rho, H, dt, rho_0=None, tau=float("inf")) -> None:
        # classic RK4, written to minimize temporaries (the update is memory-bound)
        k = self._derivative(t, rho, H, rho_0, tau)
        acc = k.copy()
        tmp = rho + (0.5 * dt) * k
        k = self._derivative(t + 0.5 * dt, tmp, H, rho_0, tau)
        acc += 2.0 * k
        tmp = rho + (0.5 * dt) * k
        k = self._derivative(t + 0.5 * dt, tmp, H, rho_0, tau)
        acc += 2.0 * k
        tmp = rho + dt * k
        k = self._derivative(t + dt, tmp, H, rho_0, tau)
        acc += k
        acc *= dt / 6.0
        rho += acc
        # keep every k-block Hermitian (rho.conj() is a copy, so no aliasing)
        rho += rho.conj().transpose(2, 1, 0)
        rho *= 0.5

    def _evolve_serial(self, rho, H, dt, total_time, rho_0, tau, observables, progress) -> None:
        n_steps = int(total_time / dt)
        if tau != float("inf") and rho_0 is None:
            rho_0 = rho.copy()
        for obs in observables:
            _rebind_hamiltonian(obs, H)
        for step in trange(n_steps, disable=not progress):
            t = step * dt
            if observables:
                ctx = _MeasureContext(H, rho, t)
                for obs in observables:
                    _measure(obs, ctx, step)
            self._step(t, rho, H, dt, rho_0, tau)
        for obs in observables:
            obs.finalize()

    def evolve(
        self,
        rho: B.Array,
        H: BlochHamiltonian,
        dt: float,
        total_time: float,
        rho_0: B.Array | None = None,
        tau: float = float("inf"),
        observables: list[Observable] | None = None,
        progress: bool = True,
        n_workers: int = 1,
    ) -> None:
        """Evolve the Bloch state; same arguments as ``RK4NeumannSolver.evolve`` plus `n_workers`.

        n_workers > 1 (CPU backend only): the k-points are split into `n_workers` chunks that
        run in separate processes. The returned observables are merged k-averages; ``rho`` is
        updated in place with the final state of every chunk.
        """
        if not isinstance(H, BlochHamiltonian):
            raise TypeError("H must be a BlochHamiltonian (use RK4NeumannSolver in real space).")
        observables = list(observables or [])
        n_workers = max(1, min(int(n_workers), H.n_k))
        if n_workers == 1:
            t0 = time.time()
            self._evolve_serial(rho, H, dt, total_time, rho_0, tau, observables, progress)
            self.last_timing = {"n_workers": 1, "worker_start_s": [0.0],
                                "worker_loop_s": [time.time() - t0], "parallel_phase_s": time.time() - t0}
            return
        if B.USE_GPU:
            raise ValueError("n_workers > 1 is for the CPU backend; on GPU all k run in one batch.")

        precision = "single" if np.dtype(B.DTYPE) == np.complex64 else "double"
        chunks = [c for c in np.array_split(np.arange(H.n_k), n_workers) if c.size]
        jobs, weights = [], []
        for idx in chunks:
            H_c, w_c = H.subset(idx)
            # deep copy observables, re-pointing references to H at the chunk Hamiltonian
            obs_c = copy.deepcopy(observables, memo={id(H): H_c, id(H.base): H.base})
            rho_c = np.ascontiguousarray(rho[:, idx, :])
            rho0_c = None if rho_0 is None else np.ascontiguousarray(rho_0[:, idx, :])
            jobs.append((self, rho_c, H_c, dt, total_time, rho0_c, tau, obs_c, precision))
            weights.append(w_c)

        # 'forkserver' (Linux) / 'spawn': forking a multi-threaded process can deadlock
        methods = mp.get_all_start_methods()
        ctx = mp.get_context("forkserver" if "forkserver" in methods else "spawn")
        with ProcessPoolExecutor(max_workers=len(jobs), mp_context=ctx) as pool:
            t_submit = time.time()
            futures = [pool.submit(_evolve_chunk, job) for job in jobs]
            outputs = [f.result() for f in tqdm(futures, disable=not progress, desc="k-chunks")]
            t_done = time.time()

        # timing of the parallel phase, e.g. to diagnose slow nodes (see last_timing)
        loops = [tm["loop_s"] for _, _, tm in outputs]
        self.last_timing = {
            "n_workers": len(jobs),
            "worker_start_s": [tm["begin"] - t_submit for _, _, tm in outputs],  # process start + imports
            "worker_loop_s": loops,                                              # RK4 steps + measurements
            "parallel_phase_s": t_done - t_submit,
        }
        for idx, (_, rho_c, _) in zip(chunks, outputs):
            rho[:, idx, :] = rho_c  # final state of every chunk back into rho
        for i, obs in enumerate(observables):
            _merge(obs, [res[i] for res, _, _ in outputs], weights)
