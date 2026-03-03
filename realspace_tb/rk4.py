from typing import Callable, List
from . import backend as B
from scipy.sparse import csr_matrix
from tqdm.auto import trange
from .hamiltonian import Hamiltonian
from .observable import Observable


class RK4NeumannSolver:
    """Implements the RK4 integration method. Currently support the basic case of von Neumann equation with simple decay term and a scalarly time-dependent Hamiltonian."""

    def _time_evolution_derivative(
        self,
        t: float,
        rho: B.Array,
        H: B.SparseArray,
        rho_0: "B.Array | None" = None,
        tau: float = float("inf"),
    ) -> B.Array:
        """
        Calculate the time derivative of the density matrix from damped Neumann equation:
        ∂ρ/∂t = -i[H, ρ] + (ρ₀ - ρ) / τ

        Parameters:
            t: Current time.
            rho: Density matrix ρ(t) (dense).
            H: Hamiltonian (sparse).
            rho_0: Initial density matrix ρ₀ (dense), used for decay term.
            tau: Relaxation time constant τ. Default is infinity (no decay).

        Returns:
            The time derivative of the density matrix (dense).
        """

        # Calculate [H, ρ]
        d_rho_dt = H @ rho - rho @ H  # TODO check if sparse multiplication works
        # else:
        #     d_rho_dt = H.dot(rho) - csr_matrix.dot(rho, H)
        d_rho_dt *= -1j

        if tau != float("inf") and rho_0 is not None:
            d_rho_dt += (rho_0 - rho) / tau

        return d_rho_dt

    def _rk4_step(
        self,
        t: float,
        rho: B.Array,
        H: Hamiltonian,
        dt: float,
        rho_0: "B.Array | None" = None,
        tau: float = float("inf"),
    ) -> None:
        """
        Perform a single Runge-Kutta 4th order step to evolve the density matrix (in-place).

        Parameters:
            t: Current time.
            rho: Density matrix ρ(t) (dense).
            H: Hamiltonian (sparse).
            dt: Time step interval.
            rho_0: Initial density matrix for the decay term.
            tau: Decay time constant τ.

        Returns:
            None. The density matrix is evolved in-place one RK4 step.
        """
        k1 = dt * self._time_evolution_derivative(t, rho, H.at_time(t), rho_0, tau)

        k2 = dt * self._time_evolution_derivative(
            t + 0.5 * dt, rho + 0.5 * k1, H.at_time(t + 0.5 * dt), rho_0, tau
        )
        k3 = dt * self._time_evolution_derivative(
            t + 0.5 * dt, rho + 0.5 * k2, H.at_time(t + 0.5 * dt), rho_0, tau
        )
        k4 = dt * self._time_evolution_derivative(t + dt, rho + k3, H.at_time(t + dt), rho_0, tau)

        rho += (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Ensure hermiticity
        rho[:] = (rho + rho.T.conj()) / 2.0

    def evolve(
        self,
        rho: B.Array,
        H: Hamiltonian,
        dt: float,
        total_time: float,
        rho_0: "B.Array | None" = None,
        tau: float = float("inf"),
        observables: list[Observable] | None = None,
        progress: bool = True,
    ) -> None:
        """
        Evolve the density matrix using the RK4 method.

        Parameters:
            rho: Initial density matrix ρ(t=0) (dense).
            H: Hamiltonian (sparse).
            dt: Time step interval.
            total_time: Total simulation time.
            rho_0: Initial density matrix for the decay term. If None, rho at t=0 is used.
            tau: Decay time constant τ.
            observables: List of observables to measure at each step.

        Returns:
            None. The density matrix is evolved in-place over the total time. Measurements are handled by Observables.
        """
        n_steps = int(total_time / dt)

        if tau != float("inf") and rho_0 is None:
            rho_0 = rho.copy()

        for step in trange(n_steps, disable=not progress):
            t = step * dt

            for observable in observables or []:
                observable.measure(rho, t, step)

            self._rk4_step(t, rho, H, dt, rho_0, tau)

        for observable in observables or []:
            observable.finalize()
