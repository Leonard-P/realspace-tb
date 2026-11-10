from abc import ABC, abstractmethod
from . import backend as B
import numpy as np


def fermi_occupation(
    energies: "B.Array",
    fermi_level: float = 0.0,
    beta: float = float("inf"),
    tol: "float | None" = None,
    matrix_size: "int | None" = None,
) -> "B.Array":
    r"""Grand-canonical occupation numbers $f(E)$ for the given eigenvalues.

    Finite `beta`: Fermi function $1/(e^{\beta(E-\mu)}+1)$, evaluated stably as
    $\tfrac12[1-\tanh(\beta(E-\mu)/2)]$ (no overflow, no clipping).

    `beta` = inf: the limit $T \to 0$ of the Fermi function,
    $f = 1$ for $E < \mu - \mathrm{tol}$, $f = 1/2$ for $|E-\mu| \le \mathrm{tol}$,
    $f = 0$ for $E > \mu + \mathrm{tol}$. Every state of a degenerate level at $\mu$
    gets the same weight, so $\rho = f(H)$ is a function of $H$ (independent of the
    eigenbasis returned by the eigensolver, and symmetry-preserving).
    If `tol` is None it defaults to $10\,n\,\epsilon\,\max|E|$ with $n$ = `matrix_size`
    (default: size of the last axis), a bound on the eigensolver's rounding error.
    """
    if beta < 0:
        raise ValueError("beta must be non-negative")
    xp = B.xp()
    energies = xp.real(xp.asarray(energies))
    delta = energies - fermi_level
    if np.isinf(beta):
        if tol is None:
            n = matrix_size if matrix_size is not None else energies.shape[-1]
            scale = float(xp.abs(energies).max()) if energies.size else 0.0
            tol = 10 * n * float(xp.finfo(energies.dtype).eps) * (scale if scale > 0 else 1.0)
        occ = xp.where(xp.abs(delta) <= tol, 0.5, (delta < 0).astype(B.FDTYPE))
        return occ.astype(B.FDTYPE)
    return (0.5 * (1.0 - xp.tanh(0.5 * beta * delta))).astype(B.FDTYPE)


class Hamiltonian(ABC):
    def __init__(self) -> None:
        self._eigenvalues: "B.Array | None" = None
        self._eigenstates: "B.Array | None" = None

    @abstractmethod
    def at_time(self, t: float) -> B.SparseArray:
        """Return the Hamiltonian at time t."""
        ...

    def _ensure_eigens(self) -> None:
        """Compute and cache eigenvalues/eigenstates (only once)."""
        if self._eigenvalues is None:
            print("Calculating eigenstates at t=0...")
            self._eigenvalues, self._eigenstates = B.xp().linalg.eigh(
                self.at_time(0.0).toarray()
            )

    @property
    def eigenvalues(self) -> B.Array:
        """Return the eigenvalues of the Hamiltonian at t=0."""
        self._ensure_eigens()
        return self._eigenvalues  # type: ignore[return-value]

    @property
    def eigenstates(self) -> B.Array:
        """Return the eigenstates of the Hamiltonian at t=0."""
        self._ensure_eigens()
        return self._eigenstates  # type: ignore[return-value]

    def ground_state_density_matrix(
        self, fermi_level: float = 0.0, tol: "float | None" = None
    ) -> B.Array:
        r"""Return the grand-canonical density matrix in the limit T -> 0.

        Returns $\sum_n f(E_n)\,|\psi_n\rangle\langle\psi_n|$ with $\mu$ = `fermi_level` and
        $f = 1$ for $E_n < \mu - \mathrm{tol}$, $f = 1/2$ for $|E_n - \mu| \le \mathrm{tol}$,
        $f = 0$ for $E_n > \mu + \mathrm{tol}$ (see `fermi_occupation`).
        """
        occupied = fermi_occupation(self.eigenvalues, fermi_level, float("inf"), tol)
        return (self.eigenstates * occupied @ self.eigenstates.T.conj()).astype(
            B.DTYPE, copy=False
        )

    def thermal_state_density_matrix(
        self,
        fermi_level: float = 0.0,
        beta: float = float("inf"),
        tol: "float | None" = None,
    ) -> B.Array:
        r"""Return the grand-canonical density matrix $f(H)$ at inverse temperature `beta`.

        `beta` = inf is the limit T -> 0: states within `tol` of the Fermi level
        get occupation 1/2 (see `fermi_occupation`).
        """
        if beta < 0:
            raise ValueError("beta must be non-negative")
        if np.isinf(beta):
            return self.ground_state_density_matrix(fermi_level, tol=tol)
        weights = fermi_occupation(self.eigenvalues, fermi_level, beta)
        return (self.eigenstates * weights @ self.eigenstates.T.conj()).astype(
            B.DTYPE, copy=False
        )
