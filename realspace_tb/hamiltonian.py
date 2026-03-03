from abc import ABC, abstractmethod
from . import backend as B
import numpy as np


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

    def ground_state_density_matrix(self, fermi_level: float = 0.0) -> B.Array:
        r"""Return the ground state density matrix.

        Returns :math:`\sum_n |\psi_n\rangle\langle\psi_n|` for all eigenstates
        with :math:`E_n \leq` ``fermi_level``.
        """
        occupied: np.ndarray = (self.eigenvalues <= fermi_level).astype(B.FDTYPE)
        return (self.eigenstates * occupied @ self.eigenstates.T.conj()).astype(
            B.DTYPE, copy=False
        )
