from ..hamiltonian import Hamiltonian
from .lattice_2d_geometry import Lattice2DGeometry
from .. import backend as B
from abc import ABC, abstractmethod
import numpy as np


def _build_hopping_csr(geometry: Lattice2DGeometry, dtype: type) -> "B.SparseArray":
    """Build a sparse -1 nearest-neighbor hopping matrix from geometry."""
    size = geometry.Lx * geometry.Ly
    nn = geometry.nearest_neighbors  # (E, 2)
    rows = np.concatenate([nn[:, 0], nn[:, 1]])
    cols = np.concatenate([nn[:, 1], nn[:, 0]])
    data = B.xp().full(len(rows), -1.0, dtype=dtype)
    return (
        B.xp_sparse()
        .coo_matrix(
            (data, (B.xp().array(rows), B.xp().array(cols))),
            shape=(size, size),
        )
        .tocsr()
    )


2


class HomogeneousFieldAmplitude(ABC):
    """Abstract base class for homogeneous electric field with only scalar time dependence."""

    @abstractmethod
    def at_time(self, time: "float | B.Array") -> "float | B.Array":
        """Return the electric field amplitude at a given time."""
        ...

    def integrate_to_time(self, time: "float | B.Array") -> "float | B.Array":
        """Return the electric field amplitude integrated from 0 to a given time.
        Needed for Peierls substitution phase factor.
        """
        raise NotImplementedError(
            "Integration not implemented for this field amplitude."
        )

    direction: B.FCPUArray = np.zeros(2)


class RampedACFieldAmplitude(HomogeneousFieldAmplitude):
    """
    Electric field amplitude ramping over time:
    E(t) = E0 * sin^2(pi * t / 2 * T_ramp) * sin(ω t), capped at E0.
    """

    def __init__(self, E0: float, omega: float, T_ramp: float, direction: B.FCPUArray):
        self.E0 = B.FDTYPE(E0)
        self.omega = B.FDTYPE(omega)
        self.T_ramp = B.FDTYPE(T_ramp)
        self.direction = B.FDTYPE(direction)

    def at_time(self, t: "float | B.Array") -> "float | B.Array":
        # TODO maybe make it CPU only and move backend transfer to Hamiltonian class
        # -> avoid confusion and minimize code scope of GPU backend
        xp = B.xp()
        if xp.isscalar(t):
            if t < self.T_ramp:
                ramp = xp.sin(np.pi * t / (2 * self.T_ramp)) ** 2
            else:
                ramp = 1.0
            return self.E0 * ramp * xp.sin(self.omega * t)

        ramp = xp.where(
            t < self.T_ramp,
            xp.sin(xp.pi * t / (2 * self.T_ramp)) ** 2,
            xp.ones_like(t, dtype=B.FDTYPE),
        )
        return self.E0 * ramp * xp.sin(self.omega * t)

    def integrate_to_time(self, t: "float | B.Array") -> "float | B.Array":
        """Integrate the field amplitude from time 0 to t. Needed for Peierls substitution."""
        xp = B.xp()
        w = self.omega
        T = self.T_ramp
        pi = xp.pi
        if xp.isscalar(t):
            integral = 0.0
            s = t if t < T else T
            if t > 0.0 and T > 0.0:
                integral += (
                    (T**2 * w**2 - pi * T * w) * xp.cos((s * T * w + pi * s) / T)
                    + (T**2 * w**2 + pi * T * w) * xp.cos((s * T * w - pi * s) / T)
                    + (2 * pi**2 - 2 * T**2 * w**2) * xp.cos(w * s)
                    - 2 * pi**2
                )
                integral /= 4 * w * (T**2 * w**2 - pi**2)
            if t > T:
                integral += (xp.cos(w * T) - xp.cos(w * t)) / w
            return self.E0 * integral

        raise NotImplementedError(
            "Integration of array time inputs not implemented yet."
        )


class LinearFieldHamiltonian(Hamiltonian):
    def __init__(
        self, geometry: Lattice2DGeometry, field_amplitude: HomogeneousFieldAmplitude
    ):
        super().__init__()

        self.geometry = geometry
        self.field_amplitude = field_amplitude

        self.H_0 = _build_hopping_csr(geometry, dtype=B.FDTYPE)

        # Sparse diagonal: diag(r_i · E_direction), centred around zero
        position_shifts = B.xp().array(
            geometry.site_positions @ field_amplitude.direction,
            dtype=B.FDTYPE,
        )
        position_shifts -= B.xp().mean(position_shifts)

        self.position_operator = B.xp_sparse().diags(
            position_shifts, format="csr", dtype=B.FDTYPE
        )

    def at_time(self, t: float) -> B.SparseArray:
        return self.H_0 + self.field_amplitude.at_time(t) * self.position_operator


class LinearFieldHamiltonianPeierls(Hamiltonian):
    """Hamiltonian with Peierls substitution for a homogeneous electric field.

    Works for both open and periodic boundary conditions; the geometry is
    responsible for providing the correct short bond vectors via
    ``geometry.bond_vectors``.
    """

    def __init__(
        self, geometry: Lattice2DGeometry, field_amplitude: HomogeneousFieldAmplitude
    ):
        super().__init__()

        self.geometry = geometry
        self.field_amplitude = field_amplitude

        self.H_0 = _build_hopping_csr(geometry, dtype=B.DTYPE)

        # for Peierls substitution, we need a phase shift matrix with elements theta_kl = (r_k - r_l) . A(t)
        size = geometry.Lx * geometry.Ly
        nn = geometry.nearest_neighbors
        bv = geometry.bond_vectors
        theta_fwd = (bv @ field_amplitude.direction).astype(float)  # (E,)

        # to add h.c., append nearest neighbors with indices swapped, and data with sign flipped
        theta_data = B.xp().array(
            np.concatenate([theta_fwd, -theta_fwd]), dtype=B.DTYPE
        )
        rows = np.concatenate([nn[:, 0], nn[:, 1]])
        cols = np.concatenate([nn[:, 1], nn[:, 0]])
        self.theta_matrix = (
            B.xp_sparse()
            .coo_matrix(
                (theta_data, (B.xp().array(rows), B.xp().array(cols))),
                shape=(size, size),
                dtype=B.DTYPE,
            )
            .tocsr()
        )

    def at_time(self, t: float) -> B.SparseArray:
        # Modify hopping amplitudes by Peierls phase: t_kl -> t_kl * exp(-i * theta_kl * t)
        phase_factors = B.xp().exp(
            -1j * self.theta_matrix.data * self.field_amplitude.integrate_to_time(t)
        )

        H_t = self.H_0.copy()
        H_t.data *= phase_factors

        return H_t
