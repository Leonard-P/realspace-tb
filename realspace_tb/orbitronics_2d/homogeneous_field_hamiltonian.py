from ..hamiltonian import Hamiltonian
from .lattice_2d_geometry import Lattice2DGeometry
from .. import backend as B
from abc import ABC, abstractmethod
import numpy as np


def _build_hopping_csr(
    geometry: Lattice2DGeometry, t_hop: float = -0.0367, dtype: type = B.DTYPE
) -> "B.SparseArray":
    """Build a sparse nearest-neighbor hopping matrix from geometry with t_hop.
    Default t_hop = -0.0367 roughly equals 1 eV"""
    size = geometry.Lx * geometry.Ly
    nn = geometry.nearest_neighbors  # (E, 2)
    rows = np.concatenate([nn[:, 0], nn[:, 1]])
    cols = np.concatenate([nn[:, 1], nn[:, 0]])
    data = B.xp().full(len(rows), t_hop, dtype=dtype)
    return (
        B.xp_sparse()
        .coo_matrix(
            (data, (B.xp().array(rows), B.xp().array(cols))),
            shape=(size, size),
        )
        .tocsr()
    )


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
    Spatially homogeneous and time-independent magnetic field B = B0
    in z-direction can be optionally included (default B0 = 0.0)
    """

    def __init__(
        self,
        E0: float,
        omega: float,
        T_ramp: float,
        direction: B.FCPUArray,
        B0: float = 0.0,
    ):
        self.E0 = B.FDTYPE(E0)
        self.B0 = B.FDTYPE(B0)
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


class RampedConstantFieldAmplitude(HomogeneousFieldAmplitude):
    """
    Initially ramped and, from T_ramp onwards, CONSTANT electric field amplitude.
    E(t) = E0 * sin^2(pi * t / 2 * T_ramp), capped at E0.
    Spatially homogeneous and time-independent magnetic field B = B0
    in z-direction can be optionally included (default B0 = 0.0)
    """

    def __init__(
        self,
        E0: float,
        T_ramp: float,
        direction: B.FCPUArray,
        B0: float = 0.0,
    ):
        self.E0 = B.FDTYPE(E0)
        self.B0 = B.FDTYPE(B0)
        self.T_ramp = B.FDTYPE(T_ramp)
        self.direction = B.FDTYPE(direction)

    def at_time(self, t: "float | B.Array") -> "float | B.Array":
        xp = B.xp()
        if xp.isscalar(t):
            if t < self.T_ramp:
                ramp = xp.sin(np.pi * t / (2 * self.T_ramp)) ** 2
            else:
                ramp = 1.0
            return self.E0 * ramp

        ramp = xp.where(
            t < self.T_ramp,
            xp.sin(xp.pi * t / (2 * self.T_ramp)) ** 2,
            xp.ones_like(t, dtype=B.FDTYPE),
        )
        return self.E0 * ramp

    def integrate_to_time(self, t: "float | B.Array") -> "float | B.Array":
        """Integrate the field amplitude from time 0 to t. Needed for Peierls substitution."""
        xp = B.xp()
        T = self.T_ramp
        pi = xp.pi
        if xp.isscalar(t):
            integral = 0.0
            s = t if t < T else T
            if t > 0.0 and T > 0.0:
                integral += -(T * xp.sin(pi * s / T) - pi * s) / (2 * pi)
            if t > T:
                integral += t - T
            return self.E0 * integral

        raise NotImplementedError(
            "Integration of array time inputs not implemented yet."
        )


class LinearFieldHamiltonian(Hamiltonian):
    """Hamiltonian with position operator for a spatially homogeneous electric field.
    Unsuitable when the system is periodic in at least one direction
    B-field has not been integrated"""

    def __init__(
        self,
        geometry: Lattice2DGeometry,
        t_hop: float,
        field_amplitude: HomogeneousFieldAmplitude,
    ):
        super().__init__()

        self.geometry = geometry
        self.field_amplitude = field_amplitude

        self.H_0 = _build_hopping_csr(geometry, t_hop, dtype=B.FDTYPE)

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
    """Hamiltonian with Peierls substitution for a spatially homogeneous electric field.

    Works for both open and periodic boundary conditions; the geometry is
    responsible for providing the correct short bond vectors via
    ``geometry.nn_bond_vectors``.
    If B0 \neq 0.0, the system has to have an open BOUNDARY in AT LEAST ONE direction
    """

    def __init__(
        self,
        geometry: Lattice2DGeometry,
        t_hop,
        field_amplitude: HomogeneousFieldAmplitude,
    ):
        super().__init__()

        self.geometry = geometry
        self.field_amplitude = field_amplitude
        try:
            B0 = self.field_amplitude.B0
            if (B0 != 0.0) and (self.geometry.pbc_x) and (self.geometry.pbc_y):
                raise ValueError
        except AttributeError:
            # Set B0 = 0.0, if it does not exist
            B0 = 0.0
        self.H_0 = _build_hopping_csr(geometry, t_hop, dtype=B.DTYPE)

        # for Peierls substitution, we need a phase shift matrix with elements theta_kl = (r_k - r_l) . A(t)
        size = geometry.Lx * geometry.Ly
        nn = geometry.nearest_neighbors
        bv = geometry.nn_bond_vectors
        spnn = geometry.site_positions[geometry.nearest_neighbors]
        theta_fwd = (bv @ field_amplitude.direction).astype(float)  # (E,)
        if geometry.pbc_y:
            # Such gauge is chosen that A (vector potential) does not depend on y
            direction2 = B.xp().array([0.0, 1.0])
            pref_theta_fwd2 = 1 / 2 * B0 * spnn[:, :, 0].sum(axis=1)
        else:
            # Such gauge is chosen that A (vector potential) does not depend on x
            direction2 = B.xp().array([1.0, 0.0])
            pref_theta_fwd2 = -1 / 2 * B0 * spnn[:, :, 1].sum(axis=1)
        theta_fwd2 = (bv @ direction2).astype(float) * pref_theta_fwd2  # (E,)
        # to add h.c., append nearest neighbors with indices swapped, and data with sign flipped
        theta_data = B.xp().array(
            np.concatenate([theta_fwd, -theta_fwd]), dtype=B.DTYPE
        )
        theta_data2 = B.xp().array(
            np.concatenate([theta_fwd2, -theta_fwd2]), dtype=B.DTYPE
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
        self.theta_matrix2 = (
            B.xp_sparse()
            .coo_matrix(
                (theta_data2, (B.xp().array(rows), B.xp().array(cols))),
                shape=(size, size),
                dtype=B.DTYPE,
            )
            .tocsr()
        )

    def at_time(self, t: float) -> B.SparseArray:
        # Modify hopping amplitudes by Peierls phase: t_kl -> t_kl * exp(-i * theta_kl * t)
        # In theta_matrix[0,1]: R_1 - R_0 instead of R_0 - R_1
        phase_factors = B.xp().exp(
            -1j
            * (
                self.theta_matrix.data * self.field_amplitude.integrate_to_time(t)
                - self.theta_matrix2.data
            )
        )

        H_t = self.H_0.copy()
        H_t.data *= phase_factors

        return H_t

    def derivative_at_time(self, t: float) -> B.SparseArray:
        # Time derivative of the Hamiltonian for Peierls substitution
        dtheta_dt = -self.theta_matrix.data * self.field_amplitude.at_time(t)
        phase_factors = B.xp().exp(
            -1j
            * (
                self.theta_matrix.data * self.field_amplitude.integrate_to_time(t)
                - self.theta_matrix2.data
            )
        )
        dH_dt = self.H_0.copy()
        dH_dt.data *= 1j * dtheta_dt * phase_factors
        return dH_dt
