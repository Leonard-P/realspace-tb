from ..hamiltonian import Hamiltonian
from .lattice_2d_geometry import Lattice2DGeometry
from .. import backend as B
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
import scipy.special as sp
import cupyx.scipy.special as csp


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
    """

    def __init__(
        self,
        E0: float,
        omega: float,
        T_ramp: float,
        direction: B.FCPUArray,
    ):
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

        integral = xp.zeros_like(t, dtype=B.FDTYPE)
        s = xp.where(t < T, t, T)
        if T > 0.0:
            ramp_integral = (
                (T**2 * w**2 - pi * T * w) * xp.cos((s * T * w + pi * s) / T)
                + (T**2 * w**2 + pi * T * w) * xp.cos((s * T * w - pi * s) / T)
                + (2 * pi**2 - 2 * T**2 * w**2) * xp.cos(w * s)
                - 2 * pi**2
            )
            ramp_integral /= 4 * w * (T**2 * w**2 - pi**2)
            # Apply ramp integral only where t > 0
            integral = xp.where(t > 0.0, ramp_integral, integral)

        # For array elements where t > T, add the post-ramp continuous oscillation integral
        post_ramp_integral = (xp.cos(w * T) - xp.cos(w * t)) / w
        integral = xp.where(t > T, integral + post_ramp_integral, integral)

        return self.E0 * integral


class LightPulseComponent(HomogeneousFieldAmplitude):
    """
    A single linearly polarized COMPONENT of an ultrashort light pulse.
    E(t) = E0 * exp(-(t-t_c)^2 / (2*sigma^2)) * cos(omega*(t-t_c) - phase_shift)

    To construct ultrashort light pulse, use EllipticalPulseFactory class

    The spatial dependence of the electromagnetic field is neglected,
    assuming the system size $L$ is much smaller than the spatial extent of the pulse.

    Validity range:
    ($kL \ll 1$ or $\omega \ll c/L$) and $L/(c\sigma) \ll 1

    Parameters
    ----------
    E0          : float
        Peak electric field amplitude.
    t_c         : float
        Pulse-center time.
    sigma       : float
        Gaussian pulse width ($\sigma$).
    omega       : float
        Pulse carrier oscillation frequency ($\omega$).
    phase_shift : float
        Extra phase of the light. CEP is encoded here.
    direction   : B.FCPUArray
        Direction along which E-field oscillates
    """

    def __init__(
        self,
        E0: float,
        t_c: float,
        sigma: float,
        omega: float,
        phase_shift: float,
        direction: B.FCPUArray,
    ):
        self.E0 = B.FDTYPE(E0)
        self.t_c = B.FDTYPE(t_c)
        self.sigma = B.FDTYPE(sigma)
        self.omega = B.FDTYPE(omega)
        self.phase_shift = B.FDTYPE(phase_shift)
        self.direction = B.FDTYPE(direction)

    def at_time(self, t: "float | B.Array") -> "float | B.Array":
        # TODO maybe make it CPU only and move backend transfer to Hamiltonian class
        # -> avoid confusion and minimize code scope of GPU backend
        xp = B.xp()
        return (
            self.E0
            * xp.exp(-((t - self.t_c) ** 2) / (2 * self.sigma**2))
            * xp.cos(self.omega * (t - self.t_c) - self.phase_shift)
        )

    def integrate_to_time(self, t: "float | B.Array") -> "float | B.Array":
        """Integrate the field amplitude from time 0 to t. Needed for Peierls substitution."""
        xp = B.xp()
        tc = self.t_c
        s = self.sigma
        w = self.omega
        d = self.phase_shift
        pi = xp.pi
        sqrt = xp.sqrt
        exp = xp.exp
        # wofz instead of exp(-z^2)*erf(z) to avoid numerical overflow
        if xp.__name__ == "cupy":
            wofz = csp.wofz
        if xp.__name__ == "numpy":
            wofz = sp.wofz

        b = s * w / sqrt(2)

        def f(a):
            return exp(-(a**2) - 2j * a * b) * wofz(-b + 1j * a)

        a1 = (tc - t) / (sqrt(2) * s)  # replaces erf(z(t))
        a2 = tc / (sqrt(2) * s)  # replaces erf(z(0))

        return self.E0 * sqrt(pi / 2) * s * xp.real(exp(-1j * d) * (f(a2) - f(a1)))
        # if xp.__name__ == "cupy":
        #    erf = csp.erf
        # if xp.__name__ == "numpy":
        #    erf = sp.erf
        # return (
        #    self.E0
        #    * sqrt(pi / 2)
        #    * s
        #    * exp(-((s * w) ** (2)) / 2)
        #    * xp.real(
        #        exp(-1j * d)
        #        * (
        #            erf((1j * s ** (2) * w + tc - t) / (sqrt(2) * s))
        #            - erf((1j * s ** (2) * w + tc) / (sqrt(2) * s))
        #        )
        #    )
        # )


class EllipticalPulseFactory:
    """Factory to generate the orthogonal field components for an elliptical pulse.

    Parameters
    ----------
    E0          : float
        Peak electric field amplitude.
    t_c         : float
        Pulse-center time.
    sigma       : float
        Gaussian pulse width ($\sigma$).
    omega       : float
        Pulse carrier oscillation frequency ($\omega$).
    phase_shift : float
        Extra phase of the light. CEP is encoded here.
    vartheta    : float
        Polar angle ($\vartheta$) of the propagation vector $\mathbf{n}$.
    varphi         : float
        Azimuthal angle ($\varphi$) of the propagation vector $\mathbf{n}$.
    """

    @staticmethod
    def create(
        E0: float,
        t_c: float,
        sigma: float,
        varepsilon: float,
        omega: float,
        phase_shift: float,
        vartheta: float,
        varphi: float,
    ) -> list[LightPulseComponent]:
        """
        Returns a list of two HomogeneousFieldAmplitude objects representing
        the e1 and e2 components of the elliptical pulse.
        e1 and e2 are truncated only to their x- and y-component.
        """
        # e1 component: cos(omega*(t-t_c) - varphi_0)
        xp = B.xp()
        e1_direction = xp.array(
            [xp.cos(vartheta) * xp.cos(varphi), xp.cos(vartheta) * xp.sin(varphi)]
        )
        comp1 = LightPulseComponent(
            E0=E0,
            t_c=t_c,
            sigma=sigma,
            omega=omega,
            phase_shift=phase_shift,
            direction=e1_direction,
        )

        # e2 component: -varepsilon * sin(omega*(t-t_c) - varphi_0)
        # -sin(x) = cos(x + pi/2), so phase_shift becomes varphi_0 - pi/2
        e2_direction = xp.array([-xp.sin(varphi), xp.cos(varphi)])
        comp2 = LightPulseComponent(
            E0=E0 * varepsilon,
            t_c=t_c,
            sigma=sigma,
            omega=omega,
            phase_shift=phase_shift - (np.pi / 2),
            direction=e2_direction,
        )

        return comp1, comp2


class DeltaKickFieldAmplitude(HomogeneousFieldAmplitude):
    """Delta-kick E(t) = E0 * \delta(t)"""

    def __init__(
        self,
        E0: float,
        direction: B.FCPUArray,
    ):
        self.E0 = B.FDTYPE(E0)
        self.direction = B.FDTYPE(direction)

    def at_time(self, t: "float | B.Array") -> "float | B.Array":
        """NOT to be literally used. Observables requiring dH/dt_t=0
        should be interpreted with CARE"""
        xp = B.xp()
        if xp.isscalar(t):
            if t == 0.0:
                val = 10 ** (4)  # xp.inf
            else:
                val = 0.0
            return self.E0 * val

        vals = xp.where(
            t == 0.0,
            10 ** (4),  # xp.inf * t
            xp.zeros_like(t, dtype=B.FDTYPE),
        )
        return self.E0 * vals

    def integrate_to_time(self, t: "float | B.Array") -> "float | B.Array":
        """Integrate the field amplitude from time -\infty to t. Needed for Peierls substitution."""
        xp = B.xp()
        if xp.isscalar(t):
            integral = 0.0
            if t > 0.0:
                integral += 1.0
            return self.E0 * integral
        integral = xp.zeros_like(t, dtype=B.FDTYPE)
        integral = xp.where(t > 0.0, 1.0, integral)
        return self.E0 * integral


class RampedConstantFieldAmplitude(HomogeneousFieldAmplitude):
    """Initially ramped and, from T_ramp onwards, CONSTANT electric field amplitude.
    E(t) = E0 * sin^2(pi * t / 2 * T_ramp), capped at E0."""

    def __init__(
        self,
        E0: float,
        T_ramp: float,
        direction: B.FCPUArray,
    ):
        self.E0 = B.FDTYPE(E0)
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

        integral = xp.zeros_like(t, dtype=B.FDTYPE)
        s = xp.where(t < T, t, T)
        if T > 0.0:
            ramp_integral = -(T * xp.sin(pi * s / T) - pi * s) / (2 * pi)
            integral = xp.where(t > 0.0, ramp_integral, integral)
        post_ramp_integral = t - T
        integral = xp.where(t > T, integral + post_ramp_integral, integral)

        return self.E0 * integral


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
        # Remark: If B0 != 0.0, H_0 is not equal H(t=0)
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
        t_hop: int | float,
        field_amplitudes: list[HomogeneousFieldAmplitude],
        B0: float = 0.0,
    ):
        super().__init__()

        self.geometry = geometry
        self.field_amplitudes = field_amplitudes
        ### CHECK the implementation again for B != 0.0. It is likely still incorrect ###
        self.B0 = B.FDTYPE(B0)
        if (self.B0 != 0.0) and (self.geometry.pbc_x) and (self.geometry.pbc_y):
            raise ValueError(
                "System with B0 != 0.0 requires open boundary in at least one direction."
            )
        # Remark: If B0 != 0.0, H_0 != H(t=0)
        self.H_0 = _build_hopping_csr(geometry, t_hop, dtype=B.DTYPE)

        # for Peierls substitution, we need a phase shift matrix with elements theta_kl = (r_k - r_l) . A(t)
        size = geometry.Lx * geometry.Ly
        nn = geometry.nearest_neighbors
        bv = geometry.nn_bond_vectors
        spnn = geometry.site_positions[geometry.nearest_neighbors]
        rows = np.concatenate([nn[:, 0], nn[:, 1]])
        cols = np.concatenate([nn[:, 1], nn[:, 0]])
        # Create independent theta matrices for each E-field component
        self.theta_matrices = []
        for field in self.field_amplitudes:
            theta_fwd = (bv @ field.direction).astype(float)
            theta_data = B.xp().array(
                np.concatenate([theta_fwd, -theta_fwd]), dtype=B.DTYPE
            )
            self.theta_matrices.append(
                B.xp_sparse()
                .coo_matrix(
                    (theta_data, (B.xp().array(rows), B.xp().array(cols))),
                    shape=(size, size),
                    dtype=B.DTYPE,
                )
                .tocsr()
            )
        # Handle B-field theta_matrix2
        if geometry.pbc_y:
            # Such gauge is chosen that A (vector potential) does not depend on y
            direction2 = B.xp().array([0.0, 1.0])
            pref_theta_fwd2 = 1 / 2 * self.B0 * spnn[:, :, 0].sum(axis=1)
        else:
            # Such gauge is chosen that A (vector potential) does not depend on x
            direction2 = B.xp().array([1.0, 0.0])
            pref_theta_fwd2 = -1 / 2 * self.B0 * spnn[:, :, 1].sum(axis=1)
        theta_fwd2 = (bv @ direction2).astype(float) * pref_theta_fwd2  # (E,)
        # to add h.c., append nearest neighbors with indices swapped, and data with sign flipped
        theta_data2 = B.xp().array(
            np.concatenate([theta_fwd2, -theta_fwd2]), dtype=B.DTYPE
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
        total_theta = sum(
            theta_mat.data * field.integrate_to_time(t)
            for field, theta_mat in zip(self.field_amplitudes, self.theta_matrices)
        )
        phase_factors = B.xp().exp(-1j * (total_theta - self.theta_matrix2.data))

        H_t = self.H_0.copy()
        H_t.data *= phase_factors

        return H_t

    def derivative_at_time(self, t: float) -> B.SparseArray:
        # Time derivative of the Hamiltonian for Peierls substitution
        dtheta_dt = sum(
            -theta_mat.data * field.at_time(t)
            for field, theta_mat in zip(self.field_amplitudes, self.theta_matrices)
        )
        total_theta = sum(
            theta_mat.data * field.integrate_to_time(t)
            for field, theta_mat in zip(self.field_amplitudes, self.theta_matrices)
        )
        phase_factors = B.xp().exp(-1j * (total_theta - self.theta_matrix2.data))
        dH_dt = self.H_0.copy()
        dH_dt.data *= 1j * dtheta_dt * phase_factors
        return dH_dt
