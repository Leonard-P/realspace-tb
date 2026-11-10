r"""
Bloch (k-space) representation of a real-space supercell that is periodic along one axis.

Physics
-------
A real-space model that is periodic along y (``pbc_y=True``) is a *ring* of finite
circumference L. The infinite system (L -> infinity) is obtained with Bloch's theorem:
for every twist angle :math:`\theta = k L` the supercell Hamiltonian is

.. math::  H_{ab}(\theta, t) = \sum_R h_{a0,\,bR}(t)\, e^{i\theta R},

i.e. the real-space supercell Hamiltonian with every hopping that crosses the periodic
boundary (from cell 0 into cell R = \pm 1) multiplied by :math:`e^{\pm i\theta}`.
The twist is exactly an Aharonov-Bohm flux :math:`\theta/2\pi` flux quanta threaded
through the ring; averaging over :math:`\theta` removes the dependence on it
(and therefore on the Landau-gauge origin). A Gamma-centred mesh of N points on a
supercell is *identical* to a real-space ring of N supercells.

As long as H(t) is translation-invariant along the periodic axis (uniform fields,
or fields that only vary along the open axis) and the initial state is too, the
density matrix stays block-diagonal in k and every block evolves independently:

.. math::  i\,\dot\rho_\theta = [H(\theta, t), \rho_\theta].

State layout
------------
The Bloch density matrix is stored as one array ``rho[a, k, b]`` of shape
``(N_s, N_k, N_s)`` (N_s = sites per supercell). This layout lets one sparse-dense
product act on *all* k-points at once: ``H @ rho.reshape(N_s, N_k * N_s)``.
Use :meth:`BlochHamiltonian.to_k_stack` to get the ``(N_k, N_s, N_s)`` view.

Observables
-----------
Observables of the infinite system are k-averages,
:math:`\langle O\rangle = \sum_k w_k\, O(\rho_\theta, H(\theta,t))`.
For observables built only from on-site and bond elements (densities, bond currents,
vorticities, ...) this equals ``O(rho_bar, H(t))`` with the untwisted real-space H(t)
and the k-averaged, phase-corrected matrix ``rho_bar`` from
:meth:`BlochHamiltonian.k_averaged_density_matrix` -- one evaluation instead of N_k.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Sequence

import numpy as np

from . import backend as B
from .hamiltonian import Hamiltonian, fermi_occupation

_AXES = {"x": 0, "y": 1}


def _host(a) -> np.ndarray:
    """NumPy copy of a NumPy or CuPy array."""
    return a.get() if hasattr(a, "get") else np.asarray(a)


class KMesh:
    r"""Twist angles :math:`\theta_n = kL` and weights for the Brillouin-zone average.

    Parameters
    ----------
    n_k : int
        Number of k-points.
    shift : float
        :math:`\theta_n = 2\pi (n + \mathrm{shift}) / n_k`. ``shift=0`` is Gamma-centred
        (identical to a real-space ring of ``n_k`` supercells); the default ``0.0``.
        Half-shifted mesh (``shift=0.5``) is symmetric under :math:`\theta\to-\theta`
        and never contains :math:`\theta=0`.
    thetas, weights : optional explicit arrays (override ``n_k``/``shift``).
    """

    def __init__(
        self,
        n_k: int | None = None,
        shift: float = 0.0,
        thetas: Sequence[float] | None = None,
        weights: Sequence[float] | None = None,
    ) -> None:
        # A uniform mesh theta_n = 2 pi (n + shift) / n_k is *exactly* a real-space ring of
        # n_k cells threaded by the flux `shift` (in flux quanta). Explicit meshes have no ring.
        self.n_cells: int | None = None
        self.shift: float | None = None
        if thetas is None:
            if n_k is None or n_k < 1:
                raise ValueError("Give n_k >= 1 or explicit thetas.")
            thetas = 2 * np.pi * (np.arange(n_k) + shift) / n_k
            self.n_cells, self.shift = int(n_k), float(shift)
        self.thetas = np.asarray(thetas, dtype=np.float64).ravel()
        if weights is None:
            weights = np.full(self.thetas.size, 1.0 / self.thetas.size)
        self.weights = np.asarray(weights, dtype=np.float64).ravel()
        if self.weights.shape != self.thetas.shape:
            raise ValueError("thetas and weights must have the same length")
        if not np.isclose(self.weights.sum(), 1.0):
            raise ValueError("weights must sum to 1")

    @property
    def n_k(self) -> int:
        return int(self.thetas.size)

    def subset(self, indices: Sequence[int]) -> tuple["KMesh", float]:
        """Sub-mesh with renormalized weights, plus its total weight in this mesh."""
        idx = np.asarray(indices, dtype=int)
        w = self.weights[idx]
        total = float(w.sum())
        return KMesh(thetas=self.thetas[idx], weights=w / total), total

    def __repr__(self) -> str:
        if self.n_cells is not None:
            return f"KMesh(n_k={self.n_k}, shift={self.shift})"
        return f"KMesh(n_k={self.n_k}, explicit)"


def _wrap_indices(geometry, axis: int, tol_rel: float = 1e-6):
    """Return (i, j, R, L) for every nearest-neighbour bond: j's image sits in cell R
    (R in {-1, 0, +1}) relative to i along `axis`; L is the cell length along `axis`."""
    nn = np.asarray(geometry.nearest_neighbors, dtype=np.int64)
    pos = np.asarray(geometry.site_positions, dtype=np.float64)
    bv = np.asarray(geometry.nn_bond_vectors, dtype=np.float64)
    shift = pos[nn[:, 0]] + bv - pos[nn[:, 1]]  # 0 for internal bonds, +-L for wrapped
    tol = tol_rel * float(np.abs(bv).max())
    comp = shift[:, axis]
    wrapped = np.abs(comp) > tol
    R = np.where(wrapped, np.sign(comp), 0.0).astype(np.int64)
    if not wrapped.any():
        raise ValueError("No bond crosses the periodic boundary along the Bloch axis.")
    period = float(np.abs(comp[wrapped]).mean())
    return nn[:, 0], nn[:, 1], R, period


class BlochHamiltonian(Hamiltonian):
    r"""k-space version of a real-space supercell Hamiltonian that is periodic along `axis`.

    Wraps any real-space :class:`Hamiltonian` (e.g. ``LinearFieldHamiltonianPeierls``)
    whose geometry is periodic along `axis`. The physics (hopping, Peierls phases of
    E and B fields, ...) is taken unchanged from ``base.at_time(t)``; this class only
    adds the Bloch twist on boundary-crossing hoppings.

    ``at_time(t)`` returns the *untwisted* supercell H(t) (exactly ``base.at_time(t)``)
    unless a k-point is activated with :meth:`at_k`; then it returns H(theta_k, t).
    That makes every existing real-space observable usable in k-space.

    `axis` defaults to the single periodic direction of the geometry: 'y' for an armchair
    ribbon (open in x), 'x' for a zigzag ribbon (open in y). The cell must be long enough
    that no pair of sites is bonded both directly and across the boundary (armchair:
    Ly >= 2 rows; zigzag: Lx >= 4 columns).

    Requirements: H(t) must be translation-invariant along `axis` in the chosen gauge
    (uniform fields; B in the Landau gauge A = (0, B x) for 'y' and A = (-B y, 0) for 'x',
    which ``LinearFieldHamiltonianPeierls`` picks automatically). A scalar potential along
    `axis` is not allowed.

    Back to real space: a uniform mesh ``KMesh(N, shift)`` is exactly a ring of N cells with seam
    twist 2 pi shift. ``ring_geometry``, ``to_real_space()``, ``real_space_hamiltonian``,
    ``real_space_density_matrix`` and ``real_space_bond_currents`` are built from the mesh, so no
    real-space geometry has to be passed in.
    """

    def __init__(
        self,
        base: Hamiltonian,
        kmesh: KMesh | int,
        axis: str | None = None,
        geometry=None,
    ) -> None:
        super().__init__()
        self.base = base
        self.geometry = geometry if geometry is not None else base.geometry
        self.kmesh = kmesh if isinstance(kmesh, KMesh) else KMesh(int(kmesh))
        if axis is None:  # the one periodic direction of the geometry
            periodic = [
                a for a in ("x", "y") if getattr(self.geometry, f"pbc_{a}", False)
            ]
            if len(periodic) != 1:
                raise ValueError(
                    "Cannot infer the Bloch axis: the geometry must be periodic along exactly "
                    "one direction (or pass axis='x' / 'y')."
                )
            axis = periodic[0]
        if axis not in _AXES:
            raise ValueError("axis must be 'x' or 'y'")
        self.axis = axis
        ax = _AXES[axis]
        if not getattr(self.geometry, f"pbc_{axis}"):
            raise ValueError(
                f"Geometry must be periodic along {axis} (pbc_{axis}=True)."
            )
        self._check_translation_invariant_field(ax)

        self.n_sites = int(self.geometry.Lx * self.geometry.Ly)
        self._ax = ax
        i, j, R, period = _wrap_indices(self.geometry, ax)
        self.period = period  # length L of the cell along the Bloch axis
        pairs = np.sort(np.stack([i, j], axis=1), axis=1)
        if np.unique(pairs, axis=0).shape[0] != pairs.shape[0]:
            raise ValueError(
                "A pair of sites is connected both directly and through the periodic "
                "boundary. Use a longer supercell along the periodic axis."
            )
        n = self.n_sites
        R_lookup = np.zeros((n, n), dtype=np.int8)
        known = np.eye(n, dtype=bool)
        R_lookup[i, j], R_lookup[j, i] = R, -R
        known[i, j] = known[j, i] = True
        self._R_cpu = R_lookup
        self._known_cpu = known
        self._wrap_rows, self._wrap_cols = np.nonzero(R_lookup)
        self._wrap_R = R_lookup[self._wrap_rows, self._wrap_cols].astype(np.float64)

        self._active_k: int | None = None
        self._device_cache: dict = {}
        self._pattern = None  # cached per-element data for the CSR sparsity pattern
        self._split_cache: dict[float, tuple] = {}

    # ------------------------------------------------------------------ basics
    def _check_translation_invariant_field(self, ax: int) -> None:
        """Reject a scalar (length-gauge) potential along the periodic axis."""
        if hasattr(self.base, "position_operator"):
            amps = getattr(self.base, "field_amplitudes", None) or [
                getattr(self.base, "field_amplitude", None)
            ]
            for amp in amps:
                if amp is not None and abs(float(np.asarray(amp.direction)[ax])) > 0:
                    raise ValueError(
                        "A scalar potential along the periodic axis breaks translation "
                        "symmetry. Use the Peierls (vector-potential) Hamiltonian."
                    )

    @property
    def n_k(self) -> int:
        return self.kmesh.n_k

    def _dev(self, name: str):
        """Backend copies of small lookup arrays (numpy or cupy)."""
        key = (name, B.USE_GPU)
        if key not in self._device_cache:
            xp = B.xp()
            src = {
                "thetas": self.kmesh.thetas,
                "weights": self.kmesh.weights,
                "R": self._R_cpu,
                "known": self._known_cpu,
                "wrap_rows": self._wrap_rows,
                "wrap_cols": self._wrap_cols,
                "wrap_R": self._wrap_R,
            }[name]
            self._device_cache[key] = xp.asarray(src)
        return self._device_cache[key]

    def subset(self, indices: Sequence[int]) -> tuple["BlochHamiltonian", float]:
        """Same model on a subset of k-points (for parallel chunks)."""
        mesh, total = self.kmesh.subset(indices)
        return BlochHamiltonian(self.base, mesh, self.axis, self.geometry), total

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_device_cache"] = {}
        state["_pattern"] = None
        state["_split_cache"] = {}
        return state

    # ------------------------------------------------------- sparsity pattern
    def _pattern_info(self, H) -> dict:
        """Per-stored-element wrap index R and index arrays of the R = 0, +1, -1 parts.
        Cached as long as the CSR sparsity pattern of H(t) does not change.
        (Computed on the host with NumPy, then moved to the active backend.)"""
        xp = B.xp()
        p = self._pattern
        if (
            p is not None
            and p["indptr"].shape == H.indptr.shape
            and p["indices"].shape == H.indices.shape
            and bool(xp.array_equal(p["indptr"], H.indptr))
            and bool(xp.array_equal(p["indices"], H.indices))
        ):
            return p
        n = self.n_sites
        indptr, cols, data = _host(H.indptr), _host(H.indices), _host(H.data)
        rows = np.repeat(np.arange(n), np.diff(indptr))
        R = self._R_cpu[rows, cols].astype(np.int64)
        if not np.all(self._known_cpu[rows, cols] | (data == 0)):
            raise ValueError(
                "H(t) has hoppings between sites that are not nearest neighbours of the "
                "geometry; their boundary-crossing index is unknown."
            )
        parts = {}
        for name, value in (("p0", 0), ("pp", 1), ("pm", -1)):
            sel = np.nonzero(R == value)[0]
            # the boundary parts only touch a few rows: keep just those rows
            active = np.arange(n) if name == "p0" else np.unique(rows[sel])
            pos = np.searchsorted(active, rows[sel])
            counts = np.bincount(pos, minlength=active.size)
            part_indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
            parts[name] = tuple(
                xp.asarray(a) for a in (sel, cols[sel], part_indptr, active)
            )
        self._pattern = {
            "indptr": H.indptr.copy(),
            "indices": H.indices.copy(),
            "R": xp.asarray(R, dtype=B.FDTYPE),
            "parts": parts,
        }
        return self._pattern

    def _twist(self, H, theta: float):
        """Return H with every boundary-crossing element multiplied by e^{i theta R}."""
        xp = B.xp()
        H = H.tocsr()
        info = self._pattern_info(H)
        Hk = H.copy().astype(B.DTYPE)
        Hk.data = Hk.data * xp.exp(1j * theta * info["R"])
        return Hk

    # --------------------------------------------------------- Hamiltonian API
    def at_time(self, t: float):
        """Untwisted supercell H(t), or H(theta_k, t) if a k-point is active."""
        H = self.base.at_time(t)
        if self._active_k is None:
            return H
        return self._twist(H, float(self.kmesh.thetas[self._active_k]))

    def derivative_at_time(self, t: float):
        dH = self.base.derivative_at_time(t)
        if self._active_k is None:
            return dH
        return self._twist(dH, float(self.kmesh.thetas[self._active_k]))

    @contextmanager
    def at_k(self, index: int) -> Iterator["BlochHamiltonian"]:
        """Temporarily make ``at_time``/``derivative_at_time`` return the twisted H(theta_index)."""
        previous = self._active_k
        self._active_k = int(index)
        try:
            yield self
        finally:
            self._active_k = previous

    def split_at_time(self, t: float, derivative: bool = False):
        """Internal: ((H_0, rows_0), (H_+, rows_+), (H_-, rows_-)) with each part restricted
        to the rows `rows_*` it touches (boundary parts only touch a few rows).
        Cached for the two most recent times (RK4 evaluates t + dt/2 twice).
        See `split_matrices_at_time` for the full-size matrices."""
        key = (float(t), bool(derivative))
        if key in self._split_cache:
            return self._split_cache[key]
        source = self.base.derivative_at_time if derivative else self.base.at_time
        H = source(t).tocsr()
        info = self._pattern_info(H)
        data = H.data.astype(B.DTYPE, copy=False)
        sp = B.xp_sparse()
        out = []
        for name in ("p0", "pp", "pm"):
            sel, cols, indptr, active = info["parts"][name]
            M = sp.csr_matrix(
                (data[sel], cols, indptr), shape=(int(active.size), self.n_sites)
            )
            out.append((M, active))
        out = tuple(out)
        if len(self._split_cache) >= 3:
            self._split_cache.pop(next(iter(self._split_cache)))
        self._split_cache[key] = out
        return out

    def split_matrices_at_time(self, t: float, derivative: bool = False):
        r"""Full-size sparse (H_0, H_+, H_-) with H(\theta,t) = H_0 + e^{i\theta}H_+ + e^{-i\theta}H_-.

        H_0: hoppings inside the supercell, H_+ (H_-): hoppings into the next (previous) cell.
        """
        xp, sp = B.xp(), B.xp_sparse()
        mats = []
        for M, active in self.split_at_time(t, derivative):
            P = sp.csr_matrix(
                (
                    xp.ones(int(active.size), dtype=B.DTYPE),
                    (active, xp.arange(int(active.size))),
                ),
                shape=(self.n_sites, int(active.size)),
            )
            mats.append((P @ M).tocsr())
        return tuple(mats)

    def bloch_matrices(self, t: float = 0.0) -> "B.Array":
        """Dense H(theta_k, t) for all k, shape (N_k, N_s, N_s)."""
        xp = B.xp()
        H0, Hp, Hm = self.split_matrices_at_time(t)
        ph = xp.exp(1j * self._dev("thetas"))[:, None, None]
        return (
            H0.toarray()[None, :, :]
            + ph * Hp.toarray()[None, :, :]
            + ph.conj() * Hm.toarray()[None, :, :]
        ).astype(B.DTYPE)

    def apply(self, rho: "B.Array", t: float) -> "B.Array":
        """Return X with X[:, k, :] = H(theta_k, t) @ rho[:, k, :] for all k at once."""
        xp = B.xp()
        n, nk = self.n_sites, self.n_k
        (H0, _), (Hp, rows_p), (Hm, rows_m) = self.split_at_time(t)
        flat = rho.reshape(n, nk * n)
        X = (H0 @ flat).reshape(n, nk, n)
        ph = xp.exp(1j * self._dev("thetas")).astype(B.DTYPE)[None, :, None]
        if Hp.nnz:
            X[rows_p] += ph * (Hp @ flat).reshape(-1, nk, n)
        if Hm.nnz:
            X[rows_m] += ph.conj() * (Hm @ flat).reshape(-1, nk, n)
        return X

    # ------------------------------------------------------- eigen and states
    def _ensure_eigens(self) -> None:
        if self._eigenvalues is None:
            print(f"Calculating eigenstates at t=0 for {self.n_k} k-points...")
            self._eigenvalues, self._eigenstates = B.xp().linalg.eigh(
                self.bloch_matrices(0.0)
            )

    def ground_state_density_matrix(
        self, fermi_level: float = 0.0, tol: float | None = None
    ):
        return self.thermal_state_density_matrix(fermi_level, float("inf"), tol)

    def thermal_state_density_matrix(
        self,
        fermi_level: float = 0.0,
        beta: float = float("inf"),
        tol: float | None = None,
    ) -> "B.Array":
        r"""Grand-canonical state :math:`\rho_\theta = f(H(\theta, 0))` for every k-point.

        Returns the Bloch state in layout ``(N_s, N_k, N_s)``. For ``beta = inf``, states
        within `tol` of the Fermi level get occupation 1/2 (see ``fermi_occupation``);
        the default `tol` is the eigensolver error bound for an N_s x N_s matrix.
        """
        xp = B.xp()
        E, V = self.eigenvalues, self.eigenstates  # (N_k, N_s), (N_k, N_s, N_s)
        f = fermi_occupation(E, fermi_level, beta, tol, matrix_size=self.n_sites)
        rho_k = (V * f[:, None, :]) @ V.conj().swapaxes(-1, -2)
        return self.from_k_stack(rho_k.astype(B.DTYPE))

    # ------------------------------------------------------------- utilities
    @staticmethod
    def to_k_stack(rho: "B.Array") -> "B.Array":
        """(N_s, N_k, N_s) Bloch state -> (N_k, N_s, N_s) view."""
        return B.xp().moveaxis(rho, 1, 0)

    @staticmethod
    def from_k_stack(rho_k: "B.Array") -> "B.Array":
        """(N_k, N_s, N_s) -> contiguous (N_s, N_k, N_s) Bloch state."""
        return B.xp().ascontiguousarray(B.xp().moveaxis(rho_k, 0, 1))

    def k_averaged_density_matrix(self, rho: "B.Array") -> "B.Array":
        r"""Supercell matrix :math:`\bar\rho_{ab} = \sum_k w_k\, e^{-i\theta_k R_{ab}} \rho_{\theta_k,ab}`.

        :math:`R_{ab}` is the cell index of b's image relative to a for bonded pairs
        (0 otherwise). With the *untwisted* H(t), every on-site or bond-local expression
        :math:`H_{ba}\bar\rho_{ab}` equals the exact infinite-system value. Do NOT use it
        for products such as ``H @ rho`` (use the k-resolved path for those).
        """
        xp = B.xp()
        w = self._dev("weights").astype(B.DTYPE)
        rho_bar = xp.tensordot(rho, w, axes=([1], [0]))  # (N_s, N_s)
        rows, cols = self._dev("wrap_rows"), self._dev("wrap_cols")
        if rows.size:
            vals = rho[rows, :, cols]  # (n_wrap, N_k)
            ph = xp.exp(
                -1j * self._dev("wrap_R")[:, None] * self._dev("thetas")[None, :]
            )
            rho_bar[rows, cols] = (vals * ph) @ w
        return rho_bar

    # ------------------------------------------------- back to real space
    #
    # A uniform mesh of N points, theta_n = 2 pi (n + s) / N, is exactly a real-space ring of N
    # cells with a twist (Aharonov-Bohm flux) of s flux quanta: the unitary change of basis
    # |a, k> = N^{-1/2} sum_n e^{i k n L} |a, n> maps the N Bloch blocks onto the ring.
    # The ring is therefore fixed by the mesh; it is built here, not passed in.

    def _require_ring(self) -> tuple[int, float]:
        n, s = self.kmesh.n_cells, self.kmesh.shift
        if n is None:
            raise ValueError(
                "This k-mesh was given as explicit thetas/weights; it is not equivalent to a "
                "finite real-space ring. Use KMesh(n_k, shift) to map back to real space."
            )
        return n, s

    @property
    def ring_geometry(self):
        """Real-space geometry of the ring equivalent to this k-mesh: the cell repeated N_k
        times along the Bloch axis (armchair: Ly * N_k rows; zigzag: Lx * N_k columns).
        """
        if getattr(self, "_ring_geometry", None) is None:
            n, _ = self._require_ring()
            reps = {"x": (n, 1), "y": (1, n)}[self.axis]
            if not hasattr(self.geometry, "replicated"):
                raise NotImplementedError(
                    f"{type(self.geometry).__name__} has no `replicated(n_x, n_y)` method."
                )
            self._ring_geometry = self.geometry.replicated(*reps)
        return self._ring_geometry

    @property
    def ring_twist(self) -> float:
        """Phase 2 pi s carried by the ring's seam bonds (s = mesh shift; 0 for Gamma-centred)."""
        _, s = self._require_ring()
        return 2 * np.pi * s

    def _ring_sites(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(alpha, n, index) with alpha[I], n[I] = cell site and cell of ring site I, and
        index[alpha, n] = ring site I. Cached."""
        if getattr(self, "_ring_site_cache", None) is None:
            n_cells, _ = self._require_ring()
            alpha, n = self._match_sites(self.ring_geometry)
            if n.min() != 0 or n.max() != n_cells - 1:
                raise RuntimeError("Unexpected cell numbering of the ring geometry.")
            index = np.empty((self.n_sites, n_cells), dtype=np.int64)
            index[alpha, n] = np.arange(alpha.size)
            self._ring_site_cache = (alpha, n, index)
        return self._ring_site_cache

    def _match_sites(self, big_geometry) -> tuple[np.ndarray, np.ndarray]:
        """Map every site of a geometry built from copies of this cell onto (alpha, n)."""
        p = np.asarray(self.geometry.site_positions, dtype=float)
        r = np.asarray(big_geometry.site_positions, dtype=float)
        e = np.zeros(2)
        e[self._ax] = 1.0
        d = r[:, None, :] - p[None, :, :]
        n = np.rint(d[..., self._ax] / self.period)
        resid = np.linalg.norm(d - n[..., None] * self.period * e, axis=-1)
        hit = resid < 1e-6 * float(
            np.abs(np.asarray(self.geometry.nn_bond_vectors)).max()
        )
        if not np.all(hit.sum(axis=1) == 1):
            raise ValueError(
                "The geometry is not built from translated copies of this cell."
            )
        alpha = np.argmax(hit, axis=1)
        return alpha, n[np.arange(r.shape[0]), alpha].astype(np.int64)

    def site_map(self) -> tuple[np.ndarray, np.ndarray]:
        """For every site I of ``ring_geometry``: (alpha[I], n[I]) = site in the cell and cell index."""
        alpha, n, _ = self._ring_sites()
        return alpha.copy(), n.copy()

    def _ring_hamiltonian_elements(
        self, t: float, I: np.ndarray, J: np.ndarray, derivative=False
    ) -> np.ndarray:
        r"""H_ring[I, J] = \sum_{R' \equiv m-n (mod N)} h_{a0, bR'} e^{i\phi (R'-(m-n))/N}."""
        n_cells, _ = self._require_ring()
        alpha, n, _ = self._ring_sites()
        R = n[J] - n[I]
        a, b = alpha[I], alpha[J]
        parts = [_host(M.toarray()) for M in self.split_matrices_at_time(t, derivative)]
        out = np.zeros(np.shape(I), dtype=complex)
        for Rp, M in zip((0, 1, -1), parts):
            wraps = Rp - R
            hit = wraps % n_cells == 0
            out += np.where(
                hit, M[a, b] * np.exp(1j * self.ring_twist * wraps / n_cells), 0
            )
        return out

    def _ring_rho_elements(
        self, rho: "B.Array", I: np.ndarray, J: np.ndarray
    ) -> np.ndarray:
        r"""rho_ring[I, J] = \sum_k w_k e^{-i\theta_k (m-n)} \rho_k[a, b] (exact change of basis)."""
        self._require_ring()
        alpha, n, _ = self._ring_sites()
        rho_k = _host(self.to_k_stack(rho))  # (N_k, N_s, N_s)
        out = np.zeros(np.shape(I), dtype=complex)
        R = n[J] - n[I]
        for w, th, rk in zip(self.kmesh.weights, self.kmesh.thetas, rho_k):
            out += w * np.exp(-1j * th * R) * rk[alpha[I], alpha[J]]
        return out

    def real_space_hamiltonian(self, t: float = 0.0) -> np.ndarray:
        """Dense H(t) of the equivalent ring (``ring_geometry``), in its site order."""
        N = self.n_sites * self._require_ring()[0]
        I, J = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
        return self._ring_hamiltonian_elements(t, I, J)

    def real_space_density_matrix(self, rho: "B.Array") -> np.ndarray:
        """Dense density matrix of the equivalent ring (``ring_geometry``), same convention as the
        real-space code (rho_ij = <c_j^dagger c_i>). Size (N_k N_s)^2: meant for checks.
        """
        N = self.n_sites * self._require_ring()[0]
        I, J = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
        return self._ring_rho_elements(rho, I, J)

    def real_space_bond_currents(
        self, rho: "B.Array", t: float, pairs=None
    ) -> np.ndarray:
        r"""Bond currents :math:`I_{i\leftarrow j} = 2\,\mathrm{Im}(H_{ij}\rho_{ji})` on the equivalent
        ring, for site pairs (i, j) given as ``ring_geometry`` indices (default: all its bonds,
        in the order of ``ring_geometry.nearest_neighbors``). Only the requested elements are
        computed, so this also works for large rings. Same convention as ``BondCurrentObservable``.
        """
        pairs = np.asarray(
            self.ring_geometry.nearest_neighbors if pairs is None else pairs,
            dtype=np.int64,
        ).reshape(-1, 2)
        i, j = pairs[:, 0], pairs[:, 1]
        return 2.0 * np.imag(
            self._ring_hamiltonian_elements(t, i, j)
            * self._ring_rho_elements(rho, j, i)
        )

    def to_real_space(self) -> "RingHamiltonian":
        """Real-space Hamiltonian of the equivalent ring (geometry ``ring_geometry``), usable with
        ``RK4NeumannSolver`` and every real-space observable. For ``KMesh(N, shift=0)`` it equals
        the user's own real-space Hamiltonian on a ring of N cells."""
        return RingHamiltonian(self)

    # infinite-system access with explicit (alpha, cell) labels (independent of any ring)
    def correlator(
        self, rho: "B.Array", site_i: tuple[int, int], site_j: tuple[int, int]
    ) -> complex:
        r""":math:`\rho_{ij}` of the k-sampled system for sites given as (alpha, cell):
        :math:`\sum_k w_k e^{-i\theta_k (n_j - n_i)} \rho_{\theta_k}[\alpha_i, \alpha_j]`.
        """
        (a, n), (b, m) = site_i, site_j
        vals = _host(rho[a, :, b])
        return complex(
            np.sum(
                self.kmesh.weights * np.exp(-1j * self.kmesh.thetas * (m - n)) * vals
            )
        )

    def hopping(
        self, t: float, site_i: tuple[int, int], site_j: tuple[int, int]
    ) -> complex:
        """H_ij(t) of the infinite system for sites given as (alpha, cell)."""
        (a, n), (b, m) = site_i, site_j
        if m - n not in (-1, 0, 1):
            return 0j
        M = self.split_matrices_at_time(t)[{0: 0, 1: 1, -1: 2}[m - n]]
        return complex(_host(M.toarray())[a, b])

    def bond_current(
        self, rho: "B.Array", t: float, site_i: tuple[int, int], site_j: tuple[int, int]
    ) -> float:
        r""":math:`I_{i\leftarrow j} = 2\,\mathrm{Im}(H_{ij}\rho_{ji})` for sites given as (alpha, cell)."""
        return 2.0 * float(
            np.imag(
                self.hopping(t, site_i, site_j) * self.correlator(rho, site_j, site_i)
            )
        )


class RingHamiltonian(Hamiltonian):
    """Real-space Hamiltonian of the ring that is exactly equivalent to a BlochHamiltonian with a
    uniform k-mesh (N cells, seam twist 2 pi * shift). Built from the Bloch blocks, so the two
    descriptions cannot get out of sync."""

    def __init__(self, bloch: BlochHamiltonian) -> None:
        super().__init__()
        self.bloch = bloch
        self.geometry = bloch.ring_geometry
        alpha, n, _ = bloch._ring_sites()
        n_cells, _ = bloch._require_ring()
        nn = np.asarray(self.geometry.nearest_neighbors)
        rows = np.concatenate([nn[:, 0], nn[:, 1], np.arange(alpha.size)])
        cols = np.concatenate([nn[:, 1], nn[:, 0], np.arange(alpha.size)])
        self._rows, self._cols = rows, cols  # pattern: all bonds + diagonal

    def _build(self, t: float, derivative: bool = False):
        vals = self.bloch._ring_hamiltonian_elements(
            t, self._rows, self._cols, derivative=derivative
        )
        N = self.geometry.Lx * self.geometry.Ly
        xp, sp = B.xp(), B.xp_sparse()
        M = sp.coo_matrix(
            (
                xp.asarray(vals.astype(np.complex128)).astype(B.DTYPE),
                (xp.asarray(self._rows), xp.asarray(self._cols)),
            ),
            shape=(N, N),
        ).tocsr()
        M.eliminate_zeros()
        return M

    def at_time(self, t: float):
        return self._build(t)

    def derivative_at_time(self, t: float):
        return self._build(t, derivative=True)
