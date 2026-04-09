import numpy as np

from .units import DEFAULT_A_NN_M, DEFAULT_T_HOP_EV, current_unit_amperes

from .observables import BondCurrentObservable
from .lattice_2d_geometry import Lattice2DGeometry



def net_current_vectors(current_obs: BondCurrentObservable, geometry: Lattice2DGeometry) -> np.ndarray:
    """Secondary observable that computes the net current (sum of all bond currents) in the system at each time step.
    Args:
        current_obs: The BondCurrentObservable instance from which to compute net currents.
        geometry: The Lattice2DGeometry instance defining the lattice structure.
    Returns:
        A (n, 2) array of net current vectors at each time, where n is len(current_obs.measurement_times).
    """
    bond_vectors = np.asarray(geometry.bond_vectors, dtype=float)
        
    return np.array([
        np.sum(bond_vectors * np.asarray(currents_t, dtype=float)[:, np.newaxis], axis=0)
        for currents_t in current_obs.values
    ])


def calculate_biot_savart_vectorized(
    r_obs,
    currents,
    geometry,
    *,
    n_images: int = 2,
    t_hop_ev: float | None = None,
    a_nn_m: float | None = None,
):
    """
    Vectorized calculation of B field.
    
    Args:
        r_obs: [x, y, z] in units of a_NN.
        currents: array-like of shape (N_bonds,) in units of one bond-current quantum.
        geometry: Lattice2DGeometry instance defining the lattice structure.
        n_images: image range for periodic replication. Uses shifts from
            -n_images to +n_images along periodic axes.
        t_hop_ev: nearest-neighbor hopping energy in eV. Used to convert current units.
        a_nn_m: nearest-neighbor distance in meters. Used to convert geometry units.
    Returns:
        B_field: np.ndarray of shape (3,) representing the magnetic field vector at r_obs.
    """

    return calculate_biot_savart_batch(
        np.asarray(r_obs)[np.newaxis, :],
        currents,
        geometry,
        n_images=n_images,
        t_hop_ev=t_hop_ev,
        a_nn_m=a_nn_m,
    )[0]


def calculate_biot_savart_batch(
    r_obs_array,
    currents,
    geometry,
    block_size=256,
    *,
    n_images: int = 2,
    t_hop_ev: float | None = None,
    a_nn_m: float | None = None,
):
    """Compute the magnetic field for many observation points at once.

    Args:
        r_obs_array: array-like of shape (N_obs, 3), points in units of a_NN.
        currents: array-like of shape (N_bonds,) in units of one bond-current quantum.
        geometry: Lattice2DGeometry instance defining the lattice structure.
        n_images: image range for periodic replication. Uses shifts from
            -n_images to +n_images along periodic axes.
        block_size: number of observation points processed per chunk.
        t_hop_ev: nearest-neighbor hopping energy in eV. Used to convert current units.
        a_nn_m: nearest-neighbor distance in meters. Used to convert geometry units.
    Returns:
        np.ndarray of shape (N_obs, 3) in Tesla.

    Unit trace:
        ``r = a_NN * r_hat`` and ``J = I_0 * J_hat`` with
        ``I_0 = e * t_hop / hbar``.
        The Biot-Savart kernel is evaluated on the dimensionless positions
        ``r_hat`` and then rescaled by ``mu_0 / (4 pi) * I_0 / a_NN``.
        The final field is therefore returned in Tesla.
    """

    r_obs_array = np.asarray(r_obs_array, dtype=float)
    if r_obs_array.ndim == 1:
        r_obs_array = r_obs_array[np.newaxis, :]
    if r_obs_array.shape[1] != 3:
        raise ValueError("r_obs_array must have shape (N_obs, 3)")

    r_i, r_k, J = geometry.prepare_3d_current_segments(
        np.asarray(currents, dtype=float),
        n_images=n_images,
        current_threshold=1e-10,
    )

    n_obs = r_obs_array.shape[0]
    if J.size == 0:
        return np.zeros((n_obs, 3), dtype=float)

    mu0_4pi = 1e-7
    if t_hop_ev is None and a_nn_m is None:
        I0 = current_unit_amperes()
        a_nn_m = DEFAULT_A_NN_M
    else:
        if t_hop_ev is None:
            t_hop_ev = DEFAULT_T_HOP_EV
        if a_nn_m is None:
            a_nn_m = DEFAULT_A_NN_M
        I0 = current_unit_amperes(t_hop_ev)

    prefactor = mu0_4pi * I0 / a_nn_m

    B_total = np.zeros((n_obs, 3), dtype=float)

    for start in range(0, n_obs, block_size):
        stop = min(start + block_size, n_obs)
        r = r_obs_array[start:stop, np.newaxis, :]  # (B, 1, 3)

        u = r - r_i[np.newaxis, :, :]  # (B, N_bonds, 3)
        v = r - r_k[np.newaxis, :, :]  # (B, N_bonds, 3)

        len_u = np.linalg.norm(u, axis=2)
        len_v = np.linalg.norm(v, axis=2)
        cross_uv = np.cross(u, v, axis=2)
        norm_cross_sq = np.sum(cross_uv**2, axis=2)
        uv = np.sum(u * v, axis=2)

        denom = len_u * len_v
        with np.errstate(divide="ignore", invalid="ignore"):
            scalar_term = (len_u + len_v) * (1.0 - (uv / denom))

        valid_obs = (norm_cross_sq > 1e-18) & (denom > 1e-18)
        with np.errstate(divide="ignore", invalid="ignore"):
            coeff = np.where(
                valid_obs,
                prefactor * J[np.newaxis, :] * scalar_term / norm_cross_sq,
                0.0,
            )

        B_segments = coeff[:, :, np.newaxis] * cross_uv
        B_total[start:stop] = np.sum(B_segments, axis=1)

    return B_total


def biot_savart_on_plane(
    x_values,
    y_values,
    z_height,
    currents,
    geometry,
    block_size=256,
    *,
    n_images: int = 2,
    t_hop_ev: float | None = None,
    a_nn_m: float | None = None,
):
    """Evaluate B on a rectangular plane parallel to the ribbon.

    Returns X, Y meshgrids and B_grid with shape (len(y_values), len(x_values), 3).
    """
    X, Y = np.meshgrid(np.asarray(x_values), np.asarray(y_values), indexing="xy")
    points = np.column_stack((X.ravel(), Y.ravel(), np.full(X.size, z_height)))
    B_flat = calculate_biot_savart_batch(
        points,
        currents,
        geometry,
        n_images=n_images,
        block_size=block_size,
        t_hop_ev=t_hop_ev,
        a_nn_m=a_nn_m,
    )
    B_grid = B_flat.reshape(X.shape + (3,))
    return X, Y, B_grid


__all__ = [
    "net_current_vectors",
    "calculate_biot_savart_vectorized",
    "calculate_biot_savart_batch",
    "biot_savart_on_plane",
]