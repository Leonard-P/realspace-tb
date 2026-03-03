
import numpy as np
from .. import backend as B

# def fourier_at_omega(signal: B.FCPUArray, dt: float, omega: float) -> np.complexfloating:
#     N = len(signal)
#     n = np.arange(N)
#     t = n * dt
#     phase = np.exp(-1j * omega * t)
#     return np.sum(signal * phase) * dt

def fourier_at_omega(signal: B.FCPUArray, dt: float, omega: float) -> np.ndarray:
    """
    Calculates the discrete Fourier component at omega for a signal of arbitrary shape.
    Assumes axis 0 is the time dimension.
    """
    arr: np.ndarray = np.asarray(signal)
    
    N = len(arr)
    t = np.arange(N) * dt
    phase = np.exp(-1j * omega * t)
    
    # Contract axis 0 (time) of signal with axis 0 of phase
    # Equivalent to: sum(signal[t, ...] * phase[t] for t)
    return np.tensordot(arr, phase, axes=([0], [0])) * dt

def ohc(orbital_current_values: B.FCPUArray, E_amplitude_values: B.FCPUArray, dt: float, omega: float) -> np.complexfloating:
    """Compute the orbital Hall conductivity from the ratio of the Fourier components at the driving frequency omega.
    Parameters:
        orbital_current_values: Array of measured orbital currents.
        E_amplitude_values: Electric field amplitude values at the same time steps.
        dt: Time step size.
        omega: Driving frequency at which the conductivity is measured.
    Returns:
        The orbital Hall conductivity as a complex number in units (e/2pi).
    """
    return 2 * np.pi * fourier_at_omega(orbital_current_values, dt, omega) / fourier_at_omega(E_amplitude_values, dt, omega)