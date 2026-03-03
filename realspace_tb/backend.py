"""
Goal of this file is to provide the correct backend (numpy or, for GPU, cupy) and data types (single, double) so that the rest of the code can be written simultaneously for both backends.
Key functions:
    xp() returns either numpy or cupy module.
    set_backend() allows switching between CPU and GPU backends as well as single and double precision.
"""

from numpy.typing import NDArray
import numpy as np
from scipy.sparse import csr_matrix
from scipy import sparse
from typing import TypeAlias
import types

try:
    import cupy as cp
    try:
        import cupy.sparse as cpsp
    except ImportError:
        import cupyx.scipy.sparse as cpsp # type: ignore
    _cupy_available = True
except ImportError:
    cp = None
    cpsp = None
    _cupy_available = False

USE_GPU: bool = False

def xp() -> types.ModuleType:
    """Return the active array module (NumPy or CuPy)."""
    return cp if USE_GPU else np

def xp_sparse() -> types.ModuleType:
    """Return the active sparse array module (scipy sparse or cupy sparse)."""
    return cpsp if USE_GPU else sparse

DTYPE: type = xp().complex128
FDTYPE: type = xp().float64
FCPUDTYPE: type = np.float64

Array: TypeAlias = "NDArray[np.complex128] | NDArray[np.complex64] | NDArray[np.float64] | NDArray[np.float32] | cp.ndarray"
SparseArray: TypeAlias = "csr_matrix | cpsp.spmatrix"
CPUArray: TypeAlias = NDArray[np.complex128] | NDArray[np.complex64] | NDArray[np.float64] | NDArray[np.float32]
FCPUArray: TypeAlias = NDArray[np.float64] | NDArray[np.float32]

def set_backend(use_gpu: bool = False, precision: str = "double") -> None:
    """
    Select computation backend and precision.

    Args:
        use_gpu: whether to use CuPy (GPU) or NumPy (CPU)
        precision: 'single' or 'double'
    """
    global USE_GPU, DTYPE, FDTYPE, FCPUDTYPE
    USE_GPU = use_gpu

    if use_gpu and not _cupy_available:
        raise RuntimeError("CuPy not available on this system.")

    if precision == "single":
        DTYPE = xp().complex64
        FDTYPE = xp().float32
        FCPUDTYPE = np.float32
    elif precision == "double":
        DTYPE = xp().complex128
        FDTYPE = xp().float64
        FCPUDTYPE = np.float64
    else:
        raise ValueError("Precision must be 'single' or 'double'.")
