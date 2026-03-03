from abc import ABC, abstractmethod
from . import backend as B
from dataclasses import dataclass


@dataclass
class MeasurementWindow:
    start_time: float = 0.0
    end_time: float = float("inf")
    stride: int = 1
    
    def should_measure(self, t: float, step_index: int) -> bool:
        return (
            self.start_time <= t <= self.end_time
            and step_index % self.stride == 0
        )


class Observable(ABC):
    """Base class for observables measured during time evolution.

    Subclasses only need to implement ``_compute``.
    The base class handles accumulation and GPU-to-CPU transfer (``finalize``).
    """

    def __init__(self, window: MeasurementWindow | None = None) -> None:
        self.window = window if window is not None else MeasurementWindow()
        self._results: list[B.Array] = []
        self._times: list[float] = []

    @abstractmethod
    def _compute(self, rho: B.Array, t: float) -> B.Array:
        """Compute and return the measurement value at the current time step."""
        ...

    def measure(self, rho: B.Array, t: float, step_index: int) -> None:
        if not self.window.should_measure(t, step_index):
            return
        # .copy() ensures we store an independent snapshot even if _compute
        # returns a view into rho (e.g. diag, real-part slices).
        self._results.append(self._compute(rho, t).copy())
        self._times.append(t)

    def finalize(self) -> None:
        """Stack accumulated results and move to CPU if needed."""
        stacked = B.xp().stack(self._results) if self._results else B.xp().empty((0,))
        times = B.xp().array(self._times, dtype=B.FDTYPE)
        if B.USE_GPU:
            self.values = stacked.get()
            self.measurement_times = times.get()
        else:
            self.values = stacked
            self.measurement_times = times

    def reset(self) -> None:
        """Clear accumulated results and times."""
        self._results.clear()
        self._times.clear()
        self.values = None
        self.measurement_times = None
