"""One-Euro filter (Casiez et al. 2012): heavy smoothing when still, low lag when moving."""

import math
from typing import Optional

import numpy as np


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 0.7, beta: float = 0.04, d_cutoff: float = 1.0) -> None:
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self) -> None:
        self._x: Optional[np.ndarray] = None
        self._dx: Optional[np.ndarray] = None
        self._t: Optional[float] = None

    @staticmethod
    def _alpha(dt: float, cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, np.float64)
        if self._x is None:
            self._x, self._dx, self._t = x.copy(), np.zeros_like(x), t
            return x.astype(np.float32)
        dt = max(t - self._t, 1e-3)
        dx = (x - self._x) / dt
        a_d = self._alpha(dt, self.d_cutoff)
        self._dx = a_d * dx + (1 - a_d) * self._dx
        # one speed per corner, so x and y of a corner share the same cutoff
        speed = np.linalg.norm(self._dx.reshape(-1, 2), axis=1, keepdims=True)
        speed = np.repeat(speed, 2, axis=1).reshape(x.shape)
        a = self._alpha(dt, self.min_cutoff + self.beta * speed)
        self._x = a * x + (1 - a) * self._x
        self._t = t
        return self._x.astype(np.float32)
