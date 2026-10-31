"""Motion filtering subsystem for SpatialX: One Euro filter."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OneEuroFilter:
    """One Euro (α-β) low-pass filter with configurable cutoffs.

    Implements the formulation from Smith (2018):
        tau = 1 / (2 * pi * min_cutoff)
        alpha = dt / (dt + tau)
        dx = (x - x_prev) / dt
        tau_d = 1 / (2 * pi * derivative_cutoff)
        alpha_d = dt / (dt + tau_d)
        dx_hat = alpha_d * dx + (1 - alpha_d) * dx_hat_prev
        x_hat = x_hat + alpha * (x - x_hat) + (alpha * (1 - alpha) / beta) * (dx_hat - dx_hat_prev)

    Attributes:
        min_cutoff: Minimum cutoff frequency in Hz.
        beta: Filter speed / derivative weighting parameter (must be > 0).
        derivative_cutoff: Derivative filter cutoff frequency in Hz.
    """

    min_cutoff: float = 0.005
    beta: float = 0.003
    derivative_cutoff: float = 0.001
    x: float = 0.0
    x_prev: float = 0.0
    dx_hat: float = 0.0
    dx_hat_prev: float = 0.0
    t_prev: Optional[float] = None

    def __post_init__(self) -> None:
        if self.min_cutoff <= 0.0:
            raise ValueError("min_cutoff must be positive.")
        if self.beta <= 0.0:
            raise ValueError("beta must be positive.")
        if self.derivative_cutoff <= 0.0:
            raise ValueError("derivative_cutoff must be positive.")

    def reset(self) -> None:
        """Reset the filter state."""
        self.x = 0.0
        self.x_prev = 0.0
        self.dx_hat = 0.0
        self.dx_hat_prev = 0.0
        self.t_prev = None

    def filter(self, x_measured: float, t: Optional[float] = None) -> float:
        """Filter a scalar measurement.

        Args:
            x_measured: New measured value.
            t: Optional sample timestamp (seconds); defaults to monotonic clock.

        Returns:
            The filtered value.
        """
        now = float(t) if t is not None else time.monotonic()
        if self.t_prev is None:
            self.t_prev = now
            self.x = self.x_prev = float(x_measured)
            return float(x_measured)
        dt = max(now - self.t_prev, 1e-6)
        tau = 1.0 / (2.0 * math.pi * self.min_cutoff)
        alpha = dt / (dt + tau)
        raw_derivative = (float(x_measured) - self.x_prev) / dt
        tau_d = 1.0 / (2.0 * math.pi * self.derivative_cutoff)
        alpha_d = dt / (dt + tau_d)
        dx_hat = alpha_d * raw_derivative + (1.0 - alpha_d) * self.dx_hat
        correction = (alpha * (1.0 - alpha) / self.beta) * (dx_hat - self.dx_hat_prev)
        x_new = self.x + alpha * (float(x_measured) - self.x) + correction
        self.x = x_new
        self.x_prev = float(x_measured)
        self.dx_hat = dx_hat
        self.dx_hat_prev = dx_hat
        self.t_prev = now
        return x_new


def create_one_euro_filter(
    min_cutoff: float = 0.005,
    beta: float = 0.003,
    derivative_cutoff: float = 0.001,
) -> OneEuroFilter:
    """Factory for a configured One Euro filter."""
    return OneEuroFilter(
        min_cutoff=min_cutoff,
        beta=beta,
        derivative_cutoff=derivative_cutoff,
    )
