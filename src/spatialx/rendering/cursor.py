"""Virtual spatial cursor for SpatialX."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from spatialx.motion import OneEuroFilter, create_one_euro_filter


@dataclass
class CursorState:
    """Current virtual cursor state."""

    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    visible: bool = True


def estimate_jitter(positions: Sequence[np.ndarray]) -> float:
    """Estimate cursor jitter from a series of 3D positions.

    Jitter is measured as the mean absolute discrete acceleration
    (second difference) of the position series. A perfectly linear
    trajectory yields zero jitter; noisy samples yield larger values.

    Args:
        positions: Sequence of 3-element position vectors.

    Returns:
        Mean absolute second difference across the series (0.0 for
        insufficient data).
    """
    if len(positions) < 3:
        return 0.0
    total = 0.0
    count = 0
    for index in range(2, len(positions)):
        acceleration = np.abs(
            np.asarray(positions[index], dtype=float)
            - 2.0 * np.asarray(positions[index - 1], dtype=float)
            + np.asarray(positions[index - 2], dtype=float)
        )
        total += float(np.mean(acceleration))
        count += 1
    return total / count if count else 0.0


class VirtualCursor:
    """Maps hand camera-space origins to filtered virtual cursor positions.

    Responsibilities:
        - Convert hand origin coordinates into cursor space.
        - Apply configurable sensitivity.
        - Clamp positions into configurable bounds.
        - Filter output with configurable One Euro filters (one per axis).
    """

    def __init__(
        self,
        filter: Optional[OneEuroFilter] = None,
        sensitivity: float = 1.0,
        bounds: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        source = filter if filter is not None else OneEuroFilter()
        self._filters = [
            create_one_euro_filter(
                min_cutoff=source.min_cutoff,
                beta=source.beta,
                derivative_cutoff=source.derivative_cutoff,
            )
            for _ in range(3)
        ]
        self.sensitivity = sensitivity
        self.bounds = bounds
        self.state = CursorState()

    def update(
        self,
        hand_origin_camera: np.ndarray,
        timestamp_s: Optional[float] = None,
    ) -> np.ndarray:
        """Compute the new filtered cursor position from a hand origin."""
        origin = np.asarray(hand_origin_camera, dtype=float).reshape(-1)
        if origin.size != 3:
            raise ValueError("hand_origin_camera must be a 3-element vector.")
        axes = [self._filters[axis].filter(float(origin[axis]), t=timestamp_s) for axis in range(3)]
        filtered = np.array(axes, dtype=float)
        position = filtered * float(self.sensitivity)
        if self.bounds is not None:
            position = np.clip(position, self.bounds[0], self.bounds[1])
        self.state.position = position.copy()
        return position

    def reset(self) -> None:
        """Reset the cursor filters and state."""
        for axis_filter in self._filters:
            axis_filter.reset()
        self.state.position = np.zeros(3)
        self.state.visible = True
