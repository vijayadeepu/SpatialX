"""Spatial coordinate mathematics for SpatialX.

This module converts MediaPipe landmark observations into stable,
camera-referenced and hand-relative spatial coordinates.

Coordinate conventions
----------------------
- Camera frame (OpenCV convention): x → right, y → down, z → forward.
- Normalized landmark coordinates: ``x``/``y`` in [0, 1] (origin top-left),
  ``z`` = depth relative to the wrist (normalized units).
- World frame: right-handed, z-up (target for downstream engines).
- Hand-relative frame: origin at the wrist, axes derived from hand geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from spatialx.vision import Landmark, TrackedHand


@dataclass(frozen=True)
class CameraCalibration:
    """Intrinsic and extrinsic camera calibration used for spatial mapping."""

    focal_length_x: float = 500.0
    focal_length_y: float = 500.0
    principal_point_x: float = 320.0
    principal_point_y: float = 240.0
    rotation: np.ndarray = None  # type: ignore[assignment]
    translation: np.ndarray = None  # type: ignore[assignment]

    def to_camera_matrix(self) -> np.ndarray:
        """Return the 3x3 intrinsic camera matrix."""
        return np.array(
            [
                [self.focal_length_x, 0.0, self.principal_point_x],
                [0.0, self.focal_length_y, self.principal_point_y],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )


def landmark_camera_space(
    landmark: Landmark,
    calibration: Optional[CameraCalibration] = None,
    image_width: float = 640.0,
    image_height: float = 480.0,
) -> np.ndarray:
    """Convert a normalized landmark into camera space.

    Camera space vector: ``(X, Y, Z)`` with ``X`` right, ``Y`` down,
    ``Z`` forward. ``Z`` is the hand-relative depth.

    Returns:
        A 3-element numpy array.
    """
    cal = calibration or CameraCalibration()
    x_px = (landmark.x or 0.0) * image_width
    y_px = (landmark.y or 0.0) * image_height
    x = (x_px - cal.principal_point_x) * cal.focal_length_x / image_width
    y = (y_px - cal.principal_point_y) * cal.focal_length_y / image_height
    z = landmark.z if landmark.z is not None else 0.0
    return np.array([x, y, z], dtype=float)


def hand_relative_vectors(hand: TrackedHand) -> np.ndarray:
    """Compute vectors from the wrist (landmark 0) to every other landmark."""
    if not hand.landmarks:
        return np.zeros((0, 3), dtype=float)
    wrist = hand.landmarks[0]
    vectors = []
    for landmark in hand.landmarks[1:]:
        if landmark.x is None or landmark.y is None:
            continue
        vectors.append(
            np.array(
                [
                    landmark.x - wrist.x,
                    landmark.y - wrist.y,
                    (landmark.z or 0.0) - (wrist.z or 0.0),
                ],
                dtype=float,
            )
        )
    return np.array(vectors, dtype=float).reshape(len(vectors), 3)


def relative_depth(hand: TrackedHand) -> np.ndarray:
    """Compute hand-relative depth (landmark depth minus wrist depth)."""
    if len(hand.landmarks) < 2:
        return np.zeros((0, 3), dtype=float)
    wrist = hand.landmarks[0]
    depths = []
    for landmark in hand.landmarks[1:]:
        depths.append(
            np.array(
                [
                    (landmark.x or 0.0) - (wrist.x or 0.0),
                    (landmark.y or 0.0) - (wrist.y or 0.0),
                    (landmark.z or 0.0) - (wrist.z or 0.0),
                ],
                dtype=float,
            )
        )
    return np.array(depths, dtype=float).reshape(len(depths), 3)


def palm_orientation(
    hand: TrackedHand,
    calibration: Optional[CameraCalibration] = None,
) -> np.ndarray:
    """Estimate palm orientation as a rotation vector (axis * angle).

    Uses the wrist-to-index-MCP and wrist-to-middle-MCP vectors as the palm
    plane basis. Returns ``[0, 0, 0]`` for degenerate (flat) palm geometry.

    Returns:
        A 3-element rotation vector.
    """
    if len(hand.landmarks) < 10:
        return np.zeros(3, dtype=float)
    wrist = hand.landmarks[0]
    index_mcp = hand.landmarks[5]
    middle_mcp = hand.landmarks[9]
    a = np.array(
        [
            (index_mcp.x or 0.0) - (wrist.x or 0.0),
            (index_mcp.y or 0.0) - (wrist.y or 0.0),
            (index_mcp.z or 0.0) - (wrist.z or 0.0),
        ],
        dtype=float,
    )
    b = np.array(
        [
            (middle_mcp.x or 0.0) - (wrist.x or 0.0),
            (middle_mcp.y or 0.0) - (wrist.y or 0.0),
            (middle_mcp.z or 0.0) - (wrist.z or 0.0),
        ],
        dtype=float,
    )
    axis = np.cross(a, b)
    angle = float(np.arctan2(np.linalg.norm(axis), np.dot(a, b)))
    return np.array([axis[0] * angle, axis[1] * angle, axis[2] * angle], dtype=float)


def hand_velocity(
    previous_origin: np.ndarray,
    current_origin: np.ndarray,
    delta_seconds: float,
) -> np.ndarray:
    """Compute hand velocity (origin displacement per second)."""
    displacement = np.asarray(current_origin, dtype=float) - np.asarray(previous_origin, dtype=float)
    if delta_seconds <= 0.0:
        return np.zeros(3, dtype=float)
    return displacement / float(delta_seconds)
