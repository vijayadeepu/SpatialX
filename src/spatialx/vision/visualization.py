"""Hand skeleton visualization for SpatialX."""

from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

from spatialx.vision.hand_landmarker import (
    LANDMARK_COUNT,
    HAND_CONNECTIONS,
    TrackedHand,
)

LEFT_HAND_COLOR = (0, 255, 0)  # green
RIGHT_HAND_COLOR = (0, 0, 255)  # red


def draw_hand_skeleton(
    image: np.ndarray,
    hands: Sequence[TrackedHand],
    connections=None,
    left_color: tuple = LEFT_HAND_COLOR,
    right_color: tuple = RIGHT_HAND_COLOR,
) -> np.ndarray:
    """Draw hand skeletons (landmarks + connections) onto a BGR frame.

    Args:
        image: OpenCV BGR frame.
        hands: Tracked hands to draw.
        connections: Iterable of ``(start, end)`` landmark index pairs.
            Defaults to the MediaPipe hand skeleton.
        left_color: Joint/connection color for left hands.
        right_color: Joint/connection color for right hands.

    Returns:
        A copy of the image annotated with the hand skeletons.
    """
    annotated = image.copy()
    height, width = annotated.shape[:2]
    connections = list(connections) if connections is not None else HAND_CONNECTIONS
    for hand in hands:
        color = left_color if hand.handedness == "LEFT" else right_color
        points: list[tuple[int, int]] = []
        for landmark in hand.landmarks[:LANDMARK_COUNT]:
            x = int(round((landmark.x or 0.0) * width))
            y = int(round((landmark.y or 0.0) * height))
            points.append((x, y))
            cv2.circle(annotated, (x, y), radius=4, color=color, thickness=2)
        for connection in connections:
            start = int(getattr(connection, "start", 0))
            end = int(getattr(connection, "end", 0))
            if start < len(points) and end < len(points):
                cv2.line(annotated, points[start], points[end], color=color, thickness=2)
    return annotated
