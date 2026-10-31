"""Milestone 2 — spatial coordinate engine tests."""

from __future__ import annotations

import numpy as np
import pytest

from spatialx.spatial import (
    CameraCalibration,
    hand_relative_vectors,
    hand_velocity,
    landmark_camera_space,
    palm_orientation,
    relative_depth,
)
from spatialx.vision import LANDMARK_COUNT, Landmark, TrackedHand


def _make_hand(landmarks: list[tuple[float, float, float]]) -> TrackedHand:
    """Build a TrackedHand from (x, y, z) tuples."""
    return TrackedHand(
        handedness="LEFT",
        landmarks=tuple(
            Landmark(index=i, x=x, y=y, z=z, visibility=1.0) for i, (x, y, z) in enumerate(landmarks)
        ),
    )


def _make_flat_hand(overrides: dict[int, tuple[float, float, float]]) -> TrackedHand:
    values = [(0.0, 0.0, 0.0)] * LANDMARK_COUNT
    for index, value in overrides.items():
        values[index] = value
    return TrackedHand(
        handedness="LEFT",
        landmarks=tuple(
            Landmark(index=i, x=x, y=y, z=z, visibility=1.0) for i, (x, y, z) in enumerate(values)
        ),
    )


CALIBRATION = CameraCalibration(
    focal_length_x=500.0,
    focal_length_y=500.0,
    principal_point_x=320.0,
    principal_point_y=240.0,
)


def test_landmark_camera_space_normalized_mapping() -> None:
    landmark = Landmark(index=0, x=0.5, y=0.5, z=0.1, visibility=1.0)
    coords = landmark_camera_space(landmark, CALIBRATION)
    assert np.allclose(coords, [0.0, 0.0, 0.1])


def test_landmark_camera_space_origin_center() -> None:
    landmark = Landmark(index=0, x=0.5, y=0.5, z=0.0, visibility=1.0)
    coords = landmark_camera_space(landmark, CALIBRATION)
    assert np.allclose(coords, [0.0, 0.0, 0.0])


def test_landmark_camera_space_nonzero_offset() -> None:
    landmark = Landmark(index=0, x=0.25, y=0.75, z=0.0, visibility=1.0)
    coords = landmark_camera_space(landmark, CALIBRATION)
    # x_px = 160, y_px = 360; X = (160 - 320) * 500 / 640 = -125, Y = (360 - 240) * 500 / 480 = 125
    assert np.allclose(coords, [-125.0, 125.0, 0.0])


def test_landmark_camera_space_defaults() -> None:
    landmark = Landmark(index=0, x=0.0, y=0.0, z=0.0, visibility=1.0)
    coords = landmark_camera_space(landmark)
    assert coords.shape == (3,)


def test_hand_relative_vectors_from_wrist() -> None:
    landmarks = [(0.0, 0.0, 0.0)] + [(0.1, 0.2, 0.3)] * (LANDMARK_COUNT - 1)
    hand = _make_hand(landmarks)
    vectors = hand_relative_vectors(hand)
    assert vectors.shape == (LANDMARK_COUNT - 1, 3)
    assert np.allclose(vectors[0], [0.1, 0.2, 0.3])


def test_relative_depth_computation() -> None:
    landmarks = [(0.0, 0.0, 0.0), (0.1, 0.0, 0.5), (0.2, 0.0, 0.25)]
    hand = _make_hand(landmarks)
    depths = relative_depth(hand)
    assert depths.shape == (2, 3)
    assert np.allclose(depths[0], [0.1, 0.0, 0.5])
    assert np.allclose(depths[1], [0.2, 0.0, 0.25])


def test_palm_orientation_rotation_vector() -> None:
    # Two distinct palm planes must produce a non-zero rotation vector.
    hand_a = _make_flat_hand({5: (0.2, 0.1, 0.0), 9: (0.05, 0.25, 0.0)})
    hand_b = _make_flat_hand({5: (0.4, 0.2, 0.0), 9: (0.1, 0.3, 0.0)})
    rot_a = palm_orientation(hand_a, CALIBRATION)
    rot_b = palm_orientation(hand_b, CALIBRATION)
    assert rot_a.shape == (3,)
    assert rot_b.shape == (3,)
    assert np.linalg.norm(rot_a) > 0.0
    assert np.linalg.norm(rot_b) > 0.0
    assert not np.allclose(rot_a, rot_b)


def test_palm_orientation_degenerate_hand() -> None:
    hand = _make_hand([(0.0, 0.0, 0.0)] * LANDMARK_COUNT)
    assert palm_orientation(hand, CALIBRATION).shape == (3,)


def test_hand_velocity_displacement_per_second() -> None:
    velocity = hand_velocity(np.zeros(3), np.array([0.0, 0.0, 0.0]), 0.0)
    assert np.allclose(velocity, [0.0, 0.0, 0.0])
    velocity = hand_velocity(np.zeros(3), np.array([1.0, 2.0, 3.0]), 2.0)
    assert np.allclose(velocity, [0.5, 1.0, 1.5])


def test_camera_calibration_defaults() -> None:
    cal = CameraCalibration()
    matrix = cal.to_camera_matrix()
    assert matrix.shape == (3, 3)
    assert matrix[0, 0] == 500.0
    assert matrix[0, 2] == 320.0
    assert matrix[1, 1] == 500.0
    assert matrix[1, 2] == 240.0
