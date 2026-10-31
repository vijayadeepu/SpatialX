"""Milestone 2 — spatial coordinate engine tests (audited conventions)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from spatialx.spatial import (
    CameraCalibration,
    camera_ray,
    hand_relative_vectors,
    hand_velocity,
    landmark_camera_space,
    normalized_to_pixel,
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
    return _make_hand(values)


CALIBRATION = CameraCalibration(
    focal_length_x=500.0,
    focal_length_y=500.0,
    principal_point_x=320.0,
    principal_point_y=240.0,
)


# --- normalized -> pixel conversion ----------------------------------------


def test_normalized_to_pixel_basic() -> None:
    landmark = Landmark(index=0, x=0.25, y=0.75, z=0.1, visibility=1.0)
    u, v, w = normalized_to_pixel(landmark, image_width=640.0, image_height=480.0)
    assert np.allclose([u, v, w], [160.0, 360.0, 64.0])


def test_normalized_to_pixel_mirrored() -> None:
    landmark = Landmark(index=0, x=0.25, y=0.75, z=0.1, visibility=1.0)
    u, v, w = normalized_to_pixel(
        landmark, image_width=640.0, image_height=480.0, mirror=True
    )
    # u = (1 - 0.25) * 640 = 480
    assert np.allclose([u, v, w], [480.0, 360.0, 64.0])


# --- camera space -----------------------------------------------------------


def test_landmark_camera_space_center_maps_to_principal_point() -> None:
    landmark = Landmark(index=0, x=0.5, y=0.5, z=0.1, visibility=1.0)
    coords = landmark_camera_space(landmark, CALIBRATION)
    # X/Y = pixel offsets from principal point; Z = width-scaled wrist-relative depth.
    assert np.allclose(coords, [0.0, 0.0, 64.0])


def test_landmark_camera_space_pixel_offsets() -> None:
    landmark = Landmark(index=0, x=0.25, y=0.75, z=0.0, visibility=1.0)
    coords = landmark_camera_space(landmark, CALIBRATION)
    # u=160 -> X=-160; v=360 -> Y=120; z=0 -> Z=0
    assert np.allclose(coords, [-160.0, 120.0, 0.0])


def test_landmark_camera_space_mirrored() -> None:
    landmark = Landmark(index=0, x=0.25, y=0.75, z=0.0, visibility=1.0)
    coords = landmark_camera_space(landmark, CALIBRATION, mirror=True)
    # mirrored u=480 -> X=+160 (horizontal axis flips)
    assert np.allclose(coords, [160.0, 120.0, 0.0])


def test_landmark_camera_space_defaults_use_image_center() -> None:
    landmark = Landmark(index=0, x=0.5, y=0.5, z=0.0, visibility=1.0)
    coords = landmark_camera_space(landmark, image_width=640.0, image_height=480.0)
    assert np.allclose(coords, [0.0, 0.0, 0.0])


def test_landmark_camera_space_z_is_relative_not_metric() -> None:
    # Z must equal the width-scaled MediaPipe depth (relative to wrist),
    # never an absolute distance in meters.
    landmark = Landmark(index=0, x=0.5, y=0.5, z=-0.05, visibility=1.0)
    coords = landmark_camera_space(landmark, image_width=640.0, image_height=480.0)
    assert coords[2] == pytest.approx(-32.0)


# --- camera rays ------------------------------------------------------------


def test_camera_ray_principal_point_is_forward() -> None:
    ray = camera_ray(320.0, 240.0, CALIBRATION)
    assert np.allclose(ray, [0.0, 0.0, 1.0])


def test_camera_ray_is_unit_length() -> None:
    for u, v in ((0.0, 0.0), (640.0, 480.0), (123.0, 456.0)):
        ray = camera_ray(u, v, CALIBRATION)
        assert np.linalg.norm(ray) == pytest.approx(1.0)


def test_camera_ray_offset_direction() -> None:
    ray = camera_ray(820.0, 240.0, CALIBRATION)  # 500px right of center, fx=500
    expected = np.array([1.0, 0.0, 1.0]) / math.sqrt(2.0)  # (u-cx)/fx = 1
    assert np.allclose(ray, expected)


# --- calibration ------------------------------------------------------------


def test_camera_calibration_defaults() -> None:
    cal = CameraCalibration()
    matrix = cal.to_camera_matrix()
    assert matrix.shape == (3, 3)
    assert matrix[0, 0] == 500.0
    assert matrix[0, 2] == 320.0
    assert matrix[1, 1] == 500.0
    assert matrix[1, 2] == 240.0
    assert np.allclose(cal.effective_rotation, np.eye(3))
    assert np.allclose(cal.effective_translation, np.zeros(3))


def test_camera_calibration_for_image_nominal() -> None:
    cal = CameraCalibration.for_image(640.0, 480.0, horizontal_fov_degrees=60.0)
    assert cal.principal_point_x == 320.0
    assert cal.principal_point_y == 240.0
    expected_fx = 320.0 / math.tan(math.radians(30.0))
    assert cal.focal_length_x == pytest.approx(expected_fx)
    assert cal.focal_length_y == pytest.approx(expected_fx)


# --- hand-relative geometry -------------------------------------------------


def test_hand_relative_vectors_from_wrist() -> None:
    landmarks = [(0.5, 0.5, 0.0)] + [(0.6, 0.625, 0.01)] * (LANDMARK_COUNT - 1)
    hand = _make_hand(landmarks)
    vectors = hand_relative_vectors(hand, image_width=1000.0, image_height=1000.0)
    assert vectors.shape == (LANDMARK_COUNT - 1, 3)
    # Pixel space: dx=100px, dy=125px, dz=0.01*1000=10px
    assert np.allclose(vectors[0], [100.0, 125.0, 10.0])


def test_hand_relative_vectors_mirror_consistent() -> None:
    landmarks = [(0.4, 0.5, 0.0)] + [(0.5, 0.5, 0.0)] * (LANDMARK_COUNT - 1)
    hand = _make_hand(landmarks)
    raw = hand_relative_vectors(hand, image_width=640.0, image_height=480.0)
    mirrored = hand_relative_vectors(
        hand, image_width=640.0, image_height=480.0, mirror=True
    )
    # Mirroring flips the sign of the x component only.
    assert np.allclose(mirrored[:, 0], -raw[:, 0])
    assert np.allclose(mirrored[:, 1:], raw[:, 1:])


def test_relative_depth_wrist_relative_width_scaled() -> None:
    landmarks = [(0.0, 0.0, 0.0), (0.1, 0.0, 0.5), (0.2, 0.0, 0.25)]
    hand = _make_hand(landmarks)
    depths = relative_depth(hand, image_width=640.0)
    assert depths.shape == (2,)
    assert np.allclose(depths, [320.0, 160.0])


# --- palm orientation -------------------------------------------------------


def test_palm_orientation_known_geometry() -> None:
    # wrist -> index MCP along +x, wrist -> middle MCP along +y (pixel space):
    # rotation of +90 degrees about +z.
    hand = _make_flat_hand(
        {0: (0.5, 0.5, 0.0), 5: (0.6, 0.5, 0.0), 9: (0.5, 0.6, 0.0)}
    )
    rotation = palm_orientation(hand, image_width=1000.0, image_height=1000.0)
    assert np.allclose(rotation, [0.0, 0.0, math.pi / 2], atol=1e-9)


def test_palm_orientation_axis_is_unit_scaled_by_angle() -> None:
    # Magnitude must equal the angle between the palm basis vectors.
    hand = _make_flat_hand(
        {0: (0.5, 0.5, 0.0), 5: (0.6, 0.5, 0.0), 9: (0.5, 0.6, 0.0)}
    )
    rotation = palm_orientation(hand, image_width=1000.0, image_height=1000.0)
    assert np.linalg.norm(rotation) == pytest.approx(math.pi / 2)


def test_palm_orientation_uses_pixel_space_not_normalized() -> None:
    # In raw normalized coordinates these basis vectors are perpendicular
    # (90 deg); in pixel space (640x480) they are not. The corrected
    # implementation must produce the pixel-space angle.
    hand = _make_flat_hand(
        {0: (0.5, 0.5, 0.0), 5: (0.6, 0.6, 0.0), 9: (0.4, 0.6, 0.0)}
    )
    rotation = palm_orientation(hand, image_width=640.0, image_height=480.0)
    a = np.array([64.0, 48.0, 0.0])
    b = np.array([-64.0, 48.0, 0.0])
    expected_angle = math.atan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b))
    assert expected_angle > math.pi / 2  # pixel-space angle exceeds 90 deg
    assert np.linalg.norm(rotation) == pytest.approx(expected_angle)


def test_palm_orientation_distinct_hands_differ() -> None:
    hand_a = _make_flat_hand({5: (0.2, 0.1, 0.0), 9: (0.05, 0.25, 0.0)})
    hand_b = _make_flat_hand({5: (0.4, 0.2, 0.0), 9: (0.1, 0.3, 0.0)})
    rot_a = palm_orientation(hand_a)
    rot_b = palm_orientation(hand_b)
    assert rot_a.shape == (3,)
    assert rot_b.shape == (3,)
    assert np.linalg.norm(rot_a) > 0.0
    assert np.linalg.norm(rot_b) > 0.0
    assert not np.allclose(rot_a, rot_b)


def test_palm_orientation_degenerate_hand() -> None:
    hand = _make_hand([(0.0, 0.0, 0.0)] * LANDMARK_COUNT)
    assert np.allclose(palm_orientation(hand), np.zeros(3))


# --- velocity ---------------------------------------------------------------


def test_hand_velocity_displacement_per_second() -> None:
    velocity = hand_velocity(np.zeros(3), np.array([0.0, 0.0, 0.0]), 0.0)
    assert np.allclose(velocity, [0.0, 0.0, 0.0])
    velocity = hand_velocity(np.zeros(3), np.array([1.0, 2.0, 3.0]), 2.0)
    assert np.allclose(velocity, [0.5, 1.0, 1.5])
