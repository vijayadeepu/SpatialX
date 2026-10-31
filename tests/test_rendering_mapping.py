"""Priority 3 — camera→scene mapping tests (no Panda3D scene needed).

Safe to run in-process: importing ``spatialx.rendering`` no longer pulls
MediaPipe (lazy ``spatialx.spatial`` exports).
"""

from __future__ import annotations

import math

import numpy as np

from spatialx.rendering import (
    camera_quat_to_scene,
    camera_space_to_scene,
    scene_quat_to_camera,
    scene_space_to_camera,
)
from spatialx.spatial.quaternion import (
    Quaternion,
    rotate_vector,
    rotation_vector_to_quaternion,
)


def test_vector_mapping_basic() -> None:
    # camera (100, 0, 0) px -> scene (1, 0, 0) at scale 0.01
    assert np.allclose(camera_space_to_scene([100, 0, 0]), [1.0, 0.0, 0.0])
    # camera y (down) maps to scene -z (down is -Z up-axis)
    assert np.allclose(camera_space_to_scene([0, 50, 0]), [0.0, 0.0, -0.5])
    # camera z (forward) maps to scene +y (forward)
    assert np.allclose(camera_space_to_scene([0, 0, 50]), [0.0, 0.5, 0.0])


def test_vector_mapping_roundtrip() -> None:
    vector = np.array([123.0, -45.0, 67.0])
    scene = camera_space_to_scene(vector, scale=0.02)
    back = scene_space_to_camera(scene, scale=0.02)
    assert np.allclose(back, vector)


def test_axis_map_quaternion_matches_vector_map() -> None:
    # The constant axis-map quaternion must implement the same (x, z, -y)
    # transform as the vector mapping.
    from spatialx.rendering.scene import AXIS_MAP_QUATERNION

    for vector in (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.3, -1.7, 2.2]),
    ):
        expected = np.array([vector[0], vector[2], -vector[1]])
        rotated = rotate_vector(AXIS_MAP_QUATERNION, vector)
        assert np.allclose(rotated, expected, atol=1e-9)


def test_orientation_mapping_is_similarity_transform() -> None:
    # Mapping must preserve rotation behaviour: rotating a camera-space
    # vector by q then mapping equals mapping the vector then rotating by
    # the mapped quaternion.
    rng = np.random.default_rng(20261031)
    for _ in range(25):
        axis = rng.normal(size=3)
        angle = float(rng.uniform(-math.pi, math.pi))
        quat = rotation_vector_to_quaternion(axis / np.linalg.norm(axis) * angle)
        vector = rng.normal(size=3)
        mapped_quat = camera_quat_to_scene(quat)
        lhs = camera_space_to_scene(rotate_vector(quat, vector), scale=1.0)
        rhs = rotate_vector(mapped_quat, camera_space_to_scene(vector, scale=1.0))
        assert np.allclose(lhs, rhs, atol=1e-9)


def test_orientation_mapping_roundtrip() -> None:
    quat = rotation_vector_to_quaternion(np.array([0.2, -0.5, 0.8]))
    back = scene_quat_to_camera(camera_quat_to_scene(quat))
    # q and -q are the same rotation; compare up to sign.
    dot = (
        back.w * quat.w + back.x * quat.x + back.y * quat.y + back.z * quat.z
    )
    assert abs(abs(dot) - 1.0) < 1e-9


def test_identity_orientation_stays_identity() -> None:
    identity = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
    mapped = camera_quat_to_scene(identity)
    assert np.allclose([mapped.w, mapped.x, mapped.y, mapped.z], [1, 0, 0, 0])
