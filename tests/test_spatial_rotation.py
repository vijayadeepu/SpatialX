"""Milestone 5 — quaternion rotation mathematics tests."""

from __future__ import annotations

import math

import numpy as np

from spatialx.spatial import (
    Quaternion,
    quaternion_conjugate,
    quaternion_multiply,
    rotate_vector,
    rotation_vector_to_quaternion,
    slerp,
    smooth_rotation,
)


def test_rotation_vector_to_quaternion_identity() -> None:
    q = rotation_vector_to_quaternion(np.array([0.0, 0.0, 0.0]))
    assert q.w == 1.0
    assert q.x == 0.0 and q.y == 0.0 and q.z == 0.0


def test_rotation_vector_to_quaternion_z_axis() -> None:
    q = rotation_vector_to_quaternion(np.array([0.0, 0.0, math.pi / 2]))
    assert q.w == math.cos(math.pi / 4)
    assert abs(q.z - math.sin(math.pi / 4)) < 1e-9
    assert q.x == 0.0 and q.y == 0.0


def test_slerp_endpoints() -> None:
    q1 = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
    q2 = Quaternion(w=math.cos(0.5 * math.pi / 2), x=0.0, y=math.sin(0.5 * math.pi / 2), z=0.0)
    assert slerp(q1, q2, 0.0) == q1
    assert slerp(q1, q2, 1.0) == q2
    mid = slerp(q1, q2, 0.5)
    assert abs(mid.w - math.cos(math.pi / 8)) < 1e-9


def test_slerp_same_quaternion() -> None:
    q = Quaternion(w=math.cos(0.4), x=0.0, y=math.sin(0.4), z=0.0)  # unit quaternion
    assert abs(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z - 1.0) < 1e-9
    result = slerp(q, q, 0.7)
    assert abs(result.w - q.w) < 1e-6
    assert abs(result.y - q.y) < 1e-6
    assert abs(result.z - q.z) < 1e-6


def test_rotate_vector_identity() -> None:
    q = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
    assert np.allclose(rotate_vector(q, np.array([1.0, 2.0, 3.0])), [1.0, 2.0, 3.0])


def test_rotate_vector_z_rotation() -> None:
    q = rotation_vector_to_quaternion(np.array([0.0, 0.0, math.pi / 2]))
    rotated = rotate_vector(q, np.array([1.0, 0.0, 0.0]))
    assert np.allclose(rotated, [0.0, 1.0, 0.0])


def test_composition_applies_q1_then_q2() -> None:
    q1 = rotation_vector_to_quaternion(np.array([0.0, 0.0, math.pi / 2]))
    q2 = rotation_vector_to_quaternion(np.array([0.0, 0.0, math.pi / 2]))
    composed = quaternion_multiply(q1, q2)
    direct = rotate_vector(composed, np.array([1.0, 0.0, 0.0]))
    sequential = rotate_vector(q2, rotate_vector(q1, np.array([1.0, 0.0, 0.0])))
    assert np.allclose(direct, sequential)


def test_smooth_rotation_returns_valid_unit_quaternion() -> None:
    q1 = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
    q2 = Quaternion(w=math.cos(0.5 * math.pi / 2), x=0.0, y=math.sin(0.5 * math.pi / 2), z=0.0)
    q = smooth_rotation(q1, q2, smoothing=0.25)
    norm = math.sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z)
    assert abs(norm - 1.0) < 1e-9


def test_quaternion_conjugate() -> None:
    q = Quaternion(w=0.5, x=0.5, y=0.5, z=0.5)
    qc = quaternion_conjugate(q)
    assert qc.w == q.w
    assert qc.x == -q.x and qc.y == -q.y and qc.z == -q.z
