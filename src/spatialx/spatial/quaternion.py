"""3D rotation mathematics (quaternions) for SpatialX."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Quaternion:
    """Quaternion in (w, x, y, z) form."""

    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


def rotation_vector_to_quaternion(axis_angle: np.ndarray) -> Quaternion:
    """Convert a rotation vector (axis * angle) to a quaternion."""
    values = np.asarray(axis_angle, dtype=float).reshape(-1)
    axis = values[:3]
    angle = float(np.linalg.norm(axis))
    if angle < 1e-12:
        return Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
    axis = axis / angle
    half = 0.5 * angle
    return Quaternion(
        w=float(math.cos(half)),
        x=float(math.sin(half) * axis[0]),
        y=float(math.sin(half) * axis[1]),
        z=float(math.sin(half) * axis[2]),
    )


def quaternion_multiply(q1: Quaternion, q2: Quaternion) -> Quaternion:
    """Hamilton product ``q1 * q2``."""
    w1, x1, y1, z1 = q1.w, q1.x, q1.y, q1.z
    w2, x2, y2, z2 = q2.w, q2.x, q2.y, q2.z
    return Quaternion(
        w=float(w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2),
        x=float(w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2),
        y=float(w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2),
        z=float(w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2),
    )


def quaternion_conjugate(q: Quaternion) -> Quaternion:
    """Conjugate of quaternion ``q``."""
    return Quaternion(w=q.w, x=-q.x, y=-q.y, z=-q.z)


def slerp(q1: Quaternion, q2: Quaternion, t: float) -> Quaternion:
    """Spherical linear interpolation between quaternions."""
    dot = q1.w * q2.w + q1.x * q2.x + q1.y * q2.y + q1.z * q2.z
    if dot < 0.0:
        q2 = quaternion_conjugate(q2)
        dot = -dot
    if dot > 0.9995:
        angle = 0.0
        s1 = 1.0 - t
        s2 = t
    else:
        theta_0 = math.acos(max(-1.0, min(1.0, dot)))
        theta = theta_0 * t
        sin_theta = math.sin(theta)
        s1 = math.sin(theta_0 - theta) / math.sin(theta_0)
        s2 = sin_theta / math.sin(theta_0)
    return Quaternion(
        w=s1 * q1.w + s2 * q2.w,
        x=s1 * q1.x + s2 * q2.x,
        y=s1 * q1.y + s2 * q2.y,
        z=s1 * q1.z + s2 * q2.z,
    )


def rotate_vector(q: Quaternion, vector: np.ndarray) -> np.ndarray:
    """Rotate a 3D vector by quaternion ``q``."""
    v = np.asarray(vector, dtype=float).reshape(-1)
    x, y, z = q.x, q.y, q.z
    w = q.w
    return v + 2.0 * np.cross(
        np.array([x, y, z]), np.cross(np.array([x, y, z]), v) + w * v
    )


def smooth_rotation(q1: Quaternion, q2: Quaternion, smoothing: float = 0.5) -> Quaternion:
    """Smoothly interpolate between two quaternions (slerp blend)."""
    return slerp(q1, q2, smoothing)


def quaternion_normalized(q: Quaternion) -> Quaternion:
    """Return a unit-norm copy of ``q`` (identity when degenerate)."""
    norm = math.sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z)
    if norm < 1e-12:
        return Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
    return Quaternion(w=q.w / norm, x=q.x / norm, y=q.y / norm, z=q.z / norm)


def quaternion_to_rotation_matrix(q: Quaternion) -> np.ndarray:
    """Convert a unit quaternion to a 3x3 rotation matrix."""
    quat = quaternion_normalized(q)
    w, x, y, z = quat.w, quat.x, quat.y, quat.z
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def rotation_matrix_to_quaternion(matrix: np.ndarray) -> Quaternion:
    """Convert a 3x3 rotation matrix to a unit quaternion (Shepperd)."""
    m = np.asarray(matrix, dtype=float).reshape(3, 3)
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return quaternion_normalized(Quaternion(w=w, x=x, y=y, z=z))


def relative_rotation(q_self: Quaternion, q_other: Quaternion) -> Quaternion:
    """World-space rotation that takes orientation ``q_self`` to ``q_other``."""
    return quaternion_normalized(
        quaternion_multiply(q_other, quaternion_conjugate(q_self))
    )
