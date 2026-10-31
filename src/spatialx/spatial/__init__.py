"""Spatial coordinate mathematics for SpatialX."""

from spatialx.spatial.coordinates import (
    CameraCalibration,
    hand_relative_vectors,
    hand_velocity,
    landmark_camera_space,
    palm_orientation,
    relative_depth,
)
from spatialx.spatial.quaternion import (
    Quaternion,
    rotate_vector,
    rotation_vector_to_quaternion,
    slerp,
    smooth_rotation,
    quaternion_conjugate,
    quaternion_multiply,
)

__all__ = [
    "CameraCalibration",
    "Quaternion",
    "hand_relative_vectors",
    "hand_velocity",
    "landmark_camera_space",
    "palm_orientation",
    "quaternion_conjugate",
    "quaternion_multiply",
    "relative_depth",
    "rotate_vector",
    "rotation_vector_to_quaternion",
    "smooth_rotation",
    "slerp",
]
