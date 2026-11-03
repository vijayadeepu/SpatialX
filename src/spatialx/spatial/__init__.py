"""Spatial coordinate mathematics for SpatialX.

Exports are resolved lazily (PEP 562) so that importing lightweight
submodules such as ``spatialx.spatial.quaternion`` does not pull the
vision stack (MediaPipe). This keeps the rendering process free of
MediaPipe, which is required because MediaPipe's bundled LLVM and
Panda3D's runtime corrupt each other's global state when both render and
track inside a single process (see PROJECT_STATUS.md, Priority 3).
"""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "CameraCalibration": "coordinates",
    "camera_ray": "coordinates",
    "hand_relative_vectors": "coordinates",
    "hand_velocity": "coordinates",
    "landmark_camera_space": "coordinates",
    "normalized_to_pixel": "coordinates",
    "palm_frame_quaternion": "coordinates",
    "palm_orientation": "coordinates",
    "relative_depth": "coordinates",
    "Quaternion": "quaternion",
    "quaternion_conjugate": "quaternion",
    "quaternion_multiply": "quaternion",
    "quaternion_normalized": "quaternion",
    "quaternion_to_rotation_matrix": "quaternion",
    "relative_rotation": "quaternion",
    "rotate_vector": "quaternion",
    "rotation_matrix_to_quaternion": "quaternion",
    "rotation_vector_to_quaternion": "quaternion",
    "smooth_rotation": "quaternion",
    "slerp": "quaternion",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve a spatial export lazily on first access."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value  # cache for subsequent lookups
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
