"""Spatial coordinate mathematics for SpatialX.

This module converts MediaPipe landmark observations into stable,
camera-referenced and hand-relative spatial coordinates.

Coordinate conventions (audited)
--------------------------------
- Image space: ``u`` right, ``v`` down, origin top-left, units of pixels.
- Normalized landmarks (MediaPipe output): ``x``/``y`` in ``[0, 1]``
  relative to the image **width** and **height** respectively; ``z`` is a
  depth offset relative to the wrist with the **same scale as ``x``**
  (width-normalized); smaller ``z`` means closer to the camera.
- Camera space (this module): pixel units with the principal point (assumed
  image center) as origin: ``[u - cx, v - cy, z * width]``. All geometry is
  computed in this isotropic pixel space so that mixed width/height
  normalization never distorts distances or orientations.
- **Depth is wrist-relative and NOT metric.** No depth sensor or validated
  intrinsic calibration is available on this hardware, so absolute distances
  (meters) are deliberately never produced here. Z is meaningful for
  relative comparisons (closer/farther, hand-relative offsets) only.
- Rays: :func:`camera_ray` provides true pinhole ray directions for callers
  that need projective geometry with assumed or measured intrinsics.
- Mirroring: webcam ("selfie") pipelines can pass ``mirror=True`` to flip
  the horizontal axis (``u -> width - u``), matching a horizontally
  mirrored preview so cursor motion follows the user's hand.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

from spatialx.spatial.quaternion import Quaternion, rotation_matrix_to_quaternion

if TYPE_CHECKING:  # pragma: no cover - typing only; importing vision here
    # would pull MediaPipe into every consumer (breaking the rendering
    # process — see spatialx.application.tracking_link).
    from spatialx.vision import Landmark, TrackedHand

IDENTITY_ROTATION = np.eye(3, dtype=float)
ZERO_TRANSLATION = np.zeros(3, dtype=float)


@dataclass(frozen=True)
class CameraCalibration:
    """Intrinsic (and optional extrinsic) camera parameters in pixel units.

    The defaults assume a 640x480 sensor, the principal point at the image
    center, and a nominal 500-pixel focal length. These are engineering
    defaults, **not a measured calibration**. Use :meth:`for_image` for
    nominal intrinsics at other resolutions, or supply measured values when
    a real calibration procedure has been performed.

    ``rotation`` / ``translation`` may be ``None`` (identity / zero); use
    :attr:`effective_rotation` / :attr:`effective_translation` to read them.
    """

    focal_length_x: float = 500.0
    focal_length_y: float = 500.0
    principal_point_x: float = 320.0
    principal_point_y: float = 240.0
    rotation: Optional[np.ndarray] = None  # 3x3, None = identity
    translation: Optional[np.ndarray] = None  # (3,), None = zero vector

    @classmethod
    def for_image(
        cls,
        image_width: float,
        image_height: float,
        horizontal_fov_degrees: float = 60.0,
    ) -> "CameraCalibration":
        """Build nominal intrinsics for an image size.

        The focal length is derived from an **assumed** horizontal field of
        view (default 60 degrees) and the principal point from the image
        center. This is a documented assumption for downstream scaling, not
        a measured camera calibration.
        """
        half_fov = math.radians(horizontal_fov_degrees) / 2.0
        focal = (float(image_width) / 2.0) / math.tan(half_fov)
        return cls(
            focal_length_x=focal,
            focal_length_y=focal,
            principal_point_x=float(image_width) / 2.0,
            principal_point_y=float(image_height) / 2.0,
        )

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

    @property
    def effective_rotation(self) -> np.ndarray:
        """Rotation matrix (identity when not set)."""
        return IDENTITY_ROTATION if self.rotation is None else self.rotation

    @property
    def effective_translation(self) -> np.ndarray:
        """Translation vector (zero when not set)."""
        return ZERO_TRANSLATION if self.translation is None else self.translation


def normalized_to_pixel(
    landmark: Landmark,
    image_width: float = 640.0,
    image_height: float = 480.0,
    mirror: bool = False,
) -> np.ndarray:
    """Convert a normalized landmark into image-space pixel coordinates.

    Args:
        landmark: Normalized landmark (MediaPipe convention).
        image_width: Frame width in pixels (z is scaled by width).
        image_height: Frame height in pixels.
        mirror: Flip horizontally (``u -> width - u``) for selfie-view
            pipelines.

    Returns:
        ``[u, v, w]`` where ``u``/``v`` are pixels from the top-left corner
        and ``w = z * image_width`` (width-scaled, wrist-relative depth).
    """
    x = landmark.x if landmark.x is not None else 0.0
    y = landmark.y if landmark.y is not None else 0.0
    z = landmark.z if landmark.z is not None else 0.0
    u = (1.0 - x) * image_width if mirror else x * image_width
    v = y * image_height
    w = z * image_width
    return np.array([u, v, w], dtype=float)


def landmark_camera_space(
    landmark: Landmark,
    calibration: Optional[CameraCalibration] = None,
    image_width: float = 640.0,
    image_height: float = 480.0,
    mirror: bool = False,
) -> np.ndarray:
    """Convert a normalized landmark into camera-space pixel coordinates.

    Camera space: origin at the principal point (assumed image center when
    no calibration is given), ``X`` right, ``Y`` down, ``Z`` forward.

    **Units and honesty:** X/Y are pixel offsets from the principal point;
    Z is the MediaPipe wrist-relative depth scaled to width-pixels
    (``z * image_width``). Z is *not* metric depth — it is only valid for
    relative comparisons (closer/farther, hand-relative offsets).

    Returns:
        A 3-element numpy array.
    """
    cal = calibration or CameraCalibration.for_image(image_width, image_height)
    u, v, w = normalized_to_pixel(landmark, image_width, image_height, mirror)
    return np.array(
        [u - cal.principal_point_x, v - cal.principal_point_y, w], dtype=float
    )


def camera_ray(
    u: float,
    v: float,
    calibration: Optional[CameraCalibration] = None,
) -> np.ndarray:
    """Unit ray direction through pixel ``(u, v)`` (true pinhole geometry).

    Args:
        u: Pixel column (right positive).
        v: Pixel row (down positive).
        calibration: Intrinsics; nominal 640x480 defaults when ``None``.

    Returns:
        A unit-norm 3-vector in camera space (``+Z`` forward).
    """
    cal = calibration or CameraCalibration()
    direction = np.array(
        [
            (u - cal.principal_point_x) / cal.focal_length_x,
            (v - cal.principal_point_y) / cal.focal_length_y,
            1.0,
        ],
        dtype=float,
    )
    norm = float(np.linalg.norm(direction))
    return direction / norm if norm > 0.0 else direction


def hand_relative_vectors(
    hand: TrackedHand,
    image_width: float = 640.0,
    image_height: float = 480.0,
    mirror: bool = False,
) -> np.ndarray:
    """Vectors from the wrist (landmark 0) to every other landmark.

    Computed in camera pixel space (isotropic x/y, width-scaled z) so
    distances are not distorted by the width/height normalization asymmetry.

    Returns:
        ``(LANDMARK_COUNT - 1, 3)`` array of pixel-space vectors.
    """
    if not hand.landmarks:
        return np.zeros((0, 3), dtype=float)
    wrist = normalized_to_pixel(hand.landmarks[0], image_width, image_height, mirror)
    vectors = []
    for landmark in hand.landmarks[1:]:
        if landmark.x is None or landmark.y is None:
            continue
        point = normalized_to_pixel(landmark, image_width, image_height, mirror)
        vectors.append(point - wrist)
    return np.array(vectors, dtype=float).reshape(len(vectors), 3)


def relative_depth(
    hand: TrackedHand,
    image_width: float = 640.0,
) -> np.ndarray:
    """Wrist-relative depth offsets for every non-wrist landmark.

    Depth is width-scaled (``(z_i - z_wrist) * image_width``, pixel units).
    Positive means farther from the camera than the wrist (MediaPipe sign
    convention: larger z is farther).

    Returns:
        A 1-D array with one entry per non-wrist landmark.
    """
    if len(hand.landmarks) < 2:
        return np.zeros(0, dtype=float)
    wrist_z = hand.landmarks[0].z if hand.landmarks[0].z is not None else 0.0
    return np.array(
        [
            ((landmark.z if landmark.z is not None else 0.0) - wrist_z) * image_width
            for landmark in hand.landmarks[1:]
        ],
        dtype=float,
    )


def palm_orientation(
    hand: TrackedHand,
    image_width: float = 640.0,
    image_height: float = 480.0,
    mirror: bool = False,
) -> np.ndarray:
    """Estimate palm orientation as a rotation vector (unit axis * angle).

    The palm plane basis is built from the wrist→index-MCP and
    wrist→middle-MCP vectors in camera pixel space (isotropic x/y with
    width-scaled z). The rotation vector magnitude equals the angle between
    the two basis vectors (radians), so downstream quaternion conversion via
    :func:`spatialx.spatial.rotation_vector_to_quaternion` is correct.

    Returns:
        A 3-element rotation vector; ``[0, 0, 0]`` for degenerate geometry.
    """
    if len(hand.landmarks) < 10:
        return np.zeros(3, dtype=float)
    wrist = normalized_to_pixel(hand.landmarks[0], image_width, image_height, mirror)
    index_mcp = normalized_to_pixel(hand.landmarks[5], image_width, image_height, mirror)
    middle_mcp = normalized_to_pixel(hand.landmarks[9], image_width, image_height, mirror)
    a = index_mcp - wrist
    b = middle_mcp - wrist
    axis = np.cross(a, b)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-9:
        return np.zeros(3, dtype=float)
    angle = math.atan2(axis_norm, float(np.dot(a, b)))
    return (axis / axis_norm) * angle


def palm_frame_quaternion(
    hand: TrackedHand,
    image_width: float = 640.0,
    image_height: float = 480.0,
    mirror: bool = False,
) -> Optional[Quaternion]:
    """Full 3D palm orientation as a camera-space quaternion.

    Builds a right-handed orthonormal palm frame from the wrist, index MCP
    and middle MCP landmarks in isotropic camera pixel space:

    - ``y``: palm "up" (wrist toward the finger knuckles, mean of index and
      middle MCP directions).
    - ``z``: palm normal (``wrist→index MCP`` cross ``wrist→middle MCP``).
    - ``x``: completes the right-handed basis.

    This is a full orientation (unlike :func:`palm_orientation`, which is a
    single-axis rotation vector). It is still **not** metric: the underlying
    landmarks use wrist-relative, width-scaled Z.

    Returns:
        A unit :class:`Quaternion`, or ``None`` when the palm geometry is
        degenerate (missing landmarks or collinear knuckle vectors). Callers
        should skip rotation updates on ``None`` rather than treat it as
        identity.
    """
    if len(hand.landmarks) < 10:
        return None
    wrist = normalized_to_pixel(hand.landmarks[0], image_width, image_height, mirror)
    index_mcp = normalized_to_pixel(hand.landmarks[5], image_width, image_height, mirror)
    middle_mcp = normalized_to_pixel(hand.landmarks[9], image_width, image_height, mirror)
    a = index_mcp - wrist
    b = middle_mcp - wrist
    if float(np.linalg.norm(a)) < 1e-9 or float(np.linalg.norm(b)) < 1e-9:
        return None
    y_axis = a + b
    y_norm = float(np.linalg.norm(y_axis))
    if y_norm < 1e-9:
        return None
    y_axis = y_axis / y_norm
    normal = np.cross(a, b)
    n_norm = float(np.linalg.norm(normal))
    if n_norm < 1e-9:
        return None
    z_axis = normal / n_norm
    x_axis = np.cross(y_axis, z_axis)
    x_norm = float(np.linalg.norm(x_axis))
    if x_norm < 1e-9:
        return None
    x_axis = x_axis / x_norm
    # Re-orthogonalize z so the matrix is exactly orthonormal.
    z_axis = np.cross(x_axis, y_axis)
    rotation = np.column_stack([x_axis, y_axis, z_axis])
    return rotation_matrix_to_quaternion(rotation)


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
