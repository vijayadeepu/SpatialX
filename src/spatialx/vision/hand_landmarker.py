"""Hand landmarker integration powered by MediaPipe Tasks."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from mediapipe.tasks.python import BaseOptions, vision
from mediapipe.tasks.python.vision import hand_landmarker as _hand_landmarker_module
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
from mediapipe.tasks.python.components.containers import category as _category_module

from spatialx.core.exceptions import VisionError

LANDMARK_COUNT = 21

MODEL_ASSET_NAME = "hand_landmarker.task"
MODEL_PATH_ENV_VAR = "SPATIALX_MODEL_PATH"
MODEL_DOWNLOAD_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

# Legacy default: the asset location in a source (editable) checkout.
# ``<repo>/src/spatialx/vision/hand_landmarker.py`` -> ``<repo>/assets/models/``.
# Prefer :func:`resolve_model_path`, which verifies existence.
DEFAULT_MODEL_PATH = str(
    Path(__file__).resolve().parents[3] / "assets" / "models" / MODEL_ASSET_NAME
)


def _model_candidates() -> list[Path]:
    """Ordered candidate locations for the hand landmarker model asset."""
    module_path = Path(__file__).resolve()
    candidates: list[Path] = []
    # 1. Environment override (checked first by resolve_model_path, strict).
    env_value = os.environ.get(MODEL_PATH_ENV_VAR)
    if env_value:
        candidates.append(Path(env_value).expanduser())
    # 2. Package-adjacent assets (packaged wheel layout: <package>/assets/models).
    candidates.append(
        module_path.parents[1] / "assets" / "models" / MODEL_ASSET_NAME
    )
    # 3. Walk up from this module to find a repo-root assets/models directory
    #    (covers the editable source checkout at any nesting depth).
    for parent in module_path.parents:
        candidates.append(parent / "assets" / "models" / MODEL_ASSET_NAME)
    return candidates


def resolve_model_path(model_path: Optional[str] = None) -> str:
    """Resolve the hand landmarker model asset to an existing file path.

    Resolution order:

    1. Explicit ``model_path`` argument (must exist).
    2. ``SPATIALX_MODEL_PATH`` environment variable (must exist when set).
    3. ``<package>/assets/models/hand_landmarker.task``.
    4. ``assets/models/hand_landmarker.task`` walking up from this module
       (the editable source checkout: ``<repo>/assets/models/``).

    Args:
        model_path: Explicit model file path, or ``None`` for auto-resolution.

    Returns:
        Absolute path string of an existing, non-empty model file.

    Raises:
        VisionError: If the resolved path does not exist or is empty. The
            message lists every searched location and the download URL.
    """
    if model_path is not None:
        path = Path(model_path).expanduser()
        if not path.is_file() or path.stat().st_size == 0:
            raise VisionError(
                f"Hand landmarker model not found or empty at explicit path: {path}"
            )
        return str(path)

    env_value = os.environ.get(MODEL_PATH_ENV_VAR)
    if env_value:
        env_path = Path(env_value).expanduser()
        if not env_path.is_file() or env_path.stat().st_size == 0:
            raise VisionError(
                f"{MODEL_PATH_ENV_VAR} is set but the model file was not found "
                f"or is empty: {env_path}"
            )
        return str(env_path)

    searched: list[str] = []
    for candidate in _model_candidates():
        key = str(candidate)
        if key in searched:
            continue
        searched.append(key)
        if candidate.is_file() and candidate.stat().st_size > 0:
            return str(candidate)

    raise VisionError(
        "Hand landmarker model asset could not be found. Searched:\n"
        + "\n".join(f"  - {path}" for path in searched)
        + f"\nSet {MODEL_PATH_ENV_VAR} to the model file, or download it from:\n"
        f"  {MODEL_DOWNLOAD_URL}"
    )


HandLandmarkerTask = vision.HandLandmarker
HAND_CONNECTIONS = _hand_landmarker_module.HandLandmarksConnections.HAND_CONNECTIONS


@dataclass(frozen=True)
class Landmark:
    """A single hand landmark in normalized image coordinates.

    - ``x`` / ``y``: normalized image coordinates (0..1), origin top-left;
      ``x`` is relative to image width, ``y`` to image height.
    - ``z``: depth offset relative to the wrist, **same scale as ``x``**
      (width-normalized); smaller values are closer to the camera. This is
      relative depth only, not metric distance.
    - ``visibility``: landmark visibility score (0..1), ``None`` when unknown.
    """

    index: int
    x: Optional[float]
    y: Optional[float]
    z: Optional[float]
    visibility: Optional[float]


@dataclass(frozen=True)
class TrackedHand:
    """A tracked hand with its 21 landmarks and identified side."""

    handedness: str  # "LEFT" | "RIGHT" | "UNKNOWN"
    landmarks: tuple[Landmark, ...]


@dataclass(frozen=True)
class HandDetection:
    """Result of a single hand-landmark inference step."""

    hands: tuple[TrackedHand, ...]
    timestamp_ms: Optional[int]
    inference_time_ms: Optional[float]


def _best_category_side(categories: Sequence) -> str:
    """Return the best-effort hand side from candidate categories."""
    if not categories:
        return "UNKNOWN"
    best = categories[0]
    side = best.category_name or best.display_name or "UNKNOWN"
    return str(side).upper()


def extract_hands(result) -> tuple[TrackedHand, ...]:
    """Convert a MediaPipe hand detection result into SpatialX hand objects.

    Hands with fewer than ``LANDMARK_COUNT`` landmarks are skipped, which
    models temporary tracking loss without crashing the pipeline.

    Args:
        result: A :class:`mediapipe.tasks.python.vision.HandLandmarkerResult`.

    Returns:
        A tuple of :class:`TrackedHand` objects (empty when no hand is detected).
    """
    hands: list[TrackedHand] = []
    for hand_index, landmarks in enumerate(result.hand_landmarks):
        if not landmarks or len(landmarks) < LANDMARK_COUNT:
            continue
        side = (
            _best_category_side(result.handedness[hand_index])
            if hand_index < len(result.handedness)
            else "UNKNOWN"
        )
        hand_landmarks = tuple(
            Landmark(
                index=index,
                x=landmark.x,
                y=landmark.y,
                z=landmark.z,
                visibility=landmark.visibility,
            )
            for index, landmark in enumerate(landmarks[:LANDMARK_COUNT])
        )
        hands.append(TrackedHand(handedness=side, landmarks=hand_landmarks))
    return tuple(hands)


class HandLandmarker:
    """MediaPipe Tasks hand landmarker wrapper with SpatialX-friendly API.

    Runs in :attr:`vision.RunningMode.VIDEO` mode so temporal tracking is
    applied by the model, and exposes inference timing for latency metrics.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        num_hands: int = 2,
        running_mode=None,
        detection_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
    ) -> None:
        self.model_path = resolve_model_path(model_path)
        self.num_hands = num_hands
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=running_mode if running_mode is not None else vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._task: Optional[HandLandmarkerTask] = None
        self._task = vision.HandLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    def detect(
        self,
        frame: np.ndarray,
        timestamp_ms: Optional[int] = None,
    ) -> Optional[HandDetection]:
        """Run hand detection / tracking on a BGR camera frame.

        The frame is converted from OpenCV's BGR channel order to RGB
        internally, since MediaPipe expects RGB input; skipping this
        conversion silently degrades detection.

        Args:
            frame: OpenCV BGR ``numpy.ndarray`` frame.
            timestamp_ms: Optional frame timestamp in milliseconds; must be
                monotonically increasing when omitted. The wrapper maintains
                an internal monotonic clock when ``None`` is given.

        Returns:
            A :class:`HandDetection`, or ``None`` when inference fails.

        Raises:
            VisionError: If inference fails.
        """
        if self._task is None:
            raise VisionError("HandLandmarker task is not initialized.")
        current_ms = self._timestamp_ms if timestamp_ms is None else int(timestamp_ms)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
        start = time.monotonic()
        try:
            result = self._task.detect_for_video(image=image, timestamp_ms=current_ms)
            inference_time_ms = (time.monotonic() - start) * 1000.0
        except Exception as error:  # pragma: no cover - defensive boundary
            raise VisionError(f"Hand landmarker inference failed: {error}") from error
        self._timestamp_ms = current_ms + 1
        return HandDetection(
            hands=extract_hands(result),
            timestamp_ms=current_ms,
            inference_time_ms=inference_time_ms,
        )

    def close(self) -> None:
        """Release the underlying MediaPipe task."""
        if self._task is not None:
            self._task.close()
            self._task = None

    def is_open(self) -> bool:
        """Return ``True`` if the underlying task is alive."""
        return self._task is not None

    def __enter__(self) -> "HandLandmarker":
        return self

    def __exit__(self, *args) -> None:
        self.close()
