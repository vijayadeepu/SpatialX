"""Milestone 1 — hand landmark detection tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers import category as category_module
from mediapipe.tasks.python.components.containers import landmark as landmark_module

from spatialx.core.exceptions import VisionError
from spatialx.vision import (
    DEFAULT_MODEL_PATH,
    LANDMARK_COUNT,
    MODEL_PATH_ENV_VAR,
    HandLandmarker,
    Landmark,
    TrackedHand,
    draw_hand_skeleton,
    extract_hands,
    resolve_model_path,
)


MODEL_PATH = str(
    Path(__file__).resolve().parents[1] / "assets" / "models" / "hand_landmarker.task"
)


def _make_result(hands: list[tuple[str, int]]) -> vision.HandLandmarkerResult:
    """Build a synthetic MediaPipe hand detection result."""
    handedness: list[list] = []
    hand_landmarks: list[list] = []
    hand_world_landmarks: list[list] = []
    for side, count in hands:
        categories = []
        if side != "NONE":
            categories.append(
                category_module.Category(
                    index=1 if side == "LEFT" else 0,
                    score=0.9,
                    display_name=side,
                    category_name=side,
                )
            )
        handedness.append(categories)
        landmarks = [
            landmark_module.NormalizedLandmark(x=0.5, y=0.5, z=0.0, visibility=1.0)
            for _ in range(count)
        ]
        hand_landmarks.append(landmarks)
        hand_world_landmarks.append(
            [landmark_module.Landmark(x=0.0, y=0.0, z=0.0) for _ in range(count)]
        )
    return vision.HandLandmarkerResult(
        handedness=handedness,
        hand_landmarks=hand_landmarks,
        hand_world_landmarks=hand_world_landmarks,
    )


def _make_tracked_hand(side: str) -> TrackedHand:
    return TrackedHand(
        handedness=side,
        landmarks=tuple(
            Landmark(index=i, x=0.5, y=0.5, z=0.0, visibility=1.0) for i in range(LANDMARK_COUNT)
        ),
    )


def test_extract_hands_returns_empty_tuple_when_no_hand_detected() -> None:
    result = _make_result([])
    assert extract_hands(result) == ()


def test_extract_hands_single_left_hand_landmarks() -> None:
    detection_hands = extract_hands(_make_result([("LEFT", LANDMARK_COUNT)]))
    assert len(detection_hands) == 1
    hand = detection_hands[0]
    assert hand.handedness == "LEFT"
    assert len(hand.landmarks) == LANDMARK_COUNT
    assert [landmark.index for landmark in hand.landmarks] == list(range(LANDMARK_COUNT))
    for landmark in hand.landmarks:
        assert landmark.x is not None and 0.0 <= landmark.x <= 1.0
        assert landmark.y is not None and 0.0 <= landmark.y <= 1.0
        assert landmark.z is not None
        assert landmark.visibility is not None


def test_extract_hands_right_hand_side() -> None:
    detection_hands = extract_hands(_make_result([("RIGHT", LANDMARK_COUNT)]))
    assert len(detection_hands) == 1
    assert detection_hands[0].handedness == "RIGHT"


def test_extract_hands_two_hands() -> None:
    detection_hands = extract_hands(
        _make_result([("LEFT", LANDMARK_COUNT), ("RIGHT", LANDMARK_COUNT)])
    )
    assert len(detection_hands) == 2
    assert [hand.handedness for hand in detection_hands] == ["LEFT", "RIGHT"]


def test_extract_hands_skips_incomplete_hand_from_tracking_loss() -> None:
    detection_hands = extract_hands(_make_result([("LEFT", 5)]))
    assert detection_hands == ()


def test_draw_hand_skeleton_returns_annotated_frame() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    annotated = draw_hand_skeleton(frame, [_make_tracked_hand("LEFT")])
    assert annotated.shape == frame.shape
    assert annotated.dtype == np.uint8


def test_hand_landmarker_detect_returns_empty_hands_for_blank_frame() -> None:
    landmarker = HandLandmarker(model_path=MODEL_PATH, num_hands=2)
    try:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detection = landmarker.detect(frame, timestamp_ms=0)
        assert detection is not None
        assert detection.hands == ()
        assert detection.inference_time_ms is not None
        assert detection.inference_time_ms >= 0.0
    finally:
        landmarker.close()


# --- Model asset path resolution (Priority 1) -------------------------------


def test_default_model_path_points_at_existing_asset() -> None:
    path = Path(DEFAULT_MODEL_PATH)
    assert path.is_file(), f"DEFAULT_MODEL_PATH does not exist: {path}"
    assert path.stat().st_size > 1_000_000  # real .task asset, not a stub


def test_resolve_model_path_auto_finds_repo_asset() -> None:
    resolved = resolve_model_path()
    resolved_path = Path(resolved)
    assert resolved_path.is_file()
    assert resolved_path.name == "hand_landmarker.task"
    assert resolved_path.stat().st_size > 1_000_000
    # Auto-resolution must find the same asset the test computes independently.
    assert resolved_path.resolve() == Path(MODEL_PATH).resolve()


def test_resolve_model_path_explicit_missing_raises_vision_error() -> None:
    with pytest.raises(VisionError, match="not found"):
        resolve_model_path("/nonexistent/models/hand_landmarker.task")


def test_resolve_model_path_env_override(monkeypatch, tmp_path: Path) -> None:
    fake_model = tmp_path / "hand_landmarker.task"
    fake_model.write_bytes(b"x" * 1024)
    monkeypatch.setenv(MODEL_PATH_ENV_VAR, str(fake_model))
    assert resolve_model_path() == str(fake_model)


def test_resolve_model_path_env_invalid_raises_vision_error(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(MODEL_PATH_ENV_VAR, str(tmp_path / "missing.task"))
    with pytest.raises(VisionError, match=MODEL_PATH_ENV_VAR):
        resolve_model_path()


def test_hand_landmarker_default_initializes_with_resolved_model() -> None:
    landmarker = HandLandmarker()  # no model_path: auto-resolution
    try:
        assert landmarker.is_open()
        assert Path(landmarker.model_path).is_file()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detection = landmarker.detect(frame, timestamp_ms=0)
        assert detection is not None
        assert detection.hands == ()
    finally:
        landmarker.close()


def test_detect_converts_bgr_frame_to_rgb(monkeypatch) -> None:
    """Regression: MediaPipe must receive RGB data, not raw OpenCV BGR."""
    import cv2 as _cv2

    calls: list[int] = []
    original_cvtColor = _cv2.cvtColor

    def _spy(src, code, *args, **kwargs):
        calls.append(code)
        return original_cvtColor(src, code, *args, **kwargs)

    monkeypatch.setattr("spatialx.vision.hand_landmarker.cv2.cvtColor", _spy)
    landmarker = HandLandmarker()
    try:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        landmarker.detect(frame, timestamp_ms=0)
        assert _cv2.COLOR_BGR2RGB in calls
    finally:
        landmarker.close()
