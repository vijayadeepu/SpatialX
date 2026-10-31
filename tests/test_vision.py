"""Milestone 1 — hand landmark detection tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers import category as category_module
from mediapipe.tasks.python.components.containers import landmark as landmark_module

from spatialx.vision import (
    LANDMARK_COUNT,
    HandLandmarker,
    Landmark,
    TrackedHand,
    draw_hand_skeleton,
    extract_hands,
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
