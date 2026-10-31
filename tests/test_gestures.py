"""Milestone 4 — deterministic gesture detection tests."""

from __future__ import annotations

from spatialx.gestures import (
    FIST,
    NONE,
    OPEN_PALM,
    PINCH,
    Gesture,
    detect_gesture,
    is_fist,
    is_open_palm,
    is_pinch,
    pinch_distance,
)
from spatialx.vision import LANDMARK_COUNT, Landmark, TrackedHand


def _hand(landmarks: list[tuple[float, float, float]]) -> TrackedHand:
    return TrackedHand(
        handedness="LEFT",
        landmarks=tuple(
            Landmark(index=i, x=x, y=y, z=z, visibility=1.0)
            for i, (x, y, z) in enumerate(landmarks)
        ),
    )


def _open_palm_hand() -> TrackedHand:
    landmarks = [(0.0, 0.0, 0.0)] * LANDMARK_COUNT
    landmarks[2] = (0.2, 0.0, 0.0)   # thumb MCP
    landmarks[3] = (0.35, 0.0, 0.0)  # thumb IP
    landmarks[4] = (0.45, 0.3, 0.0)  # thumb tip
    for finger_base, finger_tip in ((5, 8), (9, 12), (13, 16), (17, 20)):
        landmarks[finger_base] = (0.2, 0.0, 0.0)
        landmarks[finger_tip] = (0.4, 0.0, 0.0)
    return _hand(landmarks)


def _fist_hand() -> TrackedHand:
    landmarks = [(0.05, 0.0, 0.0) for _ in range(LANDMARK_COUNT)]
    landmarks[0] = (0.0, 0.0, 0.0)
    return _hand(landmarks)


def test_is_pinch_detects_thumb_index_touch() -> None:
    hand = _hand([(0.0, 0.0, 0.0)] + [(0.2, 0.0, 0.0)] * 4 + [(0.45, 0.0, 0.0)] + [(0.4, 0.0, 0.0)] + [(0.2, 0.0, 0.0)] * 12)
    assert is_pinch(hand, threshold=0.25) is True
    assert pinch_distance(hand) <= 0.25


def test_is_pinch_false_when_open() -> None:
    hand = _open_palm_hand()
    assert is_pinch(hand, threshold=0.25) is False
    assert is_open_palm(hand) is True


def test_is_open_palm() -> None:
    assert is_open_palm(_open_palm_hand()) is True


def test_is_fist() -> None:
    assert is_fist(_fist_hand()) is True


def test_detect_gesture_open_palm() -> None:
    gesture = detect_gesture([_open_palm_hand()])
    assert gesture.name == OPEN_PALM
    assert gesture.confidence > 0.9


def test_detect_gesture_fist() -> None:
    gesture = detect_gesture([_fist_hand()])
    assert gesture.name == FIST


def test_detect_gesture_pinch_with_extended_index() -> None:
    values = [(0.0, 0.0, 0.0)] * LANDMARK_COUNT
    values[0] = (0.0, 0.0, 0.0)    # wrist
    values[4] = (0.45, 0.0, 0.0)   # thumb tip
    values[5] = (0.2, 0.0, 0.0)    # index base (proximal)
    values[8] = (0.42, 0.0, 0.0)   # index tip (pinched against thumb tip)
    hand = _hand(values)
    gesture = detect_gesture([hand])
    assert gesture.name == PINCH
    assert gesture.confidence > 0.5


def test_detect_gesture_none_when_neutral() -> None:
    values = [(0.1, 0.0, 0.0)] * LANDMARK_COUNT
    values[0] = (0.0, 0.0, 0.0)  # wrist
    values[4] = (0.45, 0.0, 0.0)  # thumb tip (extended), fingers folded
    hand = _hand(values)
    gesture = detect_gesture([hand])
    assert gesture.name == NONE


def test_detect_gesture_empty_hands() -> None:
    gesture = detect_gesture([])
    assert gesture.name == NONE
    assert gesture.confidence == 0.0
