"""Deterministic geometric gesture recognition for SpatialX."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only
    # Importing vision here would pull MediaPipe into every consumer of
    # gestures (including the Panda3D rendering process). The hand objects
    # are used duck-typed / as type hints only.
    from spatialx.vision import TrackedHand

PINCH = "PINCH"
OPEN_PALM = "OPEN_PALM"
FIST = "FIST"
NONE = "NONE"

THUMB_TIP = 4
THUMB_BASE = 2
INDEX_TIP = 8
INDEX_BASE = 5
MIDDLE_TIP = 12
MIDDLE_BASE = 9
RING_TIP = 16
RING_BASE = 13
PINKY_TIP = 20
PINKY_BASE = 17

FINGER_PAIRS = (
    (INDEX_TIP, INDEX_BASE),
    (MIDDLE_TIP, MIDDLE_BASE),
    (RING_TIP, RING_BASE),
    (PINKY_TIP, PINKY_BASE),
)


@dataclass(frozen=True)
class Gesture:
    """A deterministic gesture classification result."""

    name: str  # PINCH | OPEN_PALM | FIST | NONE
    confidence: float


def _coord(landmark) -> tuple:
    return (
        landmark.x if landmark.x is not None else 0.0,
        landmark.y if landmark.y is not None else 0.0,
        landmark.z if landmark.z is not None else 0.0,
    )


def _distance(a, b) -> float:
    if a is None or b is None:
        return 0.0
    ax, ay, az = _coord(a)
    bx, by, bz = _coord(b)
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)


def _landmark(hand: TrackedHand, index: int):
    return hand.landmarks[index] if index < len(hand.landmarks) else None


def _hand_diameter(hand: TrackedHand) -> float:
    """Estimate hand size as the maximum pairwise landmark distance."""
    landmarks = list(hand.landmarks)
    diameter = 0.0
    for i in range(len(landmarks)):
        for j in range(i + 1, len(landmarks)):
            diameter = max(diameter, _distance(landmarks[i], landmarks[j]))
    return diameter


def pinch_distance(hand: TrackedHand) -> float:
    """Distance between thumb tip and index tip (pinch measure)."""
    return _distance(_landmark(hand, THUMB_TIP), _landmark(hand, INDEX_TIP))


def is_pinch(hand: TrackedHand, threshold: float = 0.25) -> bool:
    """Return ``True`` when thumb-index distance falls below pinch threshold."""
    return pinch_distance(hand) <= threshold


def finger_extended(hand: TrackedHand, tip_index: int, base_index: int) -> bool:
    """Return ``True`` when the finger at ``tip_index`` is extended."""
    tip = _landmark(hand, tip_index)
    base = _landmark(hand, base_index)
    wrist = _landmark(hand, 0)
    if tip is None or base is None or wrist is None:
        return False
    return _distance(wrist, tip) > _distance(wrist, base)


def is_open_palm(hand: TrackedHand) -> bool:
    """Return ``True`` when all four fingers are extended."""
    extended = sum(1 for tip, base in FINGER_PAIRS if finger_extended(hand, tip, base))
    return extended == len(FINGER_PAIRS)


def is_fist(hand: TrackedHand) -> bool:
    """Return ``True`` when all fingers and the thumb are folded."""
    fingers_folded = all(
        not finger_extended(hand, tip, base) for tip, base in FINGER_PAIRS
    )
    thumb_folded = not finger_extended(hand, THUMB_TIP, THUMB_BASE)
    return fingers_folded and thumb_folded


def detect_gesture(
    hands: Sequence[TrackedHand],
    pinch_threshold: float = 0.25,
) -> Gesture:
    """Deterministically classify the current gesture from tracked hands.

    Precedence: PINCH > OPEN_PALM > FIST > NONE.

    Args:
        hands: Tracked hands from the vision layer.
        pinch_threshold: Thumb-index distance threshold for pinch.

    Returns:
        A :class:`Gesture` with a deterministic name and confidence.
    """
    if not hands:
        return Gesture(name=NONE, confidence=0.0)
    hand = hands[0]
    index_extended = finger_extended(hand, INDEX_TIP, INDEX_BASE)
    if is_pinch(hand, threshold=pinch_threshold) and index_extended:
        diameter = max(_hand_diameter(hand), 1e-6)
        confidence = min(1.0, max(0.0, 1.0 - pinch_distance(hand) / diameter))
        return Gesture(name=PINCH, confidence=confidence)
    if is_open_palm(hand):
        extended = sum(1 for tip, base in FINGER_PAIRS if finger_extended(hand, tip, base))
        return Gesture(name=OPEN_PALM, confidence=extended / len(FINGER_PAIRS))
    if is_fist(hand):
        return Gesture(name=FIST, confidence=1.0)
    return Gesture(name=NONE, confidence=0.0)
