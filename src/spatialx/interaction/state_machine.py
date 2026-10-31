"""Interaction state machine for SpatialX."""

from __future__ import annotations

from enum import Enum, auto

from spatialx.gestures import FIST, NONE, OPEN_PALM, PINCH, Gesture


class InteractionState(Enum):
    """Explicit interaction states for SpatialX."""

    IDLE = auto()
    TRACKING = auto()
    HOVERING = auto()
    SELECTED = auto()
    GRABBING = auto()
    TRANSLATING = auto()
    ROTATING = auto()
    SCALING = auto()
    RELEASING = auto()
    LOST_TRACKING = auto()
    PAUSED = auto()


class InteractionStateMachine:
    """Deterministic gesture-driven interaction state machine.

    Transition rules implemented here:
        - FIST -> PAUSED (pause interaction).
        - OPEN_PALM in GRABBING -> RELEASING; in RELEASING -> IDLE (release).
        - OPEN_PALM in IDLE/TRACKING/HOVERING/SELECTED -> SELECTED (or IDLE
          when no object is available).
        - PINCH in HOVERING/SELECTED -> GRABBING (accidental-grab prevention:
          PINCH from IDLE or from non-hover states does not grab).
        - Loss of tracking -> LOST_TRACKING.
    """

    def __init__(self) -> None:
        self._state = InteractionState.IDLE
        self._object_available = False

    @property
    def state(self) -> InteractionState:
        return self._state

    @property
    def object_available(self) -> bool:
        return self._object_available

    def handle_gesture(
        self,
        gesture: Gesture,
        has_tracking: bool = True,
        object_available: bool = False,
    ) -> InteractionState:
        """Process a gesture and return the resulting interaction state."""
        self._object_available = object_available
        if not has_tracking:
            self._state = InteractionState.LOST_TRACKING
            return self._state

        name = gesture.name
        if name == FIST:
            self._state = InteractionState.PAUSED
            return self._state

        if self._state == InteractionState.PAUSED:
            if name == OPEN_PALM:
                self._state = (
                    InteractionState.SELECTED if object_available else InteractionState.IDLE
                )
            elif name == PINCH and object_available:
                self._state = InteractionState.GRABBING
            return self._state

        if name == OPEN_PALM:
            if self._state == InteractionState.GRABBING:
                self._state = InteractionState.RELEASING
                return self._state
            if self._state == InteractionState.RELEASING:
                self._state = InteractionState.IDLE
                return self._state
            self._state = (
                InteractionState.SELECTED if object_available else InteractionState.IDLE
            )
            return self._state

        if name == PINCH:
            if self._state in (InteractionState.HOVERING, InteractionState.SELECTED):
                self._state = InteractionState.GRABBING
            return self._state

        return self._state

    def release(self, gesture: Gesture) -> InteractionState:
        """Release the current object via the release gesture sequence.

        Release sequence: GRABBING --(open palm)--> RELEASING --(open palm)--> IDLE.
        """
        if self._state == InteractionState.GRABBING:
            self.handle_gesture(gesture, has_tracking=True, object_available=self._object_available)
            self.handle_gesture(gesture, has_tracking=True, object_available=self._object_available)
        return self.state
