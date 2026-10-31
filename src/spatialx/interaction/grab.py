"""Object manipulation: grab control and hand-to-object offsets."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from spatialx.gestures import Gesture, PINCH, OPEN_PALM
from spatialx.interaction.state_machine import InteractionState, InteractionStateMachine


class GrabbableObject:
    """A virtual object that can be manipulated by the hand."""

    def __init__(self, object_id: str, position: Optional[np.ndarray] = None) -> None:
        self.object_id = object_id
        self.position = np.asarray(position, dtype=float).reshape(3) if position is not None else np.zeros(3)


class GrabController:
    """Owns grabbed objects, hand-to-object offsets and translation."""

    def __init__(self) -> None:
        self._state_machine = InteractionStateMachine()
        self._objects: Dict[str, GrabbableObject] = {}
        self._grabbed: Optional[GrabbableObject] = None
        self._offset: Optional[np.ndarray] = None

    @property
    def state_machine(self) -> InteractionStateMachine:
        return self._state_machine

    @property
    def grabbed(self) -> Optional[GrabbableObject]:
        return self._grabbed

    def register_object(self, obj: GrabbableObject) -> None:
        """Register a grabbable object."""
        self._objects[obj.object_id] = obj

    def grab(self, object_id: str, hand_origin_camera: np.ndarray) -> None:
        """Grab an object, storing the hand-to-object offset.

        The offset prevents the object from jumping when grab begins:
            object.position = hand.position + offset
        """
        obj = self._objects[object_id]
        self._offset = np.asarray(obj.position, dtype=float) - np.asarray(hand_origin_camera, dtype=float)
        self._grabbed = obj
        self.state_machine.handle_gesture(
            Gesture(name=PINCH, confidence=1.0),
            has_tracking=True,
            object_available=True,
        )

    def release(self) -> None:
        """Release the grabbed object back to the idle state."""
        self._grabbed = None
        self._offset = None
        self.state_machine.release(Gesture(name=OPEN_PALM, confidence=1.0))

    def translate(self, hand_origin_camera: np.ndarray) -> None:
        """Move the grabbed object following the hand, preserving the offset."""
        if self._grabbed is None or self._offset is None:
            return
        self._grabbed.position = np.asarray(hand_origin_camera, dtype=float) + self._offset
