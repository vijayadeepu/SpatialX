"""Object manipulation: grab control, translation and quaternion rotation.

This module is the runtime bridge between tracking states and scene objects.
It never imports MediaPipe, OpenCV or Panda3D: positions are camera-space
pixel vectors and orientations are camera-space
:class:`~spatialx.spatial.quaternion.Quaternion` values.

Depth honesty: camera-space Z is wrist-relative and width-scaled, so
translation along the camera forward axis is only a relative offset, not a
metric distance. See ``spatialx.spatial.coordinates``.
"""

from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np

from spatialx.gestures import NONE, OPEN_PALM, PINCH, Gesture
from spatialx.interaction.state_machine import InteractionState, InteractionStateMachine
from spatialx.spatial.quaternion import (
    Quaternion,
    quaternion_conjugate,
    quaternion_multiply,
    quaternion_normalized,
    slerp,
)

# Conservative camera-space clamp (pixels) so a tracking spike cannot send
# the object to infinity. Generous relative to a 640x480 frame.
DEFAULT_BOUNDS_MIN = np.array([-480.0, -360.0, -200.0], dtype=float)
DEFAULT_BOUNDS_MAX = np.array([480.0, 360.0, 200.0], dtype=float)


class GrabbableObject:
    """A virtual object that can be manipulated by the hand.

    ``position`` is a 3-element camera-space vector (pixels). Orientation is
    owned by :class:`GrabController` while the object is held.
    """

    def __init__(self, object_id: str, position: Optional[np.ndarray] = None) -> None:
        self.object_id = object_id
        self.position = (
            np.asarray(position, dtype=float).reshape(3)
            if position is not None
            else np.zeros(3, dtype=float)
        )


class GestureStabilizer:
    """Require N consecutive identical gesture names before switching.

    Suppresses single-frame PINCH/OPEN_PALM flicker that would otherwise
    cause accidental grabs or releases.
    """

    def __init__(self, stable_frames: int = 2) -> None:
        if stable_frames < 1:
            raise ValueError("stable_frames must be >= 1")
        self.stable_frames = int(stable_frames)
        self._stable = NONE
        self._candidate: Optional[str] = None
        self._count = 0

    @property
    def stable_name(self) -> str:
        """The currently accepted (debounced) gesture name."""
        return self._stable

    def update(self, gesture_name: str) -> str:
        """Feed one raw gesture name; return the debounced name."""
        if gesture_name == self._stable:
            self._candidate = None
            self._count = 0
            return self._stable
        if gesture_name == self._candidate:
            self._count += 1
        else:
            self._candidate = gesture_name
            self._count = 1
        if self._count >= self.stable_frames:
            self._stable = gesture_name
            self._candidate = None
            self._count = 0
        return self._stable

    def reset(self, name: str = NONE) -> None:
        """Reset the debounce state, optionally seeding a gesture name."""
        self._stable = name
        self._candidate = None
        self._count = 0


class GrabController:
    """Owns grabbable objects, hover, translation and relative rotation.

    Use :meth:`process_frame` for the live pipeline, or the lower-level
    :meth:`grab` / :meth:`translate` / :meth:`release` API (kept for the
    existing deterministic tests and synthetic drivers).
    """

    def __init__(
        self,
        hover_radius: float = 160.0,
        stable_frames: int = 2,
        lost_grace_s: float = 1.2,
        rotation_smoothing: float = 0.35,
        bounds_min: Optional[np.ndarray] = None,
        bounds_max: Optional[np.ndarray] = None,
    ) -> None:
        self._state_machine = InteractionStateMachine()
        self._objects: Dict[str, GrabbableObject] = {}
        self._grabbed: Optional[GrabbableObject] = None
        self._offset: Optional[np.ndarray] = None
        self._hovering = False
        self._hover_radius = float(hover_radius)
        self._lost_grace_s = float(lost_grace_s)
        self._rotation_smoothing = min(max(float(rotation_smoothing), 0.0), 1.0)
        self._bounds_min = (
            DEFAULT_BOUNDS_MIN.copy()
            if bounds_min is None
            else np.asarray(bounds_min, dtype=float).reshape(3)
        )
        self._bounds_max = (
            DEFAULT_BOUNDS_MAX.copy()
            if bounds_max is None
            else np.asarray(bounds_max, dtype=float).reshape(3)
        )
        self._gesture_filter = GestureStabilizer(stable_frames=stable_frames)
        self._reference_orientation: Optional[Quaternion] = None
        self._object_orientation_at_grab: Optional[Quaternion] = None
        self._display_orientation: Optional[Quaternion] = None
        self._lost_since: Optional[float] = None
        self._last_cursor: Optional[np.ndarray] = None

    # --- introspection ------------------------------------------------------

    @property
    def state_machine(self) -> InteractionStateMachine:
        return self._state_machine

    @property
    def grabbed(self) -> Optional[GrabbableObject]:
        return self._grabbed

    @property
    def hovering(self) -> bool:
        return self._hovering

    @property
    def hover_radius(self) -> float:
        return self._hover_radius

    @property
    def display_orientation(self) -> Optional[Quaternion]:
        """Smoothed camera-space object orientation (``None`` before first grab)."""
        return self._display_orientation

    @property
    def gesture_filter(self) -> GestureStabilizer:
        return self._gesture_filter

    @property
    def primary_object(self) -> Optional[GrabbableObject]:
        """First registered object (the single-cube prototype)."""
        for obj in self._objects.values():
            return obj
        return None

    def object(self, object_id: str) -> GrabbableObject:
        return self._objects[object_id]

    # --- lower-level API (existing tests / synthetic drivers) ----------------

    def register_object(self, obj: GrabbableObject) -> None:
        """Register a grabbable object."""
        self._objects[obj.object_id] = obj

    def grab(self, object_id: str, hand_origin_camera: np.ndarray) -> None:
        """Grab an object, storing the hand-to-object offset.

        The offset prevents the object from jumping when grab begins:
            object.position = hand.position + offset
        """
        obj = self._objects[object_id]
        self._offset = np.asarray(obj.position, dtype=float) - np.asarray(
            hand_origin_camera, dtype=float
        )
        self._grabbed = obj
        if self._state_machine.state not in (
            InteractionState.HOVERING,
            InteractionState.SELECTED,
            InteractionState.GRABBING,
        ):
            self._state_machine.set_hovering(True, object_available=True)
        self._state_machine.handle_gesture(
            Gesture(name=PINCH, confidence=1.0),
            has_tracking=True,
            object_available=True,
        )

    def release(self) -> None:
        """Release the grabbed object; it keeps its last valid pose."""
        self._grabbed = None
        self._offset = None
        self._reference_orientation = None
        self._object_orientation_at_grab = None
        self._state_machine.release(Gesture(name=OPEN_PALM, confidence=1.0))
        if self._state_machine.state not in (
            InteractionState.IDLE,
            InteractionState.LOST_TRACKING,
            InteractionState.PAUSED,
        ):
            self._state_machine.handle_gesture(
                Gesture(name=OPEN_PALM, confidence=1.0),
                has_tracking=True,
                object_available=False,
            )

    def translate(self, hand_origin_camera: np.ndarray) -> None:
        """Move the grabbed object following the hand, preserving the offset."""
        if self._grabbed is None or self._offset is None:
            return
        position = np.asarray(hand_origin_camera, dtype=float).reshape(3) + self._offset
        self._grabbed.position = np.clip(position, self._bounds_min, self._bounds_max)

    # --- rotation -------------------------------------------------------------

    def begin_rotation(self, orientation: Quaternion) -> None:
        """Capture the palm reference orientation at grab time."""
        self._reference_orientation = quaternion_normalized(orientation)
        if self._display_orientation is not None:
            self._object_orientation_at_grab = self._display_orientation
        else:
            self._object_orientation_at_grab = Quaternion()
        self._display_orientation = self._object_orientation_at_grab

    def apply_rotation(self, orientation: Quaternion) -> None:
        """Apply a relative palm rotation to the held object.

        The delta is a world-space (extrinsic) rotation:
        ``q_target = (q_now * q_ref^-1) * q_object_at_grab``.
        Output is slerp-smoothed toward the target.
        """
        current = quaternion_normalized(orientation)
        if self._reference_orientation is None or self._object_orientation_at_grab is None:
            self.begin_rotation(current)
            return
        delta = quaternion_multiply(current, quaternion_conjugate(self._reference_orientation))
        delta = quaternion_normalized(delta)
        target = quaternion_multiply(delta, self._object_orientation_at_grab)
        target = quaternion_normalized(target)
        if self._display_orientation is None:
            self._display_orientation = target
        else:
            self._display_orientation = quaternion_normalized(
                slerp(self._display_orientation, target, self._rotation_smoothing)
            )

    # --- live pipeline ----------------------------------------------------------

    def process_frame(
        self,
        cursor_camera: Optional[np.ndarray],
        gesture_name: str,
        orientation: Optional[Quaternion] = None,
        timestamp_s: Optional[float] = None,
    ) -> InteractionState:
        """Process one tracking sample and update held-object transforms.

        Args:
            cursor_camera: Filtered cursor position in camera space, or
                ``None`` when no hand is tracked.
            gesture_name: Raw gesture name (PINCH / OPEN_PALM / FIST / NONE).
            orientation: Camera-space palm quaternion, or ``None`` when
                unavailable or degenerate.
            timestamp_s: Sample timestamp (monotonic seconds). Defaults to
                the local monotonic clock.

        Returns:
            The resulting :class:`InteractionState`.
        """
        has_tracking = cursor_camera is not None
        stable = self._gesture_filter.update(gesture_name if has_tracking else NONE)

        if not has_tracking:
            return self._handle_lost(timestamp_s)

        cursor = np.asarray(cursor_camera, dtype=float).reshape(3)
        self._last_cursor = cursor
        resumed = self._lost_since is not None
        self._lost_since = None

        if self._grabbed is not None:
            if resumed:
                # Re-acquire the palm reference so a reappearing hand does
                # not snap the object back to a stale orientation.
                self._reference_orientation = None
                self._object_orientation_at_grab = (
                    self._display_orientation
                    if self._display_orientation is not None
                    else Quaternion()
                )
                self._gesture_filter.reset(stable)
            if stable == OPEN_PALM:
                self.release()
                return self.state_machine.state
            # Keep holding through PINCH / FIST / transient NONE.
            self.translate(cursor)
            if orientation is not None:
                if self._reference_orientation is None:
                    self.begin_rotation(orientation)
                else:
                    self.apply_rotation(orientation)
            if self.state_machine.state != InteractionState.GRABBING:
                # Restore GRABBING if a stray transition left the machine
                # elsewhere while an object is still attached.
                self._state_machine.set_hovering(True, object_available=True)
                self._state_machine.handle_gesture(
                    Gesture(name=PINCH, confidence=1.0),
                    has_tracking=True,
                    object_available=True,
                )
            return self.state_machine.state

        self._update_hover(cursor)
        state = self._state_machine.handle_gesture(
            Gesture(name=stable, confidence=1.0),
            has_tracking=True,
            object_available=self._hovering,
        )
        if (
            state == InteractionState.GRABBING
            and self._grabbed is None
            and self.primary_object is not None
        ):
            self._attach(self.primary_object.object_id, cursor)
            if orientation is not None:
                self.begin_rotation(orientation)
        return self.state_machine.state

    def _attach(self, object_id: str, hand_origin_camera: np.ndarray) -> None:
        """Attach to an object without re-driving the state machine."""
        obj = self._objects[object_id]
        self._offset = np.asarray(obj.position, dtype=float) - np.asarray(
            hand_origin_camera, dtype=float
        )
        self._grabbed = obj

    def _update_hover(self, cursor_camera: np.ndarray) -> None:
        obj = self.primary_object
        if obj is None:
            self._hovering = False
            self._state_machine.set_hovering(False, object_available=False)
            return
        distance = float(
            np.linalg.norm(np.asarray(obj.position, dtype=float) - cursor_camera)
        )
        self._hovering = distance <= self._hover_radius
        if self._grabbed is None:
            self._state_machine.set_hovering(
                self._hovering, object_available=self._hovering
            )

    def _handle_lost(self, timestamp_s: Optional[float]) -> InteractionState:
        now = float(timestamp_s) if timestamp_s is not None else time.monotonic()
        if self._grabbed is not None:
            if self._lost_since is None:
                self._lost_since = now
            if now - self._lost_since >= self._lost_grace_s:
                self.release()
                self._gesture_filter.reset()
                return self._state_machine.handle_gesture(
                    Gesture(name=NONE, confidence=0.0),
                    has_tracking=False,
                    object_available=False,
                )
            # Within grace: freeze pose, keep holding.
            return self.state_machine.state
        self._hovering = False
        self._lost_since = None
        return self._state_machine.handle_gesture(
            Gesture(name=NONE, confidence=0.0),
            has_tracking=False,
            object_available=False,
        )
