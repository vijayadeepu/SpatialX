"""Milestone 4 — interaction state machine and grab controller tests."""

from __future__ import annotations

import numpy as np

from spatialx.gestures import FIST, NONE, OPEN_PALM, PINCH, Gesture
from spatialx.interaction import GrabbableObject, GrabController, InteractionState, InteractionStateMachine


def _machine_with_object() -> tuple[InteractionStateMachine, GrabController]:
    machine = InteractionStateMachine()
    controller = GrabController()
    controller.register_object(GrabbableObject(object_id="cube-1", position=np.array([1.0, 2.0, 3.0])))
    return machine, controller


def test_open_palm_selects_object() -> None:
    machine, controller = _machine_with_object()
    new_state = machine.handle_gesture(Gesture(name=OPEN_PALM, confidence=1.0), has_tracking=True, object_available=True)
    assert new_state == InteractionState.SELECTED
    controller.grab("cube-1", np.array([0.0, 0.0, 0.0]))


def test_pinch_grabs_from_selected() -> None:
    machine, controller = _machine_with_object()
    machine.handle_gesture(Gesture(name=OPEN_PALM, confidence=1.0), has_tracking=True, object_available=True)
    new_state = machine.handle_gesture(Gesture(name=PINCH, confidence=1.0), has_tracking=True, object_available=True)
    assert new_state == InteractionState.GRABBING


def test_accidental_grab_prevented_from_idle() -> None:
    machine, controller = _machine_with_object()
    new_state = machine.handle_gesture(Gesture(name=PINCH, confidence=1.0), has_tracking=True, object_available=True)
    assert new_state == InteractionState.IDLE


def test_grab_offset_prevents_object_jump() -> None:
    machine, controller = _machine_with_object()
    controller.state_machine.handle_gesture(Gesture(name=OPEN_PALM, confidence=1.0), has_tracking=True, object_available=True)
    controller.grab("cube-1", np.array([5.0, 6.0, 7.0]))
    assert controller.state_machine.state == InteractionState.GRABBING
    controller.translate(np.array([5.5, 6.5, 7.5]))
    # Object follows the hand via the fixed offset: moved by the hand delta.
    assert np.allclose(controller.grabbed.position, [1.5, 2.5, 3.5])


def test_release_returns_to_idle() -> None:
    machine, controller = _machine_with_object()
    controller.state_machine.handle_gesture(Gesture(name=OPEN_PALM, confidence=1.0), has_tracking=True, object_available=True)
    controller.grab("cube-1", np.array([0.0, 0.0, 0.0]))
    controller.release()
    assert controller.state_machine.state == InteractionState.IDLE
    assert controller.grabbed is None


def test_fist_pauses_interaction() -> None:
    machine, controller = _machine_with_object()
    new_state = machine.handle_gesture(Gesture(name=FIST, confidence=1.0), has_tracking=True, object_available=True)
    assert new_state == InteractionState.PAUSED


def test_lost_tracking_state() -> None:
    machine, controller = _machine_with_object()
    new_state = machine.handle_gesture(Gesture(name=NONE, confidence=0.0), has_tracking=False, object_available=True)
    assert new_state == InteractionState.LOST_TRACKING


def test_none_gesture_keeps_idle() -> None:
    machine, controller = _machine_with_object()
    new_state = machine.handle_gesture(Gesture(name=NONE, confidence=0.0), has_tracking=True, object_available=True)
    assert new_state == InteractionState.IDLE
