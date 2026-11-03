"""P3–P5 runtime interaction tests: hover, grab, translate, rotate, loss."""

from __future__ import annotations

import math

import numpy as np
import pytest

from spatialx.gestures import FIST, NONE, OPEN_PALM, PINCH
from spatialx.interaction import (
    GestureStabilizer,
    GrabbableObject,
    GrabController,
    InteractionState,
)
from spatialx.spatial.quaternion import (
    Quaternion,
    quaternion_conjugate,
    quaternion_multiply,
    quaternion_normalized,
    quaternion_to_rotation_matrix,
    relative_rotation,
    rotate_vector,
    rotation_matrix_to_quaternion,
    rotation_vector_to_quaternion,
    slerp,
)


def _controller(**kwargs) -> GrabController:
    controller = GrabController(**kwargs)
    controller.register_object(
        GrabbableObject(object_id="cube", position=np.array([0.0, -100.0, 0.0]))
    )
    return controller


def _feed(controller, name, t, quat=None, cursor=None):
    origin = np.array([0.0, 0.0, 0.0]) if cursor is None else cursor
    return controller.process_frame(origin, name, quat, t)


class TestGestureStabilizer:
    def test_requires_n_consecutive_frames(self) -> None:
        stab = GestureStabilizer(stable_frames=3)
        assert stab.update(PINCH) == NONE
        assert stab.update(PINCH) == NONE
        assert stab.update(PINCH) == PINCH
        assert stab.update(OPEN_PALM) == PINCH
        assert stab.update(OPEN_PALM) == PINCH
        assert stab.update(OPEN_PALM) == OPEN_PALM

    def test_same_gesture_keeps_stable(self) -> None:
        stab = GestureStabilizer(stable_frames=2)
        stab.update(PINCH)
        stab.update(PINCH)
        assert stab.update(PINCH) == PINCH
        assert stab.stable_name == PINCH

    def test_rejects_invalid_frame_count(self) -> None:
        with pytest.raises(ValueError):
            GestureStabilizer(stable_frames=0)

    def test_reset(self) -> None:
        stab = GestureStabilizer(stable_frames=1)
        stab.update(PINCH)
        stab.reset()
        assert stab.stable_name == NONE


class TestHoverAndGrab:
    def test_hover_when_cursor_near_object(self) -> None:
        controller = _controller(hover_radius=160.0, stable_frames=1)
        state = _feed(controller, OPEN_PALM, 0.0)
        assert controller.hovering is True
        assert state == InteractionState.HOVERING

    def test_no_hover_when_far(self) -> None:
        controller = _controller(hover_radius=80.0, stable_frames=1)
        state = _feed(controller, OPEN_PALM, 0.0, cursor=np.array([250.0, 0.0, 0.0]))
        assert controller.hovering is False
        assert state == InteractionState.IDLE

    def test_pinch_from_idle_does_not_grab(self) -> None:
        controller = _controller(hover_radius=80.0, stable_frames=1)
        state = _feed(controller, PINCH, 0.0, cursor=np.array([250.0, 0.0, 0.0]))
        assert state == InteractionState.IDLE
        assert controller.grabbed is None

    def test_pinch_from_hover_grabs(self) -> None:
        controller = _controller(hover_radius=160.0, stable_frames=1)
        _feed(controller, OPEN_PALM, 0.0)
        state = _feed(controller, PINCH, 0.033)
        assert state == InteractionState.GRABBING
        assert controller.grabbed is not None
        assert controller.grabbed.object_id == "cube"

    def test_grab_offset_prevents_jump(self) -> None:
        controller = _controller(hover_radius=160.0, stable_frames=1)
        # Object at (0, -100, 0); grab with cursor at origin.
        _feed(controller, PINCH, 0.0)
        assert controller.state_machine.state == InteractionState.GRABBING
        # Move the hand +40 x: object follows by the same delta, not to the hand.
        _feed(controller, PINCH, 0.033, cursor=np.array([40.0, 0.0, 0.0]))
        np.testing.assert_allclose(controller.primary_object.position, [40.0, -100.0, 0.0])

    def test_release_keeps_last_position(self) -> None:
        controller = _controller(hover_radius=160.0, stable_frames=1)
        _feed(controller, PINCH, 0.0)
        _feed(controller, PINCH, 0.033, cursor=np.array([50.0, 20.0, 0.0]))
        last = controller.primary_object.position.copy()
        _feed(controller, OPEN_PALM, 0.066)
        assert controller.state_machine.state == InteractionState.IDLE
        assert controller.grabbed is None
        np.testing.assert_allclose(controller.primary_object.position, last)

    def test_fist_while_holding_keeps_grab(self) -> None:
        controller = _controller(hover_radius=160.0, stable_frames=1)
        _feed(controller, PINCH, 0.0)
        state = _feed(controller, FIST, 0.033)
        assert state == InteractionState.GRABBING
        assert controller.grabbed is not None

    def test_fist_when_idle_pauses(self) -> None:
        controller = _controller(hover_radius=80.0, stable_frames=1)
        state = _feed(controller, FIST, 0.0, cursor=np.array([300.0, 0.0, 0.0]))
        assert state == InteractionState.PAUSED

    def test_transient_none_does_not_drop_object(self) -> None:
        controller = _controller(hover_radius=160.0, stable_frames=1)
        _feed(controller, PINCH, 0.0)
        _feed(controller, NONE, 0.033)
        assert controller.grabbed is not None
        assert controller.state_machine.state == InteractionState.GRABBING

    def test_position_clamped_to_bounds(self) -> None:
        from spatialx.interaction.grab import DEFAULT_BOUNDS_MAX

        controller = _controller(hover_radius=160.0, stable_frames=1)
        _feed(controller, PINCH, 0.0)
        _feed(controller, PINCH, 0.033, cursor=np.array([5000.0, 0.0, 0.0]))
        assert controller.primary_object.position[0] <= DEFAULT_BOUNDS_MAX[0] + 1e-9
        assert np.all(np.isfinite(controller.primary_object.position))


class TestTrackingLoss:
    def test_lost_when_idle(self) -> None:
        controller = _controller(hover_radius=80.0, stable_frames=1)
        state = controller.process_frame(None, NONE, None, 0.0)
        assert state == InteractionState.LOST_TRACKING

    def test_lost_while_holding_freezes_then_releases(self) -> None:
        controller = _controller(hover_radius=160.0, stable_frames=1, lost_grace_s=0.2)
        _feed(controller, PINCH, 0.0, cursor=np.array([10.0, 0.0, 0.0]))
        held = controller.primary_object.position.copy()
        # Within grace: freeze.
        state = controller.process_frame(None, NONE, None, 0.1)
        assert state == InteractionState.GRABBING
        np.testing.assert_allclose(controller.primary_object.position, held)
        # After grace: auto-release, object stays.
        state = controller.process_frame(None, NONE, None, 0.4)
        assert state == InteractionState.LOST_TRACKING
        assert controller.grabbed is None
        np.testing.assert_allclose(controller.primary_object.position, held)

    def test_resume_after_loss_keeps_orientation(self) -> None:
        controller = _controller(
            hover_radius=160.0, stable_frames=1, lost_grace_s=0.5, rotation_smoothing=1.0
        )
        q0 = Quaternion()
        _feed(controller, PINCH, 0.0, quat=q0)
        q90 = rotation_vector_to_quaternion(np.array([0.0, 0.0, math.pi / 2]))
        _feed(controller, PINCH, 0.033, quat=q90)
        held_orientation = controller.display_orientation
        controller.process_frame(None, NONE, None, 0.066)
        state = controller.process_frame(
            np.array([5.0, 0.0, 0.0]), PINCH, q90, 0.1
        )
        assert state == InteractionState.GRABBING
        # Reappearance re-captures the reference; object orientation must not reset.
        dot = (
            controller.display_orientation.w * held_orientation.w
            + controller.display_orientation.x * held_orientation.x
            + controller.display_orientation.y * held_orientation.y
            + controller.display_orientation.z * held_orientation.z
        )
        assert abs(abs(dot) - 1.0) < 1e-6


class TestRotation:
    def test_identity_palm_keeps_object_identity(self) -> None:
        controller = _controller(hover_radius=160.0, stable_frames=1, rotation_smoothing=1.0)
        _feed(controller, PINCH, 0.0, quat=Quaternion())
        _feed(controller, PINCH, 0.033, quat=Quaternion())
        display = controller.display_orientation
        np.testing.assert_allclose(
            [display.w, display.x, display.y, display.z],
            [1.0, 0.0, 0.0, 0.0],
            atol=1e-9,
        )

    def test_relative_z_rotation(self) -> None:
        controller = _controller(hover_radius=160.0, stable_frames=1, rotation_smoothing=1.0)
        _feed(controller, PINCH, 0.0, quat=Quaternion())
        q90 = rotation_vector_to_quaternion(np.array([0.0, 0.0, math.pi / 2]))
        _feed(controller, PINCH, 0.033, quat=q90)
        display = controller.display_orientation
        # World-space delta of +90° about camera Z applied to identity.
        rotated = rotate_vector(display, np.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(rotated, [0.0, 1.0, 0.0], atol=1e-9)

    def test_rotation_accumulates(self) -> None:
        controller = _controller(hover_radius=160.0, stable_frames=1, rotation_smoothing=1.0)
        _feed(controller, PINCH, 0.0, quat=Quaternion())
        q45 = rotation_vector_to_quaternion(np.array([0.0, 0.0, math.pi / 4]))
        q90 = rotation_vector_to_quaternion(np.array([0.0, 0.0, math.pi / 2]))
        _feed(controller, PINCH, 0.033, quat=q45)
        _feed(controller, PINCH, 0.066, quat=q90)
        rotated = rotate_vector(controller.display_orientation, np.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(rotated, [0.0, 1.0, 0.0], atol=1e-9)

    def test_none_orientation_skips_update(self) -> None:
        controller = _controller(hover_radius=160.0, stable_frames=1, rotation_smoothing=1.0)
        _feed(controller, PINCH, 0.0, quat=Quaternion())
        q90 = rotation_vector_to_quaternion(np.array([0.0, 0.0, math.pi / 2]))
        _feed(controller, PINCH, 0.033, quat=q90)
        before = controller.display_orientation
        _feed(controller, PINCH, 0.066, quat=None)
        assert controller.display_orientation == before

    def test_release_preserves_orientation(self) -> None:
        controller = _controller(hover_radius=160.0, stable_frames=1, rotation_smoothing=1.0)
        _feed(controller, PINCH, 0.0, quat=Quaternion())
        q90 = rotation_vector_to_quaternion(np.array([0.0, 0.0, math.pi / 2]))
        _feed(controller, PINCH, 0.033, quat=q90)
        held = controller.display_orientation
        _feed(controller, OPEN_PALM, 0.066, quat=q90)
        assert controller.display_orientation == held

    def test_slerp_smoothing_converges(self) -> None:
        controller = _controller(hover_radius=160.0, stable_frames=1, rotation_smoothing=0.3)
        _feed(controller, PINCH, 0.0, quat=Quaternion())
        q90 = rotation_vector_to_quaternion(np.array([0.0, 0.0, math.pi / 2]))
        for i in range(1, 40):
            _feed(controller, PINCH, 0.033 * i, quat=q90)
        dot = abs(
            controller.display_orientation.w * q90.w
            + controller.display_orientation.z * q90.z
        )
        assert dot > 0.999

    def test_degenerate_orientation_does_not_crash(self) -> None:
        controller = _controller(hover_radius=160.0, stable_frames=1)
        _feed(controller, PINCH, 0.0, quat=None)
        assert controller.display_orientation is None
        _feed(controller, PINCH, 0.033, quat=Quaternion())
        assert controller.display_orientation is not None


class TestQuaternionMath:
    def test_normalized_unit(self) -> None:
        q = quaternion_normalized(Quaternion(w=2.0, x=2.0, y=0.0, z=0.0))
        assert math.isclose(q.w**2 + q.x**2 + q.y**2 + q.z**2, 1.0, abs_tol=1e-12)

    def test_normalized_zero_is_identity(self) -> None:
        q = quaternion_normalized(Quaternion(w=0.0, x=0.0, y=0.0, z=0.0))
        assert (q.w, q.x, q.y, q.z) == (1.0, 0.0, 0.0, 0.0)

    def test_matrix_roundtrip(self) -> None:
        q = rotation_vector_to_quaternion(np.array([0.3, -0.7, 1.1]))
        m = quaternion_to_rotation_matrix(q)
        back = rotation_matrix_to_quaternion(m)
        dot = abs(q.w * back.w + q.x * back.x + q.y * back.y + q.z * back.z)
        assert math.isclose(dot, 1.0, abs_tol=1e-9)

    def test_matrix_is_rotation(self) -> None:
        q = rotation_vector_to_quaternion(np.array([0.2, 0.5, -0.8]))
        m = quaternion_to_rotation_matrix(q)
        np.testing.assert_allclose(m @ m.T, np.eye(3), atol=1e-12)
        assert math.isclose(np.linalg.det(m), 1.0, abs_tol=1e-12)

    def test_relative_rotation_maps_self_to_other(self) -> None:
        q1 = rotation_vector_to_quaternion(np.array([0.1, 0.4, 0.3]))
        q2 = rotation_vector_to_quaternion(np.array([0.5, -0.2, 0.9]))
        delta = relative_rotation(q1, q2)
        composed = quaternion_multiply(delta, q1)
        dot = abs(
            composed.w * q2.w
            + composed.x * q2.x
            + composed.y * q2.y
            + composed.z * q2.z
        )
        assert math.isclose(dot, 1.0, abs_tol=1e-9)


class TestPalmFrame:
    @staticmethod
    def _hand_from_camera(points):
        from spatialx.vision import Landmark, TrackedHand

        # camera (x right, y down, z fwd) -> normalized landmarks
        width, height = 640.0, 480.0
        landmarks = []
        for x, y, z in points:
            u = x + width / 2.0
            v = y + height / 2.0
            landmarks.append(
                Landmark(
                    index=len(landmarks),
                    x=u / width,
                    y=v / height,
                    z=z / width,
                    visibility=1.0,
                )
            )
        # pad to 21 landmarks
        while len(landmarks) < 21:
            landmarks.append(
                Landmark(index=len(landmarks), x=0.5, y=0.5, z=0.0, visibility=0.0)
            )
        return TrackedHand(handedness="RIGHT", landmarks=tuple(landmarks))

    def test_insufficient_landmarks_returns_none(self) -> None:
        from spatialx.spatial.coordinates import palm_frame_quaternion
        from spatialx.vision import Landmark, TrackedHand

        hand = TrackedHand(
            handedness="RIGHT",
            landmarks=tuple(
                Landmark(index=i, x=0.5, y=0.5, z=0.0, visibility=0.0) for i in range(5)
            ),
        )
        assert palm_frame_quaternion(hand) is None

    def test_collinear_palm_returns_none(self) -> None:
        from spatialx.spatial.coordinates import palm_frame_quaternion

        # wrist, index MCP and middle MCP on one line
        hand = self._hand_from_camera(
            [
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 10.0, 0.0),
                (0.0, 20.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 30.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 40.0, 0.0),
            ]
        )
        from spatialx.spatial.coordinates import palm_frame_quaternion as pfq

        assert pfq(hand) is None

    def test_flat_palm_produces_unit_quaternion(self) -> None:
        from spatialx.spatial.coordinates import palm_frame_quaternion

        # Palm in the image plane: fingers up (-y), thumb toward +x.
        hand = self._hand_from_camera(
            [
                (0.0, 0.0, 0.0),  # wrist
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (20.0, -40.0, 0.0),  # index MCP
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (40.0, -40.0, 0.0),  # middle MCP
            ]
        )
        quat = palm_frame_quaternion(hand)
        assert quat is not None
        norm = quat.w**2 + quat.x**2 + quat.y**2 + quat.z**2
        assert math.isclose(norm, 1.0, abs_tol=1e-12)

    def test_in_plane_rotation_about_camera_z(self) -> None:
        from spatialx.spatial.coordinates import palm_frame_quaternion

        def palm(theta: float):
            # Rotate the canonical palm by theta about camera Z (0,0,1).
            c, s = math.cos(theta), math.sin(theta)

            def rot(p):
                x, y, z = p
                return (c * x - s * y, s * x + c * y, z)

            return self._hand_from_camera(
                [
                    rot((0.0, 0.0, 0.0)),
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0),
                    rot((20.0, -40.0, 0.0)),
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0),
                    rot((40.0, -40.0, 0.0)),
                ]
            )

        q0 = palm_frame_quaternion(palm(0.0))
        q90 = palm_frame_quaternion(palm(math.pi / 2))
        assert q0 is not None and q90 is not None
        delta = relative_rotation(q0, q90)
        expected = rotation_vector_to_quaternion(np.array([0.0, 0.0, math.pi / 2]))
        dot = abs(delta.w * expected.w + delta.z * expected.z)
        assert dot > 0.999
        assert abs(delta.x) < 1e-6 and abs(delta.y) < 1e-6
