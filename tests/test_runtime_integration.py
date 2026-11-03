"""P3–P6 runtime integration: interaction wired into the scene graph.

These tests do not open a camera and use ``window_type="none"`` so they can
run in-process. Hardware (webcam) verification is a separate, manual/CI
step — see PROJECT_STATUS.md.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from spatialx.interaction import GrabController, GrabbableObject, InteractionState
from spatialx.spatial.quaternion import Quaternion, rotation_vector_to_quaternion

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_PREFIX = "RESULT_JSON:"


def _scene_script(body: str, timeout: int = 60) -> dict:
    script = textwrap.dedent(
        f"""
        import json
        import sys

        def _emit(payload):
            print({RESULT_PREFIX!r} + json.dumps(payload))
            sys.stdout.flush()

        try:
{textwrap.indent(textwrap.dedent(body).strip(), "            ")}
        except Exception as error:
            _emit({{"ok": False, "error": repr(error)}})
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(REPO_ROOT),
    )
    for line in completed.stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX) :])
    raise AssertionError(
        "scene script produced no result\n"
        f"stdout:\n{completed.stdout[-3000:]}\n"
        f"stderr:\n{completed.stderr[-3000:]}"
    )


def test_controller_drives_scene_cube(tmp_path: Path) -> None:
    """Synthetic tracking frames move and rotate the scene-graph cube."""
    result = _scene_script(
        """
        import numpy as np
        from spatialx.interaction import GrabController, GrabbableObject
        from spatialx.rendering import SpatialScene
        from spatialx.spatial.quaternion import (
            Quaternion,
            rotation_vector_to_quaternion,
        )

        scene = SpatialScene(window_type="none")
        cube = scene.add_cube(size=1.0)
        scene.add_cursor()

        controller = GrabController(
            hover_radius=160.0, stable_frames=1, rotation_smoothing=1.0
        )
        controller.register_object(
            GrabbableObject("cube", position=cube.get_position_camera_space())
        )

        def apply():
            obj = controller.primary_object
            if obj is not None:
                cube.set_position_camera_space(obj.position)
            if controller.display_orientation is not None:
                cube.set_orientation(controller.display_orientation)

        origin = np.array([0.0, 0.0, 0.0])
        controller.process_frame(origin, "PINCH", Quaternion(), 0.0)
        apply()
        grabbed_scene = cube.get_position_scene().copy()
        identity_cam = cube.get_orientation()

        controller.process_frame(
            np.array([80.0, -40.0, 0.0]), "PINCH", Quaternion(), 0.033
        )
        apply()
        moved_scene = cube.get_position_scene().copy()

        q90 = rotation_vector_to_quaternion(np.array([0.0, 0.0, np.pi / 2]))
        controller.process_frame(np.array([80.0, -40.0, 0.0]), "PINCH", q90, 0.066)
        apply()
        rotated = cube.get_orientation()

        # release: object keeps pose
        controller.process_frame(
            np.array([80.0, -40.0, 0.0]), "OPEN_PALM", q90, 0.099
        )
        apply()
        released_scene = cube.get_position_scene().copy()
        released_orient = cube.get_orientation()

        def qdot(a, b):
            return abs(a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z)

        _emit({
            "ok": True,
            "grabbed_scene": [round(float(v), 4) for v in grabbed_scene],
            "moved_scene": [round(float(v), 4) for v in moved_scene],
            "released_scene": [round(float(v), 4) for v in released_scene],
            "moved": bool(np.linalg.norm(moved_scene - grabbed_scene) > 0.1),
            "released_stayed": bool(
                np.allclose(released_scene, moved_scene, atol=1e-6)
            ),
            "identity_dot": round(float(qdot(identity_cam, Quaternion())), 6),
            "rotated_dot_vs_q90": round(float(qdot(rotated, q90)), 6),
            "state": controller.state_machine.state.name,
        })
        scene.close()
        """
    )
    assert result["ok"] is True, result
    assert result["moved"] is True
    assert result["released_stayed"] is True
    assert result["identity_dot"] == pytest.approx(1.0, abs=1e-6)
    assert result["rotated_dot_vs_q90"] == pytest.approx(1.0, abs=1e-6)
    assert result["state"] == "IDLE"


def test_scene_cursor_visibility_and_status(tmp_path: Path) -> None:
    result = _scene_script(
        """
        from spatialx.rendering import SpatialScene

        scene = SpatialScene(window_type="none")
        scene.add_cursor()
        scene.set_cursor_visible(False)
        hidden = scene.cursor.node.isHidden()
        scene.set_cursor_visible(True)
        shown = not scene.cursor.node.isHidden()
        scene.set_status_text("state=IDLE gesture=NONE")
        _emit({"ok": True, "hidden": bool(hidden), "shown": bool(shown)})
        scene.close()
        """
    )
    assert result["ok"] is True, result
    assert result["hidden"] is True
    assert result["shown"] is True


def test_rendering_import_does_not_load_mediapipe() -> None:
    """The Panda3D process must stay MediaPipe-free (GL corruption)."""
    script = (
        "import sys;"
        "import spatialx.rendering;"
        "import spatialx.interaction;"
        "import spatialx.gestures;"
        "bad = [m for m in sys.modules if m.startswith('mediapipe') or m == 'cv2'];"
        "print('RESULT_JSON:' + str({'ok': not bad, 'loaded': bad}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(REPO_ROOT),
    )
    for line in completed.stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            payload = eval(line[len(RESULT_PREFIX) :])  # noqa: S307 - test-only
            assert payload["ok"] is True, payload
            return
    raise AssertionError(f"no result\nstdout={completed.stdout}\nstderr={completed.stderr}")


def test_tracking_worker_reports_missing_camera() -> None:
    """An unusable camera index produces an explicit error message."""
    from spatialx.application.tracking_link import TrackingClient

    client = TrackingClient(camera_index=99, mirror=True, target_fps=5.0)
    try:
        message = client.latest_state(timeout=10.0)
    finally:
        client.close()
    assert message is not None
    assert message.get("type") == "error"
    assert message.get("kind") == "camera"
    assert "99" in str(message.get("message")) or "camera" in str(message.get("message"))


def test_tracking_worker_protocol_has_version(tmp_path: Path) -> None:
    """Worker messages are versioned JSON (v=1) with the documented fields."""
    from spatialx.application.tracking_link import TrackingClient

    # This test uses the real camera. Skip cleanly if the device is missing.
    try:
        client = TrackingClient(camera_index=0, mirror=True, target_fps=10.0)
    except Exception as error:  # pragma: no cover - no camera in CI
        pytest.skip(f"camera unavailable: {error}")
        return
    messages = []
    try:
        deadline_last = None
        import time

        end = time.monotonic() + 8.0
        while time.monotonic() < end and len(messages) < 5:
            batch = client.drain_messages(timeout=1.0)
            for message in batch:
                if message.get("type") == "error":
                    pytest.skip(f"worker error: {message.get('message')}")
                messages.append(message)
    finally:
        client.close()
    assert messages, "expected at least one tracking message"
    for message in messages:
        assert message.get("v") == 1
        assert message.get("type") == "state"
        assert set(message) >= {
            "type",
            "timestamp",
            "hands",
            "cursor_camera",
            "orientation",
            "gesture",
        }
        assert isinstance(message.get("hands"), int)
        assert message.get("gesture") in {"PINCH", "OPEN_PALM", "FIST", "NONE"}
        if message.get("cursor_camera") is not None:
            assert len(message["cursor_camera"]) == 3
        if message.get("orientation") is not None:
            assert len(message["orientation"]) == 4
