"""Priority 3 — process-isolated tracking link tests.

The parent stays MediaPipe-free; the worker process owns the camera and
MediaPipe. Both run in a fresh interpreter spawned by this test to keep
the pytest process clean of the MediaPipe/Panda3D clash.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_PREFIX = "RESULT_JSON:"

_CLIENT_SCRIPT = textwrap.dedent(
    """
    import json
    import sys
    import time

    from spatialx.application.tracking_link import TrackingClient

    client = TrackingClient(camera_index=0, mirror=True, target_fps=20.0)
    states = []
    deadline = time.monotonic() + 30.0
    try:
        while time.monotonic() < deadline and len(states) < 12:
            state = client.latest_state(timeout=1.0)
            if state is None:
                continue
            if state.get("type") == "error":
                print("RESULT_JSON:" + json.dumps(
                    {"ok": False, "error": state.get("message"),
                     "kind": state.get("kind")}))
                sys.stdout.flush()
                sys.exit(0)
            states.append(state)
    finally:
        client.close()

    hands_seen = max((s.get("hands", 0) for s in states), default=0)
    cursor_ok = all(
        s.get("cursor_camera") is None
        or (
            isinstance(s.get("cursor_camera"), list)
            and len(s["cursor_camera"]) == 3
        )
        for s in states
    )
    orientation_ok = all(
        s.get("orientation") is None
        or (
            isinstance(s.get("orientation"), list)
            and len(s["orientation"]) == 4
        )
        for s in states
    )
    gestures = sorted({s.get("gesture") for s in states})
    print("RESULT_JSON:" + json.dumps({
        "ok": len(states) >= 3,
        "n_states": len(states),
        "hands_seen": hands_seen,
        "cursor_ok": cursor_ok,
        "orientation_ok": orientation_ok,
        "gestures": gestures,
        "inference_reported": any(
            s.get("inference_ms") is not None for s in states
        ),
    }))
    sys.stdout.flush()
    """
)


def _run_client_script(tmp_path: Path, timeout: int = 120) -> dict:
    script_path = tmp_path / "tracking_client_probe.py"
    script_path.write_text(_CLIENT_SCRIPT)
    completed = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(REPO_ROOT),
    )
    for line in completed.stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX):])
    raise AssertionError(
        "tracking probe produced no result\n"
        f"stdout:\n{completed.stdout[-3000:]}\n"
        f"stderr:\n{completed.stderr[-3000:]}"
    )


def test_tracking_client_delivers_states_from_worker(tmp_path: Path) -> None:
    result = _run_client_script(tmp_path)
    if not result.get("ok") and result.get("kind") == "camera":
        # Exclusive device: another probe/instance may hold /dev/video0.
        pytest.skip(f"camera unavailable: {result.get('error')}")
    assert result["ok"] is True, result
    assert result["n_states"] >= 3
    assert result["cursor_ok"] is True, result
    assert result["orientation_ok"] is True, result
    assert result["inference_reported"] is True, result
    assert "NONE" in result["gestures"]
