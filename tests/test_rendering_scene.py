"""Priority 3 — 3D scene tests.

Scene creation and real rendering happen in isolated subprocesses:
MediaPipe (loaded by other test modules in this process) and Panda3D
corrupt each other's LLVM/GL global state when combined in one process,
so every test that opens a Panda3D window runs in a fresh interpreter.
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


def _run_scene_script(body: str, timeout: int = 120) -> dict:
    """Run a scene snippet in a fresh interpreter and parse its JSON result."""
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
            return json.loads(line[len(RESULT_PREFIX):])
    raise AssertionError(
        "scene script produced no result\n"
        f"stdout:\n{completed.stdout[-3000:]}\n"
        f"stderr:\n{completed.stderr[-3000:]}"
    )


def test_offscreen_scene_renders_screenshot(tmp_path: Path) -> None:
    """A real rendered frame: cube, axes, grid and cursor, saved as PNG."""
    output = tmp_path / "scene.png"
    result = _run_scene_script(
        f"""
        from panda3d.core import PNMImage
        from spatialx.rendering import SpatialScene

        scene = SpatialScene(
            window_type="offscreen", width=640, height=480,
            display_pipe="tinydisplay",
        )
        scene.add_cube(size=1.0)
        scene.add_cursor()
        scene.enable_cube_spin(degrees_per_second=90.0)
        for index in range(20):
            scene.update_cursor([100.0 - index * 10.0, -40.0, 0.0])
            scene.step(1)
        written = scene.save_screenshot({str(output)!r})
        image = PNMImage({str(output)!r})
        # dense pixel sweep: a real render must show object content, not
        # just the flat clear color.
        values = []
        non_background = 0
        red_axis = False
        blue_cube = False
        total = 0
        for x in range(4, image.getXSize(), 8):
            for y in range(4, image.getYSize(), 8):
                r = float(image.getRed(x, y))
                g = float(image.getGreen(x, y))
                b = float(image.getBlue(x, y))
                total += 1
                values.append((round(r, 2), round(g, 2), round(b, 2)))
                if abs(r - 0.055) > 0.03 or abs(g - 0.065) > 0.03 or abs(b - 0.09) > 0.03:
                    non_background += 1
                if r > 0.7 and g < 0.4 and b < 0.4:
                    red_axis = True
                if b > 0.5 and r < 0.4:
                    blue_cube = True
        _emit({{
            "ok": written,
            "written": written,
            "distinct_samples": len(set(values)),
            "non_background_fraction": round(non_background / max(total, 1), 3),
            "red_axis": red_axis,
            "blue_cube": blue_cube,
            "size": [image.getXSize(), image.getYSize()],
            "spin_heading": round(scene.cube.node.getH(), 1),
        }})
        scene.close()
        """
    )
    assert result["ok"] is True, result
    assert output.is_file() and output.stat().st_size > 1000
    assert result["size"] == [640, 480]
    assert result["distinct_samples"] >= 5, "render looks flat/empty"
    assert result["non_background_fraction"] > 0.05, "no object pixels found"
    assert result["red_axis"] is True, "X axis (red) not rendered"
    assert result["blue_cube"] is True, "cube not rendered"
    assert result["spin_heading"] != 0.0, "cube spin task did not run"


def test_scene_object_transform_roundtrip(tmp_path: Path) -> None:
    """Camera-space position and quaternion orientation roundtrip."""
    result = _run_scene_script(
        """
        import numpy as np
        from spatialx.rendering import SpatialScene
        from spatialx.spatial.quaternion import rotation_vector_to_quaternion

        scene = SpatialScene(
            window_type="offscreen", width=320, height=240,
            display_pipe="tinydisplay",
        )
        cube = scene.add_cube(size=1.0)
        cube.set_position_camera_space([120.0, -80.0, 40.0])
        back = cube.get_position_camera_space()

        quat = rotation_vector_to_quaternion(np.array([0.0, 0.0, 1.5707963]))
        cube.set_orientation(quat)
        got = cube.get_orientation()
        dot = (got.w * quat.w + got.x * quat.x + got.y * quat.y + got.z * quat.z)

        cursor = scene.add_cursor()
        scene.update_cursor([0.0, 0.0, 0.0])
        cursor_at_origin = cursor.get_position_camera_space()

        _emit({
            "ok": True,
            "position_back": [round(v, 4) for v in back],
            "orientation_dot": round(float(abs(dot)), 9),
            "cursor_at_origin": [round(v, 4) for v in cursor_at_origin],
        })
        scene.close()
        """
    )
    assert result["ok"] is True, result
    assert result["position_back"] == [120.0, -80.0, 40.0]
    assert result["orientation_dot"] == pytest.approx(1.0, abs=1e-6)
    assert result["cursor_at_origin"] == [0.0, 0.0, 0.0]


def test_scene_none_window_builds_scene_graph() -> None:
    """Scene graph operations work without any graphics system."""
    result = _run_scene_script(
        """
        from spatialx.rendering import SpatialScene

        scene = SpatialScene(window_type="none")
        cube = scene.add_cube(size=2.0)
        cursor = scene.add_cursor()
        cube.set_position((1.0, 2.0, 3.0))
        _emit({
            "ok": True,
            "cube_pos": [round(v, 3) for v in cube.get_position_scene()],
            "cursor_exists": cursor.node is not None,
            "has_graphics": scene.base.win is not None,
        })
        scene.close()
        """
    )
    assert result["ok"] is True, result
    assert result["cube_pos"] == [1.0, 2.0, 3.0]
    assert result["cursor_exists"] is True
    assert result["has_graphics"] is False
