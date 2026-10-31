"""Application layer: CLI entry point and runtime orchestration."""

from __future__ import annotations

import argparse
import time
from typing import TYPE_CHECKING, Sequence, Optional

from spatialx.core.exceptions import CameraError, VisionError
from spatialx.spatial import CameraCalibration, landmark_camera_space

if TYPE_CHECKING:  # pragma: no cover - typing only
    from spatialx.camera import CameraManager

MODEL_PATH_ENV_VAR = "SPATIALX_MODEL_PATH"


def build_parser() -> argparse.ArgumentParser:
    """Create the SpatialX command line argument parser."""
    parser = argparse.ArgumentParser(
        prog="spatialx",
        description="SpatialX — camera-based 3D human spatial interaction platform.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera device index to use (default: 0).",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=15,
        help="Number of frames to capture during a headless capture run (default: 15).",
    )
    parser.add_argument(
        "--detect",
        action="store_true",
        help="Run MediaPipe hand detection over captured frames (headless demo).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help=(
            "Path to the MediaPipe hand landmarker model file. When omitted, "
            f"the model is auto-resolved from the bundled assets or ${MODEL_PATH_ENV_VAR}."
        ),
    )
    parser.add_argument(
        "--num-hands",
        type=int,
        default=2,
        help="Number of hands to track (default: 2).",
    )
    parser.add_argument(
        "--cursor-demo",
        action="store_true",
        help="Run the virtual cursor pipeline over captured frames (headless demo).",
    )
    parser.add_argument(
        "--cursor-sensitivity",
        type=float,
        default=1.0,
        help="Cursor sensitivity multiplier (default: 1.0).",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show the live camera feed in a window (press 'q' to quit).",
    )
    parser.add_argument(
        "--render-demo",
        action="store_true",
        help=(
            "Open the interactive 3D scene (Panda3D window): hand-tracked "
            "spatial cursor and virtual cube; tracking runs in an isolated "
            "subprocess. Press 'q' or Escape to quit."
        ),
    )
    parser.add_argument(
        "--render-screenshot",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Render the 3D scene offscreen (software pipe, no camera "
            "needed) and save a PNG screenshot to PATH."
        ),
    )
    return parser


def _capture_sample(manager: CameraManager, count: int) -> tuple[list, float]:
    """Capture up to ``count`` frames and return frames plus elapsed seconds."""
    frames = []
    start = time.monotonic()
    while len(frames) < count:
        frame = manager.read()
        if frame is None:
            break
        frames.append(frame)
    elapsed = time.monotonic() - start
    return frames, elapsed


def _preview(manager: CameraManager) -> None:
    """Display the live camera feed until the window is closed or 'q' pressed."""
    import cv2

    window = "SpatialX — Camera Feed"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    frames_captured = 0
    frame_count = 0
    start = time.monotonic()
    while True:
        frame = manager.read()
        if frame is not None:
            fps = frame_count / max(time.monotonic() - start, 1e-6)
            frame_count += 1
            annotated = frame.image.copy()
            cv2.putText(
                annotated,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.imshow(window, annotated)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty(
            window, cv2.WND_PROP_VISIBLE
        ) <= 0:
            break
    cv2.destroyAllWindows()


def _run_detection(
    frames: list,
    model_path: Optional[str],
    num_hands: int,
) -> int:
    """Run the MediaPipe hand landmarker over captured frames.

    Returns:
        Process exit code: 0 on success, 4 on inference failure.
    """
    from spatialx.vision import HandLandmarker

    landmarker = HandLandmarker(model_path=model_path, num_hands=num_hands)
    frames_with_hands = 0
    total_hands = 0
    inference_times_ms: list[float] = []
    try:
        for frame in frames:
            detection = landmarker.detect(frame.image, timestamp_ms=int(round(frame.timestamp * 1000)))
            if detection is None:
                continue
            if detection.inference_time_ms is not None:
                inference_times_ms.append(detection.inference_time_ms)
            if detection.hands:
                frames_with_hands += 1
                total_hands += len(detection.hands)
    except VisionError as error:
        print(f"SpatialX vision error: {error}")
        return 4
    finally:
        landmarker.close()
    elapsed = 0.0
    if frames:
        start = frames[0].timestamp
        end = frames[-1].timestamp
        elapsed = end - start if end > start else 0.0
    fps = len(frames) / elapsed if elapsed > 0 else 0.0
    avg_inference = sum(inference_times_ms) / len(inference_times_ms) if inference_times_ms else 0.0
    print(
        f"SpatialX vision OK: model={landmarker.model_path} frames_captured={len(frames)} "
        f"frames_with_hands={frames_with_hands} total_hands={total_hands} "
        f"avg_inference_ms={avg_inference:.2f} fps={fps:.2f}"
    )
    return 0


def _run_cursor_demo(
    frames: list,
    model_path: Optional[str],
    num_hands: int,
    sensitivity: float,
) -> int:
    """Run the virtual cursor pipeline over captured frames.

    Returns:
        Process exit code: 0 on success, 4 on inference failure.
    """
    from spatialx.rendering import VirtualCursor, estimate_jitter
    from spatialx.vision import HandLandmarker

    landmarker = HandLandmarker(model_path=model_path, num_hands=num_hands)
    cursor = VirtualCursor(sensitivity=sensitivity)
    positions: list = []
    try:
        for frame in frames:
            detection = landmarker.detect(frame.image, timestamp_ms=int(round(frame.timestamp * 1000)))
            if detection is None or not detection.hands:
                continue
            wrist = detection.hands[0].landmarks[0]
            origin_camera = landmark_camera_space(
                wrist,
                CameraCalibration.for_image(frame.width, frame.height),
                image_width=frame.width,
                image_height=frame.height,
                mirror=True,  # selfie view: cursor follows the user's hand
            )
            positions.append(cursor.update(origin_camera, timestamp_s=frame.timestamp))
    except VisionError as error:
        print(f"SpatialX vision error: {error}")
        return 4
    finally:
        landmarker.close()
    final_position = positions[-1] if positions else None
    jitter = estimate_jitter(positions) if positions else 0.0
    print(
        f"SpatialX cursor OK: frames_captured={len(frames)} cursor_positions="
        f"{len(positions)} final_position={final_position.tolist() if final_position is not None else None} "
        f"jitter={jitter:.6f}"
    )
    return 0


def _run_render_screenshot(path: str) -> int:
    """Render the 3D scene offscreen and save a PNG screenshot.

    Uses the pure-software ``tinydisplay`` pipe (the GLX offscreen buffer
    is unreliable on some drivers) and never touches the camera.

    Returns:
        Process exit code: 0 on success, 5 on render failure.
    """
    from spatialx.rendering import SpatialScene

    scene = SpatialScene(
        window_type="offscreen",
        width=800,
        height=600,
        display_pipe="tinydisplay",
    )
    try:
        scene.add_cube(size=1.0)
        scene.add_cursor()
        scene.enable_cube_spin(degrees_per_second=60.0)
        frames = 40
        for index in range(frames):
            fraction = index / max(frames - 1, 1)
            scene.update_cursor(
                [(-0.25 + 0.5 * fraction) * 640.0,
                 (0.15 - 0.3 * fraction) * 480.0,
                 0.0]
            )
            scene.step(1)
        written = scene.save_screenshot(path)
        print(f"SpatialX render OK: screenshot={path} written={written}")
        return 0 if written else 5
    finally:
        scene.close()


def _run_render_demo(args) -> int:
    """Interactive 3D demo: hand-tracked cursor + virtual cube.

    Rendering runs in this process (Panda3D, onscreen GLX); camera and
    MediaPipe tracking run in an isolated subprocess (TrackingClient) to
    avoid the MediaPipe/Panda3D runtime clash.

    Returns:
        Process exit code: 0 on quit, 2 camera error, 4 vision error.
    """
    from spatialx.application.tracking_link import TrackingClient
    from spatialx.rendering import SpatialScene

    scene = SpatialScene(window_type="onscreen", width=800, height=600)
    cube = scene.add_cube(size=1.0)
    scene.add_cursor()
    scene.enable_cube_spin(degrees_per_second=25.0)
    client = TrackingClient(
        camera_index=args.camera_index,
        model_path=args.model_path,
        num_hands=args.num_hands,
        mirror=True,
    )
    latest = {"error": None, "state": None}

    def update(_dt: float) -> None:
        state = client.latest_state(timeout=0.0)
        if state is None:
            return
        if state.get("type") == "error":
            latest["error"] = state
            scene.base.userExit()
            return
        latest["state"] = state
        if state.get("cursor_camera") is not None:
            scene.update_cursor(state["cursor_camera"])
        gesture = state.get("gesture")
        if gesture == "FIST":
            scene.set_cube_spin(False)
            cube.set_color(0.85, 0.3, 0.25)
        elif gesture == "PINCH":
            scene.set_cube_spin(True)
            cube.set_color(0.3, 0.85, 0.4)
        elif gesture == "OPEN_PALM":
            scene.set_cube_spin(True)
            cube.set_color(0.25, 0.6, 0.9)

    # Wait briefly for the worker's first message so startup errors
    # (camera/model) surface before entering the interactive loop.
    first = client.latest_state(timeout=15.0)
    if first is not None and first.get("type") == "error":
        print(f"SpatialX tracking error: {first.get('message')}")
        client.close()
        return 2 if first.get("kind") == "camera" else 4
    latest["state"] = first

    print(
        "SpatialX render demo: show a hand to move the yellow cursor; "
        "FIST pauses the cube (red), PINCH spins it (green), "
        "OPEN_PALM resets it (blue). Press Escape or close the window to quit."
    )
    try:
        scene.run(update)
    except SystemExit:
        # Panda3D's window-close/escape path invokes sys.exit() internally;
        # the demo ends normally either way.
        pass
    finally:
        client.close()
    error = latest["error"]
    if error is not None:
        print(f"SpatialX tracking error: {error.get('message')}")
        return 2 if error.get("kind") == "camera" else 4
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the SpatialX CLI. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Render modes must never load camera/vision code (cv2 and MediaPipe)
    # in this process: importing cv2 or MediaPipe before Panda3D corrupts
    # the GL pipeline. The tracking worker validates camera/model itself.
    if args.render_screenshot:
        return _run_render_screenshot(args.render_screenshot)
    if args.render_demo:
        return _run_render_demo(args)

    # Validate the model asset up front so a missing model produces a clear
    # error before any camera or inference work happens.
    try:
        from spatialx.vision import resolve_model_path

        resolve_model_path(args.model_path)
    except VisionError as error:
        print(f"SpatialX vision error: {error}")
        return 4

    from spatialx.camera import CameraManager

    manager = CameraManager(device_index=args.camera_index)
    try:
        if not manager.open():
            print(
                f"SpatialX error: camera {args.camera_index} could not be opened."
            )
            return 2
    except CameraError as error:
        print(f"SpatialX error: {error}")
        return 2

    try:
        if args.preview:
            _preview(manager)
        else:
            frames, elapsed = _capture_sample(manager, args.frames)
            if not frames:
                print(
                    f"SpatialX error: failed to capture frames from camera "
                    f"{args.camera_index}."
                )
                return 3
            if args.detect:
                return _run_detection(
                    frames, model_path=args.model_path, num_hands=args.num_hands
                )
            if args.cursor_demo:
                return _run_cursor_demo(
                    frames,
                    model_path=args.model_path,
                    num_hands=args.num_hands,
                    sensitivity=args.cursor_sensitivity,
                )
            fps = len(frames) / elapsed if elapsed > 0 else 0.0
            print(
                f"SpatialX camera OK: index={manager.device_index} "
                f"frames_captured={len(frames)} resolution="
                f"{frames[0].width}x{frames[0].height} fps={fps:.2f}"
            )
    finally:
        manager.close()
    return 0
