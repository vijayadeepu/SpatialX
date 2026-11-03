"""Application layer: CLI entry point and runtime orchestration."""

from __future__ import annotations

import argparse
import time
from typing import TYPE_CHECKING, Sequence, Optional

from spatialx.core.exceptions import CameraError, VisionError
from spatialx.spatial import CameraCalibration, landmark_camera_space
from spatialx import __version__

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
        "--version",
        action="version",
        version=f"SpatialX {__version__}",
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
        "--hover-radius",
        type=float,
        default=160.0,
        help=(
            "Hover radius in camera-space pixels: the cursor must be within "
            "this distance of the cube to grab it (default: 160)."
        ),
    )
    parser.add_argument(
        "--tracking-fps",
        type=float,
        default=30.0,
        help="Maximum tracking-worker publish rate in FPS (default: 30).",
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
    """Interactive 3D demo: hand-tracked cursor, cube grab/translate/rotate.

    Rendering runs in this process (Panda3D, onscreen GLX); camera and
    MediaPipe tracking run in an isolated subprocess (TrackingClient) to
    avoid the MediaPipe/Panda3D runtime clash. Interaction (hover, grab,
    translation, rotation) runs in this process on plain numpy/quaternion
    data — never on MediaPipe or Panda3D objects.

    Returns:
        Process exit code: 0 on quit, 2 camera/worker error, 4 vision error.
    """
    import numpy as np

    from spatialx.application.tracking_link import TrackingClient
    from spatialx.interaction import GrabController, GrabbableObject
    from spatialx.motion import OneEuroFilter
    from spatialx.rendering import SpatialScene, VirtualCursor
    from spatialx.spatial.quaternion import Quaternion, quaternion_normalized

    def quat_from_payload(values) -> Optional[Quaternion]:
        if not isinstance(values, (list, tuple)) or len(values) != 4:
            return None
        try:
            return quaternion_normalized(
                Quaternion(*(float(v) for v in values))
            )
        except (TypeError, ValueError):
            return None

    scene = SpatialScene(window_type="onscreen", width=800, height=600)
    cube = scene.add_cube(size=1.0)
    scene.add_cursor()
    scene.set_cursor_visible(False)

    controller = GrabController(
        hover_radius=float(args.hover_radius),
        stable_frames=2,
        lost_grace_s=1.2,
        rotation_smoothing=0.35,
    )
    controller.register_object(
        GrabbableObject(
            object_id="cube",
            position=cube.get_position_camera_space(),
        )
    )

    # Interactive One Euro settings: responsive enough to follow the hand
    # without the library-default heavy smoothing (which is tuned for
    # offline jitter metrics, not live manipulation).
    cursor = VirtualCursor(
        filter=OneEuroFilter(min_cutoff=1.2, beta=0.025, derivative_cutoff=1.0),
        sensitivity=float(args.cursor_sensitivity),
        bounds=(
            np.array([-340.0, -260.0, -120.0]),
            np.array([340.0, 260.0, 120.0]),
        ),
    )

    client = TrackingClient(
        camera_index=args.camera_index,
        model_path=args.model_path,
        num_hands=args.num_hands,
        mirror=True,
        target_fps=float(args.tracking_fps),
    )
    latest_error: Optional[dict] = None
    last_seen_timestamp: Optional[float] = None
    last_shown_state = [None]
    last_frame_stats = {
        "states": 0,
        "frames": 0,
        "inference_ms": [],
        "start": time.monotonic(),
    }

    def apply_scene() -> None:
        obj = controller.primary_object
        if obj is not None:
            cube.set_position_camera_space(obj.position)
        if controller.display_orientation is not None:
            cube.set_orientation(controller.display_orientation)
        state = controller.state_machine.state
        if state.name == "GRABBING":
            cube.set_color(0.95, 0.55, 0.15)
            scene.set_cursor_color(1.0, 0.55, 0.15)
        elif state.name == "HOVERING":
            cube.set_color(0.25, 0.85, 0.45)
            scene.set_cursor_color(0.3, 0.9, 0.95)
        elif state.name == "PAUSED":
            cube.set_color(0.6, 0.6, 0.65)
            scene.set_cursor_color(0.7, 0.7, 0.75)
        else:
            cube.set_color(0.25, 0.6, 0.9)
            scene.set_cursor_color(1.0, 0.85, 0.2)

    def process_state(state: dict) -> None:
        nonlocal last_seen_timestamp
        last_frame_stats["states"] += 1
        if state.get("type") == "error":
            nonlocal latest_error
            latest_error = state
            scene.request_exit()
            return
        if state.get("type") != "state":
            return
        timestamp = state.get("timestamp")
        cursor_camera = state.get("cursor_camera")
        if cursor_camera is not None:
            raw = np.asarray(cursor_camera, dtype=float).reshape(3)
            if (
                last_seen_timestamp is None
                or timestamp is None
                or timestamp - last_seen_timestamp > 0.5
            ):
                cursor.reset()
            last_seen_timestamp = timestamp
            filtered = cursor.update(
                raw, timestamp_s=timestamp if timestamp is not None else None
            )
            scene.update_cursor(filtered)
            scene.set_cursor_visible(True)
        else:
            scene.set_cursor_visible(False)
            if last_seen_timestamp is not None and timestamp is not None:
                if timestamp - last_seen_timestamp > 0.5:
                    cursor.reset()
        if cursor_camera is not None:
            last_frame_stats["frames"] += 1
        inference = state.get("inference_ms")
        if isinstance(inference, (int, float)):
            last_frame_stats["inference_ms"].append(float(inference))
        controller.process_frame(
            cursor_camera=(
                np.asarray(cursor_camera, dtype=float).reshape(3)
                if cursor_camera is not None
                else None
            ),
            gesture_name=str(state.get("gesture") or "NONE"),
            orientation=quat_from_payload(state.get("orientation")),
            timestamp_s=timestamp if timestamp is not None else None,
        )
        apply_scene()
        interaction_state = controller.state_machine.state.name
        if interaction_state != last_shown_state[0]:
            last_shown_state[0] = interaction_state
            print(
                f"SpatialX state: {interaction_state} "
                f"gesture={state.get('gesture')} hover={controller.hovering}"
            )
        scene.set_status_text(
            f"state={interaction_state}  gesture={state.get('gesture')}  "
            f"hands={state.get('hands')}  hover={controller.hovering}"
        )

    def update(_dt: float) -> None:
        messages = client.drain_messages(timeout=0.0)
        for message in messages:
            process_state(message)
        if not client.running and latest_error is None:
            # Worker died without an error frame.
            process_state(
                {
                    "type": "error",
                    "kind": "worker",
                    "message": "tracking worker is no longer running",
                }
            )

    # Wait briefly for the worker's first message so startup errors
    # (camera/model) surface before entering the interactive loop.
    first = client.latest_state(timeout=15.0)
    if first is not None and first.get("type") == "error":
        print(f"SpatialX tracking error: {first.get('message')}")
        client.close()
        scene.close()
        return 2 if first.get("kind") in ("camera", "worker") else 4
    if first is not None:
        process_state(first)

    print(
        "SpatialX render demo: move your hand to drive the yellow cursor. "
        "Hover near the cube and PINCH to grab; move to translate; rotate "
        "your palm to rotate the cube; OPEN_PALM to release. FIST pauses. "
        "Press Escape or q to quit."
    )
    try:
        scene.run(update)
    except SystemExit:
        # Panda3D's window-close/escape path invokes sys.exit() internally.
        pass
    finally:
        client.close()
        elapsed = max(time.monotonic() - last_frame_stats["start"], 1e-6)
        inference = last_frame_stats["inference_ms"]
        avg_inference = (
            sum(inference) / len(inference) if inference else 0.0
        )
        print(
            "SpatialX render demo summary: "
            f"states={last_frame_stats['states']} "
            f"frames_with_hand={last_frame_stats['frames']} "
            f"elapsed_s={elapsed:.1f} "
            f"avg_inference_ms={avg_inference:.1f} "
            f"final_state={controller.state_machine.state.name} "
            f"final_cube_camera="
            f"{np.round(controller.primary_object.position, 1).tolist() if controller.primary_object else None}"
        )
        scene.close()

    if latest_error is not None:
        print(f"SpatialX tracking error: {latest_error.get('message')}")
        return 2 if latest_error.get("kind") in ("camera", "worker") else 4
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


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
