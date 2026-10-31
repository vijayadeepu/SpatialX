"""Application layer: CLI entry point and runtime orchestration."""

from __future__ import annotations

import argparse
import time
from typing import Sequence, Optional

import cv2

from spatialx.camera import CameraManager
from spatialx.core.exceptions import CameraError, VisionError
from spatialx.vision import DEFAULT_MODEL_PATH, HandLandmarker


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
        default=DEFAULT_MODEL_PATH,
        help="Path to the MediaPipe hand landmarker model file.",
    )
    parser.add_argument(
        "--num-hands",
        type=int,
        default=2,
        help="Number of hands to track (default: 2).",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show the live camera feed in a window (press 'q' to quit).",
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
    model_path: str,
    num_hands: int,
) -> int:
    """Run the MediaPipe hand landmarker over captured frames.

    Returns:
        Process exit code: 0 on success, 4 on inference failure.
    """
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
        f"SpatialX vision OK: model={model_path} frames_captured={len(frames)} "
        f"frames_with_hands={frames_with_hands} total_hands={total_hands} "
        f"avg_inference_ms={avg_inference:.2f} fps={fps:.2f}"
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the SpatialX CLI. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

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
            fps = len(frames) / elapsed if elapsed > 0 else 0.0
            print(
                f"SpatialX camera OK: index={manager.device_index} "
                f"frames_captured={len(frames)} resolution="
                f"{frames[0].width}x{frames[0].height} fps={fps:.2f}"
            )
    finally:
        manager.close()
    return 0
