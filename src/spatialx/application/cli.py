"""Application layer: CLI entry point and runtime orchestration."""

from __future__ import annotations

import argparse
import time
from typing import Sequence, Optional

import cv2

from spatialx.camera import CameraManager
from spatialx.core.exceptions import CameraError


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
            fps = len(frames) / elapsed if elapsed > 0 else 0.0
            print(
                f"SpatialX camera OK: index={manager.device_index} "
                f"frames_captured={len(frames)} resolution="
                f"{frames[0].width}x{frames[0].height} fps={fps:.2f}"
            )
    finally:
        manager.close()
    return 0
