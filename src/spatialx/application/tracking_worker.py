"""Isolated hand-tracking worker process for SpatialX.

Runs camera capture and MediaPipe hand tracking out-of-process (MediaPipe
and Panda3D cannot share a process — see ``tracking_link``). Speaks a
newline-delimited JSON protocol on stdout:

- emits one ``{"type": "state", ...}`` per processed frame;
- emits ``{"type": "error", "kind": "camera"|"vision", ...}`` and exits
  nonzero when startup fails;
- exits when it reads ``quit`` on stdin (or stdin closes).

All MediaPipe/TFLite log noise stays on stderr; stdout is protocol-only.

Usage: ``python3 -m spatialx.application.tracking_worker [--mirror] ...``
"""

from __future__ import annotations

import argparse
import json
import select
import sys
import time
from typing import Optional, Sequence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spatialx-tracking-worker",
        description="SpatialX isolated hand tracking worker.",
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--num-hands", type=int, default=2)
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Mirror horizontal coordinates (selfie view).",
    )
    parser.add_argument("--target-fps", type=float, default=30.0)
    return parser


def _stdin_quit_requested() -> bool:
    """Return True when a quit line (or EOF) is pending on stdin."""
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if not ready:
        return False
    line = sys.stdin.readline()
    return (not line) or line.strip() == "quit"


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Worker entry point. Returns a process exit code."""
    args = _build_parser().parse_args(argv)

    def emit(payload) -> None:
        payload = {"v": 1, **payload}
        sys.stdout.write(json.dumps(payload) + "\n")
        sys.stdout.flush()

    # Vision imports happen here so this module stays importable without
    # MediaPipe (help/wire-up paths stay light).
    from spatialx.camera import CameraManager
    from spatialx.core.exceptions import CameraError, VisionError
    from spatialx.gestures import detect_gesture
    from spatialx.spatial.coordinates import (
        CameraCalibration,
        landmark_camera_space,
        palm_frame_quaternion,
    )
    from spatialx.vision import HandLandmarker, resolve_model_path

    try:
        model_path = resolve_model_path(args.model_path)
    except VisionError as error:
        emit(
            {
                "type": "error",
                "kind": "vision",
                "message": str(error),
            }
        )
        return 4

    manager = CameraManager(device_index=args.camera_index)
    try:
        manager.open()
    except CameraError as error:
        emit(
            {
                "type": "error",
                "kind": "camera",
                "message": str(error),
            }
        )
        return 2
    if not manager.is_open():
        emit(
            {
                "type": "error",
                "kind": "camera",
                "message": f"camera {args.camera_index} could not be opened",
            }
        )
        return 2

    try:
        landmarker = HandLandmarker(
            model_path=model_path, num_hands=args.num_hands
        )
    except Exception as error:  # noqa: BLE001
        emit(
            {
                "type": "error",
                "kind": "vision",
                "message": str(error),
            }
        )
        manager.close()
        return 4

    period = 1.0 / args.target_fps if args.target_fps > 0 else 0.0
    try:
        while True:
            if _stdin_quit_requested():
                break
            try:
                frame = manager.read()
            except CameraError as error:
                emit(
                    {
                        "type": "error",
                        "kind": "camera",
                        "message": str(error),
                    }
                )
                break
            if frame is None:
                time.sleep(0.01)
                continue
            started = time.monotonic()
            try:
                detection = landmarker.detect(
                    frame.image,
                    timestamp_ms=int(round(frame.timestamp * 1000)),
                )
            except VisionError as error:
                emit(
                    {
                        "type": "error",
                        "kind": "vision",
                        "message": str(error),
                    }
                )
                break
            state = {
                "type": "state",
                "timestamp": frame.timestamp,
                "hands": len(detection.hands) if detection else 0,
                "handedness": None,
                "cursor_camera": None,
                "orientation": None,
                "gesture": "NONE",
                "inference_ms": (
                    detection.inference_time_ms if detection else None
                ),
            }
            if detection and detection.hands:
                hand = detection.hands[0]
                calibration = CameraCalibration.for_image(
                    frame.width, frame.height
                )
                wrist = landmark_camera_space(
                    hand.landmarks[0],
                    calibration,
                    image_width=frame.width,
                    image_height=frame.height,
                    mirror=args.mirror,
                )
                palm_quat = palm_frame_quaternion(
                    hand,
                    image_width=frame.width,
                    image_height=frame.height,
                    mirror=args.mirror,
                )
                gesture = detect_gesture(list(detection.hands))
                state.update(
                    {
                        "handedness": hand.handedness,
                        "cursor_camera": [float(v) for v in wrist],
                        "orientation": (
                            None
                            if palm_quat is None
                            else [
                                float(palm_quat.w),
                                float(palm_quat.x),
                                float(palm_quat.y),
                                float(palm_quat.z),
                            ]
                        ),
                        "gesture": gesture.name,
                    }
                )
            emit(state)
            elapsed = time.monotonic() - started
            if period > 0.0 and elapsed < period:
                time.sleep(period - elapsed)
    finally:
        landmarker.close()
        manager.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
