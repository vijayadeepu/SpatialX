"""Process-isolated hand tracking for SpatialX.

Why a subprocess: the camera/vision stack (OpenCV's cv2 module and
MediaPipe's bundled LLVM runtime) corrupts Panda3D's graphics state when
imported before Panda3D in one process (observed shader-pass corruption
and segfaults on this hardware). The architecture therefore keeps
tracking (camera, OpenCV, MediaPipe) in a worker process and rendering
(Panda3D) in the main process, exchanging compact newline-delimited JSON
messages over the worker's stdin/stdout pipes.

A plain :mod:`subprocess` worker (``python3 -m
spatialx.application.tracking_worker``) is used rather than
:mod:`multiprocessing`: the spawn start method re-imports the parent's
``__main__`` module, which breaks console scripts and pytest probes.

The worker owns all camera/vision imports; this module is safe to import
from a pure-rendering process.
"""

from __future__ import annotations

import json
import select
import subprocess
import sys
import time
from typing import Optional

TRACKING_MESSAGE_STATE = "state"
TRACKING_MESSAGE_ERROR = "error"
TRACKING_MESSAGE_QUIT = "quit"


class TrackingClient:
    """Client-side handle to an isolated tracking worker process.

    The parent process stays MediaPipe-free; only small JSON state
    dictionaries cross the pipe.

    Args:
        camera_index: Camera device index inside the worker.
        model_path: Optional model path forwarded to the worker.
        num_hands: MediaPipe hand count.
        mirror: Mirror horizontal coordinates (selfie view).
        target_fps: Maximum state publish rate.
    """

    def __init__(
        self,
        camera_index: int = 0,
        model_path: Optional[str] = None,
        num_hands: int = 2,
        mirror: bool = True,
        target_fps: float = 30.0,
    ) -> None:
        command = [
            sys.executable,
            "-m",
            "spatialx.application.tracking_worker",
            "--camera-index",
            str(camera_index),
            "--num-hands",
            str(num_hands),
            "--target-fps",
            str(target_fps),
        ]
        if model_path is not None:
            command += ["--model-path", str(model_path)]
        if mirror:
            command.append("--mirror")
        # stdout: JSON protocol; stderr: MediaPipe/Panda log noise (dropped).
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,  # line-buffered
        )
        self._closed = False

    @property
    def running(self) -> bool:
        """Whether the worker process is alive and the client is open."""
        return self._process.poll() is None and not self._closed

    def latest_state(self, timeout: float = 0.0) -> Optional[dict]:
        """Return the freshest tracking state, or ``None`` when idle.

        Drains queued messages so callers always act on the newest data
        (the render loop must not fall behind the camera).
        """
        if self._closed or self._process.stdout is None:
            return None
        newest: Optional[dict] = None
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            remaining = deadline - time.monotonic()
            wait = max(0.0, remaining)
            line = self._read_line(wait if newest is None else 0.0)
            if line is None:
                break
            try:
                newest = json.loads(line)
            except ValueError:
                continue  # ignore any non-protocol line
            if newest.get("type") == TRACKING_MESSAGE_ERROR:
                break
            if remaining <= 0.0:
                break
        return newest

    def _read_line(self, timeout: float) -> Optional[str]:
        """Read one stdout line with a timeout; ``None`` on timeout/EOF."""
        stdout = self._process.stdout
        if stdout is None:
            return None
        ready, _, _ = select.select([stdout], [], [], max(0.0, timeout))
        if not ready:
            return None
        line = stdout.readline()
        return line if line else None

    def close(self) -> None:
        """Ask the worker to stop and reap the process."""
        if self._closed:
            return
        self._closed = True
        try:
            if (
                self._process.poll() is None
                and self._process.stdin is not None
            ):
                self._process.stdin.write(TRACKING_MESSAGE_QUIT + "\n")
                self._process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            pass
        try:
            self._process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            self._process.kill()
            try:
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                pass
        for stream in (
            self._process.stdin,
            self._process.stdout,
            self._process.stderr,
        ):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass

    def __enter__(self) -> "TrackingClient":
        return self

    def __exit__(self, *args) -> None:
        self.close()
