"""Camera engine: discovery, frame capture, lifecycle and FPS monitoring."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from spatialx.core.exceptions import CameraError

DEFAULT_DEVICE_INDEX = 0
DEFAULT_FPS = 30.0
_PROBE_LIMIT = 5


@dataclass(frozen=True)
class Frame:
    """A single captured camera frame with capture metadata."""

    image: np.ndarray
    width: int
    height: int
    timestamp: float  # monotonic seconds

    @property
    def resolution(self) -> tuple[int, int]:
        """Return the frame resolution as ``(width, height)``."""
        return (self.width, self.height)


class CameraManager:
    """Lifecycle manager for a single camera device.

    Responsibilities:
        - Open / close the camera.
        - Capture frames with monotonic timestamps.
        - Report frame resolution and frame rate.
        - Expose camera discovery for a configurable set of indices.

    This class intentionally owns only capture concerns; inference,
    spatial mathematics and rendering are separate subsystems.
    """

    def __init__(
        self,
        device_index: int = DEFAULT_DEVICE_INDEX,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: float = DEFAULT_FPS,
    ) -> None:
        self.device_index = device_index
        self.requested_width = width
        self.requested_height = height
        self.requested_fps = fps
        self._capture: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        """Open the camera device. Returns ``True`` on success.

        Raises:
            CameraError: If the device cannot be opened.
        """
        capture = cv2.VideoCapture(self.device_index)
        if not capture.isOpened():
            capture.release()
            raise CameraError(
                f"Camera device {self.device_index} could not be opened."
            )
        if self.requested_width is not None:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.requested_width))
        if self.requested_height is not None:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.requested_height))
        self._capture = capture
        return True

    def read(self) -> Optional[Frame]:
        """Capture a single frame.

        Returns:
            A :class:`Frame` on success, or ``None`` on capture failure
            (e.g. temporary tracking loss of the camera stream).

        Raises:
            CameraError: If the camera is not open.
        """
        if self._capture is None:
            raise CameraError("CameraManager.read() called before open().")
        ok, image = self._capture.read()
        if not ok or image is None:
            return None
        height, width = image.shape[:2]
        return Frame(
            image=image,
            width=int(width),
            height=int(height),
            timestamp=time.monotonic(),
        )

    def close(self) -> None:
        """Release the camera resource."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def is_open(self) -> bool:
        """Return ``True`` if the camera handle is currently open."""
        return self._capture is not None

    @staticmethod
    def available_indices() -> list[int]:
        """Probe a bounded range of camera indices and return open ones."""
        available: list[int] = []
        for index in range(_PROBE_LIMIT):
            probe = cv2.VideoCapture(index)
            if probe.isOpened():
                available.append(index)
            probe.release()
        return available
