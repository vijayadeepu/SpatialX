"""Milestone 0 — project foundation tests."""

from __future__ import annotations

import pytest

import spatialx
from spatialx.camera import CameraManager, Frame
from spatialx.core.exceptions import CameraError, SpatialXError


def test_package_version_is_exposed() -> None:
    assert isinstance(spatialx.__version__, str)
    assert spatialx.__version__ != ""


def test_core_exceptions_hierarchy() -> None:
    assert issubclass(CameraError, SpatialXError)
    assert issubclass(SpatialXError, Exception)


def test_camera_module_exports() -> None:
    assert hasattr(CameraManager, "open")
    assert hasattr(CameraManager, "read")
    assert hasattr(CameraManager, "close")
    assert hasattr(CameraManager, "available_indices")
    assert hasattr(Frame, "resolution")


def test_camera_manager_rejects_read_before_open() -> None:
    manager = CameraManager(device_index=0)
    with pytest.raises(CameraError):
        manager.read()  # type: ignore[call-arg]


def test_cli_parser_defaults() -> None:
    from spatialx.application.cli import build_parser

    args = build_parser().parse_args([])
    assert args.camera_index == 0
    assert args.frames == 15
    assert args.preview is False
