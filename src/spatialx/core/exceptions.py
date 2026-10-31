"""Cross-cutting error types and core primitives for SpatialX."""

from __future__ import annotations


class SpatialXError(Exception):
    """Base class for all SpatialX exceptions."""


class CameraError(SpatialXError):
    """Raised when camera discovery, opening or frame capture fails."""


class ConfigurationError(SpatialXError):
    """Raised when configuration files are missing or invalid."""
