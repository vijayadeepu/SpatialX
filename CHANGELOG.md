# Changelog — SpatialX

## 0.4.0 — 2026-10-31

### Added

- Milestone 4 — First Object Manipulation:
  - Deterministic geometric gesture detection (`spatialx.gestures`).
  - Interaction state machine with explicit states and transitions.
  - Grab controller with hand-to-object offset and translation.

### Changed

- Project version bumped from 0.3.0 to 0.4.0.

## 0.3.0 — 2026-10-31

### Added

- Milestone 3 — Virtual 3D Cursor:
  - `spatialx.motion.OneEuroFilter` with configurable min cutoff, beta and
    derivative cutoff.
  - `spatialx.rendering.VirtualCursor` mapping hand origins to filtered
    cursor positions, with sensitivity scaling and boundary clamping.
  - `spatialx.rendering.estimate_jitter` cursor jitter metric.
  - CLI `--cursor-demo` and `--cursor-sensitivity` flags.

### Changed

- Project version bumped from 0.2.0 to 0.3.0.

## 0.2.0 — 2026-10-31

### Added

- Milestone 2 — Spatial Coordinate Engine:
  - `spatialx.spatial.coordinates`: camera calibration, normalized landmark →
    camera space conversion, hand-relative vectors, relative depth, palm
    orientation, hand velocity.
- Milestone 1 — Real-Time Hand Tracking:
  - MediaPipe Tasks `HandLandmarker` integration.
  - Landmark extraction, skeleton visualization, CLI `--detect`.

### Changed

- Project version bumped from 0.1.0 to 0.2.0.

## 0.1.0 — 2026-10-31

### Added

- Milestone 0 foundation: project skeleton, packaging, CLI camera capture.

### Fixed

- None.
