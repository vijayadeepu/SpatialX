# Changelog — SpatialX

## 0.2.0 — 2026-10-31

### Added

- Milestone 2 — Spatial Coordinate Engine:
  - `spatialx.spatial.coordinates`: camera calibration, normalized landmark →
    camera space conversion, hand-relative vectors, relative depth, palm
    orientation, hand velocity.
- Milestone 1 — Real-Time Hand Tracking:
  - MediaPipe Tasks `HandLandmarker` integration (`spatialx.vision.hand_landmarker`).
  - Two-hand support, 21 landmarks per hand extraction.
  - Left/right hand identification and `TrackedHand` / `HandDetection` data model.
  - Hand skeleton visualization (`spatialx.vision.visualization.draw_hand_skeleton`).
  - Tracking-loss tolerance via incomplete-hand skipping.
- CLI `--detect` mode for headless hand-detection runs with FPS and inference reporting.

### Changed

- Project version bumped from 0.1.0 to 0.2.0.

### Fixed

- None.

## 0.1.0 — 2026-10-31

### Added

- Milestone 0 foundation: project skeleton, packaging, CLI camera capture.
- Camera manager with frame capture and FPS reporting.
- Core exception hierarchy.
- Initial tests, documentation and configuration.

### Fixed

- None.
