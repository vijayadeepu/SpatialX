# Changelog — SpatialX

## Unreleased — audit repair pass (M0–M5)

### Fixed

- Model asset resolution: `DEFAULT_MODEL_PATH` pointed at a nonexistent
  `<repo>/src/assets/...` path. Added `resolve_model_path()` with verified
  existence checks, `SPATIALX_MODEL_PATH` override, and a clear `VisionError`
  (searched locations + download URL). `HandLandmarker()` auto-resolves.
- BGR/RGB channel order in `HandLandmarker.detect()`: frames are now converted
  with `cv2.COLOR_BGR2RGB` before MediaPipe inference (BGR input silently
  produced zero hand detections on live frames).
- CLI validates the model before opening the camera and reports the resolved
  model path; `--model-path` defaults to auto-resolution.
- Spatial camera space: `landmark_camera_space` returned unit-hybrid values
  `(x·W−cx)·fx/W` with X/Y lacking depth. Corrected to camera pixel space
  `[u−cx, v−cy, z·W]` (principal-point origin; wrist-relative, width-scaled,
  **non-metric** Z).
- Spatial geometry scale mixing: normalized x (per-width) and y (per-height)
  distorted distances and cross products. Hand-relative vectors and palm
  orientation now compute in isotropic pixel space.
- `palm_orientation` returned an unnormalized axis times angle; now unit
  axis × angle (correct rotation vector for quaternion conversion).
- `relative_depth` duplicated `hand_relative_vectors`; now returns 1-D
  wrist-relative width-scaled depth offsets.

### Added

- Camera mirroring (`mirror=True`) for selfie-view pipelines in
  `normalized_to_pixel`, `landmark_camera_space`, `hand_relative_vectors`,
  `palm_orientation`; CLI cursor demo uses mirrored coordinates.
- `camera_ray()` unit pinhole ray directions; `normalized_to_pixel()`.
- `CameraCalibration.for_image()` nominal intrinsics from image size and
  assumed HFOV (documented as assumption); typed optional extrinsics with
  `effective_rotation` / `effective_translation` accessors.

## 0.5.0 — 2026-10-31

### Added

- Milestone 5 — Three-Dimensional Rotation:
  - `spatialx.spatial.quaternion`: quaternion representation, axis-angle
    conversion, Hamilton product, conjugate, slerp, vector rotation, slerp-based
    rotation smoothing.

### Changed

- Project version bumped from 0.4.0 to 0.5.0.

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
