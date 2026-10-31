# Project Status — SpatialX

## Current Milestone

**Milestone 2 — Spatial Coordinate Engine: COMPLETE (software)**

**Milestone 1 — Real-Time Hand Tracking: COMPLETE**

**Milestone 0 — Environment and Project Foundation: COMPLETE**

## Completed Tasks — Milestone 2

- [x] Camera calibration model (`CameraCalibration`, camera matrix).
- [x] Normalized landmark → camera space conversion.
- [x] Hand-relative landmark vectors.
- [x] Relative hand depth computation.
- [x] Palm orientation estimation (rotation vector).
- [x] Hand velocity computation.

## Completed Tasks — Milestone 1

- [x] MediaPipe Tasks `HandLandmarker` integrated (`spatialx.vision.hand_landmarker`).
- [x] Live detection verified against real camera frames (hand detected with high confidence).
- [x] Two-hand support structure in place (detection limit configurable).
- [x] 21 landmarks per hand extraction into SpatialX data model.
- [x] Left / right hand identification.
- [x] Skeleton visualization implemented and unit-tested.
- [x] Tracking-loss tolerance (incomplete hands skipped).
- [x] CLI `--detect` mode with FPS / inference reporting.
- [x] Vision error type added (`spatialx.core.exceptions.VisionError`).
- [x] Model asset in `assets/models/hand_landmarker.task` (offline-first).

## Pending Tasks

- Milestone 3 — Virtual 3D cursor (One Euro filtering, cursor jitter).
- Milestone 4 — First object manipulation.
- Milestones 5–11 — incremental product expansion.

## Known Bugs

- **Camera device currently unavailable (environment-level).** The HP TrueVision
  HD Camera (`/dev/video0`, USB 1-4) is presently in a suspended USB power state;
  frames can no longer be opened. The application handles this gracefully
  (`SpatialX error: Camera device 0 could not be opened.`, exit code 2). Resuming
  the device requires host-level access (USB power control); may resolve on its own.

## Test Results

- `python3 -m pytest -q`: **22 passed** (5 foundation + 7 hand-vision + 10 spatial).

## Performance Measurements — Milestones 1/2

- Hand landmarker inference (MediaPipe TFLite XNNPACK CPU): mean **36.4 ms/frame**,
  min **30.4 ms**, max **57.2 ms** (10-frame sample on this hardware).
- Spatial math (NumPy): single-vector operations in microseconds.

## Next Actions

- Begin **Milestone 3 — Virtual 3D Cursor** implementation plan and work.
