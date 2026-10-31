# Project Status — SpatialX

## Current Milestone

**Milestone 1 — Real-Time Hand Tracking: COMPLETE (software)**

**Milestone 0 — Environment and Project Foundation: COMPLETE**

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

- Milestone 2 — Spatial coordinate engine.
- Milestone 3 — Virtual 3D cursor.
- Milestones 4–11 — incremental product expansion.

## Known Bugs

- **Camera device currently unavailable (environment-level).** The HP TrueVision
  HD Camera (`/dev/video0`, USB 1-4) is presently in a suspended USB power state;
  frames can no longer be opened. The application handles this gracefully
  (`SpatialX error: Camera device 0 could not be opened.`, exit code 2). Resuming
  the device requires host-level access (USB power control); may resolve on its own.

## Test Results

- `python3 -m pytest -q`: **12 passed** (5 foundation + 7 hand-vision tests).

## Performance Measurements — Milestone 1

- Hand landmarker inference (MediaPipe TFLite XNNPACK CPU): mean **36.4 ms/frame**,
  min **30.4 ms**, max **57.2 ms** (10-frame sample on this hardware).
- Theoretical inference throughput: ~27 FPS; target interaction latency < 100 ms
  (capture + inference + processing).
- Live camera hand detection previously succeeded (hand present, confidence 0.92).

## Next Actions

- Begin **Milestone 2 — Spatial Coordinate Engine** implementation plan and work.
