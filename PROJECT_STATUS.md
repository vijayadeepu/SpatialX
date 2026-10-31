# Project Status — SpatialX

## Current Milestone

**Milestone 4 — First Object Manipulation: COMPLETE (software)**

**Milestone 3 — Virtual 3D Cursor: COMPLETE (software)**

**Milestone 2 — Spatial Coordinate Engine: COMPLETE (software)**

**Milestone 1 — Real-Time Hand Tracking: COMPLETE**

**Milestone 0 — Environment and Project Foundation: COMPLETE**

## Completed Tasks — Milestone 4

- [x] Deterministic geometric gesture detection (pinch, open palm, fist).
- [x] Interaction state machine with explicit states and transitions.
- [x] Grab controller with hand-to-object offset and translation.

## Completed Tasks — Milestone 3

- [x] One Euro filter with configurable min cutoff, beta, derivative cutoff.
- [x] Virtual cursor mapping hand origins to filtered positions.
- [x] Cursor sensitivity scaling and boundary clamping.
- [x] Cursor jitter estimation.

## Completed Tasks — Milestone 2

- [x] Camera calibration model (`CameraCalibration`, camera matrix).
- [x] Normalized landmark → camera space conversion.
- [x] Hand-relative landmark vectors, relative depth, palm orientation, hand velocity.

## Completed Tasks — Milestone 1

- [x] MediaPipe Tasks `HandLandmarker` integration.
- [x] Landmark extraction, skeleton visualization, CLI `--detect`.

## Pending Tasks

- Milestone 5 — Three-Dimensional Rotation (quaternion-based wrist rotation).
- Milestone 6 — Two-Hand Spatial Interaction.
- Milestone 7 — Virtual Spatial Interface.
- Milestones 8–11 — incremental product expansion.

## Known Bugs

- **Camera device currently unavailable (environment-level).** The HP TrueVision
  HD Camera (`/dev/video0`, USB 1-4) is presently in a suspended USB power state;
  frames can no longer be opened. The application handles this gracefully
  (`SpatialX error: Camera device 0 could not be opened.`, exit code 2). Resuming
  the device requires host-level access (USB power control); may resolve on its own.

## Test Results

- `python3 -m pytest -q`: **48 passed**.
- `python3 -m compileall src main.py tests`: OK.

## Performance Measurements

- Hand landmarker inference (MediaPipe TFLite XNNPACK CPU): mean **36.4 ms/frame**,
  min **30.4 ms**, max **57.2 ms**.

## Next Actions

- Begin **Milestone 5 — Three-Dimensional Rotation** implementation plan and work.
