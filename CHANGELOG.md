# Changelog — SpatialX

## Unreleased — interaction prototype session (v0.6.0)

This entry covers the current engineering session. It does **not** claim
live hand-in-view verification; the available webcam produced a black
frame. See PROJECT_STATUS.md.

### Added

- Full palm orientation: `palm_frame_quaternion()` builds a right-handed
  camera-space quaternion from wrist / index MCP / middle MCP. Returns
  `None` on degenerate geometry. The tracking worker now publishes this
  quaternion instead of the single-axis `palm_orientation` rotation
  vector.
- Quaternion utilities: `quaternion_normalized`,
  `quaternion_to_rotation_matrix`, `rotation_matrix_to_quaternion`.
  `relative_rotation` now returns a unit quaternion.
- `GestureStabilizer`: N-frame gesture debounce (default 2) to suppress
  accidental grabs/releases.
- `GrabController.process_frame()`: live hover / grab / translate /
  rotate / release pipeline, including grab-offset preservation, bounds
  clamping, tracking-loss freeze with grace period, resume-without-snap,
  and slerp rotation smoothing.
- `InteractionStateMachine.set_hovering()` and sticky HOVERING on open
  palm. FIST while already GRABBING no longer drops the object.
- `SpatialScene`: cursor visibility, cursor colour, status text,
  `request_exit()`, `q` / Escape / window-close bindings, window title.
- `TrackingClient.drain_messages()` (preserve intermediate states for
  debounce) and synthetic `kind:"worker"` errors when the child dies.
- Worker protocol version field `"v": 1` on every message.
- CLI: `--render-demo` now performs real grab / translate / rotate;
  `--hover-radius`, `--tracking-fps`; `if __name__ == "__main__"` guard
  so `python -m spatialx.application.cli` actually runs.
- Tests: interaction runtime, palm frame, quaternion math, scene-graph
  manipulation, MediaPipe-free rendering import, worker protocol,
  missing-camera error path. Suite grew from 85 to 123 passing tests.

### Changed

- `spatialx.gestures` no longer imports `spatialx.vision` at runtime
  (TYPE_CHECKING only). This keeps MediaPipe out of the Panda3D process.
- Interactive cursor filter in `--render-demo` uses responsive One Euro
  settings (`min_cutoff=1.2`, `beta=0.025`, `derivative_cutoff=1.0`)
  instead of the library defaults, which are too heavy for live use.
- Cube auto-spin is disabled in the interactive demo so hand rotation
  and release-pose preservation are observable.
- Project version 0.5.0 → 0.6.0.

### Verified (this machine, no hand in view)

- Worker IPC v1, startup, `q` shutdown, no orphan processes, camera
  released.
- Onscreen GLX window 800×600; render loop ~66 FPS; scene contents
  (cube, grid, axes, status text, hidden cursor) confirmed by window
  dump.
- Synthetic scene-graph grab → translate → rotate → release.
- 123 tests passing; `compileall` clean.
- 5-minute worker probe: 2950 states, 9.83 Hz, inference 57.1 ms mean,
  66.6 % CPU, ~699 MB RSS, clean shutdown. Hands reported on 855 frames
  (transient; no scripted grab session).

### Not verified

- Live hand tracking session, cursor following, PINCH/OPEN_PALM/FIST on
  continuously visible landmarks, and 20+ webcam grab/rotate cycles.
  (Transient hand detections occurred during the 5-minute probe; no
  interactive manipulation was demonstrated.)

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
