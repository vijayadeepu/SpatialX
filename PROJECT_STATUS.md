# Project Status — SpatialX

## Current milestone / version

**v0.6.0 — interaction prototype.** Priorities 1–5 are implemented.
A scripted live hand acceptance (cursor follow, grab, translate, rotate,
20+ cycles) was **not** completed. The first camera frame was black; a
later 5-minute probe saw transient hand detections. See the matrix.

Last full-suite result: **123 passed** (`python3 -m pytest -q`).

## Priority completion matrix

| Priority | Status | Evidence | Remaining limitations |
|---|---|---|---|
| P1 Hand tracking | **PARTIAL** | Model resolution, BGR→RGB, worker inference and IPC verified (`tests/test_vision.py`, `tests/test_tracking_link.py`). 5-minute probe: MediaPipe reported hands on **855/2950** frames, so the detector does fire on this machine. No controlled hand-in-view session (landmarks, handedness, gesture names, 300/300) was captured. Historical 300/300 two-hand run (2026-10-31) is not reproduced. | Needs a kept-in-view hand to re-confirm landmarks, gestures and tracking continuity. |
| P2 Spatial math | **PASS** (code/tests) | 123-test suite includes mirroring, relative-depth semantics, palm frame, quaternion roundtrips. Module docstrings state Z is wrist-relative and **not** metric. | Live coordinate numbers from a real hand are NOT VERIFIED this session. |
| P3 Real-time tracking + rendering | **PARTIAL** | `spatialx --render-demo` opened an 800×600 GLX window, started the worker, published v1 JSON states (197 states / 25 s, 0 hands), hid the cursor, kept the cube at rest, and exited cleanly on `q` with no orphan processes and no camera lock. Render FPS ≈ 66 onscreen. **Cursor following a real hand is NOT VERIFIED.** | Needs a hand in view to confirm X/Y following, mirroring and bounds. |
| P4 Object translation | **PARTIAL** | `GrabController` + scene-graph integration test: hover → pinch grab, offset preserved (no jump), translate, release keeps last pose, bounds clamp, tracking-loss freeze. **Real-webcam grab/move/release is NOT VERIFIED.** Relative-Z translation is a documented limitation. | Hardware acceptance (20+ cycles) blocked by camera. |
| P5 Object rotation | **PARTIAL** | Full palm-frame quaternion, relative world-space delta, slerp smoothing, degenerate-input skip, release preserves orientation. Scene-graph test applies rotation to the actual cube node. **Real-webcam palm rotation is NOT VERIFIED.** | Hardware acceptance (20+ cycles, axis direction) blocked by camera. |
| P6 Integration / regression | **PARTIAL** | 123 tests pass; compileall clean; worker startup/IPC/teardown verified; 5-minute worker probe (see Performance). Hardware end-to-end acceptance scenario is incomplete. | Live grab/translate/rotate/release sequence not demonstrated. |
| P7 Docs / release | **PARTIAL** | README, ARCHITECTURE, CHANGELOG and this file updated to match the implementation. Version bumped to 0.6.0. **No LICENSE file; license decision required.** No git commit made automatically. | License, tagged release, and hardware re-verification still open. |

Status labels: **PASS** = implemented and verified against acceptance
criteria · **PARTIAL** = some functionality works, acceptance incomplete ·
**FAIL** = broken · **NOT VERIFIED** = no evidence.

## What was verified on this machine

- `/dev/video0` opens (640×480). The first grabbed frame was nearly
  black. During the 5-minute probe MediaPipe reported hands on 855
  frames, so the sensor/detector path is not dead.
- Worker process starts, loads the model, publishes `{"v":1,"type":"state",...}`.
  With no hand: `hands=0`, `gesture=NONE`, `cursor_camera=null`.
- `--render-demo` window titled `SpatialX` (800×600) appeared in
  `wmctrl` / `xwininfo`.
- Sending `q` to the window stopped the app, reaped the worker, released
  `/dev/video0`, and removed the window.
- Cube remained at its initial camera-space pose `[0, -100, 0]` when no
  hand was present (`LOST_TRACKING`).
- Cursor was hidden while tracking was lost.
- Synthetic scene-graph test drives the real Panda3D nodes through
  grab → translate → rotate → release (see
  `tests/test_runtime_integration.py`).
- Offscreen screenshot (`--render-screenshot`) writes a PNG via the EGL
  pipe (TinyDisplay is not installed; Panda3D falls back).

## What could not be verified

- A scripted live session: cursor following, PINCH grab, translation,
  palm rotation, release, 20+ cycles.
- Gesture stability of PINCH / OPEN_PALM / FIST on continuously visible
  landmarks.
- Tracking-to-render latency with a moving hand.

The 5-minute probe observed `hands > 0` on 855/2950 frames (transient),
so the sensor can see a hand, but no interactive manipulation was
demonstrated or recorded.

## Performance (measured this session)

Numbers below are measured, not estimated.

### 5-minute tracking-worker probe (`/tmp/opencode/sx/perf.json`)

Duration 300 s, `target_fps=20`, camera 0. No interactive grab session.

| Metric | Value |
|---|---|
| Startup to first state | **4.61 s** (model + camera open) |
| Worker states | 2950 |
| Worker publish rate | **9.83 Hz** (camera ~10 FPS, interval 100.2 ms) |
| Frames with `hands > 0` | 855 / 2950 (transient presence; no grab session) |
| Inference mean / median | **57.1 ms / 53.6 ms** |
| Inference p95 / min / max | 98.7 / 28.0 / 162.3 ms |
| Worker CPU | **66.6 %** of one core |
| Worker RSS mean / max | **699 MB / 700 MB** |
| Errors | none |
| Clean shutdown | yes |

The 855 hand frames show the camera was not black for the entire five
minutes, but no person performed a scripted grab/translate/rotate session
and nothing was recorded. Live manipulation remains NOT VERIFIED.

### Rendering (separate process, no camera)

| Metric | Value | Method |
|---|---|---|
| Scene startup | **0.29 s** | `SpatialScene(onscreen)` to ready |
| Onscreen render FPS | **66.6 FPS** | 300 `taskMgr.step()` frames, GLX 800×600 |
| Offscreen screenshot | works | EGL pipe (TinyDisplay `.so` is not installed; Panda3D falls back) |
| Tracking→render latency | NOT VERIFIED | needs a moving hand |
| Historic two-hand inference | 69.0 ms mean (2026-10-31) | not reproduced this session |

## Environment

- Linux, X11 (`DISPLAY=:0`), XFCE.
- Python 3.12, OpenCV 4.11.0, MediaPipe (CPU/XNNPACK), Panda3D 1.10.16.
- NumPy <2.0, SciPy.
- PySide6 is **not** installed and is not required by the current app.
- Model: `assets/models/hand_landmarker.task` (7,819,105 bytes).
- Camera: `/dev/video0` + `/dev/video1` (UVC). First frame was black;
  the 5-minute probe still saw transient hand detections.

## Architecture invariants (do not break)

- MediaPipe / OpenCV stay in `spatialx.application.tracking_worker`.
- Panda3D stays in the parent process (`spatialx.rendering`).
- IPC is newline-delimited JSON on the worker’s stdout; stdout is
  protocol-only (logs go to stderr).
- `spatialx.gestures` must **not** import `spatialx.vision` at runtime
  (TYPE_CHECKING only), or the rendering process loads MediaPipe.
- Rendering never imports vision internals.

## Next actions

1. Re-run live hand verification once a visible camera is available
   (P3 cursor follow, P4 grab/translate, P5 rotate, 20+ cycles each).
2. Decide the license and add a `LICENSE` file.
3. Optional: expose a short `--selftest` that reports camera black-frame
   / hand-presence so this blocker is obvious to the next operator.
4. Do **not** start Milestone 6 (two-hand interaction).

## Test inventory (this session)

- Baseline at session start: **85 passed**.
- Added: interaction runtime, palm frame, quaternion normalize/matrix,
  scene-graph manipulation, MediaPipe-free import, worker protocol,
  missing-camera error path.
- Final: **123 passed**, 0 failed, 0 skipped.
