# Architecture — SpatialX

## Overview

SpatialX is an offline-first, webcam-based spatial interaction engine.
It is organised as independent feature modules under `src/spatialx/`.
The two hard boundaries are:

1. **Vision never renders.** Camera + MediaPipe live in a worker process.
2. **Rendering never tracks.** Panda3D lives in the main process and only
   consumes plain numpy vectors and quaternions.

On this hardware, importing OpenCV or MediaPipe into a process that has
initialised Panda3D corrupts GL state (observed shader corruption and
segfaults). The split is therefore a correctness requirement, not a
performance nicety.

## Module map

| Module | Responsibility | Heavy deps |
|---|---|---|
| `camera` | Device discovery, frame capture, lifecycle, FPS. | `cv2` |
| `vision` | Hand detection, landmarks, skeleton overlay, model path. | `mediapipe`, `cv2` |
| `spatial` | Coordinates, calibration, palm frame, quaternions. | `numpy` |
| `gestures` | Deterministic geometric gestures. | none (vision is TYPE_CHECKING only) |
| `motion` | One Euro filter. | none |
| `interaction` | State machine, hover, grab, translate, rotate. | `numpy` |
| `rendering` | Panda3D scene, cursor, cube, camera↔scene mapping. | `panda3d` |
| `application` | CLI, tracking worker, tracking client, orchestration. | mixed |
| `integrations` | Reserved for external adapters. | — |
| `configuration` | JSON settings (unused by the live demo today). | — |
| `core` | Exceptions. | — |

## Process model

```
┌───────────────────────────── main process ─────────────────────────────┐
│ spatialx --render-demo                                                 │
│                                                                        │
│  SpatialScene (Panda3D / GLX onscreen)                                 │
│    ├── cube   (SceneObject)                                            │
│    ├── cursor (SceneObject, hidden when no hand)                       │
│    └── axes + grid + OnscreenText status                               │
│                                                                        │
│  GrabController          VirtualCursor (One Euro)                      │
│    state machine           filter + clamp                              │
│    hover / offset          filtered camera-space position              │
│    relative rotation                                                   │
│                                                                        │
│  TrackingClient (stdin/stdout JSON)                                    │
└───────────────▲────────────────────────────────────┬───────────────────┘
                │ state / error                      │ quit
┌───────────────┴──────────── tracking_worker ────────▼───────────────────┐
│ CameraManager → HandLandmarker → palm frame + gesture                  │
│ stdout: one JSON object per frame   stderr: MediaPipe noise (dropped)  │
└────────────────────────────────────────────────────────────────────────┘
```

The worker is a plain `subprocess.Popen` of
`python3 -m spatialx.application.tracking_worker`, **not**
`multiprocessing`. Spawn would re-import the parent `__main__`, which
breaks console scripts and pytest probes.

## IPC protocol (v1)

Newline-delimited JSON on the worker’s stdout. Every frame is flushed.
stdout carries **only** protocol lines.

State message (one per processed camera frame):

```json
{
  "v": 1,
  "type": "state",
  "timestamp": 12345.678,
  "hands": 1,
  "handedness": "RIGHT",
  "cursor_camera": [12.3, -45.6, 0.0],
  "orientation": [0.707, 0.0, 0.0, 0.707],
  "gesture": "PINCH",
  "inference_ms": 42.1
}
```

- `cursor_camera`: wrist in camera pixel space (`[x, y, z]`, origin at
  the principal point). `null` when no hand.
- `orientation`: palm-frame quaternion `(w, x, y, z)` in camera space.
  `null` when no hand or degenerate palm geometry.
- `gesture`: `PINCH | OPEN_PALM | FIST | NONE`.

Error message (then the worker exits non-zero):

```json
{"v": 1, "type": "error", "kind": "camera" | "vision", "message": "..."}
```

The client can also synthesise `kind: "worker"` when the child dies
without an error frame.

Shutdown: parent writes `quit\n` to stdin (or closes stdin). Worker
exits, releases camera and landmarker. Parent waits 3 s then kills.

`TrackingClient.drain_messages()` returns every pending message so
gesture debouncing runs at camera rate even when render is faster.
`latest_state()` keeps the newest-only contract for older callers.

## Coordinate system

Documented contract (see `spatialx.spatial.coordinates` docstrings):

| Space | Axes | Units | Origin |
|---|---|---|---|
| Image | `u` right, `v` down | pixels | top-left |
| Normalized (MediaPipe) | `x` / width, `y` / height, `z` ~ x-scale | unitless | top-left |
| Camera | `X` right, `Y` down, `Z` forward | pixels | principal point (assumed image centre) |
| Scene (Panda3D) | `X` right, `Y` forward, `Z` up | world | world origin |

Mapping: `scene = scale * (x, z, -y)` with `scale = 0.01`
(640 px ≈ 6.4 world units). This is a proper rotation
(−90° about X), so orientations map by the similarity transform
`q_scene = qA · q_camera · qA⁻¹`.

**Depth is not metric.** Camera-space Z is MediaPipe’s wrist-relative Z
scaled by image width. It is valid for relative comparisons only
(closer/farther, hand-relative offsets). No code path produces metres.
The virtual cube therefore translates reliably in the screen plane;
motion toward/away from the camera is a weak relative offset.

Mirroring (`mirror=True`) flips `u → width − u` for selfie view so the
cursor follows the user’s hand as seen in a mirror.

`palm_frame_quaternion()` builds a right-handed orthonormal palm frame
from wrist, index MCP and middle MCP:

- `y`: mean knuckle direction (palm “up”)
- `z`: palm normal (`wrist→index` × `wrist→middle`)
- `x`: completes the basis

It returns `None` on degenerate geometry; callers must skip that sample
rather than treat it as identity.

## Gesture system

`spatialx.gestures.geometric` is deterministic (no learned classifier):

- PINCH: thumb-tip / index-tip distance below threshold **and** index
  extended (so a fist is not a pinch).
- OPEN_PALM: all four fingers extended.
- FIST: all fingers and thumb folded.
- NONE: anything else.

Precedence: PINCH > OPEN_PALM > FIST > NONE.

`GestureStabilizer` (interaction layer) requires N consecutive identical
names (default 2) before the interaction state machine sees a change.
This suppresses single-frame flicker that would cause accidental grabs
or releases.

## Interaction state machine

States: `IDLE`, `TRACKING`, `HOVERING`, `SELECTED`, `GRABBING`,
`TRANSLATING`, `ROTATING`, `SCALING`, `RELEASING`, `LOST_TRACKING`,
`PAUSED`.

Live pipeline (`GrabController.process_frame`) uses:
`IDLE → HOVERING → GRABBING → (RELEASING) → IDLE`.

Rules:

- Hover when the filtered cursor is within `hover_radius` (default 160
  camera pixels) of the object centre.
- PINCH from HOVERING (or SELECTED) grabs. PINCH from IDLE does **not**
  grab (accidental-grab prevention).
- While holding, the object follows `cursor + grab_offset`, so it does
  not jump to the hand. Offset is captured at grab time.
- OPEN_PALM (debounced) releases. The object keeps its last valid
  position and orientation.
- FIST while holding keeps holding (a closed hand is not a release).
  FIST when idle pauses.
- Transient `NONE` while holding does not drop the object.
- Tracking loss while holding freezes the pose for `lost_grace_s`
  (default 1.2 s), then auto-releases into `LOST_TRACKING`.
- On resume after loss, the palm reference is re-captured so a
  reappearing hand does not snap the orientation back.

`SELECTED` remains the entry point for direct `handle_gesture` use
(unit tests, synthetic drivers) when no hover probe is available.

## Rotation pipeline

While grabbing:

1. At grab: `q_ref = palm_now`, `q_obj0 = current object orientation`.
2. Each frame: `q_delta = q_now · q_ref⁻¹` (world-space / extrinsic).
3. `q_target = q_delta · q_obj0`, normalised.
4. Display: `slerp(q_display, q_target, 0.35)`.
5. `SceneObject.set_orientation` maps camera → scene via the axis-map
   quaternion and applies it to the Panda3D node.

Quaternions are normalised after multiply/conversion. Degenerate palm
frames (`None`) are skipped. Orientation is **not** reset when the hand
reappears.

## Rendering

`SpatialScene` wraps ShowBase. Window types:

- `onscreen` — GLX, interactive demo.
- `offscreen` + `display_pipe="tinydisplay"` — screenshots / headless.
  GLX offscreen buffers are unreliable on this driver; do not use them.
- `none` — scene graph only (unit tests).

The scene never imports camera or vision modules. Colours communicate
interaction state:

| State | Cube | Cursor |
|---|---|---|
| IDLE / LOST | blue | yellow (hidden when lost) |
| HOVERING | green | cyan |
| GRABBING | orange | orange |
| PAUSED | grey | grey |

Escape and `q` call `request_exit()` (`ShowBase.userExit`). Window-close
is accepted as `window-close`.

## Data flow (interactive demo)

```
CameraManager.read()
  → HandLandmarker.detect (BGR→RGB)
  → TrackedHand[21]
  → landmark_camera_space(wrist, mirror=True)
  → palm_frame_quaternion(mirror=True)
  → detect_gesture
  → JSON state on stdout
  → TrackingClient.drain_messages
  → VirtualCursor.update (One Euro, clamp)
  → GrabController.process_frame
  → SpatialScene.update_cursor / cube.set_position_camera_space
  → cube.set_orientation
```

## Error handling and lifecycle

Startup order in `--render-demo`:

1. Create the Panda3D window (this process stays MediaPipe-free).
2. Create `TrackingClient` (spawns the worker).
3. Wait up to 15 s for the first message so camera/model errors surface
   before the interactive loop.
4. Enter `ShowBase.run()` with an update task that drains IPC.

Failure paths:

- Camera cannot open → `{"type":"error","kind":"camera"}` → exit code 2.
- Model missing / inference failure → `kind:"vision"` → exit code 4.
- Worker dies → client synthesises `kind:"worker"` → exit code 2.
- Escape / `q` / window close → exit code 0.

`finally` always closes the client (quit + wait + kill) and destroys the
scene. Camera release is verified by the worker’s `finally` block
(`landmarker.close()` then `manager.close()`).

Exit codes: 0 ok · 2 camera/worker · 3 no frames · 4 vision · 5 render.

## Testing strategy

- Pure math / interaction: in-process pytest.
- Panda3D: subprocess-isolated (`tests/test_rendering_scene.py`,
  `tests/test_runtime_integration.py`) so MediaPipe from other tests
  cannot be in the same interpreter.
- Worker IPC: real camera via `TrackingClient`
  (`tests/test_tracking_link.py`, `tests/test_runtime_integration.py`).
- Hardware hand-in-view acceptance is **not** replaced by mocks; it is
  recorded as NOT VERIFIED in PROJECT_STATUS.md when the camera cannot
  see a hand.

Do not delete tests to make the suite pass.

## Design decisions

- Plain `subprocess` over `multiprocessing` (see above).
- JSON over pickle / shared memory: debuggable, versionable, fast enough
  at ≤30 Hz for a few hundred bytes per frame.
- Deterministic gestures over a learned classifier: auditable, CPU-cheap,
  no extra model.
- Relative depth honesty over fake metric Z.
- One Euro filter for the cursor; slerp for rotation. Defaults in
  `OneEuroFilter` are conservative (offline jitter metrics). The live
  demo uses `min_cutoff=1.2, beta=0.025, derivative_cutoff=1.0`.
- No PySide6 in the critical path (not installed; not required).
