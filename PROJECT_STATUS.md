# Project Status — SpatialX

## Current Milestone

**Audit repair pass in progress (M0–M5 re-verification). Priority 1 COMPLETE.**

## Audit Priorities (user-directed repair pass)

- [x] **Priority 1 — Fix real hand tracking**: model asset path resolution fixed,
  verified, and regression-tested (details below).
- [x] **Priority 2 — Correct spatial mathematics**: conventions fixed, honest
  depth semantics, mirroring added, orientation math corrected (details below).
- [ ] Priority 3 — Implement real 3D rendering (Panda3D).
- [ ] Priority 4 — Complete object translation (rendered cube grab/move).
- [ ] Priority 5 — Complete object rotation (scene-wired orientation).
- [ ] Priority 6 — Regression testing (full coverage pass).
- [ ] Priority 7 — Documentation matching reality.

## Priority 1 Results (2026-10-31)

**Bug 1 — model asset path (fixed):** `DEFAULT_MODEL_PATH` in
`src/spatialx/vision/hand_landmarker.py` resolved to
`<repo>/src/assets/models/hand_landmarker.task` (`parents[2]` = `src`),
a nonexistent path. The real asset is at `<repo>/assets/models/`.

**Fix 1:**
- `resolve_model_path()` — verified resolution: explicit `--model-path` arg →
  `SPATIALX_MODEL_PATH` env var → package assets dir → repo-root
  `assets/models/` (walk-up). Every candidate is checked for existence and
  non-empty size; a missing model raises `VisionError` listing all searched
  locations plus the download URL.
- `HandLandmarker()` now auto-resolves the model (default `model_path=None`).
- CLI resolves/validates the model **before** opening the camera (clear error,
  exit code 4) and reports the resolved model path at runtime.

**Bug 2 — BGR/RGB channel order (fixed):** `HandLandmarker.detect()` wrapped the
raw OpenCV **BGR** frame as `ImageFormat.SRGB` without conversion. Diagnosed on
a live frame: BGR-as-SRGB → 0 hands; converted RGB → 1 hand on the same frame.
Fixed with `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` inside `detect()`.

**Verified:**
- 7 new regression tests in `tests/test_vision.py` (path resolution, env
  override, missing-model errors, default initialization + blank-frame infer,
  BGR→RGB conversion).
- Full suite: **64 passed**; `compileall` clean.
- Real end-to-end run: `spatialx --detect --frames 5` →
  `model=.../assets/models/hand_landmarker.task frames_captured=5
  avg_inference_ms=46.22`.
- **Live hand detection (hardware-verified):** 30 s / 300-frame run with both
  hands in view → hands detected on **300/300 frames** (554 hand instances,
  LEFT + RIGHT), 21-landmark skeleton overlay saved and visually verified.
  Live inference mean **69.0 ms/frame** with two hands (min 47.4, max 125.9);
  single-hand/empty-scene mean 41–42 ms.

## Priority 2 Results (2026-10-31)

**Defects found and corrected in `src/spatialx/spatial/coordinates.py`:**

1. **Incoherent camera-space units** (critical): old `landmark_camera_space`
   computed `(x_norm·W − cx)·fx/W` — a unit hybrid — and returned X/Y with no
   depth while Z carried wrist-relative `z`, so the vector was not a point in
   any consistent 3D frame. Corrected to honest camera pixel space:
   `[u − cx, v − cy, z·W]` (principal-point origin, width-scaled
   wrist-relative Z).
2. **Aspect-ratio scale mixing** (critical): normalized `x` is per-width and
   `y` per-height, so distances/cross-products on raw normalized coordinates
   were distorted (4:3 at 640×480). All geometry now computes in isotropic
   pixel space; regression test proves the pixel-space angle differs from the
   normalized-space angle.
3. **`palm_orientation` axis not unit-normalized**: returned `axis·angle` with
   `|axis| = |a||b|sinθ ≠ 1`, corrupting rotation-vector magnitude (and thus
   quaternion conversion). Corrected to unit axis × angle via `atan2`.
4. **`relative_depth` duplicated `hand_relative_vectors`**: now returns the
   1-D wrist-relative, width-scaled depth offsets its name promises.
5. **No mirroring support**: added `mirror=True` option (u → W−u) through
   `normalized_to_pixel`, `landmark_camera_space`, `hand_relative_vectors`,
   `palm_orientation`; CLI cursor demo uses selfie mirroring.
6. **`CameraCalibration` sloppy defaults**: `rotation`/`translation` are
   typed Optionals with `effective_*` accessors (identity/zero); added
   `for_image(width, height, hfov_degrees=60)` nominal constructor —
   documented as an assumption, not a measured calibration.
7. **Honesty**: no metric depth is ever produced; module docstrings state
   Z is wrist-relative and width-scaled, valid for relative comparisons only.

**Verified:**
- 13 spatial tests rewritten/added (mirroring, ray unit-norm, known-angle
  orientation, aspect-ratio regression, depth semantics, `for_image`).
- Full suite: **75 passed**; `compileall` clean.
- Live hardware run (hand in view): wrist camera space `[193, 253.5, 0]` px;
  index-tip offset `[−110.3, −283.0, −69.0]` px (311.5 px norm); monotonic
  fingertip depth offsets (−12.8 → −57.2 px, closer than wrist); palm tilt
  9.9° → unit quaternion (norm 1.000000); unit ray `[−0.147, −0.052, 0.988]`.
- `spatialx --cursor-demo --frames 30` on live data: 23 cursor positions,
  final `[208.7, 197.3, 0.0002]` px, jitter 0.0079.

## Environment Status

- **Camera restored and hardware-verified**: `/dev/video0` (640x480) captures at
  ~9–19 FPS; live two-hand landmark detection confirmed.
- Panda3D/PySide6 not yet installed (Priority 3 scope).
- Working tree note: unrelated pre-existing (uncommitted) changes remain in
  `src/spatialx/spatial/quaternion.py` / `__init__.py` (`relative_rotation`);
  preserved, not part of Priority 1, not committed.

## Test Results

- `python3 -m pytest -q`: **75 passed**.
- `python3 -m compileall src main.py tests`: OK.
- Model asset verified: `assets/models/hand_landmarker.task`, 7,819,105 bytes.

## Performance Measurements

- Hand landmarker inference, two hands live: mean **69.0 ms/frame**
  (300-frame sample), min 47.4, max 125.9.
- Hand landmarker inference, empty/single-hand scene: mean **41–42 ms/frame**.
- Historic (pre-repair) measurement: mean 36.4 ms (single hand, BGR path).

## Next Actions

- **Priority 3** — real 3D rendering with Panda3D (scene, camera, axes,
  cursor, cube, transform system; rendering/tracking separation).
