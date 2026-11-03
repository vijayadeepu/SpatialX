# SpatialX — A Universal Human Spatial Interaction Engine

SpatialX turns an ordinary webcam into a spatial input device. It tracks
hands with MediaPipe, interprets natural movement, and drives a Panda3D
scene so a person can grab, translate and rotate a virtual 3D object.

The stack is offline-first and CPU-only. There is no cloud dependency and
no extra AI model beyond the bundled MediaPipe hand landmarker.

## Current status

**Prototype (v0.6.0).** Hand tracking, spatial mathematics, process-isolated
rendering, gesture recognition and object manipulation (grab / translate /
rotate / release) are implemented and covered by automated tests.

**A scripted live hand session was not completed in the last engineering
pass.** The initial webcam frame was black; a 5-minute worker probe later
reported hands on 855/2950 frames (transient presence), but nobody
performed a recorded grab / translate / rotate session. Window lifecycle,
IPC, no-hand tracking and synthetic scene-graph manipulation *were*
verified. See [PROJECT_STATUS.md](PROJECT_STATUS.md) for the evidence
matrix.

## Features that work

- Camera capture with lifecycle and FPS reporting (`CameraManager`).
- MediaPipe Tasks hand landmarker (21 landmarks, left/right, VIDEO mode).
- Deterministic geometric gestures: `PINCH`, `OPEN_PALM`, `FIST`, `NONE`.
- Honest camera-space math (pixel units, **relative** depth only).
- One Euro filtering for the virtual cursor.
- Process-isolated tracking: MediaPipe/OpenCV in a worker subprocess,
  Panda3D in the main process, JSON-lines IPC on stdin/stdout.
- Panda3D scene: cube, cursor sphere, axes, ground grid, status text.
- Hand-controlled cursor with mirroring (selfie view).
- Hover detection, grab with offset preservation, translation, quaternion
  rotation from palm orientation, release, tracking-loss freeze.
- Headless screenshot rendering (`--render-screenshot`, TinyDisplay).
- Interactive 3D demo (`--render-demo`, onscreen GLX).

## System requirements

- Linux (tested on X11/XFCE; Wayland is untested).
- Python 3.11+.
- A UVC webcam at `/dev/video*` (640×480 or better).
- A GLX display for `--render-demo`. Offscreen screenshots use TinyDisplay
  and do not need a working GLX offscreen buffer.
- CPU only. No GPU or CUDA required.

## Installation

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

The hand landmarker model is looked up automatically:

1. `--model-path` argument
2. `SPATIALX_MODEL_PATH` environment variable
3. packaged `assets/models/hand_landmarker.task`
4. repository-root `assets/models/hand_landmarker.task`

The bundled model is `assets/models/hand_landmarker.task`
(float16, ~7.8 MB). If it is missing, download it from the URL printed in
the `VisionError` message.

## Running the application

Interactive 3D demo (webcam + Panda3D window):

```bash
spatialx --render-demo
```

Show a hand to move the yellow cursor. Hover near the cube and **pinch**
to grab. Move your hand to translate. Rotate your palm to rotate the cube.
**Open palm** to release. **Fist** pauses interaction.
Press `Escape` or `q` to quit.

Useful flags:

| Flag | Meaning |
|---|---|
| `--camera-index N` | Camera device index (default 0). |
| `--num-hands N` | MediaPipe hand count (default 2; interaction uses hand 0). |
| `--model-path PATH` | Explicit `.task` model path. |
| `--hover-radius PX` | Grab hover radius in camera pixels (default 160). |
| `--cursor-sensitivity X` | Cursor gain (default 1.0). |
| `--tracking-fps N` | Worker publish rate cap (default 30). |

Other CLI modes:

```bash
spatialx --preview                 # live camera window
spatialx --frames 15 --detect      # headless detection over N frames
spatialx --frames 15 --cursor-demo # headless virtual-cursor pipeline
spatialx --render-screenshot out.png  # offscreen 3D scene (no camera)
```

## Known limitations

- **No metric depth.** An RGB webcam cannot measure absolute distance.
  MediaPipe Z is wrist-relative and width-scaled. Translation along the
  camera forward axis is therefore a relative offset only, not a metre
  value. Screen-plane translation (left/right/up/down) is the reliable
  degrees of freedom.
- **Single-hand, single-object.** Milestone 6 (two-hand interaction) is
  out of scope. The manipulable object is one cube.
- **Palm orientation is monocular.** Rotation quality depends on landmark
  noise; it is smoothed with slerp but is not a 6-DoF IMU.
- **MediaPipe and Panda3D cannot share a process** on this hardware. The
  worker/IPC split is mandatory; do not import `cv2` or `mediapipe` in the
  rendering process.
- **GLX offscreen is unreliable** on this driver. Screenshots use
  TinyDisplay.
- **License undecided.** `pyproject.toml` currently says
  `license = { text = "Proprietary" }`. There is no `LICENSE` file. A
  license decision is still required.

## Documentation

- [PROJECT_STATUS.md](PROJECT_STATUS.md) — verified status and evidence.
- [ARCHITECTURE.md](ARCHITECTURE.md) — modules, IPC, coordinates, lifecycle.
- [CHANGELOG.md](CHANGELOG.md) — change history.
- [docs/](docs/) — documentation index (thin).
