# Architecture — SpatialX

## Overview

SpatialX is organised as a set of independent, feature-oriented subsystems
under `src/spatialx/`.

## Module Map

| Module | Responsibility |
|---|---|
| `camera` | Device discovery, frame capture, lifecycle, FPS monitoring. |
| `vision` | Hand detection, landmark extraction, skeleton visualization. |
| `spatial` | Spatial coordinate normalization, transformation, depth, orientation. |
| `gestures` | Geometric gesture recognition, two-hand logic. |
| `motion` | One Euro filtering, smoothing, prediction. |
| `interaction` | Object selection, grab, translate, rotate, scale states. |
| `rendering` | 3D environment, virtual cursor, virtual controls. |
| `integrations` | Blender adapter, external app adapters, plugin API. |
| `application` | CLI, dashboard wiring, runtime orchestration. |
| `configuration` | JSON settings loading. |
| `core` | Errors and cross-cutting primitives. |

## Current Data Flow — Milestones 0/1/2

```
main.py / CLI
    -> CameraManager (camera module) -> Frame (timestamped image)
    -> HandLandmarker (vision module, MediaPipe Tasks, VIDEO mode)
        -> HandDetection (hands, timestamp_ms, inference_time_ms)
        -> TrackedHand[] (side + 21 landmarks)
        -> draw_hand_skeleton (vision module, OpenCV)
    -> spatialx.spatial.coordinates (camera calibration, hand-relative math)
        -> reporting (FPS, latency)
```

Future flow inserts spatial mathematics between `HandDetection` and interaction.

## Vision Module Detail

- `spatialx.vision.hand_landmarker`: MediaPipe Tasks wrapper; `HandLandmarker`,
  `HandDetection`, `TrackedHand`, `Landmark`, `extract_hands`, model default path.
- `spatialx.vision.visualization`: OpenCV-based skeleton drawing.

## Design Decisions

- Strict module boundaries; no cross-module leakage yet.
- Vision emits SpatialX-native data (`TrackedHand`) — never raw MediaPipe internals.
- Errors derive from `SpatialXError`; inference failures surface as `VisionError`.

## Dependencies

- OpenCV — capture and skeleton drawing.
- MediaPipe Tasks — hand landmarks (Milestone 1, complete).
- NumPy / SciPy — mathematics.
- One Euro Filter — motion (Milestone 3).
- Panda3D / PySide6 — rendering / UI (upcoming milestones).

## Integration Boundaries

- Rendering consumes spatial coordinates; never vision internals.
- Interaction consumes gestures; never raw landmarks.
- Integrations consume interaction events; never vision internals.

## Coordinate Conventions

Documented in `docs/` as milestones land (see `docs/README.md`).
