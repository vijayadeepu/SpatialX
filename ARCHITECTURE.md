# Architecture — SpatialX

## Overview

SpatialX is organised as a set of independent, feature-oriented subsystems
under `src/spatialx/`.

## Module Map

| Module | Responsibility |
|---|---|
| `camera` | Device discovery, frame capture, lifecycle, FPS monitoring. |
| `vision` | Hand detection, tracking, landmarks, confidence. |
| `spatial` | Coordinate normalization, transformation, depth, orientation. |
| `gestures` | Geometric gesture recognition, two-hand logic. |
| `motion` | One Euro filtering, smoothing, prediction. |
| `interaction` | Object selection, grab, translate, rotate, scale states. |
| `rendering` | 3D environment, virtual cursor, virtual controls. |
| `integrations` | Blender adapter, external app adapters, plugin API. |
| `application` | CLI, dashboard wiring, runtime orchestration. |
| `configuration` | JSON settings loading. |
| `core` | Errors and cross-cutting primitives. |

## Current Data Flow (Milestone 0)

```
main.py / CLI
    -> CameraManager (camera module)
        -> Frame (timestamped image)
    -> stats (FPS, latency)
```

Future flow adds vision inference between capture and spatial mathematics.

## Design Decisions

- Strict module boundaries; no cross-module leakage yet.
- Camera ownership stays inside `spatialx.camera`.
- Errors derive from `SpatialXError`.

## Dependencies

- OpenCV — capture.
- MediaPipe Tasks — hand landmarks (Milestone 1).
- NumPy / SciPy — mathematics.
- One Euro Filter — motion (Milestone 3).
- Panda3D / PySide6 — rendering / UI (upcoming milestones).

## Integration Boundaries

- Rendering consumes spatial coordinates; never vision internals.
- Interaction consumes gestures; never raw landmarks.
- Integrations consume interaction events; never vision internals.

## Coordinate Conventions

Documented in `docs/` as milestones land (see `docs/README.md`).
