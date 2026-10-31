# SpatialX — A Universal Human Spatial Interaction Engine

## Product Overview

SpatialX transforms an ordinary camera into a spatial input device. It uses
computer vision to track hands, interpret natural movement, and drive
three-dimensional digital environments — enabling people to manipulate
digital objects with natural hand gestures instead of a mouse and keyboard.

## Current Status — Milestones 0 and 1

**Milestone 0 (Project Foundation) is complete.**

**Milestone 1 (Real-Time Hand Tracking) is complete.**

The application now has a verified development environment, a modular package
skeleton, dependency management, a camera capture pipeline, MediaPipe-based
hand tracking and landmark visualization.

## Features

- Camera capture and FPS reporting (`CameraManager`).
- Real-time hand tracking via MediaPipe Tasks (`HandLandmarker`).
- Two-hand support, 21 landmarks per hand, left/right identification.
- Hand skeleton visualization.
- Headless hand-detection runs with FPS / inference reporting (`--detect`).

## Installation

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Usage

Capture a short headless camera sample:

```bash
python main.py --frames 15
```

Run MediaPipe hand detection over captured frames (headless demo):

```bash
python main.py --frames 10 --detect
```

Show the live camera preview:

```bash
python main.py --preview
```

## Troubleshooting

- Camera not detected: check permissions (`/dev/video*`) and physical connection.

## Documentation

- [PROJECT_STATUS.md](PROJECT_STATUS.md)
- [ARCHITECTURE.md](ARCHITECTURE.md)
- [CHANGELOG.md](CHANGELOG.md)
- [docs/](docs/)
