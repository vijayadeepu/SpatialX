# SpatialX — A Universal Human Spatial Interaction Engine

## Product Overview

SpatialX transforms an ordinary camera into a spatial input device. It uses
computer vision to track hands, interpret natural movement, and drive
three-dimensional digital environments — enabling people to manipulate
digital objects with natural hand gestures instead of a mouse and keyboard.

## Current Status — Milestone 0

**Milestone 0 (Project Foundation) is complete.**

The application now has a verified development environment, a modular package
skeleton, dependency management and an initial camera capture pipeline.

## Features

- Camera-based hand tracking foundation (in progress).
- 3D spatial coordinate engine (in progress).
- Virtual 3D cursor (in progress).
- Object manipulation (in progress).
- Two-hand interaction (planned).
- Virtual spatial interface (planned).
- Configurable gesture engine (planned).
- Blender integration (planned).

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

Open the live camera preview:

```bash
python main.py --preview
```

## Troubleshooting

- Camera not detected: check permissions (`/dev/video*`) and physical connection.
- MediaPipe model download: first run downloads the hand landmark model.

## Documentation

- [PROJECT_STATUS.md](PROJECT_STATUS.md)
- [ARCHITECTURE.md](ARCHITECTURE.md)
- [CHANGELOG.md](CHANGELOG.md)
- [docs/](docs/)
