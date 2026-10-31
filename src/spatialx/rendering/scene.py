"""Real-time 3D scene for SpatialX, powered by Panda3D.

Separation of concerns: this module never imports camera or vision code.
The tracking/interaction pipeline produces plain numpy vectors and
quaternions; this module maps them into scene space and renders them.

Coordinate mapping (documented contract)
----------------------------------------
- SpatialX camera space: ``X`` right, ``Y`` down, ``Z`` forward (pixel
  units, wrist-relative Z — see ``spatialx.spatial.coordinates``).
- Panda3D scene space: ``X`` right, ``Y`` forward, ``Z`` up (world units).
- Vector mapping: ``scene = scale * (x, z, -y)`` — a proper rotation
  (right-handed to right-handed), so orientations map by the similarity
  transform ``q_scene = qA * q_camera * qA^-1`` where ``qA`` is the
  quaternion of the axis mapping (rotation of -90 degrees about X).
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

import numpy as np

from panda3d.core import (
    AmbientLight,
    DirectionalLight,
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    LineSegs,
    PNMImage,
    LQuaternion as PandaQuaternion,
    Vec4,
    loadPrcFileData,
)
from direct.showbase.ShowBase import ShowBase

from spatialx.spatial.quaternion import (
    Quaternion,
    quaternion_conjugate,
    quaternion_multiply,
    rotate_vector,
)

# Quaternion of the camera-space -> scene-space axis mapping
# (scene = (x, z, -y): a rotation of -90 degrees about the X axis).
AXIS_MAP_QUATERNION = Quaternion(
    w=math.sqrt(0.5), x=-math.sqrt(0.5), y=0.0, z=0.0
)
_AXIS_MAP_INVERSE = quaternion_conjugate(AXIS_MAP_QUATERNION)

DEFAULT_BACKGROUND = (0.055, 0.065, 0.09)


def camera_space_to_scene(
    vector: np.ndarray, scale: float = 0.01
) -> np.ndarray:
    """Map a camera-space vector to scene space: ``scale * (x, z, -y)``."""
    v = np.asarray(vector, dtype=float).reshape(-1)
    return scale * np.array([v[0], v[2], -v[1]], dtype=float)


def scene_space_to_camera(
    vector: np.ndarray, scale: float = 0.01
) -> np.ndarray:
    """Inverse of :func:`camera_space_to_scene` (unscaled)."""
    v = np.asarray(vector, dtype=float).reshape(-1)
    return np.array([v[0], -v[2], v[1]], dtype=float) / scale


def camera_quat_to_scene(quat: Quaternion) -> Quaternion:
    """Map a camera-space orientation quaternion to scene space."""
    return quaternion_multiply(
        quaternion_multiply(AXIS_MAP_QUATERNION, quat), _AXIS_MAP_INVERSE
    )


def scene_quat_to_camera(quat: Quaternion) -> Quaternion:
    """Inverse of :func:`camera_quat_to_scene`."""
    return quaternion_multiply(
        quaternion_multiply(_AXIS_MAP_INVERSE, quat), AXIS_MAP_QUATERNION
    )


def _make_cube_geom(size: float, color: Vec4) -> GeomNode:
    """Build a unit cube GeomNode with per-face normals (lit shading)."""
    half = size / 2.0
    vdata = GeomVertexData("cube", GeomVertexFormat.getV3n3(), Geom.UHStatic)
    vertex = GeomVertexWriter(vdata, "vertex")
    normal = GeomVertexWriter(vdata, "normal")
    faces = (
        ((1, 0, 0), ((half, -half, -half), (half, half, -half),
                     (half, half, half), (half, -half, half))),
        ((-1, 0, 0), ((-half, -half, -half), (-half, -half, half),
                      (-half, half, half), (-half, half, -half))),
        ((0, 1, 0), ((-half, half, -half), (-half, half, half),
                     (half, half, half), (half, half, -half))),
        ((0, -1, 0), ((-half, -half, -half), (half, -half, -half),
                      (half, -half, half), (-half, -half, half))),
        ((0, 0, 1), ((-half, -half, half), (half, -half, half),
                     (half, half, half), (-half, half, half))),
        ((0, 0, -1), ((-half, -half, -half), (-half, half, -half),
                      (half, half, -half), (half, -half, -half))),
    )
    triangles = GeomTriangles(Geom.UHStatic)
    for normal_vector, corners in faces:
        start = vdata.getNumRows()
        for corner in corners:
            vertex.addData3(*corner)
            normal.addData3(*normal_vector)
        triangles.addVertices(start, start + 1, start + 2)
        triangles.closePrimitive()
        triangles.addVertices(start, start + 2, start + 3)
        triangles.closePrimitive()
    geom = Geom(vdata)
    geom.addPrimitive(triangles)
    node = GeomNode("cube")
    node.addGeom(geom)
    node.setPythonTag("spatialx_color", color)
    return node


def _make_sphere_geom(radius: float, longitude: int = 16, latitude: int = 12) -> GeomNode:
    """Build a UV-sphere GeomNode (unshaded cursor marker)."""
    vdata = GeomVertexData("sphere", GeomVertexFormat.getV3(), Geom.UHStatic)
    vertex = GeomVertexWriter(vdata, "vertex")
    for ring in range(latitude + 1):
        theta = math.pi * ring / latitude
        for segment in range(longitude + 1):
            phi = 2.0 * math.pi * segment / longitude
            vertex.addData3(
                radius * math.sin(theta) * math.cos(phi),
                radius * math.sin(theta) * math.sin(phi),
                radius * math.cos(theta),
            )
    triangles = GeomTriangles(Geom.UHStatic)
    stride = longitude + 1
    for ring in range(latitude):
        for segment in range(longitude):
            a = ring * stride + segment
            b = a + stride
            triangles.addVertices(a, b, a + 1)
            triangles.closePrimitive()
            triangles.addVertices(a + 1, b, b + 1)
            triangles.closePrimitive()
    geom = Geom(vdata)
    geom.addPrimitive(triangles)
    node = GeomNode("sphere")
    node.addGeom(geom)
    return node


def _build_axes_node(length: float = 1.5) -> GeomNode:
    """RGB axis lines: X red, Y green, Z blue (scene space)."""
    segs = LineSegs()
    segs.setThickness(2.5)
    segs.setColor(0.85, 0.25, 0.25, 1.0)
    segs.moveTo(0.0, 0.0, 0.0)
    segs.drawTo(length, 0.0, 0.0)
    segs.setColor(0.3, 0.8, 0.3, 1.0)
    segs.moveTo(0.0, 0.0, 0.0)
    segs.drawTo(0.0, length, 0.0)
    segs.setColor(0.35, 0.5, 0.95, 1.0)
    segs.moveTo(0.0, 0.0, 0.0)
    segs.drawTo(0.0, 0.0, length)
    return segs.create()


def _build_grid_node(extent: float = 3.0, step: float = 0.5) -> GeomNode:
    """Ground-plane grid on z=0 (scene space)."""
    segs = LineSegs()
    segs.setThickness(1.0)
    segs.setColor(0.25, 0.28, 0.35, 1.0)
    value = -extent
    while value <= extent + 1e-9:
        segs.moveTo(-extent, value, 0.0)
        segs.drawTo(extent, value, 0.0)
        segs.moveTo(value, -extent, 0.0)
        segs.drawTo(value, extent, 0.0)
        value += step
    return segs.create()


class SceneObject:
    """Transform handle for an object inside a :class:`SpatialScene`.

    Positions are accepted in either scene space (world units) or camera
    space (pixel units, mapped through the scene scale). Orientations use
    SpatialX :class:`~spatialx.spatial.quaternion.Quaternion` values in
    camera space and are mapped to scene space internally.
    """

    def __init__(self, node, scene: "SpatialScene") -> None:
        self._node = node
        self._scene = scene

    @property
    def node(self):
        """The underlying Panda3D NodePath."""
        return self._node

    def set_position(self, position_scene: Sequence[float]) -> None:
        """Place the object at a scene-space position (world units)."""
        v = np.asarray(position_scene, dtype=float).reshape(-1)
        self._node.setPos(float(v[0]), float(v[1]), float(v[2]))

    def set_position_camera_space(
        self, position_camera: Sequence[float]
    ) -> None:
        """Place the object from a camera-space vector (pixel units)."""
        self.set_position(
            camera_space_to_scene(position_camera, self._scene.space_scale)
        )

    def get_position_scene(self) -> np.ndarray:
        """Scene-space position (world units)."""
        p = self._node.getPos()
        return np.array([p[0], p[1], p[2]], dtype=float)

    def get_position_camera_space(self) -> np.ndarray:
        """Position mapped back to camera space (pixel units)."""
        return scene_space_to_camera(
            self.get_position_scene(), self._scene.space_scale
        )

    def set_orientation(self, orientation: Quaternion) -> None:
        """Set orientation from a camera-space SpatialX quaternion."""
        scene_quat = camera_quat_to_scene(orientation)
        # Panda3D's reliable constructor order is (x, y, z, w).
        self._node.setQuat(
            PandaQuaternion(
                scene_quat.x, scene_quat.y, scene_quat.z, scene_quat.w
            )
        )

    def get_orientation(self) -> Quaternion:
        """Orientation mapped back to a camera-space SpatialX quaternion."""
        pq = self._node.getQuat()
        scene_quat = Quaternion(
            w=float(pq.getW()),
            x=float(pq.getX()),
            y=float(pq.getY()),
            z=float(pq.getZ()),
        )
        return scene_quat_to_camera(scene_quat)

    def set_color(self, r: float, g: float, b: float, a: float = 1.0) -> None:
        """Set the object's flat color."""
        self._node.setColor(float(r), float(g), float(b), float(a))

    def set_scale(self, scale: float) -> None:
        """Uniform scale."""
        self._node.setScale(float(scale))

    def remove(self) -> None:
        """Remove the object from the scene."""
        self._node.removeNode()


class SpatialScene:
    """Panda3D scene: environment, camera, axes, grid, cube and cursor.

    Args:
        window_type: ``"onscreen"`` (interactive window), ``"offscreen"``
            (headless buffer with real GL rendering, supports screenshots),
            or ``"none"`` (scene graph only, no graphics system).
        width/height: Render window or buffer size in pixels.
        space_scale: World units per camera-space pixel unit (default 0.01,
            i.e. a 640 px-wide frame maps to ~6.4 world units).
        background: Clear color (RGBA).
        display_pipe: Optional Panda3D display pipe to force, e.g.
            ``"tinydisplay"`` (pure software). The GLX *offscreen* buffer
            is unreliable on some drivers; use ``window_type="offscreen"``
            with ``display_pipe="tinydisplay"`` for headless rendering.
    """

    def __init__(
        self,
        window_type: str = "offscreen",
        width: int = 640,
        height: int = 480,
        space_scale: float = 0.01,
        background: Sequence[float] = DEFAULT_BACKGROUND,
        display_pipe: Optional[str] = None,
    ) -> None:
        self.space_scale = float(space_scale)
        config = [
            f"window-type {window_type}",
            f"win-size {width} {height}",
            "audio-library-name none",
            "sync-video false",
            "framebuffer-multisample 0",
            "depth-bits 24",
            "notify-level warning",
        ]
        if display_pipe:
            # e.g. "tinydisplay" — pure-software pipe, robust for offscreen
            # rendering and headless verification.
            config.append(f"load-display {display_pipe}")
        loadPrcFileData("", "\n".join(config))
        self._base = ShowBase(windowType=window_type)
        self._base.disableMouse()
        self._base.setBackgroundColor(*background, 1.0)
        self._objects: list[SceneObject] = []
        self._cube: Optional[SceneObject] = None
        self._cursor: Optional[SceneObject] = None
        self._spin_enabled = False
        self._setup_environment()
        self._base.accept("escape", self.close)

    # --- environment -------------------------------------------------------

    def _setup_environment(self) -> None:
        ambient = AmbientLight("spatialx-ambient")
        ambient.setColor(Vec4(0.45, 0.45, 0.5, 1.0))
        self._base.render.setLight(
            self._base.render.attachNewNode(ambient)
        )
        sun = DirectionalLight("spatialx-sun")
        sun.setColor(Vec4(0.85, 0.82, 0.75, 1.0))
        sun_node = self._base.render.attachNewNode(sun)
        sun_node.setHpr(-35.0, -50.0, 0.0)
        self._base.render.setLight(sun_node)

        self._base.render.attachNewNode(_build_axes_node())
        self._base.render.attachNewNode(_build_grid_node())
        self.set_camera(position=(4.5, -4.5, 3.2), target=(0.0, 0.0, 0.5))

    def set_camera(
        self,
        position: Sequence[float],
        target: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> None:
        """Position the scene camera and aim it at ``target``."""
        if self._base.camera is None:  # e.g. window_type="none"
            return
        self._base.camera.setPos(*[float(v) for v in position])
        self._base.camera.lookAt(*[float(v) for v in target])

    # --- objects -----------------------------------------------------------

    def add_cube(
        self,
        position_camera: Optional[Sequence[float]] = None,
        size: float = 1.0,
        color: Sequence[float] = (0.25, 0.6, 0.9),
    ) -> SceneObject:
        """Add the manipulable virtual cube to the scene."""
        node = self._base.render.attachNewNode(
            _make_cube_geom(size, Vec4(*[float(c) for c in color], 1.0))
        )
        node.setColor(*[float(c) for c in color], 1.0)
        obj = SceneObject(node, self)
        if position_camera is not None:
            obj.set_position_camera_space(position_camera)
        else:
            obj.set_position((0.0, 0.0, 0.5 + size / 2.0))
        self._objects.append(obj)
        self._cube = obj
        return obj

    def add_cursor(self, radius: float = 0.12, color=(1.0, 0.85, 0.2)) -> SceneObject:
        """Add the spatial cursor marker (unshaded sphere)."""
        node = self._base.render.attachNewNode(_make_sphere_geom(radius))
        node.setColor(*[float(c) for c in color], 1.0)
        node.setLightOff(1)
        obj = SceneObject(node, self)
        obj.set_position((0.0, -2.0, 1.0))
        self._objects.append(obj)
        self._cursor = obj
        return obj

    @property
    def cube(self) -> Optional[SceneObject]:
        return self._cube

    @property
    def cursor(self) -> Optional[SceneObject]:
        return self._cursor

    @property
    def base(self) -> ShowBase:
        return self._base

    def update_cursor(self, position_camera: Sequence[float]) -> None:
        """Move the cursor sphere from a camera-space hand position."""
        if self._cursor is not None:
            self._cursor.set_position_camera_space(position_camera)

    # --- rendering loop ----------------------------------------------------

    def enable_cube_spin(self, degrees_per_second: float = 25.0) -> None:
        """Spin the cube slowly (demo visualization)."""
        self._spin_rate = float(degrees_per_second)
        self._spin_enabled = True
        self._base.taskMgr.add(self._spin_task, "spatialx-cube-spin")

    def set_cube_spin(self, enabled: bool) -> None:
        """Enable/disable the demo cube spin at the existing rate."""
        if hasattr(self, "_spin_rate"):
            self._spin_enabled = bool(enabled)

    def _spin_task(self, task):
        if self._spin_enabled and self._cube is not None:
            dt = self._base.clock.getDt()
            self._cube.node.setH(
                self._cube.node.getH() + self._spin_rate * dt
            )
        return task.cont

    def step(self, frames: int = 1) -> None:
        """Advance the task manager and render ``frames`` frames."""
        for _ in range(int(frames)):
            self._base.taskMgr.step()

    def run(
        self, update_callback: Optional[Callable[[float], None]] = None
    ) -> None:
        """Run the interactive loop (onscreen demos).

        Args:
            update_callback: Called each frame with ``dt`` seconds; receives
                hand-tracking updates from the application layer.
        """
        if update_callback is not None:

            def _update_task(task):
                update_callback(self._base.clock.getDt())
                return task.cont

            self._base.taskMgr.add(_update_task, "spatialx-app-update")
        self._base.run()

    def save_screenshot(self, path: str) -> bool:
        """Render a frame and write it to ``path`` (PNG supported).

        Returns:
            ``True`` when a graphics window/buffer exists and the image
            was written.
        """
        if self._base.win is None:
            return False
        self._base.graphicsEngine.renderFrame()
        self._base.graphicsEngine.renderFrame()
        image = PNMImage()
        if not self._base.win.getScreenshot(image):
            return False
        return bool(image.write(path))

    def close(self) -> None:
        """Destroy the scene and its graphics resources."""
        self._spin_enabled = False
        try:
            self._base.destroy()
        except Exception:
            pass
