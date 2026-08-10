"""Viser 3D scene: render robot capsules, sphere, voxels, contact normals."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

from .config import MetricConfig
from .robot import RobotContext, update_kinematics, capsule_endpoints_world
from .contacts import contact_kinematics
from .viser_geometry import capsule_mesh, sphere_mesh

if TYPE_CHECKING:
    import viser


def _rotation_from_z_axis(direction_world: np.ndarray) -> Rotation:
    """Rotation that maps local +Z onto the given direction."""
    z = np.array([0.0, 0.0, 1.0], dtype=float)
    d = np.array(direction_world, dtype=float)
    n = float(np.linalg.norm(d))
    if n < 1e-12:
        return Rotation.identity()
    d = d / n
    dot = float(np.clip(np.dot(z, d), -1.0, 1.0))
    if abs(dot - 1.0) < 1e-8:
        return Rotation.identity()
    if abs(dot + 1.0) < 1e-8:
        # 180deg about X (arbitrary axis orthogonal to Z)
        return Rotation.from_rotvec(np.array([np.pi, 0.0, 0.0]))
    axis = np.cross(z, d)
    axis_n = float(np.linalg.norm(axis))
    if axis_n < 1e-12:
        return Rotation.identity()
    axis = axis / axis_n
    angle = float(np.arccos(dot))
    return Rotation.from_rotvec(axis * angle)


def add_robot_to_scene(
    server: viser.ViserServer,
    ctx: RobotContext,
    q: np.ndarray,
    cfg: MetricConfig,
    prefix: str = "/robot",
) -> dict[str, viser.MeshHandle]:
    """Add all robot link capsules to the viser scene. Returns handles by link name."""
    update_kinematics(ctx, q)
    handles: dict[str, viser.MeshHandle] = {}

    for name in ctx.link_capsules.keys():
        a, b = capsule_endpoints_world(ctx, name)
        centre = 0.5 * (a + b)
        seg = b - a
        seg_len = float(np.linalg.norm(seg))
        if seg_len < 1e-12:
            direction = np.array([0.0, 0.0, 1.0])
            half_len = 0.0
        else:
            direction = seg / seg_len
            half_len = 0.5 * seg_len

        verts, faces = capsule_mesh(cfg.link_radius_m, half_len)
        rot = _rotation_from_z_axis(direction)

        # Color: active contacts are blue, passive are gray
        if name in cfg.all_contact_links:
            color = (100, 149, 237)  # cornflower blue
        else:
            color = (180, 180, 180)

        handle = server.scene.add_mesh_simple(
            f"{prefix}/{name}",
            vertices=verts,
            faces=faces,
            position=centre,
            wxyz=rot.as_quat(scalar_first=True),
            color=color,
        )
        handles[name] = handle

    return handles


def update_robot_poses(
    ctx: RobotContext,
    q: np.ndarray,
    handles: dict[str, viser.MeshHandle],
) -> None:
    """Update mesh positions/orientations after changing q."""
    update_kinematics(ctx, q)
    for name, handle in handles.items():
        a, b = capsule_endpoints_world(ctx, name)
        centre = 0.5 * (a + b)
        seg = b - a
        seg_len = float(np.linalg.norm(seg))
        if seg_len < 1e-12:
            direction = np.array([0.0, 0.0, 1.0])
        else:
            direction = seg / seg_len
        rot = _rotation_from_z_axis(direction)
        handle.position = centre
        handle.wxyz = rot.as_quat(scalar_first=True)


def add_sphere_to_scene(
    server: viser.ViserServer,
    centre: np.ndarray,
    radius: float,
    prefix: str = "/sphere",
) -> viser.MeshHandle:
    """Add a translucent sphere to the scene."""
    verts, faces = sphere_mesh(radius)
    handle = server.scene.add_mesh_simple(
        f"{prefix}/mesh",
        vertices=verts,
        faces=faces,
        position=centre,
        color=(255, 80, 80),
        opacity=0.6,
    )
    return handle


def add_sphere_frame_to_scene(
    server: viser.ViserServer,
    axis_length: float,
    prefix: str = "/sphere_frame",
    line_width: float = 4.0,
) -> dict[str, "viser.LineSegmentsHandle"]:
    """Create 3 axis line segments in the sphere *local* frame."""
    server.scene.remove_by_name(prefix)

    points_x = np.array([[[0.0, 0.0, 0.0], [axis_length, 0.0, 0.0]]], dtype=np.float32)
    points_y = np.array([[[0.0, 0.0, 0.0], [0.0, axis_length, 0.0]]], dtype=np.float32)
    points_z = np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, axis_length]]], dtype=np.float32)

    hx = server.scene.add_line_segments(
        f"{prefix}/x",
        points=points_x,
        colors=(255, 0, 0),
        line_width=line_width,
    )
    hy = server.scene.add_line_segments(
        f"{prefix}/y",
        points=points_y,
        colors=(0, 255, 0),
        line_width=line_width,
    )
    hz = server.scene.add_line_segments(
        f"{prefix}/z",
        points=points_z,
        colors=(0, 0, 255),
        line_width=line_width,
    )

    return {"x": hx, "y": hy, "z": hz}


def _box_mesh(half: float) -> tuple[np.ndarray, np.ndarray]:
    """Unit box vertices and faces with given half-extent."""
    s = half
    verts = np.array([
        [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
        [-s, -s,  s], [s, -s,  s], [s, s,  s], [-s, s,  s],
    ], dtype=np.float32)
    faces = np.array([
        [0,2,1], [0,3,2], [4,5,6], [4,6,7],
        [0,1,5], [0,5,4], [2,3,7], [2,7,6],
        [1,2,6], [1,6,5], [0,4,7], [0,7,3],
    ], dtype=np.uint32)
    return verts, faces


def _rotation_coverage_to_color(coverage: float, max_coverage: float = 0.25) -> tuple[int, int, int]:
    """Map rotation coverage [0, max_coverage] to a color gradient.

    Uses a fixed scale so colors are consistent across different hands.
    Red (low) -> Yellow (medium) -> Green (high).

    Args:
        coverage: Rotation coverage value (fraction of orientation bins reached)
        max_coverage: Value that maps to the brightest green (default 25%)

    Returns:
        RGB tuple (0-255)
    """
    # Clamp and normalize to [0, 1]
    t = min(1.0, max(0.0, coverage / max_coverage))

    # Red -> Yellow -> Green gradient
    if t < 0.5:
        # Red to Yellow (increase green)
        r = 255
        g = int(255 * (t * 2))
        b = 50
    else:
        # Yellow to Green (decrease red)
        r = int(255 * (1 - (t - 0.5) * 2))
        g = 255
        b = 50

    return (r, g, b)


def add_voxel_markers(
    server: viser.ViserServer,
    voxel_ori_bins: dict[tuple[int,int,int], set[int]],
    origin: np.ndarray,
    voxel_size: float,
    total_bins: int,
    prefix: str = "/voxels",
    max_coverage_for_color: float = 0.25,
    R_seed: np.ndarray | None = None,
) -> dict[tuple[int,int,int], viser.MeshHandle]:
    """Add colored cubes filling each reached voxel (no gaps).

    Voxel color indicates rotation coverage using a fixed scale:
    - Red: low rotation coverage (< 5% of orientation bins)
    - Yellow: medium rotation coverage (~12.5%)
    - Green: high rotation coverage (>= 25%)

    The scale is fixed so colors are consistent across different hands.

    If R_seed is provided, voxel indices are interpreted in the seed
    manipulability frame and rotated back to world frame for display.
    """
    handles = {}
    if not voxel_ori_bins:
        return handles

    box_verts, box_faces = _box_mesh(voxel_size * 0.5)

    for v, bins in voxel_ori_bins.items():
        coverage = len(bins) / total_bins
        color = _rotation_coverage_to_color(coverage, max_coverage_for_color)
        v_arr = np.array(v, dtype=float) * voxel_size
        if R_seed is not None:
            v_arr = R_seed @ v_arr
        pos = origin + v_arr

        handle = server.scene.add_mesh_simple(
            f"{prefix}/{v[0]}_{v[1]}_{v[2]}",
            vertices=box_verts,
            faces=box_faces,
            position=pos,
            color=color,
            opacity=0.15,
        )
        handles[v] = handle

    return handles


def highlight_voxel(
    server: viser.ViserServer,
    voxel: tuple[int,int,int] | None,
    origin: np.ndarray,
    voxel_size: float,
    prefix: str = "/voxel_highlight",
    R_seed: np.ndarray | None = None,
) -> viser.MeshHandle | None:
    """Show a bright wireframe-like highlight around the selected voxel."""
    if voxel is None:
        # Remove existing
        server.scene.remove_by_name(prefix)
        return None

    # Slightly larger cube so it wraps around the voxel
    box_verts, box_faces = _box_mesh(voxel_size * 0.52)
    v_arr = np.array(voxel, dtype=float) * voxel_size
    if R_seed is not None:
        v_arr = R_seed @ v_arr
    pos = origin + v_arr

    handle = server.scene.add_mesh_simple(
        prefix,
        vertices=box_verts,
        faces=box_faces,
        position=pos,
        color=(255, 255, 0),
        wireframe=True,
    )
    return handle


def add_contact_normals(
    server: viser.ViserServer,
    ctx: RobotContext,
    q: np.ndarray,
    sphere_centre: np.ndarray,
    contact_pair: tuple[str, str],
    sphere_radius_m: float,
    link_radius_m: float,
    length: float = 0.02,
    prefix: str = "/normals",
) -> None:
    """Draw contact normal arrows at both contact points."""
    # Avoid accumulating line segments across updates.
    server.scene.remove_by_name(prefix)
    update_kinematics(ctx, q)
    colors = [(255, 80, 80), (80, 80, 255)]  # red for thumb, blue for index
    for i, geom_name in enumerate(contact_pair):
        ck = contact_kinematics(ctx, q, sphere_centre, geom_name,
                                sphere_radius_m, link_radius_m)
        start = ck.sphere_point_world
        end = start + ck.normal_world * length
        points = np.array([[start, end]], dtype=np.float32)  # (1, 2, 3)
        server.scene.add_line_segments(
            f"{prefix}/{i}",
            points=points,
            colors=colors[i],
            line_width=3.0,
        )
