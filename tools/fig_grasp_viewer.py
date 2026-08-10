#!/usr/bin/env python3
"""Figure: Interactive grasp viewer showing hand + sphere + contacts + pinch axis.

Renders:
  - Robot hand (capsule links)
  - Sphere at seed grasp
  - Two contact points with normal arrows
  - Pinch axis line through the sphere (connecting the two contacts)

Loads from saved results when available (fast); falls back to full
select_seed_and_frame (same as the metric pipeline).

Usage (run from the repo root):
    python tools/fig_grasp_viewer.py [--metric-config karma_config.yaml]
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np
import viser

# Repo root (this script lives in tools/); make `karma` importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from karma.config import load_config
from karma.robot import (
    load_robot, RobotContext, update_kinematics, neutral_configuration,
)
from karma.contacts import contact_kinematics
from karma.viser_scene import add_robot_to_scene
from karma.viser_geometry import sphere_mesh
from karma.storage import list_saved_results, load_result
from karma.orientation import (
    pinch_axis_world,
    compute_motion_primitives_frame,
    compute_rotation_primitives,
)
from karma.math_utils import unit

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _scan_robot_configs(robots_dir: Path = Path("robots")) -> dict[str, Path]:
    configs = {}
    if not robots_dir.exists():
        return configs
    for yaml_path in sorted(robots_dir.glob("robot_*.yaml")):
        stem = yaml_path.stem
        if stem == "robot_template" or stem.endswith(".tuned"):
            continue  # onboarding template / scratch tuning output, not a shipped hand
        configs[stem.replace("robot_", "")] = yaml_path
    return configs


def _find_saved_result_for_robot(urdf_name: str) -> Path | None:
    """Find the most recent saved result matching this robot's URDF name."""
    saved = list_saved_results()
    for s in saved:
        if s.urdf_name == urdf_name:
            return s.filepath
    return None


def _draw_arrow(
    server: viser.ViserServer,
    name: str,
    start: np.ndarray,
    end: np.ndarray,
    color: tuple[int, int, int],
    line_width: float = 3.0,
    head_length: float = 0.003,
    head_radius: float = 0.0015,
) -> None:
    """Draw a line segment with a cone arrowhead at the tip."""
    points = np.array([[start, end]], dtype=np.float32)
    server.scene.add_line_segments(
        f"{name}/shaft",
        points=points,
        colors=color,
        line_width=line_width,
    )
    # Arrowhead: cone mesh has apex at origin, opens along +Z.
    # We want the apex at `end` pointing along `d` (arrow direction).
    # So rotate +Z onto -d, placing the apex (origin) at `end`.
    d = end - start
    n = float(np.linalg.norm(d))
    if n < 1e-12:
        return
    d = d / n
    verts, faces = _cone_mesh(
        half_angle=math.atan(head_radius / head_length),
        height=head_length,
        n_segments=12,
    )
    wxyz = _rotation_from_z_to(-d)  # cone opens backwards (base behind tip)
    server.scene.add_mesh_simple(
        f"{name}/head",
        vertices=verts,
        faces=faces,
        position=end,  # apex at the arrow tip
        wxyz=wxyz,
        color=color,
        opacity=1.0,
        side="double",
    )


def _draw_contact_normals(
    server: viser.ViserServer,
    ctx: RobotContext,
    q: np.ndarray,
    sphere_centre: np.ndarray,
    contact_pair: tuple[str, str],
    sphere_radius_m: float,
    link_radius_m: float,
    length: float = 0.008,
    prefix: str = "/normals",
) -> None:
    """Draw contact normal arrows at both contact points."""
    server.scene.remove_by_name(prefix)
    update_kinematics(ctx, q)
    colors = [(255, 80, 80), (80, 80, 255)]  # red for thumb, blue for index
    for i, geom_name in enumerate(contact_pair):
        ck = contact_kinematics(
            ctx, q, sphere_centre, geom_name, sphere_radius_m, link_radius_m,
        )
        start = ck.sphere_point_world + ck.normal_world * 0.001
        end = start + ck.normal_world * length
        points = np.array([[start, end]], dtype=np.float32)
        server.scene.add_line_segments(
            f"{prefix}/{i}",
            points=points,
            colors=colors[i],
            line_width=3.0,
        )


def _draw_pinch_axis(
    server: viser.ViserServer,
    contact_point_thumb: np.ndarray,
    contact_point_index: np.ndarray,
    sphere_radius: float,
    prefix: str = "/pinch_axis",
    line_width: float = 4.0,
    overshoot: float = 0.8,
) -> None:
    """Draw the pinch axis: a line through the two contact points, extended for visibility."""
    server.scene.remove_by_name(prefix)

    # Direction from thumb contact to index contact
    axis_dir = contact_point_index - contact_point_thumb
    axis_len = float(np.linalg.norm(axis_dir))
    if axis_len < 1e-12:
        return
    axis_dir = axis_dir / axis_len

    # Extend beyond each contact point
    ext = sphere_radius * overshoot
    start = contact_point_thumb - ext * axis_dir
    end = contact_point_index + ext * axis_dir

    # Dotted line: alternate dash/gap segments
    total_len = float(np.linalg.norm(end - start))
    dash = sphere_radius * 0.3
    gap = sphere_radius * 0.2
    stride = dash + gap

    segments = []
    t = 0.0
    while t < total_len:
        t_end = min(t + dash, total_len)
        seg_start = start + axis_dir * t
        seg_end = start + axis_dir * t_end
        segments.append([seg_start, seg_end])
        t += stride

    if segments:
        points = np.array(segments, dtype=np.float32)
        server.scene.add_line_segments(
            f"{prefix}/line",
            points=points,
            colors=(20, 20, 20),  # near-black
            line_width=line_width,
        )


def _cone_mesh(
    half_angle: float,
    height: float,
    n_segments: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a cone mesh with apex at origin, opening along +Z.

    Args:
        half_angle: Cone half-angle in radians.
        height: Height of the cone along +Z.
        n_segments: Number of segments around the circumference.

    Returns:
        (vertices, faces) arrays.
    """
    radius = height * math.tan(half_angle)
    # Apex at origin
    verts = [np.array([0.0, 0.0, 0.0], dtype=np.float32)]
    # Ring at z = height
    for i in range(n_segments):
        theta = 2.0 * math.pi * i / n_segments
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        verts.append(np.array([x, y, height], dtype=np.float32))
    # Base centre vertex
    base_centre_idx = n_segments + 1
    verts.append(np.array([0.0, 0.0, height], dtype=np.float32))

    verts = np.array(verts, dtype=np.float32)

    faces = []
    # Side faces: apex (0) -> ring[i] -> ring[i+1]
    for i in range(n_segments):
        j = (i + 1) % n_segments
        faces.append([0, i + 1, j + 1])
    # Base cap faces (wound opposite so normal faces -Z, i.e. toward the viewer from behind)
    for i in range(n_segments):
        j = (i + 1) % n_segments
        faces.append([base_centre_idx, j + 1, i + 1])

    faces = np.array(faces, dtype=np.uint32)
    return verts, faces


def _rotation_from_z_to(direction: np.ndarray) -> np.ndarray:
    """Return a 4-element wxyz quaternion that rotates +Z onto the given direction."""
    from scipy.spatial.transform import Rotation

    z = np.array([0.0, 0.0, 1.0])
    d = unit(direction)
    dot = float(np.clip(np.dot(z, d), -1.0, 1.0))
    if abs(dot - 1.0) < 1e-8:
        return Rotation.identity().as_quat(scalar_first=True)
    if abs(dot + 1.0) < 1e-8:
        return Rotation.from_rotvec(np.array([np.pi, 0.0, 0.0])).as_quat(scalar_first=True)
    axis = np.cross(z, d)
    axis = axis / float(np.linalg.norm(axis))
    angle = float(np.arccos(dot))
    return Rotation.from_rotvec(axis * angle).as_quat(scalar_first=True)


def _draw_motion_primitives(
    server: viser.ViserServer,
    sphere_centre: np.ndarray,
    R_seed: np.ndarray,
    pinch_axis_w: np.ndarray,
    voxel_size_m: float,
    rotation_step_rad: float,
    sphere_radius_m: float,
    prefix: str = "/motion_primitives",
    line_width: float = 3.0,
    arrow_scale: float = 1.5,
) -> None:
    """Draw translation and rotation motion primitives from the sphere centre.

    Translation: 6 frame-aligned arrows (±x, ±y, ±z of R_seed) in green.
    Rotation: 4 curved arrows perpendicular to pinch axis in orange.
    """
    server.scene.remove_by_name(prefix)

    # --- Translation primitives (6 directions) ---
    trans_prims = compute_motion_primitives_frame(R_seed, voxel_size_m)
    for i, dp in enumerate(trans_prims):
        d = unit(dp)
        start = sphere_centre + sphere_radius_m * 1.5 * d
        end = start + dp * arrow_scale
        _draw_arrow(
            server, f"{prefix}/trans/{i}", start, end,
            color=(50, 180, 50), line_width=line_width,
        )

    # --- Rotation primitives (4 perpendicular to pinch axis) ---
    rot_prims = compute_rotation_primitives(pinch_axis_w, rotation_step_rad)
    for i, omega in enumerate(rot_prims):
        d = unit(omega)
        start = sphere_centre + sphere_radius_m * 1.5 * d
        end = start + d * sphere_radius_m * 0.75
        _draw_arrow(
            server, f"{prefix}/rot/{i}", start, end,
            color=(255, 140, 0), line_width=line_width,
        )


def _draw_contact_dots(
    server: viser.ViserServer,
    ctx: RobotContext,
    q: np.ndarray,
    sphere_centre: np.ndarray,
    contact_pair: tuple[str, str],
    sphere_radius_m: float,
    link_radius_m: float,
    prefix: str = "/contact_dots",
    dot_radius: float = 0.001,
) -> None:
    """Draw small spheres at the two contact points on the sphere surface."""
    server.scene.remove_by_name(prefix)
    colors = [(255, 80, 80), (80, 80, 255)]  # red for thumb, blue for index

    for i, geom_name in enumerate(contact_pair):
        ck = contact_kinematics(
            ctx, q, sphere_centre, geom_name, sphere_radius_m, link_radius_m,
        )
        verts, faces = sphere_mesh(dot_radius)
        server.scene.add_mesh_simple(
            f"{prefix}/{i}",
            vertices=verts,
            faces=faces,
            position=ck.sphere_point_world,
            color=colors[i],
        )


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Grasp viewer figure")
    parser.add_argument(
        "--metric-config",
        default="karma_config.yaml",
        help="Global fine-metric config",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    robot_configs = _scan_robot_configs()
    if not robot_configs:
        logger.error("No robot configs found in robots/ directory")
        return

    robot_names = list(robot_configs.keys())
    initial_robot = robot_names[0]

    cfg = load_config(args.metric_config, robot_configs[initial_robot])
    ctx = load_robot(cfg.urdf_path, cfg)

    current = {"name": initial_robot, "cfg": cfg, "ctx": ctx}

    server = viser.ViserServer(host=cfg.viewer_host, port=cfg.viewer_port)

    state: dict = {
        "busy": False,
        "robot_handles": None,
        "sphere_handle": None,
    }

    # Initial scene: just the hand at neutral
    q0 = neutral_configuration(ctx.model)
    state["robot_handles"] = add_robot_to_scene(server, ctx, q0, cfg)

    # --- GUI ---
    with server.gui.add_folder("Robot"):
        robot_dropdown = server.gui.add_dropdown(
            "Select Robot",
            options=robot_names,
            initial_value=initial_robot,
        )
    with server.gui.add_folder("Status"):
        status_text = server.gui.add_text(
            "Status", initial_value="Idle", disabled=True,
        )

    def _load_robot_grasp(robot_name: str) -> None:
        if state["busy"]:
            return
        state["busy"] = True
        status_text.value = f"Loading {robot_name}..."

        try:
            new_cfg = load_config(args.metric_config, robot_configs[robot_name])
            new_ctx = load_robot(new_cfg.urdf_path, new_cfg)

            # Clear old scene elements
            if state["robot_handles"] is not None:
                for h in state["robot_handles"].values():
                    h.remove()
            if state["sphere_handle"] is not None:
                state["sphere_handle"].remove()
                state["sphere_handle"] = None
            for name in ("/normals", "/contact_dots", "/pinch_axis", "/motion_primitives"):
                server.scene.remove_by_name(name)

            current["name"] = robot_name
            current["cfg"] = new_cfg
            current["ctx"] = new_ctx

            # Load seed data: saved results > full selection
            urdf_name = new_cfg.urdf_path.stem
            q = sc = cp = R_seed = None

            # 1) Saved metric results
            saved_path = _find_saved_result_for_robot(urdf_name)
            if saved_path is not None:
                status_text.value = f"Loading saved result for {robot_name}..."
                result, _meta = load_result(saved_path)
                if int(result.seed_q.shape[0]) == int(new_ctx.model.nq):
                    q = result.seed_q
                    sc = result.seed_centre
                    cp = result.seed_contact_pair
                    R_seed = getattr(result, "seed_frame", np.eye(3))

            # 2) Full seed selection (expensive)
            if q is None:
                status_text.value = f"Selecting best seed for {robot_name}..."
                from karma.seed_selection import select_seed_and_frame
                seed, R_seed, *_ = select_seed_and_frame(new_cfg, new_ctx)
                if not seed.feasible:
                    q0 = neutral_configuration(new_ctx.model)
                    state["robot_handles"] = add_robot_to_scene(
                        server, new_ctx, q0, new_cfg,
                    )
                    status_text.value = f"{robot_name}: no feasible grasp found"
                    state["busy"] = False
                    return
                q = seed.q
                sc = seed.sphere_centre_world
                cp = seed.contact_pair

            update_kinematics(new_ctx, q)

            state["robot_handles"] = add_robot_to_scene(server, new_ctx, q, new_cfg)

            # Draw sphere (more transparent than default so the pinch axis is visible)
            verts, faces = sphere_mesh(new_cfg.sphere_radius_m)
            state["sphere_handle"] = server.scene.add_mesh_simple(
                "/sphere/mesh",
                vertices=verts,
                faces=faces,
                position=sc,
                color=(255, 80, 80),
                opacity=0.4,
            )

            # Compute contact kinematics for both contacts
            ck_thumb = contact_kinematics(
                new_ctx, q, sc, cp[0],
                new_cfg.sphere_radius_m, new_cfg.link_radius_m,
            )
            ck_index = contact_kinematics(
                new_ctx, q, sc, cp[1],
                new_cfg.sphere_radius_m, new_cfg.link_radius_m,
            )

            # Draw contact normals
            _draw_contact_normals(
                server, new_ctx, q, sc, cp,
                new_cfg.sphere_radius_m, new_cfg.link_radius_m,
                length=0.008,
            )

            # Draw contact dots
            _draw_contact_dots(
                server, new_ctx, q, sc, cp,
                new_cfg.sphere_radius_m, new_cfg.link_radius_m,
            )

            # Draw pinch axis (line through both contact points)
            _draw_pinch_axis(
                server,
                ck_thumb.sphere_point_world,
                ck_index.sphere_point_world,
                new_cfg.sphere_radius_m,
            )

            # Draw motion primitives
            pinch_w = pinch_axis_world(
                ck_thumb.normal_world, ck_index.normal_world,
            )
            _draw_motion_primitives(
                server, sc, R_seed, pinch_w,
                new_cfg.voxel_size_m,
                0.15,  # ~8.6 deg, same as metric pipeline
                new_cfg.sphere_radius_m,
            )

            status_text.value = (
                f"{robot_name}: contacts {cp[0]}, {cp[1]}"
            )

        except Exception as e:
            logger.exception("Failed to load robot grasp")
            status_text.value = f"Error: {e}"
        finally:
            state["busy"] = False

    @robot_dropdown.on_update
    def _on_robot_change(_: viser.GuiEvent) -> None:
        selected = robot_dropdown.value
        threading.Thread(
            target=lambda: _load_robot_grasp(selected), daemon=True,
        ).start()

    # Auto-load initial robot
    threading.Thread(
        target=lambda: _load_robot_grasp(initial_robot), daemon=True,
    ).start()

    logger.info(
        "Grasp viewer running at http://%s:%d", cfg.viewer_host, cfg.viewer_port,
    )

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
