#!/usr/bin/env python3
"""Interactive viser visualization for KaRMA.

Controls:
  - "Compute Metric" button: runs the full pipeline, writes workspace/current.yaml + workspace/current.pkl
  - Voxel slider: select a reached voxel to inspect
  - Rotation slider: select an orientation bin at the selected voxel
"""
from __future__ import annotations

import argparse
import logging
import threading
import time
from pathlib import Path

import numpy as np
import viser
from scipy.spatial.transform import Rotation, Slerp

from karma.config import load_config, MetricConfig
from karma.robot import load_robot, update_kinematics, neutral_configuration, capsule_endpoints_world
from karma.results import write_outputs, MetricResult, State
from karma.metric import compute_metric
from karma.viser_scene import (
    add_robot_to_scene, update_robot_poses, add_sphere_to_scene,
    add_voxel_markers, highlight_voxel, add_contact_normals,
    add_sphere_frame_to_scene,
)
from karma.orientation import healpix_npix_lines, healpix_bin_to_axis
from karma.storage import (
    list_saved_results, load_result, save_result,
)

logger = logging.getLogger(__name__)


def scan_robot_configs(robots_dir: Path = Path("robots")) -> dict[str, Path]:
    """Scan for available robot config YAML files.

    Returns dict mapping display name -> config path.
    """
    configs = {}
    if not robots_dir.exists():
        return configs

    for yaml_path in sorted(robots_dir.glob("robot_*.yaml")):
        stem = yaml_path.stem
        if stem == "robot_template" or stem.endswith(".tuned"):
            continue  # onboarding template / scratch tuning output, not a shipped hand
        # Extract robot name from filename (e.g., robot_leap.yaml -> leap)
        name = stem.replace("robot_", "")
        configs[name] = yaml_path

    return configs

def _canonicalize_line_axis(axis: np.ndarray) -> np.ndarray:
    """Canonicalize a line direction (axis ~ -axis) for stable comparisons."""
    a = np.asarray(axis, dtype=float).ravel()
    n = float(np.linalg.norm(a))
    if n < 1e-12:
        return np.array([0.0, 0.0, 1.0])
    a = a / n
    # Match karma.orientation._canonicalize_axis() convention.
    if a[2] < 0 or (a[2] == 0 and a[1] < 0) or (a[2] == 0 and a[1] == 0 and a[0] < 0):
        a = -a
    return a


def _line_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between two line directions (treating a ~ -a)."""
    aa = _canonicalize_line_axis(a)
    bb = _canonicalize_line_axis(b)
    c = float(np.clip(abs(np.dot(aa, bb)), -1.0, 1.0))
    return float(np.arccos(c))


def _capsule_bounds(ctx) -> tuple[np.ndarray, float]:
    """World-frame centre and bounding radius of the robot's capsules at the
    current configuration. Assumes kinematics are already up to date."""
    pts = []
    for name in ctx.link_capsules:
        a, b = capsule_endpoints_world(ctx, name)
        pts.append(a)
        pts.append(b)
    pts = np.asarray(pts, dtype=float)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    centre = 0.5 * (lo + hi)
    radius = 0.5 * float(np.linalg.norm(hi - lo))
    return centre, radius


def _frame_camera(camera, centre: np.ndarray, radius: float) -> None:
    """Point a camera at `centre`, backed off far enough to frame `radius`.

    Hands differ a lot in where they sit (a bare hand near the origin, a hand on
    a wrist stub tens of centimetres up), so framing each one beats viser's
    default fit over the whole scene, which starts zoomed far out."""
    centre = np.asarray(centre, dtype=float)
    view = np.array([0.5, -1.0, 0.35])
    view /= np.linalg.norm(view)
    dist = max(radius, 0.03) * 2.6
    camera.position = tuple(centre + view * dist)
    camera.look_at = tuple(centre)
    camera.up_direction = (0.0, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metric-config",
        default="karma_config.yaml",
        help="Global KaRMA config (shared across robots)",
    )
    parser.add_argument(
        "--result", "-r",
        help="Load a specific result file (e.g., workspace/current.pkl or results/16_hand_batch/robot_shadowhand/current.pkl)",
    )
    parser.add_argument(
        "--results-dir",
        default="results/16_hand_batch",
        help="Directory of batch/paper runs (robot_<name>/current.pkl); offered per robot "
             "in the dropdown and auto-loaded when you switch robots.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Scan for available robot configs
    robot_configs = scan_robot_configs()
    if not robot_configs:
        logger.error("No robot configs found in robots/ directory")
        return

    robot_names = list(robot_configs.keys())
    initial_robot = robot_names[0]

    # Load initial config and robot
    cfg = load_config(args.metric_config, robot_configs[initial_robot])
    ctx = load_robot(cfg.urdf_path, cfg)

    # Mutable references for robot switching
    current_robot = {"name": initial_robot, "cfg": cfg, "ctx": ctx}

    server = viser.ViserServer(host=cfg.viewer_host, port=cfg.viewer_port)

    # State
    state: dict = {
        "busy": False,
        "result": None,
        "robot_handles": None,
        "sphere_handle": None,
        "sphere_frame_handles": None,
        "voxel_handles": None,
        "highlight_handle": None,
        "sorted_voxels": [],
        "ordered_rotation_states": [],
    }

    # Initial scene
    q0 = neutral_configuration(ctx.model)
    state["robot_handles"] = add_robot_to_scene(server, ctx, q0, cfg)
    state["cam_target"] = _capsule_bounds(ctx)

    @server.on_client_connect
    def _frame_on_connect(client: viser.ClientHandle) -> None:
        _frame_camera(client.camera, *state["cam_target"])

    # GUI - Robot Selection
    with server.gui.add_folder("Robot"):
        robot_dropdown = server.gui.add_dropdown(
            "Select Robot",
            options=robot_names,
            initial_value=initial_robot,
        )

    def _switch_robot(new_robot_name: str) -> None:
        """Switch to a different robot, clearing and rebuilding the scene."""
        if state["busy"]:
            return
        if new_robot_name == current_robot["name"]:
            return

        state["busy"] = True
        logger.info("Switching to robot: %s", new_robot_name)

        try:
            # Load new config and robot
            new_cfg = load_config(args.metric_config, robot_configs[new_robot_name])
            new_ctx = load_robot(new_cfg.urdf_path, new_cfg)

            # Clear existing scene elements
            if state["robot_handles"] is not None:
                for h in state["robot_handles"].values():
                    h.remove()
                state["robot_handles"] = None

            if state["sphere_handle"] is not None:
                state["sphere_handle"].remove()
                state["sphere_handle"] = None

            if state["sphere_frame_handles"] is not None:
                for h in state["sphere_frame_handles"].values():
                    h.remove()
                state["sphere_frame_handles"] = None

            if state["voxel_handles"] is not None:
                for h in state["voxel_handles"].values():
                    h.remove()
                state["voxel_handles"] = None

            if state["highlight_handle"] is not None:
                state["highlight_handle"].remove()
                state["highlight_handle"] = None

            # Clear result
            state["result"] = None
            state["sorted_voxels"] = []

            # Update current robot references
            current_robot["name"] = new_robot_name
            current_robot["cfg"] = new_cfg
            current_robot["ctx"] = new_ctx

            # Add new robot to scene
            new_q0 = neutral_configuration(new_ctx.model)
            state["robot_handles"] = add_robot_to_scene(server, new_ctx, new_q0, new_cfg)
            state["cam_target"] = _capsule_bounds(new_ctx)
            for client in server.get_clients().values():
                _frame_camera(client.camera, *state["cam_target"])

            # Update status
            status_text.value = f"Switched to {new_robot_name}"
            info_text.value = "No results loaded"
            voxel_slider.value = 0
            voxel_slider.max = 1

            logger.info("Successfully switched to robot: %s", new_robot_name)

        except Exception as e:
            logger.exception("Failed to switch robot")
            status_text.value = f"Switch failed: {e}"
        finally:
            state["busy"] = False

    def _switch_robot_and_refresh(new_robot_name: str) -> None:
        """Switch robot and refresh the saved results dropdown."""
        _switch_robot(new_robot_name)
        # Refresh saved results dropdown to show only results for the new robot
        nonlocal saved_options
        saved_options = _get_saved_options()
        result_dropdown.options = list(saved_options.keys())
        result_dropdown.value = "-- Compute New --"
        # Auto-load this robot's paper/batch run, if present.
        if _load_paper_run(new_robot_name) and "paper run (16_hand_batch)" in saved_options:
            result_dropdown.value = "paper run (16_hand_batch)"

    @robot_dropdown.on_update
    def _on_robot_change(_: viser.GuiEvent) -> None:
        selected = robot_dropdown.value
        threading.Thread(target=lambda: _switch_robot_and_refresh(selected), daemon=True).start()

    def _make_slerp_R(R0: np.ndarray, R1: np.ndarray) -> Slerp:
        """Create a slerp interpolator between two rotation matrices."""
        rot_pair = Rotation.from_matrix(
            np.stack([
                np.asarray(R0, dtype=float).reshape(3, 3),
                np.asarray(R1, dtype=float).reshape(3, 3),
            ], axis=0)
        )
        return Slerp([0.0, 1.0], rot_pair)

    def _update_sphere_pose(centre_world: np.ndarray, R_sphere: np.ndarray | None) -> None:
        """Update sphere mesh + attached coordinate frame."""
        if R_sphere is None:
            R_sphere = np.eye(3)

        if state["sphere_frame_handles"] is None:
            state["sphere_frame_handles"] = add_sphere_frame_to_scene(
                server,
                axis_length=float(current_robot["cfg"].sphere_radius_m) * 1.5,
            )

        centre_arr = np.asarray(centre_world, dtype=float).reshape(3)
        R_arr = np.asarray(R_sphere, dtype=float).reshape(3, 3)

        if state["sphere_handle"] is not None:
            state["sphere_handle"].position = centre_arr

        for h in state["sphere_frame_handles"].values():
            h.position = centre_arr

        try:
            quat_wxyz = Rotation.from_matrix(R_arr).as_quat(scalar_first=True)
        except Exception:
            # If R_sphere is malformed or numerically ill-conditioned, keep the prior orientation.
            return

        if state["sphere_handle"] is not None:
            state["sphere_handle"].wxyz = quat_wxyz
        for h in state["sphere_frame_handles"].values():
            h.wxyz = quat_wxyz

    def _paper_run_path(robot_name: str):
        """Path to this robot's batch/paper run (results-dir/robot_<name>/current.pkl), if it exists."""
        p = Path(args.results_dir) / f"robot_{robot_name}" / "current.pkl"
        return p if p.exists() else None

    def _score_summary(result: MetricResult) -> str:
        return (f"KaRMA-T={result.translational_score:.4f}, "
                f"KaRMA-R={result.global_rotational_score:.3f}")

    def _load_paper_run(robot_name: str) -> bool:
        """Load and display this robot's bundled paper run, if present and compatible."""
        paper_pkl = _paper_run_path(robot_name)
        if paper_pkl is None:
            return False
        try:
            result, _metadata = load_result(paper_pkl)
            if int(result.seed_q.shape[0]) != int(current_robot["ctx"].model.nq):
                return False
            state["result"] = result
            _display_result(result)
            status_text.value = (
                f"Paper run: {result.n_voxels_reached} voxels, {_score_summary(result)}"
            )
            info_text.value = f"Auto-loaded paper run for {robot_name}"
            return True
        except Exception as e:
            logger.warning("Failed to auto-load paper run for %s: %s", robot_name, e)
            return False

    def _get_saved_options() -> dict[str, str]:
        """Get dropdown options: current + saved results + 'Compute New'.

        Only shows results matching the currently selected robot.
        """
        saved = list_saved_results()
        options = {"-- Compute New --": ""}

        # Get current robot's URDF name for filtering
        current_urdf_name = current_robot["cfg"].urdf_path.stem

        current_pkl = current_robot["cfg"].out_dir / "current.pkl"
        if current_pkl.exists():
            # Check if current.pkl matches the current robot
            try:
                _, metadata = load_result(current_pkl)
                urdf_path = metadata.get("urdf_path")
                if urdf_path is None or Path(urdf_path).stem == current_urdf_name:
                    options[f"{current_robot['cfg'].out_dir.name}/current.pkl (latest)"] = str(current_pkl)
            except Exception as e:
                logger.debug("Skipping unreadable %s: %s", current_pkl, e)

        paper_pkl = _paper_run_path(current_robot["name"])
        if paper_pkl is not None:
            options["paper run (16_hand_batch)"] = str(paper_pkl)

        for s in saved:
            # Only show results matching the current robot
            if s.urdf_name == current_urdf_name:
                label = f"{s.name} ({s.n_voxels}v/{s.n_states}s)"
                options[label] = str(s.filepath)
        return options

    def _display_result(result: MetricResult) -> None:
        """Update scene to display a metric result."""
        cfg = current_robot["cfg"]
        ctx = current_robot["ctx"]

        # Update robot
        update_kinematics(ctx, result.seed_q)
        update_robot_poses(ctx, result.seed_q, state["robot_handles"])

        # Sphere
        if state["sphere_handle"] is not None:
            state["sphere_handle"].remove()
        if state["sphere_frame_handles"] is not None:
            for h in state["sphere_frame_handles"].values():
                h.remove()
            state["sphere_frame_handles"] = None
        state["sphere_handle"] = add_sphere_to_scene(
            server, result.seed_centre, cfg.sphere_radius_m,
        )
        _update_sphere_pose(result.seed_centre, np.eye(3))

        # Voxel markers
        if state["voxel_handles"] is not None:
            for h in state["voxel_handles"].values():
                h.remove()
        total_bins = healpix_npix_lines(cfg.healpix_nside)
        seed_frame = getattr(result, 'seed_frame', np.eye(3))
        state["voxel_handles"] = add_voxel_markers(
            server, result.voxel_ori_bins, result.seed_centre,
            cfg.voxel_size_m, total_bins, R_seed=seed_frame,
        )

        # Contact normals at seed
        add_contact_normals(
            server, ctx, result.seed_q, result.seed_centre,
            result.seed_contact_pair,
            cfg.sphere_radius_m, cfg.link_radius_m,
        )

        # Update slider
        sorted_v = sorted(result.voxel_ori_bins.keys())
        state["sorted_voxels"] = sorted_v
        if sorted_v:
            voxel_slider.max = len(sorted_v) - 1
            voxel_slider.value = 0

    # GUI - Saved Results
    saved_options = _get_saved_options()
    with server.gui.add_folder("Saved Results"):
        result_dropdown = server.gui.add_dropdown(
            "Select Result",
            options=list(saved_options.keys()),
            initial_value="-- Compute New --",
        )
        btn_load = server.gui.add_button("Load Selected")
        btn_save = server.gui.add_button("Save Current Result")
        btn_refresh = server.gui.add_button("Refresh List")

    # GUI - Metric
    with server.gui.add_folder("Metric"):
        btn_compute = server.gui.add_button("Compute Metric")
        status_text = server.gui.add_text("Status", initial_value="Idle", disabled=True)

    with server.gui.add_folder("Inspect"):
        voxel_slider = server.gui.add_slider(
            "Voxel Index", min=0, max=1, step=1, initial_value=0,
        )
        rotation_slider = server.gui.add_slider(
            "Rotation Index", min=0, max=1, step=1, initial_value=0,
        )
        btn_play_path = server.gui.add_button("Play Path")
        info_text = server.gui.add_text("Info", initial_value="No results yet", disabled=True)

    def _show_state(node, result):
        """Update robot, sphere, and contact normals for a given state node."""
        cfg = current_robot["cfg"]
        ctx = current_robot["ctx"]
        update_kinematics(ctx, node.q)
        update_robot_poses(ctx, node.q, state["robot_handles"])
        _update_sphere_pose(node.centre_world, getattr(node, "R_sphere", None))
        add_contact_normals(
            server, ctx, node.q, node.centre_world, node.contact_pair,
            cfg.sphere_radius_m, cfg.link_radius_m,
        )

    def _run_compute() -> None:
        if state["busy"]:
            return
        state["busy"] = True
        status_text.value = "Computing..."

        cfg = current_robot["cfg"]
        ctx = current_robot["ctx"]

        try:
            result = compute_metric(
                cfg, ctx,
                n_rotation_workers=cfg.rotation_workers,
            )
            out_path = write_outputs(cfg, result)
            state["result"] = result

            _display_result(result)

            status_text.value = (
                f"Done: {result.n_voxels_reached} voxels, "
                f"{result.n_states_reached} states, {_score_summary(result)}"
            )
            info_text.value = f"Written to {out_path}"

        except Exception as e:
            logger.exception("Compute failed")
            status_text.value = f"Error: {e}"
        finally:
            state["busy"] = False

    @btn_compute.on_click
    def _on_compute(_: viser.GuiEvent) -> None:
        threading.Thread(target=_run_compute, daemon=True).start()

    @btn_load.on_click
    def _on_load(_: viser.GuiEvent) -> None:
        if state["busy"]:
            return
        selected = result_dropdown.value
        filepath = saved_options.get(selected, "")
        if not filepath:
            status_text.value = "Select a saved result first"
            return

        state["busy"] = True
        status_text.value = "Loading..."

        try:
            result, _metadata = load_result(filepath)
            state["result"] = result
            _display_result(result)

            status_text.value = (
                f"Loaded: {result.n_voxels_reached} voxels, "
                f"{result.n_states_reached} states, {_score_summary(result)}"
            )
            info_text.value = f"Loaded from {selected}"
        except Exception as e:
            logger.exception("Load failed")
            status_text.value = f"Load error: {e}"
        finally:
            state["busy"] = False

    @btn_save.on_click
    def _on_save(_: viser.GuiEvent) -> None:
        if state["result"] is None:
            status_text.value = "No result to save - compute first"
            return
        if state["busy"]:
            return

        state["busy"] = True
        status_text.value = "Saving..."

        try:
            filepath = save_result(state["result"], current_robot["cfg"].urdf_path)
            status_text.value = f"Saved to {filepath.name}"
            # Refresh dropdown
            nonlocal saved_options
            saved_options = _get_saved_options()
            result_dropdown.options = list(saved_options.keys())
        except Exception as e:
            logger.exception("Save failed")
            status_text.value = f"Save error: {e}"
        finally:
            state["busy"] = False

    @btn_refresh.on_click
    def _on_refresh(_: viser.GuiEvent) -> None:
        nonlocal saved_options
        saved_options = _get_saved_options()
        result_dropdown.options = list(saved_options.keys())
        status_text.value = f"Found {len(saved_options) - 1} saved results"

    def _ordered_states_at_voxel(result: MetricResult, voxel: tuple, cfg: MetricConfig) -> list:
        """Collect and order all states at a voxel for smooth browsing."""
        voxel_states = [
            (s, node)
            for s, node in result.state_nodes.items()
            if node.voxel == voxel
        ]
        if len(voxel_states) <= 1:
            return voxel_states

        axes = [
            healpix_bin_to_axis(node.ori_bin, cfg.healpix_nside)
            for _s, node in voxel_states
        ]
        remaining = set(range(len(voxel_states)))
        start_i = int(np.argmin([float(np.linalg.norm(node.q - result.seed_q)) for _s, node in voxel_states]))
        order: list[int] = [start_i]
        remaining.remove(start_i)
        while remaining:
            last = order[-1]
            last_q = voxel_states[last][1].q
            last_axis = axes[last]

            def _key(j: int, last_axis=last_axis, last_q=last_q) -> tuple[float, float]:
                ang = _line_angle(last_axis, axes[j])
                dq = float(np.linalg.norm(voxel_states[j][1].q - last_q))
                return (ang, dq)

            nxt = min(remaining, key=_key)
            order.append(nxt)
            remaining.remove(nxt)

        return [voxel_states[i] for i in order]

    @voxel_slider.on_update
    def _on_voxel_change(_: viser.GuiEvent) -> None:
        result: MetricResult | None = state["result"]
        if result is None:
            return
        idx = int(voxel_slider.value)
        if idx >= len(state["sorted_voxels"]):
            return

        cfg = current_robot["cfg"]
        voxel = state["sorted_voxels"][idx]
        bins = result.voxel_ori_bins.get(voxel, set())

        # Highlight selected voxel
        if state["highlight_handle"] is not None:
            state["highlight_handle"].remove()
            state["highlight_handle"] = None
        seed_frame = getattr(result, 'seed_frame', np.eye(3))
        state["highlight_handle"] = highlight_voxel(
            server, voxel, result.seed_centre, cfg.voxel_size_m,
            R_seed=seed_frame,
        )

        # Order states at this voxel and update rotation slider
        ordered = _ordered_states_at_voxel(result, voxel, cfg)
        state["ordered_rotation_states"] = ordered
        rotation_slider.max = max(1, len(ordered) - 1)
        rotation_slider.value = 0

        # Show first state
        if ordered:
            _show_state(ordered[0][1], result)

        info_text.value = (
            f"Voxel {voxel}: {len(bins)}/{healpix_npix_lines(cfg.healpix_nside)} ori bins"
        )

    @rotation_slider.on_update
    def _on_rotation_change(_: viser.GuiEvent) -> None:
        result: MetricResult | None = state["result"]
        if result is None:
            return
        ordered = state.get("ordered_rotation_states", [])
        if not ordered:
            return
        rot_idx = int(rotation_slider.value)
        if rot_idx >= len(ordered):
            return

        cfg = current_robot["cfg"]
        _s, node = ordered[rot_idx]
        _show_state(node, result)

        voxel = node.voxel
        bins = result.voxel_ori_bins.get(voxel, set())
        info_text.value = (
            f"Voxel {voxel}: ori {rot_idx+1}/{len(ordered)}, "
            f"{len(bins)}/{healpix_npix_lines(cfg.healpix_nside)} bins"
        )

    def _play_path() -> None:
        """Animate the path from seed to the currently selected voxel with smooth interpolation."""
        result: MetricResult | None = state["result"]
        if result is None or state["busy"]:
            return
        state["busy"] = True

        cfg = current_robot["cfg"]
        ctx = current_robot["ctx"]

        idx = int(voxel_slider.value)
        if idx >= len(state["sorted_voxels"]):
            state["busy"] = False
            return

        voxel = state["sorted_voxels"][idx]

        # Find a state at this voxel (pick the first one)
        target_state: State | None = None
        for s, node in result.state_nodes.items():
            if node.voxel == voxel:
                target_state = s
                break

        if target_state is None:
            status_text.value = "No state found at this voxel"
            state["busy"] = False
            return

        # Backtrack from target to seed using parent pointers
        path: list[State] = []
        current: State | None = target_state
        while current is not None:
            path.append(current)
            parent = result.state_nodes[current].parent
            current = parent

        # Reverse to get path from seed to target
        path = path[::-1]

        n_interp = 8  # interpolation frames between each BFS state
        status_text.value = f"Playing path ({len(path)} steps, smooth)..."

        for i in range(len(path)):
            node_curr = result.state_nodes[path[i]]

            if i == 0:
                # Show first state directly
                _show_state(node_curr, result)
                info_text.value = f"Step {i+1}/{len(path)}: voxel {node_curr.voxel}"
                time.sleep(0.05)
                continue

            # Interpolate from previous state to current state
            node_prev = result.state_nodes[path[i - 1]]
            q_prev = node_prev.q
            q_curr = node_curr.q
            p_prev = node_prev.centre_world
            p_curr = node_curr.centre_world
            R_prev = getattr(node_prev, "R_sphere", None)
            R_curr = getattr(node_curr, "R_sphere", None)
            if R_prev is None:
                R_prev = np.eye(3)
            if R_curr is None:
                R_curr = np.eye(3)
            slerp_R = _make_slerp_R(R_prev, R_curr)

            for k in range(1, n_interp + 1):
                t = k / n_interp
                q_interp = (1 - t) * q_prev + t * q_curr
                p_interp = (1 - t) * p_prev + t * p_curr
                R_interp = slerp_R([t]).as_matrix()[0]

                update_kinematics(ctx, q_interp)
                update_robot_poses(ctx, q_interp, state["robot_handles"])
                _update_sphere_pose(p_interp, R_interp)

                # Only update contact normals at the final frame of each segment
                if k == n_interp:
                    add_contact_normals(
                        server, ctx, q_interp, p_interp, node_curr.contact_pair,
                        cfg.sphere_radius_m, cfg.link_radius_m,
                    )
                    info_text.value = f"Step {i+1}/{len(path)}: voxel {node_curr.voxel}"

                time.sleep(0.02)

        status_text.value = "Path playback done"
        state["busy"] = False

    @btn_play_path.on_click
    def _on_play_path(_: viser.GuiEvent) -> None:
        threading.Thread(target=_play_path, daemon=True).start()

    # Auto-load: either from --result arg or latest run_metric output.
    def _autoload_result() -> None:
        cfg = current_robot["cfg"]
        ctx = current_robot["ctx"]

        # If --result was provided, load that specific file
        if args.result:
            result_path = Path(args.result)
            if not result_path.exists():
                logger.error("Result file not found: %s", result_path)
                status_text.value = f"Error: file not found: {args.result}"
                return
            try:
                result, metadata = load_result(result_path)
                # Verify robot compatibility
                if int(result.seed_q.shape[0]) != int(ctx.model.nq):
                    logger.error(
                        "Robot mismatch: result has %d joints, loaded robot has %d",
                        int(result.seed_q.shape[0]),
                        int(ctx.model.nq),
                    )
                    status_text.value = "Error: robot config mismatch"
                    return
                state["result"] = result
                _display_result(result)
                status_text.value = (
                    f"Loaded: {result.n_voxels_reached} voxels, "
                    f"{result.n_states_reached} states, {_score_summary(result)}"
                )
                info_text.value = f"Loaded from {args.result}"
                return
            except Exception as e:
                logger.exception("Failed to load %s", args.result)
                status_text.value = f"Error: {e}"
                return

        # Prefer this robot's paper/batch run if present.
        if _load_paper_run(current_robot["name"]):
            return

        # Otherwise, try to auto-load current.pkl
        current_pkl = cfg.out_dir / "current.pkl"
        if not current_pkl.exists():
            return
        try:
            result, metadata = load_result(current_pkl)
            urdf_path = metadata.get("urdf_path")
            if urdf_path is not None and Path(urdf_path).stem != cfg.urdf_path.stem:
                logger.info(
                    "Skipping auto-load of %s (URDF mismatch: %s != %s)",
                    current_pkl,
                    Path(urdf_path).stem,
                    cfg.urdf_path.stem,
                )
                return
            if int(result.seed_q.shape[0]) != int(ctx.model.nq):
                logger.info(
                    "Skipping auto-load of %s (q size mismatch: %d != %d)",
                    current_pkl,
                    int(result.seed_q.shape[0]),
                    int(ctx.model.nq),
                )
                return
            state["result"] = result
            _display_result(result)
            status_text.value = (
                f"Loaded: {result.n_voxels_reached} voxels, "
                f"{result.n_states_reached} states, {_score_summary(result)}"
            )
            info_text.value = f"Auto-loaded {cfg.out_dir.name}/current.pkl"
        except Exception as e:
            logger.warning("Failed to auto-load %s: %s", current_pkl, e)

    threading.Thread(target=_autoload_result, daemon=True).start()

    logger.info("Viser server running at http://%s:%d", cfg.viewer_host, cfg.viewer_port)

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
