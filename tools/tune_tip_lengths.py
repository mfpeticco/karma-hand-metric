#!/usr/bin/env python3
"""Interactive tip-length tuner for a robot hand.

KaRMA models each finger link as a capsule built from the URDF joint tree. The
joint tree stops at the last joint, not at the physical fingertip, so the last
link of each finger is extended by a `tip_length_m` value you set in the robot
YAML. Getting these right matters: a tip length that is too short leaves the
capsule ending short of the real fingertip, and the hand then reaches far fewer
voxels than it should.

This tool shows the capsule skeleton next to the hand's real geometry so you can
set each tip length by eye, then writes the values back to a robot YAML. When the
URDF's visual meshes are on disk (as they are for a hand you are adding to your
own copy of the repo) they are drawn translucently behind the capsules; the
bundled example hands ship without meshes, so for those you see the capsules
alone and tune against the joint frames.

Run from the repo root:
    python tools/tune_tip_lengths.py --config robots/robot_leap.yaml

Each slider sets one fingertip's total tip length in millimetres. "Save" writes
the values back to YAML (a sibling <name>.tuned.yaml by default, or the input
file itself with --in-place). "Reset" restores the values the file loaded with.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# Repo root (this script lives in tools/); make `karma` importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from karma.config import load_config
from karma.robot import (
    load_robot, neutral_configuration, update_kinematics,
)
from karma.viser_scene import add_robot_to_scene

logger = logging.getLogger(__name__)

ROBOT_PREFIX = "/robot"
VISUAL_PREFIX = "/visual"
MESH_COLOR = (200, 200, 200)
MESH_OPACITY = 0.35


# ─────────────────────────────────────────────────────────────────────────────
# URDF visual meshes (best-effort overlay)
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_mesh_path(urdf_dir: Path, filename: str) -> Path | None:
    """Resolve a URDF <mesh filename> to a file on disk, or None if absent."""
    if filename.startswith("package://"):
        rest = filename[len("package://"):]
        after_pkg = rest.split("/", 1)[1] if "/" in rest else rest
        candidates = [urdf_dir / after_pkg, urdf_dir.parent / after_pkg, urdf_dir / rest]
    else:
        candidates = [urdf_dir / filename]
    for c in candidates:
        if c.is_file():
            return c
    return None


def _load_visual_meshes(urdf_path: str | Path, ctx) -> list[tuple[np.ndarray, np.ndarray]]:
    """Load the URDF's visual meshes, placed in KaRMA's canonicalized world frame.

    Returns a list of (vertices, faces). Every mesh is best-effort: a link whose
    frame is missing, or a mesh file that is absent or unreadable, is skipped.
    Returns an empty list when the URDF has no resolvable visual meshes.
    """
    try:
        import trimesh
        import yourdfpy
    except ImportError as e:
        logger.warning("Visual overlay needs trimesh and yourdfpy (%s); showing capsules only.", e)
        return []

    urdf_path = Path(urdf_path)
    urdf_dir = urdf_path.parent
    try:
        robot = yourdfpy.URDF.load(str(urdf_path), load_meshes=False, build_scene_graph=False)
    except Exception as e:  # noqa: BLE001 - parsing is best-effort
        logger.warning("Could not parse URDF for the mesh overlay (%s); showing capsules only.", e)
        return []

    meshes: list[tuple[np.ndarray, np.ndarray]] = []
    n_skipped = 0
    for link_name, link in robot.link_map.items():
        if not ctx.model.existFrame(link_name):
            continue
        oMf = ctx.data.oMf[ctx.model.getFrameId(link_name)]
        T_link = np.eye(4)
        T_link[:3, :3] = np.asarray(oMf.rotation)
        T_link[:3, 3] = np.asarray(oMf.translation).ravel()

        for vis in getattr(link, "visuals", []) or []:
            mesh_spec = getattr(getattr(vis, "geometry", None), "mesh", None)
            if mesh_spec is None or not getattr(mesh_spec, "filename", None):
                continue
            path = _resolve_mesh_path(urdf_dir, mesh_spec.filename)
            if path is None:
                n_skipped += 1
                continue
            try:
                m = trimesh.load(str(path), force="mesh", process=False)
                if hasattr(m, "dump"):  # a Scene
                    m = m.dump(concatenate=True)
                verts = np.asarray(m.vertices, dtype=np.float64)
                faces = np.asarray(m.faces, dtype=np.uint32)
            except Exception:  # noqa: BLE001 - one bad mesh should not stop the tool
                n_skipped += 1
                continue
            if verts.size == 0 or faces.size == 0:
                continue

            scale = getattr(mesh_spec, "scale", None)
            if scale is not None:
                verts = verts * np.asarray(scale, dtype=np.float64)
            origin = vis.origin if getattr(vis, "origin", None) is not None else np.eye(4)
            T = T_link @ np.asarray(origin, dtype=np.float64)
            world = (T[:3, :3] @ verts.T).T + T[:3, 3]
            meshes.append((world.astype(np.float32), faces))

    if n_skipped:
        logger.info("Skipped %d visual mesh(es) that were absent or unreadable.", n_skipped)
    return meshes


# ─────────────────────────────────────────────────────────────────────────────
# Tip capsules
# ─────────────────────────────────────────────────────────────────────────────

def _tunable_tips(ctx, link_radius_m: float) -> dict[str, dict]:
    """Fingertip capsules that carry a tip extension, keyed by link name.

    For each, record the unit axis of the extension and the total tip length
    (extension magnitude plus the capsule radius, matching how tip_length_m is
    defined in the robot YAML).
    """
    tips: dict[str, dict] = {}
    for name, cap in ctx.link_capsules.items():
        if cap.tip_offset_xyz_m is None:
            continue
        offset = np.asarray(cap.tip_offset_xyz_m, dtype=np.float64)
        mag = float(np.linalg.norm(offset))
        if mag < 1e-9:
            continue
        tips[name] = {
            "axis": offset / mag,
            "length_m": mag + link_radius_m,
        }
    return tips


# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────

def _save_tip_lengths(src_yaml: Path, out_yaml: Path, tip_lengths_m: dict[str, float]) -> None:
    """Write tip_length_m values into a copy of the robot YAML."""
    with open(src_yaml) as f:
        data = yaml.safe_load(f)

    links = data.get("links") or {}  # a bare "links:" key parses to None
    data["links"] = links
    for link_name, length in tip_lengths_m.items():
        entry = links.setdefault(link_name, {})
        entry["tip_length_m"] = round(float(length), 6)
        entry.pop("tip_offset_xyz_m", None)  # superseded by the scalar length

    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(out_yaml, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactively tune a hand's fingertip tip lengths.",
    )
    parser.add_argument(
        "--config", default="robots/robot_shadowhand.yaml",
        help="Robot config YAML to tune, e.g. robots/robot_leap.yaml",
    )
    parser.add_argument(
        "--metric-config", default="karma_config.yaml",
        help="Global KaRMA config (for the capsule radius and viewer host/port)",
    )
    parser.add_argument(
        "--out", default=None,
        help="Where to write tuned values (default: a sibling <name>.tuned.yaml)",
    )
    parser.add_argument(
        "--in-place", action="store_true",
        help="Overwrite the input config (a .bak backup is made on first save)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    import viser

    config_path = Path(args.config)
    if args.out:
        out_path = Path(args.out)
    elif args.in_place:
        out_path = config_path
    else:
        out_path = config_path.parent / f"{config_path.stem}.tuned.yaml"

    cfg = load_config(args.metric_config, args.config)
    ctx = load_robot(cfg.urdf_path, cfg)
    q = neutral_configuration(ctx.model)
    update_kinematics(ctx, q)

    tips = _tunable_tips(ctx, cfg.link_radius_m)
    if not tips:
        logger.warning("No fingertip capsules with a tip length were found in %s.", args.config)
    original_lengths = {name: info["length_m"] for name, info in tips.items()}
    tuned_lengths = dict(original_lengths)

    server = viser.ViserServer(host=cfg.viewer_host, port=cfg.viewer_port)
    handles = add_robot_to_scene(server, ctx, q, cfg, prefix=ROBOT_PREFIX)

    visual_meshes = _load_visual_meshes(cfg.urdf_path, ctx)

    def _draw_visuals() -> None:
        for i, (verts, faces) in enumerate(visual_meshes):
            server.scene.add_mesh_simple(
                f"{VISUAL_PREFIX}/{i}",
                vertices=verts, faces=faces,
                color=MESH_COLOR, opacity=MESH_OPACITY,
            )

    if visual_meshes:
        _draw_visuals()

    def _redraw_capsules() -> None:
        nonlocal handles
        server.scene.remove_by_name(ROBOT_PREFIX)
        handles = add_robot_to_scene(server, ctx, q, cfg, prefix=ROBOT_PREFIX)

    # ── GUI ──
    with server.gui.add_folder("Hand"):
        server.gui.add_text("Config", initial_value=config_path.name, disabled=True)
        mesh_note = (
            f"{len(visual_meshes)} visual meshes" if visual_meshes
            else "no meshes on disk (capsules only)"
        )
        server.gui.add_text("Meshes", initial_value=mesh_note, disabled=True)

    if visual_meshes:
        with server.gui.add_folder("Overlay"):
            show_meshes = server.gui.add_checkbox("Show URDF meshes", initial_value=True)

            @show_meshes.on_update
            def _(_=None) -> None:
                if show_meshes.value:
                    _draw_visuals()
                else:
                    server.scene.remove_by_name(VISUAL_PREFIX)

    slider_by_link: dict[str, object] = {}
    with server.gui.add_folder("Tip lengths (mm)"):
        for name, info in tips.items():
            length_mm = info["length_m"] * 1e3
            slider = server.gui.add_slider(
                name,
                min=round(cfg.link_radius_m * 1e3, 1),
                max=round(max(0.10, info["length_m"] * 2.5) * 1e3, 1),
                step=0.5,
                initial_value=round(length_mm, 1),
            )
            slider_by_link[name] = slider

            def _make_cb(link_name: str, axis: np.ndarray, sl):
                def _cb(_=None) -> None:
                    new_len_m = sl.value * 1e-3
                    axis_len = max(0.0, new_len_m - cfg.link_radius_m)
                    ctx.link_capsules[link_name].tip_offset_xyz_m[:] = axis * axis_len
                    tuned_lengths[link_name] = new_len_m
                    _redraw_capsules()
                return _cb

            slider.on_update(_make_cb(name, info["axis"], slider))

    with server.gui.add_folder("Save"):
        status = server.gui.add_text("Status", initial_value="No changes saved", disabled=True)
        btn_save = server.gui.add_button("Save")
        btn_reset = server.gui.add_button("Reset")

        @btn_save.on_click
        def _(_=None) -> None:
            if args.in_place and out_path == config_path:
                backup = config_path.with_suffix(config_path.suffix + ".bak")
                if not backup.exists():
                    shutil.copy2(config_path, backup)
            try:
                _save_tip_lengths(config_path, out_path, tuned_lengths)
                status.value = f"Saved {len(tuned_lengths)} tip lengths to {out_path}"
                logger.info("Saved tip lengths to %s", out_path)
            except Exception as e:  # noqa: BLE001 - surface the error in the UI
                status.value = f"Save failed: {e}"
                logger.exception("Save failed")

        @btn_reset.on_click
        def _(_=None) -> None:
            for name, info in tips.items():
                orig = original_lengths[name]
                # Assigning .value fires the slider callback, which quantizes to the
                # slider step; restore the exact values *after* so a following Save
                # writes what the file loaded rather than a rounded copy.
                slider_by_link[name].value = round(orig * 1e3, 1)
                tuned_lengths[name] = orig
                ctx.link_capsules[name].tip_offset_xyz_m[:] = info["axis"] * max(
                    0.0, orig - cfg.link_radius_m
                )
            _redraw_capsules()
            status.value = "Reset to loaded values"

    logger.info("Tip-length tuner running at http://%s:%d", cfg.viewer_host, cfg.viewer_port)
    logger.info("Tuning %d fingertip(s): %s", len(tips), ", ".join(tips) or "(none)")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
