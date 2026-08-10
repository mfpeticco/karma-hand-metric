#!/usr/bin/env python3
"""Overlay two robot-hand URDFs in viser with independent 6-DOF pose controls.

A development aid for eyeballing two hands at the same scale: pick a hand into
each of slots A and B, then translate and rotate each until they line up. Useful
when deciding how a new hand compares to one already in the set.

This renders each hand's visual meshes. It defaults to the bundled visual models
in `robots/visual_models/` (one subdirectory of meshes per hand, each paired with
the matching metric URDF in `robots/urdfs/<name>.urdf`); pass `--models-dir` to
point at a different collection, where a subdirectory may instead carry its own
URDF alongside the meshes.

Run from the repo root:
    python tools/compare_hands.py                       # bundled hands
    python tools/compare_hands.py --models-dir /path/to/hand_models
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Repo root (this script lives in tools/); make bundled models the default.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import viser
import yaml
import yourdfpy
from scipy.spatial.transform import Rotation
from viser.extras import ViserUrdf

logger = logging.getLogger(__name__)

# Mesh extensions we index when resolving package:// and relative references.
_MESH_EXTS = {".obj", ".stl", ".dae", ".ply", ".glb", ".gltf"}


def _discover_hands(models_dir: Path) -> dict[str, tuple[Path, Path]]:
    """Scan *models_dir* for hand subdirectories.

    Returns ``name -> (urdf_path, mesh_dir)``. Each subdirectory holds a hand's
    meshes; the URDF is the matching metric URDF at ``robots/urdfs/<name>.urdf``,
    or a URDF placed inside the folder if one is present (the fallback used when
    pointing ``--models-dir`` at an external collection).
    """
    hands: dict[str, tuple[Path, Path]] = {}
    for sub in sorted(models_dir.iterdir()):
        if not sub.is_dir() or sub.name.startswith("."):
            continue
        has_meshes = any(f.suffix.lower() in _MESH_EXTS for f in sub.rglob("*"))
        if not has_meshes:
            continue

        urdfs = list(sub.rglob("*.urdf"))
        if urdfs:
            dir_lower = sub.name.lower()

            def _score(p: Path) -> tuple[int, int, int, int, int]:
                name_lower = p.stem.lower()
                full_lower = str(p.relative_to(sub)).lower()
                is_glb = 1 if "glb" in name_lower else 0
                is_left = 1 if ("left" in full_lower or "-l." in full_lower
                                or "_l." in full_lower or "-l/" in full_lower
                                or "_l/" in full_lower) else 0
                is_right = 0 if ("right" in full_lower or "-r." in full_lower
                                 or "_r." in full_lower or "-r/" in full_lower
                                 or "_r/" in full_lower) else 1
                # Penalise a URDF whose name relates to no token of the hand dir.
                name_mismatch = 0 if any(tok in name_lower for tok in dir_lower.split()) else 1
                return (is_glb, is_left, name_mismatch, is_right, len(p.stem))

            urdf = min(urdfs, key=_score)
        else:
            urdf = _ROOT / "robots" / "urdfs" / f"{sub.name}.urdf"
            if not urdf.exists():
                logger.warning("No URDF for %s (expected %s)", sub.name, urdf)
                continue

        hands[sub.name] = (urdf, sub)

    return hands


def _make_filename_handler(urdf_path: Path, hand_dir: Path):
    """Return a yourdfpy ``filename_handler`` that resolves relative paths and
    ``package://`` URIs by searching *hand_dir* for the referenced file."""
    urdf_dir = urdf_path.parent

    file_index: dict[str, Path] = {}
    stem_index: dict[str, Path] = {}
    for f in hand_dir.rglob("*"):
        if f.is_file() and f.suffix.lower() in _MESH_EXTS:
            file_index[f.name.lower()] = f
            stem_index.setdefault(f.stem.lower(), f)

    def _resolve(bare_name: str) -> Path | None:
        hit = file_index.get(bare_name.lower())
        if hit is not None:
            return hit
        return stem_index.get(Path(bare_name).stem.lower())

    def handler(fname: str) -> str:
        if fname.startswith("package://"):
            resolved = _resolve(Path(fname.split("package://", 1)[1]).name)
            if resolved is not None:
                return str(resolved)
            logger.warning("Could not resolve %s in %s", fname, hand_dir.name)
            return fname

        candidate = urdf_dir / fname
        if candidate.exists():
            return str(candidate.resolve())

        resolved = _resolve(Path(fname).name)
        return str(resolved) if resolved is not None else fname

    return handler


def _load_urdf(urdf_path: Path, mesh_dir: Path) -> yourdfpy.URDF:
    """Load *urdf_path*, resolving its mesh references from *mesh_dir*.

    The URDF and its meshes may live in different directories (the metric URDF
    in ``robots/urdfs/`` and the meshes in ``robots/visual_models/<name>/``),
    so the mesh search root is passed explicitly.
    """
    return yourdfpy.URDF.load(
        str(urdf_path),
        filename_handler=_make_filename_handler(urdf_path, mesh_dir),
        build_collision_scene_graph=False,
        load_collision_meshes=False,
    )


def _add_world_frame(server: viser.ViserServer, length: float = 0.05) -> None:
    """Draw a small RGB axis triad at the world origin."""
    for axis, color, direction in [
        ("x", (255, 0, 0), [length, 0, 0]),
        ("y", (0, 255, 0), [0, length, 0]),
        ("z", (0, 0, 255), [0, 0, length]),
    ]:
        pts = np.array([[[0, 0, 0], direction]], dtype=np.float32)
        server.scene.add_line_segments(
            f"/world_frame/{axis}", points=pts, colors=color, line_width=3.0,
        )


def _load_saved_poses(path: Path) -> dict:
    """Load the pose database (``{model_name: {x,y,z,roll,pitch,yaw}}``) or {}."""
    if path.exists():
        try:
            with open(path) as f:
                data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}
        except Exception:  # noqa: BLE001 - a bad pose file should not stop the tool
            logger.warning("Could not read %s", path)
    return {}


def _save_all_poses(pose_db: dict, path: Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(pose_db, f, default_flow_style=False)
    logger.info("Saved %d model poses to %s", len(pose_db), path)


def _make_hand_controls(
    server: viser.ViserServer,
    label: str,
    hands: dict[str, Path],
    root_name: str,
    color: tuple[float, float, float, float],
    initial_hand: str,
    pose_db: dict,
) -> dict:
    """Create a GUI folder for one hand slot (dropdown + 6-DOF sliders), load it,
    and return a mutable state dict for the slot."""
    state: dict = {"urdf_vis": None}
    init_pose = pose_db.get(initial_hand, {})

    # Parent frame carrying the whole hand's 6-DOF pose.
    state["frame"] = server.scene.add_frame(
        root_name, position=(0, 0, 0), wxyz=(1, 0, 0, 0), show_axes=False,
    )

    with server.gui.add_folder(label):
        dropdown = server.gui.add_dropdown(
            "Model", options=list(hands.keys()), initial_value=initial_hand,
        )
        pos_x = server.gui.add_slider("x (m)", min=-0.5, max=0.5, step=0.005,
                                      initial_value=float(init_pose.get("x", 0.0)))
        pos_y = server.gui.add_slider("y (m)", min=-0.5, max=0.5, step=0.005,
                                      initial_value=float(init_pose.get("y", 0.0)))
        pos_z = server.gui.add_slider("z (m)", min=-0.5, max=0.5, step=0.005,
                                      initial_value=float(init_pose.get("z", 0.0)))
        rot_r = server.gui.add_slider("roll (deg)", min=-180, max=180, step=5,
                                      initial_value=float(init_pose.get("roll", 0.0)))
        rot_p = server.gui.add_slider("pitch (deg)", min=-180, max=180, step=5,
                                      initial_value=float(init_pose.get("pitch", 0.0)))
        rot_y = server.gui.add_slider("yaw (deg)", min=-180, max=180, step=5,
                                      initial_value=float(init_pose.get("yaw", 0.0)))
        btn_reset = server.gui.add_button("Reset Pose")

    state["dropdown"] = dropdown
    sliders = {"x": pos_x, "y": pos_y, "z": pos_z,
               "roll": rot_r, "pitch": rot_p, "yaw": rot_y}
    state["sliders"] = sliders

    def _update_pose(_=None) -> None:
        q = Rotation.from_euler(
            "xyz", [rot_r.value, rot_p.value, rot_y.value], degrees=True,
        ).as_quat(scalar_first=True)
        state["frame"].position = np.array([pos_x.value, pos_y.value, pos_z.value])
        state["frame"].wxyz = q

    def _apply_saved_pose(model_name: str) -> None:
        pose = pose_db.get(model_name, {})
        pos_x.value = float(pose.get("x", 0.0))
        pos_y.value = float(pose.get("y", 0.0))
        pos_z.value = float(pose.get("z", 0.0))
        rot_r.value = float(pose.get("roll", 0.0))
        rot_p.value = float(pose.get("pitch", 0.0))
        rot_y.value = float(pose.get("yaw", 0.0))
        _update_pose()

    def _load_hand(name: str) -> None:
        logger.info("Loading %s into %s ...", name, label)
        if state["urdf_vis"] is not None:
            state["urdf_vis"].remove()
            state["urdf_vis"] = None
        try:
            urdf = _load_urdf(*hands[name])
            vis = ViserUrdf(
                server, urdf, root_node_name=root_name, mesh_color_override=color,
            )
            vis.update_cfg(np.zeros(len(urdf.actuated_joint_names)))
            state["urdf_vis"] = vis
            logger.info("Loaded %s (%d joints)", name, len(urdf.actuated_joint_names))
        except Exception:  # noqa: BLE001 - one bad model should not stop the tool
            logger.exception("Failed to load %s", name)

    @btn_reset.on_click
    def _(_: viser.GuiEvent) -> None:
        for s in sliders.values():
            s.value = 0
        _update_pose()

    dropdown.on_update(lambda _: (_load_hand(dropdown.value), _apply_saved_pose(dropdown.value)))
    for ctrl in sliders.values():
        ctrl.on_update(_update_pose)

    _load_hand(initial_hand)
    _update_pose()
    return state


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlay and pose two robot-hand URDFs in viser.",
    )
    parser.add_argument(
        "--models-dir", type=Path, default=_ROOT / "robots" / "visual_models",
        help="Directory whose subdirectories each hold one hand (URDF + meshes). "
             "Defaults to the bundled robots/visual_models/.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--poses-file", type=Path, default=None,
        help="Where to store saved poses (default: <models-dir>/compare_hands_poses.yaml)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not args.models_dir.is_dir():
        logger.error("--models-dir %s is not a directory", args.models_dir)
        return
    hands = _discover_hands(args.models_dir)
    if len(hands) < 2:
        logger.error("Need at least 2 hand models in %s (found %d)", args.models_dir, len(hands))
        return
    logger.info("Found %d hands: %s", len(hands), ", ".join(hands))

    poses_file = args.poses_file or (args.models_dir / "compare_hands_poses.yaml")
    pose_db = _load_saved_poses(poses_file)

    server = viser.ViserServer(host=args.host, port=args.port)
    _add_world_frame(server)

    names = list(hands.keys())
    state_a = _make_hand_controls(
        server, "Hand A", hands, "/hand_a", (0.3, 0.5, 1.0, 0.8), names[0], pose_db,
    )
    state_b = _make_hand_controls(
        server, "Hand B", hands, "/hand_b", (1.0, 0.5, 0.2, 0.8), names[1], pose_db,
    )

    btn_save = server.gui.add_button("Save Poses")

    @btn_save.on_click
    def _(_: viser.GuiEvent) -> None:
        for st in (state_a, state_b):
            pose_db[st["dropdown"].value] = {k: float(s.value) for k, s in st["sliders"].items()}
        _save_all_poses(pose_db, poses_file)

    logger.info("Hand comparison running at http://%s:%d", args.host, args.port)

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
