"""Pinocchio robot wrapper: FK, Jacobians, joint index helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import os
import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation
from pinocchio.robot_wrapper import RobotWrapper

from .config import MetricConfig
from .math_utils import rpy_to_matrix

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MimicQConfig:
    target_q: int
    source_q: int
    multiplier: float
    offset: float


@dataclass(frozen=True)
class LinkCapsule:
    """Procedural capsule associated with a URDF link.

    The capsule axis endpoints are defined as:
      - start: link frame origin (URDF inboard joint origin)
      - end: either the selected child link frame origin, or start + R_link * tip_offset_xyz_m

    Notes:
      - Pinocchio removes fixed joints, so some URDF links will not have an associated
        Pinocchio joint. We therefore store Pinocchio *frame* ids here, not joint ids.
    """
    link_name: str
    link_frame_id: int
    distal_frame_id: int | None
    tip_offset_xyz_m: np.ndarray | None


@dataclass
class RobotContext:
    robot: RobotWrapper
    model: pin.Model
    data: pin.Data
    collision_model: pin.GeometryModel
    collision_data: pin.GeometryData
    base_link: str
    link_capsules: dict[str, LinkCapsule]
    mimic_qs: list[MimicQConfig]
    _endpoint_cache: dict[str, tuple[np.ndarray, np.ndarray]] | None = None


def _parse_urdf_joint_tree(urdf_path: Path) -> tuple[set[str], dict[str, dict]]:
    """Parse URDF for link set + joint info (kinematics only)."""
    tree = ET.parse(str(urdf_path))
    root = tree.getroot()

    links = {str(el.attrib["name"]) for el in root.findall("link")}

    joints: dict[str, dict] = {}
    for j in root.findall("joint"):
        name = str(j.attrib["name"])
        j_type = str(j.attrib.get("type", "fixed"))
        parent = str(j.find("parent").attrib["link"])
        child = str(j.find("child").attrib["link"])
        origin_el = j.find("origin")
        if origin_el is None:
            xyz = np.zeros(3)
            rpy = np.zeros(3)
        else:
            xyz = np.fromstring(origin_el.attrib.get("xyz", "0 0 0"), sep=" ", dtype=float)
            rpy = np.fromstring(origin_el.attrib.get("rpy", "0 0 0"), sep=" ", dtype=float)
        joints[name] = {
            "type": j_type,
            "parent": parent,
            "child": child,
            "origin_xyz": xyz,
            "origin_rpy": rpy,
        }

    return links, joints


def _parse_urdf_collision_tips(urdf_path: Path) -> dict[str, tuple[float, float, float]]:
    """Parse URDF collision geometry to extract fingertip positions.

    For each link with a cylinder collision geometry, computes the distal endpoint
    of the cylinder (center + half_length * cylinder_z_axis), choosing the endpoint
    further from the origin (the inboard joint).

    Returns dict mapping link_name -> tip_offset in link's local frame.
    """
    tree = ET.parse(str(urdf_path))
    root = tree.getroot()

    tips: dict[str, tuple[float, float, float]] = {}

    for link in root.findall("link"):
        link_name = link.attrib.get("name")
        if link_name is None:
            continue

        collision = link.find("collision")
        if collision is None:
            continue

        geometry = collision.find("geometry")
        if geometry is None:
            continue

        cyl = geometry.find("cylinder")
        if cyl is None:
            continue

        # Parse collision origin
        origin = collision.find("origin")
        if origin is None:
            center = np.zeros(3)
            rpy = np.zeros(3)
        else:
            center = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ", dtype=float)
            rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ", dtype=float)

        # Parse cylinder dimensions
        length = float(cyl.attrib.get("length", "0"))
        half_length = length / 2.0

        if half_length < 1e-6:
            continue

        # Cylinder axis is local Z after rpy rotation
        R = rpy_to_matrix(rpy)
        local_z = R[:, 2]

        # Two candidate endpoints
        tip_candidate_1 = center + half_length * local_z
        tip_candidate_2 = center - half_length * local_z

        # Choose the candidate further from origin (the inboard joint is at origin in link frame)
        if np.linalg.norm(tip_candidate_1) > np.linalg.norm(tip_candidate_2):
            tip = tip_candidate_1
        else:
            tip = tip_candidate_2

        tips[link_name] = (float(tip[0]), float(tip[1]), float(tip[2]))

    return tips


def _parse_urdf_inertial_coms(urdf_path: Path) -> dict[str, tuple[float, float, float]]:
    """Parse URDF inertial origins to get center of mass for each link.

    Returns dict mapping link_name -> com_xyz in link's local frame.
    Used as fallback when collision geometry is unavailable.
    """
    tree = ET.parse(str(urdf_path))
    root = tree.getroot()

    coms: dict[str, tuple[float, float, float]] = {}

    for link in root.findall("link"):
        link_name = link.attrib.get("name")
        if link_name is None:
            continue

        inertial = link.find("inertial")
        if inertial is None:
            continue

        origin = inertial.find("origin")
        if origin is None:
            continue

        xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ", dtype=float)
        coms[link_name] = (float(xyz[0]), float(xyz[1]), float(xyz[2]))

    return coms


def _build_link_capsules(model: pin.Model, cfg: MetricConfig) -> dict[str, LinkCapsule]:
    """Build procedural capsules for each link from URDF kinematics + YAML overrides.

    For terminal links (no child joints), the tip direction is auto-detected:
      1. URDF collision geometry (distal endpoint of collision cylinder)
      2. 2x inertial center of mass (rough approximation)
      3. Incoming joint translation direction (last resort)

    If tip_length_m is set in the YAML, the auto-detected direction is normalized
    and scaled to that length. Otherwise the auto-detected magnitude is used as-is.

    Note: Fixed joints are removed by Pinocchio, so we skip links with fixed inboard
    joints *unless* they are explicitly referenced by the config (e.g. contact links).
    For those referenced fixed-inboard links we can still compute kinematics via frames.
    """
    links, joints = _parse_urdf_joint_tree(cfg.urdf_path)
    collision_tips = _parse_urdf_collision_tips(cfg.urdf_path)
    inertial_coms = _parse_urdf_inertial_coms(cfg.urdf_path)

    # child_link -> inboard joint
    inboard_joint_of_link: dict[str, str] = {j["child"]: name for name, j in joints.items()}
    # parent_link -> child joints
    child_joints_of_link: dict[str, list[str]] = {}
    for name, j in joints.items():
        child_joints_of_link.setdefault(j["parent"], []).append(name)

    required_links: set[str] = set(cfg.all_contact_links)
    if cfg.seed_thumb_link:
        required_links.add(cfg.seed_thumb_link)
    if cfg.seed_index_link:
        required_links.add(cfg.seed_index_link)
    required_links.update(cfg.links.keys())

    joint_name_to_id = {str(name): i for i, name in enumerate(list(model.names))}

    def _is_fixed_joint(joint_name: str) -> bool:
        return joints[joint_name]["type"] == "fixed"

    def _find_next_movable_joint(start_link: str, visited: set[str] | None = None) -> str | None:
        """Traverse through fixed joints to find the next movable joint in Pinocchio."""
        if visited is None:
            visited = set()
        if start_link in visited:
            return None
        visited.add(start_link)

        child_joint_names = child_joints_of_link.get(start_link, [])
        for jn in child_joint_names:
            if jn in joint_name_to_id:
                # This joint exists in Pinocchio (movable)
                return jn
            # Fixed joint - traverse to its child link
            child_link = joints[jn]["child"]
            result = _find_next_movable_joint(child_link, visited)
            if result is not None:
                return result
        return None

    # Helper: pick primary child joint for axis (skipping fixed joints)
    def _pick_child_joint(parent_link: str, child_joint_names: list[str]) -> str | None:
        override = cfg.links.get(parent_link, None)
        if override is not None and override.primary_child_joint is not None:
            prim = override.primary_child_joint
            if prim in joint_name_to_id:
                return prim
            # Override is a fixed joint - traverse through it
            child_link = joints[prim]["child"]
            return _find_next_movable_joint(child_link)

        # Filter to movable joints only
        movable = [jn for jn in child_joint_names if jn in joint_name_to_id]
        if len(movable) == 1:
            return movable[0]
        if len(movable) > 1:
            # Deterministic fallback: largest origin translation magnitude
            best = movable[0]
            best_norm = -1.0
            for jn in movable:
                nrm = float(np.linalg.norm(joints[jn]["origin_xyz"]))
                if nrm > best_norm:
                    best_norm = nrm
                    best = jn
            return best

        # All child joints are fixed - traverse through them
        for jn in child_joint_names:
            child_link = joints[jn]["child"]
            result = _find_next_movable_joint(child_link)
            if result is not None:
                return result
        return None

    # Build capsules for all non-base links in the URDF.
    capsules: dict[str, LinkCapsule] = {}
    capsule_distal_links: dict[str, str] = {}  # link_name -> distal link name

    for link_name in sorted(links):
        if link_name == cfg.base_link:
            continue
        inboard_joint_name = inboard_joint_of_link.get(link_name, None)
        if inboard_joint_name is None:
            # Not reachable from base_link through joints, or malformed URDF.
            continue

        # Skip links whose inboard joint is fixed (Pinocchio merges them with parent),
        # unless this link is explicitly referenced in the config.
        if inboard_joint_name not in joint_name_to_id and link_name not in required_links:
            continue

        link_frame_id = model.getFrameId(link_name)
        if link_frame_id == model.nframes:
            # Link frame missing in Pinocchio model (unexpected, but skip defensively).
            continue

        child_joint_names = child_joints_of_link.get(link_name, [])

        # Try to find a distal joint (may need to traverse through fixed joints)
        distal_joint_name = _pick_child_joint(link_name, child_joint_names) if child_joint_names else None

        if distal_joint_name is not None:
            distal_link_name = joints[distal_joint_name]["child"]
            distal_frame_id = model.getFrameId(distal_link_name)
            if distal_frame_id == model.nframes:
                raise KeyError(
                    f"Distal link frame '{distal_link_name}' (from joint '{distal_joint_name}') "
                    f"not found in Pinocchio model; cannot build capsule for '{link_name}'"
                )
            capsules[link_name] = LinkCapsule(
                link_name=link_name,
                link_frame_id=link_frame_id,
                distal_frame_id=distal_frame_id,
                tip_offset_xyz_m=None,
            )
            capsule_distal_links[link_name] = distal_link_name
        else:
            # Terminal link (no movable child joints) - determine tip offset.
            # Auto-detect direction, then optionally override magnitude with tip_length_m.
            override = cfg.links.get(link_name, None)
            tip_length = override.tip_length_m if override is not None else None

            # Auto-detect direction via fallback chain
            tip: tuple[float, float, float] | None = None

            # Fallback 1: URDF collision geometry
            if link_name in collision_tips:
                tip = collision_tips[link_name]

            # Fallback 2: 2x inertial CoM (rough approximation for distal phalanges)
            # Skip if CoM is essentially at origin - fall through to joint heuristic
            if tip is None and link_name in inertial_coms:
                com = inertial_coms[link_name]
                com_norm = (com[0]**2 + com[1]**2 + com[2]**2) ** 0.5
                if com_norm > cfg.link_radius_m * 0.1:
                    tip = (2.0 * com[0], 2.0 * com[1], 2.0 * com[2])

            # Fallback 3: Incoming joint translation direction (last resort)
            if tip is None:
                j = joints[inboard_joint_name]
                t_parent = j["origin_xyz"]
                R_parent_child = rpy_to_matrix(j["origin_rpy"])
                tip_vec = R_parent_child.T @ t_parent
                tip = (float(tip_vec[0]), float(tip_vec[1]), float(tip_vec[2]))

            if tip_length is not None:
                # tip_length_m = distance from joint origin to the physical capsule
                # tip (end of hemisphere).  Subtract link_radius to get the capsule
                # axis endpoint (hemisphere centre).
                axis_length = max(tip_length - cfg.link_radius_m, 0.0)
                auto_arr = np.array(tip, dtype=float)
                auto_norm = float(np.linalg.norm(auto_arr))
                if auto_norm > 1e-9:
                    direction = auto_arr / auto_norm
                else:
                    direction = np.array([0.0, 0.0, 1.0])
                scaled = direction * axis_length
                tip = (float(scaled[0]), float(scaled[1]), float(scaled[2]))

            capsules[link_name] = LinkCapsule(
                link_name=link_name,
                link_frame_id=link_frame_id,
                distal_frame_id=None,
                tip_offset_xyz_m=np.array(tip, dtype=float),
            )

    # --- Post-process: remove phantom duplicate capsules ---
    # URDFs with intermediate "jointbody" links (connected by zero-offset fixed
    # or revolute joints) produce capsules that overlap geometrically.  Detect
    # via position-equivalence classes built from zero-offset joints.
    _uf: dict[str, str] = {ln: ln for ln in links}

    def _uf_find(x: str) -> str:
        while _uf[x] != x:
            _uf[x] = _uf[_uf[x]]
            x = _uf[x]
        return x

    def _uf_union(a: str, b: str) -> None:
        ra, rb = _uf_find(a), _uf_find(b)
        if ra != rb:
            _uf[ra] = rb

    for j in joints.values():
        if np.linalg.norm(j["origin_xyz"]) > 1e-6:
            continue
        if j["type"] == "fixed":
            if np.linalg.norm(j["origin_rpy"]) < 1e-4:
                _uf_union(j["parent"], j["child"])
        elif j["type"] == "revolute":
            # Revolute with zero origin: child is always at the same position
            _uf_union(j["parent"], j["child"])

    seen_keys: dict[tuple, str] = {}
    to_remove: list[str] = []
    for name in sorted(capsules.keys()):
        cap = capsules[name]
        cs = _uf_find(name)

        if cap.distal_frame_id is not None and name in capsule_distal_links:
            ce = _uf_find(capsule_distal_links[name])
            key: tuple = (cs, ce)
        else:
            # Terminal capsule: check for zero-length tip
            if cap.tip_offset_xyz_m is not None and np.linalg.norm(cap.tip_offset_xyz_m) < 1e-6:
                if name not in required_links:
                    to_remove.append(name)
                continue
            key = (cs, "terminal")

        # Zero-length: start and end in the same equivalence class
        if key[0] == key[1]:
            if name not in required_links:
                to_remove.append(name)
            continue

        if key in seen_keys:
            existing = seen_keys[key]
            # Prefer required links (they have authoritative YAML config)
            if name in required_links and existing not in required_links:
                to_remove.append(existing)
                seen_keys[key] = name
            elif name not in required_links:
                to_remove.append(name)
            # else: both required — keep both
        else:
            seen_keys[key] = name

    for name in to_remove:
        capsules.pop(name, None)

    return capsules


def _build_mimic_qs(model: pin.Model, cfg: MetricConfig) -> list[MimicQConfig]:
    if not cfg.mimic_joints:
        return []
    name_to_id = {str(name): i for i, name in enumerate(list(model.names))}

    # Build a mapping on q indices.
    mimic_qs: list[MimicQConfig] = []
    for mimic_joint, mimic in cfg.mimic_joints.items():
        if mimic_joint not in name_to_id:
            raise KeyError(f"Mimic joint '{mimic_joint}' not found in URDF/Pinocchio model")
        if mimic.joint not in name_to_id:
            raise KeyError(f"Mimic source joint '{mimic.joint}' not found in URDF/Pinocchio model")

        tgt_j = model.joints[name_to_id[mimic_joint]]
        src_j = model.joints[name_to_id[mimic.joint]]
        if tgt_j.nq != 1 or src_j.nq != 1:
            raise NotImplementedError("Only 1-DoF mimic joints are supported")
        mimic_qs.append(
            MimicQConfig(
                target_q=int(tgt_j.idx_q),
                source_q=int(src_j.idx_q),
                multiplier=float(mimic.multiplier),
                offset=float(mimic.offset),
            )
        )

    return mimic_qs


def _snap_root_placements(model: pin.Model, l_ref_m: float, decimals: int = 12) -> None:
    """Snap root-level joint placements to a fixed grid for EXACT invariance.

    ``_canonicalize_base_pose`` undoes the base pose, but the arithmetic
    ``(t0 + offset) - offset`` is not exact in IEEE float, so the canonicalized
    root joint placements carry a ~1e-16·(base offset) residual.  That residual
    propagates through FK to every link position and flips whichever discrete
    BFS/seed decision (voxel-boundary round, feasibility gap, seed ranking)
    happens to sit nearest a threshold — making the metric only *approximately*
    invariant to the robot's world pose (translation deviations up to several %
    on marginal seeds).

    Snapping the canonicalized placements to a fixed grid makes ANY base pose of
    the same robot yield a BIT-IDENTICAL model, so FK/IK/QP/BFS all produce
    bit-identical results → translation and rotation invariance become EXACT
    (not merely ≤1 voxel), and cross-platform determinism improves.

    The translation grid is L_ref-relative (``round(t / L_ref, decimals) * L_ref``)
    so it commutes with uniform scaling — preserving the exact scale invariance
    from the non-dimensionalized solver.  Rotations are dimensionless: snap the
    matrix entries, then re-orthonormalize (deterministic on the identical
    snapped matrix, so bit-identical across frames).

    The residual being killed is ~1e-15 (relative, for world offsets up to a few
    metres); the grid spacing is 1e-12 (relative), a ~10^3 safety margin, while
    the geometric distortion introduced (~1e-12·L_ref ≈ 0.25 pm) is utterly
    negligible versus mm-scale hand geometry.  Verified bit-exact across
    translation (world offsets), rotation (world rpy), and scale (0.5×/2×) on the
    representative hands; the choice trades slightly less number disturbance
    (finer grid → fewer seed-argmax flips) against margin (coarser grid → more
    robust to larger offsets).  Overridable via env KARMA_SNAP_DECIMALS.
    """
    if l_ref_m <= 0:
        return
    _env = os.environ.get("KARMA_SNAP_DECIMALS")  # test hook for precision sweep
    if _env:
        decimals = int(_env)
    for j in range(1, model.njoints):
        if model.parents[j] != 0:  # only root-level joints carry the base residual
            continue
        pl = model.jointPlacements[j]
        t = np.round(np.asarray(pl.translation, dtype=float) / l_ref_m, decimals) * l_ref_m
        R = np.round(np.asarray(pl.rotation, dtype=float), decimals)
        # Re-orthonormalize onto SO(3); SVD is deterministic on identical input,
        # so identical snapped R across frames -> identical proper rotation.
        U, _, Vt = np.linalg.svd(R)
        R_clean = U @ Vt
        if np.linalg.det(R_clean) < 0:
            U[:, -1] *= -1
            R_clean = U @ Vt
        model.jointPlacements[j] = pin.SE3(R_clean, t)


def _canonicalize_base_pose(model: pin.Model, base_link: str) -> None:
    """Transform the Pinocchio model so the base link is at identity pose.

    Undoes both translation AND rotation of the base link, ensuring that ALL
    forward-kinematics computations produce identical floating-point results
    regardless of the URDF's original base placement (e.g. a wrapper joint
    with arbitrary xyz and rpy).

    Modifies *model* in-place (jointPlacements for root-level joints).
    """
    # Compute FK at neutral to find the base link's world pose.
    tmp_data = model.createData()
    q0 = pin.neutral(model)
    pin.forwardKinematics(model, tmp_data, q0)
    pin.updateFramePlacements(model, tmp_data)

    # Look up the base link frame.
    base_fid = model.getFrameId(base_link)
    if base_fid >= model.nframes:
        logger.warning(
            "Cannot canonicalize: base_link '%s' not found in model frames", base_link
        )
        return

    base_pose = tmp_data.oMf[base_fid]
    base_pos = np.array(base_pose.translation, dtype=float)
    base_rot = np.array(base_pose.rotation, dtype=float)

    pos_offset = np.linalg.norm(base_pos)
    rot_offset = np.linalg.norm(base_rot - np.eye(3))

    if pos_offset < 1e-10 and rot_offset < 1e-10:
        return  # already at identity — nothing to do

    # Inverse of the base pose: transforms world → base-link frame.
    # Pre-multiplying each root joint placement by this undoes the base pose.
    base_inv = base_pose.inverse()

    for j in range(1, model.njoints):
        if model.parents[j] == 0:  # parent is the universe
            model.jointPlacements[j] = base_inv * model.jointPlacements[j]

    if rot_offset >= 1e-10:
        logger.info(
            "Canonicalized base pose: undid translation [%.4f, %.4f, %.4f] and rotation",
            base_pos[0], base_pos[1], base_pos[2],
        )
    else:
        logger.info("Canonicalized base position: shifted by %s", -base_pos)


def load_robot(
    urdf_path: str | Path,
    cfg: MetricConfig,
    world_offset: np.ndarray | list | None = None,
    world_rotation_rpy_deg: np.ndarray | list | None = None,
) -> RobotContext:
    """Load robot model from URDF and build a RobotContext.

    Args:
        urdf_path: Path to the URDF file.
        cfg: Metric configuration.
        world_offset: Optional 3-vector [x, y, z] to translate the robot in
            world frame *before* canonicalization.  Used by robustness tests
            to verify translation invariance without changing URDF structure
            (adding a wrapper joint changes the Pinocchio model's frame count,
            introducing irreducible floating-point differences).  The
            subsequent canonicalization + grid snap undo the transform so the
            model is bit-identical to the unshifted one.
        world_rotation_rpy_deg: Optional roll-pitch-yaw (degrees) to rotate the
            robot about the world origin *before* canonicalization.  The runtime
            analog of ``world_offset`` for verifying rotation invariance without
            rewriting the URDF.
    """
    urdf_path = Path(urdf_path)
    model = pin.buildModelFromUrdf(str(urdf_path))

    # ── Apply optional world-frame transform (for invariance testing) ──
    # Compose an SE3 = rotation-about-origin then translation, and left-multiply
    # every root-level joint placement + the base link frame by it.  The
    # subsequent canonicalization undoes it; the grid snap kills the residual.
    if world_offset is not None or world_rotation_rpy_deg is not None:
        R_w = np.eye(3)
        if world_rotation_rpy_deg is not None:
            R_w = Rotation.from_euler(
                "xyz", np.asarray(world_rotation_rpy_deg, dtype=float), degrees=True
            ).as_matrix()
        t_w = (np.asarray(world_offset, dtype=float)
               if world_offset is not None else np.zeros(3))
        T_w = pin.SE3(R_w, t_w)
        for j in range(1, model.njoints):
            if model.parents[j] == 0:
                model.jointPlacements[j] = T_w * model.jointPlacements[j]
        base_fid = model.getFrameId(cfg.base_link)
        if base_fid < model.nframes:
            model.frames[base_fid].placement = T_w * model.frames[base_fid].placement

    # ── Canonicalize base pose for translation + rotation invariance ──
    # Pinocchio FK computes ALL positions in world frame.  If the robot's base
    # link is not at identity (e.g. a wrapper joint with xyz/rpy offset),
    # floating-point arithmetic on transformed coordinates introduces tiny
    # differences that accumulate through the iterative IK solver and flip
    # marginal seeds at the feasibility boundary.
    #
    # Fix: undo the base link's full SE3 pose (translation AND rotation) so
    # the base link is always at world origin with identity rotation.
    # This makes FK produce identical bit-patterns regardless of the URDF's
    # original base placement → exact translation + rotation invariance.
    _canonicalize_base_pose(model, cfg.base_link)

    # ── Snap the canonicalized model to a fixed grid → EXACT frame invariance ──
    # Kills the ~1e-16 canonicalization residual at the source so any base pose
    # of the same robot yields a bit-identical model (see _snap_root_placements).
    # Runs unconditionally (baseline AND offset) so both sides of an invariance
    # test are snapped to the same grid and therefore bit-identical.
    _snap_root_placements(model, cfg.l_ref_m)

    data = model.createData()
    collision_model = pin.GeometryModel()  # empty on purpose (no meshes required)
    collision_data = collision_model.createData()
    robot = RobotWrapper(model=model, collision_model=collision_model, visual_model=None, verbose=False)

    link_capsules = _build_link_capsules(model, cfg)
    mimic_qs = _build_mimic_qs(model, cfg)
    return RobotContext(
        robot=robot,
        model=model,
        data=data,
        collision_model=collision_model,
        collision_data=collision_data,
        base_link=cfg.base_link,
        link_capsules=link_capsules,
        mimic_qs=mimic_qs,
    )


def neutral_configuration(model: pin.Model) -> np.ndarray:
    return pin.neutral(model)


def joint_name_to_index(model: pin.Model) -> dict[str, int]:
    return {str(name): i for i, name in enumerate(list(model.names))}


def active_velocity_indices(model: pin.Model, joint_names: list[str]) -> np.ndarray:
    name_to_id = joint_name_to_index(model)
    missing = [n for n in joint_names if n not in name_to_id]
    if missing:
        raise KeyError(f"Joint names not found in URDF: {missing}")
    v_ids = []
    for name in joint_names:
        jid = name_to_id[name]
        joint = model.joints[jid]
        if joint.nv != 1:
            raise NotImplementedError(f"Only 1-DoF revolute joints supported; {name} has nv={joint.nv}")
        v_ids.append(joint.idx_v)
    return np.array(sorted(v_ids), dtype=int)


def active_position_indices(model: pin.Model, joint_names: list[str]) -> np.ndarray:
    name_to_id = joint_name_to_index(model)
    missing = [n for n in joint_names if n not in name_to_id]
    if missing:
        raise KeyError(f"Joint names not found in URDF: {missing}")
    q_ids = []
    for name in joint_names:
        jid = name_to_id[name]
        joint = model.joints[jid]
        if joint.nq != 1:
            raise NotImplementedError(f"Only 1-DoF revolute joints supported; {name} has nq={joint.nq}")
        q_ids.append(joint.idx_q)
    return np.array(sorted(q_ids), dtype=int)


def _precompute_endpoints(ctx: RobotContext) -> None:
    """Cache capsule endpoints after FK. Uses np.copyto() on subsequent calls
    to avoid allocation."""
    if ctx._endpoint_cache is None:
        ctx._endpoint_cache = {}
        for name, cap in ctx.link_capsules.items():
            start = np.array(ctx.data.oMf[cap.link_frame_id].translation).ravel().copy()
            if cap.distal_frame_id is not None:
                end = np.array(ctx.data.oMf[cap.distal_frame_id].translation).ravel().copy()
            elif cap.tip_offset_xyz_m is not None:
                R = np.array(ctx.data.oMf[cap.link_frame_id].rotation)
                end = (start + R @ cap.tip_offset_xyz_m).copy()
            else:
                end = start.copy()
            ctx._endpoint_cache[name] = (start, end)
    else:
        for name, cap in ctx.link_capsules.items():
            cached_start, cached_end = ctx._endpoint_cache[name]
            np.copyto(cached_start, np.array(ctx.data.oMf[cap.link_frame_id].translation).ravel())
            if cap.distal_frame_id is not None:
                np.copyto(cached_end, np.array(ctx.data.oMf[cap.distal_frame_id].translation).ravel())
            elif cap.tip_offset_xyz_m is not None:
                R = np.array(ctx.data.oMf[cap.link_frame_id].rotation)
                np.copyto(cached_end, cached_start + R @ cap.tip_offset_xyz_m)
            else:
                np.copyto(cached_end, cached_start)


def update_kinematics(ctx: RobotContext, q: np.ndarray) -> None:
    """Refresh FK, frame placements, Jacobians and geometry for configuration *q*.

    Mimic/coupled joints are written into *q* in place before FK so the model stays
    consistent; pass a copy if the caller's array must not be mutated.
    """
    # Apply mimic joints in-place (if any) so kinematics are consistent.
    for m in ctx.mimic_qs:
        q[m.target_q] = m.multiplier * q[m.source_q] + m.offset

    pin.forwardKinematics(ctx.model, ctx.data, q)
    pin.updateFramePlacements(ctx.model, ctx.data)
    pin.computeJointJacobians(ctx.model, ctx.data, q)
    pin.updateGeometryPlacements(
        ctx.model, ctx.data, ctx.collision_model, ctx.collision_data
    )
    _precompute_endpoints(ctx)


def update_positions_only(ctx: RobotContext, q: np.ndarray) -> None:
    """Lightweight FK: positions and frame placements only, no Jacobians.

    ~30-40% cheaper than update_kinematics(). Use when only positions are
    needed (gap-only checks, collision checks, contact pair selection).
    Caller MUST call update_kinematics() before any Jacobian access.
    """
    for m in ctx.mimic_qs:
        q[m.target_q] = m.multiplier * q[m.source_q] + m.offset

    pin.forwardKinematics(ctx.model, ctx.data, q)
    pin.updateFramePlacements(ctx.model, ctx.data)
    _precompute_endpoints(ctx)


def fold_mimic_jacobian(
    J_full: np.ndarray,
    source_v_ids: np.ndarray,
    ctx: RobotContext,
) -> np.ndarray:
    """Fold mimic joint contributions into source joint Jacobian columns.

    When a source joint moves by dq, its mimic joints move by multiplier*dq.
    The effective velocity contribution is:
        v = J[:, source] * dq + J[:, mimic] * multiplier * dq
          = (J[:, source] + multiplier * J[:, mimic]) * dq

    Args:
        J_full: Full Jacobian (3, nv) or (6, nv)
        source_v_ids: Velocity indices of source (non-mimic) joints
        ctx: Robot context with mimic joint info

    Returns:
        J_effective: (rows, len(source_v_ids)) with mimic contributions folded in
    """
    if not ctx.mimic_qs:
        # No mimic joints - just extract source columns
        return J_full[:, source_v_ids]

    # Build mapping: source_v_idx -> list of (mimic_v_idx, multiplier)
    # First, get velocity indices for mimic joints
    mimic_v_by_source: dict[int, list[tuple[int, float]]] = {}
    for m in ctx.mimic_qs:
        # m.source_q and m.target_q are position indices
        # For 1-DoF joints, v_idx == q_idx (in Pinocchio for revolute joints)
        source_v = m.source_q  # Works for 1-DoF joints
        target_v = m.target_q
        if source_v not in mimic_v_by_source:
            mimic_v_by_source[source_v] = []
        mimic_v_by_source[source_v].append((target_v, m.multiplier))

    # Build effective Jacobian
    n_active = len(source_v_ids)
    J_eff = np.zeros((J_full.shape[0], n_active), dtype=J_full.dtype)

    for i, v_id in enumerate(source_v_ids):
        # Start with source column
        J_eff[:, i] = J_full[:, v_id]
        # Add contributions from mimic joints
        if v_id in mimic_v_by_source:
            for mimic_v, mult in mimic_v_by_source[v_id]:
                J_eff[:, i] += mult * J_full[:, mimic_v]

    return J_eff


def capsule_endpoints_world(
    ctx: RobotContext,
    link_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Get capsule axis endpoints (A,B) in world for a given link."""
    # Fast path: read from cache populated by _precompute_endpoints()
    if ctx._endpoint_cache is not None and link_name in ctx._endpoint_cache:
        return ctx._endpoint_cache[link_name]

    try:
        cap = ctx.link_capsules[link_name]
    except KeyError as e:
        raise KeyError(
            f"Link '{link_name}' not found in ctx.link_capsules. "
            f"Is it listed in the robot config contact_links/seed links? "
            f"Available capsules: {sorted(ctx.link_capsules.keys())[:25]}{'...' if len(ctx.link_capsules) > 25 else ''}"
        ) from e

    start = np.array(ctx.data.oMf[cap.link_frame_id].translation).ravel()
    if cap.distal_frame_id is not None:
        end = np.array(ctx.data.oMf[cap.distal_frame_id].translation).ravel()
        return start, end

    if cap.tip_offset_xyz_m is None:
        return start, start

    R = np.array(ctx.data.oMf[cap.link_frame_id].rotation)
    end = start + R @ cap.tip_offset_xyz_m
    return start, end
