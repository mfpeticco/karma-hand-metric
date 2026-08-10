"""Lightweight robot loading: YAML → Pinocchio model with FK/Jacobian helpers.

Intentionally standalone: replicates the joint-index resolution, tip
auto-detection, and mimic handling from ``karma/robot.py`` without importing
that package, so the baseline metrics stay decoupled from the main pipeline.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin
import yaml


# ── Inline helpers ────────────────────────────────────────────────────────

def _skew(v: np.ndarray) -> np.ndarray:
    """3x3 skew-symmetric matrix from a 3-vector."""
    v = np.asarray(v, dtype=float).ravel()
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])


def _rpy_to_rot(rpy: np.ndarray) -> np.ndarray:
    """URDF roll-pitch-yaw to rotation matrix (R = Rz*Ry*Rx)."""
    roll, pitch, yaw = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=float)


# ── Data structures ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class MimicJoint:
    target_q_idx: int
    source_q_idx: int
    multiplier: float
    offset: float


@dataclass(frozen=True)
class FingerInfo:
    name: str
    active_joint_names: list[str]
    q_indices: np.ndarray     # position indices of active joints
    v_indices: np.ndarray     # velocity indices of active joints
    tip_link_name: str
    tip_frame_id: int
    tip_offset_xyz_m: np.ndarray   # (3,)
    contact_link_names: list[str]


@dataclass
class BaselineRobot:
    robot_name: str
    urdf_path: Path
    yaml_path: Path
    base_link: str
    model: pin.Model
    data: pin.Data
    thumb: FingerInfo
    index: FingerInfo
    mimic_joints: list[MimicJoint]
    all_q_indices: np.ndarray   # sorted union of thumb + index q indices
    all_v_indices: np.ndarray   # sorted union of thumb + index v indices
    q_lower: np.ndarray         # full model lower bounds
    q_upper: np.ndarray         # full model upper bounds
    thumb_q_lower: np.ndarray   # active-only bounds for thumb
    thumb_q_upper: np.ndarray
    index_q_lower: np.ndarray   # active-only bounds for index
    index_q_upper: np.ndarray


# ── URDF tip auto-detection ──────────────────────────────────────────────

def _parse_urdf_collision_tips(urdf_path: Path) -> dict[str, np.ndarray]:
    """Distal endpoint of collision cylinders per link (link-local frame)."""
    tree = ET.parse(str(urdf_path))
    root = tree.getroot()
    tips: dict[str, np.ndarray] = {}

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

        origin = collision.find("origin")
        if origin is None:
            center = np.zeros(3)
            rpy = np.zeros(3)
        else:
            center = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ", dtype=float)
            rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ", dtype=float)

        length = float(cyl.attrib.get("length", "0"))
        half_length = length / 2.0
        if half_length < 1e-6:
            continue

        R = _rpy_to_rot(rpy)
        local_z = R[:, 2]
        c1 = center + half_length * local_z
        c2 = center - half_length * local_z
        tip = c1 if np.linalg.norm(c1) > np.linalg.norm(c2) else c2
        tips[link_name] = tip

    return tips


def _parse_urdf_inertial_coms(urdf_path: Path) -> dict[str, np.ndarray]:
    """Inertial origin (CoM) per link."""
    tree = ET.parse(str(urdf_path))
    root = tree.getroot()
    coms: dict[str, np.ndarray] = {}

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
        coms[link_name] = xyz

    return coms


def _parse_urdf_joints(urdf_path: Path) -> dict[str, dict]:
    """Return joint info dict keyed by joint name."""
    tree = ET.parse(str(urdf_path))
    root = tree.getroot()
    joints: dict[str, dict] = {}
    for j in root.findall("joint"):
        name = j.attrib["name"]
        j_type = j.attrib.get("type", "fixed")
        parent = j.find("parent").attrib["link"]
        child = j.find("child").attrib["link"]
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
    return joints


def _auto_detect_tip_offset(
    link_name: str,
    collision_tips: dict[str, np.ndarray],
    inertial_coms: dict[str, np.ndarray],
    urdf_joints: dict[str, dict],
) -> np.ndarray:
    """Auto-detect tip direction; mirrors the tip fallback chain in
    karma.robot._build_link_capsules."""
    # Fallback 1: URDF collision geometry
    if link_name in collision_tips:
        return collision_tips[link_name].copy()

    # Fallback 2: 2x inertial CoM
    if link_name in inertial_coms:
        com = inertial_coms[link_name]
        com_norm = float(np.linalg.norm(com))
        if com_norm > 0.001:
            return 2.0 * com.copy()

    # Fallback 3: incoming joint translation direction
    # Find the joint whose child is this link
    for _jname, jinfo in urdf_joints.items():
        if jinfo["child"] == link_name:
            t_parent = jinfo["origin_xyz"]
            R_parent_child = _rpy_to_rot(jinfo["origin_rpy"])
            return (R_parent_child.T @ t_parent).copy()

    return np.array([0.0, 0.0, 0.01])  # last-resort default


def _resolve_tip_offset(
    link_name: str,
    link_cfg: dict | None,
    collision_tips: dict[str, np.ndarray],
    inertial_coms: dict[str, np.ndarray],
    urdf_joints: dict[str, dict],
) -> np.ndarray:
    """Resolve tip offset for a fingertip link.

    Handles both YAML formats:
      - tip_offset_xyz_m: [x, y, z]  — used directly
      - tip_length_m: scalar          — auto-detect direction, scale to this length
    """
    if link_cfg is not None and isinstance(link_cfg, dict):
        if "tip_offset_xyz_m" in link_cfg:
            return np.array(link_cfg["tip_offset_xyz_m"], dtype=float)
        if "tip_length_m" in link_cfg:
            tip_length = float(link_cfg["tip_length_m"])
            auto = _auto_detect_tip_offset(link_name, collision_tips, inertial_coms, urdf_joints)
            auto_norm = float(np.linalg.norm(auto))
            if auto_norm > 1e-9:
                direction = auto / auto_norm
            else:
                direction = np.array([0.0, 0.0, 1.0])
            return direction * tip_length

    # No YAML config — use raw auto-detected offset
    return _auto_detect_tip_offset(link_name, collision_tips, inertial_coms, urdf_joints)


# ── Joint index resolution ───────────────────────────────────────────────

def _joint_name_to_id(model: pin.Model) -> dict[str, int]:
    return {str(name): i for i, name in enumerate(list(model.names))}


def _active_indices(
    model: pin.Model, joint_names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Return (q_indices, v_indices) for the given joint names."""
    name_to_id = _joint_name_to_id(model)
    missing = [n for n in joint_names if n not in name_to_id]
    if missing:
        raise KeyError(f"Joint names not found in URDF: {missing}")
    q_ids, v_ids = [], []
    for name in joint_names:
        jid = name_to_id[name]
        joint = model.joints[jid]
        if joint.nq != 1:
            raise NotImplementedError(f"Only 1-DoF joints supported; {name} has nq={joint.nq}")
        q_ids.append(int(joint.idx_q))
        v_ids.append(int(joint.idx_v))
    return np.array(sorted(q_ids), dtype=int), np.array(sorted(v_ids), dtype=int)


# ── Mimic joint resolution ───────────────────────────────────────────────

def _build_mimic_joints(
    model: pin.Model, robot_raw: dict
) -> list[MimicJoint]:
    joints_cfg = robot_raw.get("joints", {}) or {}
    name_to_id = _joint_name_to_id(model)
    mimics: list[MimicJoint] = []
    for joint_name, cfg in joints_cfg.items():
        if not isinstance(cfg, dict):
            continue
        mimic = cfg.get("mimic")
        if mimic is None:
            continue
        if joint_name not in name_to_id:
            raise KeyError(f"Mimic joint '{joint_name}' not found in Pinocchio model")
        source_name = mimic["joint"]
        if source_name not in name_to_id:
            raise KeyError(f"Mimic source joint '{source_name}' not found in Pinocchio model")
        tgt_j = model.joints[name_to_id[joint_name]]
        src_j = model.joints[name_to_id[source_name]]
        mimics.append(MimicJoint(
            target_q_idx=int(tgt_j.idx_q),
            source_q_idx=int(src_j.idx_q),
            multiplier=float(mimic.get("multiplier", 1.0)),
            offset=float(mimic.get("offset", 0.0)),
        ))
    return mimics


# ── Main loader ──────────────────────────────────────────────────────────

def load_baseline_robot(yaml_path: str | Path) -> BaselineRobot:
    """Load a robot from a YAML config for baseline computation.

    Parameters
    ----------
    yaml_path : path to ``robot_*.yaml``
    """
    yaml_path = Path(yaml_path)
    with open(yaml_path) as f:
        robot_raw = yaml.safe_load(f) or {}

    robot_name = str(robot_raw.get("robot_name", yaml_path.stem))
    urdf_rel = robot_raw["urdf_path"]
    urdf_path = yaml_path.parent / urdf_rel
    base_link = str(robot_raw["base_link"])

    # Load Pinocchio model (kinematics only, no meshes)
    model = pin.buildModelFromUrdf(str(urdf_path))
    data = model.createData()

    # Parse URDF for tip auto-detection
    collision_tips = _parse_urdf_collision_tips(urdf_path)
    inertial_coms = _parse_urdf_inertial_coms(urdf_path)
    urdf_joints = _parse_urdf_joints(urdf_path)

    links_cfg = robot_raw.get("links", {}) or {}
    fingers_cfg = robot_raw["fingers"]

    # Build mimic joints
    mimic_joints = _build_mimic_joints(model, robot_raw)

    def _build_finger(finger_name: str) -> FingerInfo:
        fcfg = fingers_cfg[finger_name]
        active_joint_names = list(fcfg["active_joints"])
        contact_link_names = list(fcfg["contact_links"])
        q_idx, v_idx = _active_indices(model, active_joint_names)

        # Tip link = seed link from YAML
        seed_cfg = robot_raw.get("seed", {}) or {}
        tip_link_name = str(seed_cfg.get(f"{finger_name}_link", ""))
        if not tip_link_name:
            # Fallback: last contact link
            tip_link_name = contact_link_names[-1]

        tip_frame_id = model.getFrameId(tip_link_name)
        if tip_frame_id == model.nframes:
            raise KeyError(f"Tip link '{tip_link_name}' not found in Pinocchio model")

        tip_offset = _resolve_tip_offset(
            tip_link_name,
            links_cfg.get(tip_link_name),
            collision_tips,
            inertial_coms,
            urdf_joints,
        )

        return FingerInfo(
            name=finger_name,
            active_joint_names=active_joint_names,
            q_indices=q_idx,
            v_indices=v_idx,
            tip_link_name=tip_link_name,
            tip_frame_id=tip_frame_id,
            tip_offset_xyz_m=tip_offset,
            contact_link_names=contact_link_names,
        )

    thumb = _build_finger("thumb")
    index = _build_finger("index")

    all_q = np.array(sorted(set(thumb.q_indices) | set(index.q_indices)), dtype=int)
    all_v = np.array(sorted(set(thumb.v_indices) | set(index.v_indices)), dtype=int)

    return BaselineRobot(
        robot_name=robot_name,
        urdf_path=urdf_path,
        yaml_path=yaml_path,
        base_link=base_link,
        model=model,
        data=data,
        thumb=thumb,
        index=index,
        mimic_joints=mimic_joints,
        all_q_indices=all_q,
        all_v_indices=all_v,
        q_lower=model.lowerPositionLimit.copy(),
        q_upper=model.upperPositionLimit.copy(),
        thumb_q_lower=model.lowerPositionLimit[thumb.q_indices],
        thumb_q_upper=model.upperPositionLimit[thumb.q_indices],
        index_q_lower=model.lowerPositionLimit[index.q_indices],
        index_q_upper=model.upperPositionLimit[index.q_indices],
    )


# ── FK / Jacobian helpers ────────────────────────────────────────────────

def apply_mimic(robot: BaselineRobot, q: np.ndarray) -> None:
    """Apply mimic joint constraints in-place."""
    for m in robot.mimic_joints:
        q[m.target_q_idx] = m.multiplier * q[m.source_q_idx] + m.offset


def forward_kinematics(robot: BaselineRobot, q: np.ndarray) -> None:
    """FK + frame placements only (no Jacobians). ~40% cheaper."""
    apply_mimic(robot, q)
    pin.forwardKinematics(robot.model, robot.data, q)
    pin.updateFramePlacements(robot.model, robot.data)


def forward_kinematics_jacobians(robot: BaselineRobot, q: np.ndarray) -> None:
    """FK + frame placements + Jacobian computation."""
    apply_mimic(robot, q)
    pin.forwardKinematics(robot.model, robot.data, q)
    pin.updateFramePlacements(robot.model, robot.data)
    pin.computeJointJacobians(robot.model, robot.data, q)


def fingertip_position(robot: BaselineRobot, finger: FingerInfo) -> np.ndarray:
    """World-frame fingertip position (frame origin + R @ tip_offset)."""
    oMf = robot.data.oMf[finger.tip_frame_id]
    origin = np.asarray(oMf.translation).ravel()
    R = np.asarray(oMf.rotation)
    return origin + R @ finger.tip_offset_xyz_m


def _fold_mimic_jacobian(
    J_full: np.ndarray,
    source_v_ids: np.ndarray,
    robot: BaselineRobot,
) -> np.ndarray:
    """Fold mimic contributions into source joint Jacobian columns.

    Mirrors karma.robot.fold_mimic_jacobian.
    """
    if not robot.mimic_joints:
        return J_full[:, source_v_ids].copy()

    mimic_v_by_source: dict[int, list[tuple[int, float]]] = {}
    for m in robot.mimic_joints:
        # For 1-DoF revolute joints: v_idx == q_idx
        source_v = m.source_q_idx
        target_v = m.target_q_idx
        mimic_v_by_source.setdefault(source_v, []).append((target_v, m.multiplier))

    n_active = len(source_v_ids)
    J_eff = np.zeros((J_full.shape[0], n_active), dtype=J_full.dtype)
    for i, v_id in enumerate(source_v_ids):
        J_eff[:, i] = J_full[:, v_id]
        if v_id in mimic_v_by_source:
            for mimic_v, mult in mimic_v_by_source[v_id]:
                J_eff[:, i] += mult * J_full[:, mimic_v]
    return J_eff


def fingertip_linear_jacobian(
    robot: BaselineRobot, finger: FingerInfo
) -> np.ndarray:
    """3 x n_finger linear Jacobian at the fingertip with tip offset correction.

    Jv_tip = Jv - skew(R @ tip_offset) @ Jw, mirroring the tip-lever correction
    in karma.contacts.contact_kinematics.
    Then folded via mimic joints and extracted to finger-active columns.
    """
    J6 = pin.getFrameJacobian(
        robot.model, robot.data, finger.tip_frame_id,
        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
    )
    Jv = J6[:3, :]
    Jw = J6[3:, :]

    R = np.asarray(robot.data.oMf[finger.tip_frame_id].rotation)
    lever = R @ finger.tip_offset_xyz_m
    Jv_tip = Jv - _skew(lever) @ Jw

    return _fold_mimic_jacobian(Jv_tip, finger.v_indices, robot)


def combined_linear_jacobian(robot: BaselineRobot) -> np.ndarray:
    """6 x n_total stacked [Jv_thumb; Jv_index] over all active DOFs.

    Columns correspond to the union of thumb + index active joints (sorted).
    """
    # Get full-model tip Jacobians with offset correction
    def _full_tip_jv(finger: FingerInfo) -> np.ndarray:
        J6 = pin.getFrameJacobian(
            robot.model, robot.data, finger.tip_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        Jv = J6[:3, :]
        Jw = J6[3:, :]
        R = np.asarray(robot.data.oMf[finger.tip_frame_id].rotation)
        lever = R @ finger.tip_offset_xyz_m
        return Jv - _skew(lever) @ Jw

    Jv_thumb_full = _full_tip_jv(robot.thumb)
    Jv_index_full = _full_tip_jv(robot.index)

    # Stack and fold to all_v_indices with mimic handling
    J_full = np.vstack([Jv_thumb_full, Jv_index_full])  # (6, nv)
    return _fold_mimic_jacobian(J_full, robot.all_v_indices, robot)
