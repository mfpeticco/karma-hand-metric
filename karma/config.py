"""Configuration loading and validation for KaRMA."""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from .lref import compute_lref

logger = logging.getLogger(__name__)


def _require(cond: object, msg: str) -> None:
    """Validate a config invariant. Unlike ``assert``, survives ``python -O``."""
    if not cond:
        raise ValueError(msg)


@dataclass(frozen=True)
class MimicJointConfig:
    """URDF-style mimic joint: q_mimic = multiplier * q_source + offset."""
    joint: str
    multiplier: float = 1.0
    offset: float = 0.0


@dataclass(frozen=True)
class LinkConfig:
    """Per-link kinematic info needed for procedural capsule geometry."""
    primary_child_joint: str | None = None
    tip_length_m: float | None = None


@dataclass(frozen=True)
class MetricConfig:
    # Robot identity
    robot_name: str
    urdf_path: Path
    base_link: str

    # Procedural capsule geometry hints
    links: dict[str, LinkConfig]

    # Mimic joints (keyed by mimic joint name)
    mimic_joints: dict[str, MimicJointConfig]

    # Active joints
    thumb_joint_names: list[str]
    index_joint_names: list[str]

    # Contact surface link names (no collision geometry suffixes)
    thumb_contact_links: list[str]
    index_contact_links: list[str]

    # Geometry
    sphere_radius_m: float
    link_radius_m: float

    # Discretization
    voxel_size_m: float
    step_size_m: float
    healpix_nside: int

    # Physics
    friction_coeff: float

    # Tolerances
    eps_g_m: float
    eps_col_m: float
    max_joint_step_deg: float

    # Rotation slip control (None = allow slip, value = reject if tangential slip > threshold)
    rotation_slip_threshold_mm: float | None

    # QP
    qp_regularization: float

    # BFS
    max_states: int

    # Seed
    seed_thumb_link: str
    seed_index_link: str
    seed_search_offsets_m: list[float]

    # Output
    out_dir: Path

    # Viewer
    viewer_host: str
    viewer_port: int

    # Number of parallel workers for Phase 2 rotation exploration
    rotation_workers: int

    # Reference length for non-dimensionalization (metres)
    l_ref_m: float

    @property
    def finger_joint_names(self) -> list[str]:
        return self.thumb_joint_names + self.index_joint_names

    @property
    def all_contact_links(self) -> list[str]:
        return self.thumb_contact_links + self.index_contact_links

    @property
    def max_joint_step_rad(self) -> float:
        return math.radians(self.max_joint_step_deg)


def _parse_link_configs(raw: dict) -> dict[str, LinkConfig]:
    links_raw = raw.get("links", {}) or {}
    links: dict[str, LinkConfig] = {}
    for link_name, cfg in links_raw.items():
        if cfg is None:
            cfg = {}
        primary_child_joint = cfg.get("primary_child_joint", None)
        tip_length = cfg.get("tip_length_m", None)
        links[str(link_name)] = LinkConfig(
            primary_child_joint=str(primary_child_joint) if primary_child_joint is not None else None,
            tip_length_m=float(tip_length) if tip_length is not None else None,
        )
    return links


def _parse_mimic_configs(raw: dict) -> dict[str, MimicJointConfig]:
    joints_raw = raw.get("joints", {}) or {}
    mimics: dict[str, MimicJointConfig] = {}
    for joint_name, cfg in joints_raw.items():
        if not isinstance(cfg, dict):
            continue
        mimic = cfg.get("mimic", None)
        if mimic is None:
            continue
        mimics[str(joint_name)] = MimicJointConfig(
            joint=str(mimic["joint"]),
            multiplier=float(mimic.get("multiplier", 1.0)),
            offset=float(mimic.get("offset", 0.0)),
        )
    return mimics


def load_config(
    metric_config_path: str | Path,
    robot_config_path: str | Path,
) -> MetricConfig:
    """Load runtime config.

    Args:
        metric_config_path: global fine-metric config (shared across robots)
        robot_config_path: robot model config (joint/link info only)
    """
    metric_config_path = Path(metric_config_path)
    with open(metric_config_path) as f:
        metric_raw = yaml.safe_load(f) or {}

    robot_config_path = Path(robot_config_path)
    with open(robot_config_path) as f:
        robot_raw = yaml.safe_load(f) or {}
    robot_base = robot_config_path.parent

    # Resolve URDF path relative to the robot config file
    urdf_path = Path(robot_raw["urdf_path"])
    if not urdf_path.is_absolute():
        urdf_path = robot_base / urdf_path

    # Resolve out_dir relative to metric config file
    out_dir = Path(metric_raw.get("out_dir", "workspace"))
    if not out_dir.is_absolute():
        out_dir = metric_config_path.parent / out_dir

    # rotation_workers: null / 0 / missing -> auto (one worker per CPU core). Phase 2
    # explores each voxel independently, so the worker count changes only speed, not
    # the result; the shipped config leaves it on auto.
    _rw = metric_raw.get("rotation_workers")
    rotation_workers = int(_rw) if _rw else (os.cpu_count() or 4)

    # Compute reference length for non-dimensionalization
    l_ref_m = compute_lref(robot_config_path)

    # Scale ALL length parameters by L_ref / L_ref_nominal so that robots
    # of different sizes get a geometrically equivalent problem.
    # The YAML values are nominal (tuned for a hand with L_ref ≈ l_ref_nominal).
    l_ref_nominal = float(metric_raw.get("l_ref_nominal_m", 0.2))
    if l_ref_m > 0 and l_ref_nominal > 0:
        scale = l_ref_m / l_ref_nominal
    else:
        scale = 1.0

    # Read raw (nominal) values from config
    sphere_radius_raw = float(metric_raw["sphere_radius_m"])
    link_radius_raw = float(metric_raw["link_radius_m"])
    voxel_size_raw = float(metric_raw["voxel_size_m"])
    step_size_raw = float(metric_raw.get("step_size_m", voxel_size_raw * 0.125))
    eps_g_raw = float(metric_raw.get("eps_g_m", 0.00035))
    eps_col_raw = float(metric_raw.get("eps_col_m", 0.0005))
    slip_raw = float(metric_raw["rotation_slip_threshold_mm"]) if metric_raw.get("rotation_slip_threshold_mm") is not None else None
    offsets_raw = [float(x) for x in metric_raw.get("seed_search_offsets_m", [0.0])]

    # Apply L_ref scaling to ALL length params
    sphere_radius_m = sphere_radius_raw * scale
    link_radius_m = link_radius_raw * scale
    voxel_size_m = voxel_size_raw * scale
    step_size_m = step_size_raw * scale
    eps_g_m = eps_g_raw * scale
    eps_col_m = eps_col_raw * scale
    rotation_slip_threshold_mm = slip_raw * scale if slip_raw is not None else None
    seed_search_offsets_m = [x * scale for x in offsets_raw]

    # QP regularization scales as scale^2: the rolling constraint cost ||M@x-b||^2
    # has units of m^2 (Jacobian entries scale with L_ref), so the regularization
    # term reg*||dq||^2 must scale by L_ref^2 to maintain the same damping ratio.
    qp_reg_raw = float(metric_raw.get("qp_regularization", 1e-6))
    qp_regularization = qp_reg_raw * scale ** 2

    logger.info("L_ref=%.4f m, L_ref_nominal=%.4f m, scale=%.4f → "
                "sphere=%.2f mm, link=%.2f mm, voxel=%.3f mm",
                l_ref_m, l_ref_nominal, scale,
                sphere_radius_m * 1e3, link_radius_m * 1e3, voxel_size_m * 1e3)

    fingers = robot_raw.get("fingers", {})
    seed = robot_raw.get("seed", {})

    cfg = MetricConfig(
        robot_name=str(robot_raw.get("robot_name", "robot")),
        urdf_path=urdf_path,
        base_link=str(robot_raw.get("base_link", "")),
        links=_parse_link_configs(robot_raw),
        mimic_joints=_parse_mimic_configs(robot_raw),
        thumb_joint_names=list(fingers["thumb"]["active_joints"]),
        index_joint_names=list(fingers["index"]["active_joints"]),
        thumb_contact_links=list(fingers["thumb"]["contact_links"]),
        index_contact_links=list(fingers["index"]["contact_links"]),
        sphere_radius_m=sphere_radius_m,
        link_radius_m=link_radius_m,
        voxel_size_m=voxel_size_m,
        step_size_m=step_size_m,
        healpix_nside=int(metric_raw["healpix_nside"]),
        friction_coeff=float(metric_raw["friction_coeff"]),
        eps_g_m=eps_g_m,
        eps_col_m=eps_col_m,
        max_joint_step_deg=float(metric_raw.get("max_joint_step_deg", 10.0)),
        rotation_slip_threshold_mm=rotation_slip_threshold_mm,
        qp_regularization=qp_regularization,
        max_states=int(metric_raw.get("max_states", 10000)),
        seed_thumb_link=str(seed.get("thumb_link", "")),
        seed_index_link=str(seed.get("index_link", "")),
        seed_search_offsets_m=seed_search_offsets_m,
        out_dir=out_dir,
        viewer_host=str(metric_raw.get("viewer_host", "127.0.0.1")),
        viewer_port=int(metric_raw.get("viewer_port", 8080)),
        rotation_workers=rotation_workers,
        l_ref_m=l_ref_m,
    )

    # Validation
    _require(cfg.base_link, "base_link must be set in robot config")
    _require(cfg.seed_thumb_link, "seed.thumb_link must be set in robot config")
    _require(cfg.seed_index_link, "seed.index_link must be set in robot config")

    _require(cfg.sphere_radius_m > 0, "sphere_radius_m must be positive")
    _require(cfg.link_radius_m > 0, "link_radius_m must be positive")
    _require(cfg.voxel_size_m > 0, "voxel_size_m must be positive")
    _require(cfg.step_size_m > 0, "step_size_m must be positive")
    _require(cfg.friction_coeff > 0, "friction_coeff must be positive")
    _require(cfg.healpix_nside >= 1, "healpix_nside must be >= 1")
    _require(cfg.max_states >= 1, "max_states must be >= 1")
    _require(len(cfg.thumb_joint_names) > 0, "fingers.thumb.active_joints must be non-empty")
    _require(len(cfg.index_joint_names) > 0, "fingers.index.active_joints must be non-empty")
    _require(len(cfg.thumb_contact_links) > 0, "fingers.thumb.contact_links must be non-empty")
    _require(len(cfg.index_contact_links) > 0, "fingers.index.contact_links must be non-empty")
    _require(cfg.seed_thumb_link in cfg.thumb_contact_links, "seed.thumb_link must be in fingers.thumb.contact_links")
    _require(cfg.seed_index_link in cfg.index_contact_links, "seed.index_link must be in fingers.index.contact_links")

    mimic_targets = set(cfg.mimic_joints.keys())
    active = set(cfg.finger_joint_names)
    bad_active = sorted(active & mimic_targets)
    if bad_active:
        raise ValueError(f"Active joints cannot include mimic joints: {bad_active}")

    return cfg
