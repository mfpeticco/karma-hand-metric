"""Six kinematic baselines for comparison with KaRMA.

Baselines:
  1. DOF count
  2. Joint range
  3. Workspace volume (convex hull, per-finger independent sampling)
  4. Workspace intersection (convex hull intersection of thumb ∩ index)
  5. Yoshikawa manipulability (product of singular values)
  6. Global Conditioning Index (1/condition number, averaged)
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.stats.qmc import Sobol

from .robot_loader import (
    BaselineRobot,
    FingerInfo,
    forward_kinematics,
    forward_kinematics_jacobians,
    fingertip_position,
    fingertip_linear_jacobian,
    combined_linear_jacobian,
)


@dataclass
class BaselineResults:
    robot_name: str

    # Baseline 1: DOF count
    n_joints_thumb: int = 0
    n_joints_index: int = 0
    n_joints_total: int = 0
    n_dof_independent: int = 0
    n_joints_with_mimics: int = 0

    # Baseline 2: Joint range
    mean_range_rad: float = 0.0
    median_range_rad: float = 0.0
    total_range_rad: float = 0.0
    thumb_mean_range_rad: float = 0.0
    index_mean_range_rad: float = 0.0

    # Baseline 3: Workspace volume (convex hull)
    thumb_workspace_vol_m3: float = 0.0
    index_workspace_vol_m3: float = 0.0
    avg_workspace_vol_m3: float = 0.0
    thumb_workspace_vol_norm: float = 0.0
    index_workspace_vol_norm: float = 0.0
    avg_workspace_vol_norm: float = 0.0

    # Baseline 4: Workspace intersection (convex hull)
    intersection_vol_m3: float = 0.0
    opposability_index: float = 0.0
    jaccard_similarity: float = 0.0

    # Baseline 5: Yoshikawa manipulability
    thumb_yoshikawa_mean: float = 0.0
    thumb_yoshikawa_max: float = 0.0
    index_yoshikawa_mean: float = 0.0
    index_yoshikawa_max: float = 0.0
    combined_yoshikawa_mean: float = 0.0
    combined_yoshikawa_max: float = 0.0
    thumb_log_manip_mean: float = 0.0
    index_log_manip_mean: float = 0.0
    combined_log_manip_mean: float = 0.0

    # Baseline 6: Global Conditioning Index
    thumb_gci: float = 0.0
    index_gci: float = 0.0
    combined_gci: float = 0.0
    thumb_frac_singular: float = 0.0
    index_frac_singular: float = 0.0
    combined_frac_singular: float = 0.0

    # Metadata
    l_ref_m: float = 0.0
    n_samples_actual: int = 0


# ── Sampling ──────────────────────────────────────────────────────────────

def _sobol_samples_joint(
    robot: BaselineRobot, n_samples: int
) -> np.ndarray:
    """Sobol samples over the joint (thumb+index) DOF space.

    Used for baselines 5-6 (manipulability/GCI) which need both fingers
    at a consistent joint configuration.

    Returns shape (n_actual, nq) full-model configurations.
    """
    n_active = len(robot.all_q_indices)
    log2 = max(1, math.ceil(math.log2(max(n_samples, 2))))
    n_actual = 2 ** log2

    sampler = Sobol(d=n_active, scramble=True, seed=42)
    unit_samples = sampler.random(n_actual)

    q_lo = robot.q_lower[robot.all_q_indices]
    q_hi = robot.q_upper[robot.all_q_indices]
    active_qs = q_lo + unit_samples * (q_hi - q_lo)

    q_neutral = np.zeros(robot.model.nq)
    configs = np.tile(q_neutral, (n_actual, 1))
    configs[:, robot.all_q_indices] = active_qs

    return configs


def _sobol_samples_finger(
    robot: BaselineRobot, finger: FingerInfo, n_samples: int, seed: int = 42,
) -> np.ndarray:
    """Sobol samples over a single finger's DOF space.

    Returns shape (n_actual, 3) fingertip positions in metres.
    """
    q_idx = finger.q_indices
    n_dof = len(q_idx)
    if n_dof == 0:
        return np.zeros((0, 3))

    log2 = max(1, math.ceil(math.log2(max(n_samples, 2))))
    n_actual = 2 ** log2

    sampler = Sobol(d=n_dof, scramble=True, seed=seed)
    unit = sampler.random(n_actual)

    lo = robot.q_lower[q_idx]
    hi = robot.q_upper[q_idx]
    finger_qs = lo + unit * (hi - lo)

    positions = np.empty((n_actual, 3))
    q = np.zeros(robot.model.nq)
    for i in range(n_actual):
        q[:] = 0
        q[q_idx] = finger_qs[i]
        forward_kinematics(robot, q)
        positions[i] = fingertip_position(robot, finger)

    return positions


# ── Baseline 1: DOF count ────────────────────────────────────────────────

def _compute_dof_count(robot: BaselineRobot, results: BaselineResults) -> None:
    results.n_joints_thumb = len(robot.thumb.active_joint_names)
    results.n_joints_index = len(robot.index.active_joint_names)
    results.n_joints_total = results.n_joints_thumb + results.n_joints_index
    results.n_dof_independent = len(robot.all_q_indices)
    results.n_joints_with_mimics = results.n_joints_total + len(robot.mimic_joints)


# ── Baseline 2: Joint range ──────────────────────────────────────────────

def _compute_joint_range(robot: BaselineRobot, results: BaselineResults) -> None:
    thumb_ranges = robot.thumb_q_upper - robot.thumb_q_lower
    index_ranges = robot.index_q_upper - robot.index_q_lower

    all_ranges = np.concatenate([thumb_ranges, index_ranges])
    results.mean_range_rad = float(np.mean(all_ranges))
    results.median_range_rad = float(np.median(all_ranges))
    results.total_range_rad = float(np.sum(all_ranges))
    results.thumb_mean_range_rad = float(np.mean(thumb_ranges)) if len(thumb_ranges) > 0 else 0.0
    results.index_mean_range_rad = float(np.mean(index_ranges)) if len(index_ranges) > 0 else 0.0


# ── Baseline 3 & 4: Workspace volume & intersection (convex hull) ────────

def _safe_hull_volume(points: np.ndarray) -> float:
    """ConvexHull volume, returning 0.0 for degenerate point sets."""
    if len(points) < 4:
        return 0.0
    try:
        return float(ConvexHull(points).volume)
    except Exception:
        return 0.0


def _hull_intersection_volume(
    pts_a: np.ndarray, pts_b: np.ndarray,
) -> tuple[float, float, float]:
    """Compute convex hull volumes and their intersection volume.

    Returns (vol_a, vol_b, vol_intersection).
    Points from A inside B's hull and vice versa define the intersection.
    """
    vol_a = _safe_hull_volume(pts_a)
    vol_b = _safe_hull_volume(pts_b)

    if vol_a == 0.0 or vol_b == 0.0:
        return vol_a, vol_b, 0.0

    try:
        hull_a = ConvexHull(pts_a)
        hull_b = ConvexHull(pts_b)
        del_a = Delaunay(pts_a[hull_a.vertices])
        del_b = Delaunay(pts_b[hull_b.vertices])
    except Exception:
        return vol_a, vol_b, 0.0

    a_in_b = del_b.find_simplex(pts_a) >= 0
    b_in_a = del_a.find_simplex(pts_b) >= 0

    inter_pts = np.vstack([pts_a[a_in_b], pts_b[b_in_a]]) if (a_in_b.any() or b_in_a.any()) else np.zeros((0, 3))
    vol_inter = _safe_hull_volume(inter_pts)

    return vol_a, vol_b, vol_inter


def _compute_workspace(
    robot: BaselineRobot,
    n_samples: int,
    l_ref_m: float,
    results: BaselineResults,
) -> None:
    """Workspace volume + intersection via convex hulls.

    Each finger is sampled independently for better workspace coverage.
    """
    thumb_pts = _sobol_samples_finger(robot, robot.thumb, n_samples, seed=42)
    index_pts = _sobol_samples_finger(robot, robot.index, n_samples, seed=99)

    vol_t, vol_i, vol_inter = _hull_intersection_volume(thumb_pts, index_pts)

    results.thumb_workspace_vol_m3 = vol_t
    results.index_workspace_vol_m3 = vol_i
    results.avg_workspace_vol_m3 = (vol_t + vol_i) / 2.0

    l3 = l_ref_m ** 3 if l_ref_m > 0 else 1.0
    results.thumb_workspace_vol_norm = vol_t / l3
    results.index_workspace_vol_norm = vol_i / l3
    results.avg_workspace_vol_norm = results.avg_workspace_vol_m3 / l3

    results.intersection_vol_m3 = vol_inter
    results.opposability_index = vol_inter / l3
    union_vol = vol_t + vol_i - vol_inter
    results.jaccard_similarity = vol_inter / union_vol if union_vol > 0 else 0.0


# ── Baseline 5 & 6: Yoshikawa & GCI (FK+Jacobians pass) ─────────────────

def _compute_manipulability(
    robot: BaselineRobot,
    configs: np.ndarray,
    results: BaselineResults,
) -> None:
    n = configs.shape[0]
    SINGULAR_THRESH = 1e-10

    # Accumulators
    thumb_yoshi = np.empty(n)
    index_yoshi = np.empty(n)
    combined_yoshi = np.empty(n)
    thumb_log = np.empty(n)
    index_log = np.empty(n)
    combined_log = np.empty(n)
    thumb_inv_cond = np.empty(n)
    index_inv_cond = np.empty(n)
    combined_inv_cond = np.empty(n)
    thumb_singular = np.zeros(n, dtype=bool)
    index_singular = np.zeros(n, dtype=bool)
    combined_singular = np.zeros(n, dtype=bool)

    q = np.empty(robot.model.nq)
    for i in range(n):
        np.copyto(q, configs[i])
        forward_kinematics_jacobians(robot, q)

        Jt = fingertip_linear_jacobian(robot, robot.thumb)
        Ji = fingertip_linear_jacobian(robot, robot.index)
        Jc = combined_linear_jacobian(robot)

        for J, yoshi_arr, log_arr, inv_arr, sing_arr, idx in [
            (Jt, thumb_yoshi, thumb_log, thumb_inv_cond, thumb_singular, i),
            (Ji, index_yoshi, index_log, index_inv_cond, index_singular, i),
            (Jc, combined_yoshi, combined_log, combined_inv_cond, combined_singular, i),
        ]:
            sv = np.linalg.svd(J, compute_uv=False)
            sv_min = float(sv[-1]) if len(sv) > 0 else 0.0
            sv_max = float(sv[0]) if len(sv) > 0 else 0.0

            w = float(np.prod(sv))
            yoshi_arr[idx] = w
            log_arr[idx] = math.log(w) if w > 0 else -math.inf

            if sv_min < SINGULAR_THRESH:
                sing_arr[idx] = True
                inv_arr[idx] = 0.0
            else:
                inv_arr[idx] = sv_min / sv_max

    # Aggregate Yoshikawa
    results.thumb_yoshikawa_mean = float(np.mean(thumb_yoshi))
    results.thumb_yoshikawa_max = float(np.max(thumb_yoshi))
    results.index_yoshikawa_mean = float(np.mean(index_yoshi))
    results.index_yoshikawa_max = float(np.max(index_yoshi))
    results.combined_yoshikawa_mean = float(np.mean(combined_yoshi))
    results.combined_yoshikawa_max = float(np.max(combined_yoshi))

    for log_arr, attr in [
        (thumb_log, "thumb_log_manip_mean"),
        (index_log, "index_log_manip_mean"),
        (combined_log, "combined_log_manip_mean"),
    ]:
        finite = log_arr[np.isfinite(log_arr)]
        setattr(results, attr, float(np.mean(finite)) if len(finite) > 0 else -math.inf)

    for inv_arr, sing_arr, gci_attr, frac_attr in [
        (thumb_inv_cond, thumb_singular, "thumb_gci", "thumb_frac_singular"),
        (index_inv_cond, index_singular, "index_gci", "index_frac_singular"),
        (combined_inv_cond, combined_singular, "combined_gci", "combined_frac_singular"),
    ]:
        n_sing = int(np.sum(sing_arr))
        frac_sing = n_sing / n if n > 0 else 0.0
        setattr(results, frac_attr, frac_sing)
        non_singular = inv_arr[~sing_arr]
        setattr(results, gci_attr, float(np.mean(non_singular)) if len(non_singular) > 0 else 0.0)


# ── Orchestrator ──────────────────────────────────────────────────────────

def compute_all_baselines(
    robot: BaselineRobot,
    l_ref_m: float,
    n_samples: int = 50_000,
) -> BaselineResults:
    """Compute all 6 baselines for a loaded robot.

    Parameters
    ----------
    robot : loaded BaselineRobot
    l_ref_m : reference length in metres (for normalization)
    n_samples : target number of Sobol samples (rounded up to power of 2)

    Returns
    -------
    BaselineResults with all fields populated.
    """
    results = BaselineResults(robot_name=robot.robot_name)
    results.l_ref_m = l_ref_m

    # Baselines 1-2: no sampling needed
    _compute_dof_count(robot, results)
    _compute_joint_range(robot, results)

    # Baselines 3-4: workspace volume + intersection (per-finger convex hulls)
    _compute_workspace(robot, n_samples, l_ref_m, results)

    # Baselines 5-6: Yoshikawa + GCI (joint sampling, FK+Jacobians)
    configs = _sobol_samples_joint(robot, n_samples)
    results.n_samples_actual = configs.shape[0]
    _compute_manipulability(robot, configs, results)

    return results
