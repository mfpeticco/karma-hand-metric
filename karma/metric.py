"""The KaRMA metric: Translation BFS + Parallel Rotation Exploration.

Result types and output writing live in `results.py`; this module is the search
and scoring that computes a `MetricResult`.

Phase 1: Translation-only BFS to discover all reachable voxels.
         - Uses only translation primitives (no rotation primitives)
         - Tracks voxels, not (voxel, ori_bin) states
         - Stores one valid configuration per voxel

Phase 2: Parallel rotation exploration at each discovered voxel.
         - For each voxel, explore all rotation primitives
         - Runs in parallel using ProcessPoolExecutor
         - Counts orientation bins reached at each voxel

This separation ensures that translation coverage and rotation coverage are
measured independently, and rotations don't consume the translation budget.
"""
from __future__ import annotations

import logging
import multiprocessing
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np

from .config import MetricConfig
from .robot import RobotContext, load_robot, update_kinematics, active_position_indices, active_velocity_indices
from .contacts import contact_kinematics, select_contact_pair, snapshot_ck
from . import collisions, grasp
from .orientation import (
    axis_to_healpix_bin, healpix_npix_lines, contact_line_axis_body,
    compute_motion_primitives_frame, compute_rotation_primitives,
    pinch_axis_world,
)
from .rolling_qp import solve_rotation_step
from .seed_selection import select_seed_and_frame, _run_trial_bfs, _compute_trial_frame
from .seed_selection import SeedResult, _attempt_primitive_step
from .results import MetricResult, StateNode, VoxelIdx, State
from .projection import _project_onto_contacts
from .math_utils import exp_so3, point_to_voxel

logger = logging.getLogger(__name__)


def _reorthogonalize(R: np.ndarray) -> np.ndarray:
    """Project a near-SO(3) matrix back onto SO(3) via SVD."""
    U, _, Vt = np.linalg.svd(R)
    R_clean = U @ Vt
    # Ensure det = +1 (proper rotation, not reflection)
    if np.linalg.det(R_clean) < 0:
        U[:, -1] *= -1
        R_clean = U @ Vt
    return R_clean


@dataclass
class VoxelConfig:
    """Representative configuration for a voxel (used for rotation exploration)."""
    voxel: VoxelIdx
    centre_world: np.ndarray
    q: np.ndarray
    R_sphere: np.ndarray
    contact_pair: tuple[str, str]
    ori_bin: int  # Initial orientation bin at this voxel
    parent_voxel: VoxelIdx | None = None  # Parent voxel for path playback
    arrival_prim: int = -1  # Index of primitive used to reach this voxel (-1 = seed)


@dataclass
class OrientationConfig:
    """Configuration for a specific orientation at a voxel."""
    ori_bin: int
    q: np.ndarray
    R_sphere: np.ndarray
    contact_pair: tuple[str, str]


def _explore_rotations_at_voxel(
    voxel_config: VoxelConfig,
    cfg: MetricConfig,
    ablation_level: str | None,
) -> tuple[VoxelIdx, dict[int, OrientationConfig], dict[str, int]]:
    """Explore all rotation primitives at a single voxel.

    Runs in a worker process. ``cfg`` is the parent's frozen MetricConfig (passed
    directly since it pickles cleanly); the robot context is rebuilt here because
    the Pinocchio model does not pickle. Returns (voxel, ori_configs, fail_reasons).
    """
    ctx = load_robot(cfg.urdf_path, cfg)

    # Apply ablation patches (passed explicitly since forkserver doesn't inherit env vars)
    from .ablation import apply_worker_ablation_if_needed
    apply_worker_ablation_if_needed(ctx, cfg, level_str=ablation_level)

    q_ids = active_position_indices(ctx.model, cfg.finger_joint_names)
    v_ids = active_velocity_indices(ctx.model, cfg.finger_joint_names)
    q_lower = np.array(ctx.model.lowerPositionLimit)[q_ids]
    q_upper = np.array(ctx.model.upperPositionLimit)[q_ids]

    sphere_radius_m = cfg.sphere_radius_m
    link_radius_m = cfg.link_radius_m
    eps_g_m = cfg.eps_g_m
    eps_col_m = cfg.eps_col_m
    max_joint_step_rad = cfg.max_joint_step_rad
    friction_coeff = cfg.friction_coeff
    healpix_nside = cfg.healpix_nside
    thumb_contact_links = cfg.thumb_contact_links
    index_contact_links = cfg.index_contact_links
    rotation_slip_threshold_mm = cfg.rotation_slip_threshold_mm
    qp_regularization = cfg.qp_regularization

    # Start with the initial orientation config
    ori_configs: dict[int, OrientationConfig] = {
        voxel_config.ori_bin: OrientationConfig(
            ori_bin=voxel_config.ori_bin,
            q=voxel_config.q.copy(),
            R_sphere=voxel_config.R_sphere.copy(),
            contact_pair=voxel_config.contact_pair,
        )
    }
    fail_reasons: dict[str, int] = {}

    def _fail(reason: str) -> None:
        fail_reasons[reason] = fail_reasons.get(reason, 0) + 1

    # Get initial contact kinematics
    update_kinematics(ctx, voxel_config.q)
    ck_t = contact_kinematics(
        ctx, voxel_config.q, voxel_config.centre_world,
        voxel_config.contact_pair[0], sphere_radius_m, link_radius_m,
    )
    ck_i = contact_kinematics(
        ctx, voxel_config.q, voxel_config.centre_world,
        voxel_config.contact_pair[1], sphere_radius_m, link_radius_m,
    )

    # Phase-2 rotation parameters. Both are fixed angular guards (not scaled by
    # L_ref, and not swept/tuned — see the efficiency note below).
    ROTATION_STEP_RAD = 0.15  # ~8.6 deg per tilt step; loosely tracks the ~10 deg HEALPix bin
    MAX_ROT_STEPS = 50        # safety cap per direction; in practice the slip/efficiency
                              # guards end a direction long before this binds

    pinch_w = pinch_axis_world(ck_t.normal_world, ck_i.normal_world)
    rot_primitives = compute_rotation_primitives(pinch_w, ROTATION_STEP_RAD)

    for delta_theta_dir in rot_primitives:
        # Start from voxel config and keep rotating in this direction
        q_cur = voxel_config.q.copy()
        R_cur = voxel_config.R_sphere.copy()
        p_cur = voxel_config.centre_world.copy()
        cp_cur = voxel_config.contact_pair

        for _rot_step in range(MAX_ROT_STEPS):
            # Get current contact kinematics
            update_kinematics(ctx, q_cur)
            ck_t_cur = contact_kinematics(
                ctx, q_cur, p_cur, cp_cur[0],
                sphere_radius_m, link_radius_m,
            )
            ck_i_cur = contact_kinematics(
                ctx, q_cur, p_cur, cp_cur[1],
                sphere_radius_m, link_radius_m,
            )

            rot_result = solve_rotation_step(
                [ck_t_cur, ck_i_cur],
                delta_theta_dir,
                q_cur[q_ids],
                q_lower, q_upper,
                v_ids,
                sphere_radius_m,
                link_radius_m,
                eps_g_m,
                max_joint_step_rad,
                qp_regularization=qp_regularization,
                ctx=ctx,
            )

            if not rot_result.success:
                _fail("rot_qp_fail"); break

            # Stop this direction once a step achieves under 30% of the commanded
            # rotation: near a joint-limit/singular configuration the QP returns a
            # heavily attenuated step and further tilting yields little new coverage.
            # The 0.3 value is a fixed guard, not tuned; in practice the tangential-slip
            # gate below rejects far more steps than this cutoff does.
            if rot_result.efficiency < 0.3:
                _fail("rot_low_eff"); break

            # Reject rotation steps whose tangential slip exceeds the threshold
            if rotation_slip_threshold_mm is not None:
                if rot_result.tangential_slip > rotation_slip_threshold_mm:
                    _fail("rot_slip"); break

            # Apply the rotation
            q_rot = q_cur.copy()
            q_rot[q_ids] += rot_result.delta_q_active
            R_rot = exp_so3(rot_result.delta_theta_achieved) @ R_cur
            p_rot = p_cur.copy()

            # Re-select contact pair and project
            update_kinematics(ctx, q_rot)
            cp_rot = select_contact_pair(
                ctx, q_rot, p_rot,
                thumb_contact_links, index_contact_links,
                sphere_radius_m, link_radius_m,
                current_pair=cp_cur, hysteresis_m=eps_g_m * 0.5,
            )
            q_rot, proj_dtheta, proj_ok, ck_t_rot, ck_i_rot = _project_onto_contacts(
                ctx, q_rot, p_rot, cp_rot, q_ids, v_ids,
                sphere_radius_m, link_radius_m, eps_g_m,
            )
            if not proj_ok:
                _fail("rot_projection"); break
            R_rot = exp_so3(proj_dtheta) @ R_rot

            # Validate - use cached kinematics from projection
            if ck_t_rot is None or ck_i_rot is None:
                _fail("rot_no_kinematics"); break
            if abs(ck_t_rot.g_m) > eps_g_m or abs(ck_i_rot.g_m) > eps_g_m:
                _fail("rot_gap"); break

            col_ok, _ = collisions.check_collisions(
                ctx, q_rot, p_rot, cp_rot,
                sphere_radius_m, link_radius_m, eps_col_m,
            )
            if not col_ok:
                _fail("rot_collision"); break

            self_ok, _ = collisions.check_self_collisions(
                ctx, thumb_contact_links, index_contact_links,
                link_radius_m, eps_col_m,
            )
            if not self_ok:
                _fail("rot_self_collision"); break

            normals_rot = np.array([ck_t_rot.normal_world, ck_i_rot.normal_world])
            grasp_rot = grasp.check_antipodal_friction(normals_rot, friction_coeff)
            if not grasp_rot.feasible:
                _fail("rot_grasp"); break

            if np.any(q_rot[q_ids] < q_lower - 1e-6) or np.any(q_rot[q_ids] > q_upper + 1e-6):
                _fail("rot_joint_limits"); break

            # Compute new orientation bin
            axis_b_rot = contact_line_axis_body(
                ck_t_rot.normal_world, ck_i_rot.normal_world, R_rot,
            )
            bin_rot = axis_to_healpix_bin(axis_b_rot, healpix_nside)

            # Store config for this orientation bin (only if new)
            if bin_rot not in ori_configs:
                ori_configs[bin_rot] = OrientationConfig(
                    ori_bin=bin_rot,
                    q=q_rot.copy(),
                    R_sphere=_reorthogonalize(R_rot),
                    contact_pair=cp_rot,
                )

            # Update current state for next iteration
            q_cur = q_rot
            R_cur = R_rot
            cp_cur = cp_rot

    return voxel_config.voxel, ori_configs, fail_reasons


def _score_ensemble_seed(
    seed: SeedResult,
    cfg: MetricConfig,
    ctx: RobotContext,
    max_voxels: int,
    n_rotation_workers: int,
    ablation_level: str | None,
) -> tuple[int, float]:
    """Run Phase 1 + Phase 2 for a single seed, return (n_voxels, top5_rot_mean).

    Used for ensemble scoring of seeds #2 and #3. The best seed (#1) gets
    the full pipeline with state graph; this is a lightweight scoring-only path.
    """
    q_ids = active_position_indices(ctx.model, cfg.finger_joint_names)
    v_ids = active_velocity_indices(ctx.model, cfg.finger_joint_names)
    q_lower = np.array(ctx.model.lowerPositionLimit)[q_ids]
    q_upper = np.array(ctx.model.upperPositionLimit)[q_ids]

    R_seed = _compute_trial_frame(cfg, ctx, seed, v_ids)
    primitives = compute_motion_primitives_frame(R_seed, cfg.voxel_size_m)

    # Phase 1: trial BFS with full budget, returning voxel configs
    result = _run_trial_bfs(
        cfg, ctx, seed, R_seed, trial_budget=max_voxels,
        q_ids=q_ids, v_ids=v_ids, q_lower=q_lower, q_upper=q_upper,
        primitives=primitives,
        return_configs=True,
    )
    stats, discovered = result
    n_voxels = stats.n_voxels

    if n_voxels == 0:
        return 0, 0.0

    # Convert _TrialNode -> VoxelConfig for Phase 2
    voxel_config_list: list[VoxelConfig] = []
    for voxel, node in discovered.items():
        # Compute orientation bin at this voxel
        update_kinematics(ctx, node.q)
        ck_t = contact_kinematics(
            ctx, node.q, node.p, node.contact_pair[0],
            cfg.sphere_radius_m, cfg.link_radius_m,
        )
        ck_i = contact_kinematics(
            ctx, node.q, node.p, node.contact_pair[1],
            cfg.sphere_radius_m, cfg.link_radius_m,
        )
        axis_b = contact_line_axis_body(ck_t.normal_world, ck_i.normal_world, node.R_sphere)
        ori_bin = axis_to_healpix_bin(axis_b, cfg.healpix_nside)

        voxel_config_list.append(VoxelConfig(
            voxel=voxel,
            centre_world=node.p.copy(),
            q=node.q.copy(),
            R_sphere=_reorthogonalize(node.R_sphere),
            contact_pair=node.contact_pair,
            ori_bin=ori_bin,
        ))

    # Phase 2: parallel rotation exploration
    total_bins = healpix_npix_lines(cfg.healpix_nside)
    voxel_ori: dict[VoxelIdx, set[int]] = {}

    with ProcessPoolExecutor(
        max_workers=n_rotation_workers,
        mp_context=multiprocessing.get_context("forkserver"),
    ) as executor:
        futures = {
            executor.submit(_explore_rotations_at_voxel, vc, cfg, ablation_level): vc.voxel
            for vc in voxel_config_list
        }
        for future in as_completed(futures):
            voxel, ori_configs, _ = future.result()
            voxel_ori[voxel] = set(ori_configs.keys())

    # Compute top-5 voxel mean rotation coverage
    rot_covs = sorted(
        [len(bins) / total_bins for bins in voxel_ori.values()],
        reverse=True,
    )
    top5 = rot_covs[:5]
    top5_rot_mean = float(np.mean(top5)) if top5 else 0.0

    logger.info(
        "Ensemble seed scored: n_voxels=%d, top5_rot=%.4f",
        n_voxels, top5_rot_mean,
    )
    return n_voxels, top5_rot_mean


def compute_metric(
    cfg: MetricConfig,
    ctx: RobotContext,
    max_voxels: int | None = None,
    n_rotation_workers: int | None = None,
    precomputed_seed: tuple | None = None,
) -> MetricResult:
    """Run the two-phase metric pipeline.

    Phase 1: Translation-only BFS to discover voxels
    Phase 2: Parallel rotation exploration at each voxel

    Args:
        cfg: Metric configuration
        ctx: Pinocchio robot context
        max_voxels: Maximum voxels for Phase 1 (default: cfg.max_states)
        n_rotation_workers: Number of parallel workers for Phase 2
            (default: cfg.rotation_workers)
        precomputed_seed: Optional (seed, R_seed, seed_sensitivity, top_seeds) tuple
            to skip seed selection. Used by constraint ablation to ensure the same
            seed is used across all ablation levels.

    Returns:
        MetricResult with voxel and orientation coverage data.
    """
    if max_voxels is None:
        max_voxels = cfg.max_states
    if n_rotation_workers is None:
        n_rotation_workers = cfg.rotation_workers

    q_ids = active_position_indices(ctx.model, cfg.finger_joint_names)
    v_ids = active_velocity_indices(ctx.model, cfg.finger_joint_names)
    q_lower = np.array(ctx.model.lowerPositionLimit)[q_ids]
    q_upper = np.array(ctx.model.upperPositionLimit)[q_ids]

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 0: Seed and frame selection (multi-seed trial BFS)
    # ══════════════════════════════════════════════════════════════════════════
    if precomputed_seed is not None:
        seed, R_seed, seed_sensitivity, top_seeds = precomputed_seed
        logger.info("Using precomputed seed (skipping seed selection)")
    else:
        logger.info("Selecting seed and grid frame (multi-seed trial BFS)...")
        seed, R_seed, seed_sensitivity, top_seeds = select_seed_and_frame(cfg, ctx)
    R_seed_inv = R_seed.T

    if not seed.feasible:
        logger.warning("Seed is not fully feasible (max_gap=%.4f mm)", seed.max_gap * 1000)
    logger.info(
        "Seed: contact_pair=%s, max_gap=%.4f mm, feasible=%s",
        seed.contact_pair, seed.max_gap * 1000, seed.feasible,
    )

    # Initial state
    origin = seed.sphere_centre_world.copy()
    R0 = np.eye(3)
    update_kinematics(ctx, seed.q)

    ck_thumb = contact_kinematics(
        ctx, seed.q, seed.sphere_centre_world,
        seed.contact_pair[0], cfg.sphere_radius_m, cfg.link_radius_m,
    )
    ck_index = contact_kinematics(
        ctx, seed.q, seed.sphere_centre_world,
        seed.contact_pair[1], cfg.sphere_radius_m, cfg.link_radius_m,
    )

    axis_body = contact_line_axis_body(
        ck_thumb.normal_world, ck_index.normal_world, R0,
    )
    bin0 = axis_to_healpix_bin(axis_body, cfg.healpix_nside)
    voxel0 = (0, 0, 0)

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 1: Translation-only BFS
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("Phase 1: Translation-only BFS (max_voxels=%d)...", max_voxels)

    # Track voxels and store one representative config per voxel
    voxel_configs: dict[VoxelIdx, VoxelConfig] = {
        voxel0: VoxelConfig(
            voxel=voxel0,
            centre_world=seed.sphere_centre_world.copy(),
            q=seed.q.copy(),
            R_sphere=R0.copy(),
            contact_pair=seed.contact_pair,
            ori_bin=bin0,
        )
    }

    queue: deque[VoxelIdx] = deque([voxel0])
    n_expanded = 0
    n_attempted = 0
    n_succeeded = 0
    fail_reasons: dict[str, int] = {}

    def _fail(reason: str) -> None:
        fail_reasons[reason] = fail_reasons.get(reason, 0) + 1

    # Sub-steps for linearization accuracy
    N_SUB = max(1, int(np.ceil(cfg.voxel_size_m / cfg.step_size_m)))

    # Seed-frame translation primitives (no rotation primitives in Phase 1)
    primitives = compute_motion_primitives_frame(R_seed, cfg.voxel_size_m)

    # This Phase 1 BFS shares its per-step kernel (_attempt_primitive_step) with the
    # trial BFS in seed_selection._run_trial_bfs. The two outer loops deliberately
    # differ in bookkeeping: this one reorthogonalizes R_sphere and records the
    # orientation bin to build the full state graph, whereas the trial BFS keeps
    # raw nodes and effort counters for fast seed ranking.
    while queue and len(voxel_configs) < max_voxels:
        voxel = queue.popleft()
        vc = voxel_configs[voxel]
        n_expanded += 1

        # Compute kinematics at current voxel config (snapshot to detach from Pinocchio buffers)
        update_kinematics(ctx, vc.q)
        _ck_t_start = snapshot_ck(contact_kinematics(
            ctx, vc.q, vc.centre_world, vc.contact_pair[0],
            cfg.sphere_radius_m, cfg.link_radius_m,
        ))
        _ck_i_start = snapshot_ck(contact_kinematics(
            ctx, vc.q, vc.centre_world, vc.contact_pair[1],
            cfg.sphere_radius_m, cfg.link_radius_m,
        ))

        for prim_idx, delta_p_full in enumerate(primitives):
            if len(voxel_configs) >= max_voxels:
                break
            # Skip the reverse primitive — it targets the parent voxel, which is
            # already discovered. Primitives come ±paired so the reverse of i is
            # i ^ 1 (see compute_motion_primitives_frame).
            if vc.arrival_prim >= 0 and prim_idx == (vc.arrival_prim ^ 1):
                continue
            n_attempted += 1
            outcome = _attempt_primitive_step(
                ctx, cfg, delta_p_full, N_SUB,
                vc.q, vc.centre_world, vc.R_sphere, vc.contact_pair,
                _ck_t_start, _ck_i_start,
                q_ids, v_ids, q_lower, q_upper,
            )
            if not outcome.success:
                _fail(outcome.fail_reason)
                continue

            # Compute voxel; skip if already discovered
            voxel_next = point_to_voxel(outcome.p, origin, cfg.voxel_size_m, R_seed_inv)
            if voxel_next in voxel_configs:
                _fail("duplicate_voxel")
                continue

            # Orientation bin for this config (raw R_sphere, as before Phase 2)
            axis_b = contact_line_axis_body(
                outcome.ck_t.normal_world, outcome.ck_i.normal_world, outcome.R_sphere,
            )
            bin_next = axis_to_healpix_bin(axis_b, cfg.healpix_nside)

            n_succeeded += 1
            voxel_configs[voxel_next] = VoxelConfig(
                voxel=voxel_next,
                centre_world=outcome.p.copy(),
                q=outcome.q.copy(),
                R_sphere=_reorthogonalize(outcome.R_sphere),
                contact_pair=outcome.contact_pair,
                ori_bin=bin_next,
                parent_voxel=voxel,  # Track parent for path playback
                arrival_prim=prim_idx,
            )
            queue.append(voxel_next)

        if n_expanded % 100 == 0:
            logger.info(
                "Phase 1 BFS: expanded=%d, voxels=%d",
                n_expanded, len(voxel_configs),
            )

    logger.info(
        "Phase 1 complete: expanded=%d, attempted=%d, succeeded=%d, voxels=%d",
        n_expanded, n_attempted, n_succeeded, len(voxel_configs),
    )
    logger.info("Phase 1 failure breakdown: %s", fail_reasons)

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 2: Parallel rotation exploration
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("Phase 2: Parallel rotation exploration (%d voxels, %d workers)...",
                len(voxel_configs), n_rotation_workers)

    # Internal IPC for the constraint-ablation experiment: ablation_context sets
    # this env var so Phase 2 workers re-apply the same patches (see
    # karma/ablation.py). Do not set it by hand; outside ablation_context it
    # would ablate the workers while Phase 1 runs unablated.
    ablation_level = os.environ.get("KARMA_ABLATION")
    voxel_ori: dict[VoxelIdx, set[int]] = {}
    voxel_ori_configs: dict[VoxelIdx, dict[int, OrientationConfig]] = {}
    phase2_fail_reasons: dict[str, int] = {}

    # Run rotation exploration in parallel
    voxel_list = list(voxel_configs.values())

    with ProcessPoolExecutor(max_workers=n_rotation_workers,
                             mp_context=multiprocessing.get_context("forkserver")) as executor:
        futures = {
            executor.submit(_explore_rotations_at_voxel, vc, cfg, ablation_level): vc.voxel
            for vc in voxel_list
        }

        completed = 0
        for future in as_completed(futures):
            voxel, ori_configs, fail_reasons_voxel = future.result()
            voxel_ori[voxel] = set(ori_configs.keys())
            voxel_ori_configs[voxel] = ori_configs

            # Aggregate failure reasons
            for reason, count in fail_reasons_voxel.items():
                phase2_fail_reasons[reason] = phase2_fail_reasons.get(reason, 0) + count

            completed += 1
            if completed % 50 == 0:
                logger.info("Phase 2: %d/%d voxels processed", completed, len(voxel_list))

    logger.info("Phase 2 complete: %d voxels explored", len(voxel_ori))
    logger.info("Phase 2 failure breakdown: %s", phase2_fail_reasons)

    # ══════════════════════════════════════════════════════════════════════════
    # Build result (compatible with existing MetricResult format)
    # ══════════════════════════════════════════════════════════════════════════

    # Build state nodes from all orientation configs at each voxel
    # This allows rotation playback to work properly
    nodes: dict[State, StateNode] = {}

    for voxel, vc in voxel_configs.items():
        # Compute parent state for path traversal
        parent_state: State | None = None
        if vc.parent_voxel is not None:
            parent_vc = voxel_configs[vc.parent_voxel]
            parent_state = (vc.parent_voxel, parent_vc.ori_bin)

        # Add all orientation states for this voxel from Phase 2
        ori_configs = voxel_ori_configs.get(voxel, {})
        for ori_bin, oc in ori_configs.items():
            state: State = (voxel, ori_bin)
            nodes[state] = StateNode(
                voxel=voxel,
                ori_bin=ori_bin,
                centre_world=vc.centre_world.copy(),
                q=oc.q.copy(),
                R_sphere=oc.R_sphere.copy(),
                contact_pair=oc.contact_pair,
                parent=parent_state,  # Path goes through parent voxel's initial ori_bin
            )

    # Scoring
    total_bins = healpix_npix_lines(cfg.healpix_nside)
    n_voxels = len(voxel_ori)
    n_states = len(nodes)  # Total (voxel, ori) pairs with configurations

    # Translational volume (best seed, for state graph consistency)
    voxel_vol = cfg.voxel_size_m ** 3
    l_ref = cfg.l_ref_m
    translational_volume_m3 = n_voxels * voxel_vol

    # Rotational coverage per voxel (best seed)
    rot_cov: dict[VoxelIdx, float] = {}
    for v, bins in voxel_ori.items():
        rot_cov[v] = len(bins) / total_bins

    # ══════════════════════════════════════════════════════════════════════════
    # Top-3 seed ensemble for KaRMA-T and KaRMA-R
    # ══════════════════════════════════════════════════════════════════════════
    # Best seed's scores
    best_n_voxels = float(n_voxels)
    rot_covs_best = sorted(rot_cov.values(), reverse=True)
    top5_best = rot_covs_best[:5]
    best_top5_rot = float(np.mean(top5_best)) if top5_best else 0.0

    ensemble_t = [best_n_voxels]
    ensemble_r = [best_top5_rot]

    # Run Phase 1 + Phase 2 on seeds #2 and #3 (if available)
    extra_seeds = [(v, s) for v, s in top_seeds[1:] if s is not seed]
    if extra_seeds:
        logger.info(
            "Ensemble: scoring %d additional seeds (Phase 1 + Phase 2)...",
            len(extra_seeds),
        )
        for rank, (trial_v, trial_seed) in enumerate(extra_seeds, start=2):
            logger.info("Ensemble seed #%d: trial BFS had %d voxels", rank, int(trial_v))
            try:
                ens_nvox, ens_rot = _score_ensemble_seed(
                    trial_seed, cfg, ctx, max_voxels, n_rotation_workers, ablation_level,
                )
                ensemble_t.append(float(ens_nvox))
                ensemble_r.append(ens_rot)
                logger.info(
                    "Ensemble seed #%d: n_voxels=%d, top5_rot=%.4f",
                    rank, ens_nvox, ens_rot,
                )
            except Exception as e:
                # Exclude the failed seed from the top-3 means rather than
                # counting it as 0.0, which would silently depress KaRMA-R.
                # Surface the failure loudly instead of fabricating a value.
                logger.error(
                    "Ensemble seed #%d failed, excluding from scores: %s",
                    rank, e, exc_info=True,
                )
                if seed_sensitivity is not None:
                    seed_sensitivity.setdefault("ensemble_failures", []).append(
                        {"rank": rank, "error": str(e)}
                    )

    # Average across all ensemble seeds
    mean_t = float(np.mean(ensemble_t))
    mean_r = float(np.mean(ensemble_r))

    translational_score = (mean_t * voxel_vol) / (l_ref ** 3) if l_ref > 0 else 0.0
    global_rot = mean_r

    # Store ensemble details in seed_sensitivity
    if seed_sensitivity is not None:
        seed_sensitivity["ensemble_t"] = [round(v, 2) for v in ensemble_t]
        seed_sensitivity["ensemble_r"] = [round(v, 4) for v in ensemble_r]
        seed_sensitivity["top3_mean"] = round(mean_t, 2)

    logger.info(
        "Two-phase metric complete: voxels=%d, ensemble_T=[%s] mean=%.1f, "
        "ensemble_R=[%s] mean=%.4f, KaRMA-T=%.6f",
        n_voxels,
        ", ".join(f"{v:.0f}" for v in ensemble_t), mean_t,
        ", ".join(f"{v:.4f}" for v in ensemble_r), mean_r,
        translational_score,
    )

    return MetricResult(
        voxel_ori_bins=voxel_ori,
        state_nodes=nodes,
        n_voxels_reached=n_voxels,
        n_states_reached=n_states,
        total_ori_bins=total_bins,
        translational_score=translational_score,
        translational_volume_m3=translational_volume_m3,
        l_ref_m=l_ref,
        rotational_coverage_per_voxel=rot_cov,
        global_rotational_score=global_rot,
        seed_centre=seed.sphere_centre_world,
        seed_q=seed.q,
        seed_contact_pair=seed.contact_pair,
        seed_frame=R_seed,
        seed_sensitivity=seed_sensitivity,
        phase1_fail_reasons=fail_reasons,
        phase2_fail_reasons=phase2_fail_reasons,
    )
