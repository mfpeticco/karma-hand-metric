"""Exhaustive seed selection.

Generate ALL feasible IK solutions, evaluate every one at max_states,
pick the winner.  This is both optimal (guaranteed best seed) and
efficient because BFS self-terminates when the frontier exhausts —
weak seeds cost little even at high budgets.

Algorithm:
  1. Generate ALL feasible seeds (Sobol + linspace + random), light dedup only
  2. Polish seeds (push joints toward center of range in grasp nullspace)
  3. Evaluate ALL seeds at max_states via parallel trial BFS
  4. Pick the seed with the highest n_voxels
"""
from __future__ import annotations

import logging
import multiprocessing
import os
import platform
import subprocess
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy.stats.qmc import Sobol

from .config import MetricConfig
from .contacts import contact_kinematics, select_contact_pair, snapshot_ck, _capsule_axis_segment
from . import collisions, grasp
from .math_utils import (
    closest_points_on_segments, compute_object_jacobian, compute_seed_frame,
    exp_so3, point_to_voxel,
)
from .projection import _project_onto_contacts
from .results import VoxelIdx
from .orientation import compute_motion_primitives_frame
from .robot import (
    RobotContext, update_kinematics, update_positions_only,
    active_position_indices, active_velocity_indices,
    neutral_configuration, fold_mimic_jacobian,
)
from .rolling_qp import solve_rolling_step
from .seed import (
    SeedResult, _solve_for_centre, _candidate_sphere_centres,
    _solve_candidate_worker,
)

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _physical_core_count() -> int:
    """Return the number of physical CPU cores (not hyper-threaded logical CPUs).

    macOS: ``sysctl hw.physicalcpu``
    Linux: count unique physical cores in ``/proc/cpuinfo``
    Fallback: ``os.cpu_count()``
    """
    try:
        system = platform.system()
        if system == "Darwin":
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.physicalcpu"], text=True,
            )
            return int(out.strip())
        elif system == "Linux":
            with open("/proc/cpuinfo") as f:
                cores = set()
                phys_id, core_id = None, None
                for line in f:
                    if line.startswith("physical id"):
                        phys_id = line.split(":")[1].strip()
                    elif line.startswith("core id"):
                        core_id = line.split(":")[1].strip()
                    if phys_id is not None and core_id is not None:
                        cores.add((phys_id, core_id))
                        phys_id, core_id = None, None
                if cores:
                    return len(cores)
    except Exception:
        pass
    return os.cpu_count() or 1


def _deduplicate_seeds(
    seeds: list[SeedResult],
    q_ids: np.ndarray,
    min_dist: float = 0.1,
) -> list[SeedResult]:
    """Greedy dedup in active joint space — keep seed only if >= min_dist
    from all previously kept seeds. Preserves the input ordering."""
    if not seeds:
        return seeds
    kept: list[SeedResult] = []
    kept_qs: list[np.ndarray] = []
    for s in seeds:
        q_active = s.q[q_ids]
        if any(np.linalg.norm(q_active - kq) < min_dist for kq in kept_qs):
            continue
        kept.append(s)
        kept_qs.append(q_active)
    return kept


# ── Step 1: Generate feasible seeds ──────────────────────────────────────────


def _generate_feasible_seeds(
    cfg: MetricConfig,
    ctx: RobotContext,
) -> tuple[list[SeedResult], dict]:
    """Generate feasible seed configurations using Sobol + linspace sampling.

    Returns a tuple of:
      - ALL unique feasible seeds (light dedup at 0.01 rad) sorted by
        max_gap ascending.  Falls back to best non-feasible if none found.
      - generation_stats dict with pipeline counts (n_joint_configs,
        n_reachable, n_candidates, n_ik_solutions, n_feasible, n_after_dedup).
    """
    q0 = neutral_configuration(ctx.model)
    q_ids = active_position_indices(ctx.model, cfg.finger_joint_names)
    v_ids = active_velocity_indices(ctx.model, cfg.finger_joint_names)
    q_lower = np.array(ctx.model.lowerPositionLimit)[q_ids]
    q_upper = np.array(ctx.model.upperPositionLimit)[q_ids]
    n_active = len(q_ids)

    thumb_link = cfg.seed_thumb_link
    index_link = cfg.seed_index_link

    # Maximum distance for fingertips to be "within reach"
    seed_margin = max(abs(o) for o in cfg.seed_search_offsets_m) * 2.0 if cfg.seed_search_offsets_m else 0.02
    max_reach = 2.0 * (cfg.sphere_radius_m + cfg.link_radius_m) + seed_margin

    # Generate joint samples: linspace + Sobol + random
    configs: list[np.ndarray] = [q0.copy()]

    # Linspace per joint (one joint varies at a time)
    n_per_joint = 5
    for i in range(n_active):
        vals = np.linspace(q_lower[i], q_upper[i], n_per_joint)
        for v in vals:
            q = q0.copy()
            q[q_ids[i]] = v
            configs.append(q)

    # Sobol sequence for quasi-random coverage (power-of-2 count)
    sobol = Sobol(d=n_active, scramble=False)
    sobol_samples = sobol.random(512)  # (512, n_active) in [0, 1]
    for s in sobol_samples:
        q = q0.copy()
        q[q_ids] = q_lower + s * (q_upper - q_lower)
        configs.append(q)

    # Random samples (cover different basins than Sobol)
    rng = np.random.RandomState(42)
    for _ in range(200):
        q = q0.copy()
        q[q_ids] = q_lower + rng.rand(n_active) * (q_upper - q_lower)
        configs.append(q)

    logger.info("Seed selection: sampling %d joint configurations", len(configs))

    # Filter to reachable and generate candidates
    reachable_candidates: list[tuple[np.ndarray, np.ndarray]] = []
    n_reachable = 0

    for q_sample in configs:
        update_positions_only(ctx, q_sample)
        c0, d0, h0 = _capsule_axis_segment(ctx, thumb_link)
        c1, d1, h1 = _capsule_axis_segment(ctx, index_link)
        pt0, pt1 = closest_points_on_segments(c0, d0, h0, c1, d1, h1)
        dist = np.linalg.norm(pt1 - pt0)

        if dist > max_reach:
            continue
        n_reachable += 1

        candidates = _candidate_sphere_centres(
            ctx, q_sample, thumb_link, index_link, cfg.seed_search_offsets_m,
        )
        for centre in candidates:
            reachable_candidates.append((q_sample.copy(), centre))

    logger.info("Seed selection: %d reachable configs, %d candidates to solve",
                n_reachable, len(reachable_candidates))

    # Solve all candidates in parallel (reuses seed.py's worker infrastructure)
    results: list[SeedResult] = []
    if len(reachable_candidates) > 10:
        n_workers = min(_physical_core_count(), len(reachable_candidates))
        worker_args = [
            (cfg, q_sample, centre, thumb_link, index_link,
             q_ids.copy(), v_ids.copy(), q_lower.copy(), q_upper.copy())
            for q_sample, centre in reachable_candidates
        ]
        try:
            with ProcessPoolExecutor(max_workers=n_workers,
                                 mp_context=multiprocessing.get_context("forkserver")) as executor:
                raw = list(executor.map(_solve_candidate_worker, worker_args))
            results = [r for r in raw if r is not None]
        except Exception as e:
            logger.warning("Parallel IK failed (%s), falling back to sequential", e)
            results = []

    if not results:
        # Sequential fallback
        for q_sample, centre in reachable_candidates:
            result = _solve_for_centre(
                cfg, ctx, q_sample, centre, thumb_link, index_link,
                q_ids, v_ids, q_lower, q_upper,
            )
            if result is not None:
                results.append(result)

    # Separate feasible and non-feasible
    feasible = [r for r in results if r.feasible]
    nonfeasible = [r for r in results if not r.feasible]

    logger.info("Seed selection: %d valid, %d feasible, %d non-feasible",
                len(results), len(feasible), len(nonfeasible))

    # Build generation stats from pipeline variables
    gen_stats = {
        "n_joint_configs": len(configs),
        "n_reachable": n_reachable,
        "n_candidates": len(reachable_candidates),
        "n_ik_solutions": len(results),
        "n_feasible": len(feasible),
        "n_after_dedup": 0,  # updated below if feasible
    }

    if feasible:
        # Light dedup: remove only IK-identical solutions (< 0.01 rad)
        feasible = _deduplicate_seeds(feasible, q_ids, min_dist=0.01)
        # Sort by feasibility (smallest gap first), not manipulability
        feasible.sort(key=lambda r: (r.max_gap, tuple(r.q.round(8))))
        gen_stats["n_after_dedup"] = len(feasible)
        logger.info("Seed selection: %d unique seeds after light dedup", len(feasible))
        return feasible, gen_stats  # Return ALL; select_seed_and_frame evaluates every one

    # Fallback: best non-feasible
    if nonfeasible:
        nonfeasible.sort(key=lambda r: (r.max_gap, tuple(r.q.round(8))))
        gen_stats["n_feasible"] = 0
        gen_stats["n_after_dedup"] = 1
        return nonfeasible[:1], gen_stats

    # Last resort: neutral config
    update_kinematics(ctx, q0)
    c0, d0, h0 = _capsule_axis_segment(ctx, thumb_link)
    c1, d1, h1 = _capsule_axis_segment(ctx, index_link)
    pt0, pt1 = closest_points_on_segments(c0, d0, h0, c1, d1, h1)
    gen_stats["n_feasible"] = 0
    gen_stats["n_after_dedup"] = 1
    return [SeedResult(
        q=q0, sphere_centre_world=0.5 * (pt0 + pt1),
        contact_pair=(thumb_link, index_link),
        max_gap=float("inf"), feasible=False,
    )], gen_stats


# ── Step 2: Compute trial frame ─────────────────────────────────────────────


def _compute_trial_frame(
    cfg: MetricConfig,
    ctx: RobotContext,
    seed: SeedResult,
    v_ids: np.ndarray,
) -> np.ndarray:
    """Compute R_seed for a trial seed using the manipulability ellipsoid."""
    update_kinematics(ctx, seed.q)
    ck_thumb = contact_kinematics(
        ctx, seed.q, seed.sphere_centre_world,
        seed.contact_pair[0], cfg.sphere_radius_m, cfg.link_radius_m,
    )
    ck_index = contact_kinematics(
        ctx, seed.q, seed.sphere_centre_world,
        seed.contact_pair[1], cfg.sphere_radius_m, cfg.link_radius_m,
    )
    fold_fn = (lambda J, vids: fold_mimic_jacobian(J, vids, ctx)) if ctx.mimic_qs else None
    J_obj = compute_object_jacobian(ck_thumb, ck_index, v_ids, cfg.link_radius_m, fold_fn)
    return compute_seed_frame(J_obj)


# ── Step 3: Trial BFS ───────────────────────────────────────────────────────


@dataclass
class TrialBFSStats:
    """BFS effort statistics for effort-based scoring in saturated rounds."""
    n_voxels: int
    frontier_size: int
    attempts: int          # total neighbor expansions tried
    fail_moves: int        # QP/collision/friction/gap/joint-limit failures
    revisits: int          # landed in already-discovered voxel
    unique_boundary: int   # expanded voxels where no neighbor succeeded


@dataclass
class _TrialNode:
    """Lightweight BFS node for trial BFS (no ori_bin/parent)."""
    q: np.ndarray
    p: np.ndarray
    R_sphere: np.ndarray
    contact_pair: tuple[str, str]
    arrival_prim: int = -1  # Index of primitive used to reach this node (-1 = seed)


@dataclass
class StepOutcome:
    """Result of attempting one motion primitive from a translation-BFS node."""
    success: bool
    fail_reason: str | None = None
    q: np.ndarray | None = None
    p: np.ndarray | None = None
    R_sphere: np.ndarray | None = None
    contact_pair: tuple[str, str] | None = None
    ck_t: "object | None" = None  # ContactKinematics on success
    ck_i: "object | None" = None


def _attempt_primitive_step(
    ctx: RobotContext,
    cfg: MetricConfig,
    delta_p_full: np.ndarray,
    n_sub: int,
    q_start: np.ndarray,
    p_start: np.ndarray,
    R_start: np.ndarray,
    contact_pair: tuple[str, str],
    ck_t_start,
    ck_i_start,
    q_ids: np.ndarray,
    v_ids: np.ndarray,
    q_lower: np.ndarray,
    q_upper: np.ndarray,
) -> StepOutcome:
    """Roll the sphere one primitive step (``n_sub`` sub-steps) from a node,
    project back onto the contacts, and validate the result.

    This is the SHARED inner step of both translation-BFS paths — the best-seed
    BFS in ``metric.compute_metric`` and the trial/ensemble BFS
    in ``_run_trial_bfs`` — so the two can never drift. All bookkeeping
    (fail-reason tallies, voxel dedup, orientation bins, effort stats)
    stays in the callers; this returns only success + the resulting state.

    ``ck_t_start`` / ``ck_i_start`` are snapshotted contact kinematics at the
    node, reused for the first sub-step to skip a redundant FK. On success the
    ``q``/``p``/``R_sphere``/``contact_pair`` and post-projection ``ck_t``/``ck_i``
    fields are set; ``R_sphere`` is raw (not re-orthonormalized).
    """
    slip_threshold_m = (
        cfg.rotation_slip_threshold_mm / 1000.0
        if cfg.rotation_slip_threshold_mm is not None
        else None
    )
    sub_dp = delta_p_full / n_sub
    q_cur = q_start.copy()
    p_cur = p_start.copy()
    R_cur = R_start.copy()
    cp_cur = contact_pair
    cached_ck_t = None
    cached_ck_i = None

    for _sub in range(n_sub):
        if _sub == 0:
            # Reuse snapshotted CK (detached from Pinocchio buffers, no FK restore).
            ck_t = ck_t_start
            ck_i = ck_i_start
        else:
            update_kinematics(ctx, q_cur)
            ck_t = contact_kinematics(
                ctx, q_cur, p_cur, cp_cur[0],
                cfg.sphere_radius_m, cfg.link_radius_m,
            )
            ck_i = contact_kinematics(
                ctx, q_cur, p_cur, cp_cur[1],
                cfg.sphere_radius_m, cfg.link_radius_m,
            )

        qp_result = solve_rolling_step(
            [ck_t, ck_i],
            sub_dp,
            q_cur[q_ids],
            q_lower, q_upper,
            v_ids,
            cfg.sphere_radius_m,
            cfg.link_radius_m,
            cfg.eps_g_m,
            cfg.max_joint_step_rad,
            cfg.qp_regularization,
            ctx=ctx,
            tangential_slip_threshold_m=slip_threshold_m,
        )
        if not qp_result.success:
            return StepOutcome(False, "qp_fail")

        q_cur[q_ids] += qp_result.delta_q_active
        p_cur = p_cur + sub_dp
        R_cur = exp_so3(qp_result.delta_theta) @ R_cur

        update_kinematics(ctx, q_cur)
        cp_cur = select_contact_pair(
            ctx, q_cur, p_cur,
            cfg.thumb_contact_links, cfg.index_contact_links,
            cfg.sphere_radius_m, cfg.link_radius_m,
            current_pair=cp_cur, hysteresis_m=cfg.eps_g_m * 0.5,
        )
        q_cur, proj_dtheta, proj_ok, cached_ck_t, cached_ck_i = _project_onto_contacts(
            ctx, q_cur, p_cur, cp_cur, q_ids, v_ids,
            cfg.sphere_radius_m, cfg.link_radius_m, cfg.eps_g_m,
        )
        if not proj_ok:
            return StepOutcome(False, "projection")
        R_cur = exp_so3(proj_dtheta) @ R_cur

    # Final validation (post-projection), same order in both BFS paths.
    if cached_ck_t is None or cached_ck_i is None:
        return StepOutcome(False, "no_kinematics")
    if abs(cached_ck_t.g_m) > cfg.eps_g_m or abs(cached_ck_i.g_m) > cfg.eps_g_m:
        return StepOutcome(False, "gap_post_proj")

    col_ok, _ = collisions.check_collisions(
        ctx, q_cur, p_cur, cp_cur,
        cfg.sphere_radius_m, cfg.link_radius_m, cfg.eps_col_m,
    )
    if not col_ok:
        return StepOutcome(False, "collision")

    self_ok, _ = collisions.check_self_collisions(
        ctx, cfg.thumb_contact_links, cfg.index_contact_links,
        cfg.link_radius_m, cfg.eps_col_m,
    )
    if not self_ok:
        return StepOutcome(False, "self_collision")

    normals = np.array([cached_ck_t.normal_world, cached_ck_i.normal_world])
    if not grasp.check_antipodal_friction(normals, cfg.friction_coeff).feasible:
        return StepOutcome(False, "grasp")

    if np.any(q_cur[q_ids] < q_lower - 1e-6) or np.any(q_cur[q_ids] > q_upper + 1e-6):
        return StepOutcome(False, "joint_limits")

    return StepOutcome(
        True, None, q_cur, p_cur, R_cur, cp_cur, cached_ck_t, cached_ck_i,
    )


def _run_trial_bfs(
    cfg: MetricConfig,
    ctx: RobotContext,
    seed: SeedResult,
    R_seed: np.ndarray,
    trial_budget: int,
    q_ids: np.ndarray,
    v_ids: np.ndarray,
    q_lower: np.ndarray,
    q_upper: np.ndarray,
    primitives: list[np.ndarray],
    return_configs: bool = False,
) -> TrialBFSStats | tuple:
    """Run a trial BFS capped at trial_budget voxels.

    Uses the shared per-step kernel _attempt_primitive_step, same as the best-seed
    Phase 1 BFS in metric.py.
    Returns a TrialBFSStats with voxel count, frontier size, and effort
    statistics (attempts, failures, revisits, boundary richness) that
    enable effort-based scoring in saturated rounds.

    If *return_configs* is True, returns (stats, discovered_dict) where
    discovered_dict maps VoxelIdx -> _TrialNode with per-voxel configs.
    """
    R_inv = R_seed.T
    origin = seed.sphere_centre_world.copy()
    voxel0: VoxelIdx = (0, 0, 0)

    discovered: dict[VoxelIdx, _TrialNode] = {
        voxel0: _TrialNode(
            q=seed.q.copy(),
            p=origin.copy(),
            R_sphere=np.eye(3),
            contact_pair=seed.contact_pair,
        )
    }

    queue: deque[VoxelIdx] = deque([voxel0])

    N_SUB = max(1, int(np.ceil(cfg.voxel_size_m / cfg.step_size_m)))

    # Effort counters
    attempts = 0
    fail_moves = 0
    revisits = 0
    boundary_voxels = 0

    while queue and len(discovered) < trial_budget:
        voxel = queue.popleft()
        node = discovered[voxel]

        update_kinematics(ctx, node.q)
        # Cache starting CK (snapshot to detach from Pinocchio buffers)
        _ck_t_start = snapshot_ck(contact_kinematics(
            ctx, node.q, node.p, node.contact_pair[0],
            cfg.sphere_radius_m, cfg.link_radius_m,
        ))
        _ck_i_start = snapshot_ck(contact_kinematics(
            ctx, node.q, node.p, node.contact_pair[1],
            cfg.sphere_radius_m, cfg.link_radius_m,
        ))

        any_neighbor_succeeded = False
        for prim_idx, delta_p_full in enumerate(primitives):
            if len(discovered) >= trial_budget:
                break
            # Skip the reverse primitive — it targets the parent voxel, which is
            # already discovered. Primitives come ±paired so the reverse of i is
            # i ^ 1 (see compute_motion_primitives_frame).
            if node.arrival_prim >= 0 and prim_idx == (node.arrival_prim ^ 1):
                continue

            attempts += 1
            outcome = _attempt_primitive_step(
                ctx, cfg, delta_p_full, N_SUB,
                node.q, node.p, node.R_sphere, node.contact_pair,
                _ck_t_start, _ck_i_start,
                q_ids, v_ids, q_lower, q_upper,
            )
            if not outcome.success:
                fail_moves += 1
                continue

            voxel_next = point_to_voxel(outcome.p, origin, cfg.voxel_size_m, R_inv)
            if voxel_next in discovered:
                revisits += 1
                continue

            any_neighbor_succeeded = True
            discovered[voxel_next] = _TrialNode(
                q=outcome.q.copy(),
                p=outcome.p.copy(),
                R_sphere=outcome.R_sphere.copy(),
                contact_pair=outcome.contact_pair,
                arrival_prim=prim_idx,
            )
            queue.append(voxel_next)

        if not any_neighbor_succeeded:
            boundary_voxels += 1

    stats = TrialBFSStats(
        n_voxels=len(discovered),
        frontier_size=len(queue),
        attempts=attempts,
        fail_moves=fail_moves,
        revisits=revisits,
        unique_boundary=boundary_voxels,
    )
    if return_configs:
        return stats, discovered
    return stats


# ── Parallel trial evaluation ────────────────────────────────────────────────

# Module-level globals set by _init_worker — shared across tasks in one process.
_worker_cfg = None
_worker_ctx = None


def _init_worker(cfg: MetricConfig) -> None:
    """Process initializer: pass the parent config through and load the robot once
    per worker process (the Pinocchio context does not pickle, the config does)."""
    global _worker_cfg, _worker_ctx
    from .robot import load_robot
    _worker_cfg = cfg
    _worker_ctx = load_robot(cfg.urdf_path, cfg)


def _eval_seed_trial_worker(args):
    """Evaluate a single seed's trial BFS in a child process.

    Uses the module-level _worker_cfg / _worker_ctx loaded once per process
    by _init_worker, avoiding repeated URDF parsing.
    """
    (seed, budget, q_ids, v_ids, q_lower, q_upper) = args

    cfg = _worker_cfg
    ctx = _worker_ctx
    R_seed = _compute_trial_frame(cfg, ctx, seed, v_ids)
    primitives = compute_motion_primitives_frame(R_seed, cfg.voxel_size_m)
    stats = _run_trial_bfs(
        cfg, ctx, seed, R_seed, trial_budget=budget,
        q_ids=q_ids, v_ids=v_ids, q_lower=q_lower, q_upper=q_upper,
        primitives=primitives,
    )
    return stats


def _parallel_evaluate(candidates, budget, cfg,
                       q_ids, v_ids, q_lower, q_upper):
    """Evaluate a list of seeds at a given budget using parallel workers.

    Returns list of (TrialBFSStats, seed) tuples.
    Falls back to sequential evaluation if parallel execution fails.
    """
    n_workers = min(_physical_core_count(), len(candidates))
    args_list = [(s, budget,
                  q_ids.copy(), v_ids.copy(), q_lower.copy(), q_upper.copy())
                 for s in candidates]
    try:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=(cfg,),
            mp_context=multiprocessing.get_context("forkserver"),
        ) as executor:
            results = list(executor.map(_eval_seed_trial_worker, args_list))
        return [(stats, s) for stats, s in zip(results, candidates, strict=True)]
    except Exception as e:
        logger.warning("Parallel seed eval failed (%s), falling back to sequential", e)
        # Sequential fallback — load the robot once in this process.
        from .robot import load_robot
        ctx = load_robot(cfg.urdf_path, cfg)
        out = []
        for seed in candidates:
            R_seed = _compute_trial_frame(cfg, ctx, seed, v_ids)
            primitives = compute_motion_primitives_frame(R_seed, cfg.voxel_size_m)
            stats = _run_trial_bfs(
                cfg, ctx, seed, R_seed, trial_budget=budget,
                q_ids=q_ids, v_ids=v_ids, q_lower=q_lower, q_upper=q_upper,
                primitives=primitives,
            )
            out.append((stats, seed))
        return out


# ── Step 4: Seed polishing ───────────────────────────────────────────────────


def _polish_one_seed(
    cfg: MetricConfig,
    ctx: RobotContext,
    seed: SeedResult,
    q_ids: np.ndarray,
    v_ids: np.ndarray,
    q_lower: np.ndarray,
    q_upper: np.ndarray,
    n_iters: int = 5,
    step_size: float = 0.3,
) -> SeedResult:
    """Push a seed's joints toward center of range in the grasp Jacobian nullspace.

    The grasp Jacobian J_obj maps joint velocities to sphere translation.
    Moving in its nullspace changes the joint configuration without moving
    the sphere — improving joint-limit centrality and collision margin.

    Args:
        n_iters: Number of gradient steps.
        step_size: Fraction of the nullspace displacement to apply per step.

    Returns:
        Polished SeedResult (same sphere_centre, updated q).
    """
    q = seed.q.copy()
    q_center = 0.5 * (q_lower + q_upper)

    for _it in range(n_iters):
        update_kinematics(ctx, q)

        ck_thumb = contact_kinematics(
            ctx, q, seed.sphere_centre_world,
            seed.contact_pair[0], cfg.sphere_radius_m, cfg.link_radius_m,
        )
        ck_index = contact_kinematics(
            ctx, q, seed.sphere_centre_world,
            seed.contact_pair[1], cfg.sphere_radius_m, cfg.link_radius_m,
        )
        fold_fn = (lambda J, vids: fold_mimic_jacobian(J, vids, ctx)) if ctx.mimic_qs else None
        J_obj = compute_object_jacobian(ck_thumb, ck_index, v_ids, cfg.link_radius_m, fold_fn)

        # Nullspace projector: N = I - pinv(J) @ J
        J_pinv = np.linalg.pinv(J_obj)  # (m, 3)
        m = len(q_ids)
        N = np.eye(m) - J_pinv @ J_obj  # (m, m)

        dq_desired = q_center - q[q_ids]
        dq_step = step_size * (N @ dq_desired)

        q_new_active = np.clip(q[q_ids] + dq_step, q_lower, q_upper)
        q[q_ids] = q_new_active

    # Re-validate feasibility after polishing
    update_kinematics(ctx, q)
    ck_t = contact_kinematics(
        ctx, q, seed.sphere_centre_world,
        seed.contact_pair[0], cfg.sphere_radius_m, cfg.link_radius_m,
    )
    ck_i = contact_kinematics(
        ctx, q, seed.sphere_centre_world,
        seed.contact_pair[1], cfg.sphere_radius_m, cfg.link_radius_m,
    )

    max_gap = max(abs(ck_t.g_m), abs(ck_i.g_m))
    feasible = max_gap <= cfg.eps_g_m

    # Check collision/self-collision
    if feasible:
        col_ok, _ = collisions.check_collisions(
            ctx, q, seed.sphere_centre_world, seed.contact_pair,
            cfg.sphere_radius_m, cfg.link_radius_m, cfg.eps_col_m,
        )
        if not col_ok:
            feasible = False

    if feasible:
        self_ok, _ = collisions.check_self_collisions(
            ctx, cfg.thumb_contact_links, cfg.index_contact_links,
            cfg.link_radius_m, cfg.eps_col_m,
        )
        if not self_ok:
            feasible = False

    if not feasible:
        # Polishing broke feasibility — return original
        return seed

    return SeedResult(
        q=q,
        sphere_centre_world=seed.sphere_centre_world.copy(),
        contact_pair=seed.contact_pair,
        max_gap=max_gap,
        feasible=True,
    )


def _polish_seeds(
    cfg: MetricConfig,
    ctx: RobotContext,
    seeds: list[SeedResult],
    q_ids: np.ndarray,
    v_ids: np.ndarray,
    q_lower: np.ndarray,
    q_upper: np.ndarray,
) -> list[SeedResult]:
    """Polish all seeds by pushing joints toward center of range in nullspace."""
    polished = []
    n_improved = 0
    for seed in seeds:
        p = _polish_one_seed(cfg, ctx, seed, q_ids, v_ids, q_lower, q_upper)
        if p is not seed:
            n_improved += 1
        polished.append(p)
    logger.info("Seed polishing: %d/%d seeds improved", n_improved, len(seeds))
    return polished


# ── Main entry point ─────────────────────────────────────────────────────────


def select_seed_and_frame(
    cfg: MetricConfig,
    ctx: RobotContext,
) -> tuple[SeedResult, np.ndarray, dict | None, list[tuple[float, SeedResult]]]:
    """Select the best (seed, R_seed) pair via exhaustive evaluation.

    Evaluates ALL candidate seeds at max_states and picks the winner.
    This is both optimal (guaranteed to find the best seed) and efficient
    because BFS self-terminates when the frontier exhausts — weak seeds
    cost little even at high budgets.

    Empirically, exhaustive evaluation at max_states is as fast as or
    faster than multi-round elimination schemes for high-capability hands,
    and it cannot mistakenly eliminate a strong seed on a noisy low-budget
    estimate the way early-round pruning can.

    Returns:
        (best_seed, R_seed, seed_sensitivity, top_seeds) tuple.
        seed_sensitivity is a dict with voxel count distribution across
        all evaluated seeds (None if only 1 seed).
        top_seeds is a list of up to 3 (voxel_count, SeedResult) tuples,
        sorted descending by voxel count, for ensemble scoring.
    """
    q_ids = active_position_indices(ctx.model, cfg.finger_joint_names)
    v_ids = active_velocity_indices(ctx.model, cfg.finger_joint_names)
    q_lower = np.array(ctx.model.lowerPositionLimit)[q_ids]
    q_upper = np.array(ctx.model.upperPositionLimit)[q_ids]

    # Get ALL feasible seeds (light dedup only)
    seeds, gen_stats = _generate_feasible_seeds(cfg, ctx)
    logger.info(
        "Seed generation stats: %d joint configs -> %d reachable -> "
        "%d candidates -> %d IK solutions -> %d feasible -> %d after dedup",
        gen_stats["n_joint_configs"], gen_stats["n_reachable"],
        gen_stats["n_candidates"], gen_stats["n_ik_solutions"],
        gen_stats["n_feasible"], gen_stats["n_after_dedup"],
    )
    logger.info("Seed selection: %d candidate seeds", len(seeds))

    if len(seeds) == 1:
        R_seed = _compute_trial_frame(cfg, ctx, seeds[0], v_ids)
        return seeds[0], R_seed, None, [(0.0, seeds[0])]

    # ── Seed polishing: push joints toward center of range in grasp nullspace ──
    polished = _polish_seeds(cfg, ctx, seeds, q_ids, v_ids, q_lower, q_upper)
    candidates = list(polished)

    # ── Evaluate ALL seeds at max_states ──
    logger.info("Seed selection: evaluating %d seeds at max_states=%d",
                len(candidates), cfg.max_states)
    scored_raw = _parallel_evaluate(
        candidates, cfg.max_states, cfg,
        q_ids, v_ids, q_lower, q_upper,
    )
    scored = [(float(stats.n_voxels), s) for stats, s in scored_raw]
    scored.sort(key=lambda x: -x[0])

    voxel_counts = [v for v, _ in scored]
    n_at_cap = sum(1 for v in voxel_counts if v >= cfg.max_states * 0.95)
    mean_v = float(np.mean(voxel_counts))
    std_v = float(np.std(voxel_counts))
    logger.info(
        "Seed selection: best voxels=%d, median=%d, worst=%d "
        "(%d/%d at cap)",
        int(scored[0][0]), int(scored[len(scored) // 2][0]),
        int(scored[-1][0]), n_at_cap, len(scored),
    )

    best_v = voxel_counts[0]
    median_v = voxel_counts[len(voxel_counts) // 2]
    karma_s = round(median_v / best_v, 4) if best_v > 0 else 0.0

    # Top-3 ensemble: mean of top-3 seeds' voxel counts for stable KaRMA-T
    top_k = min(3, len(voxel_counts))
    top3_voxels = voxel_counts[:top_k]
    top3_mean_v = float(np.mean(top3_voxels))

    seed_sensitivity = {
        "n_seeds_evaluated": len(scored),
        "voxel_counts": [int(v) for v in voxel_counts],
        "best": int(best_v),
        "top3_mean": round(top3_mean_v, 2),
        "top3_voxels": [int(v) for v in top3_voxels],
        "median": int(median_v),
        "worst": int(voxel_counts[-1]),
        "mean": round(mean_v, 1),
        "std": round(std_v, 1),
        "karma_s": karma_s,  # median / best voxel count; higher = less cherry-picked
    }

    best = scored[0][1]
    R_seed = _compute_trial_frame(cfg, ctx, best, v_ids)

    # Return top-3 seeds for ensemble scoring (Phase 1 + Phase 2 on each)
    top_seeds = scored[:top_k]

    logger.info(
        "Seed selection complete: best_voxels=%d, top3_mean=%.1f, pair=%s",
        int(scored[0][0]), top3_mean_v, best.contact_pair,
    )
    return best, R_seed, seed_sensitivity, top_seeds
