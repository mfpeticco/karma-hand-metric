"""Constraint ablation: context manager that patches model/functions for ablation experiments.

All ablation logic lives here. The main metric code has zero ablation awareness.

Usage:
    from karma.ablation import ablation_context, AblationLevel

    with ablation_context(AblationLevel.GAP_ONLY, ctx, cfg.finger_joint_names):
        result = compute_metric(cfg, ctx, ...)
"""
from __future__ import annotations

import enum
import os
from contextlib import contextmanager

import numpy as np

from .robot import RobotContext, active_position_indices


class AblationLevel(enum.Enum):
    """Constraint ablation levels (cumulative).

    GAP_ONLY:  gap constraint only — no joint limits, no collisions, no antipodal
    GAP_JL:    gap + joint limits
    GAP_JL_COL: gap + joint limits + collisions
    FULL:      all constraints (no ablation)
    """
    GAP_ONLY = "gap_only"
    GAP_JL = "gap_jl"
    GAP_JL_COL = "gap_jl_col"
    FULL = "full"


# ── No-op replacements ───────────────────────────────────────────────────────

def _noop_check_collisions(ctx, q, sphere_centre_world, active_contact_names,
                           sphere_radius_m, link_radius_m, eps_col_m):
    """Always-pass collision check."""
    return (True, float("inf"))


def _noop_check_self_collisions(ctx, thumb_links, index_links,
                                link_radius_m, eps_col_m):
    """Always-pass self-collision check."""
    return (True, float("inf"))


def _noop_check_antipodal_friction(contact_normals_world, friction_coeff):
    """Always-feasible antipodal check."""
    from .grasp import GraspFeasibility
    return GraspFeasibility(feasible=True, reason="ablation_skip", cos_angle=1.0)


# ── Environment variable interface for multiprocessing workers ────────────────

_ABLATION_ENV_VAR = "KARMA_ABLATION"


def apply_worker_ablation_if_needed(ctx: RobotContext, cfg,
                                    level_str: str | None = None) -> None:
    """Apply ablation patches in a worker process.

    Checks *level_str* first (passed explicitly to the worker), then falls
    back to the KARMA_ABLATION env var. No-op if neither is set.
    """
    if level_str is None:
        level_str = os.environ.get(_ABLATION_ENV_VAR)
    if not level_str:
        return
    level = AblationLevel(level_str)
    _apply_patches(level, ctx, cfg.finger_joint_names)


def _apply_patches(level: AblationLevel, ctx: RobotContext,
                   finger_joint_names: list[str]) -> None:
    """Apply ablation patches (joint limits, collision, antipodal) in-place."""
    from . import collisions, grasp

    if level == AblationLevel.FULL:
        return

    # Widen joint limits for GAP_ONLY
    if level == AblationLevel.GAP_ONLY:
        q_ids = active_position_indices(ctx.model, finger_joint_names)
        limits_lower = np.array(ctx.model.lowerPositionLimit)
        limits_upper = np.array(ctx.model.upperPositionLimit)
        limits_lower[q_ids] = -2 * np.pi
        limits_upper[q_ids] = 2 * np.pi
        ctx.model.lowerPositionLimit = limits_lower
        ctx.model.upperPositionLimit = limits_upper

    # Disable collisions for GAP_ONLY and GAP_JL
    if level in (AblationLevel.GAP_ONLY, AblationLevel.GAP_JL):
        collisions.check_collisions = _noop_check_collisions
        collisions.check_self_collisions = _noop_check_self_collisions

    # Disable antipodal for GAP_ONLY, GAP_JL, and GAP_JL_COL
    if level in (AblationLevel.GAP_ONLY, AblationLevel.GAP_JL,
                 AblationLevel.GAP_JL_COL):
        grasp.check_antipodal_friction = _noop_check_antipodal_friction


# ── Context manager ──────────────────────────────────────────────────────────

@contextmanager
def ablation_context(level: AblationLevel, ctx: RobotContext,
                     finger_joint_names: list[str]):
    """Context manager that applies ablation patches and restores on exit.

    Also sets the KARMA_ABLATION env var so that forkserver workers
    can re-apply the same patches after re-importing modules.
    """
    if level == AblationLevel.FULL:
        # No patches needed — just yield
        yield
        return

    from . import collisions, grasp

    # Save originals
    orig_lower = np.array(ctx.model.lowerPositionLimit).copy()
    orig_upper = np.array(ctx.model.upperPositionLimit).copy()
    orig_check_collisions = collisions.check_collisions
    orig_check_self_collisions = collisions.check_self_collisions
    orig_check_antipodal = grasp.check_antipodal_friction
    orig_env = os.environ.get(_ABLATION_ENV_VAR)

    try:
        # Set env var for workers
        os.environ[_ABLATION_ENV_VAR] = level.value

        # Apply patches
        _apply_patches(level, ctx, finger_joint_names)

        yield
    finally:
        # Restore everything
        ctx.model.lowerPositionLimit = orig_lower
        ctx.model.upperPositionLimit = orig_upper
        collisions.check_collisions = orig_check_collisions
        collisions.check_self_collisions = orig_check_self_collisions
        grasp.check_antipodal_friction = orig_check_antipodal

        if orig_env is None:
            os.environ.pop(_ABLATION_ENV_VAR, None)
        else:
            os.environ[_ABLATION_ENV_VAR] = orig_env
