"""Projection onto the two-contact rolling manifold.

Given a joint configuration and a fixed sphere centre, :func:`_project_onto_contacts`
runs a damped Gauss-Newton solve that closes both contact gaps while tracking the
sphere rotation needed to keep the contacts rolling (no tangential slip). This is
the numerical core shared by the Phase 1 translation BFS in :mod:`karma.metric` and
the trial BFS in :mod:`karma.seed_selection`.
"""
from __future__ import annotations

import numpy as np

from .robot import RobotContext, update_kinematics, fold_mimic_jacobian
from .contacts import contact_kinematics, contact_gap_only, ContactKinematics
from .math_utils import skew


def _compute_rolling_dtheta(
    ck0, ck1, dq_active, v_ids, sphere_radius_m, link_radius_m,
    ctx: RobotContext | None = None,
) -> np.ndarray:
    """Compute sphere rotation to maintain rolling given a joint displacement.

    Rolling constraint at each contact (with dp=0):
        Jv_surf @ dq = r_sphere * skew(n) @ dtheta

    Solves for dtheta in least-squares sense.

    Note: If ctx is provided and has mimic joints, the Jacobian extraction will
    automatically fold in mimic joint contributions to source joint columns.
    """
    def _extract_jacobian_cols(J_full: np.ndarray) -> np.ndarray:
        if ctx is not None and ctx.mimic_qs:
            return fold_mimic_jacobian(J_full, v_ids, ctx)
        return J_full[:, v_ids]

    A = np.zeros((6, 3))
    b = np.zeros(6)

    for i, ck in enumerate([ck0, ck1]):
        n = ck.normal_world
        Jw = _extract_jacobian_cols(ck.Jw_world)
        p_capsule = ck.axis_point_world + link_radius_m * n
        r = p_capsule - ck.origin_world
        Jv_surf = _extract_jacobian_cols(ck.Jv_origin_world) - skew(r) @ Jw

        # Capsule surface velocity due to joint motion
        v_capsule = Jv_surf @ dq_active

        # Rolling: r_sphere * skew(n) @ dtheta = v_capsule
        A[3*i:3*i+3, :] = sphere_radius_m * skew(n)
        b[3*i:3*i+3] = v_capsule

    # Solve least-squares
    AtA = A.T @ A + (1e-4 * sphere_radius_m ** 2) * np.eye(3)  # scale-invariant damping
    return np.linalg.solve(AtA, A.T @ b)


def _project_onto_contacts(
    ctx: RobotContext,
    q: np.ndarray,
    sphere_centre: np.ndarray,
    contact_pair: tuple[str, str],
    q_ids: np.ndarray,
    v_ids: np.ndarray,
    sphere_radius_m: float,
    link_radius_m: float,
    eps_g_m: float,
    max_iters: int = 10,
) -> tuple[np.ndarray, np.ndarray, bool, ContactKinematics | None, ContactKinematics | None]:
    """Gauss-Newton with backtracking to close contact gaps.

    Returns (q_projected, delta_theta, success, ck0, ck1) where delta_theta is the
    sphere rotation needed to maintain rolling during the projection step, and
    ck0/ck1 are the final contact kinematics (to avoid redundant recomputation).

    The rolling constraint is: Jv_surf @ dq = r * skew(n) @ dtheta + dp
    During projection, dp=0 (sphere centre fixed), so we solve for dtheta
    that compensates for the capsule surface motion caused by dq.

    We compute delta_theta incrementally at each Gauss-Newton step to account
    for the changing contact geometry.

    Note: If ctx has mimic joints, the Jacobian extraction will automatically
    fold in mimic joint contributions to source joint columns.
    """
    def _extract_jacobian_cols(J_full: np.ndarray) -> np.ndarray:
        if ctx.mimic_qs:
            return fold_mimic_jacobian(J_full, v_ids, ctx)
        return J_full[:, v_ids]

    q_lo = np.array(ctx.model.lowerPositionLimit)[q_ids]
    q_hi = np.array(ctx.model.upperPositionLimit)[q_ids]
    q_proj = q.copy()
    q_proj[q_ids] = np.clip(q_proj[q_ids], q_lo, q_hi)
    total_dtheta = np.zeros(3)

    for _ in range(max_iters):
        update_kinematics(ctx, q_proj)
        ck0 = contact_kinematics(ctx, q_proj, sphere_centre, contact_pair[0],
                                 sphere_radius_m, link_radius_m)
        ck1 = contact_kinematics(ctx, q_proj, sphere_centre, contact_pair[1],
                                 sphere_radius_m, link_radius_m)
        g = np.array([ck0.g_m, ck1.g_m])
        if np.max(np.abs(g)) <= eps_g_m:
            break

        J = np.zeros((2, len(q_ids)))
        # g = ||p_sphere - p_axis|| - (r_sphere + r_link), with p_sphere fixed here.
        # dg/dq = -n^T * dp_axis/dq
        J[0, :] = -(ck0.normal_world @ _extract_jacobian_cols(ck0.Jv_world))
        J[1, :] = -(ck1.normal_world @ _extract_jacobian_cols(ck1.Jv_world))

        JtJ = J.T @ J + (1e-2 * sphere_radius_m ** 2) * np.eye(len(q_ids))  # scale-invariant damping
        dq = -np.linalg.solve(JtJ, J.T @ g)

        # Backtracking: try full step, then halve up to 3 times
        # Use gap-only check (no Jacobians needed) for ~2x faster backtracking
        alpha = 1.0
        g_norm = np.linalg.norm(g)
        for _ in range(3):
            q_try = q_proj.copy()
            q_try[q_ids] = np.clip(q_proj[q_ids] + alpha * dq, q_lo, q_hi)
            update_kinematics(ctx, q_try)
            g0t = contact_gap_only(ctx, sphere_centre, contact_pair[0],
                                   sphere_radius_m, link_radius_m)
            g1t = contact_gap_only(ctx, sphere_centre, contact_pair[1],
                                   sphere_radius_m, link_radius_m)
            if np.linalg.norm([g0t, g1t]) < g_norm:
                break
            alpha *= 0.5

        actual_dq = alpha * dq
        q_proj[q_ids] = np.clip(q_proj[q_ids] + actual_dq, q_lo, q_hi)

        # Compute rolling-aware dtheta for this Gauss-Newton step
        # Use Jacobians at current configuration (before step applied)
        step_dtheta = _compute_rolling_dtheta(
            ck0, ck1, actual_dq, v_ids, sphere_radius_m, link_radius_m, ctx=ctx,
        )
        total_dtheta += step_dtheta

    update_kinematics(ctx, q_proj)
    ck0 = contact_kinematics(ctx, q_proj, sphere_centre, contact_pair[0],
                             sphere_radius_m, link_radius_m)
    ck1 = contact_kinematics(ctx, q_proj, sphere_centre, contact_pair[1],
                             sphere_radius_m, link_radius_m)
    ok = max(abs(ck0.g_m), abs(ck1.g_m)) <= eps_g_m
    return q_proj, total_dtheta, ok, ck0, ck1
