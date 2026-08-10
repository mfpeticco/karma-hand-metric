"""Primitives for placing a single thumb-index pinch on the sphere.

Given a joint configuration and a proposed sphere centre, :func:`_solve_for_centre`
projects onto the two-contact manifold with a least-squares IK solve and reports
whether the result is a feasible grasp.  :func:`_candidate_sphere_centres`
proposes the centres to try, and :func:`_solve_candidate_worker` carries that
work into subprocesses.

The *strategy* that decides which seeds to generate, polish, and ultimately score
lives in :mod:`karma.seed_selection`; this module only knows how to solve one
candidate.  Keeping the split means the expensive search policy can change
without touching the contact math.

The IK is posed in non-dimensional form (the free variable is the sphere-centre
displacement divided by the sphere radius), so a uniformly scaled hand solves an
identical problem, which is what makes the metric scale-invariant.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .config import MetricConfig
from .robot import (
    RobotContext, update_kinematics, update_positions_only, load_robot,
    capsule_endpoints_world,
)
from .contacts import contact_kinematics, _capsule_axis_segment
from .collisions import check_collisions
from .grasp import check_antipodal_friction
from .math_utils import closest_points_on_segments

logger = logging.getLogger(__name__)


@dataclass
class SeedResult:
    """One candidate pinch: a joint configuration plus the sphere it holds."""

    q: np.ndarray
    sphere_centre_world: np.ndarray
    contact_pair: tuple[str, str]  # (thumb_link_name, index_link_name)
    max_gap: float                 # max |contact gap| over both contacts, metres
    feasible: bool                 # gap, collision and antipodal checks all pass


def _candidate_sphere_centres(
    ctx: RobotContext,
    q: np.ndarray,
    thumb_link: str,
    index_link: str,
    offsets: list[float],
) -> list[np.ndarray]:
    """Generate candidate sphere centres between a pair of finger links."""
    update_positions_only(ctx, q)

    c0, d0, h0 = _capsule_axis_segment(ctx, thumb_link)
    c1, d1, h1 = _capsule_axis_segment(ctx, index_link)

    pt0, pt1 = closest_points_on_segments(c0, d0, h0, c1, d1, h1)

    midpoint = 0.5 * (pt0 + pt1)
    diff = pt1 - pt0
    dist = np.linalg.norm(diff)

    if dist < 1e-9:
        return [midpoint]

    centres = []
    for off in offsets:
        p = midpoint + off * d0
        centres.append(p)
        if abs(off) > 1e-9:
            p2 = midpoint + off * d1
            centres.append(p2)

    return centres


def _solve_candidate_worker(args: tuple) -> SeedResult | None:
    """Worker function for parallel seed search.

    Creates a new robot context and solves for a single (q_sample, centre) candidate.
    This function is designed to be called by ProcessPoolExecutor.
    """
    (cfg, q_sample, centre, thumb_link, index_link,
     q_ids, v_ids, q_lower, q_upper) = args

    # Rebuild the (unpicklable) robot context in this worker process. The frozen
    # MetricConfig itself pickles cleanly and is passed through directly.
    ctx = load_robot(cfg.urdf_path, cfg)
    return _solve_for_centre(
        cfg, ctx, q_sample.copy(), centre, thumb_link, index_link,
        q_ids, v_ids, q_lower, q_upper,
    )


def _solve_for_centre(
    cfg: MetricConfig,
    ctx: RobotContext,
    q_init: np.ndarray,
    centre_init: np.ndarray,
    thumb_link: str,
    index_link: str,
    q_ids: np.ndarray,
    v_ids: np.ndarray,
    q_lower: np.ndarray,
    q_upper: np.ndarray,
) -> SeedResult | None:
    """Least-squares IK to place both contacts on the sphere.

    Uses a relative displacement variable dp = p - centre_init for the sphere
    centre, so the optimizer always sees coordinates near zero regardless of
    the robot's absolute world-frame position.  This ensures translation
    invariance: a robot at [0,0,0] and one at [1,0,0] produce identical IK
    solutions (same q, sphere centre shifted by [1,0,0]).
    """
    centre = centre_init.copy()

    def residuals(x: np.ndarray) -> np.ndarray:
        q = q_init.copy()
        q[q_ids] = x[:len(q_ids)]
        dp = x[len(q_ids):]  # nondimensional displacement (dp / r_s)
        p = centre + cfg.sphere_radius_m * dp  # reconstruct physical world position

        update_kinematics(ctx, q)

        ck_thumb = contact_kinematics(
            ctx, q, p, thumb_link, cfg.sphere_radius_m, cfg.link_radius_m
        )
        ck_index = contact_kinematics(
            ctx, q, p, index_link, cfg.sphere_radius_m, cfg.link_radius_m
        )

        # Contact gap residuals (want g ≈ 0).
        # Normalize by eps_g_m (which scales with L_ref) for scale-invariance.
        # Weight 0.35 is equivalent to a fixed weight of 1000 at the nominal
        # eps_g_m = 0.35 mm.
        r_gaps = np.array([ck_thumb.g_m, ck_index.g_m]) / cfg.eps_g_m * 0.35

        # Collision penalty — normalize by eps_col_m for scale-invariance.
        # Weight 0.25 is equivalent to a fixed weight of 500 at the nominal
        # eps_col_m = 0.5 mm.
        ok, clearance = check_collisions(
            ctx, q, p, (thumb_link, index_link),
            cfg.sphere_radius_m, cfg.link_radius_m, cfg.eps_col_m,
        )
        r_col = np.array([max(0.0, (cfg.eps_col_m - clearance) / cfg.eps_col_m) * 0.25])

        # Antipodal residual: encourage dot(-n_thumb, n_index) ≈ 1.0
        # This guides the optimizer toward valid pinch configurations where
        # the contact normals oppose each other (required for force closure).
        n_thumb = ck_thumb.normal_world
        n_index = ck_index.normal_world
        n_thumb_norm = np.linalg.norm(n_thumb)
        n_index_norm = np.linalg.norm(n_index)
        if n_thumb_norm > 1e-9 and n_index_norm > 1e-9:
            cos_antipodal = np.dot(-n_thumb / n_thumb_norm, n_index / n_index_norm)
            # cos_antipodal should be ~1.0 for good pinch; penalize deviation
            r_antipodal = np.array([(1.0 - cos_antipodal) * 200.0])
        else:
            # Degenerate normals - large penalty
            r_antipodal = np.array([200.0])

        # Small regularization toward initial config
        r_reg = (x[:len(q_ids)] - q_init[q_ids]) * 0.1

        return np.concatenate([r_gaps, r_col, r_antipodal, r_reg])

    def jacobian(x: np.ndarray) -> np.ndarray:
        """Analytic Jacobian of residuals w.r.t. (q_active, dp).

        Provides exact derivatives for gap, antipodal, and regularization terms.
        Collision row uses zero approximation (small weight, rarely active).
        This eliminates finite-difference noise amplification that breaks
        translation invariance: FD amplifies 1e-14 residual noise by 1/h ≈ 1e8
        → 1e-6 Jacobian errors, while analytic keeps ~1e-14.
        """
        m = len(q_ids)
        q = q_init.copy()
        q[q_ids] = x[:m]
        dp = x[m:]  # nondimensional (dp / r_s)
        p = centre + cfg.sphere_radius_m * dp

        update_kinematics(ctx, q)

        ck_t = contact_kinematics(
            ctx, q, p, thumb_link, cfg.sphere_radius_m, cfg.link_radius_m
        )
        ck_i = contact_kinematics(
            ctx, q, p, index_link, cfg.sphere_radius_m, cfg.link_radius_m
        )

        n_res = 4 + m
        n_var = m + 3
        J = np.zeros((n_res, n_var))

        scale = 0.35 / cfg.eps_g_m
        n_t = ck_t.normal_world
        n_i = ck_i.normal_world

        # --- Compute M_k = I - d(axis_pt_k)/d(p) for each contact ---
        # Interior t: d(axis_pt)/d(p) = seg*seg^T/|seg|^2 (proj onto segment)
        # Boundary t: d(axis_pt)/d(p) = 0  →  M = I
        # We also reuse seg/ssq/t for the gap dp-Jacobian.

        a_t, b_t = capsule_endpoints_world(ctx, thumb_link)
        seg_t = b_t - a_t
        ssq_t = float(np.dot(seg_t, seg_t))
        t_t = float(np.clip(np.dot(p - a_t, seg_t) / ssq_t, 0.0, 1.0)) if ssq_t > 1e-18 else 0.0
        interior_t = 1e-12 < t_t < 1.0 - 1e-12 and ssq_t > 1e-18

        a_i, b_i = capsule_endpoints_world(ctx, index_link)
        seg_i = b_i - a_i
        ssq_i = float(np.dot(seg_i, seg_i))
        t_i = float(np.clip(np.dot(p - a_i, seg_i) / ssq_i, 0.0, 1.0)) if ssq_i > 1e-18 else 0.0
        interior_i = 1e-12 < t_i < 1.0 - 1e-12 and ssq_i > 1e-18

        def _apply_M(v, seg, ssq, interior):
            """Compute M @ v = v - proj_seg(v) if interior, else v."""
            if interior:
                return v - (np.dot(v, seg) / ssq) * seg
            return v

        # --- Row 0: d(r_gap_thumb)/d(q_active, dp) ---
        # d(gap)/d(q) = -normal^T * Jv_world
        J[0, :m] = -scale * (n_t @ ck_t.Jv_world[:, v_ids])
        # d(gap)/d(dp) = normal^T * M_t
        J[0, m:] = scale * _apply_M(n_t, seg_t, ssq_t, interior_t)

        # --- Row 1: d(r_gap_index)/d(q_active, dp) ---
        J[1, :m] = -scale * (n_i @ ck_i.Jv_world[:, v_ids])
        J[1, m:] = scale * _apply_M(n_i, seg_i, ssq_i, interior_i)

        # --- Row 2: collision — zero approximation (small weight, rarely active) ---

        # Row 3 (antipodal): zero approximation.  While the analytic formula is
        # correct (validated to 1e-10 vs FD), its large scale (~1800) amplifies
        # the 1e-14 inter-model FK noise, worsening translation invariance.
        # Zero rows are perfectly deterministic across models.

        # --- Rows 4..4+m-1: d(r_reg)/d(q_active) = 0.1 * I ---
        np.fill_diagonal(J[4:4+m, :m], 0.1)

        # Nondimensional displacement chain rule: d r / d(dp_tilde) = (d r / d dp) * r_s.
        # The existing dp-columns are d r / d dp (~1/r_s); ×r_s makes them O(1), scale-free.
        J[:, m:] *= cfg.sphere_radius_m

        return J

    # Use displacement dp = p - centre as optimization variable (starts at zero).
    # This makes the solver's internal scaling and trust-region steps independent
    # of the robot's absolute world position → translation invariance.
    x0 = np.concatenate([q_init[q_ids], np.zeros(3)])
    # Nondimensional displacement bound: dp_tilde = dp / r_s, so the physical
    # ±3(r_s + r_l) bound becomes ±3(1 + r_l/r_s) — scale-free.
    ik_margin = 3.0 * (cfg.sphere_radius_m + cfg.link_radius_m) / cfg.sphere_radius_m
    bounds_lo = np.concatenate([q_lower, -ik_margin * np.ones(3)])
    bounds_hi = np.concatenate([q_upper, +ik_margin * np.ones(3)])

    try:
        sol = least_squares(
            residuals, x0,
            jac=jacobian,
            bounds=(bounds_lo, bounds_hi),
            method="trf",
            max_nfev=500,
            ftol=1e-8,
            xtol=1e-8,
        )
    except Exception as e:
        logger.debug("IK solve failed for candidate, discarding: %s", e)
        return None

    q_sol = q_init.copy()
    q_sol[q_ids] = sol.x[:len(q_ids)]
    dp_sol = sol.x[len(q_ids):]  # nondimensional
    p_sol = centre + cfg.sphere_radius_m * dp_sol  # reconstruct physical world position

    update_kinematics(ctx, q_sol)

    ck_thumb = contact_kinematics(
        ctx, q_sol, p_sol, thumb_link, cfg.sphere_radius_m, cfg.link_radius_m
    )
    ck_index = contact_kinematics(
        ctx, q_sol, p_sol, index_link, cfg.sphere_radius_m, cfg.link_radius_m
    )

    max_gap = max(abs(ck_thumb.g_m), abs(ck_index.g_m))

    # Check feasibility
    ok_col, _ = check_collisions(
        ctx, q_sol, p_sol, (thumb_link, index_link),
        cfg.sphere_radius_m, cfg.link_radius_m, cfg.eps_col_m,
    )
    normals = np.array([ck_thumb.normal_world, ck_index.normal_world])
    grasp_ok = check_antipodal_friction(normals, cfg.friction_coeff)

    feasible = max_gap <= cfg.eps_g_m and ok_col and grasp_ok.feasible

    return SeedResult(
        q=q_sol,
        sphere_centre_world=p_sol,
        contact_pair=(thumb_link, index_link),
        max_gap=max_gap,
        feasible=feasible,
    )
