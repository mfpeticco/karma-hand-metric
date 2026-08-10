"""Collision checking: sphere vs all non-contact link capsules."""
from __future__ import annotations

import numpy as np

from .robot import RobotContext, capsule_endpoints_world


def check_collisions(
    ctx: RobotContext,
    q: np.ndarray,
    sphere_centre_world: np.ndarray,
    active_contact_names: tuple[str, str],
    sphere_radius_m: float,
    link_radius_m: float,
    eps_col_m: float,
) -> tuple[bool, float]:
    """Check the sphere against every link capsule except the active contacts.

    Returns ``(collision_free, min_clearance)``, where ``collision_free`` means
    ``min_clearance >= eps_col_m``.  Each capsule is first rejected by an AABB
    test expanded by the contact threshold, then measured exactly.
    """
    min_clearance = float("inf")
    collision_threshold = sphere_radius_m + link_radius_m + eps_col_m

    for name in ctx.link_capsules.keys():
        if name in active_contact_names:
            continue

        # Get capsule endpoints
        a, b = capsule_endpoints_world(ctx, name)

        # AABB quick rejection: compute bounding box of capsule expanded by threshold
        # and check if sphere centre is outside it
        aabb_min_x = min(a[0], b[0]) - collision_threshold
        aabb_max_x = max(a[0], b[0]) + collision_threshold
        aabb_min_y = min(a[1], b[1]) - collision_threshold
        aabb_max_y = max(a[1], b[1]) + collision_threshold
        aabb_min_z = min(a[2], b[2]) - collision_threshold
        aabb_max_z = max(a[2], b[2]) + collision_threshold

        # Check if sphere centre is outside the AABB (any axis)
        if (sphere_centre_world[0] < aabb_min_x or sphere_centre_world[0] > aabb_max_x or
            sphere_centre_world[1] < aabb_min_y or sphere_centre_world[1] > aabb_max_y or
            sphere_centre_world[2] < aabb_min_z or sphere_centre_world[2] > aabb_max_z):
            # Sphere centre is outside expanded AABB, safe from this capsule
            continue

        # Exact check: closest point on segment to sphere centre
        seg = b - a
        seg_len_sq = float(np.dot(seg, seg))
        if seg_len_sq < 1e-18:
            axis_pt = a
        else:
            t = float(np.dot(sphere_centre_world - a, seg)) / seg_len_sq
            t = max(0.0, min(1.0, t))
            axis_pt = a + t * seg

        dist = float(np.linalg.norm(sphere_centre_world - axis_pt))
        clearance = dist - sphere_radius_m - link_radius_m
        if clearance < min_clearance:
            min_clearance = clearance

    ok = min_clearance >= eps_col_m
    return ok, min_clearance


def _segment_segment_dist(a0, b0, a1, b1) -> float:
    """Closest distance between two line segments [a0,b0] and [a1,b1]."""
    d0 = b0 - a0
    d1 = b1 - a1
    r = a0 - a1

    a = float(np.dot(d0, d0))
    e = float(np.dot(d1, d1))
    f = float(np.dot(d1, r))
    EPS = 1e-12

    if a <= EPS and e <= EPS:
        return float(np.linalg.norm(r))

    if a <= EPS:
        t = max(0.0, min(1.0, f / e))
        closest = a0 - (a1 + t * d1)
        return float(np.linalg.norm(closest))

    c = float(np.dot(d0, r))
    if e <= EPS:
        s = max(0.0, min(1.0, -c / a))
        closest = (a0 + s * d0) - a1
        return float(np.linalg.norm(closest))

    b_ = float(np.dot(d0, d1))
    denom = a * e - b_ * b_

    if abs(denom) > EPS:
        s = max(0.0, min(1.0, (b_ * f - c * e) / denom))
    else:
        s = 0.0

    t = (b_ * s + f) / e
    if t < 0.0:
        t = 0.0
        s = max(0.0, min(1.0, -c / a))
    elif t > 1.0:
        t = 1.0
        s = max(0.0, min(1.0, (b_ - c) / a))

    closest = (a0 + s * d0) - (a1 + t * d1)
    return float(np.linalg.norm(closest))


def check_self_collisions(
    ctx: RobotContext,
    thumb_links: list[str],
    index_links: list[str],
    link_radius_m: float,
    eps_col_m: float,
) -> tuple[bool, float]:
    """Check for collisions between thumb and index finger capsules.

    Returns (collision_free, min_clearance).
    Capsule-capsule distance = segment distance - 2*link_radius.
    """
    min_clearance = float("inf")
    two_r = 2.0 * link_radius_m

    for t_name in thumb_links:
        if t_name not in ctx.link_capsules:
            continue
        a0, b0 = capsule_endpoints_world(ctx, t_name)

        for i_name in index_links:
            if i_name not in ctx.link_capsules:
                continue
            a1, b1 = capsule_endpoints_world(ctx, i_name)

            dist = _segment_segment_dist(a0, b0, a1, b1)
            clearance = dist - two_r
            if clearance < min_clearance:
                min_clearance = clearance

    ok = min_clearance >= eps_col_m
    return ok, min_clearance
