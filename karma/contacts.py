"""Capsule-on-sphere contact kinematics.

Each finger link is modeled as a capsule (cylinder + hemispherical caps) with
a fixed radius (link_radius_m).  The sphere is at a known centre with a known
radius.  Contact between capsule and sphere reduces to the distance from the
sphere centre to the capsule's central segment, minus the sum of radii.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pinocchio as pin

from .math_utils import skew
from .robot import RobotContext, capsule_endpoints_world


@dataclass
class ContactKinematics:
    """Kinematic quantities for one capsule-sphere contact."""
    geom_name: str
    # Closest point on the capsule axis to the sphere centre (world frame)
    axis_point_world: np.ndarray
    # Contact point on the sphere surface (world frame)
    sphere_point_world: np.ndarray
    # Unit vector from capsule axis point toward sphere centre (inward w.r.t. the sphere).
    normal_world: np.ndarray
    # Signed gap: positive = separated, negative = penetrating
    g_m: float
    # Linear velocity Jacobian of axis_point (closest-point on segment) w.r.t. full q
    # (world frame, 3×nv). This is the derivative of the *geometric closest point*
    # location, which is useful for gap projection / distance linearization.
    Jv_world: np.ndarray
    # Angular velocity Jacobian of the link (world frame, 3×nv)
    Jw_world: np.ndarray
    # World position of the link's inboard joint origin
    origin_world: np.ndarray
    # Linear velocity Jacobian of the inboard joint origin (world frame, 3×nv).
    # Together with Jw_world, this defines the link's instantaneous twist.
    Jv_origin_world: np.ndarray


def snapshot_ck(ck: ContactKinematics) -> ContactKinematics:
    """Deep-copy a ContactKinematics, detaching from Pinocchio internal buffers.

    Use this to persist CK data across subsequent update_kinematics() calls
    that would invalidate Pinocchio's internal views.
    """
    return ContactKinematics(
        geom_name=ck.geom_name,
        axis_point_world=ck.axis_point_world.copy(),
        sphere_point_world=ck.sphere_point_world.copy(),
        normal_world=ck.normal_world.copy(),
        g_m=ck.g_m,
        Jv_world=ck.Jv_world.copy(),
        Jw_world=ck.Jw_world.copy(),
        origin_world=ck.origin_world.copy(),
        Jv_origin_world=ck.Jv_origin_world.copy(),
    )


def _capsule_axis_segment(
    ctx: RobotContext,
    link_name: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (centre, unit_direction, half_length) for a link's capsule axis in world frame."""
    a, b = capsule_endpoints_world(ctx, link_name)
    centre_world = 0.5 * (a + b)
    seg = b - a
    seg_len = float(np.linalg.norm(seg))
    if seg_len < 1e-12:
        direction_world = np.array([0.0, 0.0, 1.0])
        half_len = 0.0
    else:
        direction_world = seg / seg_len
        half_len = 0.5 * seg_len
    return centre_world, direction_world, half_len


def contact_kinematics(
    ctx: RobotContext,
    q: np.ndarray,
    sphere_centre_world: np.ndarray,
    geom_name: str,
    sphere_radius_m: float,
    link_radius_m: float,
) -> ContactKinematics:
    """Compute contact kinematics between a named link capsule and the sphere."""
    # Capsule axis endpoints (world frame).
    a, b = capsule_endpoints_world(ctx, geom_name)

    # Closest point on the capsule axis segment to the sphere centre.
    seg = b - a
    seg_len_sq = float(np.dot(seg, seg))
    if seg_len_sq < 1e-18:
        t = 0.0
        axis_pt = a.copy()
    else:
        t_raw = float(np.dot(sphere_centre_world - a, seg)) / seg_len_sq
        t = float(np.clip(t_raw, 0.0, 1.0))
        axis_pt = a + t * seg

    diff = sphere_centre_world - axis_pt
    dist = float(np.linalg.norm(diff))
    if dist < 1e-12:
        # Degenerate: sphere centre on capsule axis.  Gap will be deeply
        # negative (full penetration), so downstream checks reject this.
        # Use a deterministic direction perpendicular to the segment so that
        # the (unused) Jacobian is at least rotation-invariant.
        normal = np.cross(seg, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(normal) < 1e-12:
            normal = np.cross(seg, np.array([0.0, 1.0, 0.0]))
        nrm = np.linalg.norm(normal)
        normal = normal / nrm if nrm > 1e-12 else np.array([0.0, 0.0, 1.0])
    else:
        normal = diff / dist

    gap = dist - sphere_radius_m - link_radius_m
    sphere_pt = sphere_centre_world - sphere_radius_m * normal

    # Jacobian of the closest point on the axis segment.
    #
    # For capsule-sphere distance we need the velocity of the closest point on the
    # *segment* (not a fixed point on the link). When the closest point is in the
    # interior (0<t<1), its position is:
    #   p_axis = a + t * (b - a)
    # where t depends on (a, b). We linearize with the analytic derivative of t.
    cap = ctx.link_capsules[geom_name]
    link_frame_id = cap.link_frame_id

    # NOTE: We need a Jacobian whose linear part matches the derivative of the
    # world translation of the point. In Pinocchio this corresponds to
    # LOCAL_WORLD_ALIGNED (not WORLD).
    Jfull_a = pin.getFrameJacobian(
        ctx.model, ctx.data, link_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    )
    Jv_a = Jfull_a[:3, :]
    Jw_a = Jfull_a[3:, :]
    origin_world = np.array(ctx.data.oMf[link_frame_id].translation)

    if cap.distal_frame_id is not None:
        Jfull_b = pin.getFrameJacobian(
            ctx.model, ctx.data, cap.distal_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        Jv_b = Jfull_b[:3, :]
    else:
        # Terminal link: distal endpoint is at a fixed local offset from the inboard joint.
        assert cap.tip_offset_xyz_m is not None
        R_a = np.array(ctx.data.oMf[link_frame_id].rotation)
        lever_ab = R_a @ cap.tip_offset_xyz_m
        Jv_b = Jv_a - skew(lever_ab) @ Jw_a

    if t <= 0.0 + 1e-12:
        Jv_axis_pt = Jv_a
    elif t >= 1.0 - 1e-12:
        Jv_axis_pt = Jv_b
    else:
        J_s = Jv_b - Jv_a  # (3, nv)
        w = sphere_centre_world - a  # (3,)
        u = seg_len_sq
        ws = float(np.dot(w, seg))

        # dt/dq = ( (-s^T J_a + w^T J_s)/u ) - ( (w·s)/u^2 * (2 s^T J_s) )
        dt = (-seg @ Jv_a + w @ J_s) / u - (ws / (u * u)) * (2.0 * (seg @ J_s))
        Jv_axis_pt = Jv_a + t * J_s + np.outer(seg, dt)

    return ContactKinematics(
        geom_name=geom_name,
        axis_point_world=axis_pt,
        sphere_point_world=sphere_pt,
        normal_world=normal,
        g_m=gap,
        Jv_world=Jv_axis_pt,
        Jw_world=Jw_a,
        origin_world=origin_world,
        Jv_origin_world=Jv_a,
    )


def contact_gap_only(
    ctx: RobotContext,
    sphere_centre_world: np.ndarray,
    geom_name: str,
    sphere_radius_m: float,
    link_radius_m: float,
) -> float:
    """Signed gap without Jacobians (~2x faster than contact_kinematics)."""
    a, b = capsule_endpoints_world(ctx, geom_name)
    seg = b - a
    seg_len_sq = float(np.dot(seg, seg))
    if seg_len_sq < 1e-18:
        axis_pt = a.copy()
    else:
        t = float(np.clip(float(np.dot(sphere_centre_world - a, seg)) / seg_len_sq, 0.0, 1.0))
        axis_pt = a + t * seg
    return float(np.linalg.norm(sphere_centre_world - axis_pt)) - sphere_radius_m - link_radius_m


def select_contact_pair(
    ctx: RobotContext,
    q: np.ndarray,
    sphere_centre_world: np.ndarray,
    thumb_contact_links: list[str],
    index_contact_links: list[str],
    sphere_radius_m: float,
    link_radius_m: float,
    current_pair: tuple[str, str] | None = None,
    hysteresis_m: float = 0.0,
) -> tuple[str, str]:
    """Select the closest thumb link and closest index link to the sphere.

    With *hysteresis_m* > 0 and a *current_pair*, the current link is kept
    unless another link's absolute gap is smaller by at least *hysteresis_m*.
    This prevents chattering when two links have similar gaps.
    """
    def _best(links: list[str], current: str | None) -> str:
        best_name = links[0]
        best_gap = float("inf")
        for name in links:
            ag = abs(contact_gap_only(ctx, sphere_centre_world, name, sphere_radius_m, link_radius_m))
            if ag < best_gap:
                best_gap = ag
                best_name = name
        # Hysteresis: keep current link unless new one is clearly better
        if current is not None and current != best_name and hysteresis_m > 0:
            cur_gap = abs(contact_gap_only(ctx, sphere_centre_world, current, sphere_radius_m, link_radius_m))
            if cur_gap - best_gap < hysteresis_m:
                return current
        return best_name

    cur_t = current_pair[0] if current_pair else None
    cur_i = current_pair[1] if current_pair else None
    return (_best(thumb_contact_links, cur_t), _best(index_contact_links, cur_i))
