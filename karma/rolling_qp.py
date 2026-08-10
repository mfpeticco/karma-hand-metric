"""OSQP-based rolling-contact step solver.

For translation steps, the sphere rotation is computed geometrically to enforce
pure rolling (no slip). The QP then solves for joint motion that achieves this.

For rotation steps (delta_p = 0), the rotation is a target to be achieved as
closely as possible given kinematic constraints.

The QP enforces:
  - Rolling constraint: surface velocities must match (no slip).
  - Contact gap maintenance: the signed distance must stay within eps_g.
  - Joint step bounds: |delta_q_i| <= max_joint_step_rad.
  - Joint limit bounds: q_lo <= q + delta_q <= q_hi.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import osqp
from scipy import sparse

from .contacts import ContactKinematics
from .math_utils import skew

if TYPE_CHECKING:
    from .robot import RobotContext


class QPWorkspace:
    """Pre-allocated workspace for QP solvers to avoid repeated memory allocation.

    This class holds pre-allocated numpy arrays for QP matrix construction,
    sized for a specific number of active joints and contacts.
    """

    def __init__(self, m: int, n_contacts: int = 2):
        """Initialize workspace for given dimensions.

        Args:
            m: Number of active joints
            n_contacts: Number of contacts (default 2)
        """
        self.m = m
        self.n_contacts = n_contacts
        self.n_vars = m + 3  # dq + delta_theta
        self.n_roll = 3 * n_contacts
        self.n_gap = n_contacts
        self.n_constraints = n_contacts + self.n_vars  # gap + box constraints

        # Pre-allocate arrays for solve_rolling_step
        self.M_roll = np.zeros((self.n_roll, self.n_vars))
        self.b_roll = np.zeros(self.n_roll)
        self.A_gap = np.zeros((self.n_gap, self.n_vars))
        self.l_gap = np.zeros(self.n_gap)
        self.u_gap = np.zeros(self.n_gap)
        self.P = np.zeros((self.n_vars, self.n_vars))
        self.c = np.zeros(self.n_vars)
        self.A_box = np.eye(self.n_vars)
        self.A_all = np.zeros((self.n_constraints, self.n_vars))
        self.A_all[self.n_gap:, :] = self.A_box  # Box part is always identity
        self.l_all = np.zeros(self.n_constraints)
        self.u_all = np.zeros(self.n_constraints)
        self.x_lo = np.zeros(self.n_vars)
        self.x_hi = np.zeros(self.n_vars)
        self.dq_lo = np.zeros(m)
        self.dq_hi = np.zeros(m)

        # Pre-allocate arrays for solve_rotation_step (smaller, no theta vars)
        self.M_roll_rot = np.zeros((self.n_roll, m))
        self.b_roll_rot = np.zeros(self.n_roll)
        self.A_gap_rot = np.zeros((self.n_gap, m))
        self.P_rot = np.zeros((m, m))
        self.c_rot = np.zeros(m)
        self.A_box_rot = np.eye(m)
        self.n_constraints_rot = n_contacts + m
        self.A_all_rot = np.zeros((self.n_constraints_rot, m))
        self.A_all_rot[self.n_gap:, :] = self.A_box_rot
        self.l_all_rot = np.zeros(self.n_constraints_rot)
        self.u_all_rot = np.zeros(self.n_constraints_rot)

        # Temp arrays for intermediate computations
        self.Jv_surf_active = np.zeros((3, m))
        self.r_vec = np.zeros(3)
        self.skew_n = np.zeros((3, 3))
        self.A_theta = np.zeros((self.n_roll, 3))

        # ── CSC templates (fixed sparsity, data filled in-place) ──
        # Rolling solver: P is upper-tri (n_vars x n_vars), A is dense (n_constraints x n_vars)
        n = self.n_vars
        nc = self.n_constraints
        # P upper-tri: column j has rows 0..j  →  indptr[j+1] = indptr[j] + (j+1)
        _P_indptr = np.zeros(n + 1, dtype=np.int32)
        _P_rows = []
        for j in range(n):
            _P_indptr[j + 1] = _P_indptr[j] + (j + 1)
            _P_rows.extend(range(j + 1))
        _P_nnz = int(_P_indptr[-1])
        self._P_sparse = sparse.csc_matrix(
            (np.zeros(_P_nnz), np.array(_P_rows, dtype=np.int32), _P_indptr),
            shape=(n, n),
        )
        # Pre-compute row/col extraction indices (CSC column-major order)
        self._P_triu_rows = np.array(_P_rows, dtype=int)
        self._P_triu_cols = np.empty(_P_nnz, dtype=int)
        for j in range(n):
            self._P_triu_cols[_P_indptr[j]:_P_indptr[j + 1]] = j

        # A dense: column j has all nc rows  →  indptr[j+1] = indptr[j] + nc
        _A_indptr = np.arange(0, (n + 1) * nc, nc, dtype=np.int32)
        _A_rows = np.tile(np.arange(nc, dtype=np.int32), n)
        self._A_sparse = sparse.csc_matrix(
            (np.zeros(nc * n), _A_rows, _A_indptr),
            shape=(nc, n),
        )

        # Rotation solver: P_rot upper-tri (m x m), A_rot dense (n_constraints_rot x m)
        nr = self.n_constraints_rot
        _Pr_indptr = np.zeros(m + 1, dtype=np.int32)
        _Pr_rows = []
        for j in range(m):
            _Pr_indptr[j + 1] = _Pr_indptr[j] + (j + 1)
            _Pr_rows.extend(range(j + 1))
        _Pr_nnz = int(_Pr_indptr[-1])
        self._P_rot_sparse = sparse.csc_matrix(
            (np.zeros(_Pr_nnz), np.array(_Pr_rows, dtype=np.int32), _Pr_indptr),
            shape=(m, m),
        )
        self._Pr_triu_rows = np.array(_Pr_rows, dtype=int)
        self._Pr_triu_cols = np.empty(_Pr_nnz, dtype=int)
        for j in range(m):
            self._Pr_triu_cols[_Pr_indptr[j]:_Pr_indptr[j + 1]] = j

        _Ar_indptr = np.arange(0, (m + 1) * nr, nr, dtype=np.int32)
        _Ar_rows = np.tile(np.arange(nr, dtype=np.int32), m)
        self._A_rot_sparse = sparse.csc_matrix(
            (np.zeros(nr * m), _Ar_rows, _Ar_indptr),
            shape=(nr, m),
        )

    def reset_rolling_arrays(self):
        """Zero out arrays before building new QP (rolling step)."""
        self.M_roll.fill(0.0)
        self.b_roll.fill(0.0)
        self.A_gap.fill(0.0)
        self.P.fill(0.0)
        self.c.fill(0.0)

    def reset_rotation_arrays(self):
        """Zero out arrays before building new QP (rotation step)."""
        self.M_roll_rot.fill(0.0)
        self.b_roll_rot.fill(0.0)
        self.A_gap_rot.fill(0.0)
        self.P_rot.fill(0.0)
        self.c_rot.fill(0.0)


# Workspace cache keyed by problem dimensions (m, n_contacts).
_workspace_cache: dict[tuple[int, int], QPWorkspace] = {}


def get_workspace(m: int, n_contacts: int = 2) -> QPWorkspace:
    """Get or create a workspace for the given dimensions."""
    key = (m, n_contacts)
    if key not in _workspace_cache:
        _workspace_cache[key] = QPWorkspace(m, n_contacts)
    return _workspace_cache[key]


def compute_rolling_rotation(
    contact_kin: list["ContactKinematics"],
    delta_p_world: np.ndarray,
    sphere_radius_m: float,
) -> np.ndarray:
    """Compute the sphere rotation required for pure rolling during translation.

    For a sphere rolling between two antipodal contact points:
    - Translation perpendicular to the pinch axis causes rotation
    - The rotation axis is perpendicular to both pinch and translation
    - The rotation magnitude is |delta_p_perp| / r

    With antipodal contacts the rotation terms cancel in the QP's constraint
    equations, leaving the rotation under-determined; this geometric
    construction supplies it explicitly as a soft target.

    Args:
        contact_kin: List of contact kinematics [thumb, index]
        delta_p_world: Sphere center translation vector
        sphere_radius_m: Sphere radius

    Returns:
        Rotation vector (axis * angle) in radians
    """
    if len(contact_kin) < 2:
        return np.zeros(3)

    # Use the first contact's normal as the pinch axis direction
    # (for antipodal contacts, n_thumb ≈ -n_index)
    n_thumb = contact_kin[0].normal_world
    pinch_axis = n_thumb / np.linalg.norm(n_thumb)

    # Translation magnitude
    dp_mag = np.linalg.norm(delta_p_world)
    if dp_mag < 1e-12:
        return np.zeros(3)

    # Component of translation perpendicular to pinch axis (this causes rolling)
    dp_along_component = np.dot(delta_p_world, pinch_axis)
    dp_perp = delta_p_world - dp_along_component * pinch_axis
    dp_perp_mag = np.linalg.norm(dp_perp)

    if dp_perp_mag < 1e-12:
        # Translation is along the pinch axis - no rolling rotation needed
        return np.zeros(3)

    dp_perp_dir = dp_perp / dp_perp_mag

    # Rotation axis: perpendicular to both pinch and translation direction
    # Use right-hand rule: rotation axis = pinch × translation_dir
    rot_axis = np.cross(pinch_axis, dp_perp_dir)
    rot_axis_norm = np.linalg.norm(rot_axis)
    if rot_axis_norm < 1e-12:
        return np.zeros(3)
    rot_axis = rot_axis / rot_axis_norm

    # Rotation angle: arc_length = r * theta => theta = arc_length / r
    theta = dp_perp_mag / sphere_radius_m

    return theta * rot_axis


@dataclass
class RollingStepResult:
    success: bool
    reason: str
    delta_q_active: np.ndarray  # (m,)
    delta_theta: np.ndarray     # (3,)
    roll_residual: float
    tangential_slip: float = 0.0  # Tangential slip (mm) - the "bad" slip
    spin_slip: float = 0.0        # Spin deviation from ideal (rad) - acceptable


@dataclass
class RotationStepResult:
    """Result from a pure rotation step (delta_p = 0)."""
    success: bool
    reason: str
    delta_q_active: np.ndarray  # (m,)
    delta_theta_achieved: np.ndarray  # (3,) actual rotation achieved
    roll_residual: float
    efficiency: float  # |achieved| / |target|, 0 to 1
    tangential_slip: float  # Max tangential slip at contacts (mm) - the "bad" slip
    spin_achieved: float  # Rotation about contact normals (rad) - acceptable "slip"


def solve_rolling_step(
    contact_kin: list[ContactKinematics],  # [thumb_ck, index_ck]
    delta_p_world: np.ndarray,             # desired sphere-centre displacement (3,)
    q_active: np.ndarray,                  # current active joint values (m,)
    q_lower: np.ndarray,                   # active joint lower limits (m,)
    q_upper: np.ndarray,                   # active joint upper limits (m,)
    active_v_ids: np.ndarray,              # velocity indices for active joints
    sphere_radius_m: float,
    link_radius_m: float,
    eps_g_m: float,
    max_joint_step_rad: float,
    qp_regularization: float,
    ctx: "RobotContext | None" = None,     # Optional robot context for mimic joint handling
    tangential_slip_threshold_m: float | None = None,  # None = allow slip, float = reject if slip > threshold
    workspace: QPWorkspace | None = None,  # Optional pre-allocated workspace
) -> RollingStepResult:
    """Solve one rolling step via QP.

    This formulation distinguishes between:
    - Tangential slip: contact slides on finger surface (BAD, constrained)
    - Spin: ball rotates about contact normal (OK, free)

    Decision variables: x = [delta_q_active (m), delta_theta (3)]

    The QP minimizes tangential slip while:
    - Softly preferring the geometrically-ideal rotation
    - Maintaining contact gaps
    - Respecting joint limits
    """
    from .robot import fold_mimic_jacobian

    def _extract_jacobian_cols(J_full: np.ndarray) -> np.ndarray:
        if ctx is not None and ctx.mimic_qs:
            return fold_mimic_jacobian(J_full, active_v_ids, ctx)
        return J_full[:, active_v_ids]

    n_contacts = len(contact_kin)
    m = len(q_active)
    n_vars = m + 3  # dq + delta_theta

    # Get or create workspace for pre-allocated arrays
    ws = workspace if workspace is not None else get_workspace(m, n_contacts)
    ws.reset_rolling_arrays()

    # Compute the ideal rotation for pure rolling (used for soft guidance)
    ideal_theta = compute_rolling_rotation(contact_kin, delta_p_world, sphere_radius_m)

    # Build rolling constraint matrix: M_roll @ x = b_roll
    # Jv_surf @ dq - r * skew(n) @ delta_theta = delta_p
    # Use pre-allocated arrays from workspace
    M_roll = ws.M_roll
    b_roll = ws.b_roll

    # Store normals for tangential slip decomposition
    normals = []

    for i, ck in enumerate(contact_kin):
        row = 3 * i
        n_i = ck.normal_world
        normals.append(n_i)

        # Surface Jacobian
        p_capsule = ck.axis_point_world + float(link_radius_m) * n_i
        r = p_capsule - ck.origin_world
        Jv_origin_active = _extract_jacobian_cols(ck.Jv_origin_world)
        Jw_active = _extract_jacobian_cols(ck.Jw_world)
        Jv_surf_active = Jv_origin_active - skew(r) @ Jw_active

        M_roll[row:row+3, :m] = Jv_surf_active
        M_roll[row:row+3, m:] = -sphere_radius_m * skew(n_i)
        b_roll[row:row+3] = delta_p_world

    # Contact gap constraints - use pre-allocated arrays
    A_gap = ws.A_gap
    l_gap = ws.l_gap
    u_gap = ws.u_gap

    for i, ck in enumerate(contact_kin):
        Jv_axis_active = _extract_jacobian_cols(ck.Jv_world)
        nT_J = ck.normal_world @ Jv_axis_active
        nT_dp = float(ck.normal_world @ delta_p_world)
        A_gap[i, :m] = nT_J
        l_gap[i] = ck.g_m + nT_dp - eps_g_m
        u_gap[i] = ck.g_m + nT_dp + eps_g_m

    # Nondimensionalize QP data by the sphere radius -> scale-free O(1) problem (see docs/scale_invariance_nondim.md)
    M_roll /= sphere_radius_m
    b_roll /= sphere_radius_m
    A_gap  /= sphere_radius_m
    l_gap  /= sphere_radius_m
    u_gap  /= sphere_radius_m

    # Joint and rotation bounds - use pre-allocated arrays
    dq_lo = ws.dq_lo
    dq_hi = ws.dq_hi
    np.maximum(-max_joint_step_rad, q_lower - q_active, out=dq_lo)
    np.minimum(max_joint_step_rad, q_upper - q_active, out=dq_hi)
    theta_bound = 0.2  # rad; per-step bound on the sphere-rotation QP variable

    x_lo = ws.x_lo
    x_hi = ws.x_hi
    x_lo[:m] = dq_lo
    x_lo[m:] = -theta_bound
    x_hi[:m] = dq_hi
    x_hi[m:] = theta_bound

    # Build QP objective using pre-allocated P and c
    # min ||M_roll @ x - b_roll||^2 + lambda_theta * ||theta - ideal_theta||^2 + reg * ||dq||^2
    # The QP data was nondimensionalized by sphere_radius above, so the theta
    # block of M.T@M is O(1) regardless of hand size. lambda_theta is therefore a
    # fixed relative weight on the rotation prior (no per-hand scaling needed);
    # the large value strongly ties delta_theta to the geometric ideal_theta.
    lambda_theta = 10.0 / (0.010 ** 2)  # == 1e5
    P = ws.P
    c = ws.c

    # P = M_roll.T @ M_roll + reg*I + lambda_theta penalty
    np.dot(M_roll.T, M_roll, out=P)
    P[np.diag_indices(n_vars)] += qp_regularization / sphere_radius_m ** 2
    P[m:, m:] += lambda_theta * np.eye(3)

    # c = -M_roll.T @ b_roll - lambda_theta * [0..0, ideal_theta]
    np.dot(M_roll.T, b_roll, out=c)
    c *= -1
    c[m:] -= lambda_theta * ideal_theta

    # Constraints - use pre-allocated A_all, l_all, u_all
    A_all = ws.A_all
    l_all = ws.l_all
    u_all = ws.u_all
    A_all[:n_contacts, :] = A_gap
    # Box part (A_all[n_contacts:, :]) is already identity from workspace init
    l_all[:n_contacts] = l_gap
    l_all[n_contacts:] = x_lo
    u_all[:n_contacts] = u_gap
    u_all[n_contacts:] = x_hi

    # Fill CSC templates in-place (avoids scipy csc_matrix construction)
    P_triu = np.triu(P)
    ws._P_sparse.data[:] = P_triu[ws._P_triu_rows, ws._P_triu_cols]
    ws._A_sparse.data[:] = A_all.ravel(order='F')

    solver = osqp.OSQP()
    solver.setup(
        ws._P_sparse, c, ws._A_sparse, l_all, u_all,
        # QP data is nondimensionalized by the sphere radius, so the residual is O(1) at
        # every hand scale -> a fixed absolute tolerance is correct (see docs/scale_invariance_nondim.md).
        eps_abs=1e-7, eps_rel=1e-7, max_iter=4000,
        verbose=False, polish=False,
    )
    result = solver.solve()

    if result.info.status_val not in (1, 2):
        return RollingStepResult(
            success=False,
            reason=f"qp_{result.info.status}",
            delta_q_active=np.zeros(m),
            delta_theta=np.zeros(3),
            roll_residual=float("inf"),
        )

    x = result.x
    dq = x[:m]
    dtheta = x[m:]

    # Compute residual and decompose into tangential vs normal components
    residual = M_roll @ x - b_roll
    roll_res = float(np.max(np.abs(residual)))

    # Compute tangential slip at each contact (the "bad" slip)
    max_tangential_slip = 0.0
    for i in range(n_contacts):
        res_i = residual[3*i:3*i+3]
        n_i = normals[i]
        # Tangential component = residual minus normal component
        normal_comp = np.dot(res_i, n_i) * n_i
        tangent_comp = res_i - normal_comp
        tangential_slip = np.linalg.norm(tangent_comp)
        max_tangential_slip = max(max_tangential_slip, tangential_slip)

    tangential_slip_mm = max_tangential_slip * sphere_radius_m * 1000.0

    # Compute spin deviation from ideal
    spin_slip = np.linalg.norm(dtheta - ideal_theta)

    # Only reject based on TANGENTIAL slip if threshold is specified
    # (None = lenient mode for low-DOF hands that may need some slip tolerance)
    if tangential_slip_threshold_m is not None and max_tangential_slip * sphere_radius_m > tangential_slip_threshold_m:
        return RollingStepResult(
            success=False,
            reason="tangential_slip",
            delta_q_active=dq,
            delta_theta=dtheta,
            roll_residual=roll_res,
            tangential_slip=tangential_slip_mm,
            spin_slip=spin_slip,
        )

    return RollingStepResult(
        success=True,
        reason="ok",
        delta_q_active=dq,
        delta_theta=dtheta,
        roll_residual=roll_res,
        tangential_slip=tangential_slip_mm,
        spin_slip=spin_slip,
    )


def solve_rotation_step(
    contact_kin: list[ContactKinematics],  # [thumb_ck, index_ck]
    delta_theta_target: np.ndarray,        # desired sphere rotation (3,)
    q_active: np.ndarray,                  # current active joint values (m,)
    q_lower: np.ndarray,                   # active joint lower limits (m,)
    q_upper: np.ndarray,                   # active joint upper limits (m,)
    active_v_ids: np.ndarray,              # velocity indices for active joints
    sphere_radius_m: float,
    link_radius_m: float,
    eps_g_m: float,
    max_joint_step_rad: float,
    qp_regularization: float,
    ctx: "RobotContext | None" = None,
    workspace: QPWorkspace | None = None,  # Optional pre-allocated workspace
) -> RotationStepResult:
    """Solve a pure rotation step (sphere spins in place, delta_p = 0).

    For pure rotation, the rolling constraint becomes:
        J_surf @ dq = r_sphere * skew(n) @ delta_theta

    We solve for dq that best achieves the target delta_theta while:
      - Maintaining contact gap (n · J_v @ dq ≈ 0)
      - Respecting joint limits

    This is a least-squares problem:
        minimize ||M @ dq - b||^2 + reg * ||dq||^2
        subject to gap and joint constraints

    Returns the achieved rotation (may be less than target if kinematically limited).
    """
    from .robot import fold_mimic_jacobian

    def _extract_jacobian_cols(J_full: np.ndarray) -> np.ndarray:
        if ctx is not None and ctx.mimic_qs:
            return fold_mimic_jacobian(J_full, active_v_ids, ctx)
        return J_full[:, active_v_ids]

    n_contacts = len(contact_kin)
    m = len(q_active)

    # Get or create workspace for pre-allocated arrays
    ws = workspace if workspace is not None else get_workspace(m, n_contacts)
    ws.reset_rotation_arrays()

    # Build rolling constraint matrix: M @ dq = b (least-squares target)
    # For each contact: J_surf @ dq = r * skew(n) @ dtheta
    # Use pre-allocated arrays from workspace
    M_roll = ws.M_roll_rot
    b_roll = ws.b_roll_rot

    for i, ck in enumerate(contact_kin):
        row = 3 * i
        n_i = ck.normal_world

        # Surface Jacobian (same as in solve_rolling_step)
        p_capsule = ck.axis_point_world + float(link_radius_m) * n_i
        r = p_capsule - ck.origin_world
        Jv_origin_active = _extract_jacobian_cols(ck.Jv_origin_world)
        Jw_active = _extract_jacobian_cols(ck.Jw_world)
        Jv_surf_active = Jv_origin_active - skew(r) @ Jw_active

        M_roll[row:row+3, :] = Jv_surf_active
        b_roll[row:row+3] = sphere_radius_m * skew(n_i) @ delta_theta_target

    # Gap constraint - use pre-allocated arrays
    A_gap = ws.A_gap_rot
    l_gap = ws.l_gap
    u_gap = ws.u_gap
    for i, ck in enumerate(contact_kin):
        Jv_axis_active = _extract_jacobian_cols(ck.Jv_world)
        A_gap[i, :] = ck.normal_world @ Jv_axis_active
        l_gap[i] = ck.g_m - eps_g_m
        u_gap[i] = ck.g_m + eps_g_m

    # Nondimensionalize QP data by the sphere radius -> scale-free O(1) problem (see docs/scale_invariance_nondim.md)
    M_roll /= sphere_radius_m
    b_roll /= sphere_radius_m
    A_gap  /= sphere_radius_m
    l_gap  /= sphere_radius_m
    u_gap  /= sphere_radius_m

    # Joint bounds - use pre-allocated arrays
    dq_lo = ws.dq_lo
    dq_hi = ws.dq_hi
    np.maximum(-max_joint_step_rad, q_lower - q_active, out=dq_lo)
    np.minimum(max_joint_step_rad, q_upper - q_active, out=dq_hi)

    # QP: minimize ||M @ dq - b||^2 + reg * ||dq||^2
    # Use pre-allocated P and c
    P = ws.P_rot
    c = ws.c_rot
    np.dot(M_roll.T, M_roll, out=P)
    P[np.diag_indices(m)] += qp_regularization / sphere_radius_m ** 2
    np.dot(M_roll.T, b_roll, out=c)
    c *= -1

    # Constraints - use pre-allocated arrays
    A_all = ws.A_all_rot
    l_all = ws.l_all_rot
    u_all = ws.u_all_rot
    A_all[:n_contacts, :] = A_gap
    # Box part (A_all[n_contacts:, :]) is already identity from workspace init
    l_all[:n_contacts] = l_gap
    l_all[n_contacts:] = dq_lo
    u_all[:n_contacts] = u_gap
    u_all[n_contacts:] = dq_hi

    # Fill CSC templates in-place (avoids scipy csc_matrix construction)
    P_rot_triu = np.triu(P)
    ws._P_rot_sparse.data[:] = P_rot_triu[ws._Pr_triu_rows, ws._Pr_triu_cols]
    ws._A_rot_sparse.data[:] = A_all.ravel(order='F')

    solver = osqp.OSQP()
    solver.setup(
        ws._P_rot_sparse, c, ws._A_rot_sparse, l_all, u_all,
        eps_abs=1e-7, eps_rel=1e-7, max_iter=4000,
        verbose=False, polish=False,
    )
    result = solver.solve()

    if result.info.status_val not in (1, 2):
        return RotationStepResult(
            success=False,
            reason=f"qp_{result.info.status}",
            delta_q_active=np.zeros(m),
            delta_theta_achieved=np.zeros(3),
            roll_residual=float("inf"),
            efficiency=0.0,
            tangential_slip=float("inf"),
            spin_achieved=0.0,
        )

    dq = result.x

    # Compute rolling residual
    achieved_surf_vel = M_roll @ dq
    residual = achieved_surf_vel - b_roll
    roll_res = float(np.max(np.abs(residual)))

    # Back-solve for achieved rotation from the rolling equations
    # From J_surf @ dq = r * skew(n) @ dtheta, solve for dtheta in least-squares
    # Use pre-allocated A_theta from workspace
    A_theta = ws.A_theta
    A_theta.fill(0.0)
    for i, ck in enumerate(contact_kin):
        A_theta[3*i:3*i+3, :] = skew(ck.normal_world)

    AtA = A_theta.T @ A_theta + 1e-4 * np.eye(3)  # scale-invariant damping (nondimensionalized)
    dtheta_achieved = np.linalg.solve(AtA, A_theta.T @ achieved_surf_vel)

    # Compute efficiency
    target_norm = np.linalg.norm(delta_theta_target)
    if target_norm > 1e-10:
        # Project achieved onto target direction for signed efficiency
        alignment = np.dot(delta_theta_target, dtheta_achieved) / (target_norm * target_norm)
        efficiency = float(np.clip(alignment, 0.0, 1.0))
    else:
        efficiency = 0.0

    # Compute tangential slip and spin at each contact
    # Tangential slip = slip component perpendicular to contact normal (BAD - linear sliding)
    # Spin = rotation about contact normal (OK - pivoting)
    max_tangential_slip = 0.0
    total_spin = 0.0
    for i, ck in enumerate(contact_kin):
        n_i = ck.normal_world
        slip_i = residual[3*i:3*i+3]  # slip vector at contact i

        # Decompose slip into normal and tangential components
        slip_normal = np.dot(slip_i, n_i) * n_i
        slip_tangent = slip_i - slip_normal
        tangential_slip_mag = np.linalg.norm(slip_tangent)
        max_tangential_slip = max(max_tangential_slip, tangential_slip_mag)

        # Spin = component of achieved rotation about the contact normal
        spin_i = abs(np.dot(dtheta_achieved, n_i))
        total_spin += spin_i

    # Convert tangential slip to mm for readability (residual is nondimensional -> x sphere_radius_m)
    tangential_slip_mm = float(max_tangential_slip * sphere_radius_m * 1000)

    return RotationStepResult(
        success=True,
        reason="ok",
        delta_q_active=dq,
        delta_theta_achieved=dtheta_achieved,
        roll_residual=roll_res,
        efficiency=efficiency,
        tangential_slip=tangential_slip_mm,
        spin_achieved=float(total_spin),
    )
