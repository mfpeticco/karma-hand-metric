"""Small math helpers."""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation


def point_to_voxel(
    p: np.ndarray, origin: np.ndarray, voxel_size: float, R_inv: np.ndarray,
) -> tuple[int, int, int]:
    """Map a world-space point to an integer voxel index in the seed frame."""
    d = R_inv @ (p - origin)
    idx = np.round(d / voxel_size).astype(int)
    return (int(idx[0]), int(idx[1]), int(idx[2]))


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    """URDF roll-pitch-yaw to a 3x3 rotation matrix (R = Rz(yaw) @ Ry(pitch) @ Rx(roll))."""
    roll, pitch, yaw = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def skew(v: np.ndarray) -> np.ndarray:
    """3x3 skew-symmetric matrix from a 3-vector."""
    v = np.asarray(v, dtype=float).ravel()
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])


def unit(v: np.ndarray) -> np.ndarray:
    """Normalize to unit length. Returns zero vector if input is near-zero."""
    v = np.asarray(v, dtype=float).ravel()
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.zeros_like(v)
    return v / n


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def exp_so3(omega: np.ndarray) -> np.ndarray:
    """Exponential map: rotation vector -> 3x3 rotation matrix."""
    angle = np.linalg.norm(omega)
    if angle < 1e-10:
        return np.eye(3)
    return Rotation.from_rotvec(omega).as_matrix()


def closest_points_on_segments(
    p0: np.ndarray, d0: np.ndarray, half_len0: float,
    p1: np.ndarray, d1: np.ndarray, half_len1: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Closest points between two line segments (Ericson's real-time collision detection).

    Each segment is defined by center p, unit direction d, and half-length.
    Segment spans [p - half_len*d, p + half_len*d].

    Returns (point_on_seg0, point_on_seg1).
    """
    a0 = p0 - half_len0 * d0
    b0 = p0 + half_len0 * d0
    a1 = p1 - half_len1 * d1
    b1 = p1 + half_len1 * d1

    d0_full = b0 - a0  # = 2*half_len0*d0
    d1_full = b1 - a1
    r = a0 - a1

    a = float(np.dot(d0_full, d0_full))
    e = float(np.dot(d1_full, d1_full))
    f = float(np.dot(d1_full, r))

    EPS = 1e-12

    if a <= EPS and e <= EPS:
        return a0.copy(), a1.copy()

    if a <= EPS:
        t = clamp(f / e, 0.0, 1.0)
        return a0.copy(), a1 + t * d1_full

    c = float(np.dot(d0_full, r))
    if e <= EPS:
        s = clamp(-c / a, 0.0, 1.0)
        return a0 + s * d0_full, a1.copy()

    b = float(np.dot(d0_full, d1_full))
    denom = a * e - b * b

    if abs(denom) > EPS:
        s = clamp((b * f - c * e) / denom, 0.0, 1.0)
    else:
        s = 0.0

    t = (b * s + f) / e
    if t < 0.0:
        t = 0.0
        s = clamp(-c / a, 0.0, 1.0)
    elif t > 1.0:
        t = 1.0
        s = clamp((b - c) / a, 0.0, 1.0)

    return a0 + s * d0_full, a1 + t * d1_full


# ---------------------------------------------------------------------------
# Seed manipulability frame
# ---------------------------------------------------------------------------

def compute_object_jacobian(
    ck_thumb,
    ck_index,
    v_ids: np.ndarray,
    link_radius_m: float,
    fold_mimic_fn=None,
) -> np.ndarray:
    """Object-motion Jacobian: dp_sphere/dq for a two-contact pinch grasp.

    For two contacts with surface Jacobians Jv_surf_0, Jv_surf_1, the rolling
    constraints yield (exact for perfectly antipodal contacts):

        J_obj = dp/dq = 0.5 * (Jv_surf_0 + Jv_surf_1)

    Args:
        ck_thumb: ContactKinematics for thumb contact.
        ck_index: ContactKinematics for index contact.
        v_ids: Active velocity indices.
        link_radius_m: Capsule link radius.
        fold_mimic_fn: Optional callable (J_full, v_ids) -> J_active.

    Returns:
        J_obj: (3, m) object-motion Jacobian.
    """
    def _extract(J_full):
        if fold_mimic_fn is not None:
            return fold_mimic_fn(J_full, v_ids)
        return J_full[:, v_ids]

    m = len(v_ids)
    J_surf = np.zeros((2, 3, m))
    for i, ck in enumerate([ck_thumb, ck_index]):
        n_i = ck.normal_world
        p_capsule = ck.axis_point_world + link_radius_m * n_i
        r = p_capsule - ck.origin_world
        Jv_origin_active = _extract(ck.Jv_origin_world)
        Jw_active = _extract(ck.Jw_world)
        J_surf[i] = Jv_origin_active - skew(r) @ Jw_active

    return 0.5 * (J_surf[0] + J_surf[1])


def compute_seed_frame(J_obj: np.ndarray) -> np.ndarray:
    """Canonical frame from the object-motion Jacobian's manipulability ellipsoid.

    Eigendecomposes M = J_obj @ J_obj.T to find principal mobility directions.

    Sign convention: for each eigenvector, the component with the largest
    absolute value is forced positive. Guarantees det(R) = +1.

    Returns:
        R_seed: (3, 3) rotation matrix, columns = principal directions (descending λ).
    """
    M = J_obj @ J_obj.T  # (3, 3) PSD

    eigenvalues, eigenvectors = np.linalg.eigh(M)
    # eigh returns ascending; reverse for descending
    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Only fall back to I when the Jacobian is truly zero (no mobility at all).
    # For near-spherical ellipsoids (close eigenvalues), we still use the
    # eigenvectors: they rotate covariantly with the robot, preserving rotation
    # invariance.  Returning I for near-degenerate cases would lock the grid to
    # world axes, breaking rotation invariance.
    lam_max = eigenvalues[0]
    if lam_max < 1e-12:
        return np.eye(3)

    # Deterministic sign: largest absolute component is positive.
    # Grid is symmetric under axis reflection, so this sign choice
    # doesn't affect voxel counts.
    for col in range(3):
        v = eigenvectors[:, col]
        max_idx = int(np.argmax(np.abs(v)))
        if v[max_idx] < 0:
            eigenvectors[:, col] = -v

    # Ensure right-handedness
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 2] = -eigenvectors[:, 2]

    return eigenvectors
