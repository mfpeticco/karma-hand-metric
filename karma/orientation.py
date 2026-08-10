"""Twist-invariant orientation representation and motion primitives.

For a 2-finger pinch grasp the sphere has one uncontrolled rotational DOF:
spin about the pinch axis (line connecting the two contact points).  We
represent orientation by tracking the *pinch axis direction in the sphere body
frame* — an element of S^2 with antipodal identification (a line, not a ray).
This quotients out the uncontrolled twist.

We bin S^2 directions using HEALPix (Gorski et al., 2005), whose equal-area
pixels keep the coverage fraction from being biased by where on the sphere the
bins fall.  Targeting an angular resolution delta needs roughly N ≈ 4*pi/delta^2
bins, so delta = 10 deg ≈ 0.1745 rad needs N ≈ 413.  nside=6 gives 432 pixels on
S^2, which antipodal identification reduces to the 228 line bins the metric
scores against.
"""
from __future__ import annotations

import numpy as np
import healpy as hp

from .math_utils import unit


# ── HEALPix helpers ──────────────────────────────────────────────────────────

def healpix_npix_lines(nside: int) -> int:
    """Number of unique bins when representing *lines* on S^2 with HEALPix.

    Our orientation representation identifies antipodal directions (a ~ -a), so
    the effective number of bins is smaller than the full HEALPix pixel count.

    For the default HEALPix RING indexing used by healpy, canonicalizing to the
    z>=0 hemisphere (with a deterministic tie-breaker on the equator) yields:
        N_line = 6*nside^2 + 2*nside
    """
    n = int(nside)
    return 6 * n * n + 2 * n


def _canonicalize_axis(axis: np.ndarray) -> np.ndarray:
    """Canonicalize a line direction to the 'positive' hemisphere.

    Since axis and -axis represent the same line, we pick the one in the
    z>0 hemisphere (breaking ties by y, then x).
    """
    a = np.asarray(axis, dtype=float).ravel()
    if a[2] < 0 or (a[2] == 0 and a[1] < 0) or (a[2] == 0 and a[1] == 0 and a[0] < 0):
        a = -a
    return a


def axis_to_healpix_bin(axis: np.ndarray, nside: int) -> int:
    """Map a unit direction (line) to a HEALPix pixel index."""
    a = _canonicalize_axis(unit(axis))
    theta = np.arccos(np.clip(a[2], -1.0, 1.0))
    phi = np.arctan2(a[1], a[0]) % (2.0 * np.pi)
    return int(hp.ang2pix(nside, theta, phi))


def healpix_bin_to_axis(pixel: int, nside: int) -> np.ndarray:
    """Representative unit vector for a HEALPix pixel."""
    theta, phi = hp.pix2ang(nside, pixel)
    return np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])


# ── Twist-invariant pinch axis ──────────────────────────────────────────────

def pinch_axis_world(thumb_normal_world: np.ndarray, index_normal_world: np.ndarray) -> np.ndarray:
    """The pinch axis in world frame.

    For a 2-finger antipodal grasp the two contact normals are roughly
    opposing.  The pinch axis is taken as the thumb inward-normal direction
    (= -thumb_outward_normal, pointing from thumb contact toward sphere centre).
    """
    return unit(-np.asarray(thumb_normal_world))


def contact_line_axis_body(
    thumb_normal_world: np.ndarray,
    index_normal_world: np.ndarray,
    R_sphere: np.ndarray,
) -> np.ndarray:
    """Pinch axis expressed in the sphere body frame.

    R_sphere is the 3x3 rotation matrix of the sphere body frame w.r.t. world.
    """
    axis_w = pinch_axis_world(thumb_normal_world, index_normal_world)
    R = np.asarray(R_sphere, dtype=float).reshape(3, 3)
    return unit(R.T @ axis_w)


# ── Motion primitives ────────────────────────────────────────────────────────

def compute_motion_primitives_frame(R_seed: np.ndarray, step_size: float) -> list[np.ndarray]:
    """The six Phase-1 translation primitives: ±step along each seed-frame axis.

    Args:
        R_seed: (3, 3) rotation whose columns are the principal mobility
            directions (eigenvectors of the seed manipulability ellipsoid).
        step_size: Displacement magnitude in metres.

    Returns:
        Six delta_p vectors in the world frame, ordered so that primitive ``i``
        and ``i ^ 1`` are opposites — the BFS relies on this to skip the move
        back toward a voxel's parent.
    """
    s = float(step_size)
    return [
        +s * R_seed[:, 0],
        -s * R_seed[:, 0],
        +s * R_seed[:, 1],
        -s * R_seed[:, 1],
        +s * R_seed[:, 2],
        -s * R_seed[:, 2],
    ]


def compute_rotation_primitives(
    pinch_axis_w: np.ndarray,
    rotation_step_rad: float,
) -> list[np.ndarray]:
    """Generate rotation primitives perpendicular to the pinch axis.

    For a 2-finger pinch grasp, rotation about the pinch axis is uncontrolled
    (twist DOF). Only rotations perpendicular to the pinch axis change the
    orientation bin (tilt the pinch axis to point in a new direction).

    Returns 4 rotation vectors: ±perp1, ±perp2 (perpendicular to pinch axis).

    Args:
        pinch_axis_w: Current pinch axis direction in world frame
        rotation_step_rad: Rotation magnitude in radians (~0.15 rad ~= 8.6 deg)

    Returns:
        List of 4 rotation vectors (world frame, radians)
    """
    g = unit(pinch_axis_w)

    # Build orthonormal basis perpendicular to pinch axis
    if abs(g[0]) < 0.9:
        perp1 = np.cross(g, np.array([1.0, 0.0, 0.0]))
    else:
        perp1 = np.cross(g, np.array([0.0, 1.0, 0.0]))
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(g, perp1)

    s = float(rotation_step_rad)
    return [
        +s * perp1,
        -s * perp1,
        +s * perp2,
        -s * perp2,
    ]
