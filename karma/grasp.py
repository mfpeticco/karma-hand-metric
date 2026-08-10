"""Antipodal + friction-cone feasibility for a 2-finger pinch grasp."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GraspFeasibility:
    feasible: bool
    reason: str
    cos_angle: float  # cos(angle between opposing force directions)


def check_antipodal_friction(
    contact_normals_world: np.ndarray,
    friction_coeff: float,
) -> GraspFeasibility:
    """Check 2-contact antipodal feasibility with friction cones.

    contact_normals_world: shape (2, 3), the two contact normals as produced by
        ``contact_kinematics`` (each points from the capsule axis toward the
        sphere centre). Only a *consistent* convention matters: the test below is
        invariant to flipping both normals, so inward-vs-outward does not change
        the result.

    A pinch is feasible iff force-closure holds: forces inside the two friction
    cones can sum to zero. Each cone opens by ``atan(mu)`` about its inward
    normal, so the two inward normals must be within ``2*atan(mu)`` of
    anti-parallel. With both normals pointing toward the centre, anti-parallel
    means ``n0 ~ -n1``, i.e. ``dot(-n0, n1) ~ +1``:

        dot(-n0, n1) >= cos(2 * atan(mu))
    """
    n0 = np.asarray(contact_normals_world[0], dtype=float)
    n1 = np.asarray(contact_normals_world[1], dtype=float)

    norm0 = np.linalg.norm(n0)
    norm1 = np.linalg.norm(n1)
    if norm0 < 1e-12 or norm1 < 1e-12:
        return GraspFeasibility(False, "degenerate_normal", 0.0)

    n0 = n0 / norm0
    n1 = n1 / norm1

    # Force closure: contact i can exert a force anywhere within atan(mu) of its
    # inward normal (here -n_i is redundant since both n_i point inward already).
    # Two such forces can cancel iff the inward normals are within 2*atan(mu) of
    # anti-parallel. For a good pinch n0 ~ -n1, so dot(-n0, n1) ~ +1, and the
    # feasibility threshold is cos(2*atan(mu)).
    cos_val = float(np.dot(-n0, n1))  # near +1 for a good pinch
    half_angle = math.atan(friction_coeff)
    threshold = math.cos(2.0 * half_angle)

    if cos_val >= threshold:
        return GraspFeasibility(True, "ok", cos_val)
    else:
        return GraspFeasibility(False, "friction_cone_violation", cos_val)
