"""Metric result types (MetricResult, StateNode) and output writing.

This is the foundation layer shared by seed_selection.py and metric.py, so it
must not import either of them. The manifold-projection solver lives in
:mod:`karma.projection`.
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from .config import MetricConfig

logger = logging.getLogger(__name__)

# Schema tag embedded in every saved result pickle. Metadata only: the loader in
# karma.storage records it but never gates behavior on it, and the compat
# unpickler handles older tags regardless (the shipped paper pickles carry
# dev-era tags such as "7").
RESULT_SCHEMA_VERSION = "1.0"


# ── Types ─────────────────────────────────────────────────────────────────────

VoxelIdx = tuple[int, int, int]
State = tuple[VoxelIdx, int]  # (voxel_index, healpix_bin)


@dataclass
class StateNode:
    voxel: VoxelIdx
    ori_bin: int
    centre_world: np.ndarray
    q: np.ndarray
    R_sphere: np.ndarray           # 3x3 sphere body orientation
    contact_pair: tuple[str, str]
    parent: State | None
    arrival_prim: int = -1  # Translation primitive index (-1 = seed or rotation)


@dataclass
class MetricResult:
    # Per-voxel orientation bin sets
    voxel_ori_bins: dict[VoxelIdx, set[int]]
    # Best joint config per (voxel, ori_bin)
    state_nodes: dict[State, StateNode]

    # Scalar summaries
    n_voxels_reached: int
    n_states_reached: int
    total_ori_bins: int
    translational_score: float        # KaRMA-T = top3_mean_voxels * (voxel/L_ref)^3 (dimensionless)
    translational_volume_m3: float    # V  = best_seed_voxels * voxel^3 (best seed only)
    l_ref_m: float
    rotational_coverage_per_voxel: dict[VoxelIdx, float]
    global_rotational_score: float    # KaRMA-R: top-3 seed ensemble of top-5-voxel mean rotation coverage

    seed_centre: np.ndarray
    seed_q: np.ndarray
    seed_contact_pair: tuple[str, str]

    # Seed manipulability frame (3x3 rotation matrix, columns = principal directions)
    seed_frame: np.ndarray = field(default_factory=lambda: np.eye(3))

    # Seed sensitivity (KaRMA-S): distribution of voxel counts across all evaluated seeds
    seed_sensitivity: dict | None = None

    # BFS failure reason breakdown (Phase 1 and Phase 2)
    phase1_fail_reasons: dict[str, int] = field(default_factory=dict)
    phase2_fail_reasons: dict[str, int] = field(default_factory=dict)


# ── Output ────────────────────────────────────────────────────────────────────

def write_outputs(cfg: MetricConfig, result: MetricResult) -> Path:
    """Write metric results to current.yaml in the output directory."""
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.out_dir / "current.yaml"
    out_pkl_path = cfg.out_dir / "current.pkl"

    # Build serializable output
    voxel_data = {}
    for v, bins in result.voxel_ori_bins.items():
        key = f"{v[0]}_{v[1]}_{v[2]}"
        voxel_data[key] = {
            "ori_bins": sorted(bins),
            "n_bins": len(bins),
            "rot_coverage": result.rotational_coverage_per_voxel[v],
        }

    output = {
        "summary": {
            "n_voxels_reached": result.n_voxels_reached,
            "n_states_reached": result.n_states_reached,
            "total_ori_bins": result.total_ori_bins,
            "translational_score": float(result.translational_score),
            "translational_volume_m3": float(result.translational_volume_m3),
            "l_ref_m": float(result.l_ref_m),
            "global_rotational_score": float(result.global_rotational_score),
        },
        "seed_sensitivity": result.seed_sensitivity if result.seed_sensitivity else {},
        "seed": {
            "centre_world": result.seed_centre.tolist(),
            "contact_pair": list(result.seed_contact_pair),
            "q": result.seed_q.tolist(),
            "seed_frame": result.seed_frame.tolist(),
        },
        "voxels": voxel_data,
    }

    with open(out_path, "w") as f:
        yaml.dump(output, f, default_flow_style=False, sort_keys=False)

    # Also persist the full MetricResult for interactive inspection (e.g., viser).
    # Keep the same container format as karma.storage.{save_result,load_result}.
    with open(out_pkl_path, "wb") as f:
        pickle.dump(
            {
                "result": result,
                "urdf_path": str(cfg.urdf_path),
                "timestamp": datetime.now(),
                "version": RESULT_SCHEMA_VERSION,
            },
            f,
        )

    logger.info("Wrote results to %s", out_path)
    return out_path
