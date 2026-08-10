#!/usr/bin/env python3
"""Fixed-seed stability test: isolates BFS physics from seed re-selection.

For each robot:
  1. Run baseline with full seed selection → capture seed
  2. Run all perturbations with that FIXED seed (no re-selection)
  3. Report smooth, monotonic sensitivity numbers

Usage (run from the repo root):
  python experiments/run_fixed_seed_stability.py robots/robot_leap.yaml
  python experiments/run_fixed_seed_stability.py ALL
"""
from __future__ import annotations

import argparse
import copy
import logging
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from karma.config import load_config
from karma.robot import load_robot
from karma.seed_selection import select_seed_and_frame
from karma.metric import compute_metric
from robust_tests.test_utils import (
    perturb_urdf_joint_origins,
    perturb_urdf_joint_limits,
    get_active_joint_names,
)

logging.basicConfig(level=logging.WARNING)

METRIC_CFG = str(ROOT / "karma_config.yaml")

PERTURBATIONS = {
    "link_length": [0.01, -0.01, 0.02, -0.02],
    "joint_limits_deg": [1.0, -1.0, 3.0, -3.0],
    "sphere_radius": [0.02, -0.02, 0.05, -0.05],
    "link_radius": [0.02, -0.02, 0.05, -0.05],
}

REPRESENTATIVE = [
    str(ROOT / "robots" / "robot_leap.yaml"),
    str(ROOT / "robots" / "robot_shadowhand.yaml"),
    str(ROOT / "robots" / "robot_xhand1.yaml"),
    str(ROOT / "robots" / "robot_dex3.yaml"),
    str(ROOT / "robots" / "robot_inspire.yaml"),
]


def run_metric_fixed_seed(metric_path, robot_path, precomputed_seed):
    """Run metric with a pre-selected seed (skip seed selection)."""
    cfg = load_config(metric_path, robot_path)
    ctx = load_robot(cfg.urdf_path, cfg)

    result = compute_metric(
        cfg, ctx,
        precomputed_seed=precomputed_seed,
    )
    return result


def run_robot(robot_cfg_path):
    """Run full fixed-seed stability suite for one robot."""
    name = os.path.basename(robot_cfg_path).replace("robot_", "").replace(".yaml", "")
    tmpdir = tempfile.mkdtemp(prefix=f"fixseed_{name}_")

    print(f"\n{'='*60}")
    print(f"  ROBOT: {name}")
    print(f"{'='*60}")

    # ── Baseline: full seed selection ──
    t0 = time.time()
    cfg = load_config(METRIC_CFG, robot_cfg_path)
    ctx = load_robot(cfg.urdf_path, cfg)
    seed, R_seed, seed_sensitivity, top_seeds = select_seed_and_frame(cfg, ctx)
    precomputed = (seed, R_seed, seed_sensitivity, top_seeds)

    result_base = compute_metric(
        cfg, ctx,
        precomputed_seed=precomputed,
    )
    t_base = time.time() - t0
    n_base = result_base.n_voxels_reached
    t_score_base = result_base.translational_score
    r_score_base = result_base.global_rotational_score
    print(f"  Baseline: {n_base} voxels, T={t_score_base:.5f}, R={r_score_base:.4f} ({t_base:.0f}s)")

    # Resolve URDF path
    with open(robot_cfg_path) as f:
        rcfg = yaml.safe_load(f)
    robot_dir = os.path.dirname(os.path.abspath(robot_cfg_path))
    base_urdf = os.path.join(robot_dir, rcfg["urdf_path"])

    # Joint names for the joint-limit perturbation.
    joint_names = get_active_joint_names(robot_cfg_path)

    # Get base metric params
    with open(METRIC_CFG) as f:
        metric_data = yaml.safe_load(f)
    base_sphere_r = metric_data["sphere_radius_m"]
    base_link_r = metric_data["link_radius_m"]

    results = {
        "robot": name,
        "baseline_voxels": n_base,
        "baseline_t_score": float(t_score_base),
        "baseline_r_score": float(r_score_base),
        "perturbations": {},
    }

    # ── Perturbation loop ──
    for ptype, levels in PERTURBATIONS.items():
        results["perturbations"][ptype] = []

        for level in levels:
            t0 = time.time()

            if ptype == "link_length":
                mod_urdf = os.path.join(tmpdir, f"ll_{level:+.3f}.urdf")
                perturb_urdf_joint_origins(base_urdf, level, mod_urdf)
                mod_robot = os.path.join(tmpdir, f"robot_ll_{level:+.3f}.yaml")
                rcfg_mod = copy.deepcopy(rcfg)
                rcfg_mod["urdf_path"] = mod_urdf
                with open(mod_robot, "w") as f:
                    yaml.dump(rcfg_mod, f)
                r = run_metric_fixed_seed(METRIC_CFG, mod_robot, precomputed)
                label = f"link_length {level:+.1%}"

            elif ptype == "joint_limits_deg":
                delta_rad = math.radians(level)
                mod_urdf = os.path.join(tmpdir, f"jl_{level:+.1f}.urdf")
                perturb_urdf_joint_limits(base_urdf, joint_names, delta_rad, mod_urdf)
                mod_robot = os.path.join(tmpdir, f"robot_jl_{level:+.1f}.yaml")
                rcfg_mod = copy.deepcopy(rcfg)
                rcfg_mod["urdf_path"] = mod_urdf
                with open(mod_robot, "w") as f:
                    yaml.dump(rcfg_mod, f)
                r = run_metric_fixed_seed(METRIC_CFG, mod_robot, precomputed)
                label = f"joint_limits {level:+.1f}deg"

            elif ptype == "sphere_radius":
                new_r = base_sphere_r * (1.0 + level)
                mod_metric = os.path.join(tmpdir, f"metric_sr_{level:+.3f}.yaml")
                metric_copy = copy.deepcopy(metric_data)
                metric_copy["sphere_radius_m"] = new_r
                with open(mod_metric, "w") as f:
                    yaml.dump(metric_copy, f)
                r = run_metric_fixed_seed(mod_metric, robot_cfg_path, precomputed)
                label = f"sphere_radius {level:+.1%}"

            elif ptype == "link_radius":
                new_r = base_link_r * (1.0 + level)
                mod_metric = os.path.join(tmpdir, f"metric_lr_{level:+.3f}.yaml")
                metric_copy = copy.deepcopy(metric_data)
                metric_copy["link_radius_m"] = new_r
                with open(mod_metric, "w") as f:
                    yaml.dump(metric_copy, f)
                r = run_metric_fixed_seed(mod_metric, robot_cfg_path, precomputed)
                label = f"link_radius {level:+.1%}"

            elapsed = time.time() - t0
            nv = r.n_voxels_reached
            delta = nv - n_base
            pct = delta / n_base * 100 if n_base > 0 else 0

            results["perturbations"][ptype].append({
                "label": label,
                "level": float(level),
                "voxels": int(nv),
                "delta": int(delta),
                "pct": round(pct, 1),
                "t_score": float(r.translational_score),
                "r_score": float(r.global_rotational_score),
                "elapsed": round(elapsed, 1),
            })
            print(f"  {label:>25s}: {nv:>5d} vox (Δ={delta:>+5d}, {pct:>+6.1f}%) [{elapsed:.0f}s]")

    return results


def print_summary(all_results):
    """Print a clean comparison table."""
    print("\n" + "=" * 80)
    print("FIXED-SEED STABILITY SUMMARY")
    print("=" * 80)

    robots = [r["robot"] for r in all_results]
    baselines = {r["robot"]: r["baseline_voxels"] for r in all_results}

    hdr = f"{'Perturbation':>25s}"
    for name in robots:
        hdr += f" | {name:>8s}({baselines[name]})"
    print(hdr)
    print("-" * len(hdr))

    for ptype in PERTURBATIONS:
        for i, _level in enumerate(PERTURBATIONS[ptype]):
            row = ""
            label = ""
            for r in all_results:
                entry = r["perturbations"][ptype][i]
                if not label:
                    label = entry["label"]
                delta = entry["delta"]
                pct = entry["pct"]
                row += f" | {delta:>+4d} ({pct:>+5.1f}%)"
            print(f"{label:>25s}{row}")
        print()

    # Monotonicity check
    print("MONOTONICITY CHECK:")
    all_mono = True
    for ptype in PERTURBATIONS:
        levels = PERTURBATIONS[ptype]
        pos_levels = sorted([lvl for lvl in levels if lvl > 0])
        neg_levels = sorted([lvl for lvl in levels if lvl < 0], reverse=True)

        for r in all_results:
            entries = {e["level"]: e for e in r["perturbations"][ptype]}
            pos_deltas = [entries[lvl]["delta"] for lvl in pos_levels]
            neg_deltas = [entries[lvl]["delta"] for lvl in neg_levels]

            pos_mono = all(pos_deltas[i] <= pos_deltas[i+1] for i in range(len(pos_deltas)-1)) or \
                       all(pos_deltas[i] >= pos_deltas[i+1] for i in range(len(pos_deltas)-1))
            neg_mono = all(neg_deltas[i] <= neg_deltas[i+1] for i in range(len(neg_deltas)-1)) or \
                       all(neg_deltas[i] >= neg_deltas[i+1] for i in range(len(neg_deltas)-1))

            status = "MONO" if (pos_mono and neg_mono) else "NON-MONO"
            if status == "NON-MONO":
                all_mono = False
            print(f"  {r['robot']:>12s} {ptype:>20s}: pos={pos_deltas} neg={neg_deltas} -> {status}")

    print(f"\nOverall: {'ALL_MONO' if all_mono else 'HAS_NON_MONO'}")


def main():
    parser = argparse.ArgumentParser(
        description="Fixed-seed stability: rerun each perturbation with the "
                    "baseline's seed held fixed, isolating BFS physics from "
                    "seed re-selection.",
    )
    parser.add_argument(
        "robot",
        help="Robot config to test (e.g. robots/robot_leap.yaml), or ALL for "
             "the five representative hands (several hours of compute)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "robustness" / "fixed_seed_stability.yaml",
        help="Output YAML; the default overwrites the shipped robustness data",
    )
    args = parser.parse_args()

    if args.robot != "ALL":
        results = [run_robot(args.robot)]
    else:
        results = [run_robot(rcfg) for rcfg in REPRESENTATIVE]

    print_summary(results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        yaml.safe_dump(results, f, default_flow_style=False)
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
