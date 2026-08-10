#!/usr/bin/env python3
"""Constraint ablation experiment: run BFS with constraints removed incrementally.

Ablation levels:
  (i)   Gap-only — no joint limits, no collision, no antipodal
  (ii)  Gap + joint limits
  (iii) Gap + joint limits + collisions
  (iv)  Full KaRMA (all constraints)

Runs 5 representative hands × 4 levels = 20 runs.
Uses the SAME seed for all levels (from full-constraint seed selection).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Repo root (this script lives in experiments/); make `karma` importable when run directly.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from karma.ablation import ablation_context, AblationLevel
from karma.config import load_config
from karma.robot import load_robot
from karma.metric import compute_metric
from karma.seed_selection import select_seed_and_frame

logger = logging.getLogger(__name__)

REPRESENTATIVE_HANDS = [
    "robot_leap",
    "robot_shadowhand",
    "robot_xhand1",
    "robot_dex3",
    "robot_inspire",
]

ABLATION_LEVELS = [
    {"name": "gap_only",   "label": "(i) Gap-only",     "level": AblationLevel.GAP_ONLY},
    {"name": "gap_jl",     "label": "(ii) +Joint limits","level": AblationLevel.GAP_JL},
    {"name": "gap_jl_col", "label": "(iii) +Collisions", "level": AblationLevel.GAP_JL_COL},
    {"name": "full",       "label": "(iv) Full KaRMA",   "level": AblationLevel.FULL},
]


def run_ablation(
    robot_name: str,
    metric_config_path: str,
    rotation_workers: int | None,
    out_dir: Path,
) -> dict:
    """Run all 4 ablation levels for one robot, return results dict."""
    robot_config_path = str(ROOT / "robots" / f"{robot_name}.yaml")
    results = {}

    # Compute seed ONCE with full constraints (no ablation)
    logger.info("Computing seed for %s with full constraints...", robot_name)
    cfg = load_config(metric_config_path, robot_config_path)
    ctx = load_robot(cfg.urdf_path, cfg)
    seed, R_seed, seed_sensitivity, top_seeds = select_seed_and_frame(cfg, ctx)
    precomputed_seed = (seed, R_seed, seed_sensitivity, top_seeds)
    logger.info("Seed computed: contact_pair=%s, feasible=%s", seed.contact_pair, seed.feasible)

    for level_info in ABLATION_LEVELS:
        level_name = level_info["name"]
        level = level_info["level"]
        logger.info("=" * 60)
        logger.info("%s — %s", robot_name, level_info["label"])
        logger.info("=" * 60)

        # Reload robot context for each level (ablation modifies model in-place)
        ctx = load_robot(cfg.urdf_path, cfg)

        t0 = time.perf_counter()
        with ablation_context(level, ctx, cfg.finger_joint_names):
            result = compute_metric(
                cfg, ctx,
                n_rotation_workers=rotation_workers,
                precomputed_seed=precomputed_seed,
            )
        elapsed = time.perf_counter() - t0

        results[level_name] = {
            "label": level_info["label"],
            "n_voxels": result.n_voxels_reached,
            "karma_t": result.translational_score,
            "karma_r": result.global_rotational_score,
            "n_states": result.n_states_reached,
            "elapsed_s": round(elapsed, 1),
            "phase1_fail_reasons": result.phase1_fail_reasons,
            "phase2_fail_reasons": result.phase2_fail_reasons,
        }

        logger.info(
            "%s %s: voxels=%d, KaRMA-T=%.6f, KaRMA-R=%.4f, time=%.1fs",
            robot_name, level_info["label"],
            result.n_voxels_reached,
            result.translational_score,
            result.global_rotational_score,
            elapsed,
        )

    # Save per-robot results
    robot_out = out_dir / f"{robot_name}_ablation.json"
    with open(robot_out, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved %s", robot_out)

    return results


def print_summary(all_results: dict) -> None:
    """Print summary table."""
    print("\n" + "=" * 90)
    print("CONSTRAINT ABLATION RESULTS")
    print("=" * 90)

    # KaRMA-T table
    print("\nKaRMA-T (translational score):")
    header = f"{'Hand':<15} {'(i) Gap-only':>12} {'(ii) +JL':>12} {'(iii) +Col':>12} {'(iv) Full':>12}"
    print(header)
    print("-" * len(header))
    for robot_name, results in all_results.items():
        short = robot_name.replace("robot_", "")
        vals = [f"{results[l['name']]['karma_t']:.6f}" for l in ABLATION_LEVELS]
        print(f"{short:<15} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12} {vals[3]:>12}")

    # Voxel count table
    print("\nVoxel counts:")
    header = f"{'Hand':<15} {'(i) Gap-only':>12} {'(ii) +JL':>12} {'(iii) +Col':>12} {'(iv) Full':>12}"
    print(header)
    print("-" * len(header))
    for robot_name, results in all_results.items():
        short = robot_name.replace("robot_", "")
        vals = [str(results[l["name"]]["n_voxels"]) for l in ABLATION_LEVELS]
        print(f"{short:<15} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12} {vals[3]:>12}")

    # KaRMA-R table
    print("\nKaRMA-R (rotational score):")
    header = f"{'Hand':<15} {'(i) Gap-only':>12} {'(ii) +JL':>12} {'(iii) +Col':>12} {'(iv) Full':>12}"
    print(header)
    print("-" * len(header))
    for robot_name, results in all_results.items():
        short = robot_name.replace("robot_", "")
        vals = [f"{results[l['name']]['karma_r']:.4f}" for l in ABLATION_LEVELS]
        print(f"{short:<15} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12} {vals[3]:>12}")

    # Failure breakdown for full KaRMA
    print("\nPhase 1 failure breakdown (full KaRMA):")
    for robot_name, results in all_results.items():
        short = robot_name.replace("robot_", "")
        reasons = results["full"]["phase1_fail_reasons"]
        if reasons:
            reason_str = ", ".join(f"{k}={v}" for k, v in sorted(reasons.items(), key=lambda x: -x[1]))
            print(f"  {short}: {reason_str}")

    print("\n" + "=" * 90)


def main() -> None:
    parser = argparse.ArgumentParser(description="Constraint Ablation Experiment")
    parser.add_argument(
        "--metric-config", default="karma_config.yaml",
        help="Global KaRMA config",
    )
    parser.add_argument(
        "--rotation-workers", type=int, default=None,
        help="Number of rotation workers (default: one per CPU core)",
    )
    parser.add_argument(
        "--out-dir", default="results/ablation",
        help="Output directory for results (default: overwrites the shipped Table IV data)",
    )
    parser.add_argument(
        "--robots", nargs="*", default=None,
        help="Specific robots to run by name, e.g. 'inspire leap' (default: all 5 representative)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    robots = args.robots if args.robots else REPRESENTATIVE_HANDS
    # Accept either "inspire" or the config stem "robot_inspire".
    robots = [r if r.startswith("robot_") else f"robot_{r}" for r in robots]
    all_results = {}

    for robot_name in robots:
        results = run_ablation(
            robot_name,
            args.metric_config,
            args.rotation_workers,
            out_dir,
        )
        all_results[robot_name] = results

    # Save combined results
    combined_path = out_dir / "ablation_combined.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Combined results saved to %s", combined_path)

    print_summary(all_results)


if __name__ == "__main__":
    main()
