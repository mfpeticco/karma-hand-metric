#!/usr/bin/env python3
"""CLI runner: discover robot configs, compute all baselines, write YAML + CSV.

Usage:
    python -m baselines.run_baselines                       # all robots
    python -m baselines.run_baselines --configs path/to/robot_allegro.yaml
    python -m baselines.run_baselines --n-samples 100000
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import yaml

# Run either as a module (`python -m baselines.run_baselines`) or as a script
# (`python baselines/run_baselines.py`): put the repo root on sys.path so the
# absolute imports below resolve in both cases.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from karma.lref import compute_lref
from baselines.robot_loader import load_baseline_robot
from baselines.compute_baselines import compute_all_baselines, BaselineResults

logger = logging.getLogger(__name__)


def _rel_to_root(p: str | Path) -> str:
    """Path relative to the repo root, for portable summaries (no home paths)."""
    try:
        return str(Path(p).resolve().relative_to(_REPO_ROOT))
    except ValueError:
        return Path(p).name


def _discover_configs(robot_dir: Path) -> list[Path]:
    configs = sorted(robot_dir.glob("robot_*.yaml"))
    return [p for p in configs if "template" not in p.name]


def _results_to_dict(r: BaselineResults) -> dict:
    return dataclasses.asdict(r)


def _run_one(
    yaml_path: Path,
    n_samples: int,
) -> tuple[BaselineResults, float]:
    robot = load_baseline_robot(yaml_path)
    l_ref_m = compute_lref(yaml_path)
    t0 = time.perf_counter()
    results = compute_all_baselines(robot, l_ref_m, n_samples=n_samples)
    elapsed = time.perf_counter() - t0
    return results, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute kinematic baselines for all robot hands")
    parser.add_argument(
        "--robot-dir",
        default=None,
        help="Directory with robot_*.yaml files (default: ../robots/ relative to this file)",
    )
    parser.add_argument(
        "--configs",
        nargs="*",
        default=None,
        help="Explicit YAML paths (overrides --robot-dir discovery)",
    )
    parser.add_argument("--n-samples", type=int, default=65_536, help="Sobol samples per finger; rounded up to a power of two (default: 65536 = 2^16)")
    parser.add_argument("--out-dir", default="results/baselines", help="Output directory (default: results/baselines)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Discover configs
    if args.configs:
        cfg_paths = [Path(p) for p in args.configs]
    else:
        if args.robot_dir:
            robot_dir = Path(args.robot_dir)
        else:
            robot_dir = Path(__file__).resolve().parent.parent / "robots"
        cfg_paths = _discover_configs(robot_dir)

    if not cfg_paths:
        raise SystemExit("No robot configs found.")

    logger.info("Found %d robot configs", len(cfg_paths))

    # Prepare output directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = Path(args.out_dir) / f"batch_{run_id}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    all_results: list[dict] = []

    for cfg_path in cfg_paths:
        logger.info("=== %s ===", cfg_path.stem)
        try:
            results, elapsed = _run_one(cfg_path, args.n_samples)

            # Write per-robot YAML
            robot_dir_out = batch_dir / cfg_path.stem
            robot_dir_out.mkdir(parents=True, exist_ok=True)
            results_dict = _results_to_dict(results)
            with open(robot_dir_out / "baselines.yaml", "w") as f:
                yaml.safe_dump(results_dict, f, sort_keys=False, default_flow_style=False)

            summary_rows.append({
                "robot_config": _rel_to_root(cfg_path),
                "robot_name": results.robot_name,
                "status": "ok",
                "elapsed_s": round(elapsed, 2),
                "l_ref_m": round(results.l_ref_m, 6),
                "n_joints_total": results.n_joints_total,
                "n_dof_independent": results.n_dof_independent,
                "mean_range_rad": round(results.mean_range_rad, 4),
                "avg_workspace_vol_norm": round(results.avg_workspace_vol_norm, 6),
                "opposability_index": round(results.opposability_index, 6),
                "combined_yoshikawa_mean": results.combined_yoshikawa_mean,
                "combined_gci": round(results.combined_gci, 6),
            })
            all_results.append(results_dict)

            logger.info(
                "[%s] DOF=%d  L_ref=%.1fmm  workspace_norm=%.4f  GCI=%.4f  (%.1fs)",
                results.robot_name,
                results.n_joints_total,
                results.l_ref_m * 1e3,
                results.avg_workspace_vol_norm,
                results.combined_gci,
                elapsed,
            )

        except Exception as e:
            summary_rows.append({
                "robot_config": _rel_to_root(cfg_path),
                "status": "error",
                "error": repr(e),
            })
            logger.error("Failed for %s: %s", cfg_path.stem, e)
            if args.verbose:
                traceback.print_exc()

    # Write summary YAML
    summary_path = batch_dir / "summary.yaml"
    with open(summary_path, "w") as f:
        yaml.safe_dump(
            {"batch_id": run_id, "n_samples": args.n_samples, "results": summary_rows},
            f,
            sort_keys=False,
        )

    # Write summary CSV
    csv_path = batch_dir / "summary.csv"
    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_results:
                writer.writerow(row)

    ok = sum(1 for r in summary_rows if r.get("status") == "ok")
    err = sum(1 for r in summary_rows if r.get("status") != "ok")
    logger.info("Batch complete: ok=%d error=%d", ok, err)
    logger.info("Summary: %s", summary_path)
    if all_results:
        logger.info("CSV: %s", csv_path)


if __name__ == "__main__":
    main()
