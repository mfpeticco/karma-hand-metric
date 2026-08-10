#!/usr/bin/env python3
"""Run the full metric on all 16 bundled hands and write a summary table.

The output rows mirror the committed results/16_hand_batch/summary.yaml (minus
its machine header and the two slip columns, which came from the paper's batch
harness).
"""
from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import yaml

from karma.config import load_config
from karma.robot import load_robot
from karma.metric import compute_metric

METRIC_CFG = "karma_config.yaml"
ROBOTS_DIR = Path("robots")


def run_one(robot_cfg_path: str) -> dict:
    cfg = load_config(METRIC_CFG, robot_cfg_path)
    ctx = load_robot(cfg.urdf_path, cfg)
    t0 = time.perf_counter()
    result = compute_metric(cfg, ctx)
    elapsed = time.perf_counter() - t0
    sens = result.seed_sensitivity or {}
    return {
        "robot": Path(robot_cfg_path).stem.replace("robot_", ""),
        "L_ref_mm": round(result.l_ref_m * 1000, 1),
        "n_voxels": result.n_voxels_reached,
        "KaRMA_T": round(result.translational_score, 6),
        "KaRMA_R": round(result.global_rotational_score, 6),
        "KaRMA_S": sens.get("karma_s"),
        "best_voxels": sens.get("best", result.n_voxels_reached),
        "median_voxels": sens.get("median"),
        "n_seeds": sens.get("n_seeds_evaluated"),
        "runtime_s": round(elapsed, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score all bundled hands with the shipped config "
                    "(roughly 18 minutes on a 32-thread desktop).",
    )
    parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    robot_files = sorted(
        p for p in ROBOTS_DIR.glob("robot_*.yaml") if "template" not in p.name
    )
    print(f"{'=' * 74}")
    print(f"  Running KaRMA on {len(robot_files)} hands")
    print(f"{'=' * 74}")

    rows = []
    total_t0 = time.perf_counter()

    for i, rf in enumerate(robot_files, 1):
        print(f"\n[{i}/{len(robot_files)}] {rf.stem.replace('robot_', '')}")
        try:
            r = run_one(str(rf))
            rows.append(r)
            s_str = f"{r['KaRMA_S']:.4f}" if r["KaRMA_S"] is not None else "-"
            print(f"  voxels={r['n_voxels']}  KaRMA-T={r['KaRMA_T']:.6f}  "
                  f"KaRMA-R={r['KaRMA_R']:.6f}  KaRMA-S={s_str}  ({r['runtime_s']:.0f}s)")
        except Exception as e:
            print(f"  ERROR: {e}")
            rows.append({"robot": rf.stem.replace("robot_", ""), "error": str(e)})

    total = time.perf_counter() - total_t0

    # Summary table
    print(f"\n{'=' * 74}")
    print(f"  RESULTS — {len(rows)} hands, {total:.0f}s total")
    print(f"{'=' * 74}")
    print(f"  {'Hand':<22s} {'Voxels':>7s} {'KaRMA-T':>10s} {'KaRMA-R':>10s} {'KaRMA-S':>8s} {'Time':>7s}")
    print(f"  {'-'*22} {'-'*7} {'-'*10} {'-'*10} {'-'*8} {'-'*7}")
    for r in rows:
        if "error" in r:
            print(f"  {r['robot']:<22s} {'ERROR':>7s}")
        else:
            s_str = f"{r['KaRMA_S']:.4f}" if r["KaRMA_S"] is not None else "-"
            print(f"  {r['robot']:<22s} {r['n_voxels']:>7d} {r['KaRMA_T']:>10.6f} "
                  f"{r['KaRMA_R']:>10.6f} {s_str:>8s} {r['runtime_s']:>6.0f}s")

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"all_hands_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
    with open(out_file, "w") as f:
        yaml.dump({"timestamp": datetime.now().isoformat(), "rows": rows},
                  f, default_flow_style=False, sort_keys=False)
    print(f"\n  Saved: {out_file}")


if __name__ == "__main__":
    main()
