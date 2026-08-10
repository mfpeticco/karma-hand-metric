#!/usr/bin/env python3
"""CLI entry point: compute the fine-manipulation metric and write results."""
from __future__ import annotations

import argparse
import logging
import time

from karma.config import load_config
from karma.metric import compute_metric
from karma.robot import load_robot
from karma.results import write_outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute the KaRMA fine-manipulation metric for one robot hand."
    )
    parser.add_argument(
        "--config", default="robots/robot_shadowhand.yaml",
        help="Robot config YAML, e.g. robots/robot_leap.yaml",
    )
    parser.add_argument(
        "--metric-config",
        default="karma_config.yaml",
        help="Global KaRMA config (shared across robots)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--rotation-workers", type=int, default=None,
        help="Number of parallel workers for rotation exploration (overrides config)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    cfg = load_config(args.metric_config, args.config)
    ctx = load_robot(cfg.urdf_path, cfg)

    rotation_workers = args.rotation_workers if args.rotation_workers is not None else cfg.rotation_workers

    start_time = time.perf_counter()

    logger.info("Computing metric: translation BFS + parallel rotation (%d workers)", rotation_workers)
    result = compute_metric(
        cfg, ctx,
        n_rotation_workers=rotation_workers,
    )

    elapsed = time.perf_counter() - start_time
    out_path = write_outputs(cfg, result)

    karma_s = (result.seed_sensitivity or {}).get("karma_s")

    print(f"\n{'='*54}")
    print("  KaRMA metric")
    print(f"{'='*54}")
    print(f"  Voxels reached:              {result.n_voxels_reached}")
    print(f"  States reached:              {result.n_states_reached}")
    print(f"  KaRMA-T (translational):     {result.translational_score:.6f}")
    print(f"  KaRMA-R (rotational):        {result.global_rotational_score:.4f}")
    if karma_s is not None:
        print(f"  KaRMA-S (seed sensitivity):  {karma_s:.4f}")
    print(f"  Translational volume:        {result.translational_volume_m3:.6e} m^3")
    print(f"  L_ref:                       {result.l_ref_m:.6f} m")
    print(f"  Voxel size (scaled):         {cfg.voxel_size_m*1e3:.3f} mm")
    print(f"  Elapsed time:                {elapsed:.2f}s")
    print(f"  Output written to:           {out_path}")
    print(f"{'='*54}\n")


if __name__ == "__main__":
    main()
