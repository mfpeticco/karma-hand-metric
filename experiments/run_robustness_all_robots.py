#!/usr/bin/env python3
"""Run robustness tests for all 5 representative hands.

This is a driver: for each hand it runs the single-hand robustness engine
`robust_tests/run_tests.py` (which holds the actual invariance/perturbation logic)
as a subprocess, swapping in that hand's config. Regenerates Table III.

Run from the repo root:
    python experiments/run_robustness_all_robots.py <config_file>

Example:
    python experiments/run_robustness_all_robots.py robust_tests/test_config_high.yaml
    python experiments/run_robustness_all_robots.py robust_tests/test_config.yaml
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

# Repo root (this script lives in experiments/). The child run_tests.py is
# launched with cwd=ROOT so robust_tests/ and robots/ paths resolve from there.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ROBOTS = [
    "robots/robot_leap.yaml",
    "robots/robot_shadowhand.yaml",
    "robots/robot_xhand1.yaml",
    "robots/robot_dex3.yaml",
    "robots/robot_inspire.yaml",
]


def main():
    parser = argparse.ArgumentParser(
        description="Run the robust_tests suite for the five representative "
                    "hands, one robot at a time.",
    )
    parser.add_argument(
        "config",
        help="robust_tests config to run for each hand "
             "(e.g. robust_tests/test_config_high.yaml)",
    )
    args = parser.parse_args()

    base_config_path = args.config
    with open(base_config_path) as f:
        base_config = yaml.safe_load(f)

    print(f"{'=' * 60}")
    print(f"  Config: {base_config_path}")
    print(f"  Robots: {len(ROBOTS)}")
    print(f"{'=' * 60}")

    total_t0 = time.perf_counter()

    for robot_cfg in ROBOTS:
        robot_name = Path(robot_cfg).stem.replace('robot_', '')
        print(f"\n{'─' * 60}")
        print(f"  Robot: {robot_name} ({robot_cfg})")
        print(f"{'─' * 60}")

        # Override robot_config in a temp copy
        config = dict(base_config)
        config['robot_config'] = robot_cfg

        with tempfile.NamedTemporaryFile(
            mode='w', suffix=f'_{robot_name}.yaml', delete=False
        ) as tf:
            yaml.dump(config, tf, default_flow_style=False, sort_keys=False)
            tmp_config = tf.name

        t0 = time.perf_counter()
        try:
            ret = subprocess.run(
                [sys.executable, "robust_tests/run_tests.py", "--config", tmp_config],
                cwd=ROOT,
                check=False,
            ).returncode
        finally:
            os.unlink(tmp_config)
        elapsed = time.perf_counter() - t0

        if ret != 0:
            print(f"  WARNING: Tests for {robot_name} exited with code {ret}")

        print(f"  {robot_name} completed in {elapsed:.1f}s")

    total = time.perf_counter() - total_t0
    print(f"\n{'=' * 60}")
    print(f"  All robots complete! Total: {total:.0f}s ({total/60:.1f} min)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
