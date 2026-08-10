"""3D workspace visualizer showing convex hull volumes and intersection.

Run from the repo root:
    python tools/visualize_workspace.py                       # orcahand
    python tools/visualize_workspace.py robot_allegro          # specific robot
    python tools/visualize_workspace.py --all                  # all robots, grid
    python tools/visualize_workspace.py robot_orcahand --n-samples 100000
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.stats.qmc import Sobol

# Repo root (this script lives in tools/); make `baselines` importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baselines.robot_loader import (
    BaselineRobot,
    FingerInfo,
    load_baseline_robot,
    forward_kinematics,
    fingertip_position,
)


def _sample_finger(
    robot: BaselineRobot, finger: FingerInfo, n_samples: int, seed: int = 42,
) -> np.ndarray:
    """Return (N, 3) fingertip positions in mm, sampled independently.

    Mirrors the Sobol sampling in baselines/compute_baselines.py (same
    power-of-two round-up and seeds) so this view matches the opposability
    baseline; the only deltas are mm units and returning raw points to plot.
    """
    q_idx = finger.q_indices
    n_dof = len(q_idx)
    if n_dof == 0:
        return np.zeros((0, 3))

    log2 = max(1, math.ceil(math.log2(max(n_samples, 2))))
    n_actual = 2 ** log2

    sampler = Sobol(d=n_dof, scramble=True, seed=seed)
    unit = sampler.random(n_actual)

    lo = robot.q_lower[q_idx]
    hi = robot.q_upper[q_idx]
    finger_qs = lo + unit * (hi - lo)

    positions = np.empty((n_actual, 3))
    q = np.zeros(robot.model.nq)
    for i in range(n_actual):
        q[:] = 0
        q[q_idx] = finger_qs[i]
        forward_kinematics(robot, q)
        positions[i] = fingertip_position(robot, finger)

    return positions * 1000.0  # metres → mm


def _hull_data(pts_a_mm: np.ndarray, pts_b_mm: np.ndarray):
    """Compute convex hulls, intersection points, and volumes (all in mm)."""
    vol_a = vol_b = vol_inter = 0.0
    inter_pts = np.zeros((0, 3))

    try:
        hull_a = ConvexHull(pts_a_mm)
        vol_a = hull_a.volume
    except Exception:
        hull_a = None

    try:
        hull_b = ConvexHull(pts_b_mm)
        vol_b = hull_b.volume
    except Exception:
        hull_b = None

    if hull_a is not None and hull_b is not None:
        try:
            del_a = Delaunay(pts_a_mm[hull_a.vertices])
            del_b = Delaunay(pts_b_mm[hull_b.vertices])

            a_in_b = del_b.find_simplex(pts_a_mm) >= 0
            b_in_a = del_a.find_simplex(pts_b_mm) >= 0

            parts = []
            if a_in_b.any():
                parts.append(pts_a_mm[a_in_b])
            if b_in_a.any():
                parts.append(pts_b_mm[b_in_a])

            if parts:
                inter_pts = np.vstack(parts)
                if len(inter_pts) >= 4:
                    vol_inter = ConvexHull(inter_pts).volume
        except Exception as e:
            print("  WARNING: intersection volume failed ({}); reporting 0".format(e))

    return hull_a, hull_b, inter_pts, vol_a, vol_b, vol_inter


def visualize_single(
    robot_name: str,
    robot_dir: Path,
    n_samples: int,
    ax=None,
    show: bool = True,
) -> dict:
    """Visualize workspace for one robot. Returns stats dict."""
    import matplotlib.pyplot as plt

    yaml_path = robot_dir / "{}.yaml".format(robot_name)
    if not yaml_path.exists():
        print("Not found: {}".format(yaml_path))
        return {}

    robot = load_baseline_robot(yaml_path)
    n_t = len(robot.thumb.q_indices)
    n_i = len(robot.index.q_indices)
    print("  Sampling {} (thumb={}DOF, index={}DOF) ...".format(robot_name, n_t, n_i))

    thumb_mm = _sample_finger(robot, robot.thumb, n_samples, seed=42)
    index_mm = _sample_finger(robot, robot.index, n_samples, seed=99)

    hull_t, hull_i, inter_pts, vol_t, vol_i, vol_inter = _hull_data(thumb_mm, index_mm)

    union_vol = vol_t + vol_i - vol_inter
    jaccard = vol_inter / union_vol if union_vol > 0 else 0.0

    stats = {
        "n_thumb_samples": len(thumb_mm),
        "n_index_samples": len(index_mm),
        "thumb_vol_mm3": vol_t,
        "index_vol_mm3": vol_i,
        "intersection_vol_mm3": vol_inter,
        "jaccard": jaccard,
        "n_intersection_pts": len(inter_pts),
    }

    if ax is None:
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection="3d")
        standalone = True
    else:
        standalone = False

    rng = np.random.default_rng(0)
    max_plot = 6000

    def subsample(pts, max_n):
        if len(pts) <= max_n:
            return pts
        return pts[rng.choice(len(pts), max_n, replace=False)]

    t_plot = subsample(thumb_mm, max_plot)
    i_plot = subsample(index_mm, max_plot)

    ax.scatter(t_plot[:, 0], t_plot[:, 1], t_plot[:, 2],
               c="tab:red", alpha=0.06, s=1, label="Thumb ({:.0f}k mm\u00b3)".format(vol_t / 1000))
    ax.scatter(i_plot[:, 0], i_plot[:, 1], i_plot[:, 2],
               c="tab:blue", alpha=0.06, s=1, label="Index ({:.0f}k mm\u00b3)".format(vol_i / 1000))

    if len(inter_pts) > 0:
        ip = subsample(inter_pts, max_plot)
        ax.scatter(ip[:, 0], ip[:, 1], ip[:, 2],
                   c="lime", alpha=0.5, s=6,
                   label="Intersection ({:.0f} mm\u00b3)".format(vol_inter))

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    title = "{}\ninter={:.0f} mm\u00b3  jaccard={:.3f}".format(
        robot.robot_name, vol_inter, jaccard)
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper left", fontsize=8, markerscale=5)

    if standalone and show:
        plt.tight_layout()
        plt.show()

    return stats


def visualize_all(robot_dir: Path, n_samples: int) -> None:
    """Grid of all robots found in robot_dir."""
    import matplotlib.pyplot as plt

    yamls = sorted(robot_dir.glob("robot_*.yaml"))
    n = len(yamls)
    cols = 4
    rows = math.ceil(n / cols)

    fig = plt.figure(figsize=(5 * cols, 4.5 * rows))
    for idx, yaml_path in enumerate(yamls):
        ax = fig.add_subplot(rows, cols, idx + 1, projection="3d")
        name = yaml_path.stem
        try:
            visualize_single(name, robot_dir, n_samples, ax=ax, show=False)
        except Exception as e:
            ax.set_title("{}\nERROR: {}".format(name, e), fontsize=8, color="red")
            print("  ERROR {}: {}".format(name, e))

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Workspace overlap visualizer (convex hull)")
    parser.add_argument("robot", nargs="?", default="robot_orcahand",
                        help="Robot stem name (default: robot_orcahand)")
    parser.add_argument("--all", action="store_true", help="Show all robots in a grid")
    parser.add_argument("--robot-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent / "robots",
                        help="Directory with robot_*.yaml files")
    parser.add_argument("--n-samples", type=int, default=50000)
    args = parser.parse_args()

    if args.all:
        visualize_all(args.robot_dir, args.n_samples)
    else:
        stats = visualize_single(args.robot, args.robot_dir, args.n_samples)
        if stats:
            print("\nStats for {}:".format(args.robot))
            for k, v in stats.items():
                print("  {}: {}".format(k, v))


if __name__ == "__main__":
    main()
