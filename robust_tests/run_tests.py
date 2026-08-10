#!/usr/bin/env python3
"""Robustness test suite for KaRMA — the single-hand engine.

Runs the invariance/perturbation checks for the one hand named in the config. To
run all five representative hands (Table III), use the driver
`experiments/run_robustness_all_robots.py`, which calls this script per hand.

Run from project root:
    python robust_tests/run_tests.py --config robust_tests/test_config.yaml

Configure via robust_tests/test_config.yaml (which hand, which tests, budget).
All results are written to robust_tests/results/ (or whatever results_dir is set to).
"""
from __future__ import annotations

import argparse
import logging
import math
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import yaml

# ── Project root setup ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from robust_tests.test_utils import (
    add_urdf_wrapper_joint,
    get_active_joint_names,
    perturb_urdf_joint_limits,
    perturb_urdf_joint_origins,
    prepare_metric_config,
    prepare_robot_config,
    scale_metric_config_lengths,
    scale_robot_config_lengths,
    scale_urdf,
    swap_fingers_robot_config,
)

from karma.config import load_config
from karma.robot import load_robot
from karma.metric import compute_metric
from karma.lref import compute_lref_breakdown

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("robust_tests")


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def resolve(p: str | Path) -> Path:
    """Resolve a path relative to project root."""
    p = Path(p)
    if not p.is_absolute():
        p = ROOT / p
    return p


def rel_to_root(p: str | Path) -> str:
    """Path relative to the repo root, for portable meta headers (no home paths)."""
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return Path(p).name


def extract_scores(result) -> dict:
    """Pull the key scalar scores from a MetricResult."""
    return {
        'n_voxels': result.n_voxels_reached,
        'n_states': result.n_states_reached,
        'translational_score': float(result.translational_score),
        'global_rotational_score': float(result.global_rotational_score),
    }


def compare_scores(baseline: dict, test: dict) -> dict:
    """Compute absolute and percentage deltas between two score dicts."""
    comp = {}
    for key in ('n_voxels', 'n_states', 'translational_score', 'global_rotational_score'):
        b = baseline[key]
        t = test[key]
        comp[key] = t
        comp[f'delta_{key}'] = t - b
        if isinstance(b, (int, float)) and b != 0:
            comp[f'pct_{key}'] = round(((t - b) / abs(b)) * 100, 4)
        else:
            comp[f'pct_{key}'] = None
    comp['exact_match'] = (
        baseline['n_voxels'] == test['n_voxels']
        and baseline['n_states'] == test['n_states']
        and abs(baseline['translational_score'] - test['translational_score']) < 1e-10
        and abs(baseline['global_rotational_score'] - test['global_rotational_score']) < 1e-10
    )
    # Practical match: ≤1 voxel difference, small rotational noise from parallel workers
    comp['practical_match'] = (
        abs(baseline['n_voxels'] - test['n_voxels']) <= 1
        and abs(baseline['translational_score'] - test['translational_score']) < 0.001
        and abs(baseline['global_rotational_score'] - test['global_rotational_score']) < 0.01
    )
    return comp


def run_metric(metric_cfg_path: str, robot_cfg_path: str, max_states: int | None = None):
    """Load config + robot, run two-phase metric, return (scores_dict, elapsed_s)."""
    cfg = load_config(metric_cfg_path, robot_cfg_path)
    ctx = load_robot(cfg.urdf_path, cfg)
    mv = max_states or cfg.max_states

    t0 = time.perf_counter()
    result = compute_metric(
        cfg, ctx,
        max_voxels=mv,
        n_rotation_workers=cfg.rotation_workers,
    )
    elapsed = time.perf_counter() - t0
    return extract_scores(result), elapsed


class TempEnv:
    """Context manager for a temporary directory with metric + robot configs."""

    def __init__(self, original_metric: Path, original_robot: Path,
                 metric_overrides: dict | None = None):
        self.original_metric = original_metric
        self.original_robot = original_robot
        self.metric_overrides = metric_overrides or {}
        self.tmpdir = None

    def __enter__(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix='robtest_'))

        # Write baseline metric config with overrides
        self.metric_path = self.tmpdir / 'metric_config.yaml'
        prepare_metric_config(self.original_metric, self.metric_path, self.metric_overrides)

        # Write baseline robot config with the URDF path resolved to absolute,
        # so the temp-dir copy doesn't break the config-relative path.
        self.robot_path = self.tmpdir / 'robot_config.yaml'
        cfg_raw = yaml.safe_load(open(str(self.original_robot)))
        urdf_rel = Path(cfg_raw['urdf_path'])
        if not urdf_rel.is_absolute():
            urdf_abs = (self.original_robot.parent / urdf_rel).resolve()
        else:
            urdf_abs = urdf_rel
        prepare_robot_config(self.original_robot, self.robot_path, urdf_path_override=urdf_abs)

        return self

    def __exit__(self, *exc):
        if self.tmpdir and self.tmpdir.exists():
            shutil.rmtree(self.tmpdir, ignore_errors=True)

    @property
    def urdf_path(self) -> Path:
        """Resolved absolute URDF path from the robot config."""
        with open(str(self.robot_path)) as f:
            cfg_raw = yaml.safe_load(f)
        return Path(cfg_raw['urdf_path'])


# ═══════════════════════════════════════════════════════════════════════════════
# L_ref Computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_lref(robot_cfg_path: Path) -> dict:
    """L_ref decomposition for one robot config (characterization survey 4.1).

    Single-sourced from :func:`karma.lref.compute_lref_breakdown` — the same
    L_ref the metric itself uses, so the survey can never drift from it:
      mxPr = max pairwise knuckle distance (palm diameter)
      mF   = median finger wire length

    Returns dict with L_ref_m, mxPr_m, mF_m, n_fingers, and finger_wires_m.
    """
    b = compute_lref_breakdown(robot_cfg_path)
    return {
        'L_ref_m': round(b['L_ref_m'], 6),
        'mxPr_m': round(b['mxPr_m'], 6),
        'mF_m': round(b['mF_m'], 6),
        'n_fingers': b['n_fingers'],
        'finger_wires_m': [round(w, 6) for w in b['finger_wires_m']],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Test Group 1: Exact Invariance
# ═══════════════════════════════════════════════════════════════════════════════

def test_translation(baseline: dict, cfg: dict, env_defaults: dict) -> list[dict]:
    """1.1 Translation invariance: shift robot in world frame."""
    results = []
    for t in cfg.get('translations_m', []):
        label = f"translate [{t[0]}, {t[1]}, {t[2]}]"
        logger.info("Test 1.1: %s", label)

        with TempEnv(**env_defaults) as env:
            # Create translated URDF
            mod_urdf = env.tmpdir / 'translated.urdf'
            add_urdf_wrapper_joint(env.urdf_path, mod_urdf, xyz=t, rpy=[0, 0, 0])

            # Update robot config to point at modified URDF
            prepare_robot_config(env.robot_path, env.robot_path, urdf_path_override=mod_urdf)

            scores, elapsed = run_metric(str(env.metric_path), str(env.robot_path),
                                         env_defaults['metric_overrides'].get('max_states'))
            comp = compare_scores(baseline, scores)
            comp['label'] = label
            comp['elapsed_s'] = round(elapsed, 2)
            results.append(comp)
            logger.info("  exact_match=%s  voxels: %d→%d  rot: %.4f→%.4f",
                        comp['exact_match'], baseline['n_voxels'], scores['n_voxels'],
                        baseline['global_rotational_score'], scores['global_rotational_score'])
    return results


def test_rotation(baseline: dict, cfg: dict, env_defaults: dict) -> list[dict]:
    """1.2 Rotation invariance: rotate robot in world frame."""
    results = []
    for rpy_deg in cfg.get('rotations_rpy_deg', []):
        rpy_rad = [math.radians(d) for d in rpy_deg]
        label = f"rotate rpy_deg=[{rpy_deg[0]}, {rpy_deg[1]}, {rpy_deg[2]}]"
        logger.info("Test 1.2: %s", label)

        with TempEnv(**env_defaults) as env:
            mod_urdf = env.tmpdir / 'rotated.urdf'
            add_urdf_wrapper_joint(env.urdf_path, mod_urdf, xyz=[0, 0, 0], rpy=rpy_rad)
            prepare_robot_config(env.robot_path, env.robot_path, urdf_path_override=mod_urdf)

            scores, elapsed = run_metric(str(env.metric_path), str(env.robot_path),
                                         env_defaults['metric_overrides'].get('max_states'))
            comp = compare_scores(baseline, scores)
            comp['label'] = label
            comp['elapsed_s'] = round(elapsed, 2)
            results.append(comp)
            logger.info("  exact_match=%s  voxels: %d→%d  rot: %.4f→%.4f  [EXPECTED TO FAIL]",
                        comp['exact_match'], baseline['n_voxels'], scores['n_voxels'],
                        baseline['global_rotational_score'], scores['global_rotational_score'])
    return results


def test_scale(baseline: dict, cfg: dict, env_defaults: dict) -> list[dict]:
    """1.3 Scale invariance: uniformly scale robot + all length parameters."""
    results = []
    for k in cfg.get('scale_factors', []):
        label = f"scale k={k}"
        logger.info("Test 1.3: %s", label)

        with TempEnv(**env_defaults) as env:
            # Scale URDF
            mod_urdf = env.tmpdir / 'scaled.urdf'
            scale_urdf(env.urdf_path, k, mod_urdf)

            # Scale robot config (tip offsets) + point at scaled URDF
            scaled_robot = env.tmpdir / 'robot_scaled.yaml'
            scale_robot_config_lengths(env.robot_path, k, scaled_robot, urdf_path_override=mod_urdf)

            # Scale metric config (sphere_radius, voxel_size, etc.)
            scaled_metric = env.tmpdir / 'metric_scaled.yaml'
            scale_metric_config_lengths(
                env.metric_path, k, scaled_metric,
                extra_overrides={'max_states': env_defaults['metric_overrides'].get('max_states')},
            )

            scores, elapsed = run_metric(str(scaled_metric), str(scaled_robot),
                                         env_defaults['metric_overrides'].get('max_states'))
            comp = compare_scores(baseline, scores)
            comp['label'] = label
            comp['elapsed_s'] = round(elapsed, 2)
            results.append(comp)
            logger.info("  exact_match=%s  voxels: %d→%d  rot: %.4f→%.4f",
                        comp['exact_match'], baseline['n_voxels'], scores['n_voxels'],
                        baseline['global_rotational_score'], scores['global_rotational_score'])
    return results


def test_finger_swap(baseline: dict, cfg: dict, env_defaults: dict) -> dict | None:
    """1.4 Finger swap invariance: swap thumb/index labels."""
    if not cfg.get('test_finger_swap', False):
        return None
    logger.info("Test 1.4: finger swap (thumb <-> index)")

    with TempEnv(**env_defaults) as env:
        swapped = env.tmpdir / 'robot_swapped.yaml'
        swap_fingers_robot_config(env.robot_path, swapped)

        scores, elapsed = run_metric(str(env.metric_path), str(swapped),
                                     env_defaults['metric_overrides'].get('max_states'))
        comp = compare_scores(baseline, scores)
        comp['label'] = 'finger_swap'
        comp['elapsed_s'] = round(elapsed, 2)
        logger.info("  exact_match=%s  voxels: %d→%d  rot: %.4f→%.4f",
                    comp['exact_match'], baseline['n_voxels'], scores['n_voxels'],
                    baseline['global_rotational_score'], scores['global_rotational_score'])
        return comp


# ═══════════════════════════════════════════════════════════════════════════════
# Test Group 2: Stability
# ═══════════════════════════════════════════════════════════════════════════════

def test_link_length_stability(baseline: dict, cfg: dict, env_defaults: dict) -> list[dict]:
    """2.1 Perturb all joint origin translations by (1+eps)."""
    results = []
    for eps in cfg.get('link_length_epsilons', []):
        label = f"link_length eps={eps:+.3f}"
        logger.info("Test 2.1: %s", label)

        with TempEnv(**env_defaults) as env:
            mod_urdf = env.tmpdir / 'perturbed_lengths.urdf'
            perturb_urdf_joint_origins(env.urdf_path, eps, mod_urdf)
            prepare_robot_config(env.robot_path, env.robot_path, urdf_path_override=mod_urdf)

            scores, elapsed = run_metric(str(env.metric_path), str(env.robot_path),
                                         env_defaults['metric_overrides'].get('max_states'))
            comp = compare_scores(baseline, scores)
            comp['label'] = label
            comp['epsilon'] = eps
            comp['elapsed_s'] = round(elapsed, 2)
            results.append(comp)
            logger.info("  voxels: %d→%d (%.1f%%)  rot: %.4f→%.4f",
                        baseline['n_voxels'], scores['n_voxels'],
                        comp.get('pct_n_voxels', 0) or 0,
                        baseline['global_rotational_score'], scores['global_rotational_score'])
    return results


def test_joint_limit_stability(baseline: dict, cfg: dict, env_defaults: dict,
                               robot_cfg_path: Path) -> list[dict]:
    """2.2 Widen/narrow joint limits by delta degrees."""
    results = []
    joint_names = get_active_joint_names(robot_cfg_path)

    for delta_deg in cfg.get('joint_limit_deltas_deg', []):
        delta_rad = math.radians(delta_deg)
        label = f"joint_limits delta={delta_deg:+.1f}deg"
        logger.info("Test 2.2: %s", label)

        with TempEnv(**env_defaults) as env:
            mod_urdf = env.tmpdir / 'perturbed_limits.urdf'
            perturb_urdf_joint_limits(env.urdf_path, joint_names, delta_rad, mod_urdf)
            prepare_robot_config(env.robot_path, env.robot_path, urdf_path_override=mod_urdf)

            scores, elapsed = run_metric(str(env.metric_path), str(env.robot_path),
                                         env_defaults['metric_overrides'].get('max_states'))
            comp = compare_scores(baseline, scores)
            comp['label'] = label
            comp['delta_deg'] = delta_deg
            comp['elapsed_s'] = round(elapsed, 2)
            results.append(comp)
            logger.info("  voxels: %d→%d (%.1f%%)  rot: %.4f→%.4f",
                        baseline['n_voxels'], scores['n_voxels'],
                        comp.get('pct_n_voxels', 0) or 0,
                        baseline['global_rotational_score'], scores['global_rotational_score'])
    return results


def test_sphere_radius_stability(baseline: dict, cfg: dict, env_defaults: dict) -> list[dict]:
    """2.3 Perturb sphere_radius_m by (1+eps)."""
    results = []
    # Load original sphere_radius
    with open(str(env_defaults['original_metric'])) as f:
        metric_data = yaml.safe_load(f)
    base_radius = metric_data['sphere_radius_m']

    for eps in cfg.get('sphere_radius_epsilons', []):
        new_radius = base_radius * (1.0 + eps)
        label = f"sphere_radius eps={eps:+.3f} ({base_radius:.4f}→{new_radius:.4f})"
        logger.info("Test 2.3: %s", label)

        with TempEnv(**env_defaults) as env:
            # Override sphere_radius in metric config
            prepare_metric_config(env.metric_path, env.metric_path,
                                  overrides={'sphere_radius_m': new_radius})

            scores, elapsed = run_metric(str(env.metric_path), str(env.robot_path),
                                         env_defaults['metric_overrides'].get('max_states'))
            comp = compare_scores(baseline, scores)
            comp['label'] = label
            comp['epsilon'] = eps
            comp['elapsed_s'] = round(elapsed, 2)
            results.append(comp)
            logger.info("  voxels: %d→%d (%.1f%%)  rot: %.4f→%.4f",
                        baseline['n_voxels'], scores['n_voxels'],
                        comp.get('pct_n_voxels', 0) or 0,
                        baseline['global_rotational_score'], scores['global_rotational_score'])
    return results


def test_link_radius_stability(baseline: dict, cfg: dict, env_defaults: dict) -> list[dict]:
    """2.5 Perturb link_radius_m by (1+eps)."""
    results = []
    with open(str(env_defaults['original_metric'])) as f:
        metric_data = yaml.safe_load(f)
    base_radius = metric_data['link_radius_m']

    for eps in cfg.get('link_radius_epsilons', []):
        new_radius = base_radius * (1.0 + eps)
        label = f"link_radius eps={eps:+.3f} ({base_radius:.4f}→{new_radius:.4f})"
        logger.info("Test 2.5: %s", label)

        with TempEnv(**env_defaults) as env:
            prepare_metric_config(env.metric_path, env.metric_path,
                                  overrides={'link_radius_m': new_radius})

            scores, elapsed = run_metric(str(env.metric_path), str(env.robot_path),
                                         env_defaults['metric_overrides'].get('max_states'))
            comp = compare_scores(baseline, scores)
            comp['label'] = label
            comp['epsilon'] = eps
            comp['elapsed_s'] = round(elapsed, 2)
            results.append(comp)
            logger.info("  voxels: %d→%d (%.1f%%)  rot: %.4f→%.4f",
                        baseline['n_voxels'], scores['n_voxels'],
                        comp.get('pct_n_voxels', 0) or 0,
                        baseline['global_rotational_score'], scores['global_rotational_score'])
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Test Group 3: Numerical
# ═══════════════════════════════════════════════════════════════════════════════

def test_determinism(cfg: dict, env_defaults: dict) -> dict:
    """3.1 Run metric N times and check bit-for-bit reproducibility."""
    n = cfg.get('determinism_repeats', 3)
    logger.info("Test 3.1: determinism (%d runs)", n)

    runs = []
    for i in range(n):
        with TempEnv(**env_defaults) as env:
            scores, elapsed = run_metric(str(env.metric_path), str(env.robot_path),
                                         env_defaults['metric_overrides'].get('max_states'))
            scores['elapsed_s'] = round(elapsed, 2)
            runs.append(scores)
            logger.info("  Run %d: voxels=%d states=%d rot=%.4f (%.1fs)",
                        i + 1, scores['n_voxels'], scores['n_states'],
                        scores['global_rotational_score'], elapsed)

    # Check all runs match the first
    all_match = all(
        r['n_voxels'] == runs[0]['n_voxels']
        and r['n_states'] == runs[0]['n_states']
        and abs(r['translational_score'] - runs[0]['translational_score']) < 1e-10
        and abs(r['global_rotational_score'] - runs[0]['global_rotational_score']) < 1e-10
        for r in runs[1:]
    )

    result = {
        'n_runs': n,
        'all_identical': all_match,
        'runs': runs,
    }
    logger.info("  Determinism: %s", "PASS (all identical)" if all_match else "FAIL (runs differ)")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Test Group 4: Characterization Surveys
# ═══════════════════════════════════════════════════════════════════════════════

def survey_lref(cfg: dict) -> list[dict]:
    """4.1 Compute L_ref for multiple robots."""
    results = []
    for robot_path in cfg.get('lref_robots', []):
        robot_path = resolve(robot_path)
        if not robot_path.exists():
            logger.warning("  Skipping %s (file not found)", robot_path)
            continue
        logger.info("Survey 4.1: L_ref for %s", robot_path.name)
        try:
            info = compute_lref(robot_path)
            info['robot'] = robot_path.name
            results.append(info)
            logger.info("  L_ref = %.4f m  (mxPr=%.4f, mF=%.4f, #F=%d)",
                        info['L_ref_m'], info['mxPr_m'], info['mF_m'],
                        info['n_fingers'])
        except Exception as e:
            logger.error("  Failed: %s", e)
            results.append({'robot': robot_path.name, 'error': str(e)})
    return results


def survey_step_size(cfg: dict, env_defaults: dict) -> list[dict]:
    """4.2 Run metric with different step_size fractions of voxel_size."""
    results = []
    with open(str(env_defaults['original_metric'])) as f:
        metric_data = yaml.safe_load(f)
    voxel_size = metric_data['voxel_size_m']

    for frac in cfg.get('step_size_fractions', []):
        step = voxel_size * frac
        label = f"step_size={frac}*voxel ({step:.6f}m)"
        logger.info("Survey 4.2: %s", label)

        with TempEnv(**env_defaults) as env:
            prepare_metric_config(env.metric_path, env.metric_path,
                                  overrides={'step_size_m': step})

            scores, elapsed = run_metric(str(env.metric_path), str(env.robot_path),
                                         env_defaults['metric_overrides'].get('max_states'))
            scores['label'] = label
            scores['step_size_m'] = step
            scores['step_fraction'] = frac
            scores['elapsed_s'] = round(elapsed, 2)
            results.append(scores)
            logger.info("  voxels=%d rot=%.4f (%.1fs)",
                        scores['n_voxels'], scores['global_rotational_score'], elapsed)
    return results


def survey_bfs_budget(cfg: dict, env_defaults: dict) -> list[dict]:
    """4.3 Run metric with different max_states values."""
    results = []
    for ms in cfg.get('max_states_values', []):
        label = f"max_states={ms}"
        logger.info("Survey 4.3: %s", label)

        with TempEnv(**env_defaults) as env:
            prepare_metric_config(env.metric_path, env.metric_path,
                                  overrides={'max_states': ms})

            scores, elapsed = run_metric(str(env.metric_path), str(env.robot_path), max_states=ms)
            scores['label'] = label
            scores['max_states'] = ms
            scores['elapsed_s'] = round(elapsed, 2)
            results.append(scores)
            logger.info("  voxels=%d rot=%.4f (%.1fs)",
                        scores['n_voxels'], scores['global_rotational_score'], elapsed)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # Load test configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None, help='Path to test config YAML')
    args = parser.parse_args()

    if args.config:
        test_cfg_path = Path(args.config)
    else:
        test_cfg_path = Path(__file__).parent / 'test_config.yaml'
    with open(test_cfg_path) as f:
        test_cfg = yaml.safe_load(f)

    metric_cfg_path = resolve(test_cfg['metric_config'])
    robot_cfg_path = resolve(test_cfg['robot_config'])
    results_dir = resolve(test_cfg.get('results_dir', 'robust_tests/results'))
    results_dir.mkdir(parents=True, exist_ok=True)

    max_states = test_cfg.get('max_states_override')

    # Build env_defaults dict shared by all tests
    metric_overrides = {}
    if max_states is not None:
        metric_overrides['max_states'] = max_states
    env_defaults = {
        'original_metric': metric_cfg_path,
        'original_robot': robot_cfg_path,
        'metric_overrides': metric_overrides,
    }

    all_results = {
        'meta': {
            'timestamp': datetime.now().isoformat(),
            'robot_config': rel_to_root(robot_cfg_path),
            'metric_config': rel_to_root(metric_cfg_path),
            'max_states_override': max_states,
        },
    }

    total_t0 = time.perf_counter()

    # ── Baseline ──────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Running BASELINE metric...")
    logger.info("=" * 60)
    with TempEnv(**env_defaults) as env:
        baseline, elapsed = run_metric(str(env.metric_path), str(env.robot_path), max_states)
    baseline['elapsed_s'] = round(elapsed, 2)
    all_results['baseline'] = baseline
    logger.info("Baseline: voxels=%d states=%d KaRMA-T=%.6f KaRMA-R=%.4f (%.1fs)",
                baseline['n_voxels'], baseline['n_states'],
                baseline['translational_score'], baseline['global_rotational_score'], elapsed)

    # ── Group 1: Invariance ───────────────────────────────────────────────────
    inv_cfg = test_cfg.get('invariance', {})
    if inv_cfg:
        logger.info("=" * 60)
        logger.info("Group 1: Exact Invariance Tests")
        logger.info("=" * 60)

        inv_results = {}
        inv_results['translation'] = test_translation(baseline, inv_cfg, env_defaults)
        inv_results['rotation'] = test_rotation(baseline, inv_cfg, env_defaults)
        inv_results['scale'] = test_scale(baseline, inv_cfg, env_defaults)
        inv_results['finger_swap'] = test_finger_swap(baseline, inv_cfg, env_defaults)
        all_results['invariance'] = inv_results

    # ── Group 2: Stability ────────────────────────────────────────────────────
    stab_cfg = test_cfg.get('stability', {})
    if stab_cfg:
        logger.info("=" * 60)
        logger.info("Group 2: Stability Tests")
        logger.info("=" * 60)

        stab_results = {}
        stab_results['link_length'] = test_link_length_stability(baseline, stab_cfg, env_defaults)
        stab_results['joint_limits'] = test_joint_limit_stability(
            baseline, stab_cfg, env_defaults, robot_cfg_path)
        stab_results['sphere_radius'] = test_sphere_radius_stability(baseline, stab_cfg, env_defaults)
        stab_results['link_radius'] = test_link_radius_stability(baseline, stab_cfg, env_defaults)

        # 2.4 Seed determinism — reuses the determinism test
        n_seed = stab_cfg.get('seed_determinism_repeats', 0)
        if n_seed > 1:
            stab_results['seed_determinism'] = test_determinism(
                {'determinism_repeats': n_seed}, env_defaults)

        all_results['stability'] = stab_results

    # ── Group 3: Numerical ────────────────────────────────────────────────────
    num_cfg = test_cfg.get('numerical', {})
    if num_cfg:
        logger.info("=" * 60)
        logger.info("Group 3: Numerical Tests")
        logger.info("=" * 60)

        num_results = {}
        num_results['determinism'] = test_determinism(num_cfg, env_defaults)
        all_results['numerical'] = num_results

    # ── Group 4: Characterization ─────────────────────────────────────────────
    char_cfg = test_cfg.get('characterization', {})
    if char_cfg:
        logger.info("=" * 60)
        logger.info("Group 4: Characterization Surveys")
        logger.info("=" * 60)

        char_results = {}
        char_results['lref'] = survey_lref(char_cfg)
        char_results['step_size'] = survey_step_size(char_cfg, env_defaults)
        char_results['bfs_budget'] = survey_bfs_budget(char_cfg, env_defaults)
        all_results['characterization'] = char_results

    # ── Summary ───────────────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - total_t0
    all_results['meta']['total_elapsed_s'] = round(total_elapsed, 1)

    # Write results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    robot_name = Path(test_cfg['robot_config']).stem.replace('robot_', '')
    out_file = results_dir / f'results_{robot_name}_{timestamp}.yaml'
    with open(out_file, 'w') as f:
        yaml.dump(all_results, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    # Also write a "latest" symlink/copy
    latest = results_dir / f'results_{robot_name}_latest.yaml'
    shutil.copy2(out_file, latest)

    # ── Print Summary ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  Robustness Test Suite — Summary")
    print(f"{'=' * 60}")
    print(f"  Robot:     {test_cfg['robot_config']}")
    print(f"  Baseline:  {baseline['n_voxels']} voxels, "
          f"rot={baseline['global_rotational_score']:.4f}")
    print(f"  Total time: {total_elapsed:.1f}s")
    print()

    # Invariance summary
    if 'invariance' in all_results:
        inv = all_results['invariance']
        for group_name, group_data in inv.items():
            if group_data is None:
                continue
            items = group_data if isinstance(group_data, list) else [group_data]
            exact = sum(1 for r in items if r.get('exact_match', False))
            practical = sum(1 for r in items if r.get('practical_match', False))
            total = len(items)
            if exact == total:
                status = "PASS (exact)"
            elif practical == total:
                status = f"PASS (practical {practical}/{total}, exact {exact}/{total})"
            else:
                status = f"FAIL (practical {practical}/{total}, exact {exact}/{total})"
            print(f"  1.x {group_name:20s}  {status}")

    # Stability summary
    if 'stability' in all_results:
        stab = all_results['stability']
        for group_name, group_data in stab.items():
            if isinstance(group_data, list):
                max_pct = max(abs(r.get('pct_n_voxels', 0) or 0) for r in group_data) if group_data else 0
                print(f"  2.x {group_name:20s}  max voxel change: {max_pct:.1f}%")
            elif isinstance(group_data, dict) and 'all_identical' in group_data:
                status = "PASS" if group_data['all_identical'] else "FAIL"
                print(f"  2.x {group_name:20s}  {status}")

    # Numerical summary
    if 'numerical' in all_results:
        det = all_results['numerical'].get('determinism', {})
        if det:
            status = "PASS" if det.get('all_identical') else "FAIL"
            print(f"  3.1 {'determinism':20s}  {status}")

    print(f"\n  Results: {out_file}")
    print(f"{'=' * 60}\n")


if __name__ == '__main__':
    main()
