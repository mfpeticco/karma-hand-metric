# KaRMA: Kinematic Rolling Manipulation Ability

KaRMA is a metric for evaluating a robotic hand's ability to perform fine manipulation based solely on its kinematics. 

The basic idea behind KaRMA is to have a robotic hand hold a two-finger thumb–index precision pinch on a small sphere, and see how far the hand can **translate and reorient that sphere by rolling it from an initial grasp, without letting go**. There is no regrasping, no gaiting, no motor controller, no policy, etc. The purpose is to evaluate just the kinematics of the hand.

It reports three scores:

| Score | Meaning |
| --- | --- |
| **KaRMA-T** | Translational ability: reachable sphere-centre volume, normalized by hand size |
| **KaRMA-R** | Rotational ability: fraction of 228 twist-invariant orientation bins reached |
| **KaRMA-S** | Sensitivity of the result to which initial pinch you start from |

Every step of the search enforces joint limits, self-collision and object collision, a rolling-contact constraint, and antipodal force feasibility.

This is the reference implementation for the IROS 2026 paper *"A Kinematic Metric for Fine Manipulation Ability in Robotic Hands"* ([arXiv:2605.15548](https://arxiv.org/abs/2605.15548)).

---

## Install

Requires **Python 3.10** in a conda environment: `pinocchio` and `hpp-fcl` ship as PyPI `cmeel` wheels that need it.

```bash
conda env create -f environment.yml
conda activate karma-hand-metric
```

Check that it works (about 10 seconds):

```bash
python run_metric.py --config robots/robot_inspire.yaml
```

The metric is CPU-only: there is no GPU or torch dependency.

> `pinocchio` comes from the PyPI package named **`pin`**, not the unrelated project called `pinocchio`. `environment.yml` handles this; if you install by hand, use `requirements.txt`.

---

## Explore the paper's results

This repo ships the full 16-hand run from the paper, along with a tool to explore the voxel clouds without computing anything:

```bash
python run_viser_app.py
```

Open <http://localhost:8080> and pick a hand from the dropdown. It will load the published reachable sets from `results/16_hand_batch/` by default. Each voxel is a reachable sphere-centre position, colored by how much of the orientation space is reachable there, red for the least and green for the most.

The bundled hands are Ability, Allegro, ARMS, D'Claw, Unitree Dex3, Unitree Dex5, DLR, Inspire, LEAP, OrcaHand, Shadow, Sharpa, Schunk SVH, Wuji, xHand1, and xHandLite.

## Run the metric yourself

Recompute the KaRMA scores for any hand on your own computer:

```bash
python run_metric.py --config robots/robot_leap.yaml          # ~5 min
python run_metric.py --config robots/robot_shadowhand.yaml    # ~45 s
python run_metric.py --config robots/robot_inspire.yaml       # ~10 s
```

Each run writes a summary to `workspace/current.yaml` and its full reachable set to `workspace/current.pkl`. View your run with `python run_viser_app.py --result workspace/current.pkl`.

To score all 16 bundled hands at once, run `python run_all_hands.py`. Runtime scales with how dexterous the hand is, since a more capable hand reaches more voxels. The full sweep takes roughly 18 minutes on a 32-thread desktop.

---

## Add your own hand

1. Put the URDF under `robots/urdfs/`.
2. Copy `robots/robot_template.yaml` to `robots/robot_<yourhandname>.yaml`. **The template is the guide: it is commented field by field**, walking you through the base link, the thumb and index active joints, the contact links, the tip lengths, and any coupled/mimic joints, like the PIP/DIP couplings common on many hands. An LLM can draft most of it from a URDF: give your favorite LLM the prompt at `robots/prompt.txt` together with your URDF, then **always remember to double-check the output.**
3. **Check the capsule geometry visually** (`python run_viser_app.py`, or `python tools/tune_tip_lengths.py` to set tip lengths) before trusting any number. Finger links are modeled as capsules built procedurally from the joint tree, not from URDF collision meshes, so a wrong tip length gives a tiny score instead of an error.
4. Run it: `python run_metric.py --config robots/robot_<yourhandname>.yaml`.

---

## Reproducing the paper

The camera-ready results ship in this repo:

- `results/16_hand_batch/summary.yaml`: Table I, all 16 hands
- `results/16_hand_batch/robot_<name>/current.pkl`: full reachable sets
- `results/ablation/`: the constraint ablation (Table IV)
- `results/robustness/`: invariance and sensitivity (Table III)

Regenerate the baseline correlations and the KaRMA-vs-opposability figure:

```bash
python -m baselines.run_baselines
python scripts/rebuild_analysis_data.py
python figures/plot_karma_vs_opposability.py
```

To regenerate the ablation and robustness tables themselves, use the scripts in `experiments/` (see `experiments/README.md`).

### Reproducibility

Given a fixed machine and the pinned environment, KaRMA is deterministic and **bit-reproducible**: repeated runs are identical, and the scores are exactly invariant to translating, rotating, or uniformly rescaling the URDF. That invariance is by construction: the base link is canonicalized to the world origin and joint placements are snapped to an `L_ref`-relative grid, which removes the floating-point residue that would otherwise flip a boundary voxel.

You can check this directly. The as-run evidence ships with the results — `results/16_hand_batch/determinism_3runs.txt` (three identical runs per hand) and `invariance_table_exact.txt` (Table III) — and both are regenerated by the robustness harness (`python robust_tests/run_tests.py`; `experiments/run_robustness_all_robots.py robust_tests/test_config_invariance.yaml` for invariance, see `experiments/README.md`).

Across *different* machines, expect a voxel or two of variation on a given hand, since different linear-algebra backends round the solves slightly differently and a few voxels sit right on a feasibility boundary. It never changes the ranking. The published numbers were generated on x86-64 Linux; run there to match the table exactly.

---

## Scripts

Everything runnable lives in one of these. The three at the repo root are all you
need to compute the metric; the rest are optional — inspection helpers, or scripts
that regenerate the paper's supporting data.

**Compute the metric** (repo root)

| Script | Purpose |
| --- | --- |
| `run_metric.py --config robots/robot_<name>.yaml` | one hand → `workspace/current.{yaml,pkl}` |
| `run_all_hands.py` | all 16 hands → the Table I summary |
| `run_viser_app.py` | open a computed result in the interactive viewer |

**Inspect hands** — `tools/` (none are needed to compute the metric)

| Script | Purpose |
| --- | --- |
| `tools/compare_hands.py` | overlay two hands at the same scale |
| `tools/visualize_workspace.py` | 3D reachable-fingertip workspace hull |
| `tools/fig_grasp_viewer.py` | render the paper's grasp figure |
| `tools/tune_tip_lengths.py` | set a new hand's fingertip lengths (when adding a hand) |

**Regenerate the paper's data** — `experiments/`, `baselines/`, `scripts/`, `figures/`

| Script | Purpose |
| --- | --- |
| `baselines/run_baselines.py` | the DOF / workspace / opposability / GCI baselines KaRMA is compared against |
| `experiments/run_constraint_ablation.py` | the constraint ablation (Table IV) |
| `experiments/run_robustness_all_robots.py` | invariance + sensitivity (Table III) — a driver that runs `robust_tests/run_tests.py` once per representative hand |
| `experiments/run_fixed_seed_stability.py` | fixed-seed sensitivity (isolates the BFS from seed re-selection) |
| `scripts/rebuild_analysis_data.py`, `figures/plot_karma_vs_opposability.py` | rebuild the analysis JSON and the KaRMA-vs-opposability figure |

`robust_tests/run_tests.py` is the single-hand robustness *engine*;
`run_robustness_all_robots.py` above is what drives it across all five hands — run it
directly only to test one hand.

---

## How it works

```
robot YAML + karma_config.yaml
        │
        ▼
  compute L_ref = mxPr + mF          characteristic hand length, from the URDF
        │                            (max pairwise knuckle distance + median finger length)
        ▼
  scale every length by L_ref/200mm  so all hands solve a geometrically equivalent problem
        │
        ▼
  seed selection                     generate every feasible pinch, then rank them by
        │                            running a trial search from each
        ▼
  Phase 1: translation BFS           6 primitives along the seed's manipulability axes;
        │                            each step solves a rolling-contact QP and re-projects
        ▼                            onto the two-contact manifold
  Phase 2: rotation exploration      at each reached voxel, tilt the pinch axis and record
        │                            which of the 228 HEALPix orientation bins are reachable
        ▼
  KaRMA-T, KaRMA-R, KaRMA-S          T and R are averaged over the three best
                                     seeds; S is median/best over all evaluated seeds
```

Because the search explores **object poses** (a fixed 3D voxel grid), not joint configurations, its cost grows with the voxels a hand can reach, not exponentially with the hand's DOF. A more dexterous hand is slower only because it reaches more of them.

### Layout

```
karma/                  the metric
  config.py             merges the two YAMLs, computes L_ref, scales all lengths
  lref.py               characteristic hand length from the URDF joint tree
  robot.py              pinocchio wrapper, procedural capsules, pose canonicalization
  contacts.py           capsule-sphere contact kinematics and Jacobians
  collisions.py         sphere-vs-link and thumb-vs-index clearance
  grasp.py              antipodal friction-cone feasibility
  rolling_qp.py         the rolling-contact QP (OSQP), translation and rotation steps
  seed.py               solving one candidate pinch (IK primitives)
  seed_selection.py     which seeds to try, and how they are ranked
  orientation.py        twist-invariant S^2 representation and HEALPix binning
  metric.py             the search and scoring (computes a MetricResult)
  projection.py         Gauss-Newton projection onto the two-contact rolling manifold
  results.py            result types and output writing
robots/                 per-hand configs (robot_*.yaml)
  urdfs/                kinematic URDFs the metric reads (meshless; fingers are modeled as capsules)
  visual_models/        visual meshes for the comparison tool only, one folder per hand
baselines/              DOF, workspace, opposability, Yoshikawa, GCI
results/                camera-ready results
tools/                  interactive helpers (tip-length tuner, viewers), not needed to run the metric
experiments/            scripts that regenerate the ablation and robustness data
docs/                   methodology notes (non-dimensionalization, slip tolerance)
```

The models under `robots/urdfs/` and `robots/visual_models/` are third-party, each under its own license; the MIT license here covers only the KaRMA code, and per-hand sources and licenses are listed in [`robots/visual_models/ATTRIBUTIONS.md`](robots/visual_models/ATTRIBUTIONS.md).

### Configuration

`karma_config.yaml` holds the global parameters: sphere radius, capsule radius, voxel size, friction, HEALPix resolution, search budget, contact tolerances, and the QP regularization. Every length is *nominal*, defined for a 200 mm hand, and rescaled automatically per hand. Per-hand files under `robots/` specify only that hand's joints, contact links, and geometry.

Two properties the metric leans on, exact scale invariance and the slip tolerance behind the rolling gate, are derived in `docs/`.

---

## Scope and limitations

KaRMA is a standardized **lower bound on thumb–index rolling-pinch dexterity**, not a universal dexterity score, and not a predictor of task success. 

- One two-finger pinch on a sphere. No multi-finger grasps, no gaiting, no regrasping.
- A sphere has no edges or corners, so geometry-dependent effects are out of scope by design.
- Fingers are modeled as capsules to make the math as simple and analytical as possible. So, contact can land anywhere on a finger, and we do not attempt to emulate the real shape of the finger pads on these robotic hands.
- Kinematics only: no torque limits, no friction uncertainty, no controller.
- The rolling constraint is enforced as a hard per-step gate on tangential slip rather than an exact equality. For every hand above the low-DOF floor, median residual slip stays under 0.5% of the commanded motion. The lowest-DOF hands rely on more slip (11–32%) to move at all; a strict equality would report an uninformative zero for them.

It is intended for comparing hand *kinematics* for procurement and design iteration, and complements task benchmarks rather than replacing them.

A hand is far more than its kinematics: finger shape, surface friction, intrinsic versus extrinsic actuation, gear ratios, torque, backlash, inertia, and size all matter too. KaRMA isolates the kinematic contribution and does not account for the rest. You are more than welcome to build on KaRMA and add some of these in yourself. If you use KaRMA or the idea in your own work, please cite the paper.

## Questions

Found a bug or have a question? Please open an issue on the [repository](https://github.com/mfpeticco/karma-hand-metric).

## Citation

```bibtex
@inproceedings{peticco2026karma,
  title         = {A Kinematic Metric for Fine Manipulation Ability in Robotic Hands},
  author        = {Peticco, Martin and Agrawal, Pulkit},
  booktitle     = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year          = {2026},
  eprint        = {2605.15548},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
}
```
