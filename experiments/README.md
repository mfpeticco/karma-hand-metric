# experiments

Scripts that regenerate the paper's supporting data. The main results ship
precomputed under `results/`, so you only need these to reproduce the ablation
or the robustness study from scratch. Run each from the repo root.

- `run_constraint_ablation.py`: BFS with constraints removed incrementally, over
  the five representative hands (Table IV).

  ```bash
  python experiments/run_constraint_ablation.py
  ```

  Table IV's KaRMA-T column is the single-seed, per-level value `n_voxels / 8000`
  (e.g. LEAP "Full" = 792/8000 = 0.099). The `karma_t` field in
  `results/ablation/ablation_combined.json` is instead the top-3-seed mean (Table I's
  convention, e.g. 0.097 for LEAP): the same voxels under a different average. To
  reproduce the printed Table IV, use `n_voxels / 8000`, not that field.

- `run_robustness_all_robots.py`: the URDF-transform invariance and perturbation
  sensitivity suite, driven by a `robust_tests/` config (Table III). This is a
  driver — it runs the single-hand engine `robust_tests/run_tests.py` once per
  representative hand. Run that engine directly to test just one hand.

  ```bash
  # Full suite (invariance + sensitivity), as committed in results/robustness/:
  python experiments/run_robustness_all_robots.py robust_tests/test_config.yaml

  # Invariance rows only (translation / rotation / scale):
  python experiments/run_robustness_all_robots.py robust_tests/test_config_invariance.yaml
  ```

- `run_fixed_seed_stability.py`: fixed-seed sensitivity, isolating the BFS
  physics from seed re-selection.

  ```bash
  python experiments/run_fixed_seed_stability.py robots/robot_leap.yaml   # one hand
  python experiments/run_fixed_seed_stability.py ALL                      # five hands, several hours
  ```

Where the output goes: the robustness suite writes per-run files to
`robust_tests/results/` (gitignored); the shipped copies live in
`results/robustness/`. The ablation script and the default `--out` of
`run_fixed_seed_stability.py` write into `results/` directly, overwriting the
shipped data in place.
