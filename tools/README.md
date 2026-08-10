# tools

Interactive and visualization helpers for inspecting hands and building figures.
You do not need any of these to compute the metric; they are for setup and
debugging. Run each from the repo root.

- `tune_tip_lengths.py`: set a hand's tip lengths by eye. Shows the
  capsule skeleton next to the URDF's real meshes (when they are on disk) and
  writes the tuned `tip_length_m` values back to a robot YAML. Use this when
  adding a new hand, since the joint tree stops short of the physical fingertip.
- `fig_grasp_viewer.py`: viser scene of a hand, sphere, contacts, and pinch axis
  at the seed grasp. Renders the paper's grasp figures.
- `visualize_workspace.py`: 3D convex-hull view of a hand's reachable fingertip
  workspace (thumb and index hulls plus their intersection), for one hand or all
  hands in a grid. This is the workspace baseline, shown visually.
- `compare_hands.py`: overlay two hand URDFs with independent 6-DOF pose sliders,
  to eyeball how they compare at the same scale. Renders the bundled visual
  models in `robots/visual_models/` by default; pass `--models-dir` to use your own
  collection instead.
