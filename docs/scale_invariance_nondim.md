# Scale invariance and non-dimensionalization

KaRMA compares the fine-manipulation kinematics of hands that differ in absolute
size, from the 155 mm Inspire to the 327 mm D'Claw, and it returns the same score
whether a URDF places the hand at the origin or a meter away in an arbitrary
orientation. Both properties are built into the metric rather than checked after
the fact: the scores are exactly invariant to translating, rotating, or uniformly
rescaling the URDF, and repeated runs on a fixed machine are bit-identical. This
note explains how, and points to where each piece lives in the code.

## The characteristic length L_ref

Every length in `karma_config.yaml` (sphere radius, voxel size, contact and
collision tolerances, seed-search offsets, step size) is *nominal*: it is defined
for a 200 mm reference hand and rescaled per hand by `L_ref / 200 mm` before use
(`karma/config.py`). L_ref is a characteristic hand length computed from the URDF
joint tree alone (`karma/lref.py`):

    L_ref = mxPr + mF

where `mxPr` is the maximum pairwise distance between finger-knuckle origins (palm
spread) and `mF` is the median knuckle-to-tip chain length (finger reach). The two
terms bound where a feasible pinch can form: a pinch needs fingers that can meet
(spread) and fingers long enough to wrap a sphere (reach). L_ref reads only the
fixed-joint transforms of the tree, never joint angles, so it is pose-independent
and does not change when the URDF is placed differently in the world.

Non-dimensionalizing by L_ref is also what makes KaRMA-T dimensionless. With the
voxel edge set to `h = 0.05 L_ref`, the translational score reduces to a voxel
count divided by 8000: the fraction of an `L_ref`-sided cube filled by the
reachable set, with the count averaged over the three best seeds. Two
geometrically similar hands solve the same problem and receive the same number.

## Why the three invariances hold by construction

**Translation and rotation of the world frame.** Before any forward kinematics,
`karma/robot.py` canonicalizes the base link to the world origin, so a rigid
transform applied to the whole robot cancels and cannot reach the score.

**Uniform scaling.** The rolling-contact QP and the seed IK are solved in
non-dimensional form. `karma/rolling_qp.py` divides the QP data (the rolling map,
its target, and the gap constraints) by the sphere radius before the solve, and
the QP regularization enters as `reg / sphere_radius^2`. The numerical problem is
therefore O(1) at every hand size, which is what lets OSQP use a single absolute
tolerance (`eps_abs = eps_rel = 1e-7`) instead of one that would need per-hand
tuning. A hand scaled by 2x has all of its lengths, including L_ref and the sphere
radius, scaled together, so the non-dimensional problem it hands to the solver is
unchanged.

**Floating-point residue.** Canonicalizing the base pose leaves a small
floating-point residual in the joint placements. Because KaRMA counts voxels
across discrete feasibility thresholds, a residual of a few ULP can flip a
boundary voxel and perturb the count. We remove it by snapping the canonicalized
joint placements to a fixed `L_ref`-relative grid before forward kinematics
(`_snap_root_placements`, `KARMA_SNAP_DECIMALS` in `karma/robot.py`). The grid
spacing sits far below any physical tolerance, so the snap changes no feasibility
decision on its own; it only discards the residue that world-frame placement would
otherwise leave behind.

## What is and is not guaranteed

On a fixed machine with the pinned environment, these invariances are exact. The
paper's invariance table reports zero deviation in KaRMA-T and KaRMA-R under four
translations, four rotations, and 0.5x / 2x scaling per hand, and three repeated
runs are bit-identical; the committed evidence is in
`results/16_hand_batch/invariance_table_exact.txt` and `determinism_3runs.txt`.

Across different machines, a hand can differ by a voxel or two. Different
linear-algebra backends round the QP and IK solves slightly differently, and a few
voxels sit exactly on a feasibility boundary. This never changes the ranking. The
numbers in the paper were generated on x86-64 Linux; run there to match the table
exactly.
