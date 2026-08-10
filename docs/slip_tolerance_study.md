# The slip tolerance and why it is a gate, not an equality

A perfect rolling contact has zero tangential slip: the contact point on the finger
and the contact point on the sphere move together, and the sphere only rolls and
spins, never slides. Enforcing that as a hard equality at every step is the
principled choice, but it is too brittle for the hands this metric is meant to
compare. A low-degree-of-freedom hand often cannot advance the sphere at all
without a little sliding, so a strict equality would report an uninformative zero
for exactly the hands where the distinction between "some dexterity" and "none"
matters most. KaRMA instead enforces rolling as a soft penalty inside the QP plus a
hard per-step gate on residual slip. This note explains what the gate measures,
where it is set, and what the data behind it looks like.

## What the gate measures

Each rolling step solves a QP that minimizes tangential slip while respecting joint
limits, the contact-gap constraints, and a rotation prior (`karma/rolling_qp.py`).
The objective is soft, so the solved step carries some residual slip. We decompose
that residual into two parts:

- **Tangential slip**: sliding of the contact transverse to the rolling direction.
  This is the "bad" slip. It is also the local linearization error of the step,
  the amount by which the finite step departs from ideal rolling.
- **Spin about the contact normal**: the free rotation a sphere permits at a point
  contact. This is not slip in any meaningful sense and is not penalized.

A step is accepted only if its tangential slip stays below the gate. The gate is
nominal `0.5 mm` at the 200 mm reference hand and scales per hand with L_ref, like
every other length (see `scale_invariance_nondim.md`). Equivalently it is about
40% of the per-sub-step commanded motion. Steps that exceed it are rejected and
recorded under `rot_slip` in the result's Phase 2 failure breakdown
(`phase2_fail_reasons`, also printed in the run log).

## Where the gate is set, and why there

We chose the gate at the elbow of the reach-versus-tolerance curve. As the
tolerance loosens from zero, the reachable workspace of a capable hand grows
quickly at first, because a small slip allowance lets steps through that
strict rolling would reject on linearization error alone, and then it plateaus:
past a point, loosening the tolerance buys almost no additional reach and only
lets in more sliding. The nominal 0.5 mm gate sits at that knee. It is loose enough
that well-conditioned hands are not penalized for the finite-step linearization
error, and tight enough that the accepted motion is still essentially rolling.

The data support that reading. For every hand above the low-degree-of-freedom
floor, the median residual tangential slip is below 0.5% of the commanded motion,
so KaRMA measures essentially pure rolling wherever the score is informative. The
five floor hands (Ability, Inspire, SVH, Unitree Dex3, and xHandLite, marked in
the paper) instead rely on a median of 11–32% slip to move at all. A strict
rolling equality would leave them with almost no reachable workspace and report
a near-zero score that says more about the modeling choice than about the hand.
The bounded gate keeps them on a common scale with the rest.

## Inspecting it

The committed per-hand slip evidence is in `results/16_hand_batch/summary.yaml`.
The `gated_slip_frac` column reports the fraction of steps that leaned on the
gate, and `median_alpha` reports the median slip as a fraction of the
per-sub-step commanded motion. These are the numbers that separate the floor
hands from the rest: `median_alpha` runs from about 0.001 for the
well-conditioned hands to 0.11 through 0.32 for the five floor hands. Both
columns were collected by the paper's batch harness and ship as a frozen
artifact; a fresh run reports its slip behavior through the `rot_slip`
rejection count described above.
