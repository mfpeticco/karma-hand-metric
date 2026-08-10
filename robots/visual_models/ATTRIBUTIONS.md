# Third-party robot model attributions

KaRMA evaluates 16 robotic hands. The kinematic URDFs in `robots/urdfs/` and the
visual meshes in `robots/visual_models/` (used only by `tools/compare_hands.py`) are
derived from third-party robot descriptions, each under its own license. **The
MIT license in `LICENSE` covers the KaRMA code only. It does not relicense these
robot models.** Each model stays under the terms listed below, and anyone
redistributing or using this repository must comply with each of them.

Two hands, ARMS and DLR, were rebuilt by the authors from published OpenSim
biomechanical models; they ship as meshless kinematic skeletons and carry no
third-party mesh geometry.

## License terms

Of the 14 mesh models, 13 are permissive (MIT, BSD, Apache-2.0, or — for Shadow —
CC BY 4.0): all permit reuse including commercial use, and require only that you
keep the source's attribution and copyright notice. The one exception is the
**SCHUNK SVH** hand, whose mesh geometry is **GPL-3.0 (copyleft)** — redistribution
is permitted, but under GPL terms (see the SVH note below).

ARMS and DLR are meshless kinematic skeletons the authors rebuilt from published
OpenSim models. They carry only their sources' terms — a citation request for
ARMS, and no explicit license for DLR (see the citations below). Neither upstream
affirmatively grants redistribution, so what ships here is the authors' own
kinematic reconstruction, distributed on a cite-the-source basis.

## Per-hand sources and licenses

| Hand (config) | Source | License | Copyright / authors |
| --- | --- | --- | --- |
| `ability_hand_right` | [psyonicinc/ability-hand-api](https://github.com/psyonicinc/ability-hand-api) | MIT | PSYONIC, Inc. |
| `allegro` | SimLab [allegro-hand-ros](https://github.com/felixduvallet/allegro-hand-ros), repackaged in Drake [`drake_models`](https://github.com/RobotLocomotion/models) | BSD | Wonik Robotics; Drake modifications © Toyota Research Institute |
| `ARMS_skel` | [ARMS Lab hand and wrist model](https://simtk.org/projects/arms_hand_model) (OpenSim); URDF rebuilt by the KaRMA authors | Custom Use Agreement (cite the paper below) | McFarland, Binder-Markey, Nichols, Wohlman, de Bruin, Murray (ARMS Lab) |
| `dclaw` | Google [ROBEL](https://github.com/google-research/robel) D'Claw | Apache-2.0 | The ROBEL Authors, 2019 |
| `dex3` | [unitreerobotics/unitree_ros](https://github.com/unitreerobotics/unitree_ros) (Dex3-1) | BSD-3-Clause | © 2016-2022 Hangzhou Yushu Technology Co., Ltd. (Unitree Robotics) |
| `dex5` | [unitreerobotics/unitree_ros](https://github.com/unitreerobotics/unitree_ros) (Dex5-1) | BSD-3-Clause | © 2016-2022 Hangzhou Yushu Technology Co., Ltd. (Unitree Robotics) |
| `DLR` | [OpenSim DLR Hand (Sept 2013)](https://www.handcorpus.org/?p=1077); URDF rebuilt by the KaRMA authors | No explicit license (cite the paper below) | G. Stillfried, U. Hillenbrand, M. Settles, P. van der Smagt (DLR) |
| `inspire` | [unitreerobotics/unitree_ros](https://github.com/unitreerobotics/unitree_ros) (Inspire RH56, `h1_description` meshes) | BSD-3-Clause | © 2016-2022 Hangzhou Yushu Technology (Unitree); hand geometry by Inspire Robotics |
| `leap` | [dexsuite/dex-urdf](https://github.com/dexsuite/dex-urdf) (LEAP Hand) | MIT | © 2023 Ananye Agarwal (LEAP Hand, CMU) |
| `orcahand` | [orcahand/orcahand_description](https://github.com/orcahand/orcahand_description) | MIT | © 2025 ORCA (maintainer Arturo Roberti) |
| `shadowhand` | Shadow Robot [`sr_description`](https://github.com/shadow-robot/sr_common) | CC BY 4.0 (meshes; see note) | © Shadow Robot Company Ltd.; UPMC / Guillaume Walck |
| `sharpa` | [sharpa-robotics/sharpa-urdf-usd-xml](https://github.com/sharpa-robotics/sharpa-urdf-usd-xml) (the earlier HA4 model; see note) | Permissive — BSD / Apache-2.0 (see note) | © Sharpa Group |
| `svh` | [dexsuite/dex-urdf](https://github.com/dexsuite/dex-urdf) (`schunk_hand`) | **GPL-3.0** (see note) | dex-urdf authors; SCHUNK SVH hand |
| `wuji` | [wuji-technology/wuji-description](https://github.com/wuji-technology/wuji-description) | MIT | © 2025 Wuji Technology |
| `xhand1` | RobotEra ([roboterax](https://github.com/roboterax); see note) | BSD (declared; unverified — see note) | RobotEra |
| `xhandlite` | RobotEra ([roboterax](https://github.com/roboterax); see note) | BSD (declared; unverified — see note) | RobotEra |

Each model folder ships an `ATTRIBUTION.txt` recording its source, license, and
copyright. Full upstream license texts are available at the linked sources.

## Citations for the two OpenSim-derived hands

**ARMS** (Custom Use Agreement asks that you cite the accompanying publication):

> D. C. McFarland, B. I. Binder-Markey, J. A. Nichols, S. J. Wohlman, M. de
> Bruin, and W. M. Murray, "A Musculoskeletal Model of the Hand and Wrist
> Capable of Simulating Functional Tasks," bioRxiv 2021.12.28.474357, 2021.
> doi:10.1101/2021.12.28.474357

**DLR** (requested citation):

> G. Stillfried, U. Hillenbrand, M. Settles, and P. van der Smagt, "MRI-based
> skeletal hand movement model," in *The Human Hand as an Inspiration for Robot
> Hand Development*, Springer Tracts in Advanced Robotics, 2013.

## Notes

- **Unitree (Dex3, Dex5, Inspire):** taken from `unitreerobotics/unitree_ros`,
  whose LICENSE file is BSD-3-Clause, © 2016-2022 Hangzhou Yushu Technology Co.,
  Ltd. (Unitree Robotics).
- **Shadow:** the visual meshes are Shadow's blender-exported geometry, governed
  by the CC BY 4.0 license upstream (`sr_description/blender/`) — attribution
  required, commercial use permitted, no share-alike. Upstream credits both the
  Shadow Robot Company and UPMC / Guillaume Walck; keep both. Shadow's ROS
  `sr_description` `package.xml` carries inconsistent license metadata (a BSD
  header comment and a `GPL` `<license>` tag) that does not govern the mesh
  geometry.
- **SVH (GPL-3.0):** dex-urdf's README summary table lists SVH as Apache-2.0, but
  the `LICENSE` file inside the `robots/hands/schunk_hand/` folder the meshes come
  from is GPL-3.0, as is the upstream it cites
  ([`SCHUNK-GmbH-Co-KG/schunk_svh_ros_driver`](https://github.com/SCHUNK-GmbH-Co-KG/schunk_svh_ros_driver)).
  The per-folder LICENSE governs, so we label it GPL-3.0. GPL is copyleft:
  redistribution and modification are allowed provided the GPL terms and license
  text (available at the source) travel with the files.
- **Sharpa:** our model is the earlier `Right_Sharpa_HA4`, whose own `package.xml`
  declared BSD. That standalone HA4 packaging is no longer published upstream: the
  repository now hosts the renamed "Sharpa Wave" hand (same hardware) under
  Apache-2.0, © 2025 Sharpa Group. Both are permissive; the original BSD
  declaration is no longer verifiable against a live source.
- **RobotEra XHand (xhand1, xhandlite):** the cited `roboterax/models` repository is
  BSD-3-Clause but no longer hosts (and, in its retrievable history, never held) the
  XHand URDF. We could not find a canonical, license-verified RobotEra source; the
  BSD label is the model's own `package.xml` declaration and is unverified.
- **Comparison-tool meshes:** every hand pairs the KaRMA metric URDF (from
  `robots/urdfs/`) with the source's own meshes, so the visualizer shows the exact
  model the metric uses. Inspire's meshes come from `unitree_ros` (`h1_description`),
  Orca's from `orcahand_description`, and SVH's from dex-urdf's `schunk_hand`, each
  matching its metric URDF. This affects only the visualization tool, never the metric.
