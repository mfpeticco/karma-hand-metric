"""L_ref: the characteristic hand length used to non-dimensionalize the metric.

    L_ref = mxPr + mF

where ``mxPr`` is the maximum pairwise distance between finger knuckle joint
origins and ``mF`` is the median knuckle-to-tip chain ("wire") length.  The sum
captures both palm spread and finger reach, the two geometric factors that bound
where a feasible pinch can form.

The computation is pose-independent: it reads only the URDF joint tree and the
fixed-joint transforms, never joint angles.  Every length parameter in
``karma_config.yaml`` is scaled by ``L_ref / l_ref_nominal_m`` so that hands of
different sizes are posed an equivalent geometric problem.

Algorithm
---------
1. Find the palm root: the deepest common ancestor of all finger-like leaf
   chains, which excludes wrist and forearm structure without naming fingers.
2. Per finger chain, locate the knuckle (first non-mimic revolute joint) via
   SE(3) forward kinematics through fixed joints, and sum link lengths from
   knuckle to tip for the wire length.
3. Adaptive knuckle advancement: hands with an opposition or spread joint before
   the true MCP inflate the apparent palm.  A finger is advanced to the next
   revolute with ROM >= 45 deg when its wire is an outlier (>1.35x median) or its
   knuckle ROM is under 25 deg, but only when doing so decreases mxPr.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml

from .math_utils import rpy_to_matrix

# Finger chains shorter than this are hardware stubs, not fingers.
_MIN_FINGER_WIRE_M = 0.010

# Adaptive knuckle advancement thresholds.
_WIRE_RATIO_THRESH = 1.35
_ROM_MIN_DEG = 25.0
_ROM_TARGET_DEG = 45.0


# ── SE(3) helpers ─────────────────────────────────────────────────────────

def _make_transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rpy_to_matrix(rpy)
    T[:3, 3] = xyz
    return T


# ── URDF parsing ─────────────────────────────────────────────────────────

def _parse_urdf(
    urdf_path: Path,
) -> tuple[dict, dict[str, tuple[str, str]], dict[str, list[tuple[str, str]]]]:
    """Read the joint tree, returning (joints, child->parent, parent->children)."""
    root = ET.parse(str(urdf_path)).getroot()

    joints: dict[str, dict] = {}
    child_to_parent: dict[str, tuple[str, str]] = {}
    parent_to_children: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for j in root.findall("joint"):
        name = j.attrib["name"]
        parent_link = j.find("parent").attrib["link"]
        child_link = j.find("child").attrib["link"]

        origin_el = j.find("origin")
        xyz = np.zeros(3)
        rpy = np.zeros(3)
        if origin_el is not None:
            xyz = np.fromstring(origin_el.attrib.get("xyz", "0 0 0"), sep=" ", dtype=float)
            rpy = np.fromstring(origin_el.attrib.get("rpy", "0 0 0"), sep=" ", dtype=float)

        limit_el = j.find("limit")
        limit_lower, limit_upper = 0.0, 0.0
        if limit_el is not None:
            limit_lower = float(limit_el.attrib.get("lower", "0"))
            limit_upper = float(limit_el.attrib.get("upper", "0"))

        joints[name] = {
            "type": j.attrib.get("type", "fixed"),
            "parent": parent_link,
            "child": child_link,
            "xyz": xyz,
            "rpy": rpy,
            # Mimic joints are mechanism artifacts, never real knuckles.
            "is_mimic": j.find("mimic") is not None,
            "limit_lower": limit_lower,
            "limit_upper": limit_upper,
        }

        child_to_parent[child_link] = (parent_link, name)
        parent_to_children[parent_link].append((child_link, name))

    return joints, child_to_parent, dict(parent_to_children)


def _is_real_revolute(jinfo: dict) -> bool:
    return jinfo["type"] in ("revolute", "continuous") and not jinfo.get("is_mimic", False)


# ── Tree traversal ────────────────────────────────────────────────────────

def _path_to_root(link: str, c2p: dict) -> list[str]:
    path, visited = [link], {link}
    while link in c2p:
        parent = c2p[link][0]
        if parent in visited:
            break
        path.append(parent)
        visited.add(parent)
        link = parent
    return path


def _find_lca(a: str, b: str, c2p: dict) -> str:
    bs = set(_path_to_root(b, c2p))
    for x in _path_to_root(a, c2p):
        if x in bs:
            return x
    raise ValueError(f"No common ancestor for links {a!r} and {b!r}")


def _chain_links(ancestor: str, descendant: str, c2p: dict) -> list[tuple[str, str]]:
    """Ordered (joint, link) pairs walking from ancestor down to descendant."""
    chain: list[tuple[str, str]] = []
    link = descendant
    while link != ancestor:
        if link not in c2p:
            raise ValueError(f"Cannot reach {ancestor!r} from {descendant!r}")
        parent, jname = c2p[link]
        chain.append((jname, link))
        link = parent
    chain.reverse()
    return chain


def _all_leaves(start: str, p2c: dict) -> list[str]:
    results: list[str] = []

    def dfs(link: str) -> None:
        children = p2c.get(link, [])
        if not children:
            results.append(link)
        else:
            for child, _ in children:
                dfs(child)

    dfs(start)
    return results


# ── Chain analysis ────────────────────────────────────────────────────────

def _analyze_chain(
    chain: list[tuple[str, str]],
    joints_db: dict,
    tip_offset: np.ndarray | None = None,
    start_rev_idx: int | None = None,
) -> tuple[np.ndarray, float, bool, str | None, int | None]:
    """Split a chain into (knuckle position, finger wire length).

    ``start_rev_idx`` overrides which joint counts as the knuckle, used by the
    adaptive advancement pass.  Returns ``has_finger=False`` for chains with no
    real revolute joint.
    """
    if start_rev_idx is not None:
        first_rev_idx: int | None = start_rev_idx
    else:
        first_rev_idx = None
        for i, (jname, _) in enumerate(chain):
            if _is_real_revolute(joints_db[jname]):
                first_rev_idx = i
                break

    if first_rev_idx is None:
        return np.zeros(3), 0.0, False, None, None

    # Knuckle position: FK through every joint up to and including the knuckle.
    T = np.eye(4)
    for i in range(first_rev_idx + 1):
        jinfo = joints_db[chain[i][0]]
        T = T @ _make_transform(jinfo["xyz"], jinfo["rpy"])
    knuckle_pos = T[:3, 3].copy()

    finger_wire = sum(
        float(np.linalg.norm(joints_db[chain[i][0]]["xyz"]))
        for i in range(first_rev_idx + 1, len(chain))
    )
    if tip_offset is not None:
        finger_wire += float(np.linalg.norm(tip_offset))

    return knuckle_pos, finger_wire, True, chain[first_rev_idx][0], first_rev_idx


def _max_pairwise(positions: list[np.ndarray]) -> float:
    if len(positions) < 2:
        return 0.0
    return max(
        float(np.linalg.norm(positions[i] - positions[j]))
        for i, j in combinations(range(len(positions)), 2)
    )


# ── Public API ────────────────────────────────────────────────────────────

def compute_lref(robot_yaml_path: str | Path) -> float:
    """Return L_ref in metres for a ``robots/robot_*.yaml`` config.

    Raises on a malformed config or URDF rather than falling back to a sentinel:
    a silently zeroed L_ref would zero KaRMA-T without any visible error.
    """
    return compute_lref_breakdown(robot_yaml_path)["L_ref_m"]


def compute_lref_breakdown(robot_yaml_path: str | Path) -> dict:
    """Return the L_ref decomposition for a ``robots/robot_*.yaml`` config.

    Keys: ``L_ref_m`` (= mxPr + mF, the scalar :func:`compute_lref` returns and
    the metric uses), ``mxPr_m`` (max pairwise knuckle distance = palm diameter),
    ``mF_m`` (median finger wire length), ``n_fingers``, and ``finger_wires_m``.
    This is the single source of L_ref for both the metric and the robustness
    survey; there is no separate implementation.
    """
    robot_yaml_path = Path(robot_yaml_path)
    with open(robot_yaml_path) as f:
        robot = yaml.safe_load(f) or {}

    urdf_path = robot_yaml_path.parent / robot["urdf_path"]
    joints_db, c2p, p2c = _parse_urdf(urdf_path)

    base_link = robot["base_link"]
    links_cfg = robot.get("links", {}) or {}

    def tip_off(link: str) -> np.ndarray | None:
        # Only the magnitude matters for a wire length, so direction is arbitrary.
        cfg = links_cfg.get(link)
        if cfg and isinstance(cfg, dict) and "tip_length_m" in cfg:
            return np.array([float(cfg["tip_length_m"]), 0.0, 0.0], dtype=float)
        return None

    # Palm root: deepest common ancestor of all finger-like leaves, which drops
    # wrist/forearm structure without hard-coding finger names.
    finger_leaves: list[str] = []
    for leaf in _all_leaves(base_link, p2c):
        chain = _chain_links(base_link, leaf, c2p)
        if not any(_is_real_revolute(joints_db[jn]) for jn, _ in chain):
            continue
        wire = sum(float(np.linalg.norm(joints_db[jn]["xyz"])) for jn, _ in chain)
        off = tip_off(leaf)
        if off is not None:
            wire += float(np.linalg.norm(off))
        if wire >= _MIN_FINGER_WIRE_M:
            finger_leaves.append(leaf)

    hand_root = base_link
    if len(finger_leaves) >= 2:
        hand_root = finger_leaves[0]
        for fl in finger_leaves[1:]:
            hand_root = _find_lca(hand_root, fl, c2p)

    # One entry per knuckle; when several leaves share a knuckle (e.g. the
    # xHand URDFs' "rotaback" links branching off a finger) keep the longest
    # chain.
    fingers: dict[str, tuple] = {}
    for leaf in _all_leaves(hand_root, p2c):
        chain = _chain_links(hand_root, leaf, c2p)
        off = tip_off(leaf)
        kpos, fwire, has_finger, first_rev, rev_idx = _analyze_chain(chain, joints_db, off)
        if not has_finger or fwire < _MIN_FINGER_WIRE_M:
            continue
        if first_rev not in fingers or fwire > fingers[first_rev][3]:
            fingers[first_rev] = (chain, off, kpos, fwire, rev_idx)

    _advance_knuckles(fingers, joints_db)

    knuckles = [d[2] for d in fingers.values()]
    wires = [d[3] for d in fingers.values()]

    mxPr = _max_pairwise(knuckles)
    mF = float(np.median(wires)) if wires else 0.0
    l_ref = mxPr + mF
    if l_ref <= 0.0:
        # Honor the compute_lref docstring: raise rather than return a sentinel.
        # A silently zeroed L_ref would zero KaRMA-T with no visible error.
        raise ValueError(
            f"Degenerate L_ref ({l_ref}) for {robot_yaml_path}: no finger chain "
            f"passed the revolute / minimum-wire-length filter. Check the config's "
            f"active joints and the URDF."
        )
    return {
        "L_ref_m": l_ref,
        "mxPr_m": mxPr,
        "mF_m": mF,
        "n_fingers": len(fingers),
        "finger_wires_m": [float(w) for w in wires],
    }


def _advance_knuckles(fingers: dict[str, tuple], joints_db: dict) -> None:
    """Advance outlier knuckles distally, in place, until mxPr stops improving.

    Some hands place an opposition or spread joint before the true MCP, which
    inflates the apparent palm width.  Advancement is kept only when it actually
    reduces mxPr, so hands without such a joint are left untouched.
    """
    for _ in range(5):  # converges in one or two passes in practice
        wires = [d[3] for d in fingers.values()]
        if not wires:
            return
        med_wire = float(np.median(wires))
        current_mxPr = _max_pairwise([d[2] for d in fingers.values()])

        advanced_any = False
        for frev, (chain, off, _kpos, fwire, rev_idx) in list(fingers.items()):
            jinfo = joints_db[chain[rev_idx][0]]
            rom_deg = math.degrees(jinfo["limit_upper"] - jinfo["limit_lower"])
            wire_ratio = fwire / med_wire if med_wire > 0 else 1.0
            if not (wire_ratio > _WIRE_RATIO_THRESH or rom_deg < _ROM_MIN_DEG):
                continue

            next_idx = _next_revolute(chain, joints_db, rev_idx, _ROM_TARGET_DEG)
            if next_idx is None:
                continue

            new_kpos, new_fwire, _, _, new_rev_idx = _analyze_chain(
                chain, joints_db, off, start_rev_idx=next_idx
            )
            tentative = dict(fingers)
            tentative[frev] = (chain, off, new_kpos, new_fwire, new_rev_idx)

            if _max_pairwise([d[2] for d in tentative.values()]) < current_mxPr:
                fingers[frev] = tentative[frev]
                advanced_any = True

        if not advanced_any:
            return


def _next_revolute(
    chain: list[tuple[str, str]], joints_db: dict, after_idx: int, min_rom_deg: float
) -> int | None:
    for i in range(after_idx + 1, len(chain)):
        jinfo = joints_db[chain[i][0]]
        if not _is_real_revolute(jinfo):
            continue
        if math.degrees(jinfo["limit_upper"] - jinfo["limit_lower"]) >= min_rom_deg:
            return i
    return None
