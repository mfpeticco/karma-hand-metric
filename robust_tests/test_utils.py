"""Utility functions for robustness tests: URDF transforms and config preparation.

This module has NO dependency on karma — all operations are pure file
manipulation (XML and YAML).  This makes it testable and importable without
loading heavy libraries like Pinocchio.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import yaml


# ═══════════════════════════════════════════════════════════════════════════════
# URDF Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def find_urdf_root_link(urdf_path: str | Path) -> str:
    """Find the root link of a URDF (the link that is never a child in any joint)."""
    tree = ET.parse(str(urdf_path))
    root = tree.getroot()
    all_links = {el.attrib['name'] for el in root.findall('link')}
    child_links = {j.find('child').attrib['link'] for j in root.findall('joint')}
    root_links = all_links - child_links
    if len(root_links) != 1:
        raise ValueError(f"Expected exactly 1 root link, found {len(root_links)}: {root_links}")
    return root_links.pop()


def get_active_joint_names(robot_yaml_path: str | Path) -> list[str]:
    """Extract all active joint names (thumb + index) from a robot YAML config."""
    with open(str(robot_yaml_path)) as f:
        data = yaml.safe_load(f)
    fingers = data.get('fingers', {})
    joints = []
    for finger in ('thumb', 'index'):
        joints.extend(fingers.get(finger, {}).get('active_joints', []))
    return joints


# ═══════════════════════════════════════════════════════════════════════════════
# URDF Transformation
# ═══════════════════════════════════════════════════════════════════════════════

def add_urdf_wrapper_joint(
    urdf_path: str | Path,
    output_path: str | Path,
    xyz: tuple | list = (0, 0, 0),
    rpy: tuple | list = (0, 0, 0),
) -> None:
    """Add a fixed wrapper joint before the URDF root link.

    Creates a new root link '_test_wrapper' connected to the original root link
    via a fixed joint with the given origin xyz/rpy.  This effectively
    translates and/or rotates the entire robot in world frame.

    Args:
        urdf_path: Path to the original URDF.
        output_path: Where to write the modified URDF.
        xyz: Translation offset [x, y, z] in metres.
        rpy: Rotation offset [roll, pitch, yaw] in radians.
    """
    tree = ET.parse(str(urdf_path))
    root = tree.getroot()
    original_root = find_urdf_root_link(urdf_path)

    # New wrapper link (becomes the new universe root)
    link_el = ET.SubElement(root, 'link')
    link_el.set('name', '_test_wrapper')

    # Fixed joint: _test_wrapper -> original root
    joint_el = ET.SubElement(root, 'joint')
    joint_el.set('name', '_test_wrapper_joint')
    joint_el.set('type', 'fixed')

    parent_el = ET.SubElement(joint_el, 'parent')
    parent_el.set('link', '_test_wrapper')
    child_el = ET.SubElement(joint_el, 'child')
    child_el.set('link', original_root)

    origin_el = ET.SubElement(joint_el, 'origin')
    origin_el.set('xyz', f'{xyz[0]} {xyz[1]} {xyz[2]}')
    origin_el.set('rpy', f'{rpy[0]} {rpy[1]} {rpy[2]}')

    tree.write(str(output_path), xml_declaration=True, encoding='unicode')


def scale_urdf(
    urdf_path: str | Path,
    factor: float,
    output_path: str | Path,
) -> None:
    """Scale all spatial dimensions in the URDF by a uniform factor.

    Scales:
      - All <origin xyz="..."> attributes (joints, collisions, inertials, visuals)
      - Collision geometry dimensions (cylinder length/radius, sphere radius, box size)
      - Mesh scale attributes

    Does NOT scale (angular / dimensionless):
      - Origin rpy, joint limits (lower/upper), effort, velocity
    """
    tree = ET.parse(str(urdf_path))
    root = tree.getroot()

    # Scale all origin xyz
    for origin in root.iter('origin'):
        xyz_str = origin.get('xyz')
        if xyz_str:
            vals = [float(v) * factor for v in xyz_str.split()]
            origin.set('xyz', ' '.join(str(v) for v in vals))

    # Scale collision / visual geometry dimensions
    for cyl in root.iter('cylinder'):
        for attr in ('length', 'radius'):
            val = cyl.get(attr)
            if val:
                cyl.set(attr, str(float(val) * factor))

    for sphere in root.iter('sphere'):
        val = sphere.get('radius')
        if val:
            sphere.set('radius', str(float(val) * factor))

    for box in root.iter('box'):
        val = box.get('size')
        if val:
            vals = [float(v) * factor for v in val.split()]
            box.set('size', ' '.join(str(v) for v in vals))

    for mesh in root.iter('mesh'):
        scale_str = mesh.get('scale')
        if scale_str:
            vals = [float(v) * factor for v in scale_str.split()]
            mesh.set('scale', ' '.join(str(v) for v in vals))

    tree.write(str(output_path), xml_declaration=True, encoding='unicode')


def perturb_urdf_joint_origins(
    urdf_path: str | Path,
    epsilon: float,
    output_path: str | Path,
) -> None:
    """Scale all <joint>/<origin> xyz translations by (1 + epsilon).

    Only affects kinematic chain geometry (joint translations), not collision
    or inertial origins.  Simulates manufacturing tolerances in link lengths.
    """
    tree = ET.parse(str(urdf_path))
    root = tree.getroot()

    factor = 1.0 + epsilon
    for joint in root.findall('joint'):
        origin = joint.find('origin')
        if origin is not None:
            xyz_str = origin.get('xyz')
            if xyz_str:
                vals = [float(v) * factor for v in xyz_str.split()]
                origin.set('xyz', ' '.join(str(v) for v in vals))

    tree.write(str(output_path), xml_declaration=True, encoding='unicode')


def perturb_urdf_joint_limits(
    urdf_path: str | Path,
    joint_names: list[str],
    delta_rad: float,
    output_path: str | Path,
) -> None:
    """Widen (delta > 0) or narrow (delta < 0) joint limits for named joints.

    Lower limit decreases by delta, upper limit increases by delta.
    (Negative delta narrows the range.)
    """
    tree = ET.parse(str(urdf_path))
    root = tree.getroot()

    joint_set = set(joint_names)
    for joint in root.findall('joint'):
        if joint.get('name') not in joint_set:
            continue
        limit = joint.find('limit')
        if limit is None:
            continue
        lower = float(limit.get('lower', '0'))
        upper = float(limit.get('upper', '0'))
        limit.set('lower', str(lower - delta_rad))
        limit.set('upper', str(upper + delta_rad))

    tree.write(str(output_path), xml_declaration=True, encoding='unicode')


# ═══════════════════════════════════════════════════════════════════════════════
# Config File Preparation
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_robot_config(
    original_path: str | Path,
    output_path: str | Path,
    urdf_path_override: str | Path | None = None,
) -> None:
    """Copy a robot YAML config, optionally overriding urdf_path (made absolute)."""
    with open(str(original_path)) as f:
        data = yaml.safe_load(f)

    if urdf_path_override is not None:
        data['urdf_path'] = str(Path(urdf_path_override).resolve())

    with open(str(output_path), 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def prepare_metric_config(
    original_path: str | Path,
    output_path: str | Path,
    overrides: dict | None = None,
) -> None:
    """Copy a metric YAML config with optional parameter overrides.

    For any nested-dict override, the sub-dict is merged into the existing one
    rather than replacing it wholesale.
    """
    with open(str(original_path)) as f:
        data = yaml.safe_load(f)

    if overrides:
        for key, val in overrides.items():
            if isinstance(val, dict) and key in data and isinstance(data[key], dict):
                data[key].update(val)
            else:
                data[key] = val

    with open(str(output_path), 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def scale_robot_config_lengths(
    original_path: str | Path,
    factor: float,
    output_path: str | Path,
    urdf_path_override: str | Path | None = None,
) -> None:
    """Scale tip_length_m in a robot config by factor.

    Also sets urdf_path to the given override (absolute) if provided.
    """
    with open(str(original_path)) as f:
        data = yaml.safe_load(f)

    if urdf_path_override is not None:
        data['urdf_path'] = str(Path(urdf_path_override).resolve())

    links = data.get('links', {})
    if links:
        for _link_name, link_data in links.items():
            if link_data and 'tip_length_m' in link_data:
                tip = link_data['tip_length_m']
                if tip is not None:
                    link_data['tip_length_m'] = tip * factor

    with open(str(output_path), 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def scale_metric_config_lengths(
    original_path: str | Path,
    factor: float,
    output_path: str | Path,
    extra_overrides: dict | None = None,
) -> None:
    """Scale all length-dimension parameters in a metric config by factor.

    Scales: sphere_radius_m, link_radius_m, voxel_size_m, step_size_m,
            eps_g_m, eps_col_m, rotation_slip_threshold_mm,
            seed_search_offsets_m, l_ref_nominal_m (by k),
            qp_regularization (by k^2).

    Does NOT scale: healpix_nside, friction_coeff, max_joint_step_deg,
                    max_states.
    """
    with open(str(original_path)) as f:
        data = yaml.safe_load(f)

    length_params = [
        'sphere_radius_m', 'link_radius_m', 'voxel_size_m', 'step_size_m',
        'eps_g_m', 'eps_col_m',
    ]
    for param in length_params:
        if param in data:
            data[param] = data[param] * factor

    if data.get('rotation_slip_threshold_mm') is not None:
        data['rotation_slip_threshold_mm'] = data['rotation_slip_threshold_mm'] * factor

    if 'seed_search_offsets_m' in data:
        data['seed_search_offsets_m'] = [v * factor for v in data['seed_search_offsets_m']]

    # l_ref_nominal_m scales by k so that the L_ref/l_ref_nominal ratio
    # (and hence the automatic re-scaling inside load_config) stays unchanged.
    if 'l_ref_nominal_m' in data:
        data['l_ref_nominal_m'] = data['l_ref_nominal_m'] * factor

    # qp_regularization has units of length^2 (rolling cost is ||M@x-b||^2
    # where M has length units), so it scales by k^2.
    if 'qp_regularization' in data:
        data['qp_regularization'] = data['qp_regularization'] * factor ** 2

    if extra_overrides:
        for key, val in extra_overrides.items():
            data[key] = val

    with open(str(output_path), 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def swap_fingers_robot_config(
    original_path: str | Path,
    output_path: str | Path,
) -> None:
    """Swap thumb/index finger definitions (joints, contact links, seed links).

    This tests invariance to finger labelling.
    """
    with open(str(original_path)) as f:
        data = yaml.safe_load(f)

    fingers = data.get('fingers', {})
    thumb = fingers.get('thumb', {})
    index = fingers.get('index', {})

    # Swap finger blocks
    fingers['thumb'] = index
    fingers['index'] = thumb

    # Swap seed links
    seed = data.get('seed', {})
    old_thumb = seed.get('thumb_link')
    old_index = seed.get('index_link')
    seed['thumb_link'] = old_index
    seed['index_link'] = old_thumb

    with open(str(output_path), 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
