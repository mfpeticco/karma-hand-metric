"""Procedural mesh generation for viser: capsule and sphere geometries."""
from __future__ import annotations

import numpy as np


def capsule_mesh(
    radius: float, half_length: float, n_rings: int = 12, n_segments: int = 16
) -> tuple[np.ndarray, np.ndarray]:
    """Generate vertices and faces for a capsule (cylinder + hemisphere caps).

    The capsule is oriented along the local z-axis, centred at origin.
    Returns (vertices (N,3), faces (M,3)).
    """
    verts = []
    faces = []
    half_rings = n_rings // 2

    # Top hemisphere: pole down to equator (z = half_length+radius .. half_length)
    for i in range(half_rings + 1):
        phi = (np.pi / 2) * (half_rings - i) / half_rings
        r = radius * np.cos(phi)
        z = half_length + radius * np.sin(phi)
        for j in range(n_segments):
            theta = 2 * np.pi * j / n_segments
            verts.append([r * np.cos(theta), r * np.sin(theta), z])

    # Cylinder body: top ring then bottom ring
    for z_cyl in [half_length, -half_length]:
        for j in range(n_segments):
            theta = 2 * np.pi * j / n_segments
            verts.append([radius * np.cos(theta), radius * np.sin(theta), z_cyl])

    # Bottom hemisphere: equator down to pole (z = -half_length .. -half_length-radius)
    for i in range(half_rings + 1):
        phi = (np.pi / 2) * i / half_rings
        r = radius * np.cos(phi)
        z = -half_length - radius * np.sin(phi)
        for j in range(n_segments):
            theta = 2 * np.pi * j / n_segments
            verts.append([r * np.cos(theta), r * np.sin(theta), z])

    verts_arr = np.array(verts, dtype=np.float32)

    # Generate faces by connecting adjacent rings
    n_total_rings = (n_rings // 2 + 1) + 2 + (n_rings // 2 + 1)
    for i in range(n_total_rings - 1):
        for j in range(n_segments):
            i0 = i * n_segments + j
            i1 = i * n_segments + (j + 1) % n_segments
            i2 = (i + 1) * n_segments + j
            i3 = (i + 1) * n_segments + (j + 1) % n_segments
            if i0 < len(verts_arr) and i3 < len(verts_arr):
                faces.append([i0, i2, i1])
                faces.append([i1, i2, i3])

    faces_arr = np.array(faces, dtype=np.uint32)
    # Filter out-of-bounds
    valid = np.all(faces_arr < len(verts_arr), axis=1)
    faces_arr = faces_arr[valid]

    return verts_arr, faces_arr


def sphere_mesh(
    radius: float, n_lat: int = 16, n_lon: int = 24
) -> tuple[np.ndarray, np.ndarray]:
    """Generate vertices and faces for a UV sphere."""
    verts = []
    # Top pole
    verts.append([0, 0, radius])
    for i in range(1, n_lat):
        phi = np.pi * i / n_lat
        r = radius * np.sin(phi)
        z = radius * np.cos(phi)
        for j in range(n_lon):
            theta = 2 * np.pi * j / n_lon
            verts.append([r * np.cos(theta), r * np.sin(theta), z])
    # Bottom pole
    verts.append([0, 0, -radius])

    verts_arr = np.array(verts, dtype=np.float32)
    faces = []

    # Top cap
    for j in range(n_lon):
        faces.append([0, 1 + j, 1 + (j + 1) % n_lon])

    # Body
    for i in range(n_lat - 2):
        for j in range(n_lon):
            i0 = 1 + i * n_lon + j
            i1 = 1 + i * n_lon + (j + 1) % n_lon
            i2 = 1 + (i + 1) * n_lon + j
            i3 = 1 + (i + 1) * n_lon + (j + 1) % n_lon
            faces.append([i0, i2, i1])
            faces.append([i1, i2, i3])

    # Bottom cap
    bottom = len(verts_arr) - 1
    base = 1 + (n_lat - 2) * n_lon
    for j in range(n_lon):
        faces.append([bottom, base + (j + 1) % n_lon, base + j])

    return verts_arr, np.array(faces, dtype=np.uint32)
