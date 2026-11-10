"""Engine noise on the outside of the aircraft, per mesh vertex, for the 3D heat map.

The projection works on ANY triangle mesh (positions + triangles): each vertex gets its distance
and its cos(theta) = |u . n| to every engine, exactly as for the cylinder panels (geometry.py).
`fuselage_mesh` builds a simplified 787-9 body; `load_obj` reads a real mesh (e.g. a 787 model
exported as .obj in the same coordinate system) to replace it later.
Not modelled: shielding by the wing or nacelle, diffraction, flight effects.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import metrics
from .geometry import free_field_level_db

OCTAVES_HZ = (63, 125, 250, 500, 1000, 2000, 4000)


@dataclass(frozen=True)
class Mesh:
    positions_m: np.ndarray     # (n_vertices, 3), cabinsim axes: x aft, y starboard, z up
    triangles: np.ndarray       # (n_triangles, 3) vertex indices


def body_of_revolution(stations_x: np.ndarray, radii: np.ndarray, centre_z: np.ndarray, n_around: int) -> Mesh:
    """Mesh of rings (one per x station) joined by triangles; ring k has radius radii[k] around (x, 0, z_k)."""
    angles = np.linspace(0, 2 * np.pi, n_around, endpoint=False)
    positions = np.array([[x, r * np.cos(a), z + r * np.sin(a)]
                          for x, r, z in zip(stations_x, radii, centre_z) for a in angles])
    triangles = []
    for ring in range(len(stations_x) - 1):
        for k in range(n_around):
            a = ring * n_around + k
            b = ring * n_around + (k + 1) % n_around
            c, d = a + n_around, b + n_around
            triangles += [(a, b, c), (b, d, c)]          # counter-clockwise seen from outside
    return Mesh(positions_m=positions, triangles=np.array(triangles, dtype=np.int32))


def fuselage_mesh(radius_m: float, x_nose_m: float, x_tail_m: float, axial_step_m: float, angle_step_deg: float) -> Mesh:
    """Simplified 787-9 fuselage: elliptic nose (6 m), cylinder, tapering upswept tail cone (16 m).   [ASM]"""
    nose_length, tail_length = 6.0, 16.0
    stations = np.arange(x_nose_m, x_tail_m + axial_step_m / 2, axial_step_m)
    radii, centre = np.full_like(stations, radius_m), np.zeros_like(stations)
    in_nose = stations < x_nose_m + nose_length
    s = (stations[in_nose] - (x_nose_m + nose_length)) / nose_length          # -1 at the tip, 0 at the cylinder
    radii[in_nose] = radius_m * np.sqrt(np.maximum(1 - s**2, 0.0004))
    centre[in_nose] = -0.3 * radius_m * s**2                                  # nose droops slightly
    in_tail = stations > x_tail_m - tail_length
    t = (stations[in_tail] - (x_tail_m - tail_length)) / tail_length          # 0 at the cylinder, 1 at the tail
    radii[in_tail] = radius_m - (radius_m - 0.45) * t
    centre[in_tail] = 1.4 * t**1.5                                            # upsweep
    return body_of_revolution(stations, radii, centre, int(round(360 / angle_step_deg)))


def load_obj(path: Path) -> Mesh:
    """Read vertices and faces of a Wavefront .obj file (polygons are split into triangles)."""
    positions, triangles = [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v":
            positions.append([float(v) for v in parts[1:4]])
        elif parts[0] == "f":
            corners = [int(p.split("/")[0]) - 1 for p in parts[1:]]
            triangles += [(corners[0], corners[k], corners[k + 1]) for k in range(1, len(corners) - 1)]
    return Mesh(positions_m=np.array(positions, float), triangles=np.array(triangles, dtype=np.int32))


def vertex_normals(mesh: Mesh) -> np.ndarray:
    """Outward unit normals: sum of the (area-weighted) normals of the triangles around each vertex.

    Triangles must be ordered counter-clockwise seen from outside (as in fuselage_mesh).
    """
    p = mesh.positions_m
    a, b, c = p[mesh.triangles[:, 0]], p[mesh.triangles[:, 1]], p[mesh.triangles[:, 2]]
    face_normals = np.cross(b - a, c - a)
    normals = np.zeros_like(p)
    for corner in range(3):
        np.add.at(normals, mesh.triangles[:, corner], face_normals)
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)


def surface_levels(mesh: Mesh, normals: np.ndarray, engine_positions: list[np.ndarray], band_power_w: np.ndarray,
                   bands_hz: np.ndarray, outside_impedance: float, shadow_loss_db: float) -> dict[str, np.ndarray]:
    """Incident free-field level at every vertex, summed over engines.   [DOC] projection formulas

    Lp,ext = Lw - 10 log10(4 pi r^2) + 10 log10(Z/400), minus `shadow_loss_db` where u . n > 0.
    Returns {'overall': OASPL, '63': octave level, ...} in dB re 20 uPa, one value per vertex.
    """
    band_level = metrics.power_level_db(np.maximum(band_power_w, 1e-30))
    mean_square = np.zeros((len(mesh.positions_m), len(bands_hz)))
    for source in engine_positions:
        offset = mesh.positions_m - source
        distance = np.linalg.norm(offset, axis=1)
        facing = np.sum(offset * normals, axis=1) < 0
        level = free_field_level_db(band_level[None, :], distance[:, None], outside_impedance)
        level -= np.where(facing, 0.0, shadow_loss_db)[:, None]
        mean_square += metrics.mean_square_from_level(level)
    levels_db = metrics.pressure_level_db(np.sqrt(mean_square))
    result = {"overall": metrics.db_sum(levels_db, axis=1)}
    for octave in OCTAVES_HZ:
        inside = (bands_hz > octave / 2**0.5) & (bands_hz < octave * 2**0.5)
        if inside.any():
            result[str(octave)] = metrics.db_sum(levels_db[:, inside], axis=1)
    return result
