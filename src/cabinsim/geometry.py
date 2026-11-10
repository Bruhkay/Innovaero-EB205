"""Fuselage as a cylinder of small panels, and projection of a point source onto those panels.

Coordinates (see engine.yaml): x along the axis (positive aft), y starboard, z up.
Panel angle phi is measured from +y (starboard horizontal) towards +z (up).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# p0^2 / I0 = (20e-6)^2 / 1e-12 = 400 Pa s/m: the "400" in the Lp,ext formula   [LIT] ISO 1683
REFERENCE_IMPEDANCE_RAYL = 400.0


@dataclass(frozen=True)
class PanelGrid:
    """Panels on the fuselage cylinder above the cabin floor. Arrays have shape (n_x, n_phi)."""
    x_m: np.ndarray
    phi_rad: np.ndarray
    centres_m: np.ndarray        # (n_x, n_phi, 3) on the outer skin
    normals: np.ndarray          # (n_x, n_phi, 3) outward unit normals
    areas_m2: np.ndarray         # (n_x, n_phi)
    radius_m: float

    @property
    def total_area_m2(self) -> float:
        return float(self.areas_m2.sum())


def cylinder_panels(radius_m: float, x_range_m: tuple[float, float], floor_z_m: float,
                    panel_length_m: float, panel_angle_deg: float) -> PanelGrid:
    """Split the cylinder between x_range into panels; keep those whose centre is above the floor.

    centre = (x, a cos(phi), a sin(phi)),  normal = (0, cos(phi), sin(phi)),  area = dx * a * dphi
    """
    x_start, x_end = x_range_m
    x_centres = np.arange(x_start + panel_length_m / 2, x_end, panel_length_m)
    dphi = np.deg2rad(panel_angle_deg)
    phi_all = np.arange(-np.pi + dphi / 2, np.pi, dphi)
    phi_centres = phi_all[radius_m * np.sin(phi_all) > floor_z_m]
    # Shift the port-side lower panels by +360 deg so phi runs continuously from the starboard
    # floor line, over the ceiling, to the port floor line (no gap under the floor in plots).
    phi_centres = np.sort(np.where(phi_centres < -np.pi / 2, phi_centres + 2 * np.pi, phi_centres))
    x_grid, phi_grid = np.meshgrid(x_centres, phi_centres, indexing="ij")
    normals = np.stack([np.zeros_like(x_grid), np.cos(phi_grid), np.sin(phi_grid)], axis=-1)
    centres = np.stack([x_grid, radius_m * np.cos(phi_grid), radius_m * np.sin(phi_grid)], axis=-1)
    areas = np.full_like(x_grid, panel_length_m * radius_m * dphi)
    return PanelGrid(x_m=x_grid, phi_rad=phi_grid, centres_m=centres, normals=normals,
                     areas_m2=areas, radius_m=radius_m)


@dataclass(frozen=True)
class Projection:
    """Geometry of a point source seen from each panel (shapes (n_x, n_phi))."""
    distance_m: np.ndarray
    incidence_rad: np.ndarray    # theta, from cos(theta) = |u . n|
    azimuth_rad: np.ndarray      # psi: angle of the in-plane wave direction from the fuselage axis
    lit: np.ndarray              # True if the panel faces the source


def project_point_source(panels: PanelGrid, source_position_m: np.ndarray) -> Projection:
    """r = |x_f - r_s|,  u = (x_f - r_s)/r,  cos(theta) = |u . n|   [DOC] EB205 p.18

    psi: the trace (in-plane) part of u is t = u - (u.n) n; psi = angle between t and the axis,
    0 = wave runs along the fuselage, 90 deg = wave runs around it. Needed because the curved,
    pressurised skin is stiffer for some directions than others.
    """
    offset = panels.centres_m - np.asarray(source_position_m, float)
    distance = np.linalg.norm(offset, axis=-1)
    direction = offset / distance[..., None]
    u_dot_n = np.sum(direction * panels.normals, axis=-1)
    incidence = np.arccos(np.clip(np.abs(u_dot_n), 0.0, 1.0))
    tangential = direction - u_dot_n[..., None] * panels.normals
    tangential_norm = np.linalg.norm(tangential, axis=-1)
    axial_share = np.abs(tangential[..., 0]) / np.maximum(tangential_norm, 1e-12)
    azimuth = np.arccos(np.clip(axial_share, 0.0, 1.0))
    return Projection(distance_m=distance, incidence_rad=incidence, azimuth_rad=azimuth, lit=u_dot_n < 0)


def free_field_level_db(power_level_db: np.ndarray, distance_m: np.ndarray, impedance_rayl: float) -> np.ndarray:
    """Lp,ext = Lw - 10 log10(4 pi r^2) + 10 log10(Z_inf / 400)   [DOC] EB205 p.18

    Free-field incident level of a point source. Broadcasts: (..., 1) distance with (n_bands,) Lw.
    Z_inf = rho c of the outside air; 400 = p0^2/I0 makes Lp and the intensity level line up.
    """
    return (np.asarray(power_level_db)
            - 10 * np.log10(4 * np.pi * np.asarray(distance_m) ** 2)
            + 10 * np.log10(impedance_rayl / REFERENCE_IMPEDANCE_RAYL))


def head_position(radius_m: float, sidewall_depth_m: float, floor_z_m: float, seat_x_m: float,
                  head_height_m: float, head_distance_m: float, side: str) -> np.ndarray:
    """Head centre of a window-seat passenger.   [ASM]

    z = floor + head height; the trim surface is a circle of radius (a - d);
    the head sits `head_distance_m` horizontally inboard of it.
    """
    trim_radius = radius_m - sidewall_depth_m
    z = floor_z_m + head_height_m
    y_wall = np.sqrt(trim_radius**2 - z**2)
    sign = 1.0 if side == "right" else -1.0
    return np.array([seat_x_m, sign * (y_wall - head_distance_m), z])
