"""From power through the sidewall to sound pressure in the cabin.

Two parts are added energetically at the passenger's head:
  reverberant field  p^2_rev = 4 rho c W_total / R,   R = S alpha / (1 - alpha)      [LIT] Sabine / Hopkinson-Stryker
  direct field       p^2_dir = rho c sum_i W_i cos(phi_i) / (pi r_i^2)                [ASM] Lambertian panels
The room average uses the reverberant part only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .atmosphere import Air
from .geometry import PanelGrid


@dataclass(frozen=True)
class CabinRoom:
    """Cabin section above the floor, bounded by the trim surface (radius a - d)."""
    length_m: float
    trim_radius_m: float
    floor_z_m: float

    @property
    def half_angle_rad(self) -> float:
        """Half of the angle subtended by the trim arc above the floor, measured from the top (+z)."""
        return float(np.arccos(self.floor_z_m / self.trim_radius_m))

    @property
    def cross_section_m2(self) -> float:
        """Circular segment above the floor: r^2 (2 alpha - sin(2 alpha)) / 2, alpha = half angle."""
        alpha = self.half_angle_rad
        return float(self.trim_radius_m**2 * (2 * alpha - np.sin(2 * alpha)) / 2)

    @property
    def surface_m2(self) -> float:
        """Trim arc + floor + two end walls."""
        arc = 2 * self.half_angle_rad * self.trim_radius_m * self.length_m
        floor = 2 * np.sqrt(self.trim_radius_m**2 - self.floor_z_m**2) * self.length_m
        return float(arc + floor + 2 * self.cross_section_m2)

    @property
    def volume_m3(self) -> float:
        return self.cross_section_m2 * self.length_m


def room_constant_m2(surface_m2: float, absorption: np.ndarray) -> np.ndarray:
    """R = S alpha / (1 - alpha)  [m^2]   [LIT] Bies & Hansen, Engineering Noise Control."""
    alpha = np.asarray(absorption, float)
    return surface_m2 * alpha / (1 - alpha)


def absorption_per_band(bands_hz: np.ndarray, cabin_cfg: dict) -> np.ndarray:
    """Mean absorption coefficient, interpolated (log frequency) from the cabin.yaml support points."""
    return np.interp(np.log(bands_hz), np.log(cabin_cfg["absorption_frequencies"]), cabin_cfg["absorption_coefficients"])


def reverberant_mean_square(total_power_w: np.ndarray, room_constant: np.ndarray, air: Air) -> np.ndarray:
    """p^2_rev = 4 rho c W / R   [Pa^2]."""
    return 4 * air.impedance_rayl * np.asarray(total_power_w) / room_constant


@dataclass(frozen=True)
class DirectFieldWeights:
    """Geometric weight cos(phi)/(pi r^2) from each panel to the head [1/m^2], and which panels
    lie in front of the controlled trim panel."""
    weights_per_m2: np.ndarray       # (n_x, n_phi)
    controlled_panel: np.ndarray     # bool (n_x, n_phi)


def direct_field_weights(panels: PanelGrid, trim_radius_m: float, head_m: np.ndarray,
                         trim_panel_size_m: tuple[float, float]) -> DirectFieldWeights:
    """Lambertian weights cos(phi)/(pi r^2) from the trim surface panels to the head.   [ASM]

    Each skin panel's power is radiated from the trim surface right behind it (same x, phi).
    A surface element radiating W with a cosine pattern gives intensity W cos(phi) / (pi r^2).
    The controlled trim panel is the axial x circumferential patch in front of the head.
    """
    scale = trim_radius_m / panels.radius_m
    trim_points = panels.centres_m * np.array([1.0, scale, scale])
    inward = -panels.normals
    offset = np.asarray(head_m) - trim_points
    distance = np.linalg.norm(offset, axis=-1)
    cos_angle = np.sum(offset * inward, axis=-1) / distance
    weights = np.where(cos_angle > 0, cos_angle / (np.pi * distance**2), 0.0)

    head_phi = np.arctan2(head_m[2], head_m[1])
    arc_offset = trim_radius_m * np.angle(np.exp(1j * (panels.phi_rad - head_phi)))
    controlled = ((np.abs(panels.x_m - head_m[0]) <= trim_panel_size_m[0] / 2)
                  & (np.abs(arc_offset) <= trim_panel_size_m[1] / 2))
    return DirectFieldWeights(weights_per_m2=weights, controlled_panel=controlled)


def direct_mean_square(panel_power_w: np.ndarray, weights: np.ndarray, air: Air, mask: np.ndarray | None = None) -> np.ndarray:
    """p^2_dir = rho c sum_i W_i w_i  [Pa^2]; `panel_power_w` has shape (n_x, n_phi, n_bands)."""
    w = weights if mask is None else np.where(mask, weights, 0.0)
    return air.impedance_rayl * np.einsum("ijb,ij->b", panel_power_w, w)
