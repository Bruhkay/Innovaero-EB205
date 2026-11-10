"""Transfer matrix method (TMM) for the layered sidewall.

Each layer links (pressure, normal velocity) on its two faces:
    [p_in, v_in]^T = T [p_out, v_out]^T
for a wave with angular frequency w and trace (in-plane) wavenumber kt, time convention e^{+jwt}.
The layer matrices are multiplied from the skin (outside) to the trim panel (cabin side).
All functions are vectorised over kt (and psi), so a whole angle table is one call.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..atmosphere import Air
from ..materials import Plate, miki_equivalent_fluid

SMALL_WAVENUMBER = 1e-9   # avoids 0/0 at grazing incidence (numerical guard, not physics)


@dataclass(frozen=True)
class PlateLayer:
    plate: Plate
    name: str = "plate"


@dataclass(frozen=True)
class AirLayer:
    thickness_m: float
    name: str = "air gap"


@dataclass(frozen=True)
class PorousLayer:
    thickness_m: float
    flow_resistivity: float
    bulk_density_kg_m3: float
    name: str = "blanket"


Layer = PlateLayer | AirLayer | PorousLayer


def normal_wavenumber(wavenumber: complex | float, trace_wavenumber: np.ndarray) -> np.ndarray:
    """kz = sqrt(k^2 - kt^2) on the branch Im(kz) <= 0 (decaying wave for e^{+jwt})."""
    kz = np.sqrt(np.asarray(wavenumber, complex) ** 2 - np.asarray(trace_wavenumber, float) ** 2 + 0j)
    kz = np.where(np.imag(kz) > 0, -kz, kz)
    return np.where(np.abs(kz) < SMALL_WAVENUMBER, SMALL_WAVENUMBER, kz)


def fluid_impedance(omega: float, density: complex | float, sound_speed: complex | float,
                    trace_wavenumber: np.ndarray) -> np.ndarray:
    """Normal wave impedance of a fluid half-space, Z = rho w / kz  [Pa s/m].

    kt < k: real, equals rho c / cos(theta).  kt > k: imaginary (mass-like near field).
    """
    return density * omega / normal_wavenumber(omega / np.asarray(sound_speed), trace_wavenumber)


def plate_impedance(omega: float, trace_wavenumber: np.ndarray, azimuth_rad: np.ndarray | float, plate: Plate) -> np.ndarray:
    """Wave impedance of a thin plate, Z_p = p / v   [LIT] Koval (1976), Fahy & Gardonio (2007)

    Z_p = j w m'' - (j / w) [ D (1 + j eta) kt^4 + N_x kx^2 + N_y ky^2 + (E t / R^2)(1 + j eta) cos^4(psi) ]
    kx = kt cos(psi) along the axis, ky = kt sin(psi) around the fuselage.
    The cos^4(psi) term is the Donnell curvature stiffness (kx/kt)^4; it keeps its value at kt -> 0,
    which is the breathing (ring) mode of the shell. Flat plate: R = inf, N = 0.
    """
    kt = np.asarray(trace_wavenumber, float)
    cos_psi = np.cos(azimuth_rad)
    damping = 1 + 1j * plate.loss_factor
    stiffness = (plate.bending_stiffness_nm * damping * kt**4
                 + plate.tension_axial_n_m * (kt * cos_psi) ** 2
                 + plate.tension_circ_n_m * (kt * np.sin(azimuth_rad)) ** 2
                 + plate.membrane_stiffness_n_m / plate.curvature_radius_m**2 * damping * cos_psi**4)
    return 1j * omega * plate.surface_mass_kg_m2 - 1j * stiffness / omega


def _fluid_layer_matrix(omega: float, density, sound_speed, thickness_m: float, kt: np.ndarray):
    """Matrix of a fluid slab: [[cos(kz d), j Z sin(kz d)], [j sin(kz d) / Z, cos(kz d)]]."""
    kz = normal_wavenumber(omega / np.asarray(sound_speed), kt)
    impedance = density * omega / kz
    cos_term, sin_term = np.cos(kz * thickness_m), np.sin(kz * thickness_m)
    return cos_term, 1j * impedance * sin_term, 1j * sin_term / impedance, cos_term


def layer_matrix(layer: Layer, omega: float, kt: np.ndarray, azimuth_rad, cabin_air: Air):
    """2x2 transfer matrix of one layer, returned as its four entries (t00, t01, t10, t11)."""
    if isinstance(layer, PlateLayer):
        z_plate = plate_impedance(omega, kt, azimuth_rad, layer.plate)
        one = np.ones_like(z_plate)
        return one, z_plate, 0 * one, one
    if isinstance(layer, AirLayer):
        return _fluid_layer_matrix(omega, cabin_air.density_kg_m3, cabin_air.sound_speed_m_s, layer.thickness_m, kt)
    if isinstance(layer, PorousLayer):
        density, speed = miki_equivalent_fluid(omega / (2 * np.pi), layer.flow_resistivity,
                                               cabin_air.density_kg_m3, cabin_air.sound_speed_m_s)
        return _fluid_layer_matrix(omega, density, speed, layer.thickness_m, kt)
    raise TypeError(f"unknown layer type {type(layer)}")


def total_transfer_matrix(layers: list[Layer], omega: float, kt: np.ndarray, azimuth_rad, cabin_air: Air):
    """Product T = T_1 T_2 ... T_n, entries broadcast over kt and psi."""
    shape = np.broadcast(np.asarray(kt), np.asarray(azimuth_rad)).shape
    t00, t01 = np.ones(shape, complex), np.zeros(shape, complex)
    t10, t11 = np.zeros(shape, complex), np.ones(shape, complex)
    for layer in layers:
        m00, m01, m10, m11 = layer_matrix(layer, omega, kt, azimuth_rad, cabin_air)
        t00, t01, t10, t11 = (t00 * m00 + t01 * m10, t00 * m01 + t01 * m11,
                              t10 * m00 + t11 * m10, t10 * m01 + t11 * m11)
    return t00, t01, t10, t11


@dataclass(frozen=True)
class StackResponse:
    """Response of the stack to a blocked (wall) pressure p_b with trace wavenumber kt.

    a_coefficient          A, with p_incident = A v_cabin (so v_cabin / p_b = 1 / (2 A))
    skin_velocity_ratio    v_skin / p_b   [m/s/Pa]
    cabin_velocity_ratio   v_trim / p_b   [m/s/Pa]
    outside_impedance      Z1 (radiation impedance into the outside air)
    cabin_impedance        Z2 (radiation impedance into the cabin)
    """
    a_coefficient: np.ndarray
    skin_velocity_ratio: np.ndarray
    cabin_velocity_ratio: np.ndarray
    outside_impedance: np.ndarray
    cabin_impedance: np.ndarray


def stack_response(layers: list[Layer], omega: float, kt: np.ndarray, azimuth_rad,
                   outside_air: Air, cabin_air: Air, cabin_impedance: np.ndarray | None = None) -> StackResponse:
    """Solve the stack with radiation loads on both faces.

    Boundary conditions: cabin face p_out = Z2 v_out; outside face p_b = p_in + Z1 v_in.
    With v_out = 1:  p_in = T00 Z2 + T01,  v_in = T10 Z2 + T11,
                     2A = p_b = (T00 Z2 + T01) + Z1 (T10 Z2 + T11).
    For an incident plane wave p_b = 2 p_i (rigid-wall blocked pressure), which gives the usual TMM.
    `cabin_impedance` None -> infinite cabin-side plane (Z2 = rho2 w / kz2); otherwise e.g. the
    windowed rho2 c2 sigma of a finite panel.
    """
    t00, t01, t10, t11 = total_transfer_matrix(layers, omega, kt, azimuth_rad, cabin_air)
    z_out = fluid_impedance(omega, outside_air.density_kg_m3, outside_air.sound_speed_m_s, kt)
    if cabin_impedance is None:
        z_cab = fluid_impedance(omega, cabin_air.density_kg_m3, cabin_air.sound_speed_m_s, kt)
    else:
        z_cab = np.asarray(cabin_impedance, complex)
    v_in = t10 * z_cab + t11
    a_coeff = ((t00 * z_cab + t01) + z_out * v_in) / 2
    return StackResponse(a_coefficient=a_coeff, skin_velocity_ratio=v_in / (2 * a_coeff),
                         cabin_velocity_ratio=1 / (2 * a_coeff), outside_impedance=z_out, cabin_impedance=z_cab)


def incident_transmission(layers: list[Layer], frequency_hz: float, incidence_rad: np.ndarray, azimuth_rad,
                          outside_air: Air, cabin_air: Air, cabin_impedance: np.ndarray | None = None) -> np.ndarray:
    """Power transmission coefficient tau for an incident plane wave from outside.

    tau = W_transmitted / W_incident = Re(Z2) Z1 / |A|^2,   Z1 = rho1 c1 / cos(theta)
    kt = k1 sin(theta) is conserved through all layers (Snell).
    """
    omega = 2 * np.pi * frequency_hz
    kt = omega / outside_air.sound_speed_m_s * np.sin(incidence_rad)
    response = stack_response(layers, omega, kt, azimuth_rad, outside_air, cabin_air, cabin_impedance)
    z1 = np.real(response.outside_impedance)
    return np.real(response.cabin_impedance) * z1 / np.abs(response.a_coefficient) ** 2
