"""Closed-form frequencies and the single-wall mass law from the team archive (EB205.pdf pp.18-19)."""
from __future__ import annotations

import numpy as np

from ..materials import Plate


def longitudinal_wave_speed(youngs_modulus_pa: float, density_kg_m3: float, poisson_ratio: float) -> float:
    """c_L = sqrt(E / (rho (1 - nu^2)))   [DOC]  [m/s]  (plate longitudinal wave speed)."""
    return float(np.sqrt(youngs_modulus_pa / (density_kg_m3 * (1 - poisson_ratio**2))))


def ring_frequency(longitudinal_speed_m_s: float, radius_m: float) -> float:
    """f_r = c_L / (2 pi a)   [DOC]  [Hz]: below f_r the shell's curvature stiffens it."""
    return longitudinal_speed_m_s / (2 * np.pi * radius_m)


def coincidence_frequency(sound_speed_m_s: float, surface_mass_kg_m2: float, bending_stiffness_nm: float) -> float:
    """f_c = c^2 / (2 pi) sqrt(m'' / D)   [DOC]  [Hz]: bending wave speed equals the sound speed."""
    return sound_speed_m_s**2 / (2 * np.pi) * np.sqrt(surface_mass_kg_m2 / bending_stiffness_nm)


def bay_fundamental_frequency(bending_stiffness_nm: float, surface_mass_kg_m2: float,
                              length_a_m: float, length_b_m: float) -> float:
    """f_11 = (pi/2) sqrt(D / m'') (1/a^2 + 1/b^2)   [DOC]  [Hz]  (flat, unpressurised, simply supported)."""
    return np.pi / 2 * np.sqrt(bending_stiffness_nm / surface_mass_kg_m2) * (1 / length_a_m**2 + 1 / length_b_m**2)


def bay_mode_frequency(plate: Plate, length_axial_m: float, length_circ_m: float, mode_axial: int, mode_circ: int) -> float:
    """Natural frequency of a simply supported, curved, pressurised bay.   [LIT] Leissa (1973); Koval (1976)

    k_x = m pi / a (axial), k_y = n pi / b (circumferential), k^2 = k_x^2 + k_y^2
    w^2 m'' = D k^4 + N_x k_x^2 + N_y k_y^2 + (E t / R^2) (k_x / k)^4
    The last term is the Donnell curvature stiffness; it vanishes for a flat panel (R = inf).
    With N = 0 and R = inf this reduces to the [DOC] f_11 formula above.
    """
    kx = mode_axial * np.pi / length_axial_m
    ky = mode_circ * np.pi / length_circ_m
    k2 = kx**2 + ky**2
    stiffness = (plate.bending_stiffness_nm * k2**2
                 + plate.tension_axial_n_m * kx**2 + plate.tension_circ_n_m * ky**2
                 + plate.membrane_stiffness_n_m / plate.curvature_radius_m**2 * (kx**2 / k2) ** 2)
    return float(np.sqrt(stiffness / plate.surface_mass_kg_m2) / (2 * np.pi))


def normal_incidence_mass_law_db(frequency_hz: np.ndarray, surface_mass_kg_m2: float,
                                 impedance_1_rayl: float, impedance_2_rayl: float) -> np.ndarray:
    """TL0 = 10 log10[ ((Z1 + Z2)^2 + (w m'')^2) / (4 Z1 Z2) ]   [DOC]  two-medium limp wall."""
    omega = 2 * np.pi * np.asarray(frequency_hz, float)
    z1, z2 = impedance_1_rayl, impedance_2_rayl
    return 10 * np.log10(((z1 + z2) ** 2 + (omega * surface_mass_kg_m2) ** 2) / (4 * z1 * z2))


def mass_air_mass_frequency(mass_1_kg_m2: float, mass_2_kg_m2: float, gap_m: float,
                            gas_pressure_pa: float, heat_capacity_ratio: float = 1.4) -> float:
    """f0 = (1 / 2pi) sqrt( gamma p (m1 + m2) / (d m1 m2) )   [DOC]  double-wall resonance [Hz].

    `gas_pressure_pa` is the static pressure of the air in the cavity (cabin pressure).
    """
    stiffness = heat_capacity_ratio * gas_pressure_pa / gap_m
    return float(np.sqrt(stiffness * (mass_1_kg_m2 + mass_2_kg_m2) / (mass_1_kg_m2 * mass_2_kg_m2)) / (2 * np.pi))
