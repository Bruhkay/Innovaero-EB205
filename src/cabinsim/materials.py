"""Material models: thin plates, fibrous blankets (Miki), piezoceramic coupling constants."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

VACUUM_PERMITTIVITY = 8.8541878128e-12   # F/m


@dataclass(frozen=True)
class Plate:
    """Thin plate (Kirchhoff) with optional membrane tension and curvature. SI units.

    surface_mass_kg_m2      m''
    bending_stiffness_nm    D
    loss_factor             eta (applies to the elastic terms)
    membrane_stiffness_n_m  E t, used by the curvature term (0 for flat panels)
    tension_axial_n_m       N_x, in-plane tension along the fuselage axis
    tension_circ_n_m        N_y, in-plane tension around the fuselage
    curvature_radius_m      R (inf for a flat panel)
    """
    surface_mass_kg_m2: float
    bending_stiffness_nm: float
    loss_factor: float
    thickness_m: float = 0.0
    membrane_stiffness_n_m: float = 0.0
    tension_axial_n_m: float = 0.0
    tension_circ_n_m: float = 0.0
    curvature_radius_m: float = np.inf

    def with_changes(self, **changes) -> "Plate":
        return replace(self, **changes)


def bending_stiffness(youngs_modulus_pa: float, thickness_m: float, poisson_ratio: float) -> float:
    """D = E t^3 / (12 (1 - nu^2))   [DOC] EB205 p.18  [N m]."""
    return youngs_modulus_pa * thickness_m**3 / (12 * (1 - poisson_ratio**2))


def solid_plate(solid_cfg: dict, thickness_m: float) -> Plate:
    """Plate made of one solid material from materials.yaml (m'' = rho t, D from E, t, nu)."""
    return Plate(surface_mass_kg_m2=solid_cfg["density"] * thickness_m,
                 bending_stiffness_nm=bending_stiffness(solid_cfg["youngs_modulus"], thickness_m, solid_cfg["poisson_ratio"]),
                 loss_factor=solid_cfg["loss_factor"],
                 thickness_m=thickness_m,
                 membrane_stiffness_n_m=solid_cfg["youngs_modulus"] * thickness_m)


def given_plate(panel_cfg: dict) -> Plate:
    """Plate given directly by m'' and D (e.g. honeycomb sandwich trim panel)."""
    return Plate(surface_mass_kg_m2=panel_cfg["surface_mass"], bending_stiffness_nm=panel_cfg["bending_stiffness"],
                 loss_factor=panel_cfg["loss_factor"], thickness_m=panel_cfg["thickness"])


def pressurised_cylinder(plate: Plate, radius_m: float, pressure_difference_pa: float,
                         include_tension: bool) -> Plate:
    """Add curvature and cabin-pressure tension to a skin plate.   [LIT] Koval (1976), JASA 59(6)

    Thin-walled cylinder membrane forces:  N_hoop = dp R,  N_axial = dp R / 2
    """
    tension_circ = pressure_difference_pa * radius_m if include_tension else 0.0
    tension_axial = tension_circ / 2
    return plate.with_changes(curvature_radius_m=radius_m, tension_axial_n_m=tension_axial,
                              tension_circ_n_m=tension_circ)


def miki_equivalent_fluid(frequency_hz: np.ndarray | float, flow_resistivity: float,
                          air_density: float, air_sound_speed: float) -> tuple[np.ndarray, np.ndarray]:
    """Delany-Bazley-Miki model of a fibrous blanket as an equivalent fluid.   [LIT] Miki (1990)

    X  = rho0 f / sigma
    Zc = rho0 c0 [1 + 0.0699 X^-0.632 - j 0.1071 X^-0.632]
    k  = (w / c0) [1 + 0.1093 X^-0.618 - j 0.1597 X^-0.618]
    returns rho_eq = Zc k / w and c_eq = w / k (complex), sign convention e^{+jwt}.
    Valid for roughly 0.01 < f/sigma < 1 (SI); below that it is an extrapolation.
    """
    f = np.asarray(frequency_hz, float)
    omega = 2 * np.pi * f
    x = air_density * f / flow_resistivity
    impedance = air_density * air_sound_speed * (1 + 0.0699 * x**-0.632 - 1j * 0.1071 * x**-0.632)
    wavenumber = (omega / air_sound_speed) * (1 + 0.1093 * x**-0.618 - 1j * 0.1597 * x**-0.618)
    return impedance * wavenumber / omega, omega / wavenumber


@dataclass(frozen=True)
class PiezoConstants:
    """Thin-patch (plane stress) piezo constants derived from datasheet values."""
    density_kg_m3: float
    e31_effective_c_m2: float
    permittivity_effective_f_m: float


def piezo_constants(piezo_cfg: dict) -> PiezoConstants:
    """Plane-stress constants of a thin PZT patch.   [LIT] e.g. Preumont, Vibration Control of Active Structures (2011)

    e31,eff = d31 E / (1 - nu)
    eps_eff = eps33^T - 2 d31^2 E / (1 - nu)   (capacitance of a patch whose in-plane strain is blocked)
    """
    youngs = piezo_cfg["youngs_modulus"]
    nu = piezo_cfg["poisson_ratio"]
    d31 = piezo_cfg["d31"]
    e31 = d31 * youngs / (1 - nu)
    eps_free = piezo_cfg["relative_permittivity_free"] * VACUUM_PERMITTIVITY
    eps_blocked = eps_free - 2 * d31**2 * youngs / (1 - nu)
    return PiezoConstants(density_kg_m3=piezo_cfg["density"], e31_effective_c_m2=e31,
                          permittivity_effective_f_m=eps_blocked)
