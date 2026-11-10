"""Structure-borne (bridge) path: skin vibration reaches the trim panel through frames and
trim attachment points, bypassing the blanket and air gap."""
from __future__ import annotations

import numpy as np


def field_incidence_mass_law_db(frequency_hz: np.ndarray, surface_mass_kg_m2: float) -> np.ndarray:
    """TL_M = 20 log10(f m'') - 47   [LIT] field-incidence mass law, Sharp (1978)."""
    return 20 * np.log10(np.asarray(frequency_hz, float) * surface_mass_kg_m2) - 47


def point_bridge_tl_db(frequency_hz: np.ndarray, skin_mass_kg_m2: float, trim_mass_kg_m2: float,
                       skin_coincidence_hz: float, point_spacing_m: float, isolator_gain_db: float) -> np.ndarray:
    """Sharp's point-connected double wall, bridge-path transmission loss.   [LIT] Sharp (1978)

    TL_B = TL_M(m1 + m2) + 20 log10(e f_c1) + 20 log10(m1 / (m1 + m2)) - 45 + isolator gain
    e: spacing between point connections, f_c1: coincidence frequency of the skin.
    """
    total_mass = skin_mass_kg_m2 + trim_mass_kg_m2
    return (field_incidence_mass_law_db(frequency_hz, total_mass)
            + 20 * np.log10(point_spacing_m * skin_coincidence_hz)
            + 20 * np.log10(skin_mass_kg_m2 / total_mass) - 45 + isolator_gain_db)


def bridge_velocity_transfer(bridge_tau: np.ndarray, skin_mean_square_velocity_per_intensity: np.ndarray) -> np.ndarray:
    """Bridge power per unit area per unit skin mean-square velocity.   [ASM]

    Sharp's TL_B is defined for a diffuse incident field. The bridge is driven by the skin's motion,
    so we convert it into  T_b = tau_B / (<v_skin^2> / I_inc)  using the TMM skin velocity under the
    same diffuse field. Then for ANY excitation (engine at an angle, TBL):  W''_bridge = T_b <v_skin^2>.
    Units: (W/m^2) / (m^2/s^2).
    """
    return np.asarray(bridge_tau) / np.asarray(skin_mean_square_velocity_per_intensity)
