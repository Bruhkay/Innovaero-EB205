"""Turbulent boundary layer (TBL) wall-pressure models.

Two pieces describe the TBL pressure on the skin:
  1. the point spectrum Phi_pp(f): how much mean-square pressure there is at each frequency (Goody);
  2. the normalised wavenumber spectrum phi(kx, ky): how that pressure is spread over wavelengths.
     Most TBL energy travels with the flow at U_c (convective peak at kx = w / U_c), far above the
     acoustic wavenumber, which is why the TBL couples into the cabin much more weakly than sound.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import band_edges_hz


@dataclass(frozen=True)
class BoundaryLayer:
    """Boundary-layer state at the cabin station. SI units."""
    wall_shear_stress_pa: float
    thickness_m: float
    free_stream_velocity_m_s: float
    density_kg_m3: float
    kinematic_viscosity_m2_s: float
    convection_velocity_m_s: float

    @property
    def friction_velocity_m_s(self) -> float:
        """u_tau = sqrt(tau_w / rho)."""
        return float(np.sqrt(self.wall_shear_stress_pa / self.density_kg_m3))

    @property
    def timescale_ratio(self) -> float:
        """R_T = (delta / U) / (nu / u_tau^2): outer-to-inner timescale ratio used by Goody."""
        u_tau = self.friction_velocity_m_s
        return (self.thickness_m / self.free_stream_velocity_m_s) / (self.kinematic_viscosity_m2_s / u_tau**2)


def goody_spectrum_per_hz(frequency_hz: np.ndarray, layer: BoundaryLayer) -> np.ndarray:
    """One-sided wall-pressure spectrum Phi_pp(f) [Pa^2/Hz].   [LIT] Goody (2004), AIAA J. 42(9)

    Phi(w) U / (tau_w^2 delta) = 3.0 (w delta/U)^2 / [ ((w delta/U)^0.75 + 0.5)^3.7 + (1.1 R_T^-0.57 w delta/U)^7 ]
    Phi(f) = 2 pi Phi(w)
    """
    omega = 2 * np.pi * np.asarray(frequency_hz, float)
    u, delta, tau = layer.free_stream_velocity_m_s, layer.thickness_m, layer.wall_shear_stress_pa
    x = omega * delta / u
    shape = 3.0 * x**2 / ((x**0.75 + 0.5) ** 3.7 + (1.1 * layer.timescale_ratio ** -0.57 * x) ** 7)
    return 2 * np.pi * shape * tau**2 * delta / u


def band_mean_square_pa2(bands_hz: np.ndarray, layer: BoundaryLayer, points_per_band: int = 64) -> np.ndarray:
    """Mean-square TBL pressure in each 1/3-octave band: integral of Phi_pp(f) df  [Pa^2]."""
    lower, upper = band_edges_hz(bands_hz)
    result = np.zeros(len(lower))
    for i, (f_lo, f_hi) in enumerate(zip(lower, upper)):
        f = np.geomspace(f_lo, f_hi, points_per_band)
        result[i] = np.trapezoid(goody_spectrum_per_hz(f, layer), f)
    return result


def corcos_wavenumber_spectrum(kx: np.ndarray, ky: np.ndarray, convective_wavenumber: float,
                               alpha_x: float, alpha_y: float) -> np.ndarray:
    """Normalised Corcos spectrum, integral over (kx, ky) = 1   [1/(rad/m)^2]   [LIT] Corcos (1964)

    phi = [a_x / (pi (a_x^2 + (kx - kc)^2))] [a_y / (pi (a_y^2 + ky^2))],  a_x = alpha_x kc, a_y = alpha_y kc
    kx along the flow (fuselage axis), kc = w / U_c.
    """
    ax, ay = alpha_x * convective_wavenumber, alpha_y * convective_wavenumber
    return (ax / (np.pi * (ax**2 + (kx - convective_wavenumber) ** 2))) * (ay / (np.pi * (ay**2 + ky**2)))


def mellen_wavenumber_spectrum(kx: np.ndarray, ky: np.ndarray, convective_wavenumber: float,
                               alpha_x: float, alpha_y: float) -> np.ndarray:
    """Normalised Mellen (elliptic Corcos) spectrum, integral = 1   [LIT] Mellen (1990), JASA 88(6)

    phi = 1 / (2 pi a_x a_y) [1 + ((kx - kc)/a_x)^2 + (ky/a_y)^2]^(-3/2)
    Same peak and widths as Corcos, but about 8 dB lower at low wavenumbers (kx, ky << kc),
    where the Corcos model is known to over-predict.
    """
    ax, ay = alpha_x * convective_wavenumber, alpha_y * convective_wavenumber
    radius2 = 1 + ((kx - convective_wavenumber) / ax) ** 2 + (ky / ay) ** 2
    return radius2 ** -1.5 / (2 * np.pi * ax * ay)


def wavenumber_spectrum(model: str):
    """Pick the TBL wavenumber model by name ('mellen' or 'corcos')."""
    models = {"mellen": mellen_wavenumber_spectrum, "corcos": corcos_wavenumber_spectrum}
    if model not in models:
        raise ValueError(f"unknown TBL wavenumber model {model!r}; choose from {list(models)}")
    return models[model]
