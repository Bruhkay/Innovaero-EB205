"""Radiation efficiency of a wave pattern on an infinite surface and on a finite (windowed) panel.

A plate vibrating as v(x, y) = v0 exp(-j(kx x + ky y)) radiates power
    W = 1/2 rho c sigma |v0|^2 A
sigma is the radiation efficiency. For an infinite plate sigma = k / sqrt(k^2 - kt^2) when the
trace wavenumber kt is below the acoustic wavenumber k, and zero above it. A finite panel also
radiates for kt > k (edge radiation) and does not blow up near grazing.
"""
from __future__ import annotations

import numpy as np


def infinite_radiation_efficiency(wavenumber: float, trace_wavenumber: np.ndarray) -> np.ndarray:
    """sigma_inf = k / sqrt(k^2 - kt^2) for kt < k, else 0   [LIT] Fahy & Gardonio (2007)."""
    kt = np.asarray(trace_wavenumber, float)
    radicand = wavenumber**2 - kt**2
    return np.where(radicand > 0, wavenumber / np.sqrt(np.maximum(radicand, 1e-300)), 0.0)


def _quadrature_sizes(wavenumber: float, length_x_m: float, length_y_m: float,
                      minimum: tuple[int, int]) -> tuple[int, int]:
    """Enough points to resolve the sinc^2 lobes (width 2 pi / L) in the angular integral."""
    kl = wavenumber * max(length_x_m, length_y_m)
    return max(minimum[0], int(np.ceil(1.2 * kl))), max(minimum[1], int(np.ceil(4.5 * kl)))


def windowed_radiation_efficiency(wavenumber: float, kx: np.ndarray, ky: np.ndarray,
                                  length_x_m: float, length_y_m: float,
                                  minimum_points: tuple[int, int] = (40, 80),
                                  chunk_size: int = 512) -> np.ndarray:
    """Radiation efficiency of a baffled a x b panel carrying a forced wave (kx, ky).

    [LIT] Villot, Guigou & Gagliardini (2001), J. Sound Vib. 245(3) - "spatial windowing".
    sigma(kx, ky) = 1/(4 pi^2 a b) * integral over |alpha, beta| < k of
                    k / sqrt(k^2 - alpha^2 - beta^2) * |W(alpha - kx, beta - ky)|^2
    with |W|^2 = a^2 b^2 sinc^2((alpha - kx) a / 2) sinc^2((beta - ky) b / 2).
    Substituting alpha = k sin(t) cos(c), beta = k sin(t) sin(c) removes the square-root singularity:
    sigma = (k^2 a b / (4 pi^2)) int_0^{pi/2} int_0^{2 pi} sin(t) sinc_a^2 sinc_b^2 dc dt
    Midpoint rule in t and c.  Limits: sigma -> k / kz for a large panel; -> k^2 A / (2 pi) for a small one.
    """
    kx_all, ky_all = np.broadcast_arrays(np.asarray(kx, float), np.asarray(ky, float))
    output_shape = kx_all.shape
    kx_flat, ky_flat = kx_all.ravel(), ky_all.ravel()
    n_polar, n_azimuth = _quadrature_sizes(wavenumber, length_x_m, length_y_m, minimum_points)
    polar = (np.arange(n_polar) + 0.5) * (np.pi / 2) / n_polar
    azimuth = (np.arange(n_azimuth) + 0.5) * 2 * np.pi / n_azimuth
    weight = (np.pi / 2 / n_polar) * (2 * np.pi / n_azimuth)
    sin_t = np.sin(polar)[:, None]
    alpha = (wavenumber * sin_t * np.cos(azimuth)[None, :]).ravel()
    beta = (wavenumber * sin_t * np.sin(azimuth)[None, :]).ravel()
    jacobian = np.repeat(np.sin(polar), n_azimuth)
    prefactor = wavenumber**2 * length_x_m * length_y_m / (4 * np.pi**2) * weight

    result = np.empty(kx_flat.size)
    for start in range(0, kx_flat.size, chunk_size):
        stop = start + chunk_size
        # np.sinc(x) = sin(pi x)/(pi x), so divide the argument by pi
        sinc_x = np.sinc((alpha[None, :] - kx_flat[start:stop, None]) * length_x_m / (2 * np.pi))
        sinc_y = np.sinc((beta[None, :] - ky_flat[start:stop, None]) * length_y_m / (2 * np.pi))
        result[start:stop] = prefactor * np.sum(jacobian[None, :] * sinc_x**2 * sinc_y**2, axis=1)
    return result.reshape(output_shape)
