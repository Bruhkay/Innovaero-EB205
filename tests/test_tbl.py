import numpy as np
import pytest

from cabinsim.tbl import BoundaryLayer, band_mean_square_pa2, corcos_wavenumber_spectrum, mellen_wavenumber_spectrum


def _integrate_plane(spectrum, kc):
    kx = np.linspace(-60 * kc, 60 * kc, 4001)
    ky = np.linspace(-60 * kc, 60 * kc, 4001)
    grid_x, grid_y = np.meshgrid(kx, ky, indexing="ij")
    values = spectrum(grid_x, grid_y, kc, 0.11, 0.77)
    return np.trapezoid(np.trapezoid(values, ky, axis=1), kx)


@pytest.mark.parametrize("spectrum", [corcos_wavenumber_spectrum, mellen_wavenumber_spectrum])
def test_wavenumber_spectra_are_normalised(spectrum):
    # Lorentzian/Cauchy tails decay slowly, so a finite window captures ~97-99 %
    assert _integrate_plane(spectrum, 10.0) == pytest.approx(1.0, abs=0.04)


def test_mellen_is_lower_than_corcos_at_low_wavenumber():
    kc = 20.0
    ratio_db = 10 * np.log10(mellen_wavenumber_spectrum(0.0, 0.0, kc, 0.11, 0.77)
                             / corcos_wavenumber_spectrum(0.0, 0.0, kc, 0.11, 0.77))
    assert ratio_db == pytest.approx(10 * np.log10(np.pi * 0.11 / 2), abs=0.2)     # about -7.6 dB


def test_goody_rms_pressure_is_plausible():
    """At cruise the TBL rms wall pressure is of order 0.5 % of the dynamic pressure."""
    layer = BoundaryLayer(15.5, 0.21, 252.0, 0.38, 3.77e-5, 176.0)
    bands = np.array([50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250,
                      1600, 2000, 2500, 3150, 4000, 5000], float)
    p_rms = np.sqrt(band_mean_square_pa2(bands, layer).sum())
    dynamic_pressure = 0.5 * 0.38 * 252.0**2
    assert 0.002 < p_rms / dynamic_pressure < 0.01
