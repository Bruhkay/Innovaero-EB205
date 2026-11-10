import numpy as np
import pytest

from cabinsim.materials import Plate, solid_plate
from cabinsim.sidewall import closed_form as cf
from cabinsim.sidewall.radiation import windowed_radiation_efficiency
from cabinsim.sidewall.stack import build_sidewall
from cabinsim.sidewall.tmm import PlateLayer, incident_transmission, plate_impedance
from cabinsim.sidewall.transmission import FinitePanel, diffuse_average, wavenumber_response


def test_tmm_single_plate_equals_archive_tl0(flight):
    plate = Plate(surface_mass_kg_m2=6.2, bending_stiffness_nm=350.0, loss_factor=0.01)
    frequencies = np.array([63.0, 250.0, 1000.0, 4000.0])
    tau = np.array([incident_transmission([PlateLayer(plate)], f, 0.0, 0.0, flight.outside, flight.cabin) for f in frequencies])
    expected = cf.normal_incidence_mass_law_db(frequencies, 6.2, flight.outside.impedance_rayl, flight.cabin.impedance_rayl)
    np.testing.assert_allclose(-10 * np.log10(tau), expected, atol=1e-6)


def test_mass_law_slope_is_6_db_per_octave():
    tl = cf.normal_incidence_mass_law_db(np.array([1000.0, 2000.0]), 10.0, 415.0, 415.0)
    assert tl[1] - tl[0] == pytest.approx(6.0, abs=0.1)


def test_f11_matches_archive_value(params):
    skin = solid_plate(params.materials["solids"]["CFRP_QI"], params.solutions["V0_baseline"]["skin"]["thickness"])
    f11 = cf.bay_fundamental_frequency(skin.bending_stiffness_nm, skin.surface_mass_kg_m2,
                                       params.fuselage["frame_pitch"], params.fuselage["stringer_pitch"])
    assert f11 == pytest.approx(335.0, rel=0.01)


def test_ring_and_coincidence_frequency_by_hand():
    e, rho, nu, t, radius = 60e9, 1550.0, 0.30, 0.004, 2.885
    c_l = np.sqrt(e / (rho * (1 - nu**2)))
    assert cf.longitudinal_wave_speed(e, rho, nu) == pytest.approx(6522.1, abs=0.5)
    assert cf.ring_frequency(c_l, radius) == pytest.approx(c_l / (2 * np.pi * radius))
    d = e * t**3 / (12 * (1 - nu**2))
    m = rho * t
    assert cf.coincidence_frequency(296.5, m, d) == pytest.approx(296.5**2 / (2 * np.pi) * np.sqrt(m / d))
    assert cf.coincidence_frequency(296.5, m, d) == pytest.approx(1858, abs=2)


def test_flat_unpressurised_mode_formula_reduces_to_archive_f11():
    plate = Plate(surface_mass_kg_m2=6.2, bending_stiffness_nm=351.6, loss_factor=0.01)
    assert cf.bay_mode_frequency(plate, 0.533, 0.2, 1, 1) == pytest.approx(
        cf.bay_fundamental_frequency(351.6, 6.2, 0.533, 0.2), rel=1e-12)


def test_curved_shell_breathing_mode_is_at_ring_frequency():
    """At kt = 0 along the axis, Z_p = 0 where w^2 m = E t / R^2 (the ring frequency with E, not E/(1-nu^2))."""
    plate = Plate(surface_mass_kg_m2=6.2, bending_stiffness_nm=351.6, loss_factor=0.0,
                  membrane_stiffness_n_m=60e9 * 0.004, curvature_radius_m=2.885)
    f_ring = np.sqrt(60e9 * 0.004 / (2.885**2 * 6.2)) / (2 * np.pi)
    below = plate_impedance(2 * np.pi * 0.99 * f_ring, 0.0, 0.0, plate)
    above = plate_impedance(2 * np.pi * 1.01 * f_ring, 0.0, 0.0, plate)
    assert np.imag(below) < 0 < np.imag(above)


def test_windowed_radiation_efficiency_limits():
    k = 2 * np.pi * 1000 / 344.0
    assert windowed_radiation_efficiency(k, 0.0, 0.0, 10.0, 10.0) == pytest.approx(1.0, abs=0.02)
    oblique = windowed_radiation_efficiency(k, k * np.sin(np.pi / 6), 0.0, 10.0, 10.0)
    assert oblique == pytest.approx(1 / np.cos(np.pi / 6), abs=0.02)
    k_low = 2 * np.pi * 50 / 344.0
    small = windowed_radiation_efficiency(k_low, 0.0, 0.0, 0.3, 0.3)
    assert small == pytest.approx(k_low**2 * 0.09 / (2 * np.pi), rel=0.02)     # baffled piston, low frequency


@pytest.mark.parametrize("frequency", [125.0, 1000.0])
def test_wavenumber_integral_reproduces_diffuse_transmission(params, flight, frequency):
    """A diffuse field pushed through the TBL wavenumber integral must give the same tau as the
    angle average. This checks all the factors of 2 and pi in the TBL path."""
    sidewall = build_sidewall("V0", params.solutions["V0_baseline"], params.materials, params.fuselage, flight)
    k1 = 2 * np.pi * frequency / flight.outside.sound_speed_m_s

    def diffuse(kx, ky):
        kt2 = kx**2 + ky**2
        return np.where(kt2 < k1**2, 1 / (2 * np.pi * k1 * np.sqrt(np.maximum(k1**2 - kt2, 1e-30))), 0.0)

    response = wavenumber_response(diffuse, frequency, sidewall, flight, None, k1, 2500, 48)
    tau_wavenumber = response.transmitted_per_ms * 8 * flight.outside.impedance_rayl
    theta = np.deg2rad(np.linspace(0, 89.9, 400))
    psi = np.deg2rad(np.linspace(0, 90, 25))
    theta_grid, psi_grid = np.meshgrid(theta, psi, indexing="ij")
    tau_angles = diffuse_average(incident_transmission(sidewall.layers, frequency, theta_grid, psi_grid,
                                                       flight.outside, flight.cabin), theta)
    assert 10 * np.log10(tau_wavenumber / tau_angles) == pytest.approx(0.0, abs=0.3)


def test_finite_panel_raises_low_frequency_tl(params, flight):
    """A small radiating panel radiates poorly at low frequency, so TL goes up."""
    sidewall = build_sidewall("V0", params.solutions["V0_baseline"], params.materials, params.fuselage, flight)
    panel = FinitePanel(1.066, 0.8)
    k2 = 2 * np.pi * 63.0 / flight.cabin.sound_speed_m_s
    z_finite = flight.cabin.impedance_rayl * windowed_radiation_efficiency(k2, 0.0, 0.0, panel.length_axial_m, panel.length_circ_m)
    tau_finite = incident_transmission(sidewall.layers, 63.0, 0.0, 0.0, flight.outside, flight.cabin, z_finite)
    tau_infinite = incident_transmission(sidewall.layers, 63.0, 0.0, 0.0, flight.outside, flight.cabin)
    assert tau_finite < tau_infinite
