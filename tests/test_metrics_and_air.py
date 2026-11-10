import numpy as np
import pytest

from cabinsim import metrics
from cabinsim.atmosphere import isa_temperature_pressure, air_state


def test_two_equal_levels_add_3_db():
    assert metrics.db_sum([80.0, 80.0]) == pytest.approx(83.01, abs=0.01)


def test_one_pascal_is_94_db():
    assert metrics.pressure_level_db(1.0) == pytest.approx(93.98, abs=0.01)


def test_a_weighting_reference_values():
    # IEC 61672-1 table values
    assert metrics.a_weighting_db(1000.0) == pytest.approx(0.0, abs=0.05)
    assert metrics.a_weighting_db(100.0) == pytest.approx(-19.1, abs=0.1)
    assert metrics.a_weighting_db(4000.0) == pytest.approx(1.0, abs=0.1)


def test_sil_is_mean_of_four_octaves():
    bands = np.array([400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000], float)
    levels = np.full(len(bands), 50.0)
    # each octave holds three equal 1/3-octave bands -> 50 + 10 log10(3)
    assert metrics.speech_interference_level(bands, levels) == pytest.approx(50 + 10 * np.log10(3), abs=0.01)


def test_isa_tropopause_and_sea_level():
    t, p = isa_temperature_pressure(11000.0)
    assert t == pytest.approx(216.65, abs=0.01)
    assert p == pytest.approx(22632, rel=1e-3)
    sea = air_state(*isa_temperature_pressure(0.0))
    assert sea.density_kg_m3 == pytest.approx(1.225, abs=1e-3)
    assert sea.sound_speed_m_s == pytest.approx(340.3, abs=0.1)


def test_cruise_condition(flight):
    assert flight.outside.sound_speed_m_s == pytest.approx(296.5, abs=0.2)
    assert flight.outside.density_kg_m3 == pytest.approx(0.380, abs=0.002)
    assert flight.flight_speed_m_s == pytest.approx(252.0, abs=0.5)
    assert flight.pressure_difference_pa > 50e3      # cabin pressure pushes the skin outwards
