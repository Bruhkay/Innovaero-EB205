import numpy as np
import pytest

from cabinsim.geometry import cylinder_panels, free_field_level_db, project_point_source
from cabinsim.source import engine_spectrum, sound_power_level_db


def test_engine_sound_power_level_is_141_4_db():
    assert sound_power_level_db(138.0) == pytest.approx(141.4, abs=0.05)


def test_engine_spectrum_keeps_total_power(params):
    bands = np.asarray(params.numerics["bands"], float)
    spectrum = engine_spectrum(bands, params.engine)
    assert spectrum.total_power_w.sum() == pytest.approx(params.engine["acoustic_power"], rel=1e-9)
    assert 690.0 in np.round(spectrum.tone_frequencies_hz, 6)          # BPF = 18 x 2300 / 60


def test_projection_normal_incidence_on_facing_panel():
    panels = cylinder_panels(2.0, (-1.0, 1.0), -10.0, 0.5, 10.0)
    source = np.array([0.0, 50.0, 0.0])                               # far away on +y
    projection = project_point_source(panels, source)
    i_x = np.argmin(np.abs(panels.x_m[:, 0]))
    i_phi = np.argmin(np.abs(panels.phi_rad[0]))
    assert np.rad2deg(projection.incidence_rad[i_x, i_phi]) < 6.0
    assert projection.lit[i_x, i_phi]
    assert not projection.lit[i_x, np.argmin(np.abs(panels.phi_rad[0] - np.pi))]


def test_panel_areas_cover_the_cylinder():
    panels = cylinder_panels(2.0, (0.0, 3.0), -10.0, 0.25, 4.0)       # floor below the cylinder
    assert panels.total_area_m2 == pytest.approx(2 * np.pi * 2.0 * 3.0, rel=1e-6)


def test_free_field_level_spherical_spreading():
    near = free_field_level_db(141.4, 10.0, 400.0)
    far = free_field_level_db(141.4, 20.0, 400.0)
    assert near - far == pytest.approx(6.02, abs=0.01)
    assert near == pytest.approx(141.4 - 10 * np.log10(4 * np.pi * 100), abs=1e-9)
