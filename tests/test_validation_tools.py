"""Sensitivity study, COMSOL coupon predictions, and a regression test for grazing-incidence panels."""
import csv
import json

import numpy as np
import pytest

from cabinsim.atmosphere import flight_condition
from cabinsim.coupons import coupon_bare_bay, coupon_layered_wall
from cabinsim.geometry import Projection
from cabinsim.pipeline import tau_on_panels
from cabinsim.sensitivity import VARIATIONS, markdown_table, run_study
from cabinsim.sidewall.transmission import TransmissionTables
from test_pipeline_smoke import _coarse


def test_tau_lookup_clips_grazing_angles():
    """A panel hit at 89.9 deg (outside the 0-89 deg table) must not crash the lookup."""
    incidence = np.deg2rad(np.linspace(0, 89, 5))
    azimuth = np.deg2rad(np.linspace(0, 90, 3))
    table = np.full((1, 5, 3), 1e-4)
    tables = TransmissionTables(np.array([500.0]), incidence, azimuth, table, np.ones_like(table), np.zeros(1))
    projection = Projection(distance_m=np.ones((1, 2)), incidence_rad=np.deg2rad([[45.0, 89.9]]),
                            azimuth_rad=np.zeros((1, 2)), lit=np.ones((1, 2), bool))
    np.testing.assert_allclose(tau_on_panels(tables, projection), 1e-4)


def test_sensitivity_study_runs_and_tabulates(params):
    variation = next(v for v in VARIATIONS if v.parameter == "Engine acoustic power")
    study = run_study(_coarse(params), [variation], log=lambda message: None)
    row = study["rows"][0]
    # doubling the engine power cannot make the cabin quieter, halving it cannot make it louder
    assert row["high_values"]["dba"] >= study["reference"]["dba"] >= row["low_values"]["dba"]
    assert "Engine acoustic power" in markdown_table(study)


def test_coupon_predictions_are_written(params, tmp_path):
    flight = flight_condition(params.flight)
    spec = coupon_bare_bay(params, flight, tmp_path)
    with open(tmp_path / "C1_bay_modes.csv", encoding="utf-8") as handle:
        first = next(csv.DictReader(handle))
    assert float(first["flat_hz"]) == pytest.approx(337.4, abs=0.5)        # archive f11 formula
    assert float(first["curved_pressurised_hz"]) > float(first["flat_hz"])
    assert spec["boundary"].startswith("simply supported")
    layered = coupon_layered_wall(params, flight, tmp_path)
    json.dumps(layered["layers"], default=str)
    assert (tmp_path / "C3_layered_wall_tl.csv").exists()
