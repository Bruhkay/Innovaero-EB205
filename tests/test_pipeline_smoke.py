"""Whole chain with coarse numerics: checks that everything connects and produces sane numbers."""
import json

import numpy as np

from cabinsim.export import write_json, write_spectra_csv
from cabinsim.pipeline import run


def _coarse(params):
    p = params.copy()
    p.numerics.update(bands=[125, 250, 500, 1000, 2000, 4000], incidence_angles=10, azimuth_angles=3,
                      subfrequencies_per_band=1, tbl_wavenumbers=300, tbl_directions=8,
                      shunt_incidence_angles=[2, 1], bay_modes_axial=4, bay_modes_circumferential=2,
                      bay_grid=[14, 6], panel_length=1.0, panel_angle=12.0)
    p.active.update(duration=1.5, control_start=0.3, evaluation_window=0.4, impulse_response_taps=1024)
    return p


def test_pipeline_runs_end_to_end(params, tmp_path):
    results = run(_coarse(params), fast=True, log=lambda message: None)
    names = list(results["solutions"])
    assert names == list(params.solutions)
    for solution in results["solutions"].values():
        assert np.isfinite(solution["head"]["sil_db"]) and np.isfinite(solution["room"]["dba"])
        assert solution["head"]["sil_db"] >= solution["room"]["sil_db"]          # direct field only adds
    baseline, passive = results["solutions"]["V0_baseline"], results["solutions"]["V1_passive"]
    assert passive["added_mass_kg_m2"] > 0
    assert passive["head"]["dba"] < baseline["head"]["dba"]
    assert "active_trim_result" in results["solutions"]["V3_active"]

    write_json(results, tmp_path / "results.json")
    write_spectra_csv(results, tmp_path / "spectra.csv")
    assert json.loads((tmp_path / "results.json").read_text())["bands_hz"][0] == 125
