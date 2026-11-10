"""Data behind the demo pages: heat-map mesh, quiet-zone steady state, the exported bundle."""
import base64
import json

import numpy as np
import pytest

from cabinsim.active_control.fxlms import run_tonal_fxlms, tonal_secondary_estimates
from cabinsim.active_control.quiet_zone import steady_state_weights
from cabinsim.demo_export import build_demo_data, write_demo_data
from cabinsim.heatmap import fuselage_mesh, surface_levels, vertex_normals
from cabinsim.pipeline import run
from test_fxlms import _plant
from test_pipeline_smoke import _coarse


def test_fuselage_normals_point_outwards():
    mesh = fuselage_mesh(2.885, -27.0, 36.0, 1.0, 10.0)
    normals = vertex_normals(mesh)
    cylinder = np.abs(mesh.positions_m[:, 0]) < 10
    radial = mesh.positions_m[cylinder].copy()
    radial[:, 0] = 0
    assert np.all(np.sum(normals[cylinder] * radial, axis=1) > 0)


def test_engine_side_is_loudest():
    mesh = fuselage_mesh(2.885, -27.0, 36.0, 1.0, 10.0)
    levels = surface_levels(mesh, vertex_normals(mesh), [np.array([0.0, 9.8, -1.6])], np.array([100.0]),
                            np.array([500.0]), 112.6, 15.0)["overall"]
    loudest = mesh.positions_m[np.argmax(levels)]
    assert loudest[1] > 2.0 and abs(loudest[0]) < 3.0          # starboard side, near the fan plane


def test_quiet_zone_steady_state_matches_time_simulation():
    """The closed-form fixed point must equal where the sample-by-sample tonal FxLMS ends up."""
    fs, frequency, n = 2000, 115.0, 12000
    plant = _plant(2, 2)
    secondary = tonal_secondary_estimates(plant, [frequency], fs)[0]
    amplitude = np.array([1.0 + 0.4j, -0.3 + 0.8j])                  # complex disturbance at the two mics
    t = np.arange(n) / fs
    disturbance = np.real(amplitude[None, :] * np.exp(2j * np.pi * frequency * t)[:, None])
    step, regularisation, leakage = 0.004, 0.2, 1e-4
    result = run_tonal_fxlms(disturbance, plant, [frequency], fs, step, regularisation, leakage, start_sample=0)
    weights = steady_state_weights(secondary, amplitude, step, regularisation, leakage)
    predicted = np.sum(np.abs(amplitude + secondary @ weights) ** 2) / 2
    tail = result.error[-2000:]
    phasor = np.exp(-2j * np.pi * frequency * t[-2000:])
    simulated = np.sum(np.abs(2 * (tail * phasor[:, None]).mean(axis=0)) ** 2) / 2
    assert 10 * np.log10(simulated / predicted) == pytest.approx(0.0, abs=0.5)


def test_demo_bundle_is_complete(params, tmp_path):
    results = run(_coarse(params), fast=True, log=lambda message: None)
    data = build_demo_data(params, results, log=lambda message: None)
    assert {"fdtd", "heatmap", "bay", "quiet_zone", "summary"} <= set(data)
    fdtd = data["fdtd"]
    index = base64.b64decode(fdtd["solutions"][fdtd["baseline"]]["index_b64"])
    assert len(index) == fdtd["nx"] * fdtd["nz"]
    assert len(data["heatmap"]["levels"]["overall"]) == len(data["heatmap"]["positions"]) // 3
    write_demo_data(data, tmp_path / "demo_data.js")
    text = (tmp_path / "demo_data.js").read_text()
    assert json.loads(text[text.index("{"):text.rindex("}") + 1])["fdtd"]["nx"] == fdtd["nx"]
