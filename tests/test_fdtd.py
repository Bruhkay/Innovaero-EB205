"""FDTD checks: free-field wave speed, plate as a limp mass, exactness of the layer stretching,
and that the browser core (demo/js/fdtd_core.js) computes exactly what the Python solver does."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from cabinsim.fdtd import (FdtdModel, FdtdSolver, FluidMaterial, homogeneous_model, limp_plate, material_table,
                           stretched_air, tone_source)
from cabinsim.sidewall.closed_form import normal_incidence_mass_law_db

ROOT = Path(__file__).resolve().parents[1]

OUTSIDE = FluidMaterial("outside", 0.38, 0.38 * 296.5**2)
CABIN = FluidMaterial("cabin", 0.96, 0.96 * 344.0**2)


def test_free_field_wave_speed_equals_c():
    model = homogeneous_model(OUTSIDE, (241, 241), 0.02, sponge_cells=20)
    model.source_cell = (120, 120)
    solver = FdtdSolver(model)
    near, far = (160, 120), (200, 120)                      # 0.8 m and 1.6 m from the source
    t0, width = 1.5e-3, 3e-4
    near_signal, far_signal = [], []
    for _ in range(int(9e-3 / solver.dt)):
        pulse = np.exp(-((solver.time_s - t0) / width) ** 2)
        solver.step(pulse)
        near_signal.append(solver.p[near])
        far_signal.append(solver.p[far])
    lags = np.arange(-len(near_signal) + 1, len(near_signal))
    delay = lags[np.argmax(np.correlate(far_signal, near_signal, mode="full"))] * solver.dt
    assert 0.8 / delay == pytest.approx(296.5, rel=0.02)


def _plane_wave_run(layers: list[tuple[FluidMaterial, int]], frequency: float, dx: float = 0.01) -> float:
    """1D run: outside air | layers | cabin air. Returns the transmitted rms pressure at a probe."""
    materials = [OUTSIDE, CABIN] + [m for m, _ in layers]
    index = np.zeros((1400, 1), np.uint8)
    start = 600
    for k, (_, cells) in enumerate(layers):
        index[start:start + cells] = 2 + k
        start += cells
    index[start:] = 1
    if not layers:
        index[:] = 0
    model = FdtdModel(cell_size_m=dx, y0_m=0.0, z0_m=0.0, material_index=index, materials=materials, sponge_cells=150,
                      source_cell=(300, 0))
    solver = FdtdSolver(model)
    probe = 1000
    record = []
    steps = int(0.06 / solver.dt)
    for n in range(steps):
        solver.step(tone_source(solver.time_s, frequency))
        if n > steps - int(0.015 / solver.dt):
            record.append(solver.p[probe, 0])
    return float(np.sqrt(np.mean(np.square(record))))


def test_plate_cell_follows_two_medium_mass_law():
    """A dense 2-cell ring with rho t = m'' transmits like the archive's TL0 (limp mass)."""
    frequency, surface_mass, dx = 400.0, 6.2, 0.01
    incident = _plane_wave_run([], frequency, dx)
    plate, thickness = limp_plate("skin", surface_mass, 2 * dx, 500.0)
    transmitted = _plane_wave_run([(plate, 2)], frequency, dx)
    z1, z2 = OUTSIDE.density_kg_m3 * OUTSIDE.sound_speed_m_s, CABIN.density_kg_m3 * CABIN.sound_speed_m_s
    tau = (transmitted**2 / z2) / (incident**2 / z1)
    expected = normal_incidence_mass_law_db(np.array([frequency]), surface_mass, z1, z2)[0]
    assert -10 * np.log10(tau) == pytest.approx(expected, abs=0.2)


def test_stretched_blanket_transmits_like_the_real_one():
    """Drawing a layer e times thicker with rho/e, kappa*e, sigma/e must not change transmission."""
    real, _ = stretched_air("blanket", 0.96, 0.96 * 344.0**2, 0.08, 1.0, 12000.0)
    wide, _ = stretched_air("blanket", 0.96, 0.96 * 344.0**2, 0.08, 2.5, 12000.0)
    thin_run = _plane_wave_run([(real, 8)], 300.0)
    wide_run = _plane_wave_run([(wide, 20)], 300.0)
    assert 20 * np.log10(wide_run / thin_run) == pytest.approx(0.0, abs=0.05)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
def test_browser_core_matches_python(tmp_path):
    """Same model, same source, same number of steps: JS and Python fields must agree to round-off."""
    index = np.zeros((60, 40), np.uint8)
    index[30:33, :] = 1                         # dense plate
    index[33:40, :] = 2                         # porous layer
    index[45:47, 5:35] = 3                      # rigid block
    materials = [OUTSIDE, FluidMaterial("plate", 100.0, 100.0 * 500.0**2),
                 FluidMaterial("blanket", 0.4, 0.96 * 344.0**2 * 2.5, 5000.0), FluidMaterial("floor", 1.0, 0.0, rigid=True)]
    model = FdtdModel(cell_size_m=0.02, y0_m=0.0, z0_m=0.0, material_index=index, materials=materials,
                      sponge_cells=8, source_cell=(15, 20))
    steps, frequency, amplitude = 300, 400.0, 1.0e4
    solver = FdtdSolver(model)
    for _ in range(steps):
        solver.step(amplitude * tone_source(solver.time_s, frequency))

    spec = dict(nx=60, nz=40, dx=0.02, sponge_cells=8, source_cell=[15, 20], materials=material_table(model),
                index=index.ravel().tolist(), steps=steps, frequency=frequency, amplitude=amplitude)
    (tmp_path / "model.json").write_text(json.dumps(spec))
    output = subprocess.run(["node", str(ROOT / "tests" / "js_fdtd_runner.js"), str(ROOT / "demo" / "js" / "fdtd_core.js"),
                             str(tmp_path / "model.json")], capture_output=True, text=True, check=True).stdout
    browser = json.loads(output)
    assert browser["dt"] == pytest.approx(solver.dt, rel=1e-12)
    js_field = np.array(browser["p"]).reshape(60, 40)
    assert np.max(np.abs(js_field - solver.p)) <= 1e-9 * np.max(np.abs(solver.p))
