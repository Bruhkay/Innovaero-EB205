import numpy as np
import pytest

from cabinsim.materials import Plate, piezo_constants
from cabinsim.piezo_shunt import Patch, ShuntedBay, patch_mass_per_area
from cabinsim.sidewall.closed_form import bay_mode_frequency

SKIN = Plate(surface_mass_kg_m2=6.2, bending_stiffness_nm=351.6, loss_factor=0.01, thickness_m=0.004)
A, B = 0.533, 0.2


@pytest.fixture(scope="module")
def single_patch_bay(params, flight):
    piezo = piezo_constants(params.materials["piezo"]["PZT5A"])
    patches = [Patch((0.5, 0.5), (0.16, 0.10), (1, 1))]
    return ShuntedBay(A, B, SKIN, patches, 0.25e-3, piezo, flight, 6, 3, (28, 12)), piezo, patches


def test_short_circuit_resonance_matches_analytic_mode(single_patch_bay):
    bay, piezo, patches = single_patch_bay
    added = patch_mass_per_area(patches, 0.25e-3, piezo, A * B)
    with_mass = SKIN.with_changes(surface_mass_kg_m2=SKIN.surface_mass_kg_m2 + added)
    # stiffness from the bare skin, mass incl. patches (as in the model)
    expected = bay_mode_frequency(SKIN, A, B, 1, 1) * np.sqrt(SKIN.surface_mass_kg_m2 / with_mass.surface_mass_kg_m2)
    assert bay.natural_frequencies_hz("short")[0] == pytest.approx(expected, rel=1e-9)


def test_open_circuit_raises_resonance_by_coupling_factor(single_patch_bay):
    bay, _, _ = single_patch_bay
    f_short = bay.natural_frequencies_hz("short")[0]
    k2 = bay.coupling_factor2[0, bay.target_index[0]]
    assert bay.natural_frequencies_hz("open")[0] == pytest.approx(f_short * np.sqrt(1 + k2), rel=3e-3)


def test_tuned_rl_shunt_improves_tl_at_target(single_patch_bay):
    bay, _, _ = single_patch_bay
    f_target = bay.tuning_omega[0] / (2 * np.pi)
    tau_short = bay.transmission_coefficient(f_target, 0.3, 0.2, "short")
    tau_rl = bay.transmission_coefficient(f_target, 0.3, 0.2, "RL")
    assert 10 * np.log10(tau_short / tau_rl) > 3.0


def test_rayleigh_radiation_efficiency_tends_to_one(params, flight):
    piezo = piezo_constants(params.materials["piezo"]["PZT5A"])
    bay = ShuntedBay(A, B, SKIN, [Patch((0.5, 0.5), (0.1, 0.1), (1, 1))], 0.25e-3, piezo, flight, 3, 2, (56, 24))
    low = bay.modal_radiation_efficiency(100.0, 0, flight.cabin)
    high = bay.modal_radiation_efficiency(5000.0, 0, flight.cabin)
    assert low < 0.1
    assert high == pytest.approx(1.0, abs=0.05)
