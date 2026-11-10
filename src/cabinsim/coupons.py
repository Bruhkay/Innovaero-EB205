"""Model predictions for the COMSOL coupon comparisons:  python -m cabinsim.coupons [--out out/coupons]

Each coupon is a small geometry the team can build in COMSOL (node limit) with exactly the same
assumptions as one part of this model. This module writes the model's answer as CSV, plus
`coupons_spec.json` with every dimension and material value, so both sides use the same inputs.
See docs/validation.md for the COMSOL set-up and the acceptance criteria.

C1  bare skin bay: natural frequencies (flat / curved / curved + cabin pressure) and TL at normal incidence
C2  bay with one PZT patch: short/open frequencies, coupling k^2, TL short vs RL shunt
C3  layered sidewall, infinite and flat (COMSOL: Floquet-periodic unit cell): TL at 0, 30, 60 degrees
C4  active trim panel: secondary path (actuator 1 -> head mic) and primary path (cavity -> head)
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .atmosphere import FlightCondition, flight_condition
from .materials import piezo_constants
from .params import Params, load_params
from .piezo_shunt import Patch, ShuntedBay
from .pipeline import active_trim_geometry
from .sidewall.closed_form import bay_mode_frequency
from .sidewall.stack import build_sidewall
from .sidewall.tmm import PlateLayer, incident_transmission

FREQUENCIES_HZ = np.round(np.geomspace(50, 2000, 120), 2)


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _flat(plate):
    return plate.with_changes(curvature_radius_m=np.inf, tension_axial_n_m=0.0, tension_circ_n_m=0.0)


def coupon_bare_bay(params: Params, flight: FlightCondition, folder: Path) -> dict:
    """C1: bay modes and normal-incidence TL of the bare skin bay (no patches)."""
    sidewall = build_sidewall("V0", params.solutions["V0_baseline"], params.materials, params.fuselage, flight)
    skin = sidewall.skin
    a, b = params.fuselage["frame_pitch"], params.fuselage["stringer_pitch"]
    unpressurised = skin.with_changes(tension_axial_n_m=0.0, tension_circ_n_m=0.0)
    modes = sorted(((i, j) for i in range(1, 6) for j in range(1, 3)), key=lambda m: bay_mode_frequency(skin, a, b, *m))
    _write_csv(folder / "C1_bay_modes.csv", ["mode_axial", "mode_circ", "flat_hz", "curved_hz", "curved_pressurised_hz"],
               [[i, j, round(bay_mode_frequency(_flat(skin), a, b, i, j), 2), round(bay_mode_frequency(unpressurised, a, b, i, j), 2),
                 round(bay_mode_frequency(skin, a, b, i, j), 2)] for i, j in modes])
    piezo = piezo_constants(params.materials["piezo"][params.piezo["material"]])
    num = params.numerics
    rows = []
    for name, plate in (("flat", _flat(skin)), ("pressurised", skin)):
        bay = ShuntedBay(a, b, plate, [], params.piezo["patch_thickness"], piezo, flight, num["bay_modes_axial"],
                         num["bay_modes_circumferential"], tuple(num["bay_grid"]))
        rows.append([-10 * np.log10(bay.transmission_coefficient(f, 0.0, 0.0, "short")) for f in FREQUENCIES_HZ])
    _write_csv(folder / "C1_bay_tl_normal.csv", ["frequency_hz", "tl_flat_db", "tl_pressurised_db"],
               [[f, round(rows[0][k], 2), round(rows[1][k], 2)] for k, f in enumerate(FREQUENCIES_HZ)])
    return dict(length_axial_m=a, length_circ_m=b, thickness_m=skin.thickness_m,
                surface_mass_kg_m2=skin.surface_mass_kg_m2, bending_stiffness_nm=skin.bending_stiffness_nm,
                loss_factor=skin.loss_factor, curvature_radius_m=skin.curvature_radius_m,
                tension_axial_n_m=skin.tension_axial_n_m, tension_circ_n_m=skin.tension_circ_n_m,
                boundary="simply supported on all four edges", excitation="unit plane wave, normal incidence, outside air")


def coupon_single_patch(params: Params, flight: FlightCondition, folder: Path) -> dict:
    """C2: one PZT patch on the flat bay, tuned RL shunt on mode (1,1)."""
    sidewall = build_sidewall("V0", params.solutions["V0_baseline"], params.materials, params.fuselage, flight)
    skin = _flat(sidewall.skin)
    a, b = params.fuselage["frame_pitch"], params.fuselage["stringer_pitch"]
    piezo = piezo_constants(params.materials["piezo"][params.piezo["material"]])
    first = params.piezo["patches"][0]
    patch = Patch(tuple(first["centre"]), tuple(first["size"]), tuple(first["target_mode"]))
    num = params.numerics
    bay = ShuntedBay(a, b, skin, [patch], params.piezo["patch_thickness"], piezo, flight, num["bay_modes_axial"],
                     num["bay_modes_circumferential"], tuple(num["bay_grid"]))
    short, open_ = bay.natural_frequencies_hz("short")[:4], bay.natural_frequencies_hz("open")[:4]
    _write_csv(folder / "C2_patch_frequencies.csv", ["mode_rank", "short_circuit_hz", "open_circuit_hz"],
               [[k + 1, round(s, 2), round(o, 2)] for k, (s, o) in enumerate(zip(short, open_))])
    target = bay.tuning_omega[0] / (2 * np.pi)
    band = FREQUENCIES_HZ[(FREQUENCIES_HZ > 0.6 * target) & (FREQUENCIES_HZ < 1.6 * target)]
    _write_csv(folder / "C2_patch_tl_normal.csv", ["frequency_hz", "tl_short_db", "tl_rl_db"],
               [[f, round(-10 * np.log10(bay.transmission_coefficient(f, 0.0, 0.0, "short")), 2),
                 round(-10 * np.log10(bay.transmission_coefficient(f, 0.0, 0.0, "RL")), 2)] for f in band])
    summary = bay.summary()[0]
    return dict(patch_centre_fraction=list(patch.centre_fraction), patch_size_m=list(patch.size_m),
                patch_thickness_m=params.piezo["patch_thickness"], piezo_material=params.piezo["material"],
                e31_effective_c_m2=piezo.e31_effective_c_m2, permittivity_effective_f_m=piezo.permittivity_effective_f_m,
                capacitance_nf=summary["capacitance_nf"], coupling_k2=summary["coupling_k2"],
                rl_inductance_h=summary["inductance_h"], rl_tuned_hz=summary["tuned_hz"],
                skin="flat, unpressurised (as C1 flat)")


def coupon_layered_wall(params: Params, flight: FlightCondition, folder: Path) -> dict:
    """C3: infinite flat layered wall (TMM), V0 and V1 build-ups, three angles."""
    angles = [0.0, 30.0, 60.0]
    spec = {}
    columns, data = ["frequency_hz"], []
    walls = {}
    for name in ("V0_baseline", "V1_passive"):
        sidewall = build_sidewall(name, params.solutions[name], params.materials, params.fuselage, flight)
        layers = [replace(layer, plate=_flat(layer.plate)) if isinstance(layer, PlateLayer) else layer
                  for layer in sidewall.layers]
        walls[name] = layers
        spec[name] = [dict(type=type(layer).__name__, **asdict(layer)) for layer in layers]
        columns += [f"tl_{name}_{int(angle)}deg_db" for angle in angles]
    for f in FREQUENCIES_HZ:
        row = [f]
        for layers in walls.values():
            for angle in angles:
                tau = incident_transmission(layers, f, np.deg2rad(angle), 0.0, flight.outside, flight.cabin)
                row.append(round(float(-10 * np.log10(max(tau, 1e-15))), 2))
        data.append(row)
    _write_csv(folder / "C3_layered_wall_tl.csv", columns, data)
    return dict(layers=spec, note="flat, unpressurised, infinite; outside air on the incident side, cabin air behind",
                outside_air=vars(flight.outside), cabin_air=vars(flight.cabin))


def coupon_active_panel(params: Params, flight: FlightCondition, folder: Path) -> dict | None:
    """C4: trim-panel plant, free field: actuator 1 -> head-centre point, and cavity pressure -> head."""
    name = next((n for n, s in params.solutions.items() if s["active_trim"]), None)
    if name is None:
        return None
    geometry = active_trim_geometry(params, flight, name)
    frequencies = FREQUENCIES_HZ[FREQUENCIES_HZ <= 1000]
    primary, secondary = geometry.plant.frequency_responses(frequencies, geometry.head_m[None, :])
    _write_csv(folder / "C4_active_panel_frf.csv",
               ["frequency_hz", "secondary_mag_pa_per_v", "secondary_phase_deg", "primary_mag_pa_per_pa", "primary_phase_deg"],
               [[f, f"{abs(secondary[k, 0, 0]):.4e}", round(float(np.angle(secondary[k, 0, 0], deg=True)), 1),
                 f"{abs(primary[k, 0]):.4e}", round(float(np.angle(primary[k, 0], deg=True)), 1)]
                for k, f in enumerate(frequencies)])
    plant = geometry.plant
    return dict(panel_size_m=[plant.a, plant.b], head_point_m=geometry.head_m.tolist(),
                actuator_1_centre_fraction=params.active["actuator_positions"][0],
                actuator_size_m=params.active["actuator_size"], panel_loss_factor=plant.loss_factor,
                primary_incidence_deg=params.active["primary_incidence_angle"],
                boundary="simply supported panel in an infinite rigid baffle, free field on the cabin side")


def _json_safe(value):
    """Infinite radius (flat plate) -> text, numpy -> plain types, so the spec is strict JSON."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return "infinite (flat)" if np.isinf(value) else float(value)
    return value


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write model predictions for the COMSOL coupons")
    parser.add_argument("--out", type=Path, default=Path("out/coupons"))
    parser.add_argument("--config-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    params = load_params(args.config_dir)
    flight = flight_condition(params.flight)
    spec = dict(C1=coupon_bare_bay(params, flight, args.out), C2=coupon_single_patch(params, flight, args.out),
                C3=coupon_layered_wall(params, flight, args.out), C4=coupon_active_panel(params, flight, args.out))
    (args.out / "coupons_spec.json").write_text(json.dumps(_json_safe(spec), indent=1, allow_nan=False), encoding="utf-8")
    print(f"written: {sorted(p.name for p in args.out.iterdir())}")


if __name__ == "__main__":
    main()
