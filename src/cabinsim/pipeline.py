"""End-to-end calculation: engine + TBL -> sidewall -> cabin -> passenger's head, for every solution.

Order of work:
  1. shared set-up (air, source spectrum, fuselage panels, projection, TBL, cabin geometry)
  2. per solution: sidewall layers -> transmission tables -> TBL path -> (shunt) -> cabin levels
  3. for solutions with an active trim panel: FxLMS simulation scaled to that solution's head levels
The result is a plain dict, ready for JSON.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from . import metrics
from .active_control.simulation import AncGeometry, build_geometry, simulate_broadband, simulate_tonal
from .atmosphere import FlightCondition, flight_condition
from .cabin import (CabinRoom, DirectFieldWeights, absorption_per_band, direct_field_weights, direct_mean_square,
                    reverberant_mean_square, room_constant_m2)
from .geometry import (PanelGrid, Projection, cylinder_panels, free_field_level_db, head_position,
                       project_point_source)
from .materials import given_plate, piezo_constants
from .params import Params
from .piezo_shunt import ShuntedBay, patch_mass_per_area, patches_from_config, shunt_delta_tl_db
from .sidewall import closed_form as cf
from .sidewall.stack import Sidewall, build_sidewall
from .sidewall.transmission import (FinitePanel, TransmissionTables, incidence_tables, tbl_transmission)
from .source import EngineSpectrum, band_index_of, engine_positions, engine_spectrum
from .tbl import BoundaryLayer, band_mean_square_pa2, wavenumber_spectrum

PATHS = ("engine_broadband", "engine_tonal", "tbl")


@dataclass
class Shared:
    """Everything that does not depend on the chosen solution."""
    params: Params
    fast: bool
    flight: FlightCondition
    bands_hz: np.ndarray
    engine: EngineSpectrum
    panels: PanelGrid
    projections: list[Projection]
    boundary_layer: BoundaryLayer
    tbl_mean_square_pa2: np.ndarray
    room: CabinRoom
    room_constant_m2: np.ndarray
    head_m: np.ndarray
    direct: DirectFieldWeights
    finite_panel: FinitePanel
    shunt_cache: dict = field(default_factory=dict)


def build_shared(params: Params, fast: bool = False) -> Shared:
    flight = flight_condition(params.flight)
    bands = np.asarray(params.numerics["bands"], float)
    fus, cab = params.fuselage, params.cabin
    panels = cylinder_panels(fus["radius"], tuple(fus["cabin_section_x"]), fus["floor_height"],
                             params.numerics["panel_length"], params.numerics["panel_angle"])
    projections = [project_point_source(panels, position) for position in engine_positions(params.engine)]
    tbl = params.tbl
    layer = BoundaryLayer(wall_shear_stress_pa=tbl["wall_shear_stress"], thickness_m=tbl["boundary_layer_thickness"],
                          free_stream_velocity_m_s=flight.flight_speed_m_s, density_kg_m3=flight.outside.density_kg_m3,
                          kinematic_viscosity_m2_s=flight.outside.kinematic_viscosity_m2_s,
                          convection_velocity_m_s=tbl["convection_velocity_ratio"] * flight.flight_speed_m_s)
    trim_radius = fus["radius"] - fus["sidewall_depth"]
    room = CabinRoom(length_m=fus["cabin_section_x"][1] - fus["cabin_section_x"][0], trim_radius_m=trim_radius,
                     floor_z_m=fus["floor_height"])
    head = head_position(fus["radius"], fus["sidewall_depth"], fus["floor_height"], cab["seat_x"],
                         cab["head_height_above_floor"], cab["head_distance_from_trim"], cab["seat_side"])
    return Shared(params=params, fast=fast, flight=flight, bands_hz=bands,
                  engine=engine_spectrum(bands, params.engine), panels=panels, projections=projections,
                  boundary_layer=layer, tbl_mean_square_pa2=band_mean_square_pa2(bands, layer), room=room,
                  room_constant_m2=room_constant_m2(room.surface_m2, absorption_per_band(bands, cab)),
                  head_m=head, direct=direct_field_weights(panels, trim_radius, head, tuple(fus["trim_panel_size"])),
                  finite_panel=FinitePanel(*fus["trim_panel_size"], tuple(params.numerics["window_quadrature"])))


# ------------------------------------------------------------------ engine path on every panel
def incident_mean_square(shared: Shared, projection: Projection, band_power_w: np.ndarray) -> np.ndarray:
    """Free-field p^2 of one engine at each panel (shadowed panels lowered), shape (n_x, n_phi, n_bands)."""
    outside = shared.flight.outside
    level = free_field_level_db(metrics.power_level_db(np.maximum(band_power_w, 1e-30))[None, None, :],
                                projection.distance_m[..., None], outside.impedance_rayl)
    level = level - np.where(projection.lit, 0.0, shared.params.engine["shadow_loss"])[..., None]
    return metrics.mean_square_from_level(level)


def incident_power_per_area(shared: Shared, projection: Projection, band_power_w: np.ndarray) -> np.ndarray:
    """W''_inc = p^2 cos(theta) / (rho1 c1): incident power per unit panel area [W/m^2]."""
    mean_square = incident_mean_square(shared, projection, band_power_w)
    return mean_square * np.cos(projection.incidence_rad)[..., None] / shared.flight.outside.impedance_rayl


def tau_on_panels(tables: TransmissionTables, projection: Projection) -> np.ndarray:
    """Look up tau_total(band, theta, psi) for every panel (log interpolation), shape (n_x, n_phi, n_bands).

    Angles are clipped to the table range: a panel hit at grazing incidence (theta > 89 deg) uses the
    89 deg value, where the incident power per area cos(theta) is already almost zero.
    """
    theta = np.clip(projection.incidence_rad.ravel(), tables.incidence_rad[0], tables.incidence_rad[-1])
    psi = np.clip(projection.azimuth_rad.ravel(), tables.azimuth_rad[0], tables.azimuth_rad[-1])
    points = np.stack([theta, psi], axis=-1)
    log_tau = np.log10(np.maximum(tables.tau_total, 1e-30))
    result = np.empty((points.shape[0], len(tables.bands_hz)))
    for b in range(len(tables.bands_hz)):
        interpolator = RegularGridInterpolator((tables.incidence_rad, tables.azimuth_rad), log_tau[b])
        result[:, b] = 10 ** interpolator(points)
    return result.reshape(projection.incidence_rad.shape + (len(tables.bands_hz),))


# ------------------------------------------------------------------ shunt
def shunted_bay(params: Params, flight: FlightCondition, sidewall: Sidewall) -> ShuntedBay:
    """Skin bay of this sidewall with the configured patches (the bay model adds the patch mass itself)."""
    num = params.numerics
    piezo = piezo_constants(params.materials["piezo"][params.piezo["material"]])
    skin_without_patches = sidewall.skin.with_changes(
        surface_mass_kg_m2=sidewall.skin.surface_mass_kg_m2 - sidewall.added_piezo_mass_kg_m2)
    tuning_plate = skin_without_patches
    if params.piezo["tuning_condition"] == "ground":
        tuning_plate = skin_without_patches.with_changes(tension_axial_n_m=0.0, tension_circ_n_m=0.0)
    return ShuntedBay(params.fuselage["frame_pitch"], params.fuselage["stringer_pitch"], skin_without_patches,
                      patches_from_config(params.piezo), params.piezo["patch_thickness"], piezo, flight,
                      num["bay_modes_axial"], num["bay_modes_circumferential"], tuple(num["bay_grid"]),
                      tuning_plate=tuning_plate)


def shunt_effect(shared: Shared, sidewall: Sidewall) -> tuple[np.ndarray, list[dict]]:
    """Band dTL of the shunted patches on this skin (cached: V2 and V3 share the same skin)."""
    key = (sidewall.skin.surface_mass_kg_m2, sidewall.skin.loss_factor)
    if key in shared.shunt_cache:
        return shared.shunt_cache[key]
    p, num = shared.params, shared.params.numerics
    bay = shunted_bay(p, shared.flight, sidewall)
    n_sub = 3 if shared.fast else 7
    delta = shunt_delta_tl_db(bay, shared.bands_hz, p.piezo["shunt_type"], num["shunt_max_frequency"], n_sub,
                              *num["shunt_incidence_angles"], p.piezo["negative_capacitance_ratio"])
    shared.shunt_cache[key] = (delta, bay.summary())
    return shared.shunt_cache[key]


def piezo_mass_per_area(params: Params) -> float:
    piezo = piezo_constants(params.materials["piezo"][params.piezo["material"]])
    bay_area = params.fuselage["frame_pitch"] * params.fuselage["stringer_pitch"]
    return patch_mass_per_area(patches_from_config(params.piezo), params.piezo["patch_thickness"], piezo, bay_area)


# ------------------------------------------------------------------ levels
def level_summary(bands_hz: np.ndarray, mean_square_by_path: dict[str, np.ndarray]) -> dict:
    """Band levels per path and total, plus SIL, dB(A) and OASPL of the total."""
    total = sum(mean_square_by_path.values())
    total_db = metrics.pressure_level_db(np.sqrt(np.maximum(total, 1e-30)))
    return dict(paths_db={k: metrics.pressure_level_db(np.sqrt(np.maximum(v, 1e-30))).tolist()
                          for k, v in mean_square_by_path.items()},
                total_db=total_db.tolist(),
                sil_db=metrics.speech_interference_level(bands_hz, total_db),
                dba=metrics.a_weighted_level(bands_hz, total_db),
                oaspl_db=metrics.overall_level(total_db))


def evaluate_solution(shared: Shared, name: str, log: Callable[[str], None]) -> dict:
    p, num = shared.params, shared.params.numerics
    solution = p.solutions[name]
    sidewall = build_sidewall(name, solution, p.materials, p.fuselage, shared.flight, piezo_mass_per_area(p))
    n_sub = 2 if shared.fast else num["subfrequencies_per_band"]
    tables = incidence_tables(shared.bands_hz, sidewall, shared.flight, shared.finite_panel, p.bridge,
                              num["incidence_angles"], num["azimuth_angles"], n_sub,
                              num["bridge_incidence_angles"])

    shunt_delta, shunt_rows = np.zeros(len(shared.bands_hz)), []
    if solution["piezo_shunt"]:
        log(f"  {name}: shunted bay model")
        shunt_delta, shunt_rows = shunt_effect(shared, sidewall)
    shunt_factor = 10 ** (-shunt_delta / 10)

    log(f"  {name}: TBL wavenumber integral")
    alpha_x, alpha_y = p.tbl["corcos_alpha_streamwise"], p.tbl["corcos_alpha_spanwise"]
    n_k = num["tbl_wavenumbers"] // 2 if shared.fast else num["tbl_wavenumbers"]
    tbl_models = [p.tbl["wavenumber_model"]]
    if not shared.fast:
        tbl_models += [m for m in ("mellen", "corcos") if m != p.tbl["wavenumber_model"]]
    tbl_results = {model: tbl_transmission(shared.bands_hz, sidewall, shared.flight, shared.boundary_layer,
                                           shared.tbl_mean_square_pa2, wavenumber_spectrum(model), alpha_x, alpha_y,
                                           shared.finite_panel, tables, n_k, num["tbl_directions"], n_sub)
                   for model in tbl_models}
    tbl_main = tbl_results[p.tbl["wavenumber_model"]]

    # transmitted power per panel and band [W]
    areas = shared.panels.areas_m2[..., None]
    panel_power = {path: np.zeros(shared.panels.areas_m2.shape + (len(shared.bands_hz),)) for path in PATHS}
    for projection in shared.projections:
        tau = tau_on_panels(tables, projection) * shunt_factor
        panel_power["engine_broadband"] += incident_power_per_area(shared, projection, shared.engine.broadband_power_w) * tau * areas
        panel_power["engine_tonal"] += incident_power_per_area(shared, projection, shared.engine.tonal_power_w) * tau * areas
    if p.tbl["enabled"]:
        panel_power["tbl"] += tbl_main.transmitted_total_w_m2 * shunt_factor * areas

    cabin_air = shared.flight.cabin
    reverberant = {k: reverberant_mean_square(v.sum(axis=(0, 1)), shared.room_constant_m2, cabin_air) for k, v in panel_power.items()}
    direct = {k: direct_mean_square(v, shared.direct.weights_per_m2, cabin_air) for k, v in panel_power.items()}
    controlled = {k: direct_mean_square(v, shared.direct.weights_per_m2, cabin_air, shared.direct.controlled_panel)
                  for k, v in panel_power.items()}
    head = {k: reverberant[k] + direct[k] for k in PATHS}

    total_power = {k: v.sum(axis=(0, 1)) for k, v in panel_power.items()}
    result = dict(
        label=solution["label"], description=solution["description"],
        piezo_shunt=solution["piezo_shunt"], active_trim=solution["active_trim"],
        layers=[dict(name=layer.name, type=type(layer).__name__) for layer in sidewall.layers],
        surface_mass_kg_m2=sidewall.surface_mass_kg_m2,
        tl_diffuse_db=dict(air=tables.diffuse_tl_db("air").tolist(), bridge=tables.diffuse_tl_db("bridge").tolist(),
                           total=tables.diffuse_tl_db("total").tolist()),
        shunt_delta_tl_db=shunt_delta.tolist(), shunt_patches=shunt_rows,
        tbl_coupling_penalty_db={m: r.coupling_penalty_db.tolist() for m, r in tbl_results.items()},
        transmitted_power_level_db={k: metrics.power_level_db(np.maximum(v, 1e-30)).tolist() for k, v in total_power.items()},
        room=level_summary(shared.bands_hz, reverberant),
        head=level_summary(shared.bands_hz, head),
        cabin_power_map_db=metrics.power_level_db(np.maximum(sum(panel_power.values()).sum(axis=-1), 1e-30)
                                                  / shared.panels.areas_m2).tolist(),
    )
    if len(tbl_results) > 1:
        other_model = [m for m in tbl_results if m != p.tbl["wavenumber_model"]][0]
        other_power = tbl_results[other_model].transmitted_total_w_m2 * shunt_factor * shared.panels.total_area_m2
        alt_head = dict(head)
        alt_head["tbl"] = head["tbl"] * other_power / np.maximum(total_power["tbl"], 1e-30)
        result["head_tbl_alternative"] = dict(model=other_model, **level_summary(shared.bands_hz, alt_head))
    result["_mean_square"] = dict(head=head, controlled=controlled)       # internal, removed before export
    return result


# ------------------------------------------------------------------ active trim panel
def tone_shares(shared: Shared, controlled_orders: list[int]) -> tuple[list[float], list[int], list[float]]:
    """For each controlled shaft order: tone frequency, its band, and its share of that band's tonal power."""
    engine = shared.engine
    frequencies, band_indices, shares = [], [], []
    for order in controlled_orders:
        frequency = order * engine.shaft_frequency_hz
        band = band_index_of(frequency, shared.bands_hz)
        if band is None:
            continue
        own = engine.tone_power_w[np.isclose(engine.tone_frequencies_hz, frequency)].sum()
        frequencies.append(float(frequency))
        band_indices.append(band)
        shares.append(float(own / engine.tonal_power_w[band]))
    return frequencies, band_indices, shares


def active_trim_geometry(params: Params, flight: FlightCondition, solution_name: str) -> AncGeometry:
    """Trim-panel plant, error mics and head zone for a solution (its trim panel incl. actuator mass)."""
    trim_cfg = params.solutions[solution_name]["trim"]
    trim = given_plate(params.materials["panels"][trim_cfg["panel"]])
    trim = trim.with_changes(surface_mass_kg_m2=trim.surface_mass_kg_m2 + trim_cfg["added_mass"])
    return build_geometry(params.active, tuple(params.fuselage["trim_panel_size"]), params.cabin["head_distance_from_trim"],
                          trim, piezo_constants(params.materials["piezo"][params.piezo["material"]]), flight.cabin)


def run_active_trim(shared: Shared, solution_name: str, solution_result: dict, log: Callable[[str], None]) -> dict:
    """Simulate the active trim panel with the head-level contributions of this solution, then
    apply the zone attenuation to the head spectrum (the room average is NOT changed)."""
    p = shared.params
    cfg = p.active
    head = solution_result["_mean_square"]["head"]
    controlled = solution_result["_mean_square"]["controlled"]
    duration = min(4.0, cfg["duration"]) if shared.fast else cfg["duration"]
    rng = np.random.default_rng(cfg["random_seed"])
    geometry = active_trim_geometry(p, shared.flight, solution_name)

    frequencies, band_indices, shares = tone_shares(shared, cfg["shaft_orders"])
    tone_controlled = [controlled["engine_tonal"][b] * s for b, s in zip(band_indices, shares)]
    tone_other = [head["engine_tonal"][b] * s - c for b, s, c in zip(band_indices, shares, tone_controlled)]
    log(f"  {solution_name}: tonal FxLMS ({duration:.0f} s simulated)")
    tonal = simulate_tonal(geometry, cfg, frequencies, tone_controlled, tone_other, duration, rng)

    broadband_controlled = controlled["engine_broadband"] + controlled["tbl"]
    broadband_total = head["engine_broadband"] + head["tbl"]
    log(f"  {solution_name}: broadband FxLMS ({duration:.0f} s simulated)")
    broadband = simulate_broadband(geometry, cfg, shared.bands_hz, broadband_controlled,
                                   broadband_total - broadband_controlled, duration, rng)

    tonal_factor = np.ones(len(shared.bands_hz))
    for b, share, attenuation in zip(band_indices, shares, tonal["attenuation_zone_db"]):
        tonal_factor[b] -= share * (1 - 10 ** (-attenuation / 10))
    broadband_factor = 10 ** (-broadband["attenuation_zone_db"] / 10)
    tonal_only = dict(head, engine_tonal=head["engine_tonal"] * tonal_factor)
    tonal_and_broadband = dict(engine_broadband=head["engine_broadband"] * broadband_factor,
                               engine_tonal=head["engine_tonal"] * tonal_factor,
                               tbl=head["tbl"] * broadband_factor)
    return dict(
        tone_frequencies_hz=frequencies,
        tone_controlled_share_at_head=[float(c / (c + o)) for c, o in zip(tone_controlled, tone_other)],
        tonal=tonal, broadband={k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in broadband.items()},
        broadband_controlled_share_at_head=(broadband_controlled / np.maximum(broadband_total, 1e-30)).tolist(),
        head_with_anc=level_summary(shared.bands_hz, tonal_only),
        head_with_anc_and_broadband=level_summary(shared.bands_hz, tonal_and_broadband),
        max_voltage_rms=max(tonal["voltage_rms"], broadband["voltage_rms"]),
        voltage_limit_rms=cfg["max_actuator_voltage"],
    )


# ------------------------------------------------------------------ shared outputs
def closed_form_frequencies(shared: Shared) -> dict:
    """Archive formulas evaluated with the baseline skin and trim."""
    p, flight = shared.params, shared.flight
    baseline = p.solutions[next(iter(p.solutions))]
    sidewall = build_sidewall("baseline", baseline, p.materials, p.fuselage, flight)
    solid = p.materials["solids"][baseline["skin"]["material"]]
    c_l = cf.longitudinal_wave_speed(solid["youngs_modulus"], solid["density"], solid["poisson_ratio"])
    skin, trim = sidewall.skin, sidewall.trim
    a, b = p.fuselage["frame_pitch"], p.fuselage["stringer_pitch"]
    return dict(
        longitudinal_wave_speed_m_s=c_l,
        ring_frequency_hz=cf.ring_frequency(c_l, p.fuselage["radius"]),
        coincidence_skin_outside_hz=cf.coincidence_frequency(flight.outside.sound_speed_m_s, skin.surface_mass_kg_m2, skin.bending_stiffness_nm),
        coincidence_trim_cabin_hz=cf.coincidence_frequency(flight.cabin.sound_speed_m_s, trim.surface_mass_kg_m2, trim.bending_stiffness_nm),
        bay_f11_flat_hz=cf.bay_fundamental_frequency(skin.bending_stiffness_nm, skin.surface_mass_kg_m2, a, b),
        bay_f11_curved_unpressurised_hz=cf.bay_mode_frequency(skin.with_changes(tension_axial_n_m=0.0, tension_circ_n_m=0.0), a, b, 1, 1),
        bay_f11_cruise_hz=cf.bay_mode_frequency(skin, a, b, 1, 1),
        mass_air_mass_hz=cf.mass_air_mass_frequency(skin.surface_mass_kg_m2, trim.surface_mass_kg_m2,
                                                    p.fuselage["sidewall_depth"], flight.cabin.pressure_pa),
        aerodynamic_coincidence_unpressurised_hz=cf.coincidence_frequency(shared.boundary_layer.convection_velocity_m_s,
                                                                          skin.surface_mass_kg_m2, skin.bending_stiffness_nm),
    )


def run(params: Params, fast: bool = False, log: Callable[[str], None] = print) -> dict:
    """Run the full model and return a JSON-ready dict."""
    started = time.time()
    log("set-up: air, source, projection, TBL, cabin")
    shared = build_shared(params, fast)
    flight = shared.flight
    exterior = sum(incident_mean_square(shared, projection, shared.engine.total_power_w)
                   for projection in shared.projections)
    exterior_level = metrics.pressure_level_db(np.sqrt(exterior.sum(axis=-1)))

    results: dict = dict(
        meta=dict(package="cabinsim", fast=fast),
        bands_hz=shared.bands_hz.tolist(),
        flight=dict(outside=vars(flight.outside), cabin=vars(flight.cabin), flight_speed_m_s=flight.flight_speed_m_s,
                    pressure_difference_pa=flight.pressure_difference_pa),
        source=dict(sound_power_level_db_per_engine=shared.engine.total_level_db,
                    engines=len(shared.projections), shaft_frequency_hz=shared.engine.shaft_frequency_hz,
                    tones=[dict(frequency_hz=float(f), power_w=float(w))
                           for f, w in zip(shared.engine.tone_frequencies_hz, shared.engine.tone_power_w)],
                    broadband_power_level_db=metrics.power_level_db(shared.engine.broadband_power_w).tolist(),
                    tonal_power_level_db=metrics.power_level_db(np.maximum(shared.engine.tonal_power_w, 1e-30)).tolist()),
        closed_form=closed_form_frequencies(shared),
        tbl=dict(rms_pressure_pa=float(np.sqrt(shared.tbl_mean_square_pa2.sum())),
                 band_level_db=metrics.pressure_level_db(np.sqrt(shared.tbl_mean_square_pa2)).tolist(),
                 timescale_ratio=shared.boundary_layer.timescale_ratio,
                 convection_velocity_m_s=shared.boundary_layer.convection_velocity_m_s),
        projection=dict(x_m=shared.panels.x_m[:, 0].tolist(), phi_deg=np.rad2deg(shared.panels.phi_rad[0]).tolist(),
                        exterior_oaspl_db=exterior_level.tolist(),
                        incidence_deg_right_engine=np.rad2deg(shared.projections[0].incidence_rad).tolist(),
                        lit_right_engine=shared.projections[0].lit.tolist()),
        cabin=dict(length_m=shared.room.length_m, surface_m2=shared.room.surface_m2, volume_m3=shared.room.volume_m3,
                   sidewall_area_m2=shared.panels.total_area_m2, head_position_m=shared.head_m.tolist(),
                   room_constant_m2=shared.room_constant_m2.tolist(),
                   sil_target_db=params.cabin["sil_target"]),
        solutions={},
    )
    baseline_mass = None
    for name in params.solutions:
        log(f"solution {name}")
        result = evaluate_solution(shared, name, log)
        if baseline_mass is None:
            baseline_mass = result["surface_mass_kg_m2"]
        result["added_mass_kg_m2"] = result["surface_mass_kg_m2"] - baseline_mass
        result["added_mass_section_kg"] = result["added_mass_kg_m2"] * shared.panels.total_area_m2
        if params.solutions[name]["active_trim"]:
            result["active_trim_result"] = run_active_trim(shared, name, result, log)
        result.pop("_mean_square")
        results["solutions"][name] = result
    results["meta"]["runtime_s"] = time.time() - started
    return results
