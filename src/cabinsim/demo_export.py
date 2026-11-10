"""Data for the static demo pages:  python -m cabinsim.demo_export [--results out/results.json] [--demo-dir demo]

Writes demo/data/demo_data.js as  window.CABINSIM_DATA = {...};  (a script, not a .json file, so the pages
also work when opened with file:// where browsers block fetch()) and demo/audio/*.wav (auralisation).
Uses out/results.json for the cabin-model numbers; runs the model (fast) if that file is missing.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np

from .active_control.quiet_zone import quiet_zone_maps
from .atmosphere import FlightCondition, flight_condition
from .auralisation import write_audio
from .fdtd import cross_section_model, material_table
from .geometry import head_position
from .heatmap import fuselage_mesh, surface_levels, vertex_normals
from .params import Params, load_params
from .pipeline import active_trim_geometry, piezo_mass_per_area, run, shunted_bay
from .sidewall.stack import build_sidewall
from .source import band_index_of, engine_positions, engine_spectrum


def _head(params: Params) -> np.ndarray:
    f, c = params.fuselage, params.cabin
    return head_position(f["radius"], f["sidewall_depth"], f["floor_height"], c["seat_x"],
                         c["head_height_above_floor"], c["head_distance_from_trim"], c["seat_side"])


def fdtd_data(params: Params, flight: FlightCondition, results: dict) -> dict:
    """Material map (base64 uint8) and material table of every solution's cross-section."""
    head = _head(params)
    engine = np.asarray(params.engine["position"], float)
    solutions = {}
    model = None
    for name, solution in params.solutions.items():
        model = cross_section_model(solution, params.materials, params.fuselage, engine, head, params.visuals, flight)
        index = np.ascontiguousarray(model.material_index.astype(np.uint8))
        solutions[name] = dict(label=solution["label"], materials=material_table(model),
                               index_b64=base64.b64encode(index.tobytes()).decode("ascii"),
                               head_band_db=results["solutions"].get(name, {}).get("head", {}).get("total_db"))
    nx, nz = model.shape
    return dict(nx=nx, nz=nz, dx=model.cell_size_m, y0=model.y0_m, z0=model.z0_m, sponge_cells=model.sponge_cells,
                source_cell=list(model.source_cell), probe_cell=list(model.probe_cell),
                fuselage_radius=params.fuselage["radius"], floor_z=params.fuselage["floor_height"],
                exaggeration=params.visuals["fdtd_sidewall_exaggeration"], baseline=next(iter(params.solutions)),
                compare=params.visuals["fdtd_compare_solution"], bands_hz=results["bands_hz"], solutions=solutions)


def heatmap_data(params: Params, flight: FlightCondition) -> dict:
    """Simplified fuselage mesh with exterior engine levels per vertex (overall and octave bands)."""
    x_nose, x_tail = params.visuals["heatmap_fuselage_x"]
    axial_step, angle_step = params.visuals["heatmap_resolution"]
    mesh = fuselage_mesh(params.fuselage["radius"], x_nose, x_tail, axial_step, angle_step)
    bands = np.asarray(params.numerics["bands"], float)
    spectrum = engine_spectrum(bands, params.engine)
    levels = surface_levels(mesh, vertex_normals(mesh), engine_positions(params.engine), spectrum.total_power_w, bands,
                            flight.outside.impedance_rayl, params.engine["shadow_loss"])
    return dict(positions=np.round(mesh.positions_m, 3).ravel().tolist(), triangles=mesh.triangles.ravel().tolist(),
                levels={k: np.round(v, 1).tolist() for k, v in levels.items()},
                engines=[p.tolist() for p in engine_positions(params.engine)], fan_diameter=params.engine["fan_diameter"],
                cabin_section_x=params.fuselage["cabin_section_x"], radius=params.fuselage["radius"],
                sound_power_level_db=spectrum.total_level_db)


def bay_data(params: Params, flight: FlightCondition) -> dict | None:
    """Modal velocities of one skin bay, patches short-circuited vs shunted, over a frequency sweep."""
    name = next((n for n, s in params.solutions.items() if s["piezo_shunt"]), None)
    if name is None:
        return None
    sidewall = build_sidewall(name, params.solutions[name], params.materials, params.fuselage, flight,
                              piezo_mass_per_area(params))
    bay = shunted_bay(params, flight, sidewall)
    f_low, f_high, count = params.visuals["bay_frequency_range"]
    targets = [bay.short_circuit_omega[i] / (2 * np.pi) for i in bay.target_index]
    frequencies = np.unique(np.round(np.concatenate([np.geomspace(f_low, f_high, int(count)), targets]), 1))
    incidence, azimuth = np.deg2rad(params.visuals["bay_incidence"])
    shunt = params.piezo["shunt_type"]
    fields = {"short": [], shunt: []}
    ratio_db = []
    for frequency in frequencies:
        power = {}
        for kind in fields:
            velocity, _ = bay.modal_response(frequency, incidence, azimuth, kind, params.piezo["negative_capacitance_ratio"])
            fields[kind].append(np.round(np.concatenate([velocity.real, velocity.imag]), 9).tolist())
            power[kind] = np.sum(np.abs(velocity) ** 2)            # mean-square velocity is proportional to this
        ratio_db.append(float(10 * np.log10(power["short"] / power[shunt])))
    return dict(solution=name, label=params.solutions[name]["label"], a=bay.a, b=bay.b,
                incidence_deg=list(params.visuals["bay_incidence"]),
                modes=[list(m) for m in bay.modes], frequencies_hz=frequencies.tolist(), shunt=shunt,
                fields=fields, velocity_reduction_db=ratio_db, patches=params.piezo["patches"], summary=bay.summary())


def quiet_zone_data(params: Params, flight: FlightCondition, results: dict) -> dict | None:
    """Quiet-zone maps for the first solution with an active trim panel, at its tones and band centres."""
    name = next((n for n, s in results["solutions"].items() if "active_trim_result" in s), None)
    if name is None:
        return None
    anc = results["solutions"][name]["active_trim_result"]
    bands = np.asarray(results["bands_hz"], float)
    band_share = anc["broadband_controlled_share_at_head"]
    f_low, f_high = params.active["broadband_band"]
    frequencies = [float(b) for b in bands if f_low <= b <= f_high]
    shares = [band_share[band_index_of(f, bands)] for f in frequencies]
    for tone, share in zip(anc["tone_frequencies_hz"], anc["tone_controlled_share_at_head"]):
        frequencies.append(tone)
        shares.append(share)
    order = np.argsort(frequencies)
    width, depth, n_points = params.visuals["quiet_zone_slice"]
    geometry = active_trim_geometry(params, flight, name)
    maps = quiet_zone_maps(geometry, params.active, [frequencies[i] for i in order], [shares[i] for i in order],
                           width, depth, int(n_points), params.visuals["quiet_zone_realisations"],
                           np.random.default_rng(params.active["random_seed"]))
    maps.update(solution=name, label=results["solutions"][name]["label"],
                realisations=params.visuals["quiet_zone_realisations"],
                tones_hz=anc["tone_frequencies_hz"], panel_size_m=params.fuselage["trim_panel_size"])
    return maps


def build_demo_data(params: Params, results: dict, log=print) -> dict:
    flight = flight_condition(params.flight)
    log("demo data: FDTD cross-sections")
    data = dict(fdtd=fdtd_data(params, flight, results))
    log("demo data: 3D heat map")
    data["heatmap"] = heatmap_data(params, flight)
    log("demo data: skin bay animation")
    data["bay"] = bay_data(params, flight)
    log("demo data: quiet zone maps")
    data["quiet_zone"] = quiet_zone_data(params, flight, results)
    data["summary"] = {name: dict(label=s["label"], head_sil_db=s["head"]["sil_db"], head_dba=s["head"]["dba"],
                                  added_mass_kg_m2=s["added_mass_kg_m2"])
                       for name, s in results["solutions"].items()}
    data["target_sil_db"] = results["cabin"]["sil_target_db"]
    return data


def write_demo_data(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, separators=(",", ":"))
    path.write_text(f"// generated by python -m cabinsim.demo_export\nwindow.CABINSIM_DATA = {text};\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export data and sound files for the static demo pages")
    parser.add_argument("--results", type=Path, default=Path("out/results.json"))
    parser.add_argument("--demo-dir", type=Path, default=Path("demo"), help="writes data/demo_data.js and audio/*.wav here")
    parser.add_argument("--no-audio", action="store_true", help="skip the .wav files")
    parser.add_argument("--config-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    params = load_params(args.config_dir)
    if args.results.exists():
        results = json.loads(args.results.read_text(encoding="utf-8"))
    else:
        print(f"{args.results} not found: running the model (fast)")
        results = run(params, fast=True)
    data = build_demo_data(params, results)
    if not args.no_audio:
        print("demo data: auralisation (.wav)")
        data["audio"] = write_audio(results, params.visuals, args.demo_dir / "audio")
        print(f"written: {len(data['audio']['cases'])} files in {args.demo_dir / 'audio'}")
    target = args.demo_dir / "data" / "demo_data.js"
    write_demo_data(data, target)
    print(f"written: {target} ({target.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
