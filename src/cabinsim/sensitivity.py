"""One-at-a-time sensitivity of the main results to uncertain inputs:  python -m cabinsim.sensitivity

For every variation the model is re-run (fast numerics, baseline and passive solutions only, no ANC)
and four numbers are recorded: baseline head SIL and dB(A), and the passive solution's improvement
in SIL and dB(A). The output is a Markdown table for docs/validation.md.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .params import Params, load_params
from .pipeline import run

BASELINE, PASSIVE = "V0_baseline", "V1_passive"


@dataclass(frozen=True)
class Variation:
    parameter: str
    low_label: str
    high_label: str
    low: Callable[[Params], None]
    high: Callable[[Params], None]


def _scale(section: str, key: str, factor: float) -> Callable[[Params], None]:
    def apply(p: Params) -> None:
        getattr(p, section)[key] *= factor
    return apply


def _set(section: str, key: str, value) -> Callable[[Params], None]:
    def apply(p: Params) -> None:
        getattr(p, section)[key] = value
    return apply


def _every_solution(path: tuple[str, str], value) -> Callable[[Params], None]:
    def apply(p: Params) -> None:
        for solution in p.solutions.values():
            solution[path[0]][path[1]] = value
    return apply


def _material(group: str, name: str | None, key: str, factor: float) -> Callable[[Params], None]:
    def apply(p: Params) -> None:
        for material_name, material in p.materials[group].items():
            if name is None or material_name == name:
                material[key] *= factor
    return apply


def _absorption(factor: float) -> Callable[[Params], None]:
    def apply(p: Params) -> None:
        p.cabin["absorption_coefficients"] = [min(a * factor, 0.95) for a in p.cabin["absorption_coefficients"]]
    return apply


def _engine_y(shift: float) -> Callable[[Params], None]:
    def apply(p: Params) -> None:
        p.engine["position"] = [p.engine["position"][0], p.engine["position"][1] + shift, p.engine["position"][2]]
    return apply


def _no_change(p: Params) -> None:
    return None


# Ranges: literature spread where we have one, otherwise a deliberately wide guess (see docs/validation.md).
VARIATIONS = [
    Variation("TBL wavenumber model", "Mellen", "Corcos", _set("tbl", "wavenumber_model", "mellen"),
              _set("tbl", "wavenumber_model", "corcos")),
    Variation("TBL wall shear stress", "-30 %", "+30 %", _scale("tbl", "wall_shear_stress", 0.7),
              _scale("tbl", "wall_shear_stress", 1.3)),
    Variation("TBL convection speed U_c/U", "0.6", "0.8", _set("tbl", "convection_velocity_ratio", 0.6),
              _set("tbl", "convection_velocity_ratio", 0.8)),
    Variation("Engine acoustic power", "-3 dB", "+3 dB", _scale("engine", "acoustic_power", 0.5),
              _scale("engine", "acoustic_power", 2.0)),
    Variation("Engine tone fraction", "0.15", "0.45", _set("engine", "tone_energy_fraction", 0.15),
              _set("engine", "tone_energy_fraction", 0.45)),
    Variation("Engine spanwise position", "-1 m", "+1 m", _engine_y(-1.0), _engine_y(1.0)),
    Variation("Shadow loss", "8 dB", "25 dB", _set("engine", "shadow_loss", 8.0), _set("engine", "shadow_loss", 25.0)),
    Variation("Bridge isolator gain", "10 dB", "20 dB", _set("bridge", "isolator_gain", 10.0),
              _set("bridge", "isolator_gain", 20.0)),
    Variation("Bridge attachment spacing", "0.30 m", "0.60 m", _set("bridge", "point_spacing", 0.30),
              _set("bridge", "point_spacing", 0.60)),
    Variation("Cabin absorption", "x0.7", "x1.3", _absorption(0.7), _absorption(1.3)),
    Variation("Skin tension from pressure", "off", "on (default)", _set("fuselage", "pressurised_skin_tension", False),
              _no_change),
    Variation("CFRP Young's modulus", "45 GPa", "70 GPa", _material("solids", "CFRP_QI", "youngs_modulus", 0.75),
              _material("solids", "CFRP_QI", "youngs_modulus", 70 / 60)),
    Variation("Skin thickness", "3 mm", "5 mm", _every_solution(("skin", "thickness"), 0.003),
              _every_solution(("skin", "thickness"), 0.005)),
    Variation("Blanket flow resistivity", "x0.5", "x2", _material("porous", None, "flow_resistivity", 0.5),
              _material("porous", None, "flow_resistivity", 2.0)),
    Variation("Radiating trim panel size", "0.6 x 0.5 m", "1.6 x 1.2 m", _set("fuselage", "trim_panel_size", [0.6, 0.5]),
              _set("fuselage", "trim_panel_size", [1.6, 1.2])),
    Variation("Head distance from trim", "0.30 m", "0.60 m", _set("cabin", "head_distance_from_trim", 0.30),
              _set("cabin", "head_distance_from_trim", 0.60)),
]


def study_params(params: Params) -> Params:
    """Baseline and passive solution only, without active trim (keeps each run short)."""
    p = params.copy()
    p.solutions = {name: p.solutions[name] for name in (BASELINE, PASSIVE)}
    for solution in p.solutions.values():
        solution["active_trim"] = False
    return p


def key_numbers(results: dict) -> dict[str, float]:
    base, passive = results["solutions"][BASELINE]["head"], results["solutions"][PASSIVE]["head"]
    return dict(sil=base["sil_db"], dba=base["dba"], gain_sil=base["sil_db"] - passive["sil_db"],
                gain_dba=base["dba"] - passive["dba"])


def run_study(params: Params, variations: list[Variation], log: Callable[[str], None] = print) -> dict:
    reference_params = study_params(params)
    reference = key_numbers(run(reference_params, fast=True, log=lambda m: None))
    rows = []
    for variation in variations:
        values = {}
        for label, apply in ((variation.low_label, variation.low), (variation.high_label, variation.high)):
            p = reference_params.copy()
            apply(p)
            values[label] = key_numbers(run(p, fast=True, log=lambda m: None))
        log(f"  {variation.parameter}: done")
        rows.append(dict(parameter=variation.parameter, low=variation.low_label, high=variation.high_label,
                         low_values=values[variation.low_label], high_values=values[variation.high_label]))
    return dict(reference=reference, rows=rows)


def markdown_table(study: dict) -> str:
    """Change relative to the reference run, low / high setting, in dB."""
    ref = study["reference"]
    lines = [f"Reference (fast numerics): baseline head SIL {ref['sil']:.1f} dB, {ref['dba']:.1f} dB(A); "
             f"passive solution gains {ref['gain_sil']:.1f} dB SIL, {ref['gain_dba']:.1f} dB(A).",
             "",
             "| input | low / high setting | Δ baseline SIL | Δ baseline dB(A) | Δ passive gain, SIL | Δ passive gain, dB(A) |",
             "|---|---|---|---|---|---|"]

    def pair(row: dict, key: str) -> str:
        return f"{row['low_values'][key] - ref[key]:+.1f} / {row['high_values'][key] - ref[key]:+.1f}"

    def spread(row: dict) -> float:
        return max(abs(row["low_values"][k] - ref[k]) + abs(row["high_values"][k] - ref[k]) for k in ("sil", "dba"))

    for row in sorted(study["rows"], key=spread, reverse=True):
        lines.append(f"| {row['parameter']} | {row['low']} / {row['high']} | {pair(row, 'sil')} | {pair(row, 'dba')} | "
                     f"{pair(row, 'gain_sil')} | {pair(row, 'gain_dba')} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="One-at-a-time sensitivity study")
    parser.add_argument("--out", type=Path, default=Path("out/sensitivity.md"))
    parser.add_argument("--config-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    started = time.time()
    print(f"sensitivity study: {len(VARIATIONS)} inputs x 2 settings (fast numerics)")
    study = run_study(load_params(args.config_dir), VARIATIONS)
    table = markdown_table(study)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(table, encoding="utf-8")
    print(table)
    print(f"written: {args.out} ({time.time() - started:.0f} s)")


if __name__ == "__main__":
    main()
