"""Command line entry point:  python -m cabinsim.run [--fast] [--plots] [--out out] [--config-dir DIR]

Runs the whole model and writes out/results.json, out/spectra.csv and out/summary.csv
(and PNG figures in out/figures with --plots).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .export import summary_rows, write_figures, write_json, write_spectra_csv, write_summary_csv
from .params import load_params
from .pipeline import run


def print_summary(results: dict) -> None:
    print()
    print("Head = passenger's head position (window seat, engine side). ANC columns: SIL / dB(A) at the head.")
    print(f"{'solution':<14}{'head SIL':>9}{'head dB(A)':>11}{'room SIL':>9}{'+kg/m2':>8}{'+kg section':>12}"
          f"{'tonal ANC':>14}{'tonal+bb ANC':>15}")
    for row in summary_rows(results):
        tonal = both = f"{'-':>14}"
        if "head_sil_with_anc_db" in row:
            tonal = f"{row['head_sil_with_anc_db']:8.1f} /{row['head_dba_with_anc']:5.1f}"
            both = f"{row['head_sil_with_anc_broadband_db']:9.1f} /{row['head_dba_with_anc_broadband']:5.1f}"
        print(f"{row['solution']:<14}{row['head_sil_db']:9.1f}{row['head_dba']:11.1f}{row['room_sil_db']:9.1f}"
              f"{row['added_mass_kg_m2']:8.2f}{row['added_mass_section_kg']:12.0f}{tonal}{both}")
    for name, solution in results["solutions"].items():
        anc = solution.get("active_trim_result")
        if anc is None:
            continue
        tones = ", ".join(f"{f:.0f} Hz {a:.1f} dB" for f, a in zip(anc["tone_frequencies_hz"], anc["tonal"]["attenuation_zone_db"]))
        print(f"\n{name} head-zone tone reduction: {tones}")
        print(f"{name} max actuator voltage {anc['max_voltage_rms']:.1f} V rms (limit {anc['voltage_limit_rms']:.0f} V); "
              f"tonal converged: {anc['tonal']['converged']}, broadband converged: {anc['broadband']['converged']}")
    print(f"\nruntime {results['meta']['runtime_s']:.0f} s")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="EB205 cabin noise simulation")
    parser.add_argument("--config-dir", type=Path, default=None, help="folder with the YAML files (default: built-in)")
    parser.add_argument("--out", type=Path, default=Path("out"), help="output folder (default: ./out)")
    parser.add_argument("--fast", action="store_true", help="coarser numerics and shorter ANC run")
    parser.add_argument("--plots", action="store_true", help="also save all figures as PNG in OUT/figures")
    args = parser.parse_args(argv)

    params = load_params(args.config_dir)
    results = run(params, fast=args.fast)
    write_json(results, args.out / "results.json")
    write_spectra_csv(results, args.out / "spectra.csv")
    write_summary_csv(results, args.out / "summary.csv")
    print_summary(results)
    print(f"written: {args.out / 'results.json'}, spectra.csv, summary.csv")
    if args.plots:
        figures = write_figures(results, args.out / "figures")
        print(f"written: {len(figures)} figures in {args.out / 'figures'}")


if __name__ == "__main__":
    main()
