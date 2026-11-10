"""Write results to JSON and CSV (as files, or as text for download buttons)."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import numpy as np


def _to_builtin(value):
    """Convert numpy types so json can write them."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"cannot serialise {type(value)}")


def json_text(results: dict) -> str:
    return json.dumps(results, indent=1, default=_to_builtin)


def spectra_table(results: dict) -> dict[str, list[float]]:
    """Columns: band, then head and room level of each solution (+ ANC variants if any)."""
    columns: dict[str, list[float]] = {"band_hz": list(results["bands_hz"])}
    for name, solution in results["solutions"].items():
        columns[f"{name}_head_db"] = solution["head"]["total_db"]
        columns[f"{name}_room_db"] = solution["room"]["total_db"]
        if "active_trim_result" in solution:
            anc = solution["active_trim_result"]
            columns[f"{name}_head_with_tonal_anc_db"] = anc["head_with_anc"]["total_db"]
            columns[f"{name}_head_with_tonal_and_broadband_anc_db"] = anc["head_with_anc_and_broadband"]["total_db"]
    return columns


def spectra_csv_text(results: dict) -> str:
    columns = spectra_table(results)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(list(columns))
    for i in range(len(results["bands_hz"])):
        writer.writerow([f"{values[i]:g}" if key == "band_hz" else f"{values[i]:.2f}" for key, values in columns.items()])
    return buffer.getvalue()


def summary_rows(results: dict) -> list[dict]:
    """Single-number results per solution (head position, plus ANC where present)."""
    rows = []
    for name, solution in results["solutions"].items():
        row = dict(solution=name, label=solution["label"],
                   head_sil_db=solution["head"]["sil_db"], head_dba=solution["head"]["dba"],
                   room_sil_db=solution["room"]["sil_db"], room_dba=solution["room"]["dba"],
                   added_mass_kg_m2=solution["added_mass_kg_m2"], added_mass_section_kg=solution["added_mass_section_kg"])
        if "active_trim_result" in solution:
            tonal = solution["active_trim_result"]["head_with_anc"]
            both = solution["active_trim_result"]["head_with_anc_and_broadband"]
            row.update(head_sil_with_anc_db=tonal["sil_db"], head_dba_with_anc=tonal["dba"],
                       head_sil_with_anc_broadband_db=both["sil_db"], head_dba_with_anc_broadband=both["dba"])
        rows.append(row)
    return rows


def summary_csv_text(results: dict) -> str:
    rows = summary_rows(results)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: (f"{v:.2f}" if isinstance(v, float) else v) for k, v in row.items()})
    return buffer.getvalue()


def _write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(results: dict, path: Path) -> None:
    _write_text(json_text(results), path)


def write_spectra_csv(results: dict, path: Path) -> None:
    _write_text(spectra_csv_text(results), path)


def write_summary_csv(results: dict, path: Path) -> None:
    _write_text(summary_csv_text(results), path)


def write_figures(results: dict, folder: Path, dpi: int = 160) -> list[Path]:
    """Save every figure of `cabinsim.plots` as PNG; needs matplotlib (pip install .[plots])."""
    import matplotlib.pyplot as plt

    from .plots import all_figures

    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for name, figure in all_figures(results).items():
        path = folder / f"{name}.png"
        figure.savefig(path, dpi=dpi)
        plt.close(figure)
        written.append(path)
    return written
