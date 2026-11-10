"""Figures for the results dict produced by `cabinsim.pipeline.run`.

Every function takes the results dict and returns a matplotlib Figure, so the same figures are
used by the Streamlit app, notebooks and `python -m cabinsim.run --plots`.
Colours follow the solution (fixed order in solutions.yaml), never its rank in a filtered list.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

# Validated categorical palette (light surface), fixed order   [dataviz reference palette]
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK_SECONDARY, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
LINE_WIDTH = 2.0
MARKER_SIZE = 6.0
OCTAVE_TICKS = [63, 125, 250, 500, 1000, 2000, 4000]

STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_SECONDARY, "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "grid.linestyle": "-", "axes.spines.top": False, "axes.spines.right": False, "font.size": 9.5,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "legend.frameon": False, "lines.linewidth": LINE_WIDTH, "lines.solid_capstyle": "round",
}


def solution_colors(results: dict, order: list[str] | None = None) -> dict[str, str]:
    """Colour per solution key. Pass the full solution order so colours stay stable when some are hidden."""
    names = order or list(results["solutions"])
    return {name: SERIES_COLORS[i % len(SERIES_COLORS)] for i, name in enumerate(names)}


def _new_figure(width: float = 8.0, height: float = 4.2, ncols: int = 1, **kwargs) -> tuple[Figure, np.ndarray]:
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, ncols, figsize=(width, height), constrained_layout=True, squeeze=False, **kwargs)
    return fig, axes[0]


def _apply_style(ax) -> None:
    """Re-apply style items that rc_context does not carry onto existing axes."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_SECONDARY)
    ax.grid(True, color=GRID, linewidth=0.7, linestyle="-")
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)


def _title(ax, text: str) -> None:
    ax.set_title(text, loc="left", color=INK, fontweight="bold", fontsize=11)


def _band_axis(ax, label: str = "1/3-octave band (Hz)") -> None:
    ax.set_xscale("log")
    ax.set_xticks(OCTAVE_TICKS)
    ax.set_xticklabels([str(t) if t < 1000 else f"{t // 1000}k" for t in OCTAVE_TICKS])
    ax.minorticks_off()
    ax.set_xlabel(label)


def _legend(ax, **kwargs) -> None:
    legend = ax.legend(frameon=False, **kwargs)
    for text in legend.get_texts():
        text.set_color(INK)


def _line(ax, x, y, color: str, label: str, **kwargs) -> None:
    ax.plot(x, y, color=color, lw=LINE_WIDTH, marker="o", ms=MARKER_SIZE, mec=SURFACE, mew=1.5,
            label=label, solid_joinstyle="round", **kwargs)


def _end_labels(ax, x_end: float, ends: list[tuple[float, str]], min_gap_fraction: float = 0.04) -> None:
    """Direct labels at line ends; a label that would collide is skipped (the legend carries it)."""
    low, high = ax.get_ylim()
    gap = (high - low) * min_gap_fraction
    placed: list[float] = []
    for y, text in sorted(ends):
        if all(abs(y - other) >= gap for other in placed):
            ax.annotate(text, (x_end, y), xytext=(6, 0), textcoords="offset points", va="center",
                        color=INK_SECONDARY, fontsize=8.5, annotation_clip=False)
            placed.append(y)


def _selected(results: dict, names: list[str] | None) -> list[str]:
    return [n for n in (names or list(results["solutions"])) if n in results["solutions"]]


# ------------------------------------------------------------------ source and projection
def projection_figure(results: dict) -> Figure:
    """Exterior engine OASPL on the cabin section and incidence angle of the right engine."""
    projection = results["projection"]
    x = np.asarray(projection["x_m"])
    phi = np.asarray(projection["phi_deg"])
    fig, (ax_level, ax_angle) = _new_figure(11.0, 4.0, ncols=2)
    mesh = ax_level.pcolormesh(x, phi, np.asarray(projection["exterior_oaspl_db"]).T, cmap="Blues", shading="nearest")
    fig.colorbar(mesh, ax=ax_level, label="OASPL (dB re 20 µPa)")
    _title(ax_level, "Engine noise on the fuselage (both engines)")
    mesh = ax_angle.pcolormesh(x, phi, np.asarray(projection["incidence_deg_right_engine"]).T, cmap="Oranges",
                               shading="nearest", vmin=0, vmax=90)
    fig.colorbar(mesh, ax=ax_angle, label="incidence angle θ (°)")
    _title(ax_angle, "Incidence angle θ, right engine")
    for ax in (ax_level, ax_angle):
        _apply_style(ax)
        ax.grid(False)
        ax.set_xlabel("x from fan face (m), negative = forward")
        ax.set_ylabel("φ (°): 0 = starboard side, 90 = crown, 180 = port")
    return fig


def source_spectrum_figure(results: dict) -> Figure:
    """Engine sound power per band, broadband and tonal parts (one engine)."""
    bands = np.asarray(results["bands_hz"])
    source = results["source"]
    fig, (ax,) = _new_figure(8.0, 3.8)
    _apply_style(ax)
    _line(ax, bands, source["broadband_power_level_db"], SERIES_COLORS[0], "broadband")
    tonal = np.asarray(source["tonal_power_level_db"])
    has_tone = tonal > 0
    ax.plot(bands[has_tone], tonal[has_tone], ls="none", marker="D", ms=MARKER_SIZE + 1, color=SERIES_COLORS[1],
            mec=SURFACE, mew=1.5, label="tones (shaft orders)")
    _band_axis(ax)
    ax.set_ylabel("Lw per band (dB re 1 pW)")
    _title(ax, f"Engine sound power, one engine: Lw = {source['sound_power_level_db_per_engine']:.1f} dB")
    _legend(ax, loc="lower center")
    return fig


# ------------------------------------------------------------------ sidewall
CLOSED_FORM_MARKERS = [("mass_air_mass_hz", "f₀"), ("ring_frequency_hz", "f_r"),
                       ("bay_f11_cruise_hz", "f₁₁ cruise"), ("coincidence_skin_outside_hz", "f_c")]


def transmission_loss_figure(results: dict, names: list[str] | None = None, colors: dict | None = None) -> Figure:
    """Diffuse-field TL (air + bridge) of each solution, with the archive's characteristic frequencies."""
    bands = np.asarray(results["bands_hz"])
    colors = colors or solution_colors(results)
    fig, (ax,) = _new_figure(8.5, 4.4)
    _apply_style(ax)
    ends = []
    for name in _selected(results, names):
        solution = results["solutions"][name]
        tl = np.asarray(solution["tl_diffuse_db"]["total"]) + np.asarray(solution["shunt_delta_tl_db"])
        _line(ax, bands, tl, colors[name], solution["label"])
        ends.append((tl[-1], solution["label"]))
    _band_axis(ax)
    ax.set_xlim(40, 9000)
    for key, text in CLOSED_FORM_MARKERS:
        frequency = results["closed_form"][key]
        ax.axvline(frequency, color=AXIS, lw=1.0)
        ax.annotate(text, (frequency, 1.0), xycoords=("data", "axes fraction"), xytext=(3, -12),
                    textcoords="offset points", color=INK_SECONDARY, fontsize=8.5)
    _end_labels(ax, bands[-1], ends)
    ax.set_ylabel("TL (dB), diffuse incidence")
    _title(ax, "Sidewall transmission loss (finite trim panel, incl. bridge path and shunt)")
    _legend(ax, loc="upper left")
    return fig


def coupling_penalty_figure(results: dict, names: list[str] | None = None, colors: dict | None = None) -> Figure:
    """How much less the TBL transmits than sound of the same wall pressure, vs the archive's 20-30 dB."""
    bands = np.asarray(results["bands_hz"])
    colors = colors or solution_colors(results)
    fig, (ax,) = _new_figure(8.5, 4.0)
    _apply_style(ax)
    ax.axhspan(20, 30, color=GRID, zorder=0)
    ax.annotate("archive assumption 20-30 dB", (bands[0], 30), xytext=(2, -12), textcoords="offset points",
                color=INK_SECONDARY, fontsize=8.5)
    for name in _selected(results, names):
        solution = results["solutions"][name]
        penalties = solution["tbl_coupling_penalty_db"]
        model = next(iter(penalties))
        _line(ax, bands, penalties[model], colors[name], f"{solution['label']} ({model})")
    ax.axhline(0, color=AXIS, lw=1.0)
    _band_axis(ax)
    ax.set_ylabel("TBL coupling penalty (dB)")
    _title(ax, "TBL vs diffuse sound: transmitted power difference at equal wall pressure")
    _legend(ax, loc="lower left", fontsize=8.5)
    return fig


# ------------------------------------------------------------------ cabin
def cabin_spectrum_figure(results: dict, location: str = "head", names: list[str] | None = None,
                          colors: dict | None = None) -> Figure:
    """Total band levels at the head (or room average) for each solution; ANC variant dashed."""
    bands = np.asarray(results["bands_hz"])
    colors = colors or solution_colors(results)
    fig, (ax,) = _new_figure(8.5, 4.4)
    _apply_style(ax)
    for name in _selected(results, names):
        solution = results["solutions"][name]
        _line(ax, bands, solution[location]["total_db"], colors[name], solution["label"])
        if location == "head" and "active_trim_result" in solution:
            ax.plot(bands, solution["active_trim_result"]["head_with_anc"]["total_db"], color=colors[name],
                    lw=LINE_WIDTH, ls="--", label=f"{solution['label']}, tonal ANC on")
    _band_axis(ax)
    ax.set_ylabel("Lp (dB re 20 µPa)")
    where = "passenger's head (window seat, engine side)" if location == "head" else "room average (reverberant)"
    _title(ax, f"Cabin spectrum at the {where}")
    _legend(ax, loc="upper right")
    return fig


PATH_LABELS = {"engine_broadband": "engine broadband", "engine_tonal": "engine tones", "tbl": "boundary layer (TBL)"}


def path_breakdown_figure(results: dict, name: str) -> Figure:
    """Contribution of each noise path to the head spectrum of one solution."""
    bands = np.asarray(results["bands_hz"])
    solution = results["solutions"][name]
    fig, (ax,) = _new_figure(8.5, 4.2)
    _apply_style(ax)
    for i, (path, levels) in enumerate(solution["head"]["paths_db"].items()):
        levels = np.asarray(levels)
        if path == "engine_tonal":
            visible = levels > 0
            ax.plot(bands[visible], levels[visible], ls="none", marker="D", ms=MARKER_SIZE + 1,
                    color=SERIES_COLORS[i], mec=SURFACE, mew=1.5, label=PATH_LABELS[path])
        else:
            _line(ax, bands, levels, SERIES_COLORS[i], PATH_LABELS[path])
    ax.plot(bands, solution["head"]["total_db"], color=INK, lw=1.2, label="total")
    _band_axis(ax)
    ax.set_ylim(bottom=max(0.0, ax.get_ylim()[0]))
    ax.set_ylabel("Lp (dB re 20 µPa)")
    _title(ax, f"Noise paths at the head: {solution['label']}")
    _legend(ax, loc="upper right")
    return fig


def sil_figure(results: dict, names: list[str] | None = None, colors: dict | None = None) -> Figure:
    """Head SIL of each solution vs the VIP target band; tonal-ANC result as a hatched extra bar."""
    colors = colors or solution_colors(results)
    selected = _selected(results, names)
    bars = []
    for name in selected:
        solution = results["solutions"][name]
        bars.append((solution["label"], solution["head"]["sil_db"], colors[name], None, solution["added_mass_kg_m2"]))
        if "active_trim_result" in solution:
            anc = solution["active_trim_result"]["head_with_anc"]
            bars.append((f"{solution['label']}\n+ tonal ANC", anc["sil_db"], colors[name], "//", solution["added_mass_kg_m2"]))
    fig, (ax,) = _new_figure(max(6.0, 1.6 * len(bars) + 2), 4.2)
    _apply_style(ax)
    low, high = results["cabin"]["sil_target_db"]
    ax.axhspan(low, high, color=GRID, zorder=0)
    ax.annotate(f"VIP target {low:.0f}-{high:.0f} dB", (0.0, low), xycoords=("axes fraction", "data"),
                xytext=(4, 4), textcoords="offset points", ha="left", color=INK_SECONDARY, fontsize=8.5)
    values = [b[1] for b in bars]
    floor = min(min(values), low) - 8
    for i, (label, value, color, hatch, mass) in enumerate(bars):
        ax.bar(i, value - floor, bottom=floor, width=0.55, color=color, hatch=hatch, edgecolor=SURFACE, linewidth=0)
        ax.annotate(f"{value:.1f}", (i, value), xytext=(0, 3), textcoords="offset points", ha="center", color=INK)
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([f"{b[0]}\n+{b[4]:.2f} kg/m²" for b in bars], fontsize=8.5)
    ax.set_ylim(floor, max(max(values), high) + 3)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("SIL at the head (dB)")
    _title(ax, "Speech interference level")
    return fig


# ------------------------------------------------------------------ piezo shunt
def shunt_figure(results: dict, names: list[str] | None = None, colors: dict | None = None) -> Figure:
    """Band TL gain of the RL shunts (vs short-circuited patches) for each solution that has them."""
    bands = np.asarray(results["bands_hz"])
    colors = colors or solution_colors(results)
    shunted = [n for n in _selected(results, names) if results["solutions"][n]["piezo_shunt"]]
    fig, (ax,) = _new_figure(8.5, 3.8)
    _apply_style(ax)
    width = 0.8 / max(len(shunted), 1)
    positions = np.arange(len(bands))
    for i, name in enumerate(shunted):
        solution = results["solutions"][name]
        ax.bar(positions + (i - (len(shunted) - 1) / 2) * width, solution["shunt_delta_tl_db"], width=width * 0.9,
               color=colors[name], label=solution["label"])
    ax.axhline(0, color=AXIS, lw=1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{b:g}" for b in bands], rotation=90, fontsize=8)
    ax.set_xlabel("1/3-octave band (Hz)")
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("ΔTL from shunt (dB)")
    _title(ax, "Electrical damping gain of the shunted piezo patches")
    if shunted:
        _legend(ax, loc="upper left")
    return fig


# ------------------------------------------------------------------ active trim
def anc_convergence_figure(results: dict, name: str) -> Figure:
    """Error-microphone power relative to the uncontrolled level, tonal and broadband controllers."""
    anc = results["solutions"][name]["active_trim_result"]
    fig, (ax,) = _new_figure(8.0, 3.8)
    _apply_style(ax)
    for i, (key, label) in enumerate([("tonal", "tonal FxLMS"), ("broadband", "broadband FxLMS")]):
        history = anc[key]["history"]
        ax.plot(history["time_s"], history["relative_level_db"], color=SERIES_COLORS[i], lw=LINE_WIDTH, label=label)
    ax.axhline(0, color=AXIS, lw=1.0)
    ax.set_xlabel("time (s); controller switched on at 1 s")
    ax.set_ylabel("error-mic level re. off (dB)")
    _title(ax, "FxLMS convergence at the error microphones")
    _legend(ax, loc="lower left")
    return fig


def anc_tones_figure(results: dict, name: str) -> Figure:
    """Tone reduction at the error microphones vs averaged over the head zone."""
    anc = results["solutions"][name]["active_trim_result"]
    tones = anc["tone_frequencies_hz"]
    positions = np.arange(len(tones))
    fig, (ax,) = _new_figure(7.0, 3.8)
    _apply_style(ax)
    for offset, key, label, color in [(-0.17, "attenuation_mic_db", "at the error microphones", SERIES_COLORS[0]),
                                      (0.17, "attenuation_zone_db", "head zone average", SERIES_COLORS[2])]:
        values = anc["tonal"][key]
        ax.bar(positions + offset, values, width=0.3, color=color, label=label)
        for x, value in zip(positions + offset, values):
            ax.annotate(f"{value:.1f}", (x, value), xytext=(0, 3), textcoords="offset points", ha="center",
                        color=INK_SECONDARY, fontsize=8.5)
    ax.axhline(0, color=AXIS, lw=1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{f:.0f} Hz" for f in tones])
    ax.grid(axis="x", visible=False)
    top = max(max(anc["tonal"]["attenuation_mic_db"]), max(anc["tonal"]["attenuation_zone_db"]), 1.0)
    ax.set_ylim(top=top * 1.35)                       # headroom so the legend sits above the bars
    ax.set_ylabel("reduction (dB)")
    _title(ax, "Engine tones: active trim panel")
    _legend(ax, loc="upper left", ncols=2)
    return fig


def anc_broadband_figure(results: dict, name: str) -> Figure:
    """Broadband head-zone change per band (positive = quieter, negative = ANC makes it louder)."""
    bands = np.asarray(results["bands_hz"])
    anc = results["solutions"][name]["active_trim_result"]
    values = np.asarray(anc["broadband"]["attenuation_zone_db"])
    fig, (ax,) = _new_figure(8.5, 3.6)
    _apply_style(ax)
    positions = np.arange(len(bands))
    ax.bar(positions, values, width=0.6, color=SERIES_COLORS[0])
    ax.axhline(0, color=AXIS, lw=1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{b:g}" for b in bands], rotation=90, fontsize=8)
    ax.set_xlabel("1/3-octave band (Hz)")
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("head-zone reduction (dB)")
    _title(ax, "Broadband ANC: limited by the diffuse field and reference coherence")
    return fig


def all_figures(results: dict) -> dict[str, Figure]:
    """Every figure, keyed by a file-friendly name (used by the CLI --plots option)."""
    figures = {
        "projection": projection_figure(results),
        "source_spectrum": source_spectrum_figure(results),
        "transmission_loss": transmission_loss_figure(results),
        "tbl_coupling_penalty": coupling_penalty_figure(results),
        "head_spectrum": cabin_spectrum_figure(results, "head"),
        "room_spectrum": cabin_spectrum_figure(results, "room"),
        "sil": sil_figure(results),
        "shunt": shunt_figure(results),
    }
    for name, solution in results["solutions"].items():
        figures[f"paths_{name}"] = path_breakdown_figure(results, name)
        if "active_trim_result" in solution:
            figures[f"anc_convergence_{name}"] = anc_convergence_figure(results, name)
            figures[f"anc_tones_{name}"] = anc_tones_figure(results, name)
            figures[f"anc_broadband_{name}"] = anc_broadband_figure(results, name)
    return figures
