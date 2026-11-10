"""EB205 cabin noise - team tool.

Run from the repository folder:   python -m streamlit run app/app.py
The physics is in the `cabinsim` package; this file only builds the page.
"""
from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))                   # so `import controls` works
import controls  # noqa: E402

from cabinsim import plots  # noqa: E402
from cabinsim.auralisation import common_gain, synthesise, wav_bytes  # noqa: E402
from cabinsim.config_writer import config_zip_bytes  # noqa: E402
from cabinsim.export import json_text, spectra_csv_text, spectra_table, summary_csv_text, summary_rows  # noqa: E402
from cabinsim.params import SECTIONS, Params, load_params  # noqa: E402
from cabinsim.pipeline import run  # noqa: E402

st.set_page_config(page_title="EB205 cabin noise", layout="wide")


@st.cache_resource
def default_params() -> Params:
    return load_params()


@st.cache_data(show_spinner=False, max_entries=8)
def run_model(key: str, fast: bool, _params: Params) -> dict:
    """Cached by `key` (JSON of all inputs), so the same inputs never run twice."""
    log_area = st.empty()
    return run(_params, fast=fast, log=lambda message: log_area.caption(message))


def params_key(params: Params) -> str:
    sections = {name: getattr(params, name) for name in SECTIONS}
    return json.dumps(sections, sort_keys=True, default=str)


def show(figure) -> None:
    st.pyplot(figure, clear_figure=True)
    plt.close(figure)


# ------------------------------------------------------------------ sidebar: inputs
params = default_params().copy()
tags = controls.tag_lookup(params)
with st.sidebar:
    st.title("EB205 cabin noise")
    st.caption("Values start from the YAML config. Hover the (?) to see each value's source tag.")
    fast = st.toggle("Fast mode (coarser numerics, 4 s ANC)", value=True)
    run_clicked = st.button("Run model", type="primary", width="stretch")
    controls.flight_controls(params, tags)
    controls.engine_controls(params, tags)
    controls.fuselage_controls(params, tags)
    controls.tbl_and_bridge_controls(params, tags)
    controls.piezo_controls(params, tags)
    controls.active_controls(params, tags)
    controls.materials_editor(params)
    all_names = controls.solutions_editor(params)

if run_clicked:
    if not params.solutions:
        st.error("Enable at least one solution.")
    else:
        with st.spinner("Running the model (about 1 minute; repeated inputs are cached)..."):
            st.session_state["results"] = run_model(params_key(params), fast, params)
            st.session_state["run_params"] = params

results = st.session_state.get("results")
if results is None:
    st.info("Set the inputs in the sidebar and press **Run model**.")
    st.stop()

run_params: Params = st.session_state["run_params"]
# colour per solution from the full list (incl. disabled ones), so hiding one never repaints the others
colors = plots.solution_colors(results, order=all_names + [n for n in results["solutions"] if n not in all_names])
names = list(results["solutions"])
if params_key(params) != params_key(run_params):
    st.warning("Inputs changed since the last run. Press **Run model** to update the results.")

tabs = st.tabs(["Summary", "Cabin spectra", "Sidewall", "Source & projection", "Piezo shunt", "Active trim",
                "Assumptions", "Export", "Listen"])

# ------------------------------------------------------------------ summary
with tabs[0]:
    st.subheader("Single-number results at the passenger's head")
    table = pd.DataFrame(summary_rows(results)).set_index("solution")
    st.dataframe(table.style.format(precision=1), width="stretch")
    show(plots.sil_figure(results, colors=colors))
    st.caption("Head = window seat on the engine side; room = reverberant average. "
               "Absolute levels depend on [ASM] inputs; differences between solutions are more reliable.")
    closed = pd.Series(results["closed_form"], name="value").to_frame()
    st.subheader("Characteristic frequencies (baseline skin)")
    st.dataframe(closed.style.format(precision=1), width="stretch")

# ------------------------------------------------------------------ spectra
with tabs[1]:
    location = st.radio("Location", ["head", "room"], horizontal=True,
                        format_func=lambda x: "passenger's head" if x == "head" else "room average")
    show(plots.cabin_spectrum_figure(results, location, colors=colors))
    chosen = st.selectbox("Noise paths of", names, format_func=lambda n: results["solutions"][n]["label"])
    show(plots.path_breakdown_figure(results, chosen))
    with st.expander("Data table"):
        st.dataframe(pd.DataFrame(spectra_table(results)).set_index("band_hz").round(1), width="stretch")

# ------------------------------------------------------------------ sidewall
with tabs[2]:
    show(plots.transmission_loss_figure(results, colors=colors))
    show(plots.coupling_penalty_figure(results, colors=colors))
    with st.expander("Data table"):
        tl = {results["solutions"][n]["label"]: results["solutions"][n]["tl_diffuse_db"]["total"] for n in names}
        st.dataframe(pd.DataFrame(tl, index=results["bands_hz"]).round(1), width="stretch")

# ------------------------------------------------------------------ source and projection
with tabs[3]:
    show(plots.source_spectrum_figure(results))
    show(plots.projection_figure(results))
    flight = results["flight"]
    st.dataframe(pd.DataFrame({"outside": flight["outside"], "cabin": flight["cabin"]}).style.format(precision=4),
                 width="stretch")
    st.caption(f"Flight speed {flight['flight_speed_m_s']:.1f} m/s, cabin-to-outside pressure difference "
               f"{flight['pressure_difference_pa'] / 1000:.1f} kPa, TBL rms pressure {results['tbl']['rms_pressure_pa']:.1f} Pa")

# ------------------------------------------------------------------ piezo shunt
with tabs[4]:
    shunted = [n for n in names if results["solutions"][n]["piezo_shunt"]]
    if not shunted:
        st.info("No solution with shunted piezo patches in this run.")
    else:
        show(plots.shunt_figure(results, colors=colors))
        for n in shunted:
            st.markdown(f"**{results['solutions'][n]['label']}**: patch tuning")
            st.dataframe(pd.DataFrame(results["solutions"][n]["shunt_patches"]).round(4), width="stretch")

# ------------------------------------------------------------------ active trim
with tabs[5]:
    active = [n for n in names if "active_trim_result" in results["solutions"][n]]
    if not active:
        st.info("No solution with an active trim panel in this run.")
    for n in active:
        anc = results["solutions"][n]["active_trim_result"]
        st.subheader(results["solutions"][n]["label"])
        cols = st.columns(4)
        cols[0].metric("Max actuator voltage", f"{anc['max_voltage_rms']:.1f} V rms", help=f"limit {anc['voltage_limit_rms']:.0f} V")
        cols[1].metric("Head SIL, tonal ANC", f"{anc['head_with_anc']['sil_db']:.1f} dB")
        cols[2].metric("Tonal converged", "yes" if anc["tonal"]["converged"] else "no")
        cols[3].metric("Broadband converged", "yes" if anc["broadband"]["converged"] else "no")
        show(plots.anc_convergence_figure(results, n))
        show(plots.anc_tones_figure(results, n))
        show(plots.anc_broadband_figure(results, n))
        share = pd.DataFrame({"tone (Hz)": anc["tone_frequencies_hz"],
                              "share of head tone from controlled panel": anc["tone_controlled_share_at_head"]})
        st.dataframe(share.round(3), width="stretch", hide_index=True)

# ------------------------------------------------------------------ assumptions
def value_at(params: Params, path: str):
    """Value used in the run for a dotted record path, e.g. 'engine.n1_speed' (None if removed)."""
    section, *keys = path.split(".")
    node = getattr(params, section)
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


with tabs[6]:
    rows = []
    for r in default_params().records:
        used = value_at(run_params, r.path)
        rows.append(dict(parameter=r.path, default=str(r.value), used_in_run=str(used), changed=used != r.value,
                         unit=r.unit, tag=r.source, note=r.note, verify=r.verify or ""))
    frame = pd.DataFrame(rows)
    chosen_tags = st.multiselect("Show tags", ["ASM", "LIT", "DOC"], default=["ASM"])
    hide_numerics = st.checkbox("Hide numerical settings (numerics.*)", value=True)
    view = frame[frame["tag"].isin(chosen_tags)]
    if hide_numerics:
        view = view[~view["parameter"].str.startswith("numerics.")]
    st.caption(f"{len(view)} values. 'changed' marks values edited in the sidebar; they are written as "
               "ASM [changed in app] in the exported config.")
    st.dataframe(view, width="stretch", hide_index=True)

# ------------------------------------------------------------------ export
with tabs[7]:
    st.download_button("results.json", json_text(results), "results.json", "application/json")
    st.download_button("spectra.csv", spectra_csv_text(results), "spectra.csv", "text/csv")
    st.download_button("summary.csv", summary_csv_text(results), "summary.csv", "text/csv")
    st.download_button("config (YAML, zip)", config_zip_bytes(run_params), "cabinsim_config.zip", "application/zip",
                       help="Use with: python -m cabinsim.run --config-dir config")
    if st.button("Prepare figures (PNG, zip)"):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for figure_name, figure in plots.all_figures(results).items():
                image = io.BytesIO()
                figure.savefig(image, dpi=160, format="png")
                plt.close(figure)
                archive.writestr(f"figures/{figure_name}.png", image.getvalue())
        st.download_button("figures.zip", buffer.getvalue(), "figures.zip", "application/zip")


# ------------------------------------------------------------------ listen
@st.cache_data(show_spinner=False, max_entries=4)
def listening_files(key: str, _results: dict, _visuals: dict) -> list[tuple[str, float, float, bytes, int]]:
    """(label, SIL, dB(A), wav bytes, sample rate) per case; one common gain keeps relative levels."""
    cases = synthesise(_results, _visuals)
    gain = common_gain([c.signal_pa for c in cases], _visuals["audio_headroom_db"])
    rate = int(_visuals["audio_sample_rate"])
    return [(c.label, c.sil_db, c.dba, wav_bytes(c.signal_pa, gain, rate), rate) for c in cases]


with tabs[8]:
    st.caption("Sound at the passenger's head. All files share one gain, so the differences are the model's "
               "differences; the absolute loudness on your headphones cannot be guaranteed. Set the volume once "
               "on the baseline. The jury version with instant A/B switching is demo/listen.html.")
    with st.spinner("Synthesising sound files..."):
        files = listening_files(params_key(run_params) + json.dumps(results["bands_hz"]), results, run_params.visuals)
    for label, sil, dba, data, rate in files:
        st.markdown(f"**{label}**: SIL {sil:.1f} dB, {dba:.1f} dB(A)")
        st.audio(data, format="audio/wav")
