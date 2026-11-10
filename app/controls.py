"""Sidebar input widgets for the Streamlit app.

Each function reads the default value from a Params object, shows a widget, and writes the
widget's value back into that (copied) Params object. Widget help texts show the source tag and
note from the YAML file, so the team always sees where a number comes from.
"""
from __future__ import annotations

import copy

import pandas as pd
import streamlit as st

from cabinsim.params import Params


def tag_lookup(params: Params) -> dict[str, str]:
    """'engine.n1_speed' -> '[ASM] Cruise N1 ...' for widget help texts."""
    return {r.path: f"[{r.source}] {r.note}" for r in params.records}


def number(section: dict, key: str, label: str, path: str, tags: dict, *, scale: float = 1.0,
           step: float | None = None, fmt: str | None = None, min_value: float | None = None) -> None:
    """Number input bound to section[key]; `scale` converts to display units (e.g. Pa -> GPa)."""
    shown = float(section[key]) * scale
    value = st.number_input(label, value=shown, step=step, format=fmt,
                            min_value=min_value, help=tags.get(path, ""), key=f"in:{path}")
    section[key] = unscaled(section[key], shown, value, scale)


def unscaled(original, shown: float, value: float, scale: float):
    """Write back only a real edit, so unit conversion round-off never counts as a change."""
    return original if value == shown else value / scale


def flight_controls(p: Params, tags: dict) -> None:
    with st.expander("Flight"):
        number(p.flight, "cruise_altitude", "Cruise altitude (m)", "flight.cruise_altitude", tags, step=100.0)
        number(p.flight, "cruise_mach", "Cruise Mach", "flight.cruise_mach", tags, step=0.01)
        number(p.flight, "cabin_altitude", "Cabin pressure altitude (m)", "flight.cabin_altitude", tags, step=100.0)


def engine_controls(p: Params, tags: dict) -> None:
    with st.expander("Engine source"):
        number(p.engine, "acoustic_power", "Acoustic power per engine (W)", "engine.acoustic_power", tags, step=1.0)
        number(p.engine, "n1_speed", "N1 speed (rpm)", "engine.n1_speed", tags, step=50.0)
        number(p.engine, "tone_energy_fraction", "Tone energy fraction", "engine.tone_energy_fraction", tags, step=0.05)
        position = list(p.engine["position"])
        position[1] = st.number_input("Engine spanwise position y (m)", value=float(position[1]), step=0.1,
                                      help=tags.get("engine.position", ""), key="in:engine.position.y")
        position[2] = st.number_input("Engine vertical position z (m)", value=float(position[2]), step=0.1,
                                      help=tags.get("engine.position", ""), key="in:engine.position.z")
        p.engine["position"] = position
        p.engine["mirror_left_engine"] = st.checkbox("Include left engine (mirror)", value=p.engine["mirror_left_engine"],
                                                     help=tags.get("engine.mirror_left_engine", ""))
        number(p.engine, "shadow_loss", "Shadow loss (dB)", "engine.shadow_loss", tags, step=1.0)


def fuselage_controls(p: Params, tags: dict) -> None:
    with st.expander("Fuselage and cabin"):
        number(p.fuselage, "frame_pitch", "Frame pitch (m)", "fuselage.frame_pitch", tags, step=0.01)
        number(p.fuselage, "stringer_pitch", "Stringer pitch (m)", "fuselage.stringer_pitch", tags, step=0.01)
        number(p.fuselage, "sidewall_depth", "Sidewall depth d (m)", "fuselage.sidewall_depth", tags, step=0.005, fmt="%.3f")
        p.fuselage["pressurised_skin_tension"] = st.checkbox(
            "Skin tension from cabin pressure", value=p.fuselage["pressurised_skin_tension"],
            help=tags.get("fuselage.pressurised_skin_tension", ""))
        number(p.cabin, "seat_x", "Seat position x (m)", "cabin.seat_x", tags, step=0.5)
        number(p.cabin, "head_distance_from_trim", "Head distance from trim (m)", "cabin.head_distance_from_trim", tags, step=0.05)
        scale = st.slider("Cabin absorption scale (x default)", 0.5, 1.5, 1.0, 0.05,
                          help="Multiplies the absorption coefficients of cabin.yaml (capped at 0.95)")
        p.cabin["absorption_coefficients"] = [min(a * scale, 0.95) for a in p.cabin["absorption_coefficients"]]


def tbl_and_bridge_controls(p: Params, tags: dict) -> None:
    with st.expander("Boundary layer and bridge path"):
        p.tbl["enabled"] = st.checkbox("Include TBL path", value=p.tbl["enabled"], help=tags.get("tbl.enabled", ""))
        number(p.tbl, "wall_shear_stress", "Wall shear stress (Pa)", "tbl.wall_shear_stress", tags, step=0.5)
        number(p.tbl, "boundary_layer_thickness", "Boundary layer thickness (m)", "tbl.boundary_layer_thickness", tags, step=0.01)
        number(p.tbl, "convection_velocity_ratio", "Convection velocity U_c / U", "tbl.convection_velocity_ratio", tags, step=0.05)
        models = ["mellen", "corcos"]
        p.tbl["wavenumber_model"] = st.selectbox("TBL wavenumber model", models, index=models.index(p.tbl["wavenumber_model"]),
                                                 help=tags.get("tbl.wavenumber_model", ""))
        p.bridge["enabled"] = st.checkbox("Include bridge path", value=p.bridge["enabled"], help=tags.get("bridge.enabled", ""))
        number(p.bridge, "isolator_gain", "Trim isolator gain (dB)", "bridge.isolator_gain", tags, step=1.0)
        number(p.bridge, "point_spacing", "Attachment spacing (m)", "bridge.point_spacing", tags, step=0.02)


def piezo_controls(p: Params, tags: dict) -> None:
    with st.expander("Piezo shunt"):
        kinds = ["RL", "R", "negC"]
        p.piezo["shunt_type"] = st.selectbox("Shunt circuit", kinds, index=kinds.index(p.piezo["shunt_type"]),
                                             help=tags.get("piezo.shunt_type", ""))
        conditions = ["cruise", "ground"]
        p.piezo["tuning_condition"] = st.selectbox("Tune shunts for", conditions,
                                                   index=conditions.index(p.piezo["tuning_condition"]),
                                                   help=tags.get("piezo.tuning_condition", ""))
        number(p.piezo, "patch_thickness", "Patch thickness (mm)", "piezo.patch_thickness", tags, scale=1e3, step=0.05)


def active_controls(p: Params, tags: dict) -> None:
    with st.expander("Active trim panel"):
        number(p.active, "reference_coherence", "Broadband reference coherence", "active.reference_coherence", tags,
               step=0.05, min_value=0.05)
        number(p.active, "step_size_tonal", "Tonal step size", "active.step_size_tonal", tags, step=0.0005, fmt="%.4f")
        number(p.active, "regularisation_tonal", "Tonal regularisation", "active.regularisation_tonal", tags, step=0.05)
        number(p.active, "duration", "Simulated time (s)", "active.duration", tags, step=1.0, min_value=3.0)


def materials_editor(p: Params) -> None:
    """Editable tables of solids, blankets and panels; new rows become new materials."""
    with st.expander("Material database"):
        st.caption("Edit values or add rows. New materials can then be chosen in the solutions.")
        solids = pd.DataFrame([dict(name=name, E_GPa=m["youngs_modulus"] / 1e9, density=m["density"],
                                    poisson=m["poisson_ratio"], loss_factor=m["loss_factor"])
                               for name, m in p.materials["solids"].items()])
        edited = st.data_editor(solids, num_rows="dynamic", key="ed:solids", hide_index=True)
        if not edited.equals(solids):
            p.materials["solids"] = {row["name"]: dict(youngs_modulus=row["E_GPa"] * 1e9, density=row["density"],
                                                       poisson_ratio=row["poisson"], loss_factor=row["loss_factor"])
                                     for row in edited.dropna().to_dict("records") if row["name"]}
        porous = pd.DataFrame([dict(name=name, flow_resistivity=m["flow_resistivity"], density=m["density"])
                               for name, m in p.materials["porous"].items()])
        edited = st.data_editor(porous, num_rows="dynamic", key="ed:porous", hide_index=True)
        if not edited.equals(porous):
            p.materials["porous"] = {row["name"]: dict(flow_resistivity=row["flow_resistivity"], density=row["density"])
                                     for row in edited.dropna().to_dict("records") if row["name"]}


def _solution_names(p: Params) -> list[str]:
    """Defaults plus solutions added in this session, in a stable order (drives the colours)."""
    added = st.session_state.setdefault("added_solutions", {})
    return list(p.solutions) + [name for name in added if name not in p.solutions]


def solutions_editor(p: Params) -> list[str]:
    """Edit, enable/disable and add solutions. Returns the full ordered name list."""
    added: dict = st.session_state.setdefault("added_solutions", {})
    for name, block in added.items():
        p.solutions.setdefault(name, copy.deepcopy(block))
    names = _solution_names(p)
    solids, porous = list(p.materials["solids"]), list(p.materials["porous"])

    st.subheader("Solutions")
    disabled = []
    for name in names:
        solution = p.solutions[name]
        with st.expander(solution["label"]):
            enabled = st.checkbox("Include in run", value=True, key=f"sol:{name}:enabled")
            solution["label"] = st.text_input("Label", value=solution["label"], key=f"sol:{name}:label")
            skin = solution["skin"]
            skin["material"] = st.selectbox("Skin material", solids, index=_index(solids, skin["material"]), key=f"sol:{name}:skin")
            shown = skin["thickness"] * 1e3
            value = st.number_input("Skin thickness (mm)", value=shown, step=0.25, key=f"sol:{name}:t")
            skin["thickness"] = unscaled(skin["thickness"], shown, value, 1e3)
            skin["added_mass"] = st.number_input("Damping layer mass (kg/m²)", value=float(skin["added_mass"]), step=0.1,
                                                 key=f"sol:{name}:cld_m")
            skin["added_loss_factor"] = st.number_input("Damping layer added loss factor", value=float(skin["added_loss_factor"]),
                                                        step=0.01, key=f"sol:{name}:cld_eta")
            blanket = solution["blanket"]
            blanket["material"] = st.selectbox("Blanket", porous, index=_index(porous, blanket["material"]), key=f"sol:{name}:bl")
            shown = blanket["thickness"] * 1e3
            value = st.number_input("Blanket thickness (mm)", value=shown, step=5.0, key=f"sol:{name}:bl_t")
            blanket["thickness"] = unscaled(blanket["thickness"], shown, value, 1e3)
            septum = solution["septum"]
            septum["enabled"] = st.checkbox("Septum", value=septum["enabled"], key=f"sol:{name}:sep")
            septum["surface_mass"] = st.number_input("Septum mass (kg/m²)", value=float(septum["surface_mass"]), step=0.25,
                                                     key=f"sol:{name}:sep_m")
            solution["trim"]["added_mass"] = st.number_input("Trim added mass (kg/m²)", value=float(solution["trim"]["added_mass"]),
                                                             step=0.05, key=f"sol:{name}:trim_m")
            solution["piezo_shunt"] = st.checkbox("Shunted piezo patches", value=solution["piezo_shunt"], key=f"sol:{name}:pz")
            solution["active_trim"] = st.checkbox("Active trim panel", value=solution["active_trim"], key=f"sol:{name}:anc")
            if not enabled:
                disabled.append(name)

    with st.expander("Add a solution"):
        new_name = st.text_input("Key (letters, digits, _)", value="V4_custom", key="new:name")
        source = st.selectbox("Copy from", names, key="new:source")
        if st.button("Add") and new_name and new_name not in names:
            block = copy.deepcopy(p.solutions[source])
            block.update(label=new_name.replace("_", " "), _copied_from=source)
            added[new_name] = block
            st.rerun()

    for name in disabled:
        del p.solutions[name]
    return names


def _index(options: list[str], value: str) -> int:
    return options.index(value) if value in options else 0
