import copy

from cabinsim.config_writer import CHANGED_NOTE, write_config
from cabinsim.params import SECTIONS, load_params


def test_round_trip_keeps_values_and_tags(params, tmp_path):
    write_config(params, tmp_path)
    reloaded = load_params(tmp_path)
    for section in SECTIONS:
        assert getattr(reloaded, section) == getattr(params, section)
    original = {r.path: r.source for r in params.records if not r.path.startswith("solutions.")}
    again = {r.path: r.source for r in reloaded.records if not r.path.startswith("solutions.")}
    assert original == again


def test_changed_value_is_retagged_as_assumption(params, tmp_path):
    edited = params.copy()
    edited.engine["n1_speed"] = 2400.0
    write_config(edited, tmp_path)
    record = next(r for r in load_params(tmp_path).records if r.path == "engine.n1_speed")
    assert record.value == 2400.0
    assert record.source == "ASM" and CHANGED_NOTE in record.note


def test_new_solution_copied_from_existing(params, tmp_path):
    edited = params.copy()
    v4 = copy.deepcopy(edited.solutions["V1_passive"])
    v4.update(label="V4 heavy septum", _copied_from="V1_passive")
    v4["septum"]["surface_mass"] = 2.0
    edited.solutions["V4_heavy_septum"] = v4
    write_config(edited, tmp_path)
    reloaded = load_params(tmp_path)
    assert reloaded.solutions["V4_heavy_septum"]["septum"]["surface_mass"] == 2.0
    assert reloaded.solutions["V4_heavy_septum"]["label"] == "V4 heavy septum"
