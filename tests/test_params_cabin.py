import numpy as np
import pytest

from cabinsim.cabin import CabinRoom, room_constant_m2
from cabinsim.params import VALID_SOURCES, load_params


def test_every_value_has_a_valid_tag(params):
    assert params.records
    assert all(r.source in VALID_SOURCES for r in params.records)


def test_solutions_inherit_and_stack(params):
    v1, v3 = params.solutions["V1_passive"], params.solutions["V3_active"]
    assert v3["blanket"] == v1["blanket"]                    # inherited through V2
    assert v3["piezo_shunt"] and v3["active_trim"]
    assert "extends" not in v3


def test_assumption_register_lists_asm_values(params):
    asm_paths = {r.path for r in params.records if r.source == "ASM"}
    assert "engine.n1_speed" in asm_paths
    assert "solutions.V1_passive.septum.surface_mass" in asm_paths


def test_room_geometry_full_circle_when_floor_at_bottom():
    room = CabinRoom(length_m=1.0, trim_radius_m=2.0, floor_z_m=-2.0)
    assert room.cross_section_m2 == pytest.approx(np.pi * 4.0, rel=1e-9)


def test_room_constant():
    assert room_constant_m2(100.0, np.array([0.5]))[0] == pytest.approx(100.0)


def test_custom_config_dir_is_read(tmp_path):
    import shutil
    from cabinsim.params import DEFAULT_CONFIG_DIR
    for f in DEFAULT_CONFIG_DIR.glob("*.yaml"):
        shutil.copy(f, tmp_path / f.name)
    text = (tmp_path / "engine.yaml").read_text(encoding="utf-8").replace("value: 2300.0", "value: 2400.0")
    (tmp_path / "engine.yaml").write_text(text, encoding="utf-8")
    assert load_params(tmp_path).engine["n1_speed"] == 2400.0
