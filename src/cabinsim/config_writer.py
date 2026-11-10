"""Write a (possibly edited) Params object back to tagged YAML files.

Values that still equal the source configuration keep their original tag and note. A value that
was changed (e.g. in the Streamlit app) is re-tagged ASM with "[changed in app]" in its note, and a
new number is written as ASM "[added in app]". Solutions are written fully resolved (no `extends`),
so every solution block stands on its own. YAML comments are not preserved.
"""
from __future__ import annotations

import copy
import io
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .params import DEFAULT_CONFIG_DIR, SECTIONS, Params, _deep_merge, _is_tagged_leaf

CHANGED_NOTE = "[changed in app]"
ADDED_NOTE = "[added in app]"


def _plain(value: Any) -> Any:
    """numpy scalars/arrays and tuples -> plain Python types that yaml.safe_dump accepts."""
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value


def _is_number_like(value: Any) -> bool:
    return isinstance(value, (int, float, list)) and not isinstance(value, bool)


def _resolved_raw(config_dir: Path) -> dict[str, dict]:
    """Raw tagged YAML of every section; solutions with `extends` merged in."""
    raw = {}
    for name in SECTIONS:
        with open(config_dir / f"{name}.yaml", encoding="utf-8") as handle:
            raw[name] = yaml.safe_load(handle) or {}
    resolved: dict[str, dict] = {}
    for name, block in raw["solutions"].items():
        parent = block.get("extends")
        own = {k: v for k, v in block.items() if k != "extends"}
        resolved[name] = _deep_merge(resolved[parent], own) if parent else own
    raw["solutions"] = resolved
    return raw


def _merge_value(raw_node: Any, value: Any) -> Any:
    """Put `value` into the raw (tagged) structure, re-tagging anything that changed."""
    value = _plain(value)
    if _is_tagged_leaf(raw_node):
        if raw_node["value"] == value:
            return raw_node
        updated = dict(raw_node, value=value, source="ASM")
        if CHANGED_NOTE not in str(updated.get("note", "")):
            updated["note"] = f"{updated.get('note', '')} {CHANGED_NOTE}".strip()
        return updated
    if isinstance(value, dict):
        raw_children = raw_node if isinstance(raw_node, dict) else {}
        return {key: _merge_value(raw_children.get(key), child) for key, child in value.items()}
    if raw_node is None and _is_number_like(value):
        return dict(value=value, unit="", source="ASM", note=ADDED_NOTE)
    return value


def tagged_sections(params: Params, base_dir: Path | None = None) -> dict[str, dict]:
    """Every section as a tagged dict, ready for yaml.safe_dump."""
    raw = _resolved_raw(Path(base_dir) if base_dir else DEFAULT_CONFIG_DIR)
    result = {}
    for name in SECTIONS:
        if name == "solutions":
            sources = raw["solutions"]
            result[name] = {}
            for solution_name, solution in params.solutions.items():
                template = sources.get(solution_name) or sources.get(solution.get("_copied_from", ""), {})
                clean = {k: v for k, v in solution.items() if not k.startswith("_")}
                result[name][solution_name] = _merge_value(copy.deepcopy(template), clean)
        else:
            result[name] = _merge_value(raw[name], getattr(params, name))
    return result


def yaml_texts(params: Params, base_dir: Path | None = None) -> dict[str, str]:
    """{'engine.yaml': '...', ...}"""
    header = "# Written by cabinsim.config_writer. Values marked [changed in app] differ from the source config.\n"
    return {f"{name}.yaml": header + yaml.safe_dump(section, sort_keys=False, allow_unicode=True, width=120)
            for name, section in tagged_sections(params, base_dir).items()}


def write_config(params: Params, folder: Path, base_dir: Path | None = None) -> None:
    """Write all sections to `folder`; usable afterwards with `--config-dir folder`."""
    folder.mkdir(parents=True, exist_ok=True)
    for filename, text in yaml_texts(params, base_dir).items():
        (folder / filename).write_text(text, encoding="utf-8")


def config_zip_bytes(params: Params, base_dir: Path | None = None) -> bytes:
    """All YAML files in one zip (for a download button)."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for filename, text in yaml_texts(params, base_dir).items():
            archive.writestr(f"config/{filename}", text)
    return buffer.getvalue()
