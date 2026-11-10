"""Load the YAML configuration into plain Python values and keep a record of every source tag.

Each YAML number is written as ``{value, unit, source, note, verify}``. The loader returns the
bare ``value`` to the physics code and stores the rest in a list of :class:`ParamRecord`,
which the assumption register (``python -m cabinsim.assumptions``) prints.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_DIR = Path(__file__).parent / "config"
SECTIONS = ("flight", "engine", "fuselage", "cabin", "materials", "solutions",
            "piezo", "tbl", "bridge", "active", "numerics", "visuals")
VALID_SOURCES = ("DOC", "LIT", "ASM")


@dataclass(frozen=True)
class ParamRecord:
    """One tagged number from the configuration."""
    path: str
    value: Any
    unit: str
    source: str
    note: str
    verify: str | None


@dataclass
class Params:
    """All configuration sections as nested dicts of plain values."""
    flight: dict
    engine: dict
    fuselage: dict
    cabin: dict
    materials: dict
    solutions: dict
    piezo: dict
    tbl: dict
    bridge: dict
    active: dict
    numerics: dict
    visuals: dict
    records: list[ParamRecord] = field(default_factory=list)

    def copy(self) -> "Params":
        return copy.deepcopy(self)


def _is_tagged_leaf(node: Any) -> bool:
    return isinstance(node, dict) and "value" in node and "source" in node


def _strip_tags(node: Any, path: str, records: list[ParamRecord]) -> Any:
    """Replace every tagged leaf by its value and append its tags to `records`."""
    if _is_tagged_leaf(node):
        source = str(node["source"])
        if source not in VALID_SOURCES:
            raise ValueError(f"{path}: source must be one of {VALID_SOURCES}, got {source!r}")
        records.append(ParamRecord(path=path, value=node["value"], unit=str(node.get("unit", "")),
                                   source=source, note=str(node.get("note", "")),
                                   verify=node.get("verify")))
        return node["value"]
    if isinstance(node, dict):
        return {key: _strip_tags(child, f"{path}.{key}", records) for key, child in node.items()}
    return node


def _deep_merge(base: dict, override: dict) -> dict:
    """Return `base` updated with `override`; nested dicts are merged key by key."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _resolve_solutions(raw: dict, records: list[ParamRecord]) -> dict:
    """Strip tags per solution, then apply `extends:` inheritance in file order."""
    own = {name: _strip_tags(block, f"solutions.{name}", records) for name, block in raw.items()}
    resolved: dict[str, dict] = {}
    for name, block in own.items():
        parent = block.get("extends")
        if parent is None:
            resolved[name] = block
            continue
        if parent not in resolved:
            raise ValueError(f"solutions.{name}: parent {parent!r} must be defined above it")
        child = {key: value for key, value in block.items() if key != "extends"}
        resolved[name] = _deep_merge(resolved[parent], child)
    return resolved


def load_params(config_dir: Path | str | None = None) -> Params:
    """Read all YAML sections from `config_dir` (default: the package's own config folder)."""
    folder = Path(config_dir) if config_dir is not None else DEFAULT_CONFIG_DIR
    records: list[ParamRecord] = []
    sections: dict[str, dict] = {}
    for name in SECTIONS:
        with open(folder / f"{name}.yaml", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if name == "solutions":
            sections[name] = _resolve_solutions(raw, records)
        else:
            sections[name] = _strip_tags(raw, name, records)
    return Params(**sections, records=records)
