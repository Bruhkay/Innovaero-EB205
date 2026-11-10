"""Assumption register:  python -m cabinsim.assumptions [--all] [--markdown FILE]

Lists every configuration value tagged [ASM] (or all tags with --all), with its note and the
matching item of the archive's verification list where there is one.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .params import ParamRecord, load_params


def format_value(value) -> str:
    text = str(value)
    return text if len(text) <= 40 else text[:37] + "..."


def select_records(records: list[ParamRecord], include_all: bool) -> list[ParamRecord]:
    """Default: physical [ASM] values only. Numerical resolution settings (numerics.*) and
    DOC/LIT values are shown with --all."""
    if include_all:
        return list(records)
    return [r for r in records if r.source == "ASM" and not r.path.startswith("numerics.")]


def as_markdown(records: list[ParamRecord]) -> str:
    lines = ["| parameter | value | unit | tag | note | archive verification item |",
             "|---|---|---|---|---|---|"]
    for r in records:
        note = r.note.replace("|", "/")
        verify = (r.verify or "").replace("|", "/")
        lines.append(f"| `{r.path}` | {format_value(r.value)} | {r.unit} | {r.source} | {note} | {verify} |")
    return "\n".join(lines) + "\n"


def as_text(records: list[ParamRecord]) -> str:
    lines = []
    for r in records:
        lines.append(f"[{r.source}] {r.path} = {format_value(r.value)} {r.unit}")
        lines.append(f"       {r.note}")
        if r.verify:
            lines.append(f"       verify: {r.verify}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="List tagged assumptions in the configuration")
    parser.add_argument("--all", action="store_true", help="include DOC and LIT values too")
    parser.add_argument("--markdown", type=Path, default=None, help="also write a Markdown table to this file")
    parser.add_argument("--config-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    records = select_records(load_params(args.config_dir).records, args.all)
    print(as_text(records))
    counts = {tag: sum(r.source == tag for r in records) for tag in ("DOC", "LIT", "ASM")}
    print(f"\n{len(records)} values listed  (DOC {counts['DOC']}, LIT {counts['LIT']}, ASM {counts['ASM']})")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(as_markdown(records), encoding="utf-8")
        print(f"written: {args.markdown}")


if __name__ == "__main__":
    main()
