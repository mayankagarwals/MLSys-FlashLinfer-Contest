"""
Extract workload JSONL files into human-readable Markdown files.

The output directory mirrors the source directory structure, converting:
  path/to/file.jsonl -> workloads/path/to/file.md
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_ROOT = Path("/home/simon/flashinfer-competition/mlsys26-contest/workloads")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "workloads"


def escape_md(value: Any) -> str:
    text = str(value)
    return text.replace("|", r"\|").replace("\n", "<br>")


def format_json(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=True)


def render_axes(axes: dict[str, Any]) -> list[str]:
    if not axes:
        return ["_none_"]

    lines = [
        "| Axis | Value |",
        "| --- | --- |",
    ]
    for key, value in axes.items():
        lines.append(f"| `{escape_md(key)}` | `{escape_md(format_json(value))}` |")
    return lines


def render_inputs(inputs: dict[str, Any]) -> list[str]:
    if not inputs:
        return ["_none_"]

    lines = [
        "| Input | Type | Value | Tensor Key | Path |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, spec in inputs.items():
        input_type = spec.get("type", "-")
        value = "-" if "value" not in spec else format_json(spec["value"])
        tensor_key = spec.get("tensor_key", "-")
        path = spec.get("path", "-")
        lines.append(
            f"| `{escape_md(name)}` | `{escape_md(input_type)}` | "
            f"`{escape_md(value)}` | `{escape_md(tensor_key)}` | `{escape_md(path)}` |"
        )
    return lines


def render_record(idx: int, record: dict[str, Any]) -> list[str]:
    definition = record.get("definition", "-")
    solution = record.get("solution")
    evaluation = record.get("evaluation")
    workload = record.get("workload", {})
    uuid = workload.get("uuid", f"record-{idx}")
    axes = workload.get("axes", {})
    inputs = workload.get("inputs", {})

    lines = [
        f"## Record {idx}: `{escape_md(uuid)}`",
        "",
        f"- Definition: `{escape_md(definition)}`",
        f"- Solution: `{escape_md(format_json(solution))}`",
        f"- Evaluation: `{escape_md(format_json(evaluation))}`",
        "",
        "### Axes",
        "",
    ]
    lines.extend(render_axes(axes))
    lines.extend(["", "### Inputs", ""])
    lines.extend(render_inputs(inputs))
    lines.append("")
    return lines


def render_markdown(records: list[dict[str, Any]], source_rel_path: Path) -> str:
    title = source_rel_path.stem
    first_definition = records[0].get("definition", "-") if records else "-"

    lines = [
        f"# {escape_md(title)}",
        "",
        "## Summary",
        "",
        f"- Source JSONL: `{source_rel_path.as_posix()}`",
        f"- Definition: `{escape_md(first_definition)}`",
        "",
    ]

    for idx, record in enumerate(records, start=1):
        lines.extend(render_record(idx, record))

    return "\n".join(lines).rstrip() + "\n"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_no}, got {type(row).__name__}")
            records.append(row)
    return records


def extract_workloads(source_root: Path, output_root: Path, clean: bool = False) -> int:
    if not source_root.exists():
        raise FileNotFoundError(f"Source directory not found: {source_root}")

    if clean and output_root.exists():
        shutil.rmtree(output_root)

    jsonl_files = sorted(source_root.rglob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found under: {source_root}")

    converted = 0
    for jsonl_path in jsonl_files:
        rel_path = jsonl_path.relative_to(source_root)
        md_path = (output_root / rel_path).with_suffix(".md")
        md_path.parent.mkdir(parents=True, exist_ok=True)

        records = read_jsonl(jsonl_path)
        markdown = render_markdown(records, rel_path)
        md_path.write_text(markdown, encoding="utf-8")
        converted += 1

    return converted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert workload JSONL files to markdown documents."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help=f"Directory containing workload JSONL files (default: {DEFAULT_SOURCE_ROOT})",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Directory where markdown files are written (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete output-root before writing new markdown files.",
    )
    args = parser.parse_args()

    count = extract_workloads(
        source_root=args.source_root,
        output_root=args.output_root,
        clean=args.clean,
    )
    print(f"Converted {count} workload files into: {args.output_root}")


if __name__ == "__main__":
    main()
