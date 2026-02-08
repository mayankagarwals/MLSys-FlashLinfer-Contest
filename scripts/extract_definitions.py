"""
Extract workload definition JSON files into human-readable Markdown files.

The output directory mirrors the source directory structure, converting:
  path/to/file.json -> definitions/path/to/file.md
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_ROOT = Path("/home/simon/flashinfer-competition/mlsys26-contest/definitions")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "definitions"


def escape_md(value: Any) -> str:
    """Escape basic markdown table delimiters and line breaks."""
    text = str(value)
    return text.replace("|", r"\|").replace("\n", "<br>")


def format_json_value(value: Any) -> str:
    """Format a JSON value for compact markdown display."""
    if value is None:
        return "null"
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=True)


def format_shape(shape: Any) -> str:
    if shape is None:
        return "scalar"
    return json.dumps(shape, ensure_ascii=True)


def render_axes(axes: dict[str, Any]) -> list[str]:
    if not axes:
        return []

    lines = [
        "## Axes",
        "",
        "| Axis | Type | Value | Description |",
        "| --- | --- | --- | --- |",
    ]
    for axis_name, axis_meta in axes.items():
        axis_type = format_json_value(axis_meta.get("type", "-"))
        axis_value = format_json_value(axis_meta["value"]) if "value" in axis_meta else "-"
        description = escape_md(axis_meta.get("description", ""))
        lines.append(
            f"| `{escape_md(axis_name)}` | `{escape_md(axis_type)}` | "
            f"`{escape_md(axis_value)}` | {description} |"
        )
    return lines + [""]


def render_constraints(constraints: list[Any]) -> list[str]:
    if not constraints:
        return []

    lines = ["## Constraints", ""]
    for item in constraints:
        lines.append(f"- `{escape_md(item)}`")
    return lines + [""]


def render_io_table(section_name: str, io_spec: dict[str, Any]) -> list[str]:
    if not io_spec:
        return []

    lines = [
        f"## {section_name}",
        "",
        "| Name | Shape | Dtype | Optional | Description |",
        "| --- | --- | --- | --- | --- |",
    ]

    for name, meta in io_spec.items():
        shape = escape_md(format_shape(meta.get("shape")))
        dtype = escape_md(format_json_value(meta.get("dtype", "-")))
        optional = "yes" if bool(meta.get("optional", False)) else "no"
        description = escape_md(meta.get("description", ""))
        lines.append(
            f"| `{escape_md(name)}` | `{shape}` | `{dtype}` | `{optional}` | {description} |"
        )
    return lines + [""]


def render_markdown(definition: dict[str, Any], source_rel_path: Path) -> str:
    name = definition.get("name", source_rel_path.stem)
    description = definition.get("description", "")
    op_type = definition.get("op_type", "")
    tags = definition.get("tags", [])

    lines = [
        f"# {escape_md(name)}",
        "",
        "## Summary",
        "",
        f"- Source JSON: `{source_rel_path.as_posix()}`",
        f"- Operation Type: `{escape_md(op_type)}`",
        f"- Description: {escape_md(description)}",
        "- Tags: "
        + (", ".join(f"`{escape_md(tag)}`" for tag in tags) if tags else "_none_"),
        "",
    ]

    lines.extend(render_axes(definition.get("axes", {})))
    lines.extend(render_constraints(definition.get("constraints", [])))
    lines.extend(render_io_table("Inputs", definition.get("inputs", {})))
    lines.extend(render_io_table("Outputs", definition.get("outputs", {})))

    reference = definition.get("reference")
    if reference:
        lines.extend(["## Reference", "", "```python", reference, "```", ""])

    known_keys = {
        "name",
        "description",
        "op_type",
        "tags",
        "axes",
        "constraints",
        "inputs",
        "outputs",
        "reference",
    }
    extras = {k: v for k, v in definition.items() if k not in known_keys}
    if extras:
        lines.extend(
            [
                "## Additional Fields",
                "",
                "```json",
                json.dumps(extras, indent=2, ensure_ascii=True),
                "```",
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def extract_definitions(source_root: Path, output_root: Path, clean: bool = False) -> int:
    if not source_root.exists():
        raise FileNotFoundError(f"Source directory not found: {source_root}")

    if clean and output_root.exists():
        shutil.rmtree(output_root)

    json_files = sorted(source_root.rglob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found under: {source_root}")

    converted = 0
    for json_path in json_files:
        rel_path = json_path.relative_to(source_root)
        md_path = (output_root / rel_path).with_suffix(".md")
        md_path.parent.mkdir(parents=True, exist_ok=True)

        data = json.loads(json_path.read_text(encoding="utf-8"))
        markdown = render_markdown(data, rel_path)
        md_path.write_text(markdown, encoding="utf-8")
        converted += 1

    return converted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert workload definition JSON files to markdown documents."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help=f"Directory containing workload JSON definitions (default: {DEFAULT_SOURCE_ROOT})",
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

    count = extract_definitions(
        source_root=args.source_root,
        output_root=args.output_root,
        clean=args.clean,
    )
    print(f"Converted {count} definition files into: {args.output_root}")


if __name__ == "__main__":
    main()
