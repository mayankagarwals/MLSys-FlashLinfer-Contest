"""
Extract solution JSON files into human-readable Markdown files.

The output directory mirrors the source directory structure, converting:
  path/to/file.json -> solutions/path/to/file.md
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from dataset_utils import resolve_dataset_root


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "reference_solutions"


def escape_md(value: Any) -> str:
    text = str(value)
    return text.replace("|", r"\|").replace("\n", "<br>")


def render_list(items: list[Any]) -> str:
    if not items:
        return "_none_"
    return ", ".join(f"`{escape_md(item)}`" for item in items)


def render_sources(sources: list[dict[str, Any]]) -> list[str]:
    if not sources:
        return []

    lines = ["## Sources", ""]
    for idx, source in enumerate(sources, start=1):
        path = source.get("path", f"source-{idx}")
        content = source.get("content", "")
        suffix = Path(path).suffix.lstrip(".") or "text"
        lines.extend(
            [
                f"### `{escape_md(path)}`",
                "",
                f"- Index: `{idx}`",
                "",
                f"```{suffix}",
                content,
                "```",
                "",
            ]
        )
    return lines


def render_markdown(solution: dict[str, Any], source_rel_path: Path) -> str:
    name = solution.get("name", source_rel_path.stem)
    definition = solution.get("definition", "")
    author = solution.get("author", "")
    description = solution.get("description", "")
    spec = solution.get("spec", {})

    lines = [
        f"# {escape_md(name)}",
        "",
        "## Summary",
        "",
        f"- Source JSON: `{source_rel_path.as_posix()}`",
        f"- Definition: `{escape_md(definition)}`",
        f"- Author: `{escape_md(author)}`",
        f"- Description: {escape_md(description) if description else '_none_'}",
        "",
        "## Build Spec",
        "",
        f"- Language: `{escape_md(spec.get('language', '-'))}`",
        f"- Entry Point: `{escape_md(spec.get('entry_point', '-'))}`",
        "- Target Hardware: " + render_list(spec.get("target_hardware", [])),
        "- Dependencies: " + render_list(spec.get("dependencies", [])),
        "- Destination Passing Style: "
        + f"`{escape_md(spec.get('destination_passing_style', '-'))}`",
        "",
    ]

    lines.extend(render_sources(solution.get("sources", [])))

    known_keys = {
        "name",
        "definition",
        "author",
        "description",
        "spec",
        "sources",
    }
    extras = {k: v for k, v in solution.items() if k not in known_keys}
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


def extract_solutions(source_root: Path, output_root: Path, clean: bool = False) -> int:
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
        description="Convert solution JSON files to markdown documents."
    )
    parser.add_argument(
        "--local",
        type=str,
        default=None,
        help="Path to a local dataset root. If omitted, use the cached Hugging Face dataset.",
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

    dataset_root = resolve_dataset_root(args.local)
    source_root = dataset_root / "solutions"
    count = extract_solutions(
        source_root=source_root,
        output_root=args.output_root,
        clean=args.clean,
    )
    print(f"Dataset root: {dataset_root}")
    print(f"Converted {count} solution files into: {args.output_root}")


if __name__ == "__main__":
    main()
