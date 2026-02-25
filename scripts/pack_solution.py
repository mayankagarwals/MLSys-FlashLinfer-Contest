"""
Pack solution source files into solution.json.

Reads configuration from config.toml and packs the appropriate source files
(Python, Triton, or CUDA) into a Solution JSON file for submission.
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from flashinfer_bench import BuildSpec, Solution, SourceFile
from flashinfer_bench.agents import pack_solution_from_files


def load_config() -> dict:
    """Load configuration from config.toml."""
    config_path = PROJECT_ROOT / "config.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "rb") as f:
        return tomllib.load(f)


def patch_cuda_solution(solution: Solution):
    # submit it as a Python submission so that we can control compile flags
    assert solution.spec.language == "cuda"
    solution.spec.language = "python"

    entrypoint_file, entrypoint_symbol = solution.spec.entry_point.split("::")
    main_content = rf"""
import tvm_ffi
from pathlib import Path

CURRENT_DIR = Path(__file__).parent

lib_path = tvm_ffi.cpp.build(
    name="my_package",
    cuda_files=[str(CURRENT_DIR / "{entrypoint_file}")],
    extra_cflags=["-O3"],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "-lineinfo",
    ]
)
mod = tvm_ffi.load_module(lib_path)
run = mod.{entrypoint_symbol}
"""

    # swap the entrypoint
    solution.sources.append(SourceFile(path="main.py",content=main_content))
    solution.spec.entry_point = "main.py::run"


def pack_solution(output_path: Path = None) -> Path:
    """Pack solution files into a Solution JSON."""
    config = load_config()

    solution_config = config["solution"]
    build_config = config["build"]

    language = build_config["language"]
    entry_point = build_config["entry_point"]

    # Determine source directory based on language
    if language == "python":
        source_dir = PROJECT_ROOT / "solution" / "python"
    elif language == "triton":
        source_dir = PROJECT_ROOT / "solution" / "triton"
    elif language == "cuda":
        source_dir = PROJECT_ROOT / "solution" / "cuda"
    else:
        raise ValueError(f"Unsupported language: {language}")

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    destination_passing_style = build_config.get("destination_passing_style", True)

    # Create build spec
    spec = BuildSpec(
        language=language,
        target_hardware=["cuda"],
        entry_point=entry_point,
        destination_passing_style=destination_passing_style,
    )

    # Pack the solution
    solution = pack_solution_from_files(
        path=str(source_dir),
        spec=spec,
        name=solution_config["name"],
        definition=solution_config["definition"],
        author=solution_config["author"],
    )

    # patch CUDA submission (modify in-place)
    if spec.language == "cuda":
        patch_cuda_solution(solution)

    # Write to output file
    if output_path is None:
        output_path = PROJECT_ROOT / "solution.json"

    output_path.write_text(solution.model_dump_json(indent=2))
    print(f"Solution packed: {output_path}")
    print(f"  Name: {solution.name}")
    print(f"  Definition: {solution.definition}")
    print(f"  Author: {solution.author}")
    print(f"  Language: {language}")
    print(f"  Destination Passing Style: {destination_passing_style}")

    return output_path


def main():
    """Entry point for pack_solution script."""
    import argparse

    parser = argparse.ArgumentParser(description="Pack solution files into solution.json")
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output path for solution.json (default: ./solution.json)"
    )
    args = parser.parse_args()

    try:
        pack_solution(args.output)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
