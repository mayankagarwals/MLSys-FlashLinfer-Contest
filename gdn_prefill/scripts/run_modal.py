"""
FlashInfer-Bench Modal Cloud Benchmark Runner.

Reads the solution source files locally (pure file I/O — deliberately *no*
``flashinfer_bench`` import on the host) and ships them to Modal, where the
solution is packed and benchmarked on an NVIDIA B200. Keeping the import off the
host is what lets ``modal run`` work on macOS without any Linux/GPU-only deps or
import shims — see MODAL_RUNBOOK.md.

Setup (one-time):
    pip install modal
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace <dataset>/definitions /definitions
    modal volume put flashinfer-trace <dataset>/workloads   /workloads
    modal volume put flashinfer-trace <dataset>/solutions   /solutions
    modal volume put flashinfer-trace <dataset>/blob        /blob

Run:
    modal run gdn_prefill/scripts/run_modal.py                       # all workloads
    modal run gdn_prefill/scripts/run_modal.py --workload-uuid <id>  # one workload
"""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ImportError:  # Python < 3.11
    import tomli as tomllib

import modal

PROJECT_ROOT = Path(__file__).parent.parent

app = modal.App("flashinfer-bench")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

# The contest's CUDA 13.2 CI image — the same toolchain (nvcc, CUTLASS, headers)
# the official grader uses (see docker/Dockerfile). Our kernels use CUDA 13 syntax
# (e.g. `__block_size__`, `__grid_constant__`), so a CUDA 12.x image can't compile
# them. flashinfer_bench builds each solution in a *spawned* worker process; that
# worker inherits the container env but the CI image doesn't expose CUDA to it, so
# we set CUDA_HOME (lets tvm-ffi find nvcc at $CUDA_HOME/bin/nvcc) and register the
# CUDA libs with ldconfig (lets the runtime `libcudart.so` preload resolve).
# Without these the worker fails with COMPILE_ERROR.
image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132", add_python=None)
    .run_commands(
        # Modal needs `python` on PATH; the CI image ships `python3`.
        "ln -sf $(command -v python3) /usr/local/bin/python",
        # Make libcudart.so resolvable for any process (incl. the spawn worker).
        "printf '/usr/local/cuda/lib64\\n/usr/local/cuda/targets/x86_64-linux/lib\\n'"
        " > /etc/ld.so.conf.d/cuda.conf && ldconfig",
    )
    .pip_install(
        "pandas",
        "huggingface_hub",
        "git+https://github.com/flashinfer-ai/flashinfer-bench.git",
    )
    .env({"CUDA_HOME": "/usr/local/cuda"})
)


def read_source_files() -> dict[str, str]:
    """Read config.toml + solution source files into a ``{relpath: content}`` dict.

    Pure file I/O; intentionally imports nothing from flashinfer_bench so this
    module imports cleanly on macOS when ``modal run`` loads it locally.
    """
    config_path = PROJECT_ROOT / "config.toml"
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    language = config["build"]["language"]
    # An explicit `source_dir` in [build] overrides the language->dir default,
    # letting parallel solutions (e.g. solution/cutedsl) coexist with the main one.
    rel_source_dir = config["build"].get("source_dir") or f"solution/{language}"
    source_dir = PROJECT_ROOT / rel_source_dir
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    files = {"config.toml": config_path.read_text()}
    for src in sorted(source_dir.rglob("*")):
        if src.is_file() and "__pycache__" not in src.parts:
            rel = str(src.relative_to(PROJECT_ROOT))
            files[rel] = src.read_text()
    return files


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={TRACE_SET_PATH: trace_volume})
def run_benchmark(config_toml: str, source_files: dict, workload_uuid: str = None) -> dict:
    """Pack the solution and benchmark it on a B200.

    If ``workload_uuid`` is given, only the workload whose UUID matches exactly
    is benchmarked; otherwise all workloads run.
    """
    from flashinfer_bench import Benchmark, BenchmarkConfig, BuildSpec, TraceSet
    from flashinfer_bench.agents import pack_solution_from_files

    config = tomllib.loads(config_toml)
    solution_config = config["solution"]
    build_config = config["build"]
    language = build_config["language"]
    rel_source_dir = build_config.get("source_dir") or f"solution/{language}"

    # Reconstruct the source tree on the worker, then pack remotely.
    workspace = Path("/workspace")
    for rel_path, content in source_files.items():
        if rel_path.startswith(f"{rel_source_dir}/"):
            out = workspace / rel_path
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(content)
    source_dir = workspace / rel_source_dir

    spec = BuildSpec(
        language=language,
        target_hardware=["cuda"],
        entry_point=build_config["entry_point"],
        destination_passing_style=build_config.get("destination_passing_style", True),
    )
    solution = pack_solution_from_files(
        path=str(source_dir),
        spec=spec,
        name=solution_config["name"],
        definition=solution_config["definition"],
        author=solution_config["author"],
    )
    print(f"Packed solution: {solution.name} ({solution.definition})")

    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    if solution.definition not in trace_set.definitions:
        available = ", ".join(sorted(trace_set.definitions))
        raise ValueError(f"Definition '{solution.definition}' not found. Available: {available}")

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])
    if not workloads:
        raise ValueError(f"No workloads found for '{solution.definition}'")

    if workload_uuid is not None:
        matches = [w for w in workloads if w.workload.uuid == workload_uuid]
        if not matches:
            available = ", ".join(w.workload.uuid for w in workloads)
            raise ValueError(
                f"Workload '{workload_uuid}' not found for definition "
                f"'{solution.definition}'. Available: {available}"
            )
        workloads = matches

    bench_config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)
    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    benchmark = Benchmark(bench_trace_set, bench_config)
    result_trace_set = benchmark.run_all(dump_traces=True)

    traces = result_trace_set.traces.get(definition.name, [])
    results = {definition.name: {}}

    for trace in traces:
        if trace.evaluation:
            entry = {
                "status": trace.evaluation.status.value,
                "solution": trace.solution,
                "log": trace.evaluation.log,
            }
            if trace.evaluation.performance:
                entry["latency_ms"] = trace.evaluation.performance.latency_ms
                entry["reference_latency_ms"] = trace.evaluation.performance.reference_latency_ms
                entry["speedup_factor"] = trace.evaluation.performance.speedup_factor
            if trace.evaluation.correctness:
                entry["max_abs_error"] = trace.evaluation.correctness.max_absolute_error
                entry["max_rel_error"] = trace.evaluation.correctness.max_relative_error
            results[definition.name][trace.workload.uuid] = entry

    return results


def print_results(results: dict):
    """Print benchmark results in a formatted way."""
    for def_name, traces in results.items():
        print(f"\n{def_name}:")
        for workload_uuid, result in traces.items():
            status = result.get("status")
            print(f"  Workload {workload_uuid[:8]}...: {status}", end="")

            if result.get("latency_ms") is not None:
                print(f" | {result['latency_ms']:.3f} ms", end="")

            if result.get("speedup_factor") is not None:
                print(f" | {result['speedup_factor']:.2f}x speedup", end="")

            if result.get("max_abs_error") is not None:
                abs_err = result["max_abs_error"]
                rel_err = result.get("max_rel_error", 0)
                print(f" | abs_err={abs_err:.2e}, rel_err={rel_err:.2e}", end="")

            print()

            # For any non-passing status, dump the captured compiler/runtime log.
            if status and status != "PASSED" and result.get("log"):
                print("    --- log ---")
                for log_line in result["log"].rstrip().splitlines():
                    print(f"    {log_line}")
                print("    --- end log ---")


@app.local_entrypoint()
def main(workload_uuid: str = None):
    """Read source files locally, then pack + benchmark on Modal.

    Pass ``--workload-uuid <uuid>`` to benchmark the single workload whose UUID
    matches exactly; omit it to run all workloads.
    """
    source_files = read_source_files()
    config_toml = source_files["config.toml"]

    if workload_uuid:
        print(f"Sending sources to Modal; benchmarking workload {workload_uuid} on B200...")
    else:
        print("Sending sources to Modal; benchmarking all workloads on B200...")
    results = run_benchmark.remote(config_toml, source_files, workload_uuid=workload_uuid)

    if not results:
        print("No results returned!")
        return

    print_results(results)
