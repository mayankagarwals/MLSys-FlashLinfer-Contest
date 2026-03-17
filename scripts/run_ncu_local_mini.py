# modified from https://github.com/flashinfer-ai/flashinfer-bench/blob/c1fd980f/flashinfer_bench/agents/_solution_runner.py
#
# run with ncu
#  ncu --set full --import-source on --nvtx --nvtx-include flashinfer_bench_ncu_profile] -o profile -f uv run python scripts/run_ncu_local_mini.py
#
# you can select a particular workload by passing --uuid

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from flashinfer_bench.bench.evaluators.utils import allocate_outputs
from flashinfer_bench.bench.utils import gen_inputs, load_safetensors
from flashinfer_bench.compile import BuilderRegistry
from flashinfer_bench.data import Definition, Solution, Workload
from pack_solution import pack_solution

DEFAULT_FULL_DATASET_PATH = "/home/simon/flashinfer-competition/mlsys26-contest"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def ensure_local_mini_dataset(def_name: str, parent: str) -> Path:
    mini_path = PROJECT_ROOT / "mini_datasets" / f"{def_name}_single"
    def_path = mini_path / "definitions" / parent / f"{def_name}.json"
    workload_path = mini_path / "workloads" / parent / f"{def_name}.jsonl"
    blob_path = mini_path / "blob"

    if def_path.exists() and workload_path.exists() and blob_path.exists():
        return mini_path

    full_path = Path(
        os.environ.get("FIB_FULL_DATASET_PATH", DEFAULT_FULL_DATASET_PATH)
    ).expanduser().resolve()
    if not full_path.exists():
        raise FileNotFoundError(f"Full dataset path not found: {full_path}")

    src_def_path = full_path / "definitions" / parent / f"{def_name}.json"
    src_workload_path = full_path / "workloads" / parent / f"{def_name}.jsonl"
    src_blob_path = full_path / "blob"

    if not src_def_path.exists():
        raise FileNotFoundError(f"Definition file not found: {src_def_path}")
    if not src_workload_path.exists():
        raise FileNotFoundError(f"Workload file not found: {src_workload_path}")
    if not src_blob_path.exists():
        raise FileNotFoundError(f"Blob directory not found: {src_blob_path}")

    def_path.parent.mkdir(parents=True, exist_ok=True)
    workload_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_def_path, def_path)

    first_line = None
    for line in src_workload_path.read_text().splitlines():
        if line.strip():
            first_line = line
            break
    if first_line is None:
        raise ValueError(f"No workload rows found in {src_workload_path}")
    workload_path.write_text(first_line + "\n")

    if blob_path.exists() or blob_path.is_symlink():
        if blob_path.is_symlink():
            blob_path.unlink()
        else:
            raise FileExistsError(f"Expected blob symlink path is a directory: {blob_path}")
    blob_path.symlink_to(src_blob_path, target_is_directory=True)

    print(f"Created mini dataset: {mini_path}")
    return mini_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uuid")
    args = parser.parse_args()

    # pack and load solution
    solution_path = pack_solution()
    solution = Solution.model_validate_json(solution_path.read_text())

    # get definition from solution
    def_name = solution.definition

    if def_name.startswith("dsa_"):
        parent = "dsa_paged"
    elif def_name.startswith("gdn_"):
        parent = "gdn"
    elif def_name.startswith("moe_"):
        parent = "moe"
    else:
        raise ValueError("Unsupported definition")

    trace_set_path = ensure_local_mini_dataset(def_name, parent)
    print(f"Using local mini dataset: {trace_set_path}")

    # load definition
    def_path = trace_set_path / "definitions" / parent / f"{def_name}.json"
    definition = Definition.model_validate_json(def_path.read_text())

    # load workloads
    workload_path = trace_set_path / "workloads" / parent / f"{def_name}.jsonl"
    workloads = [
        Workload.model_validate(json.loads(line)["workload"])
        for line in workload_path.read_text().splitlines()
        if line.strip()
    ]

    # select the workload from the workload list
    if args.uuid is None:
        print("uuid is not provided. Selecting the first workload")
        workload = workloads[0]
    else:
        workload = next(w for w in workloads if w.uuid == args.uuid)

    # Build the solution
    registry = BuilderRegistry.get_instance()
    runnable = registry.build(definition, solution)

    print(workload.uuid, workload.axes)

    # Load safetensors if needed
    safe_tensors = None
    if any(inp.type == "safetensors" for inp in workload.inputs.values()):
        safe_tensors = load_safetensors(definition, workload, trace_set_path)

    # Generate inputs
    device = "cuda"
    inputs = gen_inputs(definition, workload, device, safe_tensors)

    # Allocate output tensors
    outputs = allocate_outputs(definition, inputs, device)

    # Warmup run to trigger JIT compilation
    with torch.no_grad():
        runnable.call_destination_passing(*inputs, *outputs)
    torch.cuda.synchronize()

    # Actual run for profiling (marked with NVTX for NCU filtering)
    with torch.cuda.nvtx.range("flashinfer_bench_ncu_profile"):
        with torch.no_grad():
            runnable.call_destination_passing(*inputs, *outputs)
        torch.cuda.synchronize()

    # Cleanup
    runnable.cleanup()


if __name__ == "__main__":
    main()
