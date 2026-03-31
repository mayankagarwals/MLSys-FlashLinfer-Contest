import argparse
import json
import os
import statistics
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import torch
from flashinfer.testing import bench_gpu_time_with_cupti
from flashinfer_bench.bench.evaluators.utils import allocate_outputs
from flashinfer_bench.bench.utils import gen_inputs, load_safetensors
from flashinfer_bench.compile import BuilderRegistry
from flashinfer_bench.data import Definition, Solution, Workload
from pack_solution import pack_solution

DEFAULT_FULL_DATASET_PATH = "/home/simon/flashinfer-competition/mlsys26-contest"


def infer_parent(def_name: str) -> str:
    if def_name.startswith("dsa_"):
        return "dsa_paged"
    if def_name.startswith("gdn_"):
        return "gdn"
    if def_name.startswith("moe_"):
        return "moe"
    raise ValueError(f"Unsupported definition: {def_name}")


def resolve_dataset_path(dataset_path_arg: Optional[str]) -> Path:
    path = (
        dataset_path_arg
        or os.environ.get("FIB_FULL_DATASET_PATH")
        or DEFAULT_FULL_DATASET_PATH
    )
    dataset_path = Path(path).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    return dataset_path


def load_definition_and_workloads(
    dataset_path: Path, def_name: str
) -> Tuple[Definition, List[Workload]]:
    parent = infer_parent(def_name)
    def_path = dataset_path / "definitions" / parent / f"{def_name}.json"
    workload_path = dataset_path / "workloads" / parent / f"{def_name}.jsonl"

    if not def_path.exists():
        raise FileNotFoundError(f"Definition file not found: {def_path}")
    if not workload_path.exists():
        raise FileNotFoundError(f"Workload file not found: {workload_path}")

    definition = Definition.model_validate_json(def_path.read_text())
    workloads = [
        Workload.model_validate(json.loads(line)["workload"])
        for line in workload_path.read_text().splitlines()
        if line.strip()
    ]
    return definition, workloads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--max-workloads", type=int, default=None)
    parser.add_argument("--uuid", type=str, default=None)
    parser.add_argument("--dry-run-iters", type=int, default=10)
    parser.add_argument("--repeat-iters", type=int, default=1000)
    parser.add_argument("--output-tsv", type=str, default="benchmark.tsv")
    return parser.parse_args()


def main():
    cli_args = parse_args()

    # pack and load solution
    solution_path = pack_solution()
    solution = Solution.model_validate_json(solution_path.read_text())

    # get definition from solution
    def_name = solution.definition
    dataset_path = resolve_dataset_path(cli_args.dataset_path)
    definition, workloads = load_definition_and_workloads(dataset_path, def_name)

    if cli_args.uuid is not None:
        workloads = [w for w in workloads if w.uuid == cli_args.uuid]
        if not workloads:
            raise ValueError(f"No workload found for uuid={cli_args.uuid}")

    if cli_args.max_workloads is not None:
        workloads = workloads[: cli_args.max_workloads]

    print(f"Using dataset path: {dataset_path}")
    print(f"Definition: {def_name}")
    print(f"Workloads: {len(workloads)}")

    # Build the solution
    registry = BuilderRegistry.get_instance()
    runnable = registry.build(definition, solution)

    rows = []

    for workload in workloads:
        print(workload.uuid, workload.axes)

        safe_tensors = None
        if any(inp.type == "safetensors" for inp in workload.inputs.values()):
            safe_tensors = load_safetensors(definition, workload, dataset_path)

        device = "cuda"
        inputs = gen_inputs(definition, workload, device, safe_tensors)
        outputs = allocate_outputs(definition, inputs, device)

        if solution.spec.destination_passing_style:
            call_args = tuple(inputs) + tuple(outputs)
        else:
            call_args = tuple(inputs)

        # Warmup run to trigger JIT compilation
        with torch.no_grad():
            runnable(*call_args)
        torch.cuda.synchronize()

        timings = bench_gpu_time_with_cupti(
            runnable,
            dry_run_iters=cli_args.dry_run_iters,
            repeat_iters=cli_args.repeat_iters,
            input_args=call_args,
        )
        latency_us = statistics.median(timings) * 1e3
        rows.append(dict(uuid=workload.uuid, latency_us=latency_us))

    runnable.cleanup()

    # Keep TSV schema aligned with scripts/run_local_fast.py.
    df = pd.DataFrame(rows, columns=["uuid", "latency_us"])
    print(df)

    output_path = Path(cli_args.output_tsv)
    if not output_path.is_absolute() and output_path.parent == Path("."):
        output_path = Path("results") / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, sep="\t")
    print(f"Wrote TSV: {output_path}")


if __name__ == "__main__":
    main()
