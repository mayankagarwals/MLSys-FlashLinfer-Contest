import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from flashinfer.testing import bench_gpu_time_with_cupti
from flashinfer_bench.bench.evaluators.utils import allocate_outputs
from flashinfer_bench.bench.utils import gen_inputs, load_safetensors
from flashinfer_bench.compile import BuilderRegistry
from flashinfer_bench.data import Definition, Solution, Workload

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pack_solution import pack_solution
from solution.python import gdn_prefill_reference

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


def clone_arg(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    return value


def tensor_summary(name: str, actual: torch.Tensor, expected: torch.Tensor) -> str:
    actual_f = actual.float()
    expected_f = expected.float()
    diff = (actual_f - expected_f).abs()
    if diff.numel() == 0:
        return f"{name}: shape={tuple(actual.shape)} dtype={actual.dtype} empty tensor"
    diff_flat = diff.reshape(-1)
    actual_flat = actual_f.reshape(-1)
    expected_flat = expected_f.reshape(-1)
    flat_idx = int(diff_flat.argmax().item())
    max_abs = diff_flat[flat_idx]
    max_idx = tuple(
        int(i) for i in torch.unravel_index(torch.tensor(flat_idx), diff.shape)
    )
    expected_at_max = expected_flat[flat_idx].item()
    actual_at_max = actual_flat[flat_idx].item()
    rel = diff / expected_f.abs().clamp_min(1e-12)
    max_rel = rel.max().item()
    return (
        f"{name}: shape={tuple(actual.shape)} dtype={actual.dtype} "
        f"max_abs={max_abs.item():.6e} max_rel={max_rel:.6e} "
        f"idx={max_idx} actual={actual_at_max:.6e} expected={expected_at_max:.6e}"
    )


def bench_latency_us(fn, call_args, dry_run_iters: int, repeat_iters: int) -> float:
    timings = bench_gpu_time_with_cupti(
        fn,
        dry_run_iters=dry_run_iters,
        repeat_iters=repeat_iters,
        input_args=call_args,
    )
    return statistics.median(timings) * 1e3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--uuid", type=str, required=True)
    parser.add_argument("--dry-run-iters", type=int, default=10)
    parser.add_argument("--repeat-iters", type=int, default=100)
    args = parser.parse_args()

    solution_path = pack_solution()
    solution = Solution.model_validate_json(solution_path.read_text())
    if solution.definition != "gdn_prefill_qk4_v8_d128_k_last":
        raise ValueError(
            "This debug script currently supports only gdn_prefill_qk4_v8_d128_k_last"
        )

    dataset_path = resolve_dataset_path(args.dataset_path)
    definition, workloads = load_definition_and_workloads(dataset_path, solution.definition)
    workload = next((w for w in workloads if w.uuid == args.uuid), None)
    if workload is None:
        raise ValueError(f"No workload found for uuid={args.uuid}")

    registry = BuilderRegistry.get_instance()
    runnable = registry.build(definition, solution)

    safe_tensors = None
    if any(inp.type == "safetensors" for inp in workload.inputs.values()):
        safe_tensors = load_safetensors(definition, workload, dataset_path)

    inputs = gen_inputs(definition, workload, "cuda", safe_tensors)
    outputs = allocate_outputs(definition, inputs, "cuda")
    call_args = tuple(inputs) + tuple(outputs)

    ref_inputs = [clone_arg(x) for x in inputs]
    with torch.no_grad():
        expected_output, expected_new_state = gdn_prefill_reference.run(*ref_inputs)
        runnable(*call_args)
    torch.cuda.synchronize()

    actual_by_name = dict(zip(definition.outputs.keys(), outputs))
    expected_by_name = {
        "output": expected_output,
        "new_state": expected_new_state,
    }

    print(f"Dataset path: {dataset_path}")
    print(f"Definition: {solution.definition}")
    print(f"Workload: {workload.uuid}")
    print("Reference comparison:")
    for name in definition.outputs.keys():
        actual = actual_by_name[name]
        expected = expected_by_name[name]
        print(f"  {tensor_summary(name, actual, expected)}")

    bench_outputs = allocate_outputs(definition, inputs, "cuda")
    bench_call_args = tuple(inputs) + tuple(bench_outputs)
    ref_bench_args = tuple(clone_arg(x) for x in inputs)

    with torch.no_grad():
        runnable(*bench_call_args)
        gdn_prefill_reference.run(*ref_bench_args)
    torch.cuda.synchronize()

    solution_latency_us = bench_latency_us(
        runnable, bench_call_args, args.dry_run_iters, args.repeat_iters
    )
    reference_latency_us = bench_latency_us(
        gdn_prefill_reference.run,
        ref_bench_args,
        args.dry_run_iters,
        args.repeat_iters,
    )
    speedup = reference_latency_us / solution_latency_us

    print("Latency comparison:")
    print(f"  solution_latency_us={solution_latency_us:.3f}")
    print(f"  reference_latency_us={reference_latency_us:.3f}")
    print(f"  speedup_vs_reference={speedup:.3f}x")

    runnable.cleanup()


if __name__ == "__main__":
    main()
