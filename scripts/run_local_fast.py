"""
Fast single-process benchmark using flashinfer-bench evaluator.

Reads config.toml for the solution to benchmark, runs all workloads in a single
process (no subprocess isolation), and reports latency + correctness.

Usage:
  cd scripts/
  uv run python run_local_fast.py                          # download dataset from HuggingFace
  uv run python run_local_fast.py --local /path/to/dataset  # use local dataset
  uv run python run_local_fast.py --run_baseline gdn_prefill  # run FlashInfer baseline
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from flashinfer_bench.bench.evaluators import DefaultEvaluator
from flashinfer_bench.bench.evaluators.utils import allocate_outputs
from flashinfer_bench.bench.utils import BenchmarkConfig, gen_inputs, load_safetensors
from flashinfer_bench.compile import BuilderRegistry
from flashinfer_bench.data import Definition, Solution, Workload
from pack_solution import pack_solution


def main(args: argparse.Namespace):
    # Resolve dataset path: local flag > HuggingFace download
    if args.local:
        repo_path = Path(args.local)
        if not repo_path.exists():
            raise FileNotFoundError(f"Local dataset not found: {repo_path}")
    else:
        from huggingface_hub import snapshot_download
        REPO_NAME = "flashinfer-ai/mlsys26-contest"
        repo_path = Path(snapshot_download(REPO_NAME, repo_type="dataset"))

    if args.run_baseline:
        # load hard-coded baseline path
        solution_path = dict(
            gdn_decode=repo_path
            / "solutions/baseline/gdn/gdn_decode_qk4_v8_d128_k_last/flashinfer_wrapper_9b7f1e.json",
            gdn_prefill=repo_path
            / "solutions/baseline/gdn/gdn_prefill_qk4_v8_d128_k_last/flashinfer_wrapper_123ca6.json",
        )[args.run_baseline]
    else:
        # pack our solution
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

    # load definition
    filename = f"definitions/{parent}/{def_name}.json"
    definition = Definition.model_validate_json(open(repo_path / filename).read())

    # load workloads
    filename = f"workloads/{parent}/{def_name}.jsonl"
    workloads = [
        Workload.model_validate(json.loads(line)["workload"])
        for line in open(repo_path / filename)
    ]

    # Build the solution
    device = "cuda"
    registry = BuilderRegistry.get_instance()
    runnable_ref = registry.build_reference(definition)
    runnable = registry.build(definition, solution)

    rows = []

    for workload in workloads:
        # Load safetensors if needed
        safe_tensors = None
        if any(inp.type == "safetensors" for inp in workload.inputs.values()):
            safe_tensors = load_safetensors(definition, workload, repo_path)

        inputs = gen_inputs(definition, workload, device, safe_tensors)
        outputs_ref = allocate_outputs(definition, inputs, device)
        runnable_ref.call_destination_passing(*inputs, *outputs_ref)

        evaluation = DefaultEvaluator.evaluate(
            definition,
            runnable,
            inputs=[inputs],
            ref_outputs=[outputs_ref],
            ref_mean_latency_ms=1.0,  # arbitrary number
            cfg=BenchmarkConfig(),
            log_path="/tmp/flashinfer-bench",
            device=device,
        )

        sample = dict(uuid=workload.uuid, **workload.axes)
        correct = False
        if evaluation.correctness is not None:
            sample.update(max_abs_error=evaluation.correctness.max_absolute_error)
            sample.update(max_rel_error=evaluation.correctness.max_relative_error)
        if evaluation.performance is not None:
            sample.update(latency_us=evaluation.performance.latency_ms * 1e3)
            correct=True
        rows.append(sample)

        print(workload.uuid, workload.axes, f"{correct=}")

    # Cleanup
    runnable.cleanup()

    # show results
    df = pd.DataFrame(rows)
    print(df)
    df.to_csv("benchmark.tsv", index=False, sep="\t")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast single-process benchmark")
    parser.add_argument("--run_baseline", choices=["gdn_decode", "gdn_prefill"],
                        help="Run FlashInfer baseline instead of config.toml solution")
    parser.add_argument("--local", type=str, default=None,
                        help="Path to local dataset (skip HuggingFace download)")
    args = parser.parse_args()

    main(args)
