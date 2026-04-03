"""
Fast single-process benchmark using flashinfer-bench evaluator.

Reads config.toml for the solution to benchmark, runs all workloads in a single
process (no subprocess isolation), and reports latency + correctness.

This variant rewrites each prefill workload's `cu_seqlens` so that
`num_seqs - 1` sequences have length 1 and the final sequence receives all
remaining tokens.

Usage:
  cd scripts/
  uv run python run_local_fast_uneven.py                          # download dataset from HuggingFace
  uv run python run_local_fast_uneven.py --local /path/to/dataset  # use local dataset
  uv run python run_local_fast_uneven.py --run_baseline gdn_prefill  # run FlashInfer baseline
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from flashinfer_bench.bench.evaluators import DefaultEvaluator
from flashinfer_bench.bench.evaluators.utils import allocate_outputs
from flashinfer_bench.bench.utils import BenchmarkConfig, gen_inputs, load_safetensors
from flashinfer_bench.compile import BuilderRegistry
from flashinfer_bench.data import Definition, Solution, Workload
from pack_solution import pack_solution


def build_uneven_cu_seqlens(total_seq_len: int, num_seqs: int, device: str) -> torch.Tensor:
    if num_seqs <= 0:
        raise ValueError(f"Expected num_seqs > 0, got {num_seqs}")
    if total_seq_len < num_seqs:
        raise ValueError(
            f"Expected total_seq_len >= num_seqs for uneven layout, got {total_seq_len=} {num_seqs=}"
        )

    lengths = torch.ones((num_seqs,), dtype=torch.int64, device=device)
    lengths[-1] = total_seq_len - (num_seqs - 1)

    cu_seqlens = torch.empty(num_seqs + 1, dtype=torch.int64, device=device)
    cu_seqlens[0] = 0
    cu_seqlens[1:] = lengths.cumsum(dim=0)
    return cu_seqlens


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

    if args.max_workloads is not None:
        workloads = workloads[: args.max_workloads]

    # Build the solution
    device = "cuda"
    registry = BuilderRegistry.get_instance()
    runnable_ref = registry.build_reference(definition)
    runnable = registry.build(definition, solution)

    input_names = list(definition.inputs.keys())
    cu_seqlens_idx = input_names.index("cu_seqlens")

    rows = []

    for workload in workloads:
        # Load safetensors if needed
        safe_tensors = None
        if any(inp.type == "safetensors" for inp in workload.inputs.values()):
            safe_tensors = load_safetensors(definition, workload, repo_path)

        inputs = gen_inputs(definition, workload, device, safe_tensors)

        total_seq_len = int(workload.axes["total_seq_len"])
        num_seqs = int(workload.axes["num_seqs"])
        uneven_cu_seqlens = build_uneven_cu_seqlens(total_seq_len, num_seqs, device)
        inputs[cu_seqlens_idx] = uneven_cu_seqlens

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

        long_seq_len = total_seq_len - (num_seqs - 1)
        sample = dict(
            uuid=workload.uuid,
            **workload.axes,
            short_seq_len=1,
            num_short_seqs=max(num_seqs - 1, 0),
            long_seq_len=long_seq_len,
            cu_seqlens_mode="uneven",
        )
        correct = False
        if evaluation.correctness is not None:
            sample.update(max_abs_error=evaluation.correctness.max_absolute_error)
            sample.update(max_rel_error=evaluation.correctness.max_relative_error)
        if evaluation.performance is not None:
            sample.update(latency_us=evaluation.performance.latency_ms * 1e3)
            correct = True
        rows.append(sample)

        print(
            workload.uuid,
            workload.axes,
            f"uneven_seq_len=[1 x {max(num_seqs - 1, 0)}, {long_seq_len}]",
            f"{correct=}",
        )

    # Cleanup
    runnable.cleanup()

    # show results
    df = pd.DataFrame(rows)
    print(df)
    df.to_csv("benchmark.tsv", index=False, sep="\t")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast single-process benchmark with uneven cu_seqlens")
    parser.add_argument(
        "--run_baseline",
        choices=["gdn_decode", "gdn_prefill"],
        help="Run FlashInfer baseline instead of config.toml solution",
    )
    parser.add_argument(
        "--local",
        type=str,
        default=None,
        help="Path to local dataset (skip HuggingFace download)",
    )
    parser.add_argument(
        "--max-workloads",
        type=int,
        default=None,
        help="Only run the first N workloads from the dataset",
    )
    args = parser.parse_args()

    main(args)
