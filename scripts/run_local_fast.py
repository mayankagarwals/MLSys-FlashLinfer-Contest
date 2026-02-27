import json
import statistics
from pathlib import Path

import pandas as pd
import torch
from flashinfer.testing import bench_gpu_time_with_cupti
from flashinfer_bench.bench.evaluators.utils import allocate_outputs
from flashinfer_bench.bench.utils import gen_inputs, load_safetensors
from flashinfer_bench.compile import BuilderRegistry
from flashinfer_bench.data import Definition, Solution, Workload
from huggingface_hub import hf_hub_download
from pack_solution import pack_solution


def main():
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

    REPO_NAME = "flashinfer-ai/mlsys26-contest"

    # load definition
    filename = f"definitions/{parent}/{def_name}.json"
    path = hf_hub_download(REPO_NAME, filename, repo_type="dataset")
    definition = Definition.model_validate_json(open(path).read())

    # load workloads
    filename = f"workloads/{parent}/{def_name}.jsonl"
    path = hf_hub_download(REPO_NAME, filename, repo_type="dataset")
    workloads = [
        Workload.model_validate(json.loads(line)["workload"]) for line in open(path)
    ]

    # root path for safetensors path
    trace_set_path = Path(
        hf_hub_download(REPO_NAME, "README.md", repo_type="dataset")
    ).parent

    # Build the solution
    registry = BuilderRegistry.get_instance()
    runnable = registry.build(definition, solution)

    rows = []

    for workload in workloads:
        print(workload.uuid, workload.axes)

        # Load safetensors if needed
        safe_tensors = None
        if any(inp.type == "safetensors" for inp in workload.inputs.values()):
            safe_tensors = load_safetensors(definition, workload, trace_set_path)

        # Generate inputs and allocate outputs
        device = "cuda"
        inputs = gen_inputs(definition, workload, device, safe_tensors)
        outputs = allocate_outputs(definition, inputs, device)

        if solution.spec.destination_passing_style:
            args = tuple(inputs) + tuple(outputs)
        else:
            args = tuple(inputs)

        # Warmup run to trigger JIT compilation
        with torch.no_grad():
            runnable(*args)
        torch.cuda.synchronize()

        timings = bench_gpu_time_with_cupti(
            runnable, dry_run_iters=3, repeat_iters=100, input_args=args
        )
        latency_us = statistics.median(timings) * 1e3

        rows.append(dict(uuid=workload.uuid, latency_us=latency_us))

    # Cleanup
    runnable.cleanup()

    # show results
    df = pd.DataFrame(rows)
    print(df)
    df.to_csv("benchmark.tsv", index=False, sep="\t")


if __name__ == "__main__":
    main()
