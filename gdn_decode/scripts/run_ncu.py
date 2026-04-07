# modified from https://github.com/flashinfer-ai/flashinfer-bench/blob/c1fd980f/flashinfer_bench/agents/_solution_runner.py
#
# run with ncu
#  ncu --set full --import-source on --nvtx --nvtx-include flashinfer_bench_ncu_profile/ -o profile -f python scripts/run_ncu.py
#
# you can select a particular workload by passing --uuid

import argparse
import json
from pathlib import Path

import torch
from flashinfer_bench.bench.evaluators.utils import allocate_outputs
from flashinfer_bench.bench.utils import gen_inputs, load_safetensors
from flashinfer_bench.compile import BuilderRegistry
from flashinfer_bench.data import Definition, Solution, Workload
from pack_solution import pack_solution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local",
        type=str,
        default=None,
        help="Path to local dataset (skip HuggingFace download)",
    )
    parser.add_argument("--uuid")
    args = parser.parse_args()

    if args.local:
        repo_path = Path(args.local)
        if not repo_path.exists():
            raise FileNotFoundError(f"Local dataset not found: {repo_path}")
    else:
        from huggingface_hub import snapshot_download

        REPO_NAME = "flashinfer-ai/mlsys26-contest"
        repo_path = Path(snapshot_download(REPO_NAME, repo_type="dataset"))

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

    # load definition
    filename = f"definitions/{parent}/{def_name}.json"
    definition = Definition.model_validate_json(open(repo_path / filename).read())

    # load workloads
    filename = f"workloads/{parent}/{def_name}.jsonl"
    workloads = [
        Workload.model_validate(json.loads(line)["workload"])
        for line in open(repo_path / filename)
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
        safe_tensors = load_safetensors(definition, workload, repo_path)

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
