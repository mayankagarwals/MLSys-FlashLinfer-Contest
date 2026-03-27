import json
from pathlib import Path

import pandas as pd
from flashinfer_bench.bench.evaluators import DefaultEvaluator
from flashinfer_bench.bench.evaluators.utils import allocate_outputs
from flashinfer_bench.bench.utils import BenchmarkConfig, gen_inputs, load_safetensors
from flashinfer_bench.compile import BuilderRegistry
from flashinfer_bench.data import Definition, Solution, Workload
from huggingface_hub import snapshot_download
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
    repo_path = Path(snapshot_download(REPO_NAME, repo_type="dataset"))

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
        print(workload.uuid, workload.axes)

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
            log_path="/tmp/flashinfer",
            device=device,
        )
        rows.append(
            dict(
                uuid=workload.uuid,
                **workload.axes,
                latency_us=evaluation.performance.latency_ms * 1e3,
            )
        )

    # Cleanup
    runnable.cleanup()

    # show results
    df = pd.DataFrame(rows)
    print(df)
    df.to_csv("benchmark.tsv", index=False, sep="\t")


if __name__ == "__main__":
    main()
