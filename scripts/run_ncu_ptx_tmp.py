# modified from https://github.com/flashinfer-ai/flashinfer-bench/blob/c1fd980f/flashinfer_bench/agents/_solution_runner.py

'''
Usage: 

/usr/local/cuda-13.1/bin/ncu \
  --set full \
  --import-source on \
  --nvtx \
  --nvtx-include flashinfer_bench_ncu_profile/ \
  -o "$RUN_ROOT/profile" \
  -f python \
  scripts/run_ncu_ptx_tmp.py


'''
# you can select a particular workload by passing --uuid

import argparse
import json
from pathlib import Path

import torch
from flashinfer_bench.bench.evaluators.utils import allocate_outputs
from flashinfer_bench.bench.utils import gen_inputs, load_safetensors
from flashinfer_bench.compile import BuilderRegistry
from flashinfer_bench.data import Definition, Solution, Workload
from huggingface_hub import hf_hub_download
from pack_solution import pack_solution

import os
import tvm_ffi


def _patch_tvm_ffi_build_for_ptx():
    keep_dir = os.environ.get("FIB_NCU_KEEP_DIR", "/tmp/mlsys_ncu_profile_run/cuda_keep")
    os.makedirs(keep_dir, exist_ok=True)

    cap = torch.cuda.get_device_capability()
    cc = f"{cap[0]}{cap[1]}"
    ptx_gencode = f"-gencode=arch=compute_{cc},code=compute_{cc}"

    orig_build = tvm_ffi.cpp.build

    def patched_build(*args, **kwargs):
        extra = list(kwargs.get("extra_cuda_cflags") or [])
        wanted = [
            "-lineinfo",
            "--keep",
            f"--keep-dir={keep_dir}",
            ptx_gencode,
        ]
        for flag in wanted:
            if flag not in extra:
                extra.append(flag)
        kwargs["extra_cuda_cflags"] = extra
        return orig_build(*args, **kwargs)

    tvm_ffi.cpp.build = patched_build
    print(f"patched tvm_ffi.cpp.build for PTX/lineinfo, keep_dir={keep_dir}, gencode={ptx_gencode}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uuid")
    args = parser.parse_args()
    _patch_tvm_ffi_build_for_ptx()

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

    # select the workload from the workload list
    if args.uuid is None:
        print("uuid is not provided. Selecting the first workload")
        workload = workloads[0]
    else:
        workload = next(w for w in workloads if w.uuid == args.uuid)

    # root path for safetensors path
    trace_set_path = Path(
        hf_hub_download(REPO_NAME, "README.md", repo_type="dataset")
    ).parent

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
