"""Direct timing comparison: our CUDA kernel vs Triton v4 mix."""
import sys, json, time, torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solution" / "python"))

# Import both kernels
from cuda_parallel_v3 import run as cuda_run
from triton_v4 import run as triton_v4_run
from triton_v2b import run as triton_v2_run

# Load workloads
from flashinfer_bench.bench.utils import gen_inputs, load_safetensors
from flashinfer_bench.data import Definition, Workload

repo_path = Path('/home/yue/mlsys26-contest')
def_name = "gdn_prefill_qk4_v8_d128_k_last"
definition = Definition.model_validate_json(
    open(repo_path / f'definitions/gdn/{def_name}.json').read())
workloads = [Workload.model_validate(json.loads(line)['workload'])
             for line in open(repo_path / f'workloads/gdn/{def_name}.jsonl')]

def time_kernel(run_fn, inputs, n_warmup=5, n_iter=20):
    """Time a kernel with CUDA events."""
    for _ in range(n_warmup):
        run_fn(**inputs)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        run_fn(**inputs)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iter * 1000  # us

def triton_mix_run(**kwargs):
    T = kwargs['q'].shape[0]
    if T >= 4096:
        return triton_v4_run(**kwargs)
    elif T >= 256:
        return triton_v2_run(**kwargs)
    else:
        return triton_v4_run(**kwargs)  # fallback

cuda_total = 0
triton_total = 0

for i, wl in enumerate(workloads):
    safe_tensors = None
    if any(inp.type == "safetensors" for inp in wl.inputs.values()):
        safe_tensors = load_safetensors(definition, wl, repo_path)
    raw_inputs = gen_inputs(definition, wl, 'cuda', safe_tensors)
    inputs = {name: raw_inputs[j] for j, name in enumerate(definition.inputs.keys())}

    T = wl.axes.get('total_seq_len', '?')
    N = wl.axes.get('num_seqs', '?')

    cuda_us = time_kernel(cuda_run, inputs)
    triton_us = time_kernel(triton_mix_run, inputs)

    cuda_total += cuda_us
    triton_total += triton_us
    ratio = triton_us / cuda_us if cuda_us > 0 else 0
    if i < 5 or int(T) >= 4096:
        print(f"WL{i:3d} T={T:>5} N={N:>3}: CUDA={cuda_us:8.1f}us  Triton={triton_us:8.1f}us  ratio={ratio:.2f}x")

print(f"\nTotal CUDA:   {cuda_total:.0f} us")
print(f"Total Triton: {triton_total:.0f} us")
print(f"Speedup: {triton_total/cuda_total:.2f}x")
