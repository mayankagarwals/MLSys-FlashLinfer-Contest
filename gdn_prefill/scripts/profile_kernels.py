"""Profile individual kernel times using CUDA events."""
import sys, json, torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solution" / "python"))

from cuda_parallel_v3 import run as cuda_run

# Load one large workload
from flashinfer_bench.bench.utils import gen_inputs, load_safetensors
from flashinfer_bench.data import Definition, Workload

repo_path = Path('/home/yue/mlsys26-contest')
def_name = "gdn_prefill_qk4_v8_d128_k_last"
definition = Definition.model_validate_json(
    open(repo_path / f'definitions/gdn/{def_name}.json').read())
workloads = [Workload.model_validate(json.loads(line)['workload'])
             for line in open(repo_path / f'workloads/gdn/{def_name}.jsonl')]

# Find T=8192, N=32 workload
for wl in workloads:
    T = int(wl.axes.get('total_seq_len', 0))
    N = int(wl.axes.get('num_seqs', 0))
    if T == 8192 and N == 32:
        break

safe_tensors = None
if any(inp.type == "safetensors" for inp in wl.inputs.values()):
    safe_tensors = load_safetensors(definition, wl, repo_path)
raw_inputs = gen_inputs(definition, wl, 'cuda', safe_tensors)
inputs = {name: raw_inputs[j] for j, name in enumerate(definition.inputs.keys())}

print(f"Profiling T={T}, N={N}")

# Warmup
for _ in range(5):
    cuda_run(**inputs)
torch.cuda.synchronize()

# Use nsight-like approach: time total kernel
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
n_iter = 50

start.record()
for _ in range(n_iter):
    cuda_run(**inputs)
end.record()
torch.cuda.synchronize()
total_us = start.elapsed_time(end) / n_iter * 1000
print(f"Total per-call: {total_us:.1f} us")

# Now use torch profiler for per-kernel breakdown
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=False,
) as prof:
    for _ in range(3):
        cuda_run(**inputs)
    torch.cuda.synchronize()

print("\nPer-kernel breakdown (3 iterations):")
events = prof.key_averages()
for e in sorted(events, key=lambda x: -x.device_time_total):
    if e.device_time_total > 100:  # > 100us total
        print(f"  {e.key[:60]:60s} {e.device_time_total/3:8.1f} us/iter  (count={e.count//3})")
