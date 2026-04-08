"""Profile Triton v4 per-kernel times."""
import sys, json, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "solution" / "python"))
from triton_v4 import run as triton_run
from flashinfer_bench.bench.utils import gen_inputs, load_safetensors
from flashinfer_bench.data import Definition, Workload

repo_path = Path('/home/yue/mlsys26-contest')
def_name = "gdn_prefill_qk4_v8_d128_k_last"
definition = Definition.model_validate_json(
    open(repo_path / f'definitions/gdn/{def_name}.json').read())
workloads = [Workload.model_validate(json.loads(line)['workload'])
             for line in open(repo_path / f'workloads/gdn/{def_name}.jsonl')]

for wl in workloads:
    if int(wl.axes.get('total_seq_len', 0)) == 8192 and int(wl.axes.get('num_seqs', 0)) == 32:
        break

safe_tensors = None
if any(inp.type == "safetensors" for inp in wl.inputs.values()):
    safe_tensors = load_safetensors(definition, wl, repo_path)
raw_inputs = gen_inputs(definition, wl, 'cuda', safe_tensors)
inputs = {name: raw_inputs[j] for j, name in enumerate(definition.inputs.keys())}

print(f"Profiling Triton T=8192, N=32")
for _ in range(5):
    triton_run(**inputs)
torch.cuda.synchronize()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(20):
    triton_run(**inputs)
end.record()
torch.cuda.synchronize()
print(f"Total: {start.elapsed_time(end)/20*1000:.1f} us")

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    for _ in range(3):
        triton_run(**inputs)
    torch.cuda.synchronize()

print("\nTriton per-kernel (3 iters):")
events = prof.key_averages()
for e in sorted(events, key=lambda x: -x.device_time_total):
    if e.device_time_total > 50:
        print(f"  {e.key[:70]:70s} {e.device_time_total/3:8.1f} us/iter  (n={e.count//3})")
