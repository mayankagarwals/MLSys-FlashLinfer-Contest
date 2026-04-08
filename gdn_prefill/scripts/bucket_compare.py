"""Full bucket comparison: CUDA vs Triton across all 100 workloads."""
import sys, json, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "solution" / "python"))
from cuda_parallel_v3 import run as cuda_run
from triton_v4 import run as triton_v4_run
from triton_v2b import run as triton_v2_run
from flashinfer_bench.bench.utils import gen_inputs, load_safetensors
from flashinfer_bench.data import Definition, Workload

repo_path = Path('/home/yue/mlsys26-contest')
def_name = "gdn_prefill_qk4_v8_d128_k_last"
definition = Definition.model_validate_json(open(repo_path / "definitions/gdn" / (def_name+".json")).read())
workloads = [Workload.model_validate(json.loads(line)["workload"]) for line in open(repo_path / "workloads/gdn" / (def_name+".jsonl"))]

def time_fn(fn, inputs, nw=3, ni=10):
    for _ in range(nw): fn(**inputs)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(ni): fn(**inputs)
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/ni*1000

def triton_mix(**kw):
    T = kw["q"].shape[0]
    return triton_v4_run(**kw) if T >= 4096 else (triton_v2_run(**kw) if T >= 256 else triton_v4_run(**kw))

buckets = {}
for i, wl in enumerate(workloads):
    st = None
    if any(inp.type=="safetensors" for inp in wl.inputs.values()): st = load_safetensors(definition, wl, repo_path)
    raw = gen_inputs(definition, wl, "cuda", st)
    inp = {name: raw[j] for j, name in enumerate(definition.inputs.keys())}
    T = int(wl.axes.get("total_seq_len", 0))
    cu = time_fn(cuda_run, inp)
    tr = time_fn(triton_mix, inp)
    if T < 256: k = "T<256"
    elif T < 4096: k = "256-4095"
    else: k = "T>=4096"
    if k not in buckets: buckets[k] = [0,0,0]
    buckets[k][0] += cu; buckets[k][1] += tr; buckets[k][2] += 1

print("%-12s %5s %8s %8s %6s %8s" % ("Bucket", "N", "CUDA", "Triton", "Ratio", "Gap"))
tc=tt=0
for k in ["T<256", "256-4095", "T>=4096"]:
    if k in buckets:
        c,t,n = buckets[k]
        print("%-12s %5d %7.0fus %7.0fus %5.2fx %7.0fus" % (k,n,c,t,c/t if t else 0,c-t))
        tc+=c; tt+=t
print("%-12s %5d %7.0fus %7.0fus %5.2fx %7.0fus" % ("TOTAL",100,tc,tt,tc/tt,tc-tt))
