#!/usr/bin/env python3
"""
find_thresholds.py — find optimal dispatch thresholds for gdn_prefill_mix.py.

Method: run the full benchmark 3 times with each kernel forced, then compare
per-workload latencies to find the T values where each kernel wins.

Usage (from gdn_prefill/scripts/):
    python find_thresholds.py --local /path/to/dataset
"""
import argparse, shutil, subprocess, sys, csv
from pathlib import Path

SCRIPTS = Path(__file__).parent
MIX     = SCRIPTS.parents[0] / "solution/python/gdn_prefill_mix.py"
PYTHON  = shutil.which("python") or sys.executable
# prefer the venv python if present
VENV_PY = Path(__file__).parents[2] / ".venv/bin/python"
if VENV_PY.exists():
    PYTHON = str(VENV_PY)

KERNEL_OVERRIDES = {
    "recurrent": """\
import ctypes; ctypes.CDLL("libcudart.so", mode=ctypes.RTLD_GLOBAL)
import torch
from torch import Tensor
from .cuda_recurrent_v1 import run as _recurrent

def run(q: Tensor, k: Tensor, v: Tensor, state: Tensor,
        A_log: Tensor, a: Tensor, dt_bias: Tensor, b: Tensor,
        cu_seqlens: Tensor, scale: float):
    o = torch.empty_like(v)
    ns = torch.empty_like(state)
    _recurrent(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, o, ns)
    return o, ns
""",
    "v3": """\
import ctypes; ctypes.CDLL("libcudart.so", mode=ctypes.RTLD_GLOBAL)
from torch import Tensor
from .cuda_parallel_v3 import run as _v3

def run(q: Tensor, k: Tensor, v: Tensor, state: Tensor,
        A_log: Tensor, a: Tensor, dt_bias: Tensor, b: Tensor,
        cu_seqlens: Tensor, scale: float):
    return _v3(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)
""",
    "chunk": """\
import ctypes; ctypes.CDLL("libcudart.so", mode=ctypes.RTLD_GLOBAL)
from torch import Tensor
from .chunk_v6b import run as _chunk

def run(q: Tensor, k: Tensor, v: Tensor, state: Tensor,
        A_log: Tensor, a: Tensor, dt_bias: Tensor, b: Tensor,
        cu_seqlens: Tensor, scale: float):
    return _chunk(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)
""",
}

def run_benchmark(local_path: str, tsv_out: Path):
    """Run run_local_fast.py and save benchmark.tsv to tsv_out."""
    result = subprocess.run(
        [PYTHON, "run_local_fast.py", "--local", local_path],
        cwd=SCRIPTS, capture_output=True, text=True
    )
    if result.returncode != 0:
        print(result.stderr[-2000:])
        raise RuntimeError("Benchmark failed")
    shutil.copy(SCRIPTS / "benchmark.tsv", tsv_out)

def load_tsv(path: Path) -> dict:
    """Return {uuid: {T, N, latency_us}} from benchmark.tsv."""
    rows = {}
    for r in csv.DictReader(path.read_text().splitlines(), delimiter="\t"):
        rows[r["uuid"]] = {
            "T": int(r["total_seq_len"]),
            "N": int(r["num_seqs"]),
            "lat": float(r["latency_us"]) if r["latency_us"] else float("inf"),
        }
    return rows

def first_crossover(data_a, data_b, name_a, name_b):
    """Return smallest T where b beats a, plus a per-T table string."""
    # group by T, take the minimum latency across workloads with same T
    from collections import defaultdict
    a_by_T, b_by_T = defaultdict(list), defaultdict(list)
    for r in data_a.values():
        a_by_T[r["T"]].append(r["lat"])
    for r in data_b.values():
        b_by_T[r["T"]].append(r["lat"])

    lines = [f"{'T':>6} | {name_a:>12} | {name_b:>12} | winner"]
    lines.append("-" * 45)
    crossover = None
    for T in sorted(set(a_by_T) & set(b_by_T)):
        la = min(a_by_T[T])
        lb = min(b_by_T[T])
        w = name_b if lb < la else name_a
        lines.append(f"T={T:4d} | {la:12.1f} | {lb:12.1f} | {w}")
        if w == name_b and crossover is None:
            crossover = T
    return crossover, "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", required=True)
    args = parser.parse_args()

    original = MIX.read_text()
    results = {}

    try:
        for kernel, override in KERNEL_OVERRIDES.items():
            print(f"\n{'='*50}\nRunning benchmark: always {kernel} ...\n{'='*50}")
            MIX.write_text(override)
            tsv = SCRIPTS / f"benchmark_{kernel}.tsv"
            run_benchmark(args.local, tsv)
            results[kernel] = load_tsv(tsv)
            print(f"  → saved {tsv.name}")
    finally:
        MIX.write_text(original)
        print("\nRestored original gdn_prefill_mix.py")

    # ── Threshold 1: recurrent vs v3 ──────────────────────────────────────
    rec = {u: r for u, r in results["recurrent"].items() if r["T"] <= 120}
    v3  = {u: r for u, r in results["v3"].items()        if r["T"] <= 120}
    t1, table1 = first_crossover(rec, v3, "recurrent", "v3")
    print(f"\n=== Threshold 1: recurrent → v3 ===\n{table1}")
    print(f"\n  → recurrent wins for T < {t1}, v3 wins for T >= {t1}")

    # ── Threshold 2: v3 vs chunk ───────────────────────────────────────────
    v3c  = {u: r for u, r in results["v3"].items()    if 60 <= r["T"] <= 1024}
    ckc  = {u: r for u, r in results["chunk"].items() if 60 <= r["T"] <= 1024}
    t2, table2 = first_crossover(v3c, ckc, "v3", "chunk")
    print(f"\n=== Threshold 2: v3 → chunk_v6b ===\n{table2}")
    print(f"\n  → v3 wins for T < {t2}, chunk wins for T >= {t2}")

    print(f"""
{'='*50}
RECOMMENDED gdn_prefill_mix.py thresholds:
  T < {t1:4d}  →  recurrent
  T < {t2:4d}  →  v3
  T >= {t2:4d}  →  chunk_v6b
{'='*50}""")


if __name__ == "__main__":
    main()
