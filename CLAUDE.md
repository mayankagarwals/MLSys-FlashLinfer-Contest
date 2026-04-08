# Project: MLSys FlashLinfer Contest

## Remote Server
- SSH: `SSH_AUTH_SOCK="" ssh -o IdentitiesOnly=yes -F /dev/null -i ~/.ssh/id_ed25519_9701account yue@95.133.252.11`
- Datasets: `/home/yue/mlsys26-contest` (official 100 workloads)
- Build: `scripts/pack_solution.py` packs solution into JSON

## How to Benchmark
- From remote server, in the gdn_prefill directory:
  ```
  cd /home/yue/MLSys-FlashLinfer-Contest/gdn_prefill/scripts
  uv run python run_local_fast.py --local /home/yue/mlsys26-contest
  ```
- This reads `gdn_prefill/config.toml` to pick the entry point (currently `cuda_parallel_v3.py::run`)
- To compare against Triton baseline, change config.toml entry_point to `triton_v4.py::run` and re-run
- Output shows per-workload latency, speedup factor, and correctness (PASSED/FAILED)

## Current Goal
- Make `cuda_parallel_v3.cu` faster than `triton_v4.py` across all 100 workloads
- Both correctness AND performance matter — a faster but incorrect kernel is useless
- Big refactor is fine we have enough tokens and time

## Optimization Workflow — AUTONOMOUS LOOP

When optimizing CUDA/Triton kernels, you MUST follow this loop WITHOUT stopping:

1. Make ONE targeted change to the kernel
2. Build and test on remote server
3. Run benchmark, record timing
4. Compare to target — if not met, analyze WHY and pick next approach
5. **Go back to step 1 — do NOT stop to ask what to do next**

**STOP ONLY when:**
- Target performance is achieved
- You've tried 10+ approaches with no progress
- You're genuinely stuck and need human input (not just "what should I try next")
- A correctness regression you cannot diagnose

**NEVER do these:**
- Do NOT ask "what should I try next?" — decide yourself and keep going
- Do NOT summarize intermediate results and wait for approval — just continue
- Do NOT stop after one successful change to celebrate — immediately try the next optimization
- If a change makes things worse or breaks correctness, revert and try something else WITHOUT asking

**After each successful improvement** (correct + faster), git commit the change to preserve progress.
