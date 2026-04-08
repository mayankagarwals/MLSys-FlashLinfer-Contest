---
name: CUDA parallel v1 optimization status
description: Current optimization state of cuda_parallel_v1.cu kernel with tcgen05 on B200
type: project
---

# CUDA Parallel v1 Optimization Status (2026-04-03)

## Current Changes vs Original Baseline
1. **MMA attn@vnew**: Replaced serial dot product in FusedRecurrence step 4b with tcgen05 MMA. Formats masked attn[64,64]→tile_a bf16, vnew^T→tile_b bf16, does BK=64 MMA. Requires k TMA reload after (overlapped with TMEM read). ~3-6% speedup on long sequences.

2. **Precomputed q@k^T**: New ComputeQKKernel_TC runs after ComputeWU, reuses d_A buffer. Stores masked+gated q@k^T. FusedRecurrence loads from global instead of computing q@k^T MMA inline. Removes 1 MMA + 1 TMA from critical sequential path. Additional 2-5% on long sequences, but adds overhead for short multi-seq workloads.

## Performance (vs original cuda_parallel_v1)
- T=5709, 2 seqs: 3766→3382us (11.4% faster)
- T=2107, 1 seq: 1450→1343us (8.0% faster)
- T=8192 multi-seq: ~neutral (extra kernel launch overhead offsets savings)
- Total across 100 WL: 74835→72992us (2.5% overall)
- Triton v4 target: still ~8x faster on largest workloads

## BV=64 Attempt — FAILED
- Changing kBV from 32 to 64 causes NaN in state output for ~35/100 workloads
- Failure pattern is data-dependent (not chunk-count dependent)
- Race condition in k→tile_a transpose vs vnew→tile_b transpose was found and fixed (added __syncthreads) but didn't resolve the NaN
- BV=32 serial dot product also fails with BV=64 → bug is in BV=64 itself, not MMA change
- Root cause unknown — needs deeper investigation

## Key Bug Found: OOB Write in Precomputed QK
- ComputeQK must limit writes to clen rows (not kBT) to avoid corrupting packed sequence data
- FusedRecurrence must zero-fill s_wh rows >= clen when loading from global
- Same pattern as ComputeA which correctly uses `clen * kBT` as loop bound

## Next Steps (from optimization_ideas.md)
- Fuse ComputeA + SolveTril to eliminate global memory round-trip
- Block-recursive SolveTril (much more parallel than row-by-row forward sub)
- Pipeline chunk iterations with double-buffering in FusedRecurrence
- Consider removing precomputed qk for short-sequence workloads (conditional launch)
