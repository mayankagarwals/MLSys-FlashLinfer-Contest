# GDN Prefill Optimization Log

## Baseline (2026-04-12)
- Total: 8837.7 us across 100 workloads
- chunk_v7b: 5111.2 us (57.8%) — 31 workloads, T>=525, N>2
- cuda_v4: 2040.5 us (23.1%) — 37 workloads, 64<=T<525
- chunk_v6b: 1097.3 us (12.4%) — 9 workloads, T>=525, N<=2
- recurrent: 588.6 us (6.7%) — 23 workloads, T<64
- Target: 10% improvement = save 883.8 us

## Per-Kernel Profiling (T=8192, N=32, chunk_v7b path)
| Kernel | Time | % |
|--------|------|---|
| h_kernel (CUDA h_v1) | 75.3 us | 40.8% |
| inverse (Triton) | 47.5 us | 25.7% |
| O kernel (Triton) | 37.4 us | 20.3% |
| kkt (CUDA) | 17.8 us | 9.7% |
| compute_chunks (Triton) | 6.6 us | 3.6% |

## Optimization Attempts

### 1. tf32 off-diagonal in inverse kernel (APPLIED)
- Changed off-diagonal block products from tf32x3 to tf32 in chunk_v6c.py
- Kept tf32x3 for Ai_41 (deepest chain) to maintain correctness
- Result: inverse kernel 12% faster (47.5 → 41.7 us)
- Correctness: 100/100 with hybrid tf32/tf32x3
- Overall impact: +1.8% on chunk path, +1.2% overall

### 2. O kernel num_warps tuning (APPLIED)
- Tested BV=32/64/128 with num_warps=4/8/16
- Best: BV=64 num_warps=4 (36.8 us vs 36.9 us for num_warps=8)
- Marginal improvement (<1%)

### 3. Inverse kernel num_warps tuning (TESTED)
- num_warps=1: 72.4 us (worse)
- num_warps=2: 47.8 us (current, best)
- num_warps=4: 70.1 us (worse)
- num_warps=8: 129.4 us (much worse)

### 4. v6b vs v7b dispatch for N<=2 (APPLIED)
- Tested all 9 v6b workloads with both paths
- v7b faster for 7/9 workloads (T<2500)
- Changed dispatch: T>=2500 N<=2 → v6b, else → v7b
- Total savings: ~27 us (0.3%)

### 5. Use v4 pipeline for T>=525 (REJECTED)
- v4 is 3-4x slower for T>=525 due to FusedPrepKernel's launch_bounds(128,1) limiting occupancy

### 6. torch.compile (REJECTED)
- 34% speedup on T=8192 N=32 but produces INCORRECT results
- TVM FFI calls not handled correctly by torch.compile graph tracer
- Recompilation overhead for varying shapes

### 7. CUDA Graph capture (REJECTED)
- Graph capture fails with mixed CUDA + Triton kernels
- "The CUDA Graph is empty" warning

### 8. Fused kkt+inverse Triton kernel (ABANDONED)
- Register pressure too high: A_mat[64,64] = 4096 fp32 in registers
- Triton tensor slicing from 64x64 to 16x16 blocks uncertain
- A matrix fits in L2 cache anyway (16MB << 96MB L2)

### 9. CUDA inverse kernel (ESTIMATED)
- Would replace Triton inverse with CUDA using mma.sync + wmma
- Estimated 7% faster due to better instruction scheduling
- But insufficient improvement (~112 us total) for the implementation effort

### 10. Full CUDA chunk pipeline (IN PROGRESS)
- Replace ALL Triton kernels with CUDA equivalents
- Single C++ driver function eliminates ~34 us Python overhead per call
- Estimated impact: 40 workloads × 34 us = 1360 us (15.4%)
- Status: background agent writing cuda_chunk_pipeline.cu

### 11. Fix v6b import to use optimized chunk_v6c (APPLIED)
- Changed `from .chunk_v6b import run as chunk_v6c` to `from .chunk_v6c import run as chunk_v6c`
- Previously the v6b dispatch path used the OLD inverse (tf32 DOT_PRECISION without bf16+correction)
- Now uses the optimized inverse (bf16+correction + tf32 off-diagonal)
- Additional ~12 us savings on the 2 v6b workloads

### 12. CUDA inverse+WU kernel (TESTED, REJECTED)
- Wrote a full CUDA InverseWUKernel (1258 lines) adapted from FusedPrepKernel Phase 2+3
- Correctness: GOOD (odiff=0.000244)
- Performance: 2x SLOWER than Triton inverse (368 us vs 190 us for T=8192 N=32)
- Reason: Triton generates efficient code on B200; wmma approach in CUDA has higher overhead
- Conclusion: Triton's mma.sync is competitive with hand-written CUDA on Blackwell

### 13. Triton H kernel BV tuning (TESTED, NO CHANGE NEEDED)
- Tested BV=8/16/32 with num_warps=2/4/8 and num_stages=2/3 for T=1377 N=1
- Current setting (BV=16, num_warps=4, num_stages=3) is already optimal at 41.0 us
- BV=8: worse (43-66 us) despite better SM utilization — per-TB overhead dominates
- BV=32: worse (51-70 us) — too few TBs

### 14. Fuse chunk metadata into kkt kernel (APPLIED)
- Added prep_meta_kernel (128 threads, parallel per-seq) + kkt_v1b_with_meta to cuda_kkt_v1b.cu
- Eliminates Triton compute_chunks_kernel launch and Python-side setup
- Saves ~1 Python function call + 1 Triton kernel launch + tensor allocations
- prep_meta takes 7.3 us (vs Triton's 6.6 us) — slower kernel but fewer launches
- Net savings: ~38 us over 40 chunk workloads

## Session 2 Optimizations (2026-04-14)

### 15. L2 EVICT_LAST cache hint for K in H kernel (APPLIED)
- K tensor shared across H/Hg=2 head groups — EVICT_LAST keeps it in L2
- H kernel: 79.4 us vs 79.7 us (0.4% faster per call)
- Overall: ~5 us total savings

### 16. Dispatch threshold tuning: recurrent for T<67/N=1 + T<64/N=2 (APPLIED)
- Profiled crossover: recurrent 40 us vs v5 49 us for T=49/N=1
- Changed from `T>=64 || (N==1 && T>=46)` to `T>=64 && (N>=3 || T>=67)`
- Saves ~30 us from 8 workloads switched to faster path

### 17. v6c/v7b dispatch: v7b for T<600 N<=2 (APPLIED)
- v7b (CUDA H) 5-6 us faster than v6c (Triton H) for small T, N<=2
- Changed: T>=525 with N<=2 AND T>=600 → v6c, else → v7b
- Saves ~12 us from 2 workloads

### 18. v5 framework integration (INFRASTRUCTURE)
- Created cuda_h_kernel.h with tcgen05 H kernel in namespace hv1
- Created RunHAndO C++ function for potential H+O fusion
- NOT used in production — OOutputKernel precision doesn't match Triton O

### 19. FusedPrep for T>=525 (TESTED, REJECTED)
- FusedPrep is 2-3x SLOWER than Triton KKT+inverse for T>=525
- T=8192: 219 us (FusedPrep) vs 70 us (Triton)
- Only competitive for T<700

### 20. H kernel dual-pass fusion (TESTED, REJECTED)
- Merged bf16 conversion + scaling passes into single pass
- SLOWER (80.2 us vs 79.4 us) because breaks pipeline between wh and vk MMAs
- Two-pass design is intentional for latency hiding

### 21. Tensor pool for v7b/v6c (TESTED, REJECTED)
- Pre-allocated tensors with view slicing to avoid per-call malloc
- SLOWER (8522 vs 8410 us) — view creation overhead exceeds allocation savings

### 22. CUDA inverse kernel (IN PROGRESS, BLOCKED)
- 1240-line cuda_inverse_v1.cu using mma.sync bf16 m16n8k16
- CRITICAL: Triton uses mma.sync (not tcgen05) for 16x16 bf16 on sm_100a
- Has fundamental MMA register/smem mapping bug (identity test fails with diff ~3.9)
- Needs standalone MMA test harness to debug
- If fixed: could replace Triton inverse (42 us) and enable kernel fusion

### Key Precision Findings
- FusedPrep inverse IS precise enough (max W/U diff 0.004 vs Triton)
- The ONLY precision issue is OOutputKernel vs Triton O kernel
- FusedPrep + CUDA H + Triton O → 0/0 failures
- Python overhead is only 3.3 us/workload (not a bottleneck)

## Current Result (after session 2)
- Optimized total: ~8365 us (measurement varies 8360-8390)
- Total improvement: ~473 us (5.4% from baseline 8837.7 us)
- Remaining gap to 10%: ~411 us

## Remaining Optimization Opportunities
1. **CUDA inverse kernel** (highest impact, ~400 us if debugged and faster than Triton)
   - Need to fix mma_16x16_bf16 helper function
   - Test with standalone harness before integrating
2. **Algorithmic changes** (e.g., different chunk sizes for specific workloads)
3. **ncu profiling** of hottest kernels to find micro-optimization opportunities

## Key Findings
1. The Triton compiler generates excellent code on B200 Blackwell — hand-written CUDA is NOT faster
2. Python overhead is ~10-16 us per chunk call, but mostly overlaps with GPU execution
3. The H kernel (sequential recurrence) is the fundamental bottleneck for large T
4. The inverse kernel's tf32x3→tf32 off-diagonal saves ~12% on the inverse but only ~1.9% on chunk path
5. torch.compile shows 34% speedup potential but has correctness issues with TVM FFI
6. CUDA graph capture fails with mixed Triton+CUDA kernels
