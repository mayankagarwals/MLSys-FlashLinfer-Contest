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

## Current Result
- Optimized total: 8721.8 us
- Improvement: 115.8 us (1.31%)
- Remaining target: ~768 us more needed

## Key Findings
1. The Triton compiler generates excellent code on B200 Blackwell — hand-written CUDA is NOT faster
2. Python overhead is ~10-16 us per chunk call, but mostly overlaps with GPU execution
3. The H kernel (sequential recurrence) is the fundamental bottleneck for large T
4. The inverse kernel's tf32x3→tf32 off-diagonal saves ~12% on the inverse but only ~1.9% on chunk path
5. torch.compile shows 34% speedup potential but has correctness issues with TVM FFI
6. CUDA graph capture fails with mixed Triton+CUDA kernels
