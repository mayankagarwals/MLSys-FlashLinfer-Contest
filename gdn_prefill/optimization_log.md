# GDN Prefill v3 Optimization Log

## Session: April 9, 2026

### Baseline
- Branch: yue/prefill-v3-perf (commit 847691a)
- Total: 11,777 us, 100/100 correct
- v3 (42 WLs, T=64-1023): ~3,362 us
- chunk_v5 (30 WLs, T>=1024): ~7,550 us
- recurrent (28 WLs, T<64): ~865 us

### Per-kernel baseline (T=8192, N=32, v3 only)
- H-kernel: 332us (46%)
- FusedPrep: 268us (37%)
- O-kernel: 116us (16%)
- Total v3: 716us

---

### Optimization 1: H-kernel tile-format smem (SUCCESS)
**Commit**: d1cce99
**Hypothesis**: Row-major smem has 8-way bank conflicts on ldmatrix reads (stride 256B → bank diff 0). Tile format (no-swizzle, 8-row blocks) has 16B stride → all different banks.
**What changed**: Converted s_k[64,128], s_w[64,128], s_h_T[128,16], s_vnew_T[16,64] from row-major to tile format using tile_byte_offset(). cp.async writes directly to tile-format addresses. ldmatrix reads use tile_byte_offset for address computation.
**Result**:
- H-kernel: 332 → 248 us (**-25.3%**)
- FusedPrep: 268 → 267 us (unchanged)
- O-kernel: 116 → 115 us (unchanged)
- Total v3 (T=8192): 716 → 630 us (-12.0%)
- Overall: 11,777 → 11,525 us (-2.1%)
**Why it worked**: Bank conflicts were the dominant bottleneck for ldmatrix. Tile format places consecutive rows within an 8-row block at 16B stride, ensuring all 8 threads in an ldmatrix group hit different banks.

---

### Optimization 2: H-kernel TMA k loading + remove 115KB smem pad (MIXED)
**Commit**: 7d932da
**Hypothesis**: (A) TMA bulk load is faster than 128 per-thread cp.async calls. (B) 115KB smem pad may no longer be needed with tile format.
**What changed**: TMA for full chunks (clen==kBT) via k_tmap_64, cp.async fallback for partial chunks. Removed 115KB smem pad, reduced to actual ~62KB.
**Result**:
- TMA: marginal improvement (~15 us overall)
- Smem pad removal: **important finding** — multi-block scheduling (2 blocks/SM) is now FASTER than 1 block/SM with tile format. 115KB pad forced 1 block → 670us, vs 62KB allowing multi-block → 633us for T=8192.
- Overall: 11,525 → 11,485 us (-0.3%)
**Why smem pad removal helped**: With tile format eliminating bank conflicts, the bottleneck shifted from smem access to instruction throughput. Multi-block scheduling gives better warp-level parallelism (8 warps vs 4) to hide memory latencies.

---

### Optimization 3: FP Phase 3 tile-format smem (FAILED — REVERTED)
**Commit**: 999894e (reverted in e721644)
**Hypothesis**: Same bank conflict issue as H-kernel — tile format should help FP Phase 3 ldmatrix reads.
**What changed**: Converted s_Ainv_bf16[64,64] and s_input_bf16[64,128] to tile format.
**Result**: 10x SLOWER (T=134: 63→522 us). Massive regression.
**Why it failed**: Tile format has 16-way WRITE bank conflicts for regular stores! SBO=128 means all K-groups within the same 8-row block start at the same bank offset. cp.async (used in H-kernel) bypasses the bank conflict issue, but FP Phase 3 uses regular stores for scaling. 16 threads writing same row with different K-groups → all hit same bank → 16-way conflict.
**Lesson**: Tile format is ONLY beneficial when writes use cp.async or TMA. For regular store writes, use XOR swizzle instead.

---

### Optimization 4: FP Phase 3 XOR swizzle (SUCCESS)
**Commit**: f699e53
**Hypothesis**: XOR swizzle (XOR column-group index with lower bits of row) eliminates read bank conflicts while keeping write bank conflicts at 2-4x (same as row-major).
**What changed**: Added xor_swizzle_col(row, col, cg_mask) function. Applied to both s_Ainv_bf16 and s_input_bf16 writes and reads.
**Bug found & fixed**: Initial implementation used `row & 15` for both 64-col and 128-col buffers. For 64-col buffer (8 column groups), XOR with 0..15 produces swizzled_cg values 0..15, but valid range is 0..7 → buffer overflow! Fix: use cg_mask = (num_cols/8 - 1) to clamp the XOR range.
**Result**:
- T=134: 63.5 → 55.4 us (-12.7%)
- v3 total: ~3,362 → 2,829 us (-15.9%)
- Overall: 11,777 → 11,229 us (-4.7%)
**Why it worked**: Read bank conflicts eliminated (8-way → 0-way). Write conflicts unchanged at 2-way (XOR permutes within the same set of banks). Net win because reads happen more frequently (in the inner MMA loop).

---

### v3 vs chunk_v5 gap (current status)
| T | N | v3 (us) | chunk_v5 (us) | Gap |
|---|---|---------|---------------|-----|
| 1377 | 1 | 157.8 | 142.5 | v3 11% slower |
| 1592 | 3 | 147.7 | 138.6 | v3 7% slower |
| 1800 | 3 | 209.0 | 161.9 | v3 23% slower |
| 2107 | 1 | 202.5 | 163.7 | v3 19% slower |
| 8192 | 32 | ~630 | ~250 | v3 2.5x slower |

v3 is still significantly slower than chunk_v5 for T>=1024 due to:
1. Sequential H-kernel recurrence (chunks processed serially)
2. SASS-level instruction scheduling (nvcc can't match Triton's MLIR backend)
3. More smem traffic per MMA (774 LDS vs Triton's 27 for equivalent compute)

---

### Optimization 5: Pre-load all MMA2 fragments before MMA (FAILED — no improvement)
**Hypothesis**: Matching Triton's scheduling pattern (all ldmatrix before any mma.sync) would improve ILP.
**Result**: T=134: 55.5 vs 55.4 us (no change). T=959: 107.3 vs 108.3 us (-0.9%, noise).
**Why it failed**: nvcc already optimizes ldmatrix/mma interleaving when bank conflicts are eliminated. The compiler schedules loads and computes optimally without manual pre-loading. Confirmed earlier finding from memory.

---

### Per-kernel breakdown (T=959, N=4, after all optimizations)
| Kernel | Time (us) | % |
|--------|-----------|---|
| H-kernel | 58.1 | 56.6% |
| FusedPrep | 26.8 | 26.1% |
| O-kernel | 17.7 | 17.2% |
| Total | 102.6 | 100% |

H-kernel dominates at 57%. Within H-kernel, Step 0 (h store) and vnew computation are the main costs now that MMA bank conflicts are eliminated.

### Analysis: s_vnew_T write bank conflicts
- Tile format has 4-way write conflict (vs 2-way in row-major)
- XOR swizzle doesn't help because: all threads at a given j have the SAME bv value → XOR with (bv & mask) is the same for all → no redistribution
- The bank conflicts are inherent to the access pattern (same bv, different t)
- Read conflicts (0-way with tile format) outweigh the write regression since reads happen 16x more often (in MMA loop)

---

### Optimization 6: TMA w loading in H-kernel (FAILED — GPU hang)
**Hypothesis**: Replace 128 per-thread cp.async w loads with single TMA instruction. TMA for w is safe even for partial chunks because NaN in wh[t>=clen] is never read by vnew computation.
**Result**: GPU hang (mbarrier_wait never completes). CUDA_ERROR_UNKNOWN.
**Root cause**: Unknown — TMA descriptor and mbarrier pattern match the working k TMA implementation. Possible issues: parameter passing order (__grid_constant__ as last param), or init_tma_desc_3d incompatibility with the w scratch buffer layout. Needs deeper investigation.
**Reverted**: Code reverted to cp.async w loading.

---

### Optimization 7: Pre-compute tile offsets for vnew/h_T writes (NO IMPROVEMENT)
**What changed**: Pre-compute tile_byte_offset base once, use stride constants (j*16, +LBO) instead of re-calling tile_byte_offset for each write.
**Result**: T=134: 55.5 us (unchanged). Compiler already optimized this.

---

### Next steps for -10% target (630 us gap remaining)
1. **Debug TMA w loading**: The hang is unexpected since TMA k works fine with identical pattern. Could be the __grid_constant__ parameter position or some TMA descriptor issue.
2. **Fuse kernel launches**: single persistent kernel or CUDA graph to reduce 3 × 5us launch overhead
3. **Architectural**: The 2.5x v3 vs chunk_v5 gap for T>=1024 is mainly from SASS scheduling — nvcc's code gen can't match Triton's MLIR. Potential solutions:
   a. Write critical H-kernel inner loop in inline SASS
   b. Use CuTe/CUTLASS for the matmul portions
   c. Custom Triton backend generating CUDA code
4. **KEY GOAL**: v3 must beat chunk_v5 for ALL T>=64 (currently 2.5x slower for T=8192)
