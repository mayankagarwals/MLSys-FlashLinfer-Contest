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

### Next steps
1. Profile per-kernel breakdown with nsys for current code
2. Optimize H-kernel further: vnew computation, step0, warp specialization
3. Optimize FP Phase 3 MMA scheduling (pre-load all fragments before MMA)
4. Consider architectural changes: persistent kernel, kernel fusion
5. **KEY GOAL**: Close v3 vs chunk_v5 gap for T>=1024
