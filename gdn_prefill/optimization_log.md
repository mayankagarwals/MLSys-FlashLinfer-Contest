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

---

## Session: April 10, 2026

### Optimization 8: FP Phase 3 Triton-style Ai scaling (FAILED — precision issues)
**Hypothesis**: Scale Ai matrix (64x64) instead of input (64x128): W = (Ai*beta*exp(g)) @ k. Eliminates staging buffer and scaling step. From Triton's merge_16x16_to_64x64_inverse_kernel.
**What changed**: Rewrote Phase 3 to compute Ai*bg→bf16 before MMA, load k directly to XOR-swizzled smem via cp.async.
**Bugs found**:
- Missing __syncthreads between s_bg write and Ai scaling read → race condition → non-deterministic output
- cp.async with pred=false does NOT zero smem → stale data in rows >= clen
- After fixing both: 6/42 v3 pass, 36/42 fail due to PRECISION differences
**Root cause**: Different rounding path: old code rounds Ai→bf16 then scales input as bf16×bf16. New code scales Ai in fp32 then rounds→bf16. The intermediate products differ at the bf16 precision boundary, causing ~1e-2 error that exceeds atol tolerance.
**Lesson**: Can't change the multiplication order in (Ai @ (scale * input)) without changing numerical results at bf16 precision. The tolerance (atol=1e-2) is very tight relative to output magnitudes (~1e-2).

### Optimization 9: Lower v3 dispatch threshold for N=1, T>=46 (SUCCESS)
**Commit**: 7b5a77f
**Hypothesis**: v3 is faster than recurrent for single-sequence workloads with T>=46 (1 chunk, ~48us) vs recurrent (T-dependent, ~52-65us for T=46-61).
**What changed**: Added condition `N == 1 and T >= 46` to route small single-seq workloads to v3 instead of recurrent.
**Result**: 11,252 → 11,192 us (-0.5%). 100/100 correct.
**Per-workload savings**: T=61(-16us), T=49×3(-15us), T=48(-6us) = ~37us total.
**Why it worked**: v3's fixed overhead (~42us from 3 kernel launches) is lower than recurrent's T-proportional cost (~1us/token + 10us base) for T>46 with N=1. Multi-seq workloads stay with recurrent because v3 overhead doesn't scale well with N.

### Optimization 10: chunk_v5 eliminate CUDA sync (FAILED — slower)
**Hypothesis**: Replace `chunk_offsets[-1].item()` with CPU-computed total_num_chunks from `cu_seqlens.tolist()` to avoid CUDA sync.
**Result**: 11,192 → 11,707 us (+515 us, WORSE). chunk_v5 went from 7,557 to 8,052.
**Why it failed**: `cu_seqlens.tolist()` has Python list creation overhead (~5-10us per call × 30 WLs = 150-300us), while the original `.item()` sync overhead is negligible (compute_chunks_kernel completes in ~1us).

### Optimization 11: H-kernel launch_bounds(128, 2) (NO IMPROVEMENT)
**Result**: T=134: 55.5 us (unchanged). T=959: 108.6 us (unchanged).
**Why**: Compiler already generates code compatible with 2 blocks/SM under (128, 1).

### Optimization 12: H-kernel BV_H=32 (FAILED — 22-47% slower)
**Hypothesis**: Larger BV tile (32 vs 16) gives better MMA data reuse by sharing B fragments across 2 M-subtiles.
**What changed**: kBV_H=32, kNVT_H=4. h_reg[2][4][4]. MMA2 M-subtiling. 2-pass vnew.
**Result**: T=134: 55.5→67.8 us (+22%), T=959: 108.3→159.0 us (+47%). MASSIVE regression.
**Why it failed**: 
- 2x more MMA calls per block (32 vs 16 in MMA2)
- 2-pass vnew adds sequential overhead
- Fewer blocks (4 vs 8 v-tiles) reduces parallelism on 192-SM B200
- Smem grows from 60KB to 68KB, potentially limiting multi-block scheduling
- The B fragment reuse (loading B once for 2 MMAs) doesn't compensate for the 2x compute increase
**Lesson**: BV_H=16 is the sweet spot for B200. Smaller tiles = more parallelism from more blocks.

### Research Agent Findings Summary (Apr 10)

**Triton H-kernel analysis:**
- Uses 3-stage pipeline (num_stages=3): prefetches 3 chunks ahead vs our 1.5
- Triton loads all K-fragments upfront → all MMAs back-to-back
- ~450-500 cycles of latency hiding vs our ~200 cycles
- Serial h_reg dependency prevents cross-chunk overlap regardless
- Key gap: SASS instruction scheduling (verified)

**Algorithmic optimization analysis:**
1. H+O kernel fusion: eliminate d_h roundtrip (~268MB/seq) — HIGH IMPACT but complex
2. Pre-compute exp(g): save ~384 expf/chunk across H+O — MEDIUM
3. Phase 2 reduce iterations: 3→2 only works for depth<8, but 16x16 block needs all 3
4. W/U computation sharing: Triton does this, but precision constraints prevent in CUDA
5. d_h workaround: pass h through smem/L2 instead of global — easier than full fusion

### Remaining viable approaches (by estimated impact):
1. **H+O fusion**: -200+ us (eliminates kernel launch + d_h traffic). Multi-day effort.
2. **Pre-compute exp(g)**: -50 us (eliminates redundant expf). 1-2 hour effort.
3. **Tune dispatch threshold**: Already done, diminishing returns.
4. **CuTe/CUTLASS for MMA**: Could match Triton SASS. Multi-day effort.

### Optimization 13: chunk_v5 use upper_bound_chunks (FAILED — race conditions)
**Hypothesis**: Use upper_bound_chunks instead of total_num_chunks to avoid CUDA sync. Zero-init chunk_indices so excess blocks safely re-process chunk 0.
**Result**: 81/100 correct. 19 workloads failed.
**Why it failed**: Excess blocks write to the same output locations as the real chunk 0 block. Concurrent writes from multiple blocks to the same address create race conditions (non-atomic partial writes). Even though the computation is identical, the timing of writes differs.
**Lesson**: Can't use upper_bound_chunks with duplicate work — all grid blocks must process UNIQUE chunks.

### Optimization 14: Cache intermediate tensor allocations (SUCCESS — BIG WIN)
**Commit**: 634cc28 + 8ea813a + f609ca7
**Hypothesis**: torch.empty calls have measurable CUDA memory allocator overhead that shows up in benchmark timing. Caching tensors across calls avoids this overhead.
**What changed**: 
- chunk_v5: Cache 10 intermediate tensors + 3 metadata tensors (keyed by problem dimensions)
- cuda_v3: Cache output + new_state tensors
- gdn_prefill_mix: Cache recurrent output tensors
**Result**: 11,202 → 10,911 us (-2.6%, cumulative -7.3%). 100/100 correct.
- chunk_v5 savings: 7,547 → 7,259 us (-288 us from caching 10 tensors)
- v3 savings: minimal (only 2 tensors cached)
- recurrent savings: minimal (only 2 tensors cached)
**Why it worked**: CUDA memory allocator (via PyTorch caching allocator) has per-allocation overhead of ~3-10us. With 10 allocations per chunk_v5 call × 30 calls = 300 calls eliminated → ~900-3000us savings potential. Actual savings ~288us consistent with ~10us/alloc overhead.
**Key insight**: Python-level optimization (tensor caching) gave the second-largest improvement after H-kernel bank conflict elimination. Always check for allocation overhead.

### Optimization 15: Compute chunk metadata on CPU (FAILED — CUDA error)
**Hypothesis**: Replace Triton compute_chunks_kernel + .item() sync with CPU computation of chunk_indices via cu_seqlens.tolist() + torch.tensor transfer.
**Result**: CUDA_ERROR_UNKNOWN. The CPU-created tensor might have format issues or the transfer races with pending GPU work.
**Reverted**: Kept tensor caching for metadata but kept Triton kernel for computation.

### Current Status
- **10,889 us (-7.5%)**, 100/100 correct
- Gap to target: 290 us
- Python-level optimizations exhausted (tensor caching applied everywhere)
- GPU compute is the remaining bottleneck
- chunk_v5 .item() sync overhead is ~5-10us × 30 = 150-300us but can't be eliminated

### Optimization 16: Recurrent kernel sm_100a + libcuda (FAILED — runtime crash)
**What changed**: Added `TVM_FFI_CUDA_ARCH_LIST=10.0a` and `-lcuda` to recurrent kernel build.
**Result**: Compilation succeeds but benchmark crashes (30/100 correct).
**Root cause**: `-lcuda` likely conflicts with the recurrent kernel's simpler CUDA usage.
**Reverted**.

### Final Status: 10,885 us (-7.6%), 100/100 correct
- Gap to -10% target: ~286 us (purely GPU compute)
- All Python-level optimizations exhausted (tensor caching, dispatch threshold)
- All accessible CUDA-level optimizations exhausted (bank conflicts, TMA, XOR swizzle)
- Remaining gap requires: H+O kernel fusion, CuTe/CUTLASS, or Triton kernel modifications

## Session: April 10, 2026 (continued)

### Optimization 17: Triton H-kernel BV=32 for large N (FAILED — slower)
**Result**: 10,885 → 10,961 us (+76 us). cv5 top workloads got slower (331 vs 324 us).
**Why**: Fewer v-tile blocks (4 vs 8) reduces parallelism and occupancy hiding.

### Optimization 18: Overlap .item() sync with kkt_v1 (FAILED — excess work)
**Hypothesis**: Launch kkt_v1 with upper_bound_chunks before .item() sync to overlap CPU wait with GPU compute.
**Result**: 10,885 → 11,143 us (+258 us, much worse).
**Why**: kkt_v1 processes upper_bound_chunks iterations (up to 44% more for N=57) instead of total_num_chunks. Excess work >> sync savings.

### Optimization 19: Triton H-kernel num_stages=4 (FAILED — crash)
**Result**: Benchmark crashes (GPU idle, no TSV). Likely register overflow from 4-stage pipeline.

### Optimization 20: Triton H-kernel BV=32 for large N*H (FAILED — slower)  
**Result**: +76 us overall. Fewer v-tile blocks hurt parallelism.

### Current confirmed best: 10,874 us (-7.7%). Gap: 275 us to -10% target.

### Optimization 21: Block-wise W/U in merge kernel (FAILED — incorrect scaling)
**Hypothesis**: Skip global Ai store+reload+debug_barrier in merge_16x16_to_64x64_inverse_kernel.
Compute W/U directly from register-resident Ai blocks. Also fewer MMA calls (160 vs 256, skips zero upper-triangular blocks).
**Result**: First attempt: 70/100 (beta scaled rows instead of columns). Fix attempt: 10/100 (Triton [None, :] indexing issue).
**Root cause**: `Ai * beta` broadcasts beta along columns (j), not rows (i). The block-wise version needs `Ai_21 * beta[col_range][None, :]` but Triton's `[None, :]` indexing doesn't behave as expected on sliced 1D tensors. Needs `tl.expand_dims` or `reshape` instead.
**Status**: Reverted. The approach is SOUND but needs correct Triton tensor manipulation.
**Potential savings**: Eliminates debug_barrier sync + global Ai roundtrip + 37% fewer MMA calls. Could save 50-100+ us on chunk_v5 workloads.
**TODO for next session**: Fix using tl.expand_dims(beta[0:16], 0) instead of beta[0:16][None, :].

### Optimization 22: Block-wise W/U in merge kernel (SUCCESS — 98/100, -900us cv5)
**Commit**: 62910ee
**What changed**: In merge_16x16_to_64x64_inverse_kernel, skip the global Ai reload after debug_barrier. Instead, compute W=(Ai*beta*exp(g))@K and U=(Ai*beta)@V block-wise from the 10 register-resident Ai blocks (lower-triangular structure, skip 6 zero blocks).
**Key fix**: 1D beta broadcasting (`[16,16] * [16]`) correctly scales columns, not rows. Earlier attempts with `[None,:]` or `[:, None]` failed.
**Result**: cv5: 7,240 → 6,340 us (-900 us, -12.4%). Total: 10,874 → 10,012 us (-15.0% from baseline).
**2 failures**: T=8192 N=38 and N=43 fail consistently. Block-wise tf32x3 accumulation rounds differently than single large matmul. Edge cases where rounding crosses atol=1e-2.

### Optimization 23: Block-wise W/U with bf16 roundtrip (SUCCESS — 100/100, 10,670 us)
**Commit**: f7dca7b
**Hypothesis**: The 2 failures come from Ai blocks retaining full float32 precision. The L2 reload approach stores Ai as bf16 then loads back, introducing a roundtrip that changes Ab values. Adding `.to(tl.bfloat16).to(tl.float32)` to each Ai block matches this precision profile. Combined with `acc=` chaining, the MMA accumulation order is identical to the single [64,64] dot.
**What changed**: After computing 10 Ai blocks in float32, cast each to bf16 and back. Use `acc=` parameter for block-wise tl.dot calls to chain the accumulation (K=16+16+... = same as K=64 in one dot).
**Result**: 100/100 correct. Total: 10,894 → 10,670 us (-2.1%, cumulative -9.4% from baseline).
**Why it worked**: The bf16 roundtrip ensures `round_bf16(Ai * beta)` produces identical bf16 values to the L2 approach. The `acc=` chaining feeds the MMA accumulator across dot calls, matching the single-dot accumulation order exactly.
**Confirmed**: Without bf16 roundtrip (pure acc= chaining), still 98/100 — same 2 failures (T=8192 N=38, N=43). Roundtrip IS needed.
**Also tried**: num_warps=4 → benchmark hangs (register spilling causes 10x regression).

### Optimization 24: Ai inverse DOT_PRECISION=tf32 (SUCCESS — 100/100, 10,246 us)
**Commit**: 90c07fd
**Hypothesis**: The bf16 roundtrip masks reduced Ai precision. tf32 (1 MMA pass) is 3x fewer passes than tf32x3 for the 36 Ai inverse dots per CTA. The bf16 rounding after the inverse discards the extra precision that tf32x3 provides.
**What changed**: Changed DOT_PRECISION from "tf32x3" to "tf32" for both chunk_v5 and triton_v4 merge kernel calls.
**Result**: 100/100 correct. Total: 10,670 → 10,246 us (-4.0%, cumulative **-13.0%** from baseline).
- cv5: 7,021 → 6,597 us (-424 us, -6.1%)
**Why it worked**: After the Ai inverse, all blocks are cast to bf16 (`.to(tl.bfloat16).to(tl.float32)`). This discards mantissa bits beyond bf16's 8-bit significand. The difference between tf32 (11-bit) and tf32x3 (~23-bit) Ai values is entirely lost in the bf16 truncation. So the W/U dot products see identical bf16 inputs regardless of Ai inverse precision.
**Key insight**: The bf16 roundtrip not only fixes the precision mismatch (Opt 23) but also enables lower precision in the inverse computation. The two optimizations are synergistic.

### Current Status: 10,246 us (-13.0%), 100/100 correct
- **TARGET (-10% = 10,599 us) EXCEEDED by 353 us**
- cv5: 6,597 us (-12.5% from 7,550 baseline)
- v3: ~3,063 us (-8.9%)
- recurrent: ~586 us
