# GDN Prefill Kernel Profiling & Optimization Analysis

## Current Performance: 8,427 us total (100 workloads, 100/100 correct)

---

## 1. Dispatch Paths & Per-Workload Latencies

| Path | Workloads | Total (us) | Avg (us) | Condition |
|------|-----------|-----------|----------|-----------|
| chunk_v8b (CUDA tcgen05 H) | 26 | ~4,300 | 165 | T>=525, high N |
| chunk_v8c (Triton H) | 14 | ~1,550 | 111 | T>=525, low N |
| cuda_v4 (FusedPrep+H+O) | 32 | ~1,800 | 56 | 64<=T<525 |
| cuda_recurrent_v1 | 28 | ~870 | 31 | T<64 |

---

## 2. Per-Kernel Breakdown (nsys)

### v8b path: T=8192 N=32 (total ~187 us)

| Kernel | Time (us) | % | Grid | Block |
|--------|----------|---|------|-------|
| h_kernel_cutlass (tcgen05 H) | 74.1 | 39.6 | (8, 32) | 320 |
| merge_16x16_to_64x64_inverse | 39.1 | 20.9 | (159, 8) | 64 |
| chunk_fwd_kernel_o (Triton O) | 37.3 | 19.9 | (159, 8) | 256 |
| kkt_v1b_kernel_cutlass | 17.5 | 9.3 | (37, 4) | 192 |
| prep_meta_kernel | 4.3 | 2.3 | (1, 1) | 128 |

### v8c path: T=5709 N=2 (total ~242 us)

| Kernel | Time (us) | % | Grid | Block |
|--------|----------|---|------|-------|
| chunk_gated_delta_rule_fwd_kernel_h (Triton H) | 139.6 | 63.0 | (8, 16) | 128 |
| merge_16x16_to_64x64_inverse | 35.2 | 15.9 | (~90, 8) | 64 |
| chunk_fwd_kernel_o (Triton O) | 28.0 | 12.6 | (~90, 8) | 256 |
| kkt_v1b_kernel_cutlass | 14.7 | 6.6 | (37, 4) | 192 |
| prep_meta_kernel | 4.3 | 1.9 | (1, 1) | 128 |

### cuda_v4 path: T=461 N=2 (total ~68 us)

| Kernel | Time (us) | % |
|--------|----------|---|
| HRecurrenceKernel (mma.sync H) | 30.6 | 44.9 |
| FusedPrepKernel | 25.4 | 37.3 |
| OOutputKernel (tcgen05 O) | 12.1 | 17.8 |

---

## 3. ncu Analysis: What Are The Threads Actually Doing?

Note: low occupancy is NOT the problem. Flash Attention also runs at low occupancy.
The real question: **what are threads waiting on when not doing useful work?**

### v8b path: T=8192 N=32

| Kernel | Dur(us) | Compute% | Mem% | Idle% | Root Cause |
|--------|---------|----------|------|-------|------------|
| H kernel | 81 | 8% | 23% | **69%** | **Barrier sync between warp groups** — TMA/MMA/CUDA warps wait for each other via mbarrier. 10 warps but only 1-2 active at any time. |
| inverse | 50 | 17% | 45% | **38%** | **DRAM latency** — L2 hit rate 16.5%, most loads go to HBM. Pipeline stalls waiting for data. |
| O kernel | 40 | 31% | 44% | **25%** | **Most efficient** — TMEM pipeline 23% utilized. Room to improve memory throughput. |
| KKT | 20 | 16% | 24% | **60%** | **Warp specialization overhead** — similar to H kernel, TMA/MMA/CUDA warps idle-waiting. |

### v8c path: T=5709 N=2

| Kernel | Dur(us) | Compute% | Mem% | Idle% | Root Cause |
|--------|---------|----------|------|-------|------------|
| Triton H | 144 | 10% | 18% | **72%** | **Sequential chunk loop with per-chunk global loads** — each chunk does 3 global loads (w,v,k) + 1 global store (h,v_new). Pipeline can't hide this latency with only num_stages=5. |
| inverse | 45 | 11% | 29% | **60%** | Same DRAM latency issue as v8b path. |
| O kernel | 32 | 25% | 36% | **39%** | Fewer blocks (low-N) → less parallelism to saturate memory bus. |
| KKT | 18 | 12% | 19% | **69%** | Same warp specialization overhead. |

### cuda_v4 path: T=294 N=3

| Kernel | Dur(us) | Compute% | Mem% | Idle% | Root Cause |
|--------|---------|----------|------|-------|------------|
| FusedPrep | 29 | 4% | 10% | **86%** | **Small grid (3×8=24 blocks for 192 SMs)** — most SMs idle. Also tf32 WMMA inverse is compute-heavy but few blocks. |
| H (mma.sync) | 25 | 6% | 22% | **72%** | **Sequential chunks with strided global loads** — each chunk loads u,w,k with large strides. |
| O (tcgen05) | 15 | 6% | 10% | **84%** | **Very small grid** — few blocks, most SMs idle. |

---

## 4. Optimization Ideas (ranked by actual stall reason, not occupancy)

### 4.1 H kernel (69% idle — v8b, 39.6% of total time) — HIGHEST PRIORITY

**ncu warp stall breakdown (T=8192 N=32):**

| Stall Reason | % | What It Means |
|---|---|---|
| **long_scoreboard** | **31.8%** | Waiting for global memory loads (g_cu reads, TMA completion) |
| **barrier** | **19.6%** | bar_sync between H warps and V warps |
| **short_scoreboard** | **13.2%** | Waiting for TMEM ops (tcgen05 ld/st) |
| **sleeping** | **7.3%** | mbarrier_wait spin-loops (inherent to warp specialization) |

**Target the top stalls:**

- **A. Prefetch g_cu to reduce long_scoreboard (31.8%)**: H warps load g_cu_last from global memory at the START of each chunk (line 387: `g_cu_ptr[last_idx * H + head_id]`). V warps also load g_cu (line 588). These are scattered reads with high latency.
  - *Fix*: Pre-load g_cu for the NEXT chunk into smem or registers during the current chunk's PROCESS_SCALED_H/PROCESS_SCALED_V phase (which overlaps with vk MMA). Use PTX `ld.global.L1::no_allocate` to avoid polluting L1.
  - *Fix*: For V warps, the v_scale computation (line 584-594) loads BT=64 g_cu values. Pre-compute these into shared memory before the wh MMA wait, hiding the latency behind MMA.

- **B. Reduce bar_sync count to cut barrier stalls (19.6%)**: Currently 2 bar_sync per chunk: `bar_sync<1>(128)` in H warps (line 396/411) and `bar_sync<2>(128)` in V warps. These synchronize 4 warps each.
  - *Fix*: Check if warp_id_ == 3's `cp_async_bulk_wait_group_read<0>()` can be moved to overlap with other work, reducing the bar_sync's critical path.
  - *Fix*: The bar_sync at end of PROCESS_SCALED_H (line 488) synchronizes for the TMA store — could use a lighter `__syncwarp()` + fence instead.

- **C. Reduce tcgen05 ld/st latency (short_scoreboard 13.2%)**: tcgen05_ld/st for TMEM access.
  - *Fix*: Interleave tcgen05_st with scalar computation (e.g., h_scale multiply) using PTX asm to prevent the compiler from serializing them.
  - *Fix*: Use `tcgen05.st` for multiple tiles in a batch before calling `tcgen05_wait_st`, reducing fence overhead.

- **D. Double-buffer g_cu in shared memory**: Allocate 2× BT floats in smem. While chunk N uses buffer A, pre-load chunk N+1's g_cu into buffer B. Eliminates global memory latency from the critical path entirely.

### 4.2 Inverse DRAM latency (38% idle — v8b, 20.9% of total time) — HIGH PRIORITY

**Stall reason:** 83% of L2 accesses miss → DRAM round-trips (~200 cycles each). The inverse loads 8 blocks of k/v (each 16×128 bf16 = 4KB) + 10 A blocks (16×16 fp32 = 1KB each). Total ~42KB per thread block — fits in L2 but gets evicted by other blocks.

**Ideas:**
- **A. CUDA inverse with TMA bulk loads**: Replace Triton's scattered `tl.load` with TMA bulk copies to shared memory. TMA loads are asynchronous and can pipeline with compute. The inverse's Neumann computation (~24 MMA calls) provides enough compute to hide TMA latency.
- **B. tcgen05 MMA for W/U computation**: Column-based approach: pack A_inv columns into [64,16] smem, use 4 tcgen05 calls for [64,128]=[64,16]@[16,128] instead of 640 mma.sync calls. Dramatically reduces instruction issue overhead.
- **C. L2 residency hints**: Use `prefetch.global.L2` for K data early in the kernel (K is shared across 2 head groups). Pre-warming L2 before the main load phase.
- **D. Interleave loads with inverse MMA**: Issue k/v/beta/g_cu loads BEFORE the off-diagonal computation so they overlap with the MMA-intensive inverse chain.

### 4.3 O kernel memory throughput (25% idle — v8b, 19.9% of total time) — MEDIUM PRIORITY

**Stall reason:** 44% memory throughput means bus isn't saturated. The kernel loads h[BV,K_dim], v[BT,BV], q[BT,K_dim], k[BT,K_dim] — mostly sequential, well-coalesced, but could be faster.

**Ideas:**
- **A. CUDA O kernel with TMA + tcgen05**: Replace Triton mma.sync with TMA loads → smem → tcgen05 MMA. ~28 tcgen05 calls replaces ~896 mma.sync. TMA enables pipelining loads with MMA.
- **B. Vectorized loads (PTX ld.global.v4)**: Verify Triton generates 128-bit loads. If not, use a CUDA kernel with explicit `ld.global.v4.b32` for q/k/h/v loads.
- **C. Pipeline the BV loop**: The O kernel has `for i_v in static_range(2)` loop. Pre-load h/v for iteration 1 while computing iteration 0's MMA. Triton's static_range doesn't pipeline — would need manual restructuring or switching to `tl.range` with `num_stages=2`.
- **D. Shared memory for q@k^T result**: The 64×64 attention matrix A is in registers (~128 regs/thread). Moving to smem frees registers and allows the compiler to better schedule other instructions.

### 4.4 FusedPrep (cuda_v4 path, 37% of v4 time) — MEDIUM PRIORITY

**Stall reason:** Small grid for low-N workloads + tf32 WMMA inverse is compute-heavy.

**Ideas:**
- **A. bf16 inverse (same as v8c optimization)**: Replace tf32 WMMA with bf16 mma.sync for inverse off-diagonal. Apply the same early-roundtrip trick.
- **B. Persistent kernel**: Instead of N×Hg blocks, use persistent blocks that loop over chunks. Keeps SMs busy for small-N workloads.

### 4.5 Triton H kernel (v8c path, 63% of v8c time) — MEDIUM-HIGH PRIORITY

**Stall reason:** 72% idle — sequential chunk loop, each chunk does 3 global loads + 2 global stores. num_stages=5 hides some latency but not enough for large T.

**Ideas:**
- **A. Increase num_stages further**: Try 7-8 stages (tested 7 earlier, was slightly worse — may need retesting with current inverse optimization).
- **B. CUDA H kernel with smaller BV**: A tcgen05 H kernel with BV=32 or BV=16 would give more blocks and use TMA for loads (vs Triton's global loads). TMA's asynchronous pipeline would hide latency much better.
- **C. Vectorized global stores**: v_new and h stores may not be 128-bit vectorized by Triton. PTX-level stores (`st.global.v4.b32`) could help.

### 4.6 Cross-cutting: PTX-level instruction optimization

- **A. ILP in scalar sections**: Between MMA calls, kernels do scalar math (exp, multiply, address compute). Inspect SASS to verify these are interleaved with memory ops, not serialized.
- **B. Register reuse**: Avoid spilling by using PTX `asm volatile` with explicit register constraints for hot inner loops.
- **C. Warp-level prefetch**: Use `prefetch.global.L2` in idle warps to pre-warm cache for next chunk's data.

---

## 5. Priority Ranking (by actual impact, not occupancy)

| Priority | Target | Stall Reason | Est. Savings | Effort |
|----------|--------|-------------|-------------|--------|
| 1 | H kernel barrier reduction | 69% idle on sync | 150-300 us | High |
| 2 | CUDA tcgen05 inverse (TMA + tcgen05) | 38% idle on DRAM | 200-400 us | High |
| 3 | CUDA tcgen05 O kernel (TMA + tcgen05) | 25% idle on memory | 150-300 us | High |
| 4 | FusedPrep bf16 inverse (v4 path) | 86% idle (small grid) | 50-100 us | Medium |
| 5 | CUDA H kernel with BV=16/32 for v8c | 72% idle on loads | 50-100 us | Medium |
| 6 | PTX-level ILP + vectorized loads | instruction stalls | 20-50 us | Medium |
| 7 | L2 prefetch / cache hints | DRAM latency | 10-30 us | Low |
