/*
* The chunked formulas are:
*
*   T = inv(I + strictLower(B * Gamma * (K @ K.T))) * B.T
*   W = (T * G.T) @ K          — WY correction keys
*   U = T @ V                  — WY corrected values
*
*   V' = U - W @ S             — delta-corrected values
*   O  = G * (Q @ S)           — inter-chunk output (query old state)
*      + ((Q @ K.T) * M') @ V' — intra-chunk output (attention within chunk)
*   S  = G[-1] * S + K.T @ (V' * (G[-1] / G))  — state update
*
* where:
*   G      = cumulative gate products within chunk, shape (C, 1)
*   G[-1]  = total gate decay across chunk (scalar)
*   Gamma  = G / G.T, pairwise gate ratios, shape (C, C)
*   M'     = causal_mask * Gamma, gate-weighted causal mask (C, C)
*   B      = per-token learning rates (beta), shape (C, 1)
*   S      = recurrent state carried between chunks, shape (D_k, D_v)
*
* Key insight: T, W, U depend only on data within the chunk (no state
* dependency), so Kernels 1-3 can run in parallel across ALL chunks.
* Only Kernel 4 (which uses S) must run sequentially per sequence.
*
* === Kernel Pipeline ===
*
* Kernel 1: PreprocessKernel (all chunks in parallel)
*   - Computes g_cumsum = cumulative_sum(log_gate) and beta = sigmoid(b)
*   - These are scalar per-token values, no tensor cores needed
*   - Grid: (total_chunks, num_heads)
*
* Kernel 2a: ComputeAKernel_TC (all chunks in parallel)
*   - Computes A = strictLower(B * Gamma * (K @ K.T))
*   - K @ K.T via tcgen05 tensor cores (the expensive part)
*   - Post-multiply by beta * exp(g_diff) with strict lower mask
*   - K tiles loaded via TMA (async DMA, no thread involvement)
*   - Output: A_mat (C × C) per chunk, strictly lower triangular
*
* Kernel 2b: SolveTrilKernel (all chunks in parallel)
*   - Computes inv(I + A) via column-parallel forward substitution
*   - I is never materialized: diagonal implicitly 1, identity column
*     generated on the fly as (tid == i) ? 1 : 0
*   - Unit lower triangular → always invertible, no pivoting needed
*   - Each of 64 threads solves one column independently
*   - Result overwrites A_mat in-place
*
* Kernel 3: ComputeWUKernel_TC (all chunks in parallel)
*   - Computes W = (T * G.T) @ K and U = T @ V
*   - Both share the same A_inv left operand (loaded once, fp32→bf16)
*   - The "* B.T" from T = inv(...) * B.T is absorbed into input scaling:
*     W input: K scaled by beta * exp(g)  (fuses B.T and G.T)
*     U input: V scaled by beta           (fuses B.T only)
*   - 4 tile jobs via blockIdx.z: {W_lo, W_hi, U_lo, U_hi} (64-col halves)
*
* Kernel 4: FusedRecurrenceOutput (sequential over chunks, parallel over seqs/heads)
*   - The only kernel with state dependency → sequential chunk loop
*   - Fuses V', O (both terms), and S update into one kernel because:
*     (a) All depend on state S which lives in shared memory (s_h)
*     (b) Splitting would require global memory round-trips for S and V'
*     (c) Enables TMA overlap between steps
*   - State s_h[BV=32][K=128] persists in shared memory across all chunks
*   - M' is never materialized: causal mask applied as (col <= row) check,
*     gate ratio computed on the fly as exp(g[row] - g[col])
*
*   Per-chunk steps with TMA overlap:
*
*     Step 1: W @ S^T via tcgen05
*       TMA loads W → tile_a (async)
*       OVERLAP: convert s_h fp32 → tile_b bf16 while TMA runs
*       MMA: tile_a @ tile_b → s_wh
*       Prefetch g_cumsum → s_gc (needed in steps 3-5, latency hiding)
*
*     Step 2: V' = U - W@S
*       TMA loads Q → tile_a (async, for step 3)
*       OVERLAP: compute s_vnew = u_in - s_wh while TMA runs
*
*     Step 3: inter-chunk output = G * (Q @ S)
*       Q already in tile_a (from step 2 TMA), S still in tile_b
*       MMA: Q @ S^T → s_wh
*       Write scale * exp(g) * (Q @ S) to output
*
*     Step 4: intra-chunk output = ((Q @ K.T) * M') @ V'
*       TMA loads K → tile_b (async, overwrites S — no longer needed)
*       MMA: Q @ K^T → s_wh
*       Apply causal mask + gate ratio element-wise (M' on the fly)
*       Manual dot product: masked_attn @ V' (small matmul, not worth tcgen05)
*       Accumulate into output (read-modify-write)
*
*     Step 5: state update S = G[-1] * S + K.T @ (V' * G[-1]/G)
*       Decay state: s_h *= exp(g_last)
*       Gate V': s_vnew *= exp(g_last - g[t])
*       Transpose K and gated V' into tile layout
*       MMA: K^T @ gated_V' → accumulate into s_h
*
*   After all chunks: write s_h → new_state in global memory
*
* === Hardware Details (Blackwell SM_100a) ===
*
* - tcgen05 tensor cores: BM must be 128 (hardware constraint), so 64-row
*   matrices are padded to 128 rows; only first 64 rows of output are read
* - TMA (Tensor Memory Access): async global→smem loads with no thread
*   involvement, coordinated via mbarriers. Used for plain bf16 loads
*   (K, Q, W). NOT used when fp32→bf16 conversion or scaling is needed
* - TMEM: tensor memory private to the MMA unit, allocated once per kernel,
*   reused across all MMA operations
* - Tile layout: non-standard byte addressing for tcgen05 compatibility
*   byte_off = tc * LBO + tr * SBO + wr * 16 + wc * 2
*
* === Dimensions ===
*
* K=128 (key/query dim), V=128 (value dim), BT=64 (chunk size), BV=32 (v tile)
* Hq=4 (query heads), Hk=4 (key heads), Hv=8 (value heads, GQA)

 */

#include "cuda_utils.h"

#include <cstdint>
#include <math.h>
#include <vector>

namespace {

constexpr int kK = 128, kV = 128;
constexpr int64_t kHq = 4, kHk = 4, kHv = 8;
constexpr int kBT = 64, kBV = 32;
constexpr int kNVT = kV / kBV;   // 4
constexpr int MMA_K = 16;
constexpr int SBO_CONST = 128;  // 8 rows * 16 bytes

// tcgen05 tile layout: convert (row, col) of bf16 matrix with H rows to byte offset
__device__ __forceinline__
int tile_byte_offset(int row, int col, int H) {
  int tr = row / 8;
  int tc = col / 8;
  int wr = row % 8;
  int wc = col % 8;
  int LBO = H * 16;
  return tc * LBO + tr * SBO_CONST + wr * 16 + wc * 2;
}

// Vectorized tile load: load 8 bf16 values from global row-major into one tile row
__device__ __forceinline__
void tile_store_vec8(__nv_bfloat16 *base, int row, int col_start, int H,
                     const __nv_bfloat16 *src) {
  int tr = row / 8, tc = col_start / 8, wr = row % 8;
  int byte_off = tc * (H * 16) + tr * SBO_CONST + wr * 16;
  *reinterpret_cast<int4 *>(reinterpret_cast<char *>(base) + byte_off) =
      *reinterpret_cast<const int4 *>(src);
}

// Store a bf16 value into tile-layout shared memory
__device__ __forceinline__
void tile_store(__nv_bfloat16 *base, int row, int col, int H, __nv_bfloat16 val) {
  int boff = tile_byte_offset(row, col, H);
  *reinterpret_cast<__nv_bfloat16 *>(reinterpret_cast<char *>(base) + boff) = val;
}

// Load global row-major [rows, cols] bf16 into tile-layout smem (kept for fallback)
__device__ __forceinline__
void load_global_to_tile(
    __nv_bfloat16 *smem, int H, int cols,
    const __nv_bfloat16 *global_ptr, int global_stride,
    int pad_rows, int tid) {
  const int total_chunks = H * (cols / 8);
  for (int i = tid; i < total_chunks; i += 128) {
    int row = i / (cols / 8);
    int col8 = (i % (cols / 8)) * 8;
    if (row < pad_rows) {
      tile_store_vec8(smem, row, col8, H, global_ptr + row * global_stride + col8);
    } else {
      int tr = row / 8, tc = col8 / 8, wr = row % 8;
      int byte_off = tc * (H * 16) + tr * SBO_CONST + wr * 16;
      *reinterpret_cast<int4 *>(reinterpret_cast<char *>(smem) + byte_off) = make_int4(0,0,0,0);
    }
  }
}

// Load fp32 smem data (e.g., h state) into tile-layout bf16 smem
__device__ __forceinline__
void load_fp32_to_tile(
    __nv_bfloat16 *smem, int H, int cols,
    const float *fp32_src, int fp32_stride,
    int actual_rows, int tid) {
  const int total_chunks = H * (cols / 8);
  __nv_bfloat16 tmp[8];
  for (int i = tid; i < total_chunks; i += 128) {
    int row = i / (cols / 8);
    int col8 = (i % (cols / 8)) * 8;
    if (row < actual_rows) {
      for (int j = 0; j < 8; j++)
        tmp[j] = __float2bfloat16(fp32_src[row * fp32_stride + col8 + j]);
    } else {
      for (int j = 0; j < 8; j++) tmp[j] = __float2bfloat16(0.0f);
    }
    int tr = row / 8, tc = col8 / 8, wr = row % 8;
    int byte_off = tc * (H * 16) + tr * SBO_CONST + wr * 16;
    *reinterpret_cast<int4 *>(reinterpret_cast<char *>(smem) + byte_off) =
        *reinterpret_cast<int4 *>(tmp);
  }
}

// Load fp32 GLOBAL memory into tile-layout bf16 smem
__device__ __forceinline__
void load_fp32_global_to_tile(
    __nv_bfloat16 *smem, int H, int cols,
    const float *global_ptr, int global_stride,
    int actual_rows, int tid) {
  const int total_chunks = H * (cols / 8);
  __nv_bfloat16 tmp[8];
  for (int i = tid; i < total_chunks; i += 128) {
    int row = i / (cols / 8);
    int col8 = (i % (cols / 8)) * 8;
    if (row < actual_rows) {
      for (int j = 0; j < 8; j++)
        tmp[j] = __float2bfloat16(global_ptr[row * global_stride + col8 + j]);
    } else {
      for (int j = 0; j < 8; j++) tmp[j] = __float2bfloat16(0.0f);
    }
    int tr = row / 8, tc = col8 / 8, wr = row % 8;
    int byte_off = tc * (H * 16) + tr * SBO_CONST + wr * 16;
    *reinterpret_cast<int4 *>(reinterpret_cast<char *>(smem) + byte_off) =
        *reinterpret_cast<int4 *>(tmp);
  }
}

__device__ __forceinline__ float softplus_s(float x) {
  return log1pf(expf(-fabsf(x))) + fmaxf(x, 0.0f);
}

// ═══════════════════════════════════════════════════════════════════
// TMA helper: issue TMA load + arrive on mbarrier
// ═══════════════════════════════════════════════════════════════════
__device__ __forceinline__
void tma_load_tile_3d(uint32_t smem_addr, const CUtensorMap *tmap,
                      int row_offset, int col_group_offset,
                      uint32_t mbar_addr, int cp_bytes,
                      int warp_id) {
  if (warp_id == 0 && elect_sync()) {
    tma_load_3d(smem_addr, tmap, 0, row_offset, col_group_offset, mbar_addr);
    asm volatile(
      "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
      :: "r"(mbar_addr), "r"(cp_bytes) : "memory");
  }
}

// Issue TWO TMA loads on the same mbarrier (for ComputeA which loads s_A and s_B)
__device__ __forceinline__
void tma_load_two_tiles_3d(
    uint32_t smem_addr_a, const CUtensorMap *tmap_a, int row_a, int col_group_a,
    uint32_t smem_addr_b, const CUtensorMap *tmap_b, int row_b, int col_group_b,
    uint32_t mbar_addr, int total_cp_bytes,
    int warp_id) {
  if (warp_id == 0 && elect_sync()) {
    tma_load_3d(smem_addr_a, tmap_a, 0, row_a, col_group_a, mbar_addr);
    tma_load_3d(smem_addr_b, tmap_b, 0, row_b, col_group_b, mbar_addr);
    asm volatile(
      "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
      :: "r"(mbar_addr), "r"(total_cp_bytes) : "memory");
  }
}


// ═══════════════════════════════════════════════════════════════════
// Kernel 1: Preprocess — g_cumsum + beta (no tensor cores)
// Grid: (total_chunks, kHv), Block: kBT
// ═══════════════════════════════════════════════════════════════════
__global__ void PreprocessKernel(
    const __nv_bfloat16 *__restrict__ a_in,
    const __nv_bfloat16 *__restrict__ b_in,
    const float *__restrict__ A_log,
    const float *__restrict__ dt_bias,
    const int64_t *__restrict__ cu_seqlens,
    const int32_t *__restrict__ chunk_indices,
    float *__restrict__ g_cumsum,
    float *__restrict__ beta_out,
    int64_t total_chunks) {

  const int chunk_id = blockIdx.x, hv = blockIdx.y;
  if (chunk_id >= total_chunks) return;

  const int seq_idx = chunk_indices[chunk_id * 2];
  const int local_chunk = chunk_indices[chunk_id * 2 + 1];
  const int64_t s0 = cu_seqlens[seq_idx];
  const int64_t s1 = cu_seqlens[seq_idx + 1];
  const int64_t cstart = s0 + (int64_t)local_chunk * kBT;
  const int clen = min(kBT, (int)(s1 - cstart));
  const int tid = threadIdx.x;
  const bool valid = tid < clen;
  const int64_t gt = cstart + tid;

  float g_val = 0.0f, beta_val = 0.0f;
  if (valid) {
    float x = __bfloat162float(a_in[gt * kHv + hv]) + dt_bias[hv];
    g_val = -expf(A_log[hv]) * softplus_s(x);
    beta_val = 1.0f / (1.0f + expf(-__bfloat162float(b_in[gt * kHv + hv])));
  }

  __shared__ float s_g[kBT];
  s_g[tid] = g_val;
  __syncthreads();
  if (tid == 0) { for (int i = 1; i < clen; i++) s_g[i] += s_g[i-1]; }
  __syncthreads();

  if (valid) {
    g_cumsum[gt * kHv + hv] = s_g[tid];
    beta_out[gt * kHv + hv] = beta_val;
  }
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 2a: ComputeAKernel_TC — tcgen05 k@k^T + preprocess + gating → A_mat
// Grid: (total_chunks, kHk=4), Block: 128
// Computes k@k^T ONCE per k-head, then applies gating for kHv/kHk=2 v-heads
// Also fuses PreprocessKernel (g_cumsum + beta) into this kernel
// ═══════════════════════════════════════════════════════════════════
__global__ void __launch_bounds__(128, 1)
ComputeAKernel_TC(
    const __grid_constant__ CUtensorMap k_tmap_128,
    const __grid_constant__ CUtensorMap k_tmap_64,
    const __nv_bfloat16 *__restrict__ a_in,
    const __nv_bfloat16 *__restrict__ b_in,
    const float *__restrict__ A_log,
    const float *__restrict__ dt_bias,
    const int64_t *__restrict__ cu_seqlens,
    const int32_t *__restrict__ chunk_indices,
    float *__restrict__ g_cumsum,
    float *__restrict__ beta_out,
    float *__restrict__ A_mat,
    int64_t total_chunks) {

  const int chunk_id = blockIdx.x, k_head = blockIdx.y;
  if (chunk_id >= total_chunks) return;

  const int seq_idx = chunk_indices[chunk_id * 2];
  const int local_chunk = chunk_indices[chunk_id * 2 + 1];
  const int64_t s0 = cu_seqlens[seq_idx], s1 = cu_seqlens[seq_idx + 1];
  const int64_t cstart = s0 + (int64_t)local_chunk * kBT;
  const int clen = min(kBT, (int)(s1 - cstart));

  const int tid = threadIdx.x;
  const int warp_id = tid / 32;

  constexpr int BM = 128, BN = 64, BK = 128;

  extern __shared__ __align__(1024) char smem[];
  __nv_bfloat16 *s_A = reinterpret_cast<__nv_bfloat16 *>(smem);
  __nv_bfloat16 *s_B = reinterpret_cast<__nv_bfloat16 *>(smem + BM * BK * 2);
  char *after_tiles = smem + BM * BK * 2 + BN * BK * 2;
  uint64_t *mbars = reinterpret_cast<uint64_t *>(after_tiles);
  int *tmem_buf = reinterpret_cast<int *>(mbars + 2);
  float *s_result = reinterpret_cast<float *>(tmem_buf + 1);
  float *s_g = s_result + kBT * kBT;
  float *s_beta = s_g + kBT;

  const uint32_t A_smem_base = cvt_smem_ptr(s_A);
  const uint32_t B_smem_base = cvt_smem_ptr(s_B);
  const uint32_t mbar_tma = cvt_smem_ptr(&mbars[0]);
  const uint32_t mbar_mma = cvt_smem_ptr(&mbars[1]);

  if (warp_id == 0 && elect_sync()) {
    mbarrier_init(mbar_tma, 1);
    mbarrier_init(mbar_mma, 1);
    asm volatile("fence.mbarrier_init.release.cluster;");
  } else if (warp_id == 1) {
    tcgen05_alloc(cvt_smem_ptr(tmem_buf), BN);
  }
  __syncthreads();
  const int taddr = tmem_buf[0];
  constexpr uint32_t idesc = make_tcgen05_idesc<BM, BN>();

  // TMA load k — same k_head for both v-heads
  int col_group = (int)(k_head * kK / 8);
  constexpr int cp_A = BM * BK * (int)sizeof(__nv_bfloat16);
  constexpr int cp_B = BN * BK * (int)sizeof(__nv_bfloat16);
  tma_load_two_tiles_3d(
      A_smem_base, &k_tmap_128, (int)cstart, col_group,
      B_smem_base, &k_tmap_64,  (int)cstart, col_group,
      mbar_tma, cp_A + cp_B, warp_id);
  mbarrier_wait(mbar_tma, 0);
  __syncthreads();

  // MMA: k@k^T — done ONCE, shared across 2 v-heads
  tcgen05_fence();
  if (warp_id == 0 && elect_sync()) {
    for (int ki = 0; ki < BK / MMA_K; ki++) {
      uint32_t a_base = A_smem_base + ki * 2 * (BM * 16);
      uint32_t b_base = B_smem_base + ki * 2 * (BN * 16);
      tcgen05_mma(taddr,
          make_tcgen05_desc_noswizzle(a_base, BM, SBO_CONST),
          make_tcgen05_desc_noswizzle(b_base, BN, SBO_CONST),
          idesc, ki);
    }
    tcgen05_commit(mbar_mma);
  }
  mbarrier_wait(mbar_mma, 0);
  tcgen05_fence();

  // Read TMEM → s_result [64,64]
  for (int n = 0; n < BN / 8; n++) {
    float tmp[8];
    tcgen05_ld_32x8(tmp, taddr, warp_id * 32, n * 8);
    int my_row = warp_id * 32 + (tid % 32);
    if (my_row < kBT)
      for (int c = 0; c < 8; c++)
        s_result[my_row * kBT + n * 8 + c] = tmp[c];
  }
  __syncthreads();
  if (warp_id == 0) tcgen05_dealloc(taddr, BN);

  // Loop over kHv/kHk = 2 v-heads that share this k-head
  constexpr int HEADS_PER_K = kHv / kHk;
  for (int hi = 0; hi < HEADS_PER_K; hi++) {
    const int hv = k_head * HEADS_PER_K + hi;

    // Fused preprocess: compute g_cumsum and beta for this v-head
    if (tid < kBT) {
      float g_val = 0.0f, beta_val = 0.0f;
      if (tid < clen) {
        const int64_t gt = cstart + tid;
        float x = __bfloat162float(a_in[gt * kHv + hv]) + dt_bias[hv];
        g_val = -expf(A_log[hv]) * softplus_s(x);
        beta_val = 1.0f / (1.0f + expf(-__bfloat162float(b_in[gt * kHv + hv])));
      }
      s_g[tid] = g_val;
      s_beta[tid] = beta_val;
    }
    __syncthreads();
    if (tid == 0) { for (int i = 1; i < clen; i++) s_g[i] += s_g[i-1]; }
    __syncthreads();

    // Store g_cumsum and beta
    if (tid < clen) {
      const int64_t gt = cstart + tid;
      g_cumsum[gt * kHv + hv] = s_g[tid];
      beta_out[gt * kHv + hv] = s_beta[tid];
    }

    // Apply gating + mask → A_mat (only write clen rows)
    for (int i = tid; i < clen * kBT; i += 128) {
      int row = i / kBT, col = i % kBT;
      float val = 0.0f;
      if (col < row)
        val = s_beta[row] * s_result[row * kBT + col] * expf(s_g[row] - s_g[col]);
      A_mat[(cstart + row) * kHv * kBT + hv * kBT + col] = val;
    }
    __syncthreads();
  }
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 2b: SolveTrilKernel — forward substitution (I+A)^{-1} in-place
// Grid: (total_chunks, kHv), Block: kBT=64
// ═══════════════════════════════════════════════════════════════════
__global__ void SolveTrilKernel(
    const int64_t *__restrict__ cu_seqlens,
    const int32_t *__restrict__ chunk_indices,
    float *__restrict__ A_mat,
    int64_t total_chunks) {

  const int chunk_id = blockIdx.x, hv = blockIdx.y;
  if (chunk_id >= total_chunks) return;

  const int seq_idx = chunk_indices[chunk_id * 2];
  const int local_chunk = chunk_indices[chunk_id * 2 + 1];
  const int64_t s0 = cu_seqlens[seq_idx], s1 = cu_seqlens[seq_idx + 1];
  const int64_t cstart = s0 + (int64_t)local_chunk * kBT;
  const int clen = min(kBT, (int)(s1 - cstart));

  const int tid = threadIdx.x;  // 0..63

  __shared__ float s_A[kBT][kBT];
  __shared__ float s_Ai[kBT][kBT];

  // Load A from global
  for (int row = 0; row < kBT; row++)
    s_A[row][tid] = A_mat[(cstart + row) * kHv * kBT + hv * kBT + tid];
  __syncthreads();

  // Initialize s_Ai to zero
  for (int row = 0; row < kBT; row++)
    s_Ai[row][tid] = 0.0f;
  __syncthreads();

  // Column-parallel forward substitution: each thread handles one column
  for (int i = 0; i < clen; i++) {
    float val = (tid == i) ? 1.0f : 0.0f;
    for (int j = 0; j < i; j++)
      val -= s_A[i][j] * s_Ai[j][tid];
    s_Ai[i][tid] = val;
    __syncthreads();
  }

  // Fill identity for rows >= clen
  for (int i = clen; i < kBT; i++)
    s_Ai[i][tid] = (tid == i) ? 1.0f : 0.0f;
  __syncthreads();

  // Write A_inv back to A_mat
  for (int row = 0; row < clen; row++)
    A_mat[(cstart + row) * kHv * kBT + hv * kBT + tid] = s_Ai[row][tid];
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 3: ComputeWU — tcgen05 for A_inv @ input
// Grid: (total_chunks, kHv, 4), Block: 128
// NO TMA: input tile requires transpose+scale, A_inv is fp32→bf16
// ═══════════════════════════════════════════════════════════════════
__global__ void __launch_bounds__(128, 1)
ComputeWUKernel_TC(
    const __nv_bfloat16 *__restrict__ k_in,
    const __nv_bfloat16 *__restrict__ v_in,
    const float *__restrict__ g_cumsum,
    const float *__restrict__ beta_f,
    const float *__restrict__ A_inv,
    const int64_t *__restrict__ cu_seqlens,
    const int32_t *__restrict__ chunk_indices,
    __nv_bfloat16 *__restrict__ w_out,
    __nv_bfloat16 *__restrict__ u_out,
    int64_t total_chunks) {

  const int chunk_id = blockIdx.x, hv = blockIdx.y;
  const int tile_id = blockIdx.z;
  if (chunk_id >= total_chunks) return;

  const int seq_idx = chunk_indices[chunk_id * 2];
  const int local_chunk = chunk_indices[chunk_id * 2 + 1];
  const int64_t s0 = cu_seqlens[seq_idx], s1 = cu_seqlens[seq_idx + 1];
  const int64_t cstart = s0 + (int64_t)local_chunk * kBT;
  const int clen = min(kBT, (int)(s1 - cstart));
  const int64_t k_head = hv / (kHv / kHk);

  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const bool is_w = (tile_id < 2);
  const int out_col = (tile_id % 2) * 64;

  constexpr int BM = 128, BN = 64, BK_inner = 64;

  extern __shared__ __align__(1024) char smem[];
  __nv_bfloat16 *s_Ainv = reinterpret_cast<__nv_bfloat16 *>(smem);
  __nv_bfloat16 *s_input = reinterpret_cast<__nv_bfloat16 *>(
      smem + BM * BK_inner * 2);
  char *after_tiles = smem + BM * BK_inner * 2 + BN * BK_inner * 2;
  uint64_t *mbars = reinterpret_cast<uint64_t *>(after_tiles);
  int *tmem_buf = reinterpret_cast<int *>(mbars + 1);
  float *s_bg = reinterpret_cast<float *>(tmem_buf + 1);

  const uint32_t Ainv_smem = cvt_smem_ptr(s_Ainv);
  const uint32_t input_smem = cvt_smem_ptr(s_input);
  const uint32_t mbar_addr = cvt_smem_ptr(mbars);

  if (warp_id == 0 && elect_sync()) {
    mbarrier_init(mbar_addr, 1);
    asm volatile("fence.mbarrier_init.release.cluster;");
  } else if (warp_id == 1) {
    tcgen05_alloc(cvt_smem_ptr(tmem_buf), BN);
  }
  __syncthreads();
  const int taddr = tmem_buf[0];
  constexpr uint32_t idesc = make_tcgen05_idesc<BM, BN>();

  if (tid < kBT) {
    float b = (tid < clen) ? beta_f[(cstart + tid) * kHv + hv] : 0.0f;
    if (is_w) {
      float g = (tid < clen) ? g_cumsum[(cstart + tid) * kHv + hv] : 0.0f;
      s_bg[tid] = b * expf(g);
    } else {
      s_bg[tid] = b;
    }
  }
  __syncthreads();

  // A_inv is fp32→bf16 (can't use TMA)
  load_fp32_global_to_tile(s_Ainv, BM, BK_inner,
      A_inv + cstart * kHv * kBT + hv * kBT, kHv * kBT, clen, tid);

  // Input tile: transposed + scaled (can't use TMA)
  for (int i = tid; i < BN * BK_inner; i += 128) {
    int j = i / BK_inner, k = i % BK_inner;
    __nv_bfloat16 val = __float2bfloat16(0.0f);
    if (k < clen) {
      float raw;
      if (is_w)
        raw = __bfloat162float(
            k_in[(cstart + k) * kHk * kK + k_head * kK + out_col + j]) * s_bg[k];
      else
        raw = __bfloat162float(
            v_in[(cstart + k) * kHv * kV + hv * kV + out_col + j]) * s_bg[k];
      val = __float2bfloat16(raw);
    }
    tile_store(s_input, j, k, BN, val);
  }
  __syncthreads();

  int phase = 0;
  for (int ki = 0; ki < BK_inner / MMA_K; ki++) {
    tcgen05_fence();
    if (warp_id == 0 && elect_sync()) {
      uint32_t a_base = Ainv_smem + ki * 2 * (BM * 16);
      uint32_t b_base = input_smem + ki * 2 * (BN * 16);
      uint64_t a_desc = make_tcgen05_desc_noswizzle(a_base, BM, SBO_CONST);
      uint64_t b_desc = make_tcgen05_desc_noswizzle(b_base, BN, SBO_CONST);
      tcgen05_mma(taddr, a_desc, b_desc, idesc, ki);
      tcgen05_commit(mbar_addr);
    }
    mbarrier_wait(mbar_addr, phase);
    phase ^= 1;
  }
  tcgen05_fence();

  for (int n = 0; n < BN / 8; n++) {
    float tmp[8];
    tcgen05_ld_32x8(tmp, taddr, warp_id * 32, n * 8);
    int my_row = warp_id * 32 + (tid % 32);
    if (my_row < clen) {
      if (is_w) {
        for (int c = 0; c < 8; c++)
          w_out[(cstart + my_row) * kHv * kK + hv * kK + out_col + n * 8 + c] =
              __float2bfloat16_rn(tmp[c]);
      } else {
        for (int c = 0; c < 8; c++)
          u_out[(cstart + my_row) * kHv * kV + hv * kV + out_col + n * 8 + c] =
              __float2bfloat16_rn(tmp[c]);
      }
    }
  }

  __syncthreads();
  if (warp_id == 0) tcgen05_dealloc(taddr, BN);
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 4: FusedRecurrenceOutput — InterChunkFwdH + ChunkFwdO fused
// Grid: (kNVT, num_seqs * kHv), Block: 128
// TMA for: w→s_tile_a, q→s_tile_a, k→s_tile_b
// Manual for: h→s_tile_b (fp32→bf16 from smem)
//
// Shared memory layout (~85KB):
//   s_h[BV][K+1] fp32 padded = 16.5KB (persistent state)
//   s_tile_a[128*128] bf16 = 32KB (A operand, reused for w/q tiles)
//   s_tile_b[64*128] bf16 = 16KB (B operand, reused for h/k tiles)
//   mbar_tma(8) + mbar_mma(8) + tmem_buf(4) = 20 bytes
//   s_gc[64] = 256B
//   s_wh[64*64] fp32 = 16KB (tcgen05 result, reusable across MMA ops)
//   s_vnew[64*32] bf16 = 4KB
// ═══════════════════════════════════════════════════════════════════
__global__ void __launch_bounds__(128, 1)
FusedRecurrenceOutput(
    const __grid_constant__ CUtensorMap w_tmap_128,
    const __grid_constant__ CUtensorMap q_tmap_128,
    const __grid_constant__ CUtensorMap k_tmap_64,
    const __nv_bfloat16 *__restrict__ k_in,
    const __nv_bfloat16 *__restrict__ u_in,
    const float *__restrict__ g_cumsum,
    const float *__restrict__ state0,
    const int64_t *__restrict__ cu_seqlens,
    const int64_t *__restrict__ chunk_offsets,
    float scale,
    __nv_bfloat16 *__restrict__ output,
    float *__restrict__ new_state,
    int64_t num_seqs) {

  const int v_tile = blockIdx.x;
  const int bh = blockIdx.y;
  const int seq_idx = bh / kHv, hv = bh % kHv;
  if (seq_idx >= num_seqs) return;

  const int64_t q_head = hv / (kHv / kHq);
  const int64_t k_head = hv / (kHv / kHk);
  const int64_t s0 = cu_seqlens[seq_idx], s1 = cu_seqlens[seq_idx + 1];
  const int NT = (int)((s1 - s0 + kBT - 1) / kBT);

  const int warp_id = threadIdx.x / 32;
  const int tid = threadIdx.x;
  const int v_start = v_tile * kBV;

  constexpr int TC_BM = 128, TC_BN = 64, TC_BK = kK;  // 128, 64, 128

  // ── Shared memory layout ──
  constexpr int s_h_bytes = ((kBV * (kK + 1) * (int)sizeof(float) + 1023) / 1024) * 1024;

  extern __shared__ __align__(1024) char smem[];
  float (*s_h)[kK + 1] = reinterpret_cast<float (*)[kK + 1]>(smem);

  char *dyn = smem + s_h_bytes;
  __nv_bfloat16 *s_tile_a = reinterpret_cast<__nv_bfloat16 *>(dyn);                      // [128,128] = 32KB
  __nv_bfloat16 *s_tile_b = reinterpret_cast<__nv_bfloat16 *>(dyn + TC_BM * TC_BK * 2);  // [64,128] = 16KB
  char *after_tiles = dyn + TC_BM * TC_BK * 2 + TC_BN * TC_BK * 2;
  uint64_t *mbars = reinterpret_cast<uint64_t *>(after_tiles);  // 2 mbarriers: [0]=TMA, [1]=MMA
  int *tmem_buf = reinterpret_cast<int *>(mbars + 2);
  float *s_gc = reinterpret_cast<float *>(tmem_buf + 1);                          // [64]
  float *s_wh = s_gc + kBT;                                                       // [64*64] = 16KB
  __nv_bfloat16 *s_vnew = reinterpret_cast<__nv_bfloat16 *>(s_wh + kBT * TC_BN); // [64*32] = 4KB

  const uint32_t tileA_smem = cvt_smem_ptr(s_tile_a);
  const uint32_t tileB_smem = cvt_smem_ptr(s_tile_b);
  const uint32_t mbar_tma = cvt_smem_ptr(&mbars[0]);
  const uint32_t mbar_mma = cvt_smem_ptr(&mbars[1]);

  // Initialize state h
  if (state0) {
    const float *h0 = state0 + ((int64_t)seq_idx * kHv + hv) * kV * kK;
    for (int bv = 0; bv < kBV; bv++)
      for (int k = tid; k < kK; k += 128)
        s_h[bv][k] = h0[(v_start + bv) * kK + k];
  } else {
    for (int bv = 0; bv < kBV; bv++)
      for (int k = tid; k < kK; k += 128)
        s_h[bv][k] = 0.0f;
  }
  __syncthreads();

  // Init mbarriers + alloc TMEM once (reused for all MMA operations across all chunks)
  if (warp_id == 0 && elect_sync()) {
    mbarrier_init(mbar_tma, 1);
    mbarrier_init(mbar_mma, 1);
    asm volatile("fence.mbarrier_init.release.cluster;");
  } else if (warp_id == 1) {
    tcgen05_alloc(cvt_smem_ptr(tmem_buf), TC_BN);
  }
  __syncthreads();
  const int taddr = tmem_buf[0];
  constexpr uint32_t idesc = make_tcgen05_idesc<TC_BM, TC_BN>();

  // TMA mbarrier phase tracking (persists across chunks)
  int tma_phase = 0;

  // Precompute TMA col_group offsets for this block's head assignment
  const int w_col_group = (int)(hv * kK / 8);      // w_in: [T, Hv*K], head=hv
  const int q_col_group = (int)(q_head * kK / 8);   // q_in: [T, Hq*K], head=q_head
  const int k_col_group = (int)(k_head * kK / 8);   // k_in: [T, Hk*K], head=k_head

  constexpr int cp_tileA = TC_BM * TC_BK * (int)sizeof(__nv_bfloat16);  // 32KB
  constexpr int cp_tileB = TC_BN * TC_BK * (int)sizeof(__nv_bfloat16);  // 16KB

  for (int ct = 0; ct < NT; ct++) {
    const int64_t cstart = s0 + (int64_t)ct * kBT;
    const int clen = min(kBT, (int)(s1 - cstart));

    // ════════════════════════════════════════════════
    // Step 1: Compute w@h^T via tcgen05
    // A = w [64,128] padded to [128,128], B = h [32,128] padded to [64,128]
    // ════════════════════════════════════════════════

    // TMA load w into tile A (async) + h tile load in parallel
    tma_load_tile_3d(tileA_smem, &w_tmap_128, (int)cstart, w_col_group,
                     mbar_tma, cp_tileA, warp_id);
    // Load h into tile B while TMA w transfers in background
    load_fp32_to_tile(s_tile_b, TC_BN, TC_BK, &s_h[0][0], kK + 1, kBV, tid);
    // Wait for TMA w
    mbarrier_wait(mbar_tma, tma_phase);
    tma_phase ^= 1;
    __syncthreads();

    // MMA: issue ALL 8 K-tiles, single commit at end
    {
      tcgen05_fence();
      if (warp_id == 0 && elect_sync()) {
        for (int ki = 0; ki < TC_BK / MMA_K; ki++) {
          uint32_t a_base = tileA_smem + ki * 2 * (TC_BM * 16);
          uint32_t b_base = tileB_smem + ki * 2 * (TC_BN * 16);
          tcgen05_mma(taddr,
              make_tcgen05_desc_noswizzle(a_base, TC_BM, SBO_CONST),
              make_tcgen05_desc_noswizzle(b_base, TC_BN, SBO_CONST),
              idesc, ki);
        }
        tcgen05_commit(mbar_mma);
      }
      mbarrier_wait(mbar_mma, 0);
      // Reset mbarrier for next use
      if (warp_id == 0 && elect_sync())
        mbarrier_init(mbar_mma, 1);
      __syncthreads();
      tcgen05_fence();
    }

    // Read w@h^T TMEM → s_wh; overlap with g_cumsum load
    for (int n = 0; n < TC_BN / 8; n++) {
      float tmp[8];
      tcgen05_ld_32x8(tmp, taddr, warp_id * 32, n * 8);
      int my_row = warp_id * 32 + (tid % 32);
      if (my_row < kBT)
        for (int c = 0; c < 8; c++)
          s_wh[my_row * TC_BN + n * 8 + c] = tmp[c];
    }
    // Load g_cumsum early (overlapped with TMEM read, needed later)
    if (tid < kBT)
      s_gc[tid] = (tid < clen) ? g_cumsum[(cstart + tid) * kHv + hv] : 0.0f;
    __syncthreads();

    // ════════════════════════════════════════════════
    // Step 2: v_new = u - w@h^T
    // OVERLAP: Issue TMA q → tile_a (step 3 prep, tile_a no longer needed)
    // ════════════════════════════════════════════════
    tma_load_tile_3d(tileA_smem, &q_tmap_128, (int)cstart, q_col_group,
                     mbar_tma, cp_tileA, warp_id);
    // v_new computation runs while TMA q loads in background
    for (int i = tid; i < kBT * kBV; i += 128) {
      int t = i / kBV, bv = i % kBV;
      float val = 0.0f;
      if (t < clen) {
        float uv = __bfloat162float(
            u_in[(cstart + t) * kHv * kV + hv * kV + v_start + bv]);
        val = uv - s_wh[t * TC_BN + bv];
      }
      s_vnew[i] = __float2bfloat16(val);
    }
    // Wait for TMA q to complete
    mbarrier_wait(mbar_tma, tma_phase);
    tma_phase ^= 1;
    __syncthreads();

    // ════════════════════════════════════════════════
    // Step 3: q@h^T via tcgen05 (h STILL in tile_b, q just loaded into tile_a)
    // ════════════════════════════════════════════════
    {
      tcgen05_fence();
      if (warp_id == 0 && elect_sync()) {
        for (int ki = 0; ki < TC_BK / MMA_K; ki++) {
          uint32_t a_base = tileA_smem + ki * 2 * (TC_BM * 16);
          uint32_t b_base = tileB_smem + ki * 2 * (TC_BN * 16);
          tcgen05_mma(taddr, make_tcgen05_desc_noswizzle(a_base, TC_BM, SBO_CONST),
                      make_tcgen05_desc_noswizzle(b_base, TC_BN, SBO_CONST), idesc, ki);
        }
        tcgen05_commit(mbar_mma);
      }
      mbarrier_wait(mbar_mma, 0);
      if (warp_id == 0 && elect_sync()) mbarrier_init(mbar_mma, 1);
      __syncthreads();
      tcgen05_fence();
    }

    // Read q@h^T TMEM → s_wh (first kBV cols = q@h^T result for output)
    for (int n = 0; n < TC_BN / 8; n++) {
      float tmp[8];
      tcgen05_ld_32x8(tmp, taddr, warp_id * 32, n * 8);
      int my_row = warp_id * 32 + (tid % 32);
      if (my_row < kBT)
        for (int c = 0; c < 8; c++)
          s_wh[my_row * TC_BN + n * 8 + c] = tmp[c];
    }
    __syncthreads();

    // Write scale * q@h^T * exp(g) to output (first pass)
    for (int i = tid; i < clen * kBV; i += 128) {
      int t = i / kBV, bv = i % kBV;
      float oval = scale * s_wh[t * TC_BN + bv] * expf(s_gc[t]);
      output[(cstart + t) * kHv * kV + hv * kV + v_start + bv] =
          __float2bfloat16_rn(oval);
    }

    // ════════════════════════════════════════════════
    // Step 4: q@k^T via tcgen05 (q still in tile_a)
    // OVERLAP: TMA k issued while output write runs
    // ════════════════════════════════════════════════
    tma_load_tile_3d(tileB_smem, &k_tmap_64, (int)cstart, k_col_group,
                     mbar_tma, cp_tileB, warp_id);
    mbarrier_wait(mbar_tma, tma_phase);
    tma_phase ^= 1;
    __syncthreads();

    // MMA q@k^T — batch all K-tiles, single commit
    {
      tcgen05_fence();
      if (warp_id == 0 && elect_sync()) {
        for (int ki = 0; ki < TC_BK / MMA_K; ki++) {
          uint32_t a_base = tileA_smem + ki * 2 * (TC_BM * 16);
          uint32_t b_base = tileB_smem + ki * 2 * (TC_BN * 16);
          tcgen05_mma(taddr, make_tcgen05_desc_noswizzle(a_base, TC_BM, SBO_CONST),
                      make_tcgen05_desc_noswizzle(b_base, TC_BN, SBO_CONST), idesc, ki);
        }
        tcgen05_commit(mbar_mma);
      }
      mbarrier_wait(mbar_mma, 0);
      if (warp_id == 0 && elect_sync()) mbarrier_init(mbar_mma, 1);
      __syncthreads();
      tcgen05_fence();
    }

    // Read q@k^T → s_wh [64,64] cols [0..63] (overwrites cols [0..31] but qh_out in [32..63])
    for (int n = 0; n < TC_BN / 8; n++) {
      float tmp[8];
      tcgen05_ld_32x8(tmp, taddr, warp_id * 32, n * 8);
      int my_row = warp_id * 32 + (tid % 32);
      if (my_row < kBT)
        for (int c = 0; c < 8; c++)
          s_wh[my_row * TC_BN + n * 8 + c] = tmp[c];
    }
    __syncthreads();

    // Apply gating + causal mask to q@k^T (cols [0..63])
    for (int i = tid; i < kBT * kBT; i += 128) {
      int row = i / kBT, col = i % kBT;
      float val = 0.0f;
      if (row < clen && col < clen && col <= row)
        val = s_wh[row * TC_BN + col] * expf(s_gc[row] - s_gc[col]);
      s_wh[row * TC_BN + col] = val;
    }
    __syncthreads();

    // Add scale * (attn @ v_new) to output (second pass, read-modify-write)
    for (int i = tid; i < clen * kBV; i += 128) {
      int t = i / kBV, bv = i % kBV;
      float dot = 0.0f;
      for (int j = 0; j <= t; j++)
        dot += s_wh[t * TC_BN + j] * __bfloat162float(s_vnew[j * kBV + bv]);
      float existing = __bfloat162float(
          output[(cstart + t) * kHv * kV + hv * kV + v_start + bv]);
      output[(cstart + t) * kHv * kV + hv * kV + v_start + bv] =
          __float2bfloat16_rn(existing + scale * dot);
    }
    __syncthreads();

    // ════════════════════════════════════════════════
    // Step 5: Update state h
    // h *= exp(g_last)
    // v_new *= exp(g_last - g)  (gating for state update)
    // h += k^T @ v_new_gated
    // ════════════════════════════════════════════════

    float g_last = s_gc[clen - 1];
    float g_exp = expf(g_last);

    // Combined: gate v_new + scale h (saves one __syncthreads)
    for (int i = tid; i < kBT * kBV; i += 128) {
      int t = i / kBV;
      if (t < clen) {
        float sc = expf(g_last - s_gc[t]);
        s_vnew[i] = __float2bfloat16(__bfloat162float(s_vnew[i]) * sc);
      }
    }
    for (int bv = 0; bv < kBV; bv++)
      for (int k = tid; k < kK; k += 128)
        s_h[bv][k] *= g_exp;
    __syncthreads();

    // h += k^T @ v_new_gated via tcgen05
    {
      constexpr int BK_kv = kBT;
      // Load k^T into tile_a [128,64] — transpose from tile_b (k still from TMA step 4)
      const int total_a = TC_BM * (BK_kv / 8);
      for (int i = tid; i < total_a; i += 128) {
        int col_k = i / (BK_kv / 8);
        int t_grp = i % (BK_kv / 8);
        int t_base = t_grp * 8;
        __nv_bfloat16 tmp[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
          int t = t_base + j;
          if (t < clen) {
            int boff = (col_k / 8) * (TC_BN * 16) + (t / 8) * SBO_CONST + (t % 8) * 16 + (col_k % 8) * 2;
            tmp[j] = *reinterpret_cast<__nv_bfloat16 *>(reinterpret_cast<char *>(s_tile_b) + boff);
          } else {
            tmp[j] = __float2bfloat16(0.0f);
          }
        }
        int tr = col_k / 8, tc = t_grp, wr = col_k % 8;
        int off = tc * (TC_BM * 16) + tr * SBO_CONST + wr * 16;
        *reinterpret_cast<int4 *>(reinterpret_cast<char *>(s_tile_a) + off) =
            *reinterpret_cast<int4 *>(tmp);
      }
      // Transpose gated vnew [BT,BV]→[BV,BT] padded to [64,64] in tile_b
      const int total_b = TC_BN * (BK_kv / 8);
      for (int i = tid; i < total_b; i += 128) {
        int row = i / (BK_kv / 8), col8 = (i % (BK_kv / 8)) * 8;
        __nv_bfloat16 tmp[8];
        if (row < kBV) {
          #pragma unroll
          for (int j = 0; j < 8; j++) {
            int t = col8 + j;
            tmp[j] = (t < clen) ? s_vnew[t * kBV + row] : __float2bfloat16(0.0f);
          }
        } else {
          #pragma unroll
          for (int j = 0; j < 8; j++) tmp[j] = __float2bfloat16(0.0f);
        }
        int tr = row / 8, tc = col8 / 8, wr = row % 8;
        int off = tc * (TC_BN * 16) + tr * SBO_CONST + wr * 16;
        *reinterpret_cast<int4 *>(reinterpret_cast<char *>(s_tile_b) + off) =
            *reinterpret_cast<int4 *>(tmp);
      }
      __syncthreads();

      // MMA: k^T @ vnew_gated → TMEM [128,64], 4 K-tiles
      {
        tcgen05_fence();
        if (warp_id == 0 && elect_sync()) {
          for (int ki = 0; ki < BK_kv / MMA_K; ki++) {
            uint32_t a_base = tileA_smem + ki * 2 * (TC_BM * 16);
            uint32_t b_base = tileB_smem + ki * 2 * (TC_BN * 16);
            tcgen05_mma(taddr,
                make_tcgen05_desc_noswizzle(a_base, TC_BM, SBO_CONST),
                make_tcgen05_desc_noswizzle(b_base, TC_BN, SBO_CONST),
                idesc, ki);
          }
          tcgen05_commit(mbar_mma);
        }
        mbarrier_wait(mbar_mma, 0);
        if (warp_id == 0 && elect_sync())
          mbarrier_init(mbar_mma, 1);
        __syncthreads();
        tcgen05_fence();
      }

      // Read TMEM → add to s_h[bv][col_k]
      for (int n = 0; n < TC_BN / 8; n++) {
        float tmp[8];
        tcgen05_ld_32x8(tmp, taddr, warp_id * 32, n * 8);
        int col_k = warp_id * 32 + (tid % 32);
        for (int c = 0; c < 8; c++) {
          int bv = n * 8 + c;
          if (bv < kBV)
            s_h[bv][col_k] += tmp[c];
        }
      }
      __syncthreads();
    }
  } // end chunk loop

  // Dealloc TMEM
  __syncthreads();
  if (warp_id == 0) tcgen05_dealloc(taddr, TC_BN);

  // Store final state
  float *ns = new_state + ((int64_t)seq_idx * kHv + hv) * kV * kK;
  for (int bv = 0; bv < kBV; bv++)
    for (int k = tid; k < kK; k += 128)
      ns[(v_start + bv) * kK + k] = s_h[bv][k];
}

// ═══════════════════════════════════════════════════════════════════
// GPU metadata kernel (UNCHANGED)
// ═══════════════════════════════════════════════════════════════════
__global__ void PrepMetaKernel(
    const int64_t *__restrict__ cu_seqlens,
    int32_t *__restrict__ chunk_indices,
    int64_t *__restrict__ chunk_offsets,
    int32_t *__restrict__ total_chunks_out,
    int64_t num_seqs) {
  int32_t total = 0;
  chunk_offsets[0] = 0;
  for (int64_t i = 0; i < num_seqs; i++) {
    int64_t slen = cu_seqlens[i + 1] - cu_seqlens[i];
    int32_t nc = (int32_t)((slen + kBT - 1) / kBT);
    for (int32_t c = 0; c < nc; c++) {
      chunk_indices[(total + c) * 2] = (int32_t)i;
      chunk_indices[(total + c) * 2 + 1] = c;
    }
    total += nc;
    chunk_offsets[i + 1] = (int64_t)total;
  }
  total_chunks_out[0] = total;
}

// ═══════════════════════════════════════════════════════════════════
// Host — fused pipeline with TMA
// ═══════════════════════════════════════════════════════════════════
__host__ __forceinline__ float ResolveScale(double scale) {
  float s = static_cast<float>(scale);
  return (s == 0.0f) ? (1.0f / sqrtf((float)kK)) : s;
}

// Persistent scratch buffer (survives across calls)
static char *g_scratch = nullptr;
static size_t g_scratch_size = 0;
static bool g_attrs_set = false;

void RunGdnPrefillTcgen05(
    TensorView q, TensorView k, TensorView v,
    ffi::Optional<TensorView> state_opt, TensorView A_log, TensorView a,
    TensorView dt_bias, TensorView b, TensorView cu_seqlens, double scale,
    TensorView output, TensorView new_state) {

  gdn_prefill::ValidateShapesAndTypes(q, k, v, A_log, a, dt_bias, b,
                                      cu_seqlens, output, new_state);

  const int64_t T = q.size(0);
  const int64_t num_seqs = cu_seqlens.size(0) - 1;

  const float *state_ptr = nullptr;
  if (state_opt.has_value()) {
    TensorView state = state_opt.value();
    gdn_prefill::ValidateState(state, num_seqs);
    CHECK_DEVICE(q, state);
    state_ptr = static_cast<const float *>(state.data_ptr());
  }

  const float scale_f = ResolveScale(scale);
  ffi::CUDADeviceGuard guard(q.device().device_id);
  const cudaStream_t stream = get_cuda_stream(q.device());

  // Upper bound for total chunks (used for scratch sizing)
  const int64_t max_chunks = num_seqs + (T + kBT - 1) / kBT;

  // Compute scratch size — NO d_h, NO d_vnew (fused away!)
  auto align256 = [](size_t x) -> size_t { return (x + 255) & ~255ULL; };
  const size_t sz_g    = align256(T * kHv * sizeof(float));
  const size_t sz_beta = align256(T * kHv * sizeof(float));
  const size_t sz_A    = align256(T * kHv * kBT * sizeof(float));
  const size_t sz_w    = align256(T * kHv * kK * sizeof(__nv_bfloat16));
  const size_t sz_u    = align256(T * kHv * kV * sizeof(__nv_bfloat16));
  const size_t sz_co   = align256((num_seqs + 2) * sizeof(int64_t));
  const size_t sz_ci   = align256(max_chunks * 2 * sizeof(int32_t));
  const size_t sz_tc   = align256(sizeof(int32_t));
  const size_t total_sz = sz_g + sz_beta + sz_A + sz_w + sz_u + sz_co + sz_ci + sz_tc;

  // Grow persistent scratch if needed (never shrinks)
  if (total_sz > g_scratch_size) {
    if (g_scratch) cudaFree(g_scratch);
    cudaError_t me = cudaMalloc(&g_scratch, total_sz);
    if (me != cudaSuccess) {
      TVM_FFI_THROW(RuntimeError) << "scratch cudaMalloc(" << total_sz << ") failed: " << cudaGetErrorString(me);
    }
    g_scratch_size = total_sz;
  }

  // Partition scratch — no d_h, no d_vnew
  char *p = g_scratch;
  float *d_g = (float *)p; p += sz_g;
  float *d_beta = (float *)p; p += sz_beta;
  float *d_A = (float *)p; p += sz_A;
  __nv_bfloat16 *d_w = (__nv_bfloat16 *)p; p += sz_w;
  __nv_bfloat16 *d_u = (__nv_bfloat16 *)p; p += sz_u;
  int64_t *d_co = (int64_t *)p; p += sz_co;
  int32_t *d_ci = (int32_t *)p; p += sz_ci;
  int32_t *d_tc = (int32_t *)p;

  auto *q_p = static_cast<const __nv_bfloat16 *>(q.data_ptr());
  auto *k_p = static_cast<const __nv_bfloat16 *>(k.data_ptr());
  auto *v_p = static_cast<const __nv_bfloat16 *>(v.data_ptr());
  auto *a_p = static_cast<const __nv_bfloat16 *>(a.data_ptr());
  auto *b_p = static_cast<const __nv_bfloat16 *>(b.data_ptr());
  auto *Alog_p = static_cast<const float *>(A_log.data_ptr());
  auto *dtb_p = static_cast<const float *>(dt_bias.data_ptr());
  auto *cusl_p = static_cast<const int64_t *>(cu_seqlens.data_ptr());
  auto *out_p = static_cast<__nv_bfloat16 *>(output.data_ptr());
  auto *ns_p = static_cast<float *>(new_state.data_ptr());

  // Set smem attributes once
  if (!g_attrs_set) {
    g_attrs_set = true;
    int smem_CA = (128*128*2 + 64*128*2 + 2*8+4 + 64*64*4 + 64*4 + 64*4 + 1023) & ~1023;
    cudaFuncSetAttribute(ComputeAKernel_TC, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_CA);
    int smem_WU = (128*64*2 + 64*64*2 + 8+4 + 64*4 + 1023) & ~1023;
    cudaFuncSetAttribute(ComputeWUKernel_TC, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_WU);
    constexpr int s_h_b = ((kBV*(kK+1)*4 + 1023)/1024)*1024;
    int smem_FRO = (s_h_b + 128*128*2 + 64*128*2 + 2*8+4 + kBT*4 + kBT*64*4 + kBT*kBV*2 + 1023) & ~1023;
    cudaFuncSetAttribute(FusedRecurrenceOutput, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_FRO);
  }

  // Launch PrepMeta FIRST, then create TMA descs on CPU while GPU runs
  PrepMetaKernel<<<1, 1, 0, stream>>>(cusl_p, d_ci, d_co, d_tc, num_seqs);
  int32_t h_tc;
  cudaMemcpyAsync(&h_tc, d_tc, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);

  // Create TMA descriptors while PrepMeta + memcpy runs on GPU
  static CUtensorMap k_tmap_128, k_tmap_64, q_tmap_128, w_tmap_128;
  static const void *c_kp = nullptr, *c_qp = nullptr;
  static void *c_wp = nullptr;
  static int64_t c_T = 0;
  if ((const void*)k_p != c_kp || (const void*)q_p != c_qp || (void*)d_w != c_wp || T != c_T) {
    init_tma_desc_3d(&k_tmap_128, k_p, (uint64_t)T, (uint64_t)(kHk * kK), 128, kK);
    init_tma_desc_3d(&k_tmap_64,  k_p, (uint64_t)T, (uint64_t)(kHk * kK), 64,  kK);
    init_tma_desc_3d(&q_tmap_128, q_p, (uint64_t)T, (uint64_t)(kHq * kK), 128, kK);
    init_tma_desc_3d(&w_tmap_128, d_w, (uint64_t)T, (uint64_t)(kHv * kK), 128, kK);
    c_kp = k_p; c_qp = q_p; c_wp = d_w; c_T = T;
  }

  // Compute smem sizes for launch
  int smem_CA = (128*128*2 + 64*128*2 + 2*8+4 + 64*64*4 + 64*4 + 64*4 + 1023) & ~1023;
  int smem_WU = (128*64*2 + 64*64*2 + 8+4 + 64*4 + 1023) & ~1023;
  constexpr int s_h_b = ((kBV*(kK+1)*4 + 1023)/1024)*1024;
  int smem_FRO = (s_h_b + 128*128*2 + 64*128*2 + 2*8+4 + kBT*4 + kBT*64*4 + kBT*kBV*2 + 1023) & ~1023;

  // Now sync to get total_chunks
  cudaStreamSynchronize(stream);
  const int64_t total_chunks = h_tc;

  // Clear stale errors from previous calls
  cudaGetLastError();

  // 1+2a. Fused preprocess + k@k^T → A_mat (grid kHk=4, not kHv=8)
  ComputeAKernel_TC<<<dim3(total_chunks, kHk), 128, smem_CA, stream>>>(
      k_tmap_128, k_tmap_64, a_p, b_p, Alog_p, dtb_p, cusl_p, d_ci,
      d_g, d_beta, d_A, total_chunks);

  // 2b. SolveTrilKernel — forward substitution (I+A)^{-1} in-place
  SolveTrilKernel<<<dim3(total_chunks, kHv), kBT, 0, stream>>>(
      cusl_p, d_ci, d_A, total_chunks);

  // 3. ComputeWU (no TMA — scaled+transposed loads)
  ComputeWUKernel_TC<<<dim3(total_chunks, kHv, 4), 128, smem_WU, stream>>>(
      k_p, v_p, d_g, d_beta, d_A, cusl_p, d_ci, d_w, d_u, total_chunks);

  // 4. FusedRecurrenceOutput [TMA for w, q, k loads]
  FusedRecurrenceOutput<<<dim3(kNVT, num_seqs * kHv), 128, smem_FRO, stream>>>(
      w_tmap_128, q_tmap_128, k_tmap_64,
      k_p, d_u, d_g, state_ptr, cusl_p, d_co,
      scale_f, out_p, ns_p, num_seqs);

  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    TVM_FFI_THROW(RuntimeError)
        << "GdnPrefillTcgen05 failed: " << cudaGetErrorString(err);
}

} // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(gdn_prefill_tcgen05, RunGdnPrefillTcgen05);
