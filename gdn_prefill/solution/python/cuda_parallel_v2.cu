/*
 * GDN Prefill v2 — Optimized CUDA Kernel for Blackwell SM_100a (B200)
 *
 * ═══════════════════════════════════════════════════════════════════
 * Optimizations over cuda_parallel_v1.cu:
 *
 * 1. FUSED PREPROCESSING (FusedPrepKernel)
 *    - Merged ComputeA + SolveTril + ComputeWU into one kernel
 *    - Eliminates 2 global memory round-trips for A_mat/A_inv (32MB for T=8192)
 *    - Block-recursive 16×16 inverse using tf32 wmma tensor cores:
 *      (I+A)^{-1} = (I-A)(I+A²)(I+A⁴)(I+A⁸)  [exact for strict lower tri]
 *      Replaces 64-step sequential forward substitution from v1
 *    - Off-diagonal blocks via Schur complement with fp32 scalar matmuls
 *
 * 2. H/O KERNEL SPLIT (matching Triton v4 architecture)
 *    - HRecurrenceKernel (tcgen05, BV=32): sequential state propagation
 *      Only 2 MMAs per chunk (was 5 in v1's FusedRecurrenceOutput)
 *      Cross-chunk TMA pipelining: w[ct+1] prefetched during h_update,
 *      k[ct] loaded during v_new computation
 *    - OOutputKernel (wmma/mma.sync, BV=64): parallel output computation
 *      3 MMAs per chunk, embarrassingly parallel across all chunks
 *      Uses s_qh smem buffer to avoid global read-modify-write
 *
 * 3. TMEM NON-DETERMINISM FIX
 *    - Root cause: tcgen05 TMEM alloc/dealloc under heavy concurrent block
 *      load on B200 causes non-deterministic output corruption
 *    - Evidence: Triton's o-kernel uses mma.sync (register-based), NOT tcgen05
 *    - Fix: OKernel uses nvcuda::wmma (generates mma.sync PTX) — no TMEM
 *
 * Results: 100/100 correct (stable), 1.21x speedup vs v1, 4.6x gap to Triton
 * ═══════════════════════════════════════════════════════════════════
 *
 * Pipeline:
 * 1. PrepMetaKernel:    GPU metadata (chunk indices, offsets)
 * 2. PreprocessKernel:  g_cumsum, beta (no tensor cores)
 * 3. FusedPrepKernel:   k@k^T → block-inverse → W,U computation (tcgen05+tf32 wmma)
 * 4. HRecurrenceKernel: State recurrence (tcgen05 TMA+TMEM, 2 MMAs/chunk serial)
 * 5. OOutputKernel:     Output computation (wmma/mma.sync, 3 MMAs/chunk parallel)
 *
 * tcgen05 tile layout:
 *   byte_off = tc * LBO + tr * SBO + wr * 16 + wc * 2
 *   tc = col/8, tr = row/8, wr = row%8, wc = col%8
 *   LBO = H * 16, SBO = 128
 *   BM MUST be 128 (hardware constraint). Pad A to 128 rows, read first 64 rows.
 */

#include "cuda_utils.h"

#include <cstdint>
#include <math.h>
#include <mma.h>
#include <vector>

namespace {

constexpr int kK = 128, kV = 128;
constexpr int64_t kHq = 4, kHk = 4, kHv = 8;
constexpr int kBT = 64, kBV = 32;
constexpr int kNVT = kV / kBV;   // 4
constexpr int MMA_K = 16;
constexpr int SBO_CONST = 128;  // 8 rows * 16 bytes

// H/O kernel split constants
constexpr int kBV_H = 32;              // BV for h-kernel (tcgen05-based)
constexpr int kBV_O = 64;              // BV for o-kernel (tcgen05-based)
constexpr int kNVT_H = kV / kBV_H;    // 4
constexpr int kNVT_O = kV / kBV_O;    // 2

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
    mbarrier_arrive_expect_tx(mbar_addr, cp_bytes);
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
    mbarrier_arrive_expect_tx(mbar_addr, total_cp_bytes);
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
// Kernel 2a: ComputeAKernel_TC — tcgen05 k@k^T + beta*exp(g)*mask → A_mat
// Grid: (total_chunks, kHv), Block: 128
// NOW WITH TMA: k_tmap_128 for s_A, k_tmap_64 for s_B
// ═══════════════════════════════════════════════════════════════════
__global__ void __launch_bounds__(128, 1)
ComputeAKernel_TC(
    const __grid_constant__ CUtensorMap k_tmap_128,
    const __grid_constant__ CUtensorMap k_tmap_64,
    const float *__restrict__ g_cumsum,
    const float *__restrict__ beta_f,
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
  const int64_t k_head = hv / (kHv / kHk);

  const int tid = threadIdx.x;
  const int warp_id = tid / 32;

  constexpr int BM = 128, BN = 64, BK = 128;

  extern __shared__ __align__(1024) char smem[];
  __nv_bfloat16 *s_A = reinterpret_cast<__nv_bfloat16 *>(smem);                // [128,128] tile = 32KB
  __nv_bfloat16 *s_B = reinterpret_cast<__nv_bfloat16 *>(smem + BM * BK * 2);  // [64,128] tile = 16KB
  char *after_tiles = smem + BM * BK * 2 + BN * BK * 2;
  uint64_t *mbars = reinterpret_cast<uint64_t *>(after_tiles);  // 2 mbarriers: [0]=TMA, [1]=MMA
  int *tmem_buf = reinterpret_cast<int *>(mbars + 2);
  float *s_result = reinterpret_cast<float *>(tmem_buf + 1);  // [64*64] = 16KB
  float *s_g = s_result + kBT * kBT;                           // [64]
  float *s_beta = s_g + kBT;                                   // [64]

  const uint32_t A_smem_base = __cvta_generic_to_shared(s_A);
  const uint32_t B_smem_base = __cvta_generic_to_shared(s_B);
  const uint32_t mbar_tma = __cvta_generic_to_shared(&mbars[0]);
  const uint32_t mbar_mma = __cvta_generic_to_shared(&mbars[1]);

  // Init mbarriers + alloc TMEM
  if (warp_id == 0 && elect_sync()) {
    mbarrier_init(mbar_tma, 1);
    mbarrier_init(mbar_mma, 1);
    asm volatile("fence.mbarrier_init.release.cluster;");
  } else if (warp_id == 1) {
    tcgen05_alloc(__cvta_generic_to_shared(tmem_buf), BN);
  }
  __syncthreads();
  const int taddr = tmem_buf[0];
  constexpr uint32_t idesc = make_tcgen05_idesc<BM, BN>();

  // TMA load k into s_A [128,128] and s_B [64,128]
  // 3D coords: (0, row_offset=cstart, col_group=k_head * K / 8)
  int col_group = (int)(k_head * kK / 8);
  constexpr int cp_A = BM * BK * (int)sizeof(__nv_bfloat16);  // 32KB
  constexpr int cp_B = BN * BK * (int)sizeof(__nv_bfloat16);  // 16KB
  tma_load_two_tiles_3d(
      A_smem_base, &k_tmap_128, (int)cstart, col_group,
      B_smem_base, &k_tmap_64,  (int)cstart, col_group,
      mbar_tma, cp_A + cp_B, warp_id);

  // Wait for TMA load to complete
  mbarrier_wait(mbar_tma, 0);
  // TMA mbarrier phase is now 1 (not reused in this kernel)

  __syncthreads();

  // MMA: batch all 8 K-tiles, single commit
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

  // Read TMEM → s_result [64,64] (padded), only first 64 rows of 128x64 output
  for (int n = 0; n < BN / 8; n++) {
    float tmp[8];
    tcgen05_ld<SHAPE::_32x32b, 8>(tmp, warp_id * 32, n * 8);
    tcgen05_wait_ld();
    int my_row = warp_id * 32 + (tid % 32);
    if (my_row < kBT) {
      for (int c = 0; c < 8; c++)
        s_result[my_row * kBT + n * 8 + c] = tmp[c];
    }
  }
  __syncthreads();

  if (warp_id == 0) tcgen05_dealloc(taddr, BN);

  // Load g_cumsum and beta
  if (tid < kBT) {
    s_g[tid] = (tid < clen) ? g_cumsum[(cstart + tid) * kHv + hv] : 0.0f;
    s_beta[tid] = (tid < clen) ? beta_f[(cstart + tid) * kHv + hv] : 0.0f;
  }
  __syncthreads();

  // Apply beta * exp(g_diff) * strictly-lower-triangular mask, write to global A_mat
  for (int i = tid; i < clen * kBT; i += 128) {
    int row = i / kBT, col = i % kBT;
    float val = 0.0f;
    if (col < row)
      val = s_beta[row] * s_result[row * kBT + col] * expf(s_g[row] - s_g[col]);
    A_mat[(cstart + row) * kHv * kBT + hv * kBT + col] = val;
  }
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 2c: ComputeQKKernel_TC — precompute q@k^T with causal mask + gating
// Grid: (total_chunks, kHv), Block: 128
// Reuses A_mat buffer (safe: runs after ComputeWU consumes A_inv)
// ═══════════════════════════════════════════════════════════════════
__global__ void __launch_bounds__(128, 1)
ComputeQKKernel_TC(
    const __grid_constant__ CUtensorMap q_tmap_128,
    const __grid_constant__ CUtensorMap k_tmap_64,
    const float *__restrict__ g_cumsum,
    const int64_t *__restrict__ cu_seqlens,
    const int32_t *__restrict__ chunk_indices,
    float *__restrict__ qk_mat,
    int64_t total_chunks) {

  const int chunk_id = blockIdx.x, hv = blockIdx.y;
  if (chunk_id >= total_chunks) return;

  const int seq_idx = chunk_indices[chunk_id * 2];
  const int local_chunk = chunk_indices[chunk_id * 2 + 1];
  const int64_t s0 = cu_seqlens[seq_idx], s1 = cu_seqlens[seq_idx + 1];
  const int64_t cstart = s0 + (int64_t)local_chunk * kBT;
  const int clen = min(kBT, (int)(s1 - cstart));
  const int64_t q_head = hv / (kHv / kHq);
  const int64_t k_head = hv / (kHv / kHk);

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

  const uint32_t A_smem_base = __cvta_generic_to_shared(s_A);
  const uint32_t B_smem_base = __cvta_generic_to_shared(s_B);
  const uint32_t mbar_tma = __cvta_generic_to_shared(&mbars[0]);
  const uint32_t mbar_mma = __cvta_generic_to_shared(&mbars[1]);

  if (warp_id == 0 && elect_sync()) {
    mbarrier_init(mbar_tma, 1);
    mbarrier_init(mbar_mma, 1);
    asm volatile("fence.mbarrier_init.release.cluster;");
  } else if (warp_id == 1) {
    tcgen05_alloc(__cvta_generic_to_shared(tmem_buf), BN);
  }
  __syncthreads();
  const int taddr = tmem_buf[0];
  constexpr uint32_t idesc = make_tcgen05_idesc<BM, BN>();

  // TMA load q into s_A [128,128] and k into s_B [64,128]
  int q_col = (int)(q_head * kK / 8);
  int k_col = (int)(k_head * kK / 8);
  constexpr int cp_A = BM * BK * (int)sizeof(__nv_bfloat16);
  constexpr int cp_B = BN * BK * (int)sizeof(__nv_bfloat16);
  tma_load_two_tiles_3d(
      A_smem_base, &q_tmap_128, (int)cstart, q_col,
      B_smem_base, &k_tmap_64,  (int)cstart, k_col,
      mbar_tma, cp_A + cp_B, warp_id);

  mbarrier_wait(mbar_tma, 0);
  __syncthreads();

  // MMA: q @ k^T
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
    tcgen05_ld<SHAPE::_32x32b, 8>(tmp, warp_id * 32, n * 8);
    tcgen05_wait_ld();
    int my_row = warp_id * 32 + (tid % 32);
    if (my_row < kBT)
      for (int c = 0; c < 8; c++)
        s_result[my_row * kBT + n * 8 + c] = tmp[c];
  }
  __syncthreads();

  if (warp_id == 0) tcgen05_dealloc(taddr, BN);

  // Load g_cumsum
  if (tid < kBT)
    s_g[tid] = (tid < clen) ? g_cumsum[(cstart + tid) * kHv + hv] : 0.0f;
  __syncthreads();

  // Apply exp(g_diff) * causal mask (col <= row), write to global qk_mat
  // Only write valid rows (0..clen-1) to avoid corrupting other sequences' data
  for (int i = tid; i < clen * kBT; i += 128) {
    int row = i / kBT, col = i % kBT;
    float val = 0.0f;
    if (col < clen && col <= row)
      val = s_result[row * kBT + col] * expf(s_g[row] - s_g[col]);
    qk_mat[(cstart + row) * kHv * kBT + hv * kBT + col] = val;
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

  const uint32_t Ainv_smem = __cvta_generic_to_shared(s_Ainv);
  const uint32_t input_smem = __cvta_generic_to_shared(s_input);
  const uint32_t mbar_addr = __cvta_generic_to_shared(mbars);

  if (warp_id == 0 && elect_sync()) {
    mbarrier_init(mbar_addr, 1);
    asm volatile("fence.mbarrier_init.release.cluster;");
  } else if (warp_id == 1) {
    tcgen05_alloc(__cvta_generic_to_shared(tmem_buf), BN);
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
    tcgen05_ld<SHAPE::_32x32b, 8>(tmp, warp_id * 32, n * 8);
    tcgen05_wait_ld();
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
// Kernel FUSED: FusedPrepKernel — ComputeA + SolveTril + ComputeWU in one kernel
// Grid: (total_chunks, kHv), Block: 128
// Eliminates 2 global memory round-trips for A_mat/A_inv.
//
// Shared memory layout (~97KB, fits B200's 228KB):
//   s_A_tiles: [128,128] bf16 = 32KB  (TMA tile for k, reused for A_inv tile in Phase 3)
//   s_B_tiles: [64,128] bf16 = 16KB   (TMA tile for k, reused for input tile in Phase 3)
//   mbar_tma(8) + mbar_mma(8) + tmem_buf(4) = 20B
//   s_result: [64*64] fp32 = 16KB     (k@k^T result, then A matrix)
//   s_result_inv: [64*64] fp32 = 16KB (A_inv from forward substitution)
//   s_g: [64] fp32 = 256B
//   s_beta: [64] fp32 = 256B
//   s_bg: [64] fp32 = 256B            (beta*exp(g) or beta, for WU scaling)
// Total: ~81KB
// ═══════════════════════════════════════════════════════════════════
__global__ void __launch_bounds__(128, 1)
FusedPrepKernel(
    const __grid_constant__ CUtensorMap k_tmap_128,
    const __grid_constant__ CUtensorMap k_tmap_64,
    const __nv_bfloat16 *__restrict__ k_in,
    const __nv_bfloat16 *__restrict__ v_in,
    const float *__restrict__ g_cumsum,
    const float *__restrict__ beta_f,
    const int64_t *__restrict__ cu_seqlens,
    const int32_t *__restrict__ chunk_indices,
    __nv_bfloat16 *__restrict__ w_out,
    __nv_bfloat16 *__restrict__ u_out,
    int64_t total_chunks) {

  const int chunk_id = blockIdx.x, hv = blockIdx.y;
  if (chunk_id >= total_chunks) return;

  const int seq_idx = chunk_indices[chunk_id * 2];
  const int local_chunk = chunk_indices[chunk_id * 2 + 1];
  const int64_t s0 = cu_seqlens[seq_idx], s1 = cu_seqlens[seq_idx + 1];
  const int64_t cstart = s0 + (int64_t)local_chunk * kBT;
  const int clen = min(kBT, (int)(s1 - cstart));
  const int64_t k_head = hv / (kHv / kHk);

  const int tid = threadIdx.x;
  const int warp_id = tid / 32;

  constexpr int BM = 128, BN = 64, BK = 128;
  constexpr int BK_inner = 64;  // K-dim for Phase 3 MMA (A_inv is 64x64)

  extern __shared__ __align__(1024) char smem[];
  __nv_bfloat16 *s_A_tiles = reinterpret_cast<__nv_bfloat16 *>(smem);                    // [128,128] tile = 32KB
  __nv_bfloat16 *s_B_tiles = reinterpret_cast<__nv_bfloat16 *>(smem + BM * BK * 2);      // [64,128] tile = 16KB
  char *after_tiles = smem + BM * BK * 2 + BN * BK * 2;
  uint64_t *mbars = reinterpret_cast<uint64_t *>(after_tiles);  // 2 mbarriers: [0]=TMA, [1]=MMA
  int *tmem_buf = reinterpret_cast<int *>(mbars + 2);
  float *s_result = reinterpret_cast<float *>(tmem_buf + 1);       // [64*64] = 16KB
  float *s_result_inv = s_result + kBT * kBT;                      // [64*64] = 16KB
  float *s_g = s_result_inv + kBT * kBT;                           // [64]
  float *s_beta = s_g + kBT;                                       // [64]
  float *s_bg = s_beta + kBT;                                      // [64]

  const uint32_t A_smem_base = __cvta_generic_to_shared(s_A_tiles);
  const uint32_t B_smem_base = __cvta_generic_to_shared(s_B_tiles);
  const uint32_t mbar_tma = __cvta_generic_to_shared(&mbars[0]);
  const uint32_t mbar_mma = __cvta_generic_to_shared(&mbars[1]);

  // Init mbarriers + alloc TMEM
  if (warp_id == 0 && elect_sync()) {
    mbarrier_init(mbar_tma, 1);
    mbarrier_init(mbar_mma, 1);
    asm volatile("fence.mbarrier_init.release.cluster;");
  } else if (warp_id == 1) {
    tcgen05_alloc(__cvta_generic_to_shared(tmem_buf), BN);
  }
  __syncthreads();
  const int taddr = tmem_buf[0];
  constexpr uint32_t idesc = make_tcgen05_idesc<BM, BN>();

  // ════════════════════════════════════════════════════════════
  // Phase 1: Compute A_mat (same as ComputeAKernel_TC)
  // TMA load k → s_A_tiles[128,128] and s_B_tiles[64,128]
  // MMA k@k^T → TMEM → s_result[64,64]
  // Apply beta*exp(g)*mask → s_result becomes A matrix
  // ════════════════════════════════════════════════════════════

  int col_group = (int)(k_head * kK / 8);
  constexpr int cp_A = BM * BK * (int)sizeof(__nv_bfloat16);  // 32KB
  constexpr int cp_B = BN * BK * (int)sizeof(__nv_bfloat16);  // 16KB
  tma_load_two_tiles_3d(
      A_smem_base, &k_tmap_128, (int)cstart, col_group,
      B_smem_base, &k_tmap_64,  (int)cstart, col_group,
      mbar_tma, cp_A + cp_B, warp_id);

  // Wait for TMA load
  mbarrier_wait(mbar_tma, 0);
  __syncthreads();

  // MMA: batch all 8 K-tiles, single commit
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
  if (warp_id == 0 && elect_sync()) mbarrier_init(mbar_mma, 1);
  tcgen05_fence();

  // Read TMEM → s_result [64,64] (padded), only first 64 rows of 128x64 output
  for (int n = 0; n < BN / 8; n++) {
    float tmp[8];
    tcgen05_ld<SHAPE::_32x32b, 8>(tmp, warp_id * 32, n * 8);
    tcgen05_wait_ld();
    int my_row = warp_id * 32 + (tid % 32);
    if (my_row < kBT) {
      for (int c = 0; c < 8; c++)
        s_result[my_row * kBT + n * 8 + c] = tmp[c];
    }
  }
  __syncthreads();

  // Load g_cumsum and beta
  if (tid < kBT) {
    s_g[tid] = (tid < clen) ? g_cumsum[(cstart + tid) * kHv + hv] : 0.0f;
    s_beta[tid] = (tid < clen) ? beta_f[(cstart + tid) * kHv + hv] : 0.0f;
  }
  __syncthreads();

  // Apply beta * exp(g_diff) * strictly-lower-triangular mask IN PLACE on s_result
  for (int i = tid; i < kBT * kBT; i += 128) {
    int row = i / kBT, col = i % kBT;
    float val = 0.0f;
    if (row < clen && col < row)
      val = s_beta[row] * s_result[row * kBT + col] * expf(s_g[row] - s_g[col]);
    s_result[row * kBT + col] = val;
  }
  __syncthreads();

  // ════════════════════════════════════════════════════════════
  // Phase 2: SolveTril — block-recursive (I+A)^{-1} via Neumann series
  // s_result has A matrix [64*64], s_result_inv will have A_inv [64*64]
  // Split 64x64 into 4x4 grid of 16x16 blocks.
  // Step 1: Invert diagonal blocks with tf32 wmma (parallel across 4 warps)
  // Step 2: Off-diagonal blocks via Schur complement (scalar fp32 matmul)
  // ════════════════════════════════════════════════════════════

  // Initialize s_result_inv to identity
  for (int idx = tid; idx < kBT * kBT; idx += 128) {
    int row = idx / kBT, col = idx % kBT;
    s_result_inv[row * kBT + col] = (row == col) ? 1.0f : 0.0f;
  }
  __syncthreads();

  // --- Step 1: Diagonal block inversion using tf32 wmma ---
  // Each warp (0-3) inverts one 16x16 diagonal block independently.
  // Formula: (I+D)^{-1} = (I-D)(I+D^2)(I+D^4)(I+D^8) for strictly lower triangular D.
  //
  // Temp space in s_A_tiles area (32KB available, we use 12KB):
  //   Per warp w: power[16][16], Ai[16][16], tmp[16][16] — each stride 16
  {
    using namespace nvcuda::wmma;
    constexpr int BS = 16;  // block size
    constexpr int S = kBT;  // stride in s_result/s_result_inv (64)

    // Reuse s_A_tiles (bf16 tile area, 32KB) as fp32 temp during Phase 2
    float *s_tmp = reinterpret_cast<float *>(s_A_tiles);
    // Layout per warp w: power at s_tmp[w*768], Ai at s_tmp[w*768+256], tmp at s_tmp[w*768+512]
    float *my_power = s_tmp + warp_id * 768;          // [16][16] stride 16
    float *my_Ai    = s_tmp + warp_id * 768 + 256;    // [16][16] stride 16
    float *my_tmp   = s_tmp + warp_id * 768 + 512;    // [16][16] stride 16

    const int bi = warp_id;  // diagonal block index (0..3)
    const int blk_row = bi * BS;
    const int blk_col = bi * BS;
    const int lane = tid % 32;

    // Initialize power = D (the diagonal block from s_result)
    // Initialize Ai = I - D
    for (int idx2 = lane; idx2 < BS * BS; idx2 += 32) {
      int r = idx2 / BS, c = idx2 % BS;
      float d_val = s_result[(blk_row + r) * S + (blk_col + c)];
      my_power[r * BS + c] = d_val;
      my_Ai[r * BS + c] = ((r == c) ? 1.0f : 0.0f) - d_val;
    }
    __syncwarp();

    // 3 iterations of squaring: power = power @ power, Ai = Ai @ (I + power)
    // wmma tf32: m16n16k8, A row_major [M,K], B row_major [K,N]
    // For C = A @ B: a_frag loads A[16,8] at col offset k*8, b_frag loads B[8,16] at row offset k*8
    for (int iter = 0; iter < 3; iter++) {
      // --- Compute tmp = power @ power (16x16 @ 16x16 tf32 wmma) ---
      {
        fragment<accumulator, 16, 16, 8, float> acc;
        fill_fragment(acc, 0.0f);

        // k-tile 0: k=0..7
        {
          fragment<matrix_a, 16, 16, 8, precision::tf32, row_major> a_frag;
          fragment<matrix_b, 16, 16, 8, precision::tf32, row_major> b_frag;
          load_matrix_sync(a_frag, my_power, BS);           // A[16,8] from col 0, stride=16
          load_matrix_sync(b_frag, my_power + 0 * BS, BS);  // B[8,16] from row 0, stride=16
          mma_sync(acc, a_frag, b_frag, acc);
        }
        // k-tile 1: k=8..15
        {
          fragment<matrix_a, 16, 16, 8, precision::tf32, row_major> a_frag;
          fragment<matrix_b, 16, 16, 8, precision::tf32, row_major> b_frag;
          load_matrix_sync(a_frag, my_power + 8, BS);       // A[16,8] from col 8, stride=16
          load_matrix_sync(b_frag, my_power + 8 * BS, BS);  // B[8,16] from row 8, stride=16
          mma_sync(acc, a_frag, b_frag, acc);
        }
        store_matrix_sync(my_tmp, acc, BS, mem_row_major);
      }
      __syncwarp();

      // Copy tmp → power (power = power^2)
      for (int idx2 = lane; idx2 < BS * BS; idx2 += 32)
        my_power[idx2] = my_tmp[idx2];
      __syncwarp();

      // --- Compute Ai_new = Ai @ (I + power) ---
      // Form (I + power) in my_tmp
      for (int idx2 = lane; idx2 < BS * BS; idx2 += 32) {
        int r = idx2 / BS, c = idx2 % BS;
        my_tmp[r * BS + c] = ((r == c) ? 1.0f : 0.0f) + my_power[r * BS + c];
      }
      __syncwarp();

      // Compute Ai @ my_tmp → result stored directly to my_Ai
      // (wmma reads all of my_tmp and my_Ai into registers before storing back)
      {
        fragment<accumulator, 16, 16, 8, float> acc;
        fill_fragment(acc, 0.0f);

        // k-tile 0
        {
          fragment<matrix_a, 16, 16, 8, precision::tf32, row_major> a_frag;
          fragment<matrix_b, 16, 16, 8, precision::tf32, row_major> b_frag;
          load_matrix_sync(a_frag, my_Ai, BS);
          load_matrix_sync(b_frag, my_tmp + 0 * BS, BS);
          mma_sync(acc, a_frag, b_frag, acc);
        }
        // k-tile 1
        {
          fragment<matrix_a, 16, 16, 8, precision::tf32, row_major> a_frag;
          fragment<matrix_b, 16, 16, 8, precision::tf32, row_major> b_frag;
          load_matrix_sync(a_frag, my_Ai + 8, BS);
          load_matrix_sync(b_frag, my_tmp + 8 * BS, BS);
          mma_sync(acc, a_frag, b_frag, acc);
        }
        store_matrix_sync(my_Ai, acc, BS, mem_row_major);
      }
      __syncwarp();
    }

    // Write inverted diagonal block to s_result_inv
    for (int idx2 = lane; idx2 < BS * BS; idx2 += 32) {
      int r = idx2 / BS, c = idx2 % BS;
      s_result_inv[(blk_row + r) * S + (blk_col + c)] = my_Ai[r * BS + c];
    }
  }
  __syncthreads();

  // --- Step 2: Off-diagonal blocks via Schur complement ---
  // Uses scalar fp32 16x16 matmul with all 128 threads.
  // A_ij aliases: s_result[(bi*16)*64 + bj*16], stride 64
  // Ai_ij aliases: s_result_inv[(bi*16)*64 + bj*16], stride 64
  //
  // Helper lambda: C[16][16] += A[16][16] @ B[16][16], all in s_result/s_result_inv with stride S=64
  // We compute into a temp buffer, then negate and multiply by Ai_ii.
  //
  // Notation: blk(buf, bi, bj) = &buf[(bi*16)*64 + bj*16]
  // Ai_10 = -Ai_11 @ A_10 @ Ai_00
  // Ai_20 = -Ai_22 @ (A_20 @ Ai_00 + A_21 @ Ai_10)
  // Ai_21 = -Ai_22 @ A_21 @ Ai_11
  // Ai_30 = -Ai_33 @ (A_30 @ Ai_00 + A_31 @ Ai_10 + A_32 @ Ai_20)
  // Ai_31 = -Ai_33 @ (A_31 @ Ai_11 + A_32 @ Ai_21)
  // Ai_32 = -Ai_33 @ A_32 @ Ai_22

  // Reuse s_A_tiles as temp for off-diagonal matmul results
  // tmp1[16][16] at offset 0, tmp2[16][16] at offset 256 (all stride 16)
  {
    constexpr int BS = 16;
    constexpr int S = kBT;
    float *tmp1 = reinterpret_cast<float *>(s_A_tiles);          // [16][16] stride 16

    // Macro-like helpers using lambdas would be ideal but CUDA doesn't support
    // device lambdas easily in __global__. Use inline code.

    // matmul16x16: C = A @ B, where A is at (a_ptr, a_stride), B at (b_ptr, b_stride)
    // Result stored at (c_ptr, c_stride). All 128 threads participate.
    // Each thread computes ~2 elements of the 256-element output.

    // --- Ai_10 = -Ai_11 @ A_10 @ Ai_00 ---
    // Step a: tmp1 = A_10 @ Ai_00
    for (int idx2 = tid; idx2 < BS * BS; idx2 += 128) {
      int r = idx2 / BS, c = idx2 % BS;
      float sum = 0.0f;
      for (int k = 0; k < BS; k++)
        sum += s_result[(1*BS + r) * S + (0*BS + k)] * s_result_inv[(0*BS + k) * S + (0*BS + c)];
      tmp1[r * BS + c] = sum;
    }
    __syncthreads();
    // Step b: Ai_10 = -Ai_11 @ tmp1
    for (int idx2 = tid; idx2 < BS * BS; idx2 += 128) {
      int r = idx2 / BS, c = idx2 % BS;
      float sum = 0.0f;
      for (int k = 0; k < BS; k++)
        sum += s_result_inv[(1*BS + r) * S + (1*BS + k)] * tmp1[k * BS + c];
      s_result_inv[(1*BS + r) * S + (0*BS + c)] = -sum;
    }
    __syncthreads();

    // --- Ai_21 = -Ai_22 @ A_21 @ Ai_11 ---
    // Step a: tmp1 = A_21 @ Ai_11
    for (int idx2 = tid; idx2 < BS * BS; idx2 += 128) {
      int r = idx2 / BS, c = idx2 % BS;
      float sum = 0.0f;
      for (int k = 0; k < BS; k++)
        sum += s_result[(2*BS + r) * S + (1*BS + k)] * s_result_inv[(1*BS + k) * S + (1*BS + c)];
      tmp1[r * BS + c] = sum;
    }
    __syncthreads();
    // Step b: Ai_21 = -Ai_22 @ tmp1
    for (int idx2 = tid; idx2 < BS * BS; idx2 += 128) {
      int r = idx2 / BS, c = idx2 % BS;
      float sum = 0.0f;
      for (int k = 0; k < BS; k++)
        sum += s_result_inv[(2*BS + r) * S + (2*BS + k)] * tmp1[k * BS + c];
      s_result_inv[(2*BS + r) * S + (1*BS + c)] = -sum;
    }
    __syncthreads();

    // --- Ai_20 = -Ai_22 @ (A_20 @ Ai_00 + A_21 @ Ai_10) ---
    // Step a: tmp1 = A_20 @ Ai_00
    for (int idx2 = tid; idx2 < BS * BS; idx2 += 128) {
      int r = idx2 / BS, c = idx2 % BS;
      float sum = 0.0f;
      for (int k = 0; k < BS; k++)
        sum += s_result[(2*BS + r) * S + (0*BS + k)] * s_result_inv[(0*BS + k) * S + (0*BS + c)];
      tmp1[r * BS + c] = sum;
    }
    __syncthreads();
    // Step b: tmp1 += A_21 @ Ai_10
    for (int idx2 = tid; idx2 < BS * BS; idx2 += 128) {
      int r = idx2 / BS, c = idx2 % BS;
      float sum = 0.0f;
      for (int k = 0; k < BS; k++)
        sum += s_result[(2*BS + r) * S + (1*BS + k)] * s_result_inv[(1*BS + k) * S + (0*BS + c)];
      tmp1[r * BS + c] += sum;
    }
    __syncthreads();
    // Step c: Ai_20 = -Ai_22 @ tmp1
    for (int idx2 = tid; idx2 < BS * BS; idx2 += 128) {
      int r = idx2 / BS, c = idx2 % BS;
      float sum = 0.0f;
      for (int k = 0; k < BS; k++)
        sum += s_result_inv[(2*BS + r) * S + (2*BS + k)] * tmp1[k * BS + c];
      s_result_inv[(2*BS + r) * S + (0*BS + c)] = -sum;
    }
    __syncthreads();

    // --- Ai_32 = -Ai_33 @ A_32 @ Ai_22 ---
    // Step a: tmp1 = A_32 @ Ai_22
    for (int idx2 = tid; idx2 < BS * BS; idx2 += 128) {
      int r = idx2 / BS, c = idx2 % BS;
      float sum = 0.0f;
      for (int k = 0; k < BS; k++)
        sum += s_result[(3*BS + r) * S + (2*BS + k)] * s_result_inv[(2*BS + k) * S + (2*BS + c)];
      tmp1[r * BS + c] = sum;
    }
    __syncthreads();
    // Step b: Ai_32 = -Ai_33 @ tmp1
    for (int idx2 = tid; idx2 < BS * BS; idx2 += 128) {
      int r = idx2 / BS, c = idx2 % BS;
      float sum = 0.0f;
      for (int k = 0; k < BS; k++)
        sum += s_result_inv[(3*BS + r) * S + (3*BS + k)] * tmp1[k * BS + c];
      s_result_inv[(3*BS + r) * S + (2*BS + c)] = -sum;
    }
    __syncthreads();

    // --- Ai_31 = -Ai_33 @ (A_31 @ Ai_11 + A_32 @ Ai_21) ---
    // Step a: tmp1 = A_31 @ Ai_11
    for (int idx2 = tid; idx2 < BS * BS; idx2 += 128) {
      int r = idx2 / BS, c = idx2 % BS;
      float sum = 0.0f;
      for (int k = 0; k < BS; k++)
        sum += s_result[(3*BS + r) * S + (1*BS + k)] * s_result_inv[(1*BS + k) * S + (1*BS + c)];
      tmp1[r * BS + c] = sum;
    }
    __syncthreads();
    // Step b: tmp1 += A_32 @ Ai_21
    for (int idx2 = tid; idx2 < BS * BS; idx2 += 128) {
      int r = idx2 / BS, c = idx2 % BS;
      float sum = 0.0f;
      for (int k = 0; k < BS; k++)
        sum += s_result[(3*BS + r) * S + (2*BS + k)] * s_result_inv[(2*BS + k) * S + (1*BS + c)];
      tmp1[r * BS + c] += sum;
    }
    __syncthreads();
    // Step c: Ai_31 = -Ai_33 @ tmp1
    for (int idx2 = tid; idx2 < BS * BS; idx2 += 128) {
      int r = idx2 / BS, c = idx2 % BS;
      float sum = 0.0f;
      for (int k = 0; k < BS; k++)
        sum += s_result_inv[(3*BS + r) * S + (3*BS + k)] * tmp1[k * BS + c];
      s_result_inv[(3*BS + r) * S + (1*BS + c)] = -sum;
    }
    __syncthreads();

    // --- Ai_30 = -Ai_33 @ (A_30 @ Ai_00 + A_31 @ Ai_10 + A_32 @ Ai_20) ---
    // Step a: tmp1 = A_30 @ Ai_00
    for (int idx2 = tid; idx2 < BS * BS; idx2 += 128) {
      int r = idx2 / BS, c = idx2 % BS;
      float sum = 0.0f;
      for (int k = 0; k < BS; k++)
        sum += s_result[(3*BS + r) * S + (0*BS + k)] * s_result_inv[(0*BS + k) * S + (0*BS + c)];
      tmp1[r * BS + c] = sum;
    }
    __syncthreads();
    // Step b: tmp1 += A_31 @ Ai_10
    for (int idx2 = tid; idx2 < BS * BS; idx2 += 128) {
      int r = idx2 / BS, c = idx2 % BS;
      float sum = 0.0f;
      for (int k = 0; k < BS; k++)
        sum += s_result[(3*BS + r) * S + (1*BS + k)] * s_result_inv[(1*BS + k) * S + (0*BS + c)];
      tmp1[r * BS + c] += sum;
    }
    __syncthreads();
    // Step c: tmp1 += A_32 @ Ai_20
    for (int idx2 = tid; idx2 < BS * BS; idx2 += 128) {
      int r = idx2 / BS, c = idx2 % BS;
      float sum = 0.0f;
      for (int k = 0; k < BS; k++)
        sum += s_result[(3*BS + r) * S + (2*BS + k)] * s_result_inv[(2*BS + k) * S + (0*BS + c)];
      tmp1[r * BS + c] += sum;
    }
    __syncthreads();
    // Step d: Ai_30 = -Ai_33 @ tmp1
    for (int idx2 = tid; idx2 < BS * BS; idx2 += 128) {
      int r = idx2 / BS, c = idx2 % BS;
      float sum = 0.0f;
      for (int k = 0; k < BS; k++)
        sum += s_result_inv[(3*BS + r) * S + (3*BS + k)] * tmp1[k * BS + c];
      s_result_inv[(3*BS + r) * S + (0*BS + c)] = -sum;
    }
    __syncthreads();
  }

  // Handle rows >= clen: set to identity
  if (clen < kBT) {
    for (int idx2 = tid; idx2 < kBT * kBT; idx2 += 128) {
      int row = idx2 / kBT, col = idx2 % kBT;
      if (row >= clen || col >= clen)
        s_result_inv[row * kBT + col] = (row == col) ? 1.0f : 0.0f;
    }
    __syncthreads();
  }

  // ════════════════════════════════════════════════════════════
  // Phase 3: Compute W and U (replaces ComputeWUKernel_TC)
  // A_inv is in s_result_inv [64,64] fp32 → load to s_A_tiles [128,64] bf16 ONCE
  // Then iterate 4 output tiles (w_col0, w_col1, u_col0, u_col1)
  // ════════════════════════════════════════════════════════════

  // Load A_inv from s_result_inv → s_A_tiles [128,64] bf16 (fp32→bf16, rows 64-127 = 0)
  // This is the same load_fp32_to_tile pattern but from flat array with stride kBT
  {
    const int total_chunks_tile = BM * (BK_inner / 8);  // 128 * 8 = 1024
    __nv_bfloat16 tmp_tile[8];
    for (int i = tid; i < total_chunks_tile; i += 128) {
      int row = i / (BK_inner / 8);
      int col8 = (i % (BK_inner / 8)) * 8;
      if (row < kBT) {
        for (int j = 0; j < 8; j++)
          tmp_tile[j] = __float2bfloat16(s_result_inv[row * kBT + col8 + j]);
      } else {
        for (int j = 0; j < 8; j++) tmp_tile[j] = __float2bfloat16(0.0f);
      }
      int tr = row / 8, tc = col8 / 8, wr = row % 8;
      int byte_off = tc * (BM * 16) + tr * SBO_CONST + wr * 16;
      *reinterpret_cast<int4 *>(reinterpret_cast<char *>(s_A_tiles) + byte_off) =
          *reinterpret_cast<int4 *>(tmp_tile);
    }
  }
  __syncthreads();

  // Reuse idesc for BK_inner=64 MMA (BM=128, BN=64, but K-tiles = 64/16 = 4)
  const uint32_t Ainv_smem = A_smem_base;
  const uint32_t input_smem = B_smem_base;

  // Iterate over 4 output tiles
  for (int tile_id = 0; tile_id < 4; tile_id++) {
    const bool is_w = (tile_id < 2);
    const int out_col = (tile_id % 2) * 64;

    // Precompute scaling factors for this tile type
    if (tid < kBT) {
      if (is_w) {
        float b = s_beta[tid];
        float g = s_g[tid];
        s_bg[tid] = (tid < clen) ? b * expf(g) : 0.0f;
      } else {
        s_bg[tid] = (tid < clen) ? s_beta[tid] : 0.0f;
      }
    }
    __syncthreads();

    // Load scaled+transposed input → s_B_tiles [64,64] bf16
    // Input tile: row=output_col_idx (j), col=time (k) — transposed layout
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
      tile_store(s_B_tiles, j, k, BN, val);
    }
    __syncthreads();

    // MMA: A_inv[128,64] @ input[64,64] → TMEM [128,64]
    // 4 K-tiles (BK_inner=64, MMA_K=16)
    int phase = 0;
    for (int ki = 0; ki < BK_inner / MMA_K; ki++) {
      tcgen05_fence();
      if (warp_id == 0 && elect_sync()) {
        uint32_t a_base = Ainv_smem + ki * 2 * (BM * 16);
        uint32_t b_base = input_smem + ki * 2 * (BN * 16);
        uint64_t a_desc = make_tcgen05_desc_noswizzle(a_base, BM, SBO_CONST);
        uint64_t b_desc = make_tcgen05_desc_noswizzle(b_base, BN, SBO_CONST);
        tcgen05_mma(taddr, a_desc, b_desc, idesc, ki);
        tcgen05_commit(mbar_mma);
      }
      mbarrier_wait(mbar_mma, phase);
      phase ^= 1;
    }
    tcgen05_fence();

    // Read TMEM → write to global w_out or u_out
    for (int n = 0; n < BN / 8; n++) {
      float tmp[8];
      tcgen05_ld<SHAPE::_32x32b, 8>(tmp, warp_id * 32, n * 8);
      tcgen05_wait_ld();
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
  } // end tile loop

  // Dealloc TMEM
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

  const uint32_t tileA_smem = __cvta_generic_to_shared(s_tile_a);
  const uint32_t tileB_smem = __cvta_generic_to_shared(s_tile_b);
  const uint32_t mbar_tma = __cvta_generic_to_shared(&mbars[0]);
  const uint32_t mbar_mma = __cvta_generic_to_shared(&mbars[1]);

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
    tcgen05_alloc(__cvta_generic_to_shared(tmem_buf), TC_BN);
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
      tcgen05_ld<SHAPE::_32x32b, 8>(tmp, warp_id * 32, n * 8);
      tcgen05_wait_ld();
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
      tcgen05_ld<SHAPE::_32x32b, 8>(tmp, warp_id * 32, n * 8);
      tcgen05_wait_ld();
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

    // Read q@k^T → s_wh [64,64]
    for (int n = 0; n < TC_BN / 8; n++) {
      float tmp[8];
      tcgen05_ld<SHAPE::_32x32b, 8>(tmp, warp_id * 32, n * 8);
      tcgen05_wait_ld();
      int my_row = warp_id * 32 + (tid % 32);
      if (my_row < kBT)
        for (int c = 0; c < 8; c++)
          s_wh[my_row * TC_BN + n * 8 + c] = tmp[c];
    }
    __syncthreads();

    // Apply gating + causal mask to q@k^T
    for (int i = tid; i < kBT * kBT; i += 128) {
      int row = i / kBT, col = i % kBT;
      float val = 0.0f;
      if (row < clen && col < clen && col <= row)
        val = s_wh[row * TC_BN + col] * expf(s_gc[row] - s_gc[col]);
      s_wh[row * TC_BN + col] = val;
    }
    __syncthreads();

    // ════════════════════════════════════════════════
    // Step 4b: attn @ v_new via tcgen05 (replaces serial dot product)
    // Format masked attn→tile_a [128,64], vnew^T→tile_b [64,64]; MMA; reload k
    // ════════════════════════════════════════════════
    {
      constexpr int ATTN_BK = kBT;  // 64
      // Load masked attn [64,64] fp32 → tile_a [128,64] bf16 (rows 64-127 = 0)
      constexpr int attn_a_total = TC_BM * (ATTN_BK / 8);  // 128*8=1024
      for (int i = tid; i < attn_a_total; i += 128) {
        int row = i / (ATTN_BK / 8);
        int col8 = (i % (ATTN_BK / 8)) * 8;
        __nv_bfloat16 tmp[8];
        if (row < kBT) {
          #pragma unroll
          for (int j = 0; j < 8; j++)
            tmp[j] = __float2bfloat16(s_wh[row * TC_BN + col8 + j]);
        } else {
          #pragma unroll
          for (int j = 0; j < 8; j++) tmp[j] = __float2bfloat16(0.0f);
        }
        int tr = row / 8, tc = col8 / 8, wr = row % 8;
        int off = tc * (TC_BM * 16) + tr * SBO_CONST + wr * 16;
        *reinterpret_cast<int4 *>(reinterpret_cast<char *>(s_tile_a) + off) =
            *reinterpret_cast<int4 *>(tmp);
      }
      // Load vnew [BT,BV] → tile_b [64,64] bf16 (transposed: row=bv, col=t)
      // MMA computes A @ B^T, so C[t,bv] = sum_j attn[t,j] * B[bv,j] = sum_j attn[t,j] * vnew[j,bv]
      constexpr int attn_b_total = TC_BN * (ATTN_BK / 8);  // 64*8=512
      for (int i = tid; i < attn_b_total; i += 128) {
        int row = i / (ATTN_BK / 8);  // bv (0..63, padded)
        int col8 = (i % (ATTN_BK / 8)) * 8;  // t_base
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

      // MMA: attn[128,64] @ vnew_T[64,64]^T → C[128,64]
      tcgen05_fence();
      if (warp_id == 0 && elect_sync()) {
        for (int ki = 0; ki < ATTN_BK / MMA_K; ki++) {
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

      // Overlap: reload k via TMA (needed for state update step 5)
      tma_load_tile_3d(tileB_smem, &k_tmap_64, (int)cstart, k_col_group,
                       mbar_tma, cp_tileB, warp_id);

      if (warp_id == 0 && elect_sync()) mbarrier_init(mbar_mma, 1);
      tcgen05_fence();

      // Read TMEM → scale and add to output
      for (int n = 0; n < kBV / 8; n++) {
        float tmp[8];
        tcgen05_ld<SHAPE::_32x32b, 8>(tmp, warp_id * 32, n * 8);
        tcgen05_wait_ld();
        int my_row = warp_id * 32 + (tid % 32);
        if (my_row < clen) {
          #pragma unroll
          for (int c = 0; c < 8; c++) {
            int idx = (cstart + my_row) * kHv * kV + hv * kV + v_start + n * 8 + c;
            float existing = __bfloat162float(output[idx]);
            output[idx] = __float2bfloat16_rn(existing + scale * tmp[c]);
          }
        }
      }

      // Wait for k TMA reload to complete before state update
      mbarrier_wait(mbar_tma, tma_phase);
      tma_phase ^= 1;
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
        tcgen05_ld<SHAPE::_32x32b, 8>(tmp, warp_id * 32, n * 8);
        tcgen05_wait_ld();
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
// Kernel H: HRecurrenceKernel — Sequential state propagation (tcgen05, BV=32)
// Grid: (kNVT_H=4, num_seqs * kHv), Block: 128
// Per-chunk: store h -> d_h, TMA w + tcgen05 w@h^T, v_new = u - wh,
//            store v_new -> d_u, gate v_new, scale h, TMA k + tcgen05 k^T@vnew -> update h
// Uses tcgen05 TMEM + TMA (same layout as FusedRecurrenceOutput)
//
// Shared memory layout (~85KB, same as FusedRecurrenceOutput):
//   s_h[BV_H][K+1] fp32 padded = 16.5KB (persistent state)
//   s_tile_a[128*128] bf16 = 32KB (A operand, for w tile)
//   s_tile_b[64*128] bf16 = 16KB (B operand, for h/k tiles)
//   mbar_tma(8) + mbar_mma(8) + tmem_buf(4) = 20 bytes
//   s_gc[64] = 256B
//   s_wh[64*64] fp32 = 16KB (tcgen05 result)
//   s_vnew[64*32] bf16 = 4KB
// ═══════════════════════════════════════════════════════════════════
__global__ void __launch_bounds__(128, 1)
HRecurrenceKernel(
    const __grid_constant__ CUtensorMap w_tmap_128,
    const __grid_constant__ CUtensorMap k_tmap_64,
    const __nv_bfloat16 *__restrict__ k_in,           // k: [T, Hk*K] bf16 (unused, kept for compat)
    __nv_bfloat16 *__restrict__ u_inout,              // d_u: [T, Hv*V] bf16 (overwritten with v_new)
    const float *__restrict__ g_cumsum,
    const float *__restrict__ state0,
    const int64_t *__restrict__ cu_seqlens,
    const int64_t *__restrict__ chunk_offsets,
    __nv_bfloat16 *__restrict__ d_h,                  // scratch: [total_chunks * kHv * kV * kK] bf16
    float *__restrict__ new_state,
    int64_t num_seqs) {

  const int v_tile = blockIdx.x;  // 0..kNVT_H-1 (4 tiles of BV=32)
  const int bh = blockIdx.y;
  const int seq_idx = bh / kHv, hv = bh % kHv;
  if (seq_idx >= num_seqs) return;

  const int64_t k_head = hv / (kHv / kHk);
  const int64_t s0 = cu_seqlens[seq_idx], s1 = cu_seqlens[seq_idx + 1];
  const int NT = (int)((s1 - s0 + kBT - 1) / kBT);
  const int64_t chunk_base = chunk_offsets[seq_idx];

  const int warp_id = threadIdx.x / 32;
  const int tid = threadIdx.x;
  const int v_start = v_tile * kBV_H;

  constexpr int TC_BM = 128, TC_BN = 64, TC_BK = kK;  // 128, 64, 128

  // Shared memory layout (same as FusedRecurrenceOutput)
  constexpr int s_h_bytes = ((kBV_H * (kK + 1) * (int)sizeof(float) + 1023) / 1024) * 1024;

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

  const uint32_t tileA_smem = __cvta_generic_to_shared(s_tile_a);
  const uint32_t tileB_smem = __cvta_generic_to_shared(s_tile_b);
  const uint32_t mbar_tma = __cvta_generic_to_shared(&mbars[0]);
  const uint32_t mbar_mma = __cvta_generic_to_shared(&mbars[1]);

  // Initialize state h
  if (state0) {
    const float *h0 = state0 + ((int64_t)seq_idx * kHv + hv) * kV * kK;
    for (int bv = 0; bv < kBV_H; bv++)
      for (int k = tid; k < kK; k += 128)
        s_h[bv][k] = h0[(v_start + bv) * kK + k];
  } else {
    for (int bv = 0; bv < kBV_H; bv++)
      for (int k = tid; k < kK; k += 128)
        s_h[bv][k] = 0.0f;
  }
  __syncthreads();

  // Init mbarriers + alloc TMEM once
  if (warp_id == 0 && elect_sync()) {
    mbarrier_init(mbar_tma, 1);
    mbarrier_init(mbar_mma, 1);
    asm volatile("fence.mbarrier_init.release.cluster;");
  } else if (warp_id == 1) {
    tcgen05_alloc(__cvta_generic_to_shared(tmem_buf), TC_BN);
  }
  __syncthreads();
  const int taddr = tmem_buf[0];
  constexpr uint32_t idesc = make_tcgen05_idesc<TC_BM, TC_BN>();

  // TMA mbarrier phase tracking
  int tma_phase = 0;

  // Precompute TMA col_group offsets
  const int w_col_group = (int)(hv * kK / 8);
  const int k_col_group = (int)(k_head * kK / 8);

  constexpr int cp_tileA = TC_BM * TC_BK * (int)sizeof(__nv_bfloat16);  // 32KB
  constexpr int cp_tileB = TC_BN * TC_BK * (int)sizeof(__nv_bfloat16);  // 16KB

  // Prefetch w for first chunk via TMA
  {
    const int64_t cstart_0 = s0;
    tma_load_tile_3d(tileA_smem, &w_tmap_128, (int)cstart_0, w_col_group,
                     mbar_tma, cp_tileA, warp_id);
    mbarrier_wait(mbar_tma, tma_phase);
    tma_phase ^= 1;
    __syncthreads();
  }


  for (int ct = 0; ct < NT; ct++) {
    const int64_t cstart = s0 + (int64_t)ct * kBT;
    const int clen = min(kBT, (int)(s1 - cstart));
    const int64_t chunk_id = chunk_base + ct;

    // Step 0: Store current h to d_h[chunk_id] for o-kernel
    {
      __nv_bfloat16 *h_dst = d_h + (chunk_id * kHv + hv) * kV * kK;
      for (int bv = 0; bv < kBV_H; bv++)
        for (int kk = tid; kk < kK; kk += 128)
          h_dst[(v_start + bv) * kK + kk] = __float2bfloat16(s_h[bv][kk]);
    }

    // Step 1: Compute w@h^T via tcgen05
    // w already in tile_a from TMA prefetch, load h into tile_b
    load_fp32_to_tile(s_tile_b, TC_BN, TC_BK, &s_h[0][0], kK + 1, kBV_H, tid);
    __syncthreads();

    // MMA: w @ h^T
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
      if (warp_id == 0 && elect_sync())
        mbarrier_init(mbar_mma, 1);
      __syncthreads();
      tcgen05_fence();
    }

    // Read w@h^T TMEM -> s_wh; overlap with g_cumsum load
    for (int n = 0; n < TC_BN / 8; n++) {
      float tmp[8];
      tcgen05_ld<SHAPE::_32x32b, 8>(tmp, warp_id * 32, n * 8);
      tcgen05_wait_ld();
      int my_row = warp_id * 32 + (tid % 32);
      if (my_row < kBT)
        for (int c = 0; c < 8; c++)
          s_wh[my_row * TC_BN + n * 8 + c] = tmp[c];
    }
    if (tid < kBT)
      s_gc[tid] = (tid < clen) ? g_cumsum[(cstart + tid) * kHv + hv] : 0.0f;
    __syncthreads();

    // Step 2: v_new = u - w@h^T, store v_new to d_u, gate v_new, scale h
    // OVERLAP: Issue TMA k -> tile_b (needed for step 5)
    tma_load_tile_3d(tileB_smem, &k_tmap_64, (int)cstart, k_col_group,
                     mbar_tma, cp_tileB, warp_id);

    {
      float g_last = s_gc[clen - 1];

      for (int i = tid; i < kBT * kBV_H; i += 128) {
        int t = i / kBV_H, bv = i % kBV_H;
        __nv_bfloat16 vnew_bf = __float2bfloat16(0.0f);
        if (t < clen) {
          float uv = __bfloat162float(
              u_inout[(cstart + t) * kHv * kV + hv * kV + v_start + bv]);
          float vnew_val = uv - s_wh[t * TC_BN + bv];
          // Store v_new as bf16 (for o-kernel)
          __nv_bfloat16 vnew_stored = __float2bfloat16(vnew_val);
          u_inout[(cstart + t) * kHv * kV + hv * kV + v_start + bv] = vnew_stored;
          // Gate from bf16-truncated value (matches original precision)
          float gated = __bfloat162float(vnew_stored) * expf(g_last - s_gc[t]);
          vnew_bf = __float2bfloat16(gated);
        }
        s_vnew[i] = vnew_bf;
      }

      // Scale h: h *= exp(g_last)
      float g_exp = expf(g_last);
      for (int bv = 0; bv < kBV_H; bv++)
        for (int kk = tid; kk < kK; kk += 128)
          s_h[bv][kk] *= g_exp;
    }

    // Wait for TMA k to complete
    mbarrier_wait(mbar_tma, tma_phase);
    tma_phase ^= 1;
    __syncthreads();

    // Step 5: Update state h += k^T @ v_new_gated via tcgen05
    {
      constexpr int BK_kv = kBT;
      // Load k^T into tile_a [128,64] -- transpose from tile_b (k from TMA)
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
      // Transpose gated vnew [BT,BV_H]->[BV_H,BT] padded to [64,64] in tile_b
      const int total_b = TC_BN * (BK_kv / 8);
      for (int i = tid; i < total_b; i += 128) {
        int row = i / (BK_kv / 8), col8 = (i % (BK_kv / 8)) * 8;
        __nv_bfloat16 tmp[8];
        if (row < kBV_H) {
          #pragma unroll
          for (int j = 0; j < 8; j++) {
            int t = col8 + j;
            tmp[j] = (t < clen) ? s_vnew[t * kBV_H + row] : __float2bfloat16(0.0f);
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

      // MMA: k^T @ vnew_gated -> TMEM [128,64], 4 K-tiles
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

      // Prefetch w for NEXT chunk (overlapped with TMEM read)
      if (ct + 1 < NT) {
        const int64_t next_cstart = s0 + (int64_t)(ct + 1) * kBT;
        tma_load_tile_3d(tileA_smem, &w_tmap_128, (int)next_cstart, w_col_group,
                         mbar_tma, cp_tileA, warp_id);
      }

      // Read TMEM -> add to s_h[bv][col_k]
      for (int n = 0; n < TC_BN / 8; n++) {
        float tmp[8];
        tcgen05_ld<SHAPE::_32x32b, 8>(tmp, warp_id * 32, n * 8);
        tcgen05_wait_ld();
        int col_k = warp_id * 32 + (tid % 32);
        for (int c = 0; c < 8; c++) {
          int bv = n * 8 + c;
          if (bv < kBV_H)
            s_h[bv][col_k] += tmp[c];
        }
      }

      // Wait for w TMA prefetch to complete before next iteration
      if (ct + 1 < NT) {
        mbarrier_wait(mbar_tma, tma_phase);
        tma_phase ^= 1;
      }
      __syncthreads();
    }
  } // end chunk loop

  // Dealloc TMEM
  __syncthreads();
  if (warp_id == 0) tcgen05_dealloc(taddr, TC_BN);

  // Store final state
  float *ns = new_state + ((int64_t)seq_idx * kHv + hv) * kV * kK;
  for (int bv = 0; bv < kBV_H; bv++)
    for (int kk = tid; kk < kK; kk += 128)
      ns[(v_start + bv) * kK + kk] = s_h[bv][kk];
}

// ═══════════════════════════════════════════════════════════════════
// Kernel O: OOutputKernel — Parallel output computation (wmma, BV=64)
// Grid: (kNVT_O, total_chunks, kHv), Block: 128 (4 warps)
// Uses nvcuda::wmma instead of tcgen05 — NO TMEM, deterministic output.
// Reads precomputed h from d_h, v_new from d_u
// ═══════════════════════════════════════════════════════════════════
__global__ void __launch_bounds__(128, 1)
OOutputKernel(
    const __nv_bfloat16 *__restrict__ q_in,
    const __nv_bfloat16 *__restrict__ k_in,
    const __nv_bfloat16 *__restrict__ u_in,       // d_u (contains v_new after h-kernel)
    const __nv_bfloat16 *__restrict__ d_h,          // precomputed h: [total_chunks * kHv * kV * kK] bf16
    const float *__restrict__ g_cumsum,
    const int64_t *__restrict__ cu_seqlens,
    const int32_t *__restrict__ chunk_indices,
    float scale,
    __nv_bfloat16 *__restrict__ output,
    int64_t total_chunks) {

  using namespace nvcuda;

  const int v_tile = blockIdx.x;
  const int hv = blockIdx.z;

  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int v_start = v_tile * kBV_O;

  const int chunk_id = blockIdx.y;
  if (chunk_id >= total_chunks) return;

  // ── Shared memory layout (row-major bf16, NO tile layout) ──
  // s_A[64*128] bf16 = 16KB   (q data, then reused for attn bf16)
  // s_B[64*128] bf16 = 16KB   (h data, then k data, then vnew)
  // s_wh[64*64] fp32 = 16KB   (MMA result / workspace)
  // s_qh[64*64] fp32 = 16KB   (scaled q@h^T * exp(g), kept until end)
  // s_gc[64] fp32 = 256B
  // Total ≈ 64KB

  extern __shared__ __align__(1024) char smem[];
  __nv_bfloat16 *s_A = reinterpret_cast<__nv_bfloat16 *>(smem);                        // [64, 128]
  __nv_bfloat16 *s_B = reinterpret_cast<__nv_bfloat16 *>(smem + kBT * kK * 2);        // [64, 128]
  float *s_wh = reinterpret_cast<float *>(smem + kBT * kK * 2 + kBT * kK * 2);        // [64, 64]
  float *s_qh = s_wh + kBT * kBT;                                                       // [64, 64]
  float *s_gc = s_qh + kBT * kBV_O;                                                     // [64]

  // ── Chunk metadata ──
  const int seq_idx = chunk_indices[chunk_id * 2];
  const int local_chunk = chunk_indices[chunk_id * 2 + 1];
  const int64_t s0 = cu_seqlens[seq_idx], s1 = cu_seqlens[seq_idx + 1];
  const int64_t cstart = s0 + (int64_t)local_chunk * kBT;
  const int clen = min(kBT, (int)(s1 - cstart));
  const int64_t q_head = hv / (kHv / kHq);
  const int64_t k_head = hv / (kHv / kHk);

  // ── Load g_cumsum ──
  if (tid < kBT)
    s_gc[tid] = (tid < clen) ? g_cumsum[(cstart + tid) * kHv + hv] : 0.0f;

  // ════════════════════════════════════════════════
  // Step 1: Load q → s_A, h → s_B, compute q@h^T via wmma
  // ════════════════════════════════════════════════
  {
    // Load q[64, 128] from global → s_A (row-major, coalesced)
    const __nv_bfloat16 *q_ptr = q_in + cstart * kHq * kK + q_head * kK;
    for (int i = tid; i < kBT * kK; i += 128) {
      int row = i / kK, col = i % kK;
      s_A[i] = (row < clen) ? q_ptr[row * kHq * kK + col] : __float2bfloat16(0.0f);
    }
    // Load h[kBV_O, 128] from d_h → s_B (pad to 64 rows, row-major)
    const __nv_bfloat16 *h_src = d_h + ((int64_t)chunk_id * kHv + hv) * kV * kK + v_start * kK;
    for (int i = tid; i < kBT * kK; i += 128) {
      int row = i / kK, col = i % kK;
      s_B[i] = (row < kBV_O) ? h_src[row * kK + col] : __float2bfloat16(0.0f);
    }
    __syncthreads();

    // wmma: q[64,128] @ h[64,128]^T → s_qh[64,64]
    // Output tiles: m_tiles=4, n_tiles=kBV_O/16=4, total=16, 4 per warp
    constexpr int M_TILES = kBT / 16;        // 4
    constexpr int N_TILES = kBV_O / 16;      // 4
    constexpr int K_TILES = kK / 16;         // 8
    constexpr int TOTAL_TILES = M_TILES * N_TILES;  // 16

    for (int tile_idx = warp_id; tile_idx < TOTAL_TILES; tile_idx += 4) {
      int mt = tile_idx / N_TILES;
      int nt = tile_idx % N_TILES;

      wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
      wmma::fill_fragment(acc, 0.0f);

      for (int kt = 0; kt < K_TILES; kt++) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;

        // a_frag: q[mt*16..mt*16+15, kt*16..kt*16+15], row-major, ld=128
        wmma::load_matrix_sync(a_frag, s_A + mt * 16 * kK + kt * 16, kK);
        // b_frag: h[nt*16..nt*16+15, kt*16..kt*16+15], col-major (= h^T row-major), ld=128
        wmma::load_matrix_sync(b_frag, s_B + nt * 16 * kK + kt * 16, kK);

        wmma::mma_sync(acc, a_frag, b_frag, acc);
      }

      // Store q@h^T result to s_qh (fp32), apply exp(g) scaling
      wmma::store_matrix_sync(s_wh + mt * 16 * kBT + nt * 16, acc, kBT, wmma::mem_row_major);
    }
    __syncthreads();

    // Apply exp(g) scaling: s_qh[t, bv] = s_wh[t, bv] * exp(g[t])
    for (int i = tid; i < kBT * kBV_O; i += 128) {
      int t = i / kBV_O, bv = i % kBV_O;
      float val = (t < clen) ? s_wh[t * kBT + bv] * expf(s_gc[t]) : 0.0f;
      s_qh[t * kBV_O + bv] = val;
    }
    __syncthreads();
  }

  // ════════════════════════════════════════════════
  // Step 2: Load k → s_B, compute q@k^T via wmma (q still in s_A)
  // ════════════════════════════════════════════════
  {
    // Load k[64, 128] from global → s_B (row-major, coalesced)
    const __nv_bfloat16 *k_ptr = k_in + cstart * kHk * kK + k_head * kK;
    for (int i = tid; i < kBT * kK; i += 128) {
      int row = i / kK, col = i % kK;
      s_B[i] = (row < clen) ? k_ptr[row * kHk * kK + col] : __float2bfloat16(0.0f);
    }
    __syncthreads();

    // wmma: q[64,128] @ k[64,128]^T → s_wh[64,64]
    constexpr int M_TILES = kBT / 16;   // 4
    constexpr int N_TILES = kBT / 16;   // 4
    constexpr int K_TILES = kK / 16;    // 8
    constexpr int TOTAL_TILES = M_TILES * N_TILES;  // 16

    for (int tile_idx = warp_id; tile_idx < TOTAL_TILES; tile_idx += 4) {
      int mt = tile_idx / N_TILES;
      int nt = tile_idx % N_TILES;

      wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
      wmma::fill_fragment(acc, 0.0f);

      for (int kt = 0; kt < K_TILES; kt++) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;

        wmma::load_matrix_sync(a_frag, s_A + mt * 16 * kK + kt * 16, kK);
        wmma::load_matrix_sync(b_frag, s_B + nt * 16 * kK + kt * 16, kK);

        wmma::mma_sync(acc, a_frag, b_frag, acc);
      }

      // Store q@k^T result to s_wh (fp32)
      wmma::store_matrix_sync(s_wh + mt * 16 * kBT + nt * 16, acc, kBT, wmma::mem_row_major);
    }
    __syncthreads();

    // Apply gating + causal mask to q@k^T
    for (int i = tid; i < kBT * kBT; i += 128) {
      int row = i / kBT, col = i % kBT;
      float val = 0.0f;
      if (row < clen && col < clen && col <= row)
        val = s_wh[row * kBT + col] * expf(s_gc[row] - s_gc[col]);
      s_wh[row * kBT + col] = val;
    }
    __syncthreads();
  }

  // ════════════════════════════════════════════════
  // Step 3: attn @ v_new via wmma, combine with q@h^T, write output
  // ════════════════════════════════════════════════
  {
    // Convert masked attn [64,64] fp32 → s_A bf16 [64,64] (reuse q buffer)
    for (int i = tid; i < kBT * kBT; i += 128) {
      s_A[i] = __float2bfloat16(s_wh[i]);
    }

    // Load v_new [64, kBV_O] from d_u → s_B (row-major, coalesced)
    // s_B is [64,128] but we only use [64, kBV_O=64]
    for (int i = tid; i < kBT * kBV_O; i += 128) {
      int t = i / kBV_O, bv = i % kBV_O;
      // Store at [t, bv] in s_B with ld=kBV_O
      s_B[t * kBV_O + bv] = (t < clen)
          ? u_in[(cstart + t) * kHv * kV + hv * kV + v_start + bv]
          : __float2bfloat16(0.0f);
    }
    __syncthreads();

    // wmma: attn[64,64] @ vnew[64,kBV_O] → output contrib [64, kBV_O]
    // A is attn (row-major in s_A, ld=kBT=64), B is vnew (row-major in s_B, ld=kBV_O)
    // We want C[m,n] = sum_k attn[m,k] * vnew[k,n]
    // With matrix_a row_major and matrix_b col_major:
    //   C[m,n] = sum_k A[m,k] * B^T[k,n] = sum_k attn[m,k] * vnew[n,k]
    // But vnew is [t, bv] not [bv, t]! So we need matrix_b row_major:
    //   With row_major B: C[m,n] = sum_k A[m,k] * B[k,n] = attn[m,k] * vnew[k,n]  ✓

    constexpr int M_TILES = kBT / 16;          // 4
    constexpr int N_TILES = kBV_O / 16;        // 4
    constexpr int K_TILES_ATTN = kBT / 16;    // 4
    constexpr int TOTAL_TILES = M_TILES * N_TILES;  // 16

    for (int tile_idx = warp_id; tile_idx < TOTAL_TILES; tile_idx += 4) {
      int mt = tile_idx / N_TILES;
      int nt = tile_idx % N_TILES;

      wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
      wmma::fill_fragment(acc, 0.0f);

      for (int kt = 0; kt < K_TILES_ATTN; kt++) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b_frag;

        // attn in s_A[mt*16, kt*16], ld=kBT
        wmma::load_matrix_sync(a_frag, s_A + mt * 16 * kBT + kt * 16, kBT);
        // vnew in s_B[kt*16, nt*16], ld=kBV_O
        wmma::load_matrix_sync(b_frag, s_B + kt * 16 * kBV_O + nt * 16, kBV_O);

        wmma::mma_sync(acc, a_frag, b_frag, acc);
      }

      // Store attn@vnew result to s_wh (fp32, reuse workspace)
      // s_wh used as [64, kBV_O] here (fits in [64,64])
      wmma::store_matrix_sync(s_wh + mt * 16 * kBV_O + nt * 16, acc, kBV_O, wmma::mem_row_major);
    }
    __syncthreads();

    // Combine attn@vnew (s_wh) + q@h^T*exp(g) (s_qh) → write final output
    for (int i = tid; i < kBT * kBV_O; i += 128) {
      int t = i / kBV_O, bv = i % kBV_O;
      if (t < clen) {
        float attn_vnew = s_wh[t * kBV_O + bv];
        float qh_part = s_qh[t * kBV_O + bv];
        float combined = qh_part + attn_vnew;
        int idx = (cstart + t) * kHv * kV + hv * kV + v_start + bv;
        output[idx] = __float2bfloat16_rn(scale * combined);
      }
    }
  }
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

  // Compute scratch size — includes d_h for h/o kernel split
  auto align256 = [](size_t x) -> size_t { return (x + 255) & ~255ULL; };
  const size_t sz_g    = align256(T * kHv * sizeof(float));
  const size_t sz_beta = align256(T * kHv * sizeof(float));
  const size_t sz_A    = align256(T * kHv * kBT * sizeof(float));
  const size_t sz_w    = align256(T * kHv * kK * sizeof(__nv_bfloat16));
  const size_t sz_u    = align256(T * kHv * kV * sizeof(__nv_bfloat16));
  const size_t sz_h    = align256((size_t)max_chunks * kHv * kV * kK * sizeof(__nv_bfloat16));  // d_h for o-kernel (bf16)
  const size_t sz_co   = align256((num_seqs + 2) * sizeof(int64_t));
  const size_t sz_ci   = align256(max_chunks * 2 * sizeof(int32_t));
  const size_t sz_tc   = align256(sizeof(int32_t));
  const size_t total_sz = sz_g + sz_beta + sz_A + sz_w + sz_u + sz_h + sz_co + sz_ci + sz_tc;

  // Grow persistent scratch if needed (never shrinks)
  if (total_sz > g_scratch_size) {
    if (g_scratch) cudaFree(g_scratch);
    cudaError_t me = cudaMalloc(&g_scratch, total_sz);
    if (me != cudaSuccess) {
      TVM_FFI_THROW(RuntimeError) << "scratch cudaMalloc(" << total_sz << ") failed: " << cudaGetErrorString(me);
    }
    g_scratch_size = total_sz;
  }

  // Partition scratch — includes d_h for h/o kernel split
  char *p = g_scratch;
  float *d_g = (float *)p; p += sz_g;
  float *d_beta = (float *)p; p += sz_beta;
  float *d_A = (float *)p; p += sz_A;
  __nv_bfloat16 *d_w = (__nv_bfloat16 *)p; p += sz_w;
  __nv_bfloat16 *d_u = (__nv_bfloat16 *)p; p += sz_u;
  __nv_bfloat16 *d_h = (__nv_bfloat16 *)p; p += sz_h;    // h state per chunk for o-kernel (bf16)
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
    // FusedPrepKernel: s_A_tiles(32KB) + s_B_tiles(16KB) + mbars(20B) + s_result(16KB) + s_result_inv(16KB) + s_g(256B) + s_beta(256B) + s_bg(256B) ≈ 81KB
    int smem_FP = (128*128*2 + 64*128*2 + 2*8+4 + 64*64*4 + 64*64*4 + 64*4 + 64*4 + 64*4 + 1023) & ~1023;
    cudaFuncSetAttribute(FusedPrepKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_FP);
    // FusedRecurrenceOutput (kept as dead code, but set attr for completeness)
    constexpr int s_h_b_fro = ((kBV*(kK+1)*4 + 1023)/1024)*1024;
    int smem_FRO = (s_h_b_fro + 128*128*2 + 64*128*2 + 2*8+4 + kBT*4 + kBT*64*4 + kBT*kBV*2 + 1023) & ~1023;
    cudaFuncSetAttribute(FusedRecurrenceOutput, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_FRO);
    // HRecurrenceKernel (tcgen05): s_h(~16.5KB) + tile_a(32KB) + tile_b(16KB) + mbars(20B) + s_gc(256B) + s_wh(16KB) + s_vnew(4KB) ~= 85KB
    constexpr int s_h_b_H = ((kBV_H*(kK+1)*4 + 1023)/1024)*1024;
    int smem_H = (s_h_b_H + 128*128*2 + 64*128*2 + 2*8+4 + kBT*4 + kBT*64*4 + kBT*kBV_H*2 + 1023) & ~1023;
    cudaFuncSetAttribute(HRecurrenceKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_H);
    // OOutputKernel (wmma): s_A(16KB) + s_B(16KB) + s_wh(16KB) + s_qh(16KB) + s_gc(256B) ≈ 64KB
    int smem_O = (kBT*kK*2 + kBT*kK*2 + kBT*kBT*4 + kBT*kBV_O*4 + kBT*4 + 1023) & ~1023;
    cudaFuncSetAttribute(OOutputKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_O);
  }

  // Launch PrepMeta FIRST, then create TMA descs on CPU while GPU runs
  PrepMetaKernel<<<1, 1, 0, stream>>>(cusl_p, d_ci, d_co, d_tc, num_seqs);
  int32_t h_tc;
  cudaMemcpyAsync(&h_tc, d_tc, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);

  // Create TMA descriptors EVERY call (no caching — prevents stale descriptor issues)
  CUtensorMap k_tmap_128, k_tmap_64, q_tmap_128, w_tmap_128;
  init_tma_desc_3d(&k_tmap_128, k_p, (uint64_t)T, (uint64_t)(kHk * kK), 128, kK);
  init_tma_desc_3d(&k_tmap_64,  k_p, (uint64_t)T, (uint64_t)(kHk * kK), 64,  kK);
  init_tma_desc_3d(&q_tmap_128, q_p, (uint64_t)T, (uint64_t)(kHq * kK), 128, kK);
  init_tma_desc_3d(&w_tmap_128, d_w, (uint64_t)T, (uint64_t)(kHv * kK), 128, kK);

  // Compute smem sizes for launch
  int smem_FP = (128*128*2 + 64*128*2 + 2*8+4 + 64*64*4 + 64*64*4 + 64*4 + 64*4 + 64*4 + 1023) & ~1023;
  // HRecurrenceKernel smem (tcgen05)
  constexpr int s_h_b_H_launch = ((kBV_H*(kK+1)*4 + 1023)/1024)*1024;
  int smem_H = (s_h_b_H_launch + 128*128*2 + 64*128*2 + 2*8+4 + kBT*4 + kBT*64*4 + kBT*kBV_H*2 + 1023) & ~1023;
  // OOutputKernel smem (wmma)
  int smem_O = (kBT*kK*2 + kBT*kK*2 + kBT*kBT*4 + kBT*kBV_O*4 + kBT*4 + 1023) & ~1023;

  // Now sync to get total_chunks
  cudaStreamSynchronize(stream);
  const int64_t total_chunks = h_tc;

  // Clear stale errors from previous calls
  cudaGetLastError();

  // 1. Preprocess
  PreprocessKernel<<<dim3(total_chunks, kHv), kBT, 0, stream>>>(
      a_p, b_p, Alog_p, dtb_p, cusl_p, d_ci, d_g, d_beta, total_chunks);

  // 2+3. FusedPrepKernel — ComputeA + SolveTril + ComputeWU in one kernel
  // Eliminates 2 global memory round-trips for A_mat/A_inv
  FusedPrepKernel<<<dim3(total_chunks, kHv), 128, smem_FP, stream>>>(
      k_tmap_128, k_tmap_64, k_p, v_p, d_g, d_beta, cusl_p, d_ci,
      d_w, d_u, total_chunks);

  // H/O split path: h-kernel (tcgen05) computes recurrence, o-kernel (wmma) computes output
  // OKernel now uses wmma (no TMEM) — fully deterministic, no block count limit
  {
    // Zero d_h buffer to prevent stale data from previous calls
    cudaMemsetAsync(d_h, 0, sz_h, stream);
    HRecurrenceKernel<<<dim3(kNVT_H, num_seqs * kHv), 128, smem_H, stream>>>(
        w_tmap_128, k_tmap_64, k_p,
        d_u, d_g, state_ptr, cusl_p, d_co,
        d_h, ns_p, num_seqs);

    OOutputKernel<<<dim3(kNVT_O, total_chunks, kHv), 128, smem_O, stream>>>(
        q_p, k_p,
        d_u, d_h, d_g, cusl_p, d_ci,
        scale_f, out_p, total_chunks);
  }

  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    TVM_FFI_THROW(RuntimeError)
        << "GdnPrefillTcgen05 failed: " << cudaGetErrorString(err);
}

} // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(gdn_prefill_tcgen05, RunGdnPrefillTcgen05);
