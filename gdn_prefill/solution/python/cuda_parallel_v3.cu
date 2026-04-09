/*
 * GDN Prefill v3 — Optimized CUDA Kernel for Blackwell SM_100a (B200)
 *
 * Faster than chunk_v5 (CUDA kkt + Triton) for T < 1024 workloads due to:
 * - Kernel fusion: k@k^T + block-inverse + W/U in single FusedPrepKernel
 *   (no global memory round-trips for intermediate A, Ai, g_cu, beta tensors)
 * - tcgen05 tensor cores for k@k^T (Phase 1) and O-kernel MMAs
 * - Single pre-allocated scratch buffer (no per-call torch.empty allocations)
 * - No CPU-GPU sync (chunk mapping inlined in GPU kernels)
 * - 3 kernel launches vs chunk_v5's 5 + sync
 *
 * Pipeline:
 * 1. FusedPrepKernel (128 threads, tcgen05+wmma+mma.sync):
 *    Phase 1: TMA k load → tcgen05 k@k^T → midpoint-normalized gating → A matrix
 *    Phase 2: Block-recursive (I+A)^{-1} via Neumann series (tf32 wmma)
 *    Phase 3: W = A_inv @ (beta*exp(g)*k), U = A_inv @ (beta*v) via mma.sync
 * 2. HRecurrenceKernel (128 threads, mma.sync):
 *    Sequential state recurrence h per sequence, BV=16
 * 3. OOutputKernel (256 threads, tcgen05 swizzled BM=64/BN=64):
 *    q@k^T → gating → q@h^T → exp(g) scaling → attn@vnew → output
 *
 */

 #include "cuda_utils.h"

 #include <cstdint>
 #include <math.h>
 #include <mma.h>
 #include <vector>


 namespace {

 // tcgen05 TMEM ld/st for 16x32bx2 tile shape (O-kernel TMEM access)
 // addr = pre-composed TMEM address: taddr + tmem_offset
 __device__ __forceinline__
 void tcgen05_ld_16x32bx2(uint32_t (&out)[16], int addr) {
   asm volatile(
     "tcgen05.ld.sync.aligned.16x32bx2.x16.b32 "
     "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16 + 0], 16;"
     : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3]),
       "=r"(out[4]), "=r"(out[5]), "=r"(out[6]), "=r"(out[7]),
       "=r"(out[8]), "=r"(out[9]), "=r"(out[10]), "=r"(out[11]),
       "=r"(out[12]), "=r"(out[13]), "=r"(out[14]), "=r"(out[15])
     : "r"(addr));
   asm volatile("tcgen05.wait::ld.sync.aligned;");
 }

 __device__ __forceinline__
 void tcgen05_st_16x32bx2(int addr, const uint32_t (&in)[16]) {
   asm volatile(
     "{\n\t.reg .pred p;\n\tmov.pred p, -1;\n\t"
     "@p tcgen05.st.sync.aligned.16x32bx2.x16.b32 [%0 + 0], 16, "
     "{%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};\n\t}"
     :: "r"(addr),
        "r"(in[0]), "r"(in[1]), "r"(in[2]), "r"(in[3]),
        "r"(in[4]), "r"(in[5]), "r"(in[6]), "r"(in[7]),
        "r"(in[8]), "r"(in[9]), "r"(in[10]), "r"(in[11]),
        "r"(in[12]), "r"(in[13]), "r"(in[14]), "r"(in[15])
     : "memory");
   asm volatile("tcgen05.wait::st.sync.aligned;");
 }
 
 constexpr int kK = 128, kV = 128;
 constexpr int64_t kHq = 4, kHk = 4, kHv = 8;
 constexpr int kBT = 64, kBV = 32;
 constexpr int kNVT = kV / kBV;   // 4
 constexpr int MMA_K = 16;
 constexpr int SBO_CONST = 128;  // 8 rows * 16 bytes
 
 // H/O kernel split constants
 constexpr int kBV_H = 16;              // BV for h-kernel (tcgen05-based)
 constexpr int kBV_O = 64;              // BV for o-kernel (tcgen05-based)
 constexpr int kNVT_H = kV / kBV_H;    // 8
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
   return log1pf(__expf(-fabsf(x))) + fmaxf(x, 0.0f);
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
 
 
 // Dead kernels removed — all fused into FusedPrepKernel
 
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
     const __nv_bfloat16 *__restrict__ a_in,
     const __nv_bfloat16 *__restrict__ b_in,
     const float *__restrict__ A_log,
     const float *__restrict__ dt_bias,
     float *__restrict__ g_cumsum_out,
     const int64_t *__restrict__ cu_seqlens,
     const int32_t *__restrict__ chunk_indices,
     __nv_bfloat16 *__restrict__ w_out,
     __nv_bfloat16 *__restrict__ u_out,
     const int32_t *__restrict__ total_chunks_ptr,
     int64_t max_chunks,
     int64_t num_seqs) {
 
   const int chunk_id = blockIdx.x, hv = blockIdx.y;
   if (chunk_id >= max_chunks) return;
 
   // Inline chunk_id → (seq_idx, local_chunk) mapping (replaces PrepMeta lookup)
   int seq_idx = -1, local_chunk = -1;
   {
     int running = 0;
     for (int i = 0; i < num_seqs; i++) {
       int64_t slen = cu_seqlens[i + 1] - cu_seqlens[i];
       int nc = (int)((slen + kBT - 1) / kBT);
       if (chunk_id < running + nc) {
         seq_idx = i;
         local_chunk = chunk_id - running;
         break;
       }
       running += nc;
     }
     if (seq_idx < 0) return;  // excess block beyond actual total_chunks
   }
   const int64_t s0 = cu_seqlens[seq_idx], s1 = cu_seqlens[seq_idx + 1];
   const int64_t cstart = s0 + (int64_t)local_chunk * kBT;
   const int clen = min(kBT, (int)(s1 - cstart));
   const int64_t k_head = hv / (kHv / kHk);
   const int64_t q_head = hv / (kHv / kHq);
 
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
   // Align s_result to 16 bytes for cp.async staging in Phase 3
   float *s_result = reinterpret_cast<float *>(
       (reinterpret_cast<uintptr_t>(tmem_buf + 1) + 15) & ~15ULL);  // [64*64] = 16KB
   float *s_result_inv = s_result + kBT * kBT;                      // [64*64] = 16KB
   float *s_g = s_result_inv + kBT * kBT;                           // [64]
   float *s_beta = s_g + kBT;                                       // [64]
   float *s_bg = s_beta + kBT;                                      // [64]
 
   const uint32_t A_smem_base = cvt_smem_ptr(s_A_tiles);
   const uint32_t B_smem_base = cvt_smem_ptr(s_B_tiles);
   const uint32_t mbar_tma = cvt_smem_ptr(&mbars[0]);
   const uint32_t mbar_mma = cvt_smem_ptr(&mbars[1]);
 
   // Init mbarriers + alloc TMEM
   if (warp_id == 0 && elect_sync()) {
     mbarrier_init(mbar_tma, 1);
     mbarrier_init(mbar_mma, 1);
     asm volatile("fence.mbarrier_init.release.cluster;");
   } else if (warp_id == 1) {
     tcgen05_alloc(cvt_smem_ptr(tmem_buf), BN);
   }
   __syncthreads();
   const int taddr = tmem_buf[0];
   constexpr uint32_t idesc = make_tcgen05_idesc(BM, BN);
 
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
 
   // Wait for TMA load (mbarrier acquire provides visibility for tcgen05)
   mbarrier_wait(mbar_tma, 0);
 
   // MMA: batch all 8 K-tiles, single commit
   tcgen05_fence_after_thread_sync();
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
   tcgen05_fence_after_thread_sync();
 
   // Overlap: compute g/beta from global memory WHILE reading TMEM → s_result
   // g/beta writes to s_g/s_beta (independent of s_result)
   {
     float g_val = 0.0f, beta_val = 0.0f;
     if (tid < clen) {
       float x = __bfloat162float(a_in[(cstart + tid) * kHv + hv]) + dt_bias[hv];
       g_val = -__expf(A_log[hv]) * softplus_s(x);
       beta_val = 1.0f / (1.0f + __expf(-__bfloat162float(b_in[(cstart + tid) * kHv + hv])));
     }
     // Write g/beta to smem before TMEM read (hides global load latency)
     if (tid < kBT) {
       s_g[tid] = g_val;
       s_beta[tid] = beta_val;
     }
   }
 
   // Read TMEM → s_result [64,64] (padded), only first 64 rows of 128x64 output
   for (int n = 0; n < BN / 8; n++) {
     float tmp[8];
     {
      uint32_t addr = taddr + ((warp_id * 32) << 16) + n * 8;
      asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
        : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3]),
          "=f"(tmp[4]), "=f"(tmp[5]), "=f"(tmp[6]), "=f"(tmp[7])
        : "r"(addr));
      tcgen05_wait_ld();
    }
     int my_row = warp_id * 32 + (tid % 32);
     if (my_row < kBT) {
       for (int c = 0; c < 8; c++)
         s_result[my_row * kBT + n * 8 + c] = tmp[c];
     }
   }
   // Dealloc TMEM immediately — not needed for Phase 2 (wmma) or Phase 3 (mma.sync)
   if (warp_id == 0) tcgen05_dealloc(taddr, BN);
   __syncthreads();  // Ensures both s_g/s_beta and s_result are visible
 
   // Prefix scan for g_cumsum (sequential, single-threaded)
   if (tid == 0) {
     for (int i = 1; i < clen; i++) s_g[i] += s_g[i-1];
   }
   __syncthreads();
   if (tid < clen)
     g_cumsum_out[(cstart + tid) * kHv + hv] = s_g[tid];
 
   // Apply beta * exp(g_diff) * strictly-lower-triangular mask IN PLACE on s_result
   // Precompute: beta_expg[row] = beta[row]*exp(g[row]) stored in first 64 floats of s_result_inv
   //             neg_expg[col] = exp(-g[col]) stored in s_bg
   // Inner loop: NO per-element expf (eliminates ~4096 expf → 128)
   {
     float *s_beta_expg = s_result_inv;  // reuse first 64 floats
     float g_mid = (clen > 1) ? s_g[clen >> 1] : ((clen > 0) ? s_g[0] : 0.0f);
     if (tid < kBT) {
       s_bg[tid] = (tid < clen) ? __expf(g_mid - s_g[tid]) : 0.0f;
       s_beta_expg[tid] = (tid < clen) ? s_beta[tid] * __expf(s_g[tid] - g_mid) : 0.0f;
     }
     __syncthreads();
     for (int i = tid; i < kBT * kBT; i += 128) {
       int row = i >> 6, col = i & 63;
       s_result[i] = (col < row && row < clen && col < clen) ? s_beta_expg[row] * s_result[i] * s_bg[col] : 0.0f;
     }
   }
   __syncthreads();
 
   // Keep TMEM allocated through Phase 2+3 (used for Phase 1, deallocated after Phase 3)
   // Safe with launch_bounds(128, 1) — only 1 block per SM, no TMEM pressure
 
   // ════════════════════════════════════════════════════════════
   // Phase 2: SolveTril — block-recursive (I+A)^{-1} via Neumann series
   // s_result has A matrix [64*64], s_result_inv will have A_inv [64*64]
   // Split 64x64 into 4x4 grid of 16x16 blocks.
   // Step 1: Invert diagonal blocks with tf32 wmma (parallel across 4 warps)
   // Step 2: Off-diagonal blocks via Schur complement (scalar fp32 matmul)
   // ════════════════════════════════════════════════════════════
 
   // Initialize s_result_inv to zero — vectorized with int4 (4 floats per store)
   {
     int4 *dst = reinterpret_cast<int4 *>(s_result_inv);
     int4 zero = make_int4(0, 0, 0, 0);
     for (int idx = tid; idx < kBT * kBT / 4; idx += 128)
       dst[idx] = zero;
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
   // Early exit for small chunks: skip unnecessary off-diagonal levels
   if (clen <= 16) goto phase3;  // 1 block → no off-diagonal work
   // Uses wmma tf32 16x16 matmul with warp-level parallelism.
   // Dependency DAG:
   //   Level 0 (parallel): Ai_10, Ai_21, Ai_32
   //   Level 1 (parallel): Ai_20, Ai_31
   //   Level 2: Ai_30
   //
   // Each Ai_ij = -Ai_ii @ (sum_k A_ik @ Ai_kj)
   // Per-warp temp buffer in s_A_tiles area for intermediate results.
   {
     using namespace nvcuda::wmma;
     constexpr int BS = 16;
     constexpr int S = kBT;  // stride=64 in s_result/s_result_inv
     const int lane = tid % 32;
 
     // Per-warp temp: 4 warps × [16][16] fp32 = 4KB total, fits in s_A_tiles (32KB)
     float *my_tmp = reinterpret_cast<float *>(s_A_tiles) + warp_id * 256;  // [16][16] stride 16
 
     // Helper macros as inline code: wmma 16x16 matmul with stride-64 source
     // C[16,16] = A_blk[bi,bk] @ B_blk[bk,bj]  (both in stride-S arrays)
     // Result stored to my_tmp[16,16] stride 16
 
     // Level 0: Ai_10, Ai_21, Ai_32 in parallel (warps 0,1,2)
     if (warp_id < 3) {
       // Warp 0: Ai_10 = -Ai_11 @ A_10 @ Ai_00  (bi=1,bj=0, via bk=0)
       // Warp 1: Ai_21 = -Ai_22 @ A_21 @ Ai_11  (bi=2,bj=1, via bk=1)
       // Warp 2: Ai_32 = -Ai_33 @ A_32 @ Ai_22  (bi=3,bj=2, via bk=2)
       const int bi = warp_id + 1;  // 1,2,3
       const int bj = warp_id;      // 0,1,2
       const int bk = warp_id;      // 0,1,2 (single term)
 
       // Step a: my_tmp = A[bi,bk] @ Ai[bk,bj]
       {
         fragment<accumulator, 16, 16, 8, float> acc;
         fill_fragment(acc, 0.0f);
         fragment<matrix_a, 16, 16, 8, precision::tf32, row_major> a_frag;
         fragment<matrix_b, 16, 16, 8, precision::tf32, row_major> b_frag;
         load_matrix_sync(a_frag, &s_result[(bi*BS)*S + bk*BS], S);
         load_matrix_sync(b_frag, &s_result_inv[(bk*BS)*S + bj*BS], S);
         mma_sync(acc, a_frag, b_frag, acc);
         load_matrix_sync(a_frag, &s_result[(bi*BS)*S + bk*BS + 8], S);
         load_matrix_sync(b_frag, &s_result_inv[(bk*BS+8)*S + bj*BS], S);
         mma_sync(acc, a_frag, b_frag, acc);
         store_matrix_sync(my_tmp, acc, BS, mem_row_major);
       }
       __syncwarp();
 
       // Step b: Ai[bi,bj] = -Ai[bi,bi] @ my_tmp
       {
         fragment<accumulator, 16, 16, 8, float> acc;
         fill_fragment(acc, 0.0f);
         fragment<matrix_a, 16, 16, 8, precision::tf32, row_major> a_frag;
         fragment<matrix_b, 16, 16, 8, precision::tf32, row_major> b_frag;
         load_matrix_sync(a_frag, &s_result_inv[(bi*BS)*S + bi*BS], S);
         load_matrix_sync(b_frag, my_tmp, BS);
         mma_sync(acc, a_frag, b_frag, acc);
         load_matrix_sync(a_frag, &s_result_inv[(bi*BS)*S + bi*BS + 8], S);
         load_matrix_sync(b_frag, my_tmp + 8 * BS, BS);
         mma_sync(acc, a_frag, b_frag, acc);
         // Store negated result to s_result_inv
         store_matrix_sync(my_tmp, acc, BS, mem_row_major);
       }
       __syncwarp();
       for (int idx2 = lane; idx2 < BS * BS; idx2 += 32) {
         int r = idx2 / BS, c = idx2 % BS;
         s_result_inv[(bi*BS + r) * S + (bj*BS + c)] = -my_tmp[idx2];
       }
     }
     __syncthreads();
 
     // nblocks = number of 16x16 diagonal blocks needed = ceil(clen/16)
     // Level 1 computes blocks (2,0) and (3,1): need nblocks >= 3
     if (((clen + 15) / 16) <= 2) goto phase2_done;  // 2 blocks → only Level 0 needed
 
     // Level 1: Ai_20, Ai_31 in parallel (warps 0,1)
     if (warp_id < 2) {
       // Warp 0: Ai_20 = -Ai_22 @ (A_20@Ai_00 + A_21@Ai_10)  bi=2,bj=0
       // Warp 1: Ai_31 = -Ai_33 @ (A_31@Ai_11 + A_32@Ai_21)  bi=3,bj=1
       const int bi = warp_id + 2;  // 2,3
       const int bj = warp_id;      // 0,1
 
       // Accumulate sum: A[bi,bj]@Ai[bj,bj] + A[bi,bj+1]@Ai[bj+1,bj]
       {
         fragment<accumulator, 16, 16, 8, float> acc;
         fill_fragment(acc, 0.0f);
         // Term 1: A[bi, bj] @ Ai[bj, bj]
         {
           fragment<matrix_a, 16, 16, 8, precision::tf32, row_major> a_frag;
           fragment<matrix_b, 16, 16, 8, precision::tf32, row_major> b_frag;
           load_matrix_sync(a_frag, &s_result[(bi*BS)*S + bj*BS], S);
           load_matrix_sync(b_frag, &s_result_inv[(bj*BS)*S + bj*BS], S);
           mma_sync(acc, a_frag, b_frag, acc);
           load_matrix_sync(a_frag, &s_result[(bi*BS)*S + bj*BS + 8], S);
           load_matrix_sync(b_frag, &s_result_inv[(bj*BS+8)*S + bj*BS], S);
           mma_sync(acc, a_frag, b_frag, acc);
         }
         // Term 2: A[bi, bj+1] @ Ai[bj+1, bj]
         {
           fragment<matrix_a, 16, 16, 8, precision::tf32, row_major> a_frag;
           fragment<matrix_b, 16, 16, 8, precision::tf32, row_major> b_frag;
           load_matrix_sync(a_frag, &s_result[(bi*BS)*S + (bj+1)*BS], S);
           load_matrix_sync(b_frag, &s_result_inv[((bj+1)*BS)*S + bj*BS], S);
           mma_sync(acc, a_frag, b_frag, acc);
           load_matrix_sync(a_frag, &s_result[(bi*BS)*S + (bj+1)*BS + 8], S);
           load_matrix_sync(b_frag, &s_result_inv[((bj+1)*BS+8)*S + bj*BS], S);
           mma_sync(acc, a_frag, b_frag, acc);
         }
         store_matrix_sync(my_tmp, acc, BS, mem_row_major);
       }
       __syncwarp();
 
       // Ai[bi,bj] = -Ai[bi,bi] @ my_tmp
       {
         fragment<accumulator, 16, 16, 8, float> acc;
         fill_fragment(acc, 0.0f);
         fragment<matrix_a, 16, 16, 8, precision::tf32, row_major> a_frag;
         fragment<matrix_b, 16, 16, 8, precision::tf32, row_major> b_frag;
         load_matrix_sync(a_frag, &s_result_inv[(bi*BS)*S + bi*BS], S);
         load_matrix_sync(b_frag, my_tmp, BS);
         mma_sync(acc, a_frag, b_frag, acc);
         load_matrix_sync(a_frag, &s_result_inv[(bi*BS)*S + bi*BS + 8], S);
         load_matrix_sync(b_frag, my_tmp + 8 * BS, BS);
         mma_sync(acc, a_frag, b_frag, acc);
         store_matrix_sync(my_tmp, acc, BS, mem_row_major);
       }
       __syncwarp();
       for (int idx2 = lane; idx2 < BS * BS; idx2 += 32) {
         int r = idx2 / BS, c = idx2 % BS;
         s_result_inv[(bi*BS + r) * S + (bj*BS + c)] = -my_tmp[idx2];
       }
     }
     __syncthreads();
 
     // Level 2 computes block (3,0): need nblocks >= 4 (full 64x64)
     if (((clen + 15) / 16) <= 3) goto phase2_done;  // 3 blocks → only Level 0+1 needed
 
     // Level 2: Ai_30 = -Ai_33 @ (A_30@Ai_00 + A_31@Ai_10 + A_32@Ai_20)
     // Use warp 0 only (other warps idle)
     if (warp_id == 0) {
       constexpr int bi = 3, bj = 0;
       fragment<accumulator, 16, 16, 8, float> acc;
       fill_fragment(acc, 0.0f);
       // 3 terms: k=0,1,2
       for (int bk = 0; bk < 3; bk++) {
         fragment<matrix_a, 16, 16, 8, precision::tf32, row_major> a_frag;
         fragment<matrix_b, 16, 16, 8, precision::tf32, row_major> b_frag;
         load_matrix_sync(a_frag, &s_result[(bi*BS)*S + bk*BS], S);
         load_matrix_sync(b_frag, &s_result_inv[(bk*BS)*S + bj*BS], S);
         mma_sync(acc, a_frag, b_frag, acc);
         load_matrix_sync(a_frag, &s_result[(bi*BS)*S + bk*BS + 8], S);
         load_matrix_sync(b_frag, &s_result_inv[(bk*BS+8)*S + bj*BS], S);
         mma_sync(acc, a_frag, b_frag, acc);
       }
       store_matrix_sync(my_tmp, acc, BS, mem_row_major);
       __syncwarp();
 
       // Ai_30 = -Ai_33 @ my_tmp
       {
         fragment<accumulator, 16, 16, 8, float> acc2;
         fill_fragment(acc2, 0.0f);
         fragment<matrix_a, 16, 16, 8, precision::tf32, row_major> a_frag;
         fragment<matrix_b, 16, 16, 8, precision::tf32, row_major> b_frag;
         load_matrix_sync(a_frag, &s_result_inv[(bi*BS)*S + bi*BS], S);
         load_matrix_sync(b_frag, my_tmp, BS);
         mma_sync(acc2, a_frag, b_frag, acc2);
         load_matrix_sync(a_frag, &s_result_inv[(bi*BS)*S + bi*BS + 8], S);
         load_matrix_sync(b_frag, my_tmp + 8 * BS, BS);
         mma_sync(acc2, a_frag, b_frag, acc2);
         store_matrix_sync(my_tmp, acc2, BS, mem_row_major);
       }
       __syncwarp();
       for (int idx2 = lane; idx2 < BS * BS; idx2 += 32) {
         int r = idx2 / BS, c = idx2 % BS;
         s_result_inv[(bi*BS + r) * S + (bj*BS + c)] = -my_tmp[idx2];
       }
     }
     __syncthreads();
   }
 
   phase2_done:
   // Handle rows >= clen: set identity on unused portion
   // Only needed when chunk is partial (last chunk of a sequence)
   if (clen < kBT) {
     // Zero rows >= clen (except diagonal) and set diagonal to 1
     for (int idx2 = tid; idx2 < kBT * kBT; idx2 += 128) {
       int row = idx2 >> 6, col = idx2 & 63;
       if (row >= clen || col >= clen)
         s_result_inv[idx2] = (row == col) ? 1.0f : 0.0f;
     }
     __syncthreads();
   }
 
   phase3:
   // ════════════════════════════════════════════════════════════
   // Phase 3: Compute W and U via mma.sync (no tcgen05/TMEM)
   // A_inv[64,64] @ input[64,64] for 4 tiles (w_col0, w_col1, u_col0, u_col1)
   // A = A_inv, row-major bf16 [64,64] in s_A_tiles (converted once)
   // B = scaled_input[t, j], row-major bf16 [64,64] in s_B_tiles
   // B is already in [K_mma=t, N=j] format → no transpose needed for ldmatrix.x2.trans!
   // ════════════════════════════════════════════════════════════
 
   // Reuse s_result as staging buffer for k/v data (16KB)
   __nv_bfloat16 *s_stage = reinterpret_cast<__nv_bfloat16 *>(s_result);
 
   // Convert A_inv fp32 → bf16 row-major in s_A_tiles[64,64] (first 8KB of 32KB buffer)
   // Vectorized: convert 8 fp32 → 4 bf16x2 → 1 int4 store
   __nv_bfloat16 *s_Ainv_bf16 = reinterpret_cast<__nv_bfloat16 *>(s_A_tiles);  // [64,64] bf16
   for (int i = tid; i < kBT * kBT / 8; i += 128) {
     int idx = i * 8;
     const float *src = &s_result_inv[idx];
     int4 dst;
     __nv_bfloat162 *dp = reinterpret_cast<__nv_bfloat162 *>(&dst);
     dp[0] = {__float2bfloat16(src[0]), __float2bfloat16(src[1])};
     dp[1] = {__float2bfloat16(src[2]), __float2bfloat16(src[3])};
     dp[2] = {__float2bfloat16(src[4]), __float2bfloat16(src[5])};
     dp[3] = {__float2bfloat16(src[6]), __float2bfloat16(src[7])};
     *reinterpret_cast<int4 *>(&s_Ainv_bf16[idx]) = dst;
   }
 
   // cp.async load k_in → s_stage (overlapped with A_inv conversion)
   {
     const __nv_bfloat16 *k_ptr = k_in + cstart * kHk * kK + k_head * kK;
     const uint32_t stage_base = cvt_smem_ptr(s_stage);
     for (int i = tid; i < kBT * kK / 8; i += 128) {
       int row = i / (kK / 8), col8 = (i % (kK / 8)) * 8;
       uint32_t dst = stage_base + (row * kK + col8) * 2;
       cp_async_cg_128(dst, &k_ptr[row * kHk * kK + col8], row < clen);
     }
     asm volatile("cp.async.commit_group;");
   }
 
   // Compute w scaling: s_bg[t] = beta[t] * exp(g[t])
   if (tid < kBT)
     s_bg[tid] = (tid < clen) ? s_beta[tid] * __expf(s_g[tid]) : 0.0f;
 
   asm volatile("cp.async.wait_group 0;");
   __syncthreads();
 
   // B operand buffer: s_B_tiles reinterpreted as bf16 row-major [64,128] for full-width pass
   __nv_bfloat16 *s_input_bf16 = reinterpret_cast<__nv_bfloat16 *>(s_B_tiles);  // [64,128]
   const int lane_p3 = tid % 32;
   const int m_base_p3 = warp_id * 16;
 
   // Process W: scale full 128-col k input into s_input_bf16[64,128], single MMA pass with 16 N-tiles
   {
     // Scale all 128 cols at once: s_input_bf16[t, j] = s_stage[t, j] * s_bg[t]
     // Vectorized with bf16x2 multiply (native hw instruction, no float conversion)
     for (int i = tid; i < kBT * kK / 8; i += 128) {
       int t = (i * 8) / kK, j = (i * 8) % kK;
       if (t < clen) {
         __nv_bfloat162 bg2 = __float2bfloat162_rn(s_bg[t]);
         int4 src = *reinterpret_cast<const int4 *>(&s_stage[t * kK + j]);
         __nv_bfloat162 *sp = reinterpret_cast<__nv_bfloat162 *>(&src);
         #pragma unroll
         for (int p = 0; p < 4; p++)
           sp[p] = __hmul2(sp[p], bg2);
         *reinterpret_cast<int4 *>(&s_input_bf16[t * kK + j]) = src;
       } else {
         *reinterpret_cast<int4 *>(&s_input_bf16[t * kK + j]) = make_int4(0, 0, 0, 0);
       }
     }
     __syncthreads();
 
     // Issue v cp.async EARLY — overlaps with W MMA below
     // s_stage (s_result) is no longer read after scaling above, safe to overwrite
     {
       const __nv_bfloat16 *v_ptr = v_in + cstart * kHv * kV + hv * kV;
       const uint32_t stage_base = cvt_smem_ptr(s_stage);
       for (int i = tid; i < kBT * kK / 8; i += 128) {
         int row = i / (kK / 8), col8 = (i % (kK / 8)) * 8;
         uint32_t dst = stage_base + (row * kK + col8) * 2;
         cp_async_cg_128(dst, &v_ptr[row * kHv * kV + col8], row < clen);
       }
       asm volatile("cp.async.commit_group;");
     }
 
     // mma.sync: C[row, j] = A_inv[row, t] * input[t, j], 16 N-tiles for full 128 cols
     // A = s_Ainv_bf16[64,64] stride kBT, B = s_input_bf16[64,128] stride kK
     // v cp.async loading in background into s_stage
     float p3_acc[16][4] = {};
     for (int kt = 0; kt < kBT / 16; kt++) {
       uint32_t a[4];
       {
         int r = (lane_p3 % 8) + ((lane_p3 & 8) ? 8 : 0) + m_base_p3;
         int c = (lane_p3 >= 16) ? 8 : 0;
         ldmatrix<4>(a, cvt_smem_ptr(&s_Ainv_bf16[r * kBT + kt * 16 + c]));
       }
       for (int nt = 0; nt < 16; nt++) {
         uint32_t b[2];
         {
           int kr = lane_p3 % 16;
           ldmatrix_trans<2>(b, cvt_smem_ptr(&s_input_bf16[(kt * 16 + kr) * kK + nt * 8]));
         }
         mma_m16n8k16_bf16(
             p3_acc[nt][0], p3_acc[nt][1], p3_acc[nt][2], p3_acc[nt][3],
             a[0], a[1], a[2], a[3], b[0], b[1],
             p3_acc[nt][0], p3_acc[nt][1], p3_acc[nt][2], p3_acc[nt][3]);
       }
     }
     // Write to global w_out (packed 32-bit stores), all 128 cols
     {
       int gID = lane_p3 / 4, tIG = lane_p3 % 4;
       int r0 = m_base_p3 + gID, r1 = r0 + 8;
       for (int nt = 0; nt < 16; nt++) {
         int c0 = nt * 8 + tIG * 2;
         if (r0 < clen) {
           __nv_bfloat162 pair0 = {__float2bfloat16_rn(p3_acc[nt][0]), __float2bfloat16_rn(p3_acc[nt][1])};
           *reinterpret_cast<__nv_bfloat162 *>(&w_out[(cstart + r0) * kHv * kK + hv * kK + c0]) = pair0;
         }
         if (r1 < clen) {
           __nv_bfloat162 pair1 = {__float2bfloat16_rn(p3_acc[nt][2]), __float2bfloat16_rn(p3_acc[nt][3])};
           *reinterpret_cast<__nv_bfloat162 *>(&w_out[(cstart + r1) * kHv * kK + hv * kK + c0]) = pair1;
         }
       }
     }
   }
 
   if (tid < kBT)
     s_bg[tid] = (tid < clen) ? s_beta[tid] : 0.0f;
 
   // Wait for v cp.async to complete
   asm volatile("cp.async.wait_group 0;");
   __syncthreads();
 
   // Process U: scale full 128-col v input into s_input_bf16[64,128], single MMA pass
   {
     // Vectorized with bf16x2 multiply
     for (int i = tid; i < kBT * kK / 8; i += 128) {
       int t = (i * 8) / kK, j = (i * 8) % kK;
       if (t < clen) {
         __nv_bfloat162 bg2 = __float2bfloat162_rn(s_bg[t]);
         int4 src = *reinterpret_cast<const int4 *>(&s_stage[t * kK + j]);
         __nv_bfloat162 *sp = reinterpret_cast<__nv_bfloat162 *>(&src);
         #pragma unroll
         for (int p = 0; p < 4; p++)
           sp[p] = __hmul2(sp[p], bg2);
         *reinterpret_cast<int4 *>(&s_input_bf16[t * kK + j]) = src;
       } else {
         *reinterpret_cast<int4 *>(&s_input_bf16[t * kK + j]) = make_int4(0, 0, 0, 0);
       }
     }
     __syncthreads();
 
     float p3_acc[16][4] = {};
     for (int kt = 0; kt < kBT / 16; kt++) {
       uint32_t a[4];
       {
         int r = (lane_p3 % 8) + ((lane_p3 & 8) ? 8 : 0) + m_base_p3;
         int c = (lane_p3 >= 16) ? 8 : 0;
         ldmatrix<4>(a, cvt_smem_ptr(&s_Ainv_bf16[r * kBT + kt * 16 + c]));
       }
       for (int nt = 0; nt < 16; nt++) {
         uint32_t b[2];
         {
           int kr = lane_p3 % 16;
           ldmatrix_trans<2>(b, cvt_smem_ptr(&s_input_bf16[(kt * 16 + kr) * kK + nt * 8]));
         }
         mma_m16n8k16_bf16(
             p3_acc[nt][0], p3_acc[nt][1], p3_acc[nt][2], p3_acc[nt][3],
             a[0], a[1], a[2], a[3], b[0], b[1],
             p3_acc[nt][0], p3_acc[nt][1], p3_acc[nt][2], p3_acc[nt][3]);
       }
     }
     // After U MMA, all smem reads are done. Sync before U global writes.
     __syncthreads();

     // TMEM already deallocated after Phase 1

     // Write U results to global
     {
       int gID = lane_p3 / 4, tIG = lane_p3 % 4;
       int r0 = m_base_p3 + gID, r1 = r0 + 8;
       for (int nt = 0; nt < 16; nt++) {
         int c0 = nt * 8 + tIG * 2;
         if (r0 < clen) {
           __nv_bfloat162 pair0 = {__float2bfloat16_rn(p3_acc[nt][0]), __float2bfloat16_rn(p3_acc[nt][1])};
           *reinterpret_cast<__nv_bfloat162 *>(&u_out[(cstart + r0) * kHv * kV + hv * kV + c0]) = pair0;
         }
         if (r1 < clen) {
           __nv_bfloat162 pair1 = {__float2bfloat16_rn(p3_acc[nt][2]), __float2bfloat16_rn(p3_acc[nt][3])};
           *reinterpret_cast<__nv_bfloat162 *>(&u_out[(cstart + r1) * kHv * kV + hv * kV + c0]) = pair1;
         }
       }
     }
   }
   // Phase 4 eliminated — q@k^T now computed inside OOutputKernel
 }
 
 // Dead kernel FusedRecurrenceOutput removed — replaced by H+O kernel split
 
 // ═══════════════════════════════════════════════════════════════════
 // Kernel H v3: ALL mma.sync, NO tcgen05/TMEM
 // Grid: (kNVT_H=8, num_seqs * kHv), Block: 128
 // h in persistent fp32 registers. Both MMA1 and MMA2 via inline PTX.
 // No TMEM allocation → no TMEM non-determinism.
 //
 // Smem (~43KB): s_w[64*128]bf16=16KB, s_h_T[128*16]bf16=4KB,
 //   s_wh[64*16]fp32=4KB, s_gc[64]fp32=256B, s_k[64*128]bf16=16KB, s_vnew_T[16*64]bf16=2KB
 // ═══════════════════════════════════════════════════════════════════
 __global__ void __launch_bounds__(128, 1)
 HRecurrenceKernel(
     const __nv_bfloat16 *__restrict__ w_in,
     const __nv_bfloat16 *__restrict__ k_in,
     __nv_bfloat16 *__restrict__ u_inout,
     const float *__restrict__ g_cumsum,
     const float *__restrict__ state0,
     const int64_t *__restrict__ cu_seqlens,
     __nv_bfloat16 *__restrict__ d_h,
     float *__restrict__ new_state,
     int64_t num_seqs) {
 
   const int v_tile = blockIdx.x;
   const int bh = blockIdx.y;
   const int seq_idx = bh / kHv, hv = bh % kHv;
   if (seq_idx >= num_seqs) return;
 
   const int64_t k_head = hv / (kHv / kHk);
   const int64_t s0 = cu_seqlens[seq_idx], s1 = cu_seqlens[seq_idx + 1];
   const int NT = (int)((s1 - s0 + kBT - 1) / kBT);
   // Compute chunk_base inline (replaces chunk_offsets lookup)
   int64_t chunk_base = 0;
   for (int i = 0; i < seq_idx; i++) {
     int64_t slen_i = cu_seqlens[i + 1] - cu_seqlens[i];
     chunk_base += (slen_i + kBT - 1) / kBT;
   }
 
   const int warp_id = threadIdx.x / 32;
   const int lane_id = threadIdx.x % 32;
   const int tid = threadIdx.x;
   const int v_start = v_tile * kBV_H;
 
   extern __shared__ __align__(256) char smem[];
   // Double-buffered w: s_w[0] and s_w[1], each [64,128] = 16KB
   __nv_bfloat16 *s_w_buf[2];
   s_w_buf[0] = reinterpret_cast<__nv_bfloat16 *>(smem);                   // [64,128] = 16KB
   s_w_buf[1] = s_w_buf[0] + kBT * kK;                                     // [64,128] = 16KB
   __nv_bfloat16 *s_h_T = s_w_buf[1] + kBT * kK;                           // [128,16] = 4KB (transposed h)
   float *s_wh = reinterpret_cast<float *>(s_h_T + kK * kBV_H);            // [64,16] = 4KB
   float *s_gc = s_wh + kBT * kBV_H;                                        // [64] = 256B
   __nv_bfloat16 *s_k = reinterpret_cast<__nv_bfloat16 *>(s_gc + kBT);     // [64,128] = 16KB
   __nv_bfloat16 *s_vnew_T = s_k + kBT * kK;                               // [16,64] = 2KB
   __nv_bfloat16 *s_u = s_vnew_T + kBV_H * kBT;                            // [64,16] = 2KB (prefetched u_inout)

   // Persistent h state: h_reg[nt][reg], 4 N-tiles × 4 fp32 = 16 regs/thread
   // row0 = lane_id/4 (groupID, 0..7), row1 = row0+8 (8..15)
   // col = warp_id*32 + nt*8 + (lane_id%4)*2
   float h_reg[4][4];
   {
     int row0 = lane_id / 4, row1 = row0 + 8;
     int col_pair = lane_id % 4;
     int wcb = warp_id * 32;
     if (state0) {
       const float *h0 = state0 + ((int64_t)seq_idx * kHv + hv) * kV * kK;
       for (int nt = 0; nt < 4; nt++) {
         int col0 = wcb + nt * 8 + col_pair * 2;
         h_reg[nt][0] = h0[(v_start + row0) * kK + col0];
         h_reg[nt][1] = h0[(v_start + row0) * kK + col0 + 1];
         h_reg[nt][2] = h0[(v_start + row1) * kK + col0];
         h_reg[nt][3] = h0[(v_start + row1) * kK + col0 + 1];
       }
     } else {
       for (int nt = 0; nt < 4; nt++)
         h_reg[nt][0] = h_reg[nt][1] = h_reg[nt][2] = h_reg[nt][3] = 0.0f;
     }
   }
   __syncthreads();
 
   // Prefetch w[0] before the chunk loop
   {
     const int64_t cstart0 = s0;
     const int clen0 = min(kBT, (int)(s1 - cstart0));
     const __nv_bfloat16 *w_ptr = w_in + cstart0 * kHv * kK + hv * kK;
     __nv_bfloat16 *s_w = s_w_buf[0];
     for (int i = tid; i < kBT * kK / 8; i += 128) {
       int row = i / (kK / 8), col8 = (i % (kK / 8)) * 8;
       uint32_t dst = cvt_smem_ptr(&s_w[row * kK + col8]);
       if (row < clen0) {
         cp_async_cg_128(dst, &w_ptr[row * kHv * kK + col8], true);
       } else {
         *reinterpret_cast<int4 *>(&s_w[row * kK + col8]) = make_int4(0,0,0,0);
       }
     }
     asm volatile("cp.async.commit_group;");
   }
 
   for (int ct = 0; ct < NT; ct++) {
     const int64_t cstart = s0 + (int64_t)ct * kBT;
     const int clen = min(kBT, (int)(s1 - cstart));
     const int64_t chunk_id = chunk_base + ct;
     __nv_bfloat16 *s_w = s_w_buf[ct & 1];

     // Step 0: Store h → d_h (NON-TRANSPOSED: h[bv,k]) + s_h_T, wait for w[ct] cp.async
     {
       __nv_bfloat16 *h_dst = d_h + (chunk_id * kHv + hv) * kV * kK;
       int row0 = lane_id / 4, row1 = row0 + 8, col_pair = lane_id % 4;
       int wcb = warp_id * 32;
       for (int nt = 0; nt < 4; nt++) {
         int col0 = wcb + nt * 8 + col_pair * 2;
         __nv_bfloat16 b0 = __float2bfloat16(h_reg[nt][0]);
         __nv_bfloat16 b1 = __float2bfloat16(h_reg[nt][1]);
         __nv_bfloat16 b2 = __float2bfloat16(h_reg[nt][2]);
         __nv_bfloat16 b3 = __float2bfloat16(h_reg[nt][3]);
         // d_h: NON-TRANSPOSED layout h[bv, k] with stride kK=128
         // Vectorized bf16x2 writes (col0 and col0+1 are adjacent in [bv,k] layout)
         *reinterpret_cast<__nv_bfloat162 *>(&h_dst[(v_start + row0) * kK + col0]) =
             __nv_bfloat162{b0, b1};
         *reinterpret_cast<__nv_bfloat162 *>(&h_dst[(v_start + row1) * kK + col0]) =
             __nv_bfloat162{b2, b3};
         // s_h_T for MMA1 within H-kernel (unchanged)
         s_h_T[col0       * kBV_H + row0] = b0;
         s_h_T[(col0 + 1) * kBV_H + row0] = b1;
         s_h_T[col0       * kBV_H + row1] = b2;
         s_h_T[(col0 + 1) * kBV_H + row1] = b3;
       }
       // Load g_cumsum early (overlapped with h store)
       if (tid < kBT)
         s_gc[tid] = (tid < clen) ? g_cumsum[(cstart + tid) * kHv + hv] : 0.0f;
       // Wait for w[ct] cp.async (issued in previous iteration or before loop)
       asm volatile("cp.async.wait_group 0;");
     }
     __syncthreads();
 
     // Issue k + u loads early — both overlap with MMA1, single commit group
     {
       const __nv_bfloat16 *k_src = k_in + cstart * kHk * kK + k_head * kK;
       for (int i = tid; i < kBT * kK / 8; i += 128) {
         int row = i / (kK / 8), col8 = (i % (kK / 8)) * 8;
         uint32_t dst = cvt_smem_ptr(&s_k[row * kK + col8]);
         if (row < clen) {
           cp_async_cg_128(dst, &k_src[row * kHk * kK + col8], true);
         } else {
           *reinterpret_cast<int4 *>(&s_k[row * kK + col8]) = make_int4(0,0,0,0);
         }
       }
       // Prefetch u_inout into s_u (hides strided global load latency behind MMA1)
       {
         const __nv_bfloat16 *u_base = u_inout + cstart * kHv * kV + hv * kV + v_start;
         for (int i = tid; i < kBT * kBV_H / 8; i += 128) {
           int t = i / (kBV_H / 8), bv8 = (i % (kBV_H / 8)) * 8;
           uint32_t dst = cvt_smem_ptr(&s_u[t * kBV_H + bv8]);
           if (t < clen) {
             cp_async_cg_128(dst, &u_base[t * kHv * kV + bv8], true);
           } else {
             *reinterpret_cast<int4 *>(&s_u[t * kBV_H + bv8]) = make_int4(0,0,0,0);
           }
         }
       }
       asm volatile("cp.async.commit_group;");
     }
 
     // Step 1: MMA1 — w[64,128] @ h^T[128,16] → wh[64,16]
     // (k loading in background via cp.async)
     {
       float wh_acc[2][4] = {};
       #pragma unroll
       for (int kt = 0; kt < 8; kt++) {
         uint32_t a[4];
         {
           int r = (lane_id % 8) + ((lane_id & 8) ? 8 : 0) + warp_id * 16;
           int c = (lane_id >= 16) ? 8 : 0;
           ldmatrix<4>(a, cvt_smem_ptr(&s_w[r * kK + kt * 16 + c]));
         }
         #pragma unroll
         for (int nt = 0; nt < 2; nt++) {
           uint32_t b[2];
           {
             int k_row = lane_id % 16;
             ldmatrix_trans<2>(b, cvt_smem_ptr(&s_h_T[(kt * 16 + k_row) * kBV_H + nt * 8]));
           }
           mma_m16n8k16_bf16(
               wh_acc[nt][0], wh_acc[nt][1], wh_acc[nt][2], wh_acc[nt][3],
               a[0], a[1], a[2], a[3], b[0], b[1],
               wh_acc[nt][0], wh_acc[nt][1], wh_acc[nt][2], wh_acc[nt][3]);
         }
       }
       // Scatter wh → s_wh[64, 16] — per-warp output
       {
         int r0 = warp_id * 16 + lane_id / 4, r1 = r0 + 8;
         int cp = lane_id % 4;
         for (int nt = 0; nt < 2; nt++) {
           int c0 = nt * 8 + cp * 2;
           if (r0 < kBT) { s_wh[r0 * kBV_H + c0] = wh_acc[nt][0]; s_wh[r0 * kBV_H + c0 + 1] = wh_acc[nt][1]; }
           if (r1 < kBT) { s_wh[r1 * kBV_H + c0] = wh_acc[nt][2]; s_wh[r1 * kBV_H + c0 + 1] = wh_acc[nt][3]; }
         }
       }
     }
     // Wait for k+u cp.async (issued before MMA1, latency hidden behind MMA1 compute)
     // Must complete before vnew reads s_u and before MMA2 reads s_k
     asm volatile("cp.async.wait_group 0;");
     // NO __syncthreads here — v_new is per-warp (reads only own MMA1 rows)
     // s_u and s_k are filled by cp.async (thread-local, no cross-thread dependency)

     // Step 2: v_new (per-warp), gate, scale h, prefetch w[ct+1] via cp.async
     {
       // Prefetch w[ct+1] into alternate buffer (all threads)
       if (ct + 1 < NT) {
         const int64_t next_cstart = s0 + (int64_t)(ct + 1) * kBT;
         const int next_clen = min(kBT, (int)(s1 - next_cstart));
         const __nv_bfloat16 *w_ptr = w_in + next_cstart * kHv * kK + hv * kK;
         __nv_bfloat16 *s_w_next = s_w_buf[(ct + 1) & 1];
         for (int i = tid; i < kBT * kK / 8; i += 128) {
           int row = i / (kK / 8), col8 = (i % (kK / 8)) * 8;
           uint32_t dst = cvt_smem_ptr(&s_w_next[row * kK + col8]);
           if (row < next_clen) {
             cp_async_cg_128(dst, &w_ptr[row * kHv * kK + col8], true);
           } else {
             *reinterpret_cast<int4 *>(&s_w_next[row * kK + col8]) = make_int4(0,0,0,0);
           }
         }
         asm volatile("cp.async.commit_group;");
       }
 
       // Per-warp v_new: read from s_u (prefetched), compute v_new, writeback to global
       __syncwarp();
       float g_last = s_gc[clen - 1];
       {
         int bv8 = (lane_id < 16) ? 0 : 8;
         int t = warp_id * 16 + (lane_id & 15);
         if (t < clen) {
           // Read from smem (prefetched via cp.async, latency hidden behind MMA1)
           int4 u_vec = *reinterpret_cast<const int4 *>(&s_u[t * kBV_H + bv8]);
           __nv_bfloat16 *u_arr = reinterpret_cast<__nv_bfloat16 *>(&u_vec);
           float gc_t = s_gc[t];
           float gate = __expf(g_last - gc_t);
           #pragma unroll
           for (int j = 0; j < 8; j++) {
             float vn = __bfloat162float(u_arr[j]) - s_wh[t * kBV_H + bv8 + j];
             u_arr[j] = __float2bfloat16(vn);
             s_vnew_T[(bv8 + j) * kBT + t] = __float2bfloat16(vn * gate);
           }
           // Writeback modified u to global
           *reinterpret_cast<int4 *>(&u_inout[(cstart + t) * kHv * kV + hv * kV + v_start + bv8]) = u_vec;
         } else if (t < kBT) {
           #pragma unroll
           for (int j = 0; j < 8; j++)
             s_vnew_T[(bv8 + j) * kBT + t] = __float2bfloat16(0.0f);
         }
       }
       float g_exp = __expf(g_last);
       for (int nt = 0; nt < 4; nt++) {
         h_reg[nt][0] *= g_exp; h_reg[nt][1] *= g_exp;
         h_reg[nt][2] *= g_exp; h_reg[nt][3] *= g_exp;
       }
       // k+u already waited above. Only w[ct+1] may be in flight.
       // Next iteration's step0 will wait for it with wait_group(0).
       if (ct + 1 >= NT)
         asm volatile("cp.async.wait_group 0;");
       __syncthreads();
     }
 
     // Step 3: MMA2 — h += vnew^T[16,64] @ k[64,128] via mma.sync
     {
       const int wn = warp_id * 32;
       #pragma unroll
       for (int kt = 0; kt < 4; kt++) {
         uint32_t a[4];
         {
           int r = (lane_id % 8) + ((lane_id & 8) ? 8 : 0);
           int c = (lane_id >= 16) ? 8 : 0;
           ldmatrix<4>(a, cvt_smem_ptr(&s_vnew_T[r * kBT + kt * 16 + c]));
         }
         #pragma unroll
         for (int nt = 0; nt < 4; nt++) {
           uint32_t b[2];
           {
             int kr = lane_id % 16;
             ldmatrix_trans<2>(b, cvt_smem_ptr(&s_k[(kt * 16 + kr) * kK + wn + nt * 8]));
           }
           mma_m16n8k16_bf16(
               h_reg[nt][0], h_reg[nt][1], h_reg[nt][2], h_reg[nt][3],
               a[0], a[1], a[2], a[3], b[0], b[1],
               h_reg[nt][0], h_reg[nt][1], h_reg[nt][2], h_reg[nt][3]);
         }
       }
     }
     // No __syncthreads needed: MMA_H2 only writes to h_reg (registers).
     // Next iteration's Step 0 has __syncthreads after h store + w wait.
   }
 
   // DEBUG: print per-step timing from one representative block
   // Store final state
   {
     float *ns = new_state + ((int64_t)seq_idx * kHv + hv) * kV * kK;
     int row0 = lane_id / 4, row1 = row0 + 8, cp = lane_id % 4;
     int wcb = warp_id * 32;
     for (int nt = 0; nt < 4; nt++) {
       int c0 = wcb + nt * 8 + cp * 2;
       ns[(v_start + row0) * kK + c0]     = h_reg[nt][0];
       ns[(v_start + row0) * kK + c0 + 1] = h_reg[nt][1];
       ns[(v_start + row1) * kK + c0]     = h_reg[nt][2];
       ns[(v_start + row1) * kK + c0 + 1] = h_reg[nt][3];
     }
   }
 }
 
 
 // ═══════════════════════════════════════════════════════════════════
 // Swizzle helpers for tcgen05 MMA with [64, 128] and [64, 64] tiles
 // ═══════════════════════════════════════════════════════════════════
 
 // Compute the byte offset in a swizzled [64, 128] bf16 tile for element (row, col).
 // row: 0..63, col: 0..127 (bf16 element indices)
 // Returns: byte offset in shared memory (0..16383)
 __device__ __host__ __forceinline__
 int swizzle_byte_offset_64x128(int row, int col) {
   int row16 = row & 15;
   int row_group = row >> 4;
   int cg = col >> 3;
   int cg_lo = cg & 7;
   int cg_hi = cg >> 3;
   return ((cg_lo ^ (row16 & 7)) << 4)
        | (row16 << 7)
        | (cg_hi << 13)
        | (row_group << 11)
        | ((col & 7) << 1);
 }
 
 // Per-thread swizzle offset for non-transposed loading (Triton pattern).
 // 256 threads load [64, 128] tile: each thread loads 4 rows x 8 bf16.
 __device__ __forceinline__
 int swizzle_offset_q(int tid) {
   int off = ((tid & 7) << 4) | ((tid & 0xF0) << 3);
   off ^= (tid & 0x70);
   off |= ((tid << 10) & 8192);
   return off;
 }
 
 // Build swizzled descriptor for tcgen05.mma
 __device__ __forceinline__
 uint64_t make_swizzle_desc(uint32_t smem_addr) {
   uint64_t addr_enc = ((uint64_t)(smem_addr) >> 4ULL) & 0x3FFFULL;
   return addr_enc | 0x4000004002000000ULL;
 }
 
 // Scatter-transpose: h_T[K, V] → swizzled tile[BV, K] in smem
 // Source: h_T[k, v_start + bv] for k=0..K-1, bv=0..BV-1
 // Dest: swizzled tile position (row=bv, col=k)
 // For [64,128] tile: 1024 work items, 4 per thread with 256 threads
 __device__ __forceinline__
 void scatter_transpose_64x128(
     const __nv_bfloat16 *__restrict__ h_T, // [K, V_full] in global
     char *smem_tile,                        // swizzled [64, 128] bf16 tile
     int v_start,
     int V_stride,
     int tid)
 {
   for (int item = 0; item < 4; item++) {
     int item_id = tid + item * 256;
     int bv = item_id >> 4;           // bv 0..63
     int kg = item_id & 15;           // k_group 0..15
     int k_start = kg * 8;
 
     uint32_t packed[4];
     #pragma unroll
     for (int j = 0; j < 4; j++) {
       __nv_bfloat16 v0 = h_T[(k_start + j*2) * V_stride + v_start + bv];
       __nv_bfloat16 v1 = h_T[(k_start + j*2 + 1) * V_stride + v_start + bv];
       packed[j] = (uint32_t)__bfloat16_as_ushort(v0) | ((uint32_t)__bfloat16_as_ushort(v1) << 16);
     }
 
     int byte_off = swizzle_byte_offset_64x128(bv, k_start);
     *reinterpret_cast<uint4*>(smem_tile + byte_off) = *reinterpret_cast<uint4*>(packed);
   }
 }
 
 // Scatter-transpose for [64, 64] tile (BV=64, K=64)
 // Source: src[t, v_start + bv] with t_stride between rows
 // Dest: swizzled tile[bv, t]
 // 64*8 = 512 work items, 2 per thread with 256 threads
 __device__ __forceinline__
 void scatter_transpose_64x64(
     const __nv_bfloat16 *__restrict__ src, // [T, ...] in global
     char *smem_tile,                        // swizzled [64, 64] bf16 tile
     int v_start,
     int src_stride,                         // stride between consecutive t rows
     int clen,                               // actual number of valid t rows
     int tid)
 {
   // 64 bv * 8 t_groups = 512 work items, 2 per thread
   for (int item = 0; item < 2; item++) {
     int item_id = tid + item * 256;
     int bv = item_id >> 3;           // bv 0..63
     int tg = item_id & 7;           // t_group 0..7
     int t_start = tg * 8;
 
     uint32_t packed[4];
     #pragma unroll
     for (int j = 0; j < 4; j++) {
       int t0 = t_start + j*2, t1 = t0 + 1;
       __nv_bfloat16 v0 = (t0 < clen) ? src[t0 * src_stride + v_start + bv] : __float2bfloat16(0.0f);
       __nv_bfloat16 v1 = (t1 < clen) ? src[t1 * src_stride + v_start + bv] : __float2bfloat16(0.0f);
       packed[j] = (uint32_t)__bfloat16_as_ushort(v0) | ((uint32_t)__bfloat16_as_ushort(v1) << 16);
     }
 
     int byte_off = swizzle_byte_offset_64x128(bv, t_start);
     *reinterpret_cast<uint4*>(smem_tile + byte_off) = *reinterpret_cast<uint4*>(packed);
   }
 }
 
 // ═══════════════════════════════════════════════════════════════════
 // Kernel O v9: tcgen05 with inline q@k^T (Phase 4 fused), 256 threads
 // Grid: (kNVT_O, total_chunks, kHv), Block: 256
 //
 // NEW flow (q@k^T computed here, not in FusedPrepKernel):
 //   1. Load q + k (cp.async) → MMA_qk: q@k^T → TMEM
 //   2. Read TMEM, apply gating + causal mask → bf16 → s_attn_tile (swizzled)
 //   3. Load h (cp.async, reuse s_h_tile) + scatter vnew → wait
 //   4. MMA1: q@h^T → TMEM (overwrites MMA_qk result)
 //   5. Read TMEM, scale by exp(g), write back to TMEM
 //   6. MMA3: attn@vnew → accumulate onto TMEM
 //   7. Read TMEM → output
 //
 // Smem layout (~49KB):
 //   s_q:    [64,128] swizzled = 16KB  (A for MMA_qk and MMA1)
 //   s_h:    [64,128] swizzled = 16KB  (B for MMA_qk: k; then reloaded with h for MMA1)
 //   s_attn: [64,64] swizzled = 8KB   (A for MMA3, written by gating code)
 //   s_vnew: [64,64] swizzled = 8KB   (B for MMA3, scatter-transposed from u_in)
 //   s_gc:   [64] fp32 = 256B
 //   mbars:  3 × 8B
 //   tmem_buf: 4B
 // ═══════════════════════════════════════════════════════════════════
 __global__ void __launch_bounds__(256, 1)
 OOutputKernel(
     const __nv_bfloat16 *__restrict__ q_in,
     const __nv_bfloat16 *__restrict__ k_in,
     const __nv_bfloat16 *__restrict__ u_in,
     const __nv_bfloat16 *__restrict__ d_h,
     const float *__restrict__ g_cumsum,
     const int64_t *__restrict__ cu_seqlens,
     const int32_t *__restrict__ chunk_indices,
     float scale,
     __nv_bfloat16 *__restrict__ output,
     const int32_t *__restrict__ total_chunks_ptr,
     int64_t num_seqs) {

   const int v_tile = blockIdx.x, hv = blockIdx.z;
   const int tid = threadIdx.x, warp_id = tid / 32, lane_id = tid % 32;
   const int v_start = v_tile * kBV_O;
   const int chunk_id = blockIdx.y;

   // Inline chunk_id → (seq_idx, local_chunk) mapping
   int seq_idx = -1, local_chunk = -1;
   {
     int running = 0;
     for (int i = 0; i < num_seqs; i++) {
       int64_t slen = cu_seqlens[i + 1] - cu_seqlens[i];
       int nc = (int)((slen + kBT - 1) / kBT);
       if (chunk_id < running + nc) {
         seq_idx = i;
         local_chunk = chunk_id - running;
         break;
       }
       running += nc;
     }
     if (seq_idx < 0) return;  // excess block
   }

   // Smem layout: s_q(16KB) | s_h(16KB) | s_attn(8KB) | s_vnew(8KB) | s_gc(256B) | mbars(24B) | tmem_buf(4B)
   extern __shared__ __align__(128) char smem[];
   char *s_q_tile    = smem;                    // [64,128] swizzled = 16384 bytes
   char *s_h_tile    = smem + 16384;            // [64,128] swizzled = 16384 bytes (k first, then h)
   char *s_attn_tile = smem + 32768;            // [64,64] swizzled = 8192 bytes
   char *s_vnew_tile = smem + 40960;            // [64,64] swizzled = 8192 bytes
   float *s_gc       = reinterpret_cast<float *>(smem + 49152);  // [64] fp32 = 256 bytes
   uint64_t *mbar    = reinterpret_cast<uint64_t *>(smem + 49408);  // 3 mbars × 8B
   int *tmem_buf     = reinterpret_cast<int *>(smem + 49432);       // 4B

   // seq_idx and local_chunk already computed inline above
   const int64_t s0 = cu_seqlens[seq_idx], s1 = cu_seqlens[seq_idx + 1];
   const int64_t cstart = s0 + (int64_t)local_chunk * kBT;
   const int clen = min(kBT, (int)(s1 - cstart));
   const int64_t q_head = hv / (kHv / kHq);
   const int64_t k_head = hv / (kHv / kHk);

   // Load g_cumsum
   if (tid < kBT)
     s_gc[tid] = (tid < clen) ? g_cumsum[(cstart + tid) * kHv + hv] : 0.0f;

   // ─── Allocate TMEM (128 cols for BM=64, BN=64) ───
   if (tid < 32) {
     asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(cvt_smem_ptr(tmem_buf)), "r"(128));
   }
   __syncthreads();
   int taddr = *tmem_buf;
   __syncthreads();
   if (tid < 32) {
     asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
   }

   // ─── Init mbarriers (3: MMA_qk, MMA1, MMA3) ───
   uint32_t mbar0_addr = cvt_smem_ptr(&mbar[0]);  // MMA_qk
   uint32_t mbar1_addr = cvt_smem_ptr(&mbar[1]);  // MMA1
   uint32_t mbar2_addr = cvt_smem_ptr(&mbar[2]);  // MMA3
   if (tid == 0) {
     asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(mbar0_addr));
     asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(mbar1_addr));
     asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(mbar2_addr));
   }
   __syncthreads();

   // ═══════════════════════════════════════════════════════════════
   // STEP 1: Load q (cp.async) + Load k (cp.async into s_h_tile)
   // ═══════════════════════════════════════════════════════════════
   {
     // q loading via cp.async to swizzled smem
     const __nv_bfloat16 *q_ptr = q_in + cstart * kHq * kK + q_head * kK;
     uint32_t q_smem_base = cvt_smem_ptr(s_q_tile);
     int row_base = (tid >> 4) & 0xF;
     int col_start = (tid & 0xF) * 8;
     int base_off = swizzle_offset_q(tid);

     for (int batch = 0; batch < 4; batch++) {
       int row = row_base + batch * 16;
       cp_async_cg_128(q_smem_base + base_off + batch * 2048,
           &q_ptr[row * kHq * kK + col_start], row < clen);
     }
     asm volatile("cp.async.commit_group;");
   }

   // k loading via cp.async into s_h_tile (reusing h's buffer for k first)
   // k layout: k_in[t, k_head, K] with stride kHk*kK between rows
   {
     const __nv_bfloat16 *k_ptr = k_in + cstart * kHk * kK + k_head * kK;
     uint32_t k_smem_base = cvt_smem_ptr(s_h_tile);
     int row_base = (tid >> 4) & 0xF;
     int col_start = (tid & 0xF) * 8;
     int base_off = swizzle_offset_q(tid);

     for (int batch = 0; batch < 4; batch++) {
       int row = row_base + batch * 16;
       cp_async_cg_128(k_smem_base + base_off + batch * 2048,
           &k_ptr[row * kHk * kK + col_start], row < clen);
     }
     asm volatile("cp.async.commit_group;");
   }

   // Wait for both q and k cp.async
   asm volatile("cp.async.wait_group 0;");
   __syncthreads();
   asm volatile("fence.proxy.async.shared::cta;");

   uint32_t smem_base = cvt_smem_ptr(smem);

   // ═══════════════════════════════════════════════════════════════
   // STEP 2: MMA_qk: q[64,128] @ k[64,128]^T → qk[64,64] in TMEM
   // ═══════════════════════════════════════════════════════════════
   if (warp_id == 0 && elect_sync()) {
     uint32_t idesc = 0x04100490; // BM=64, BN=64, bf16→f32
     for (int step = 0; step < 8; step++) {
       int k_off = (step & 3) * 32 + (step >> 2) * 8192;
       uint64_t a_desc = make_swizzle_desc(smem_base + k_off);          // s_q
       uint64_t b_desc = make_swizzle_desc(smem_base + 16384 + k_off);  // s_h (has k)
       tcgen05_mma(taddr, a_desc, b_desc, idesc, step);
     }
     tcgen05_commit(mbar0_addr);
   }

   // Wait for MMA_qk
   mbarrier_wait(mbar0_addr, 0);

   // ═══════════════════════════════════════════════════════════════
   // STEP 3: Read TMEM → apply gating + causal mask → bf16 → s_attn_tile
   // ═══════════════════════════════════════════════════════════════
   {
     // Ensure g_cumsum is visible (loaded at top of kernel, synced already)
     uint32_t shfl_warp;
     asm volatile("shfl.sync.idx.b32 %0, %1, 0, 31, -1;" : "=r"(shfl_warp) : "r"(warp_id));
     int tmem_off = ((shfl_warp << 21) & 6291456) | ((shfl_warp & 4) << 3);
     int my_taddr = tmem_off + taddr;

     uint32_t vals[16];
     tcgen05_ld_16x32bx2(vals, my_taddr);

     int row = (shfl_warp % 4) * 16 + (lane_id % 16);
     int col_base = (shfl_warp >= 4 ? 32 : 0) + (lane_id >= 16 ? 16 : 0);

     // Midpoint normalization for exp stability
     float g_mid = (clen > 1) ? s_gc[clen >> 1] : ((clen > 0) ? s_gc[0] : 0.0f);
     float eg_row = (row < clen) ? __expf(s_gc[row] - g_mid) : 0.0f;

     // Apply gating + causal mask, convert to bf16, write to s_attn_tile
     uint32_t attn_smem_base_addr = cvt_smem_ptr(s_attn_tile);
     for (int r = 0; r < 16; r += 2) {
       int col0 = col_base + r;
       int col1 = col0 + 1;
       float raw0 = *reinterpret_cast<float*>(&vals[r]);
       float raw1 = *reinterpret_cast<float*>(&vals[r+1]);

       float eg_col0 = (col0 < clen) ? __expf(g_mid - s_gc[col0]) : 0.0f;
       float eg_col1 = (col1 < clen) ? __expf(g_mid - s_gc[col1]) : 0.0f;

       // Causal mask: col <= row (lower triangular including diagonal)
       float gated0 = (col0 <= row && row < clen && col0 < clen) ? raw0 * eg_row * eg_col0 : 0.0f;
       float gated1 = (col1 <= row && row < clen && col1 < clen) ? raw1 * eg_row * eg_col1 : 0.0f;

       __nv_bfloat162 pair = {__float2bfloat16(gated0), __float2bfloat16(gated1)};

       // Write to swizzled s_attn_tile at (row, col0)
       int byte_off = swizzle_byte_offset_64x128(row, col0);
       *reinterpret_cast<__nv_bfloat162*>(s_attn_tile + byte_off) = pair;
     }
   }
   __syncthreads();

   // ═══════════════════════════════════════════════════════════════
   // STEP 4: Load h (cp.async into s_h_tile, overwriting k) + scatter vnew
   // ═══════════════════════════════════════════════════════════════
   {
     const __nv_bfloat16 *h_src = d_h + ((int64_t)chunk_id * kHv + hv) * kV * kK
                                  + v_start * kK;  // h[v_start, 0]
     uint32_t h_smem_base = cvt_smem_ptr(s_h_tile);
     int h_row_base = (tid >> 4) & 0xF;
     int h_col_start = (tid & 0xF) * 8;
     int h_base_off = swizzle_offset_q(tid);

     for (int batch = 0; batch < 4; batch++) {
       int row = h_row_base + batch * 16;
       cp_async_cg_128(h_smem_base + h_base_off + batch * 2048,
           &h_src[row * kK + h_col_start], row < kBV_O);
     }
     asm volatile("cp.async.commit_group;");
   }

   // Scatter-transpose vnew (global → smem, can't use cp.async easily)
   {
     const __nv_bfloat16 *vn_ptr = u_in + cstart * kHv * kV + hv * kV;
     scatter_transpose_64x64(vn_ptr, s_vnew_tile, v_start, kHv * kV, clen, tid);
   }

   // Wait for h cp.async
   asm volatile("cp.async.wait_group 0;");
   __syncthreads();
   asm volatile("fence.proxy.async.shared::cta;");

   // ═══════════════════════════════════════════════════════════════
   // STEP 5: MMA1: q[64,128] @ h[64,128]^T → qh[64,64] in TMEM
   // step=0 overwrites MMA_qk result (enable_input_d=0)
   // ═══════════════════════════════════════════════════════════════
   if (warp_id == 0 && elect_sync()) {
     // Re-init mbar1 for MMA1 commit
     asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(mbar1_addr));
     asm volatile("fence.mbarrier_init.release.cluster;");
   }
   __syncthreads();

   if (warp_id == 0 && elect_sync()) {
     uint32_t idesc = 0x04100490; // BM=64, BN=64, bf16→f32
     for (int step = 0; step < 8; step++) {
       int k_off = (step & 3) * 32 + (step >> 2) * 8192;
       uint64_t a_desc = make_swizzle_desc(smem_base + k_off);          // s_q
       uint64_t b_desc = make_swizzle_desc(smem_base + 16384 + k_off);  // s_h (now has h)
       tcgen05_mma(taddr, a_desc, b_desc, idesc, step);
     }
     tcgen05_commit(mbar1_addr);
   }

   // Wait for MMA1
   mbarrier_wait(mbar1_addr, 0);

   // ═══════════════════════════════════════════════════════════════
   // STEP 6: Read MMA1 TMEM → scale by exp(g) → write back to TMEM
   // ═══════════════════════════════════════════════════════════════
   {
     uint32_t shfl_warp;
     asm volatile("shfl.sync.idx.b32 %0, %1, 0, 31, -1;" : "=r"(shfl_warp) : "r"(warp_id));
     int tmem_off = ((shfl_warp << 21) & 6291456) | ((shfl_warp & 4) << 3);
     int my_taddr = tmem_off + taddr;

     uint32_t vals[16];
     tcgen05_ld_16x32bx2(vals, my_taddr);

     int row = (shfl_warp % 4) * 16 + (lane_id % 16);
     float eg = (row < clen) ? __expf(s_gc[row]) : 0.0f;

     uint32_t scaled[16];
     for (int i = 0; i < 16; i++) {
       float v = *reinterpret_cast<float*>(&vals[i]) * eg;
       scaled[i] = *reinterpret_cast<uint32_t*>(&v);
     }

     // Write scaled values back to TMEM
     tcgen05_st_16x32bx2(my_taddr, scaled);
   }

   // s_attn_tile and s_vnew_tile are already ready (written in steps 3+4)
   __syncthreads();

   // ═══════════════════════════════════════════════════════════════
   // STEP 7: MMA3: attn[64,64] @ vnew[64,64]^T → accumulate onto TMEM
   // ═══════════════════════════════════════════════════════════════
   if (warp_id == 0 && elect_sync()) {
     // Re-init mbar2 for MMA3 commit
     asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(mbar2_addr));
     asm volatile("fence.mbarrier_init.release.cluster;");
   }
   __syncthreads();

   if (warp_id == 0 && elect_sync()) {
     uint32_t idesc = 0x04100490; // BM=64, BN=64, bf16→f32
     for (int step = 0; step < 4; step++) {
       int k_off = step * 32;
       uint64_t a_desc = make_swizzle_desc(smem_base + 32768 + k_off);
       uint64_t b_desc = make_swizzle_desc(smem_base + 40960 + k_off);
       tcgen05_mma(taddr, a_desc, b_desc, idesc, 1);  // always accumulate
     }
     tcgen05_commit(mbar2_addr);
   }
   mbarrier_wait(mbar2_addr, 0);

   // ═══════════════════════════════════════════════════════════════
   // STEP 8: Read final TMEM → scale → write output
   // ═══════════════════════════════════════════════════════════════
   {
     uint32_t shfl_warp;
     asm volatile("shfl.sync.idx.b32 %0, %1, 0, 31, -1;" : "=r"(shfl_warp) : "r"(warp_id));
     int tmem_off = ((shfl_warp << 21) & 6291456) | ((shfl_warp & 4) << 3);
     int my_taddr = tmem_off + taddr;

     uint32_t vals[16];
     tcgen05_ld_16x32bx2(vals, my_taddr);

     // TMEM layout: row = (warp_id%4)*16 + (lane_id%16)
     //   col_base = (warp_id>=4 ? 32 : 0) + (lane_id>=16 ? 16 : 0)
     //   regs 0-15 → cols col_base..col_base+15
     int row = (shfl_warp % 4) * 16 + (lane_id % 16);
     int col_base = (shfl_warp >= 4 ? 32 : 0) + (lane_id >= 16 ? 16 : 0);

     if (row < clen) {
       __nv_bfloat16 *out_row = output + (cstart + row) * kHv * kV + hv * kV + v_start;
       for (int r = 0; r < 16; r += 2) {
         float v0 = *reinterpret_cast<float*>(&vals[r]);
         float v1 = *reinterpret_cast<float*>(&vals[r+1]);
         __nv_bfloat162 packed = {__float2bfloat16_rn(scale * v0),
                                  __float2bfloat16_rn(scale * v1)};
         *reinterpret_cast<__nv_bfloat162*>(&out_row[col_base + r]) = packed;
       }
     }
   }

   // ─── Dealloc TMEM ───
   __syncthreads();
   if (tid < 32) {
     asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(taddr), "r"(128));
   }
 }
 
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
   // qk_mat (d_A) eliminated — q@k^T now computed inside OOutputKernel
   auto align256 = [](size_t x) -> size_t { return (x + 255) & ~255ULL; };
   const size_t sz_g    = align256(T * kHv * sizeof(float));
   const size_t sz_w    = align256(T * kHv * kK * sizeof(__nv_bfloat16));
   const size_t sz_u    = align256(T * kHv * kV * sizeof(__nv_bfloat16));
   const size_t sz_h    = align256((size_t)max_chunks * kHv * kV * kK * sizeof(__nv_bfloat16));
   const size_t sz_ci   = align256(max_chunks * 2 * sizeof(int32_t));
   const size_t sz_tc   = align256(sizeof(int32_t));
   const size_t total_sz = sz_g + sz_w + sz_u + sz_h + sz_ci + sz_tc;
 
   // Grow persistent scratch if needed (never shrinks)
   if (total_sz > g_scratch_size) {
     if (g_scratch) cudaFree(g_scratch);
     cudaError_t me = cudaMalloc(&g_scratch, total_sz);
     if (me != cudaSuccess) {
       TVM_FFI_THROW(RuntimeError) << "scratch cudaMalloc(" << total_sz << ") failed: " << cudaGetErrorString(me);
     }
     g_scratch_size = total_sz;
   }
 
   // Partition scratch
   char *p = g_scratch;
   float *d_g = (float *)p; p += sz_g;
   __nv_bfloat16 *d_w = (__nv_bfloat16 *)p; p += sz_w;
   __nv_bfloat16 *d_u = (__nv_bfloat16 *)p; p += sz_u;
   __nv_bfloat16 *d_h = (__nv_bfloat16 *)p; p += sz_h;
   int32_t *d_ci = (int32_t *)p; p += sz_ci;  // unused (kept for API compat)
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
     // FusedPrepKernel: s_A_tiles(32KB) + s_B_tiles(16KB) + mbars(20B) + s_result(16KB) + s_result_inv(16KB) + s_g(256B) + s_beta(256B) + s_bg(256B) ≈ 81KB
     int smem_FP_attr = (128*128*2 + 64*128*2 + 256 + 64*64*4 + 64*64*4 + 64*4 + 64*4 + 64*4 + 1023) & ~1023;
     cudaFuncSetAttribute(FusedPrepKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_FP_attr);
     // HRecurrenceKernel: force 1 block/SM via 115KB smem pad (228KB/2 = 114KB)
     // H-kernel: actual ~59KB but pad to 115KB to force 1 block/SM (sequential recurrence
    // needs max registers; multi-block causes register pressure → 13% slower)
    int smem_H = 115 * 1024;
     cudaFuncSetAttribute(HRecurrenceKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_H);
     // OOutputKernel v9 (tcgen05 swizzled, inline q@k^T): s_q(16KB) + s_h(16KB) + s_attn(8KB) + s_vnew(8KB) + s_gc(256B) + mbars(24B) + tmem(4B) ≈ 49KB
     int smem_O = (16384 + 16384 + 8192 + 8192 + 256 + 24 + 4 + 1023) & ~1023;
     cudaFuncSetAttribute(OOutputKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_O);
   }
 
   // Create TMA descriptors (PrepMeta eliminated — inline chunk mapping)
   // q_tmap_128 removed — q@k^T moved to O-kernel (uses cp.async, not TMA)
   CUtensorMap k_tmap_128, k_tmap_64, w_tmap_128;
   init_tma_desc_3d(&k_tmap_128, k_p, (uint64_t)T, (uint64_t)(kHk * kK), 128, kK);
   init_tma_desc_3d(&k_tmap_64,  k_p, (uint64_t)T, (uint64_t)(kHk * kK), 64,  kK);
   init_tma_desc_3d(&w_tmap_128, d_w, (uint64_t)T, (uint64_t)(kHv * kK), 128, kK);
 
   // Compute smem sizes for launch
   int smem_FP = (128*128*2 + 64*128*2 + 256 + 64*64*4 + 64*64*4 + 64*4 + 64*4 + 64*4 + 1023) & ~1023;
   // H-kernel smem: 2×w(16KB) + h_T(4KB) + wh(4KB) + gc(256B) + k(16KB) + vnew_T(2KB) + u(2KB)
   int smem_H = (2*kBT*kK*2 + kK*kBV_H*2 + kBT*kBV_H*4 + kBT*4 + kBT*kK*2 + kBV_H*kBT*2 + kBT*kBV_H*2 + 1023) & ~1023;
   int smem_O = (16384 + 16384 + 8192 + 8192 + 256 + 24 + 4 + 1023) & ~1023;

   // Use upper-bound for grid — excess blocks exit early via inline chunk mapping
   const int64_t total_chunks = max_chunks;
 
   // Clear stale errors from previous calls
   cudaGetLastError();
 
   // 1+2+3. FusedPrepKernel — Preprocess + ComputeA + SolveTril + ComputeWU
   // Phase 4 (q@k^T) eliminated — moved to OOutputKernel
   FusedPrepKernel<<<dim3(total_chunks, kHv), 128, smem_FP, stream>>>(
       k_tmap_128, k_tmap_64, k_p, v_p,
       a_p, b_p, Alog_p, dtb_p, d_g,
       cusl_p, d_ci,
       d_w, d_u, d_tc, total_chunks, num_seqs);
 
   // H/O split path
   {
     HRecurrenceKernel<<<dim3(kNVT_H, num_seqs * kHv), 128, smem_H, stream>>>(
         d_w, k_p,
         d_u, d_g, state_ptr, cusl_p,
         d_h, ns_p, num_seqs);
 
     OOutputKernel<<<dim3(kNVT_O, total_chunks, kHv), 256, smem_O, stream>>>(
         q_p, k_p,
         d_u, d_h, d_g, cusl_p, d_ci,
         scale_f, out_p, d_tc, num_seqs);
   }
 
   const cudaError_t err = cudaGetLastError();
   if (err != cudaSuccess)
     TVM_FFI_THROW(RuntimeError)
         << "GdnPrefillTcgen05 failed: " << cudaGetErrorString(err);
 }
 
 } // namespace
 
 TVM_FFI_DLL_EXPORT_TYPED_FUNC(gdn_prefill_tcgen05, RunGdnPrefillTcgen05);
 
 
 