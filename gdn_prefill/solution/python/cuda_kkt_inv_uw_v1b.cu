// Fused kkt_v1b + inv_uw_v1 kernel.
//
// Math (per chunk, per head; BT=64, K_dim=V_dim=128):
//
//   beta   = sigmoid(b)                                    in  [BT]
//   g      = -exp(A_log) * softplus(a + dt_bias)           in  [BT]
//   g_cu   = cumsum(g) along chunk                         in  [BT]
//
//   KKT    = K @ K^T                                       in  [BT, BT]
//   A[i,j] = KKT[i,j] * beta[i] * exp(g_cu[i] - g_cu[j])   if i > j  (strictly lower tri)
//          = 0                                             otherwise
//
//   Ai     = (I + A)^{-1}                                  in  [BT, BT]
//
//   U      = (Ai * beta[j])               @ V              in  [BT, V_dim]
//   W      = (Ai * beta[j] * exp(g_cu[j])) @ K             in  [BT, K_dim]
//
// Inputs (gmem):  K, V, a, b, A_log, dt_bias
// Outputs (gmem): U, W (bf16), g_cu (fp32, used by downstream H/O kernels)
// Intermediate A lives only in smem — it does not round-trip through HBM.
//
// Warp specialization (10 warps = 320 threads per CTA):
//   warps 0..3 : EPI — epilogue warps. Drain U/W from tmem, pack fp32→bf16 into
//                smem in swizzled layout, then TMA-store U and W to gmem.
//   warps 4..7 : INV — compute warps. Do beta/g_cu prep, apply KKT mask into A,
//                invert (I+A), produce Ab/Abg in tmem for the U/W MMA.
//   warp 8     : MMA — single warp whose elected lane issues tcgen05.mma and
//                tcgen05.commit for KKT (MMA #1) and U/W (MMA #2).
//   warp 9     : TMA — single warp whose elected lane issues tma_load for K, V
//                and drives the TMA->smem pipeline stages.
//
// How it works (per chunk, steady state):
//   TMA:    load K, V                            [mma_mbar wait → tma_mbar arrive]
//   MMA #1: KKT = K @ K^T                        [tma_mbar + epi_mbar wait → kkt_mbar arrive]
//   INV P1: beta, g_cu from a, b, A_log          (parallel with KKT MMA)
//             beta  = sigmoid(b), writes to beta_smem
//             g_cu  = cumsum(-exp(A_log)*softplus(a+dt_bias)) via warp scan
//             stores g_cu to gmem + exp(g_cu) to g_cu_smem
//   INV P2: wait kkt_mbar, tcgen05_ld KKT from tmem into regs,
//           mask (strict-lower + time-valid) and scale by beta·exp(g-g.T),
//           sts_b32x4 the bf16 result into A_smem in ldmatrix-swizzled layout
//   INV P3: block-triangular inverse of (I+A) in smem
//             - each warp owns one diagonal [16,16] tile: Newton-Schulz (3 iters)
//             - off-diag by 1 (warps 1..3): Ai_{i,i-1} = -Ai_{i,i} @ A_{i,i-1} @ Ai_{i-1,i-1}
//             - off-diag by 2 (warps 0..1): Ai_{2,0}, Ai_{3,1} with 2-term sum
//             - off-diag by 3 (warp 0):     Ai_{3,0} with 3-term sum
//   INV P4: Ab = Ai*β[j], Abg = Ab*exp(g_cu[j])  [mma_mbar wait (prev) → inv_mbar arrive]
//             ldmatrix Ai_smem → scale by beta/g_cu → tcgen05_st into Ab/Abg tmem
//   MMA #2: U = Ab@V, W = Abg@K                  [inv_mbar wait → mma_mbar arrive]
//             tcgen05_mma_tmem: A-operand lives in tmem (Ab/Abg), B in smem (V/K)
//   EPI:    tmem → smem → TMA store U, W         [mma_mbar wait → epi_mbar arrive]
//             tcgen05_ld U/W (fp32) → fp32x2_to_bf16x2 pack → stmatrix into
//             swizzled U_smem/W_smem → tma_store_4d to gmem
//
// Fusion helps in:
//   (a) A stays in smem — no HBM round-trip for the per-chunk 64x64 attention-like matrix.
//   (b) bf16 packing + swizzle happen in one pass directly after tmem read
//       (A_smem is written in the layout ldmatrix expects for Phase 3).
//   (c) TMEM aliasing keeps footprint tight: KKT output aliases the low 64 cols
//       of U's tmem slot; U overwrites once inv has drained KKT into A_smem.
//   (d) beta/g prep overlaps with KKT MMA instead of running in its own kernel.

#include <cuda_bf16.h>
#include <cudaTypedefs.h>
#include <cstdint>
#include "cuda_utils.h"

__host__ __device__ inline
int cdiv_fuse(int a, int b) { return (a + b - 1) / b; }

__device__ inline
uint32_t fp32x2_to_bf16x2_fuse(float a, float b) {
  uint32_t tmp;
  asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(tmp) : "f"(b), "f"(a));
  return tmp;
}
__device__ inline
void bf16x2_to_fp32x2_fuse(float *out, uint32_t data) {
  asm volatile("shl.b32 %0, %2, 16;\n"
               "and.b32 %1, %2, 0xFFFF0000;"
              : "=f"(out[0]), "=f"(out[1]) : "r"(data));
}

__device__ inline
void lds_f32x2_fuse(float *data, uint32_t addr) {
  asm volatile("ld.shared.v2.f32 {%0, %1}, [%2];" : "=f"(data[0]), "=f"(data[1]) : "r"(addr));
};

__device__ inline
void sts_b32x4_fuse(uint32_t addr, const uint32_t *data) {
  asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
              :: "r"(addr),
                 "r"(data[0]), "r"(data[1]), "r"(data[2]), "r"(data[3]));
};

__device__ inline
void sts_b32_fuse(uint32_t addr, uint32_t data) {
  asm volatile("st.shared.b32 [%0], %1;" :: "r"(addr), "r"(data));
};

__device__ __forceinline__
void mma_bf16_fuse(float d[4], uint32_t a[4], uint32_t b[2], float c[4]) {
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
    : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
      "r"(b[0]), "r"(b[1]),
      "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

constexpr int H = 8;
constexpr int Hg = 4;
constexpr int BT = 64;
constexpr int K_dim = 128;
constexpr int V_dim = 128;

constexpr int A_size = BT * BT * sizeof(nv_bfloat16);
constexpr int K_size = BT * K_dim * sizeof(nv_bfloat16);
constexpr int V_size = BT * V_dim * sizeof(nv_bfloat16);
constexpr int STAGE_SIZE = A_size + K_size + V_size;

constexpr int NUM_WARPS = 4 + 4 + 2;
constexpr int WARP_SIZE = 32;
constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;

template <int NUM_STAGES>
__global__
__block_size__((TB_SIZE, 1, 1))
void kkt_inv_uw_v1b_kernel_cutlass(
  const __grid_constant__ CUtensorMap K_tmap,  // [total_T, Hg, K_dim]
  const __grid_constant__ CUtensorMap V_tmap,  // [total_T, H, V_dim]
  const __grid_constant__ CUtensorMap U_tmap,  // [pad_T, H, V_dim]
  const __grid_constant__ CUtensorMap W_tmap,  // [pad_T, H, V_dim]
  const float       *A_log_ptr,                // [H]
  const nv_bfloat16 *a_ptr,                    // [total_T, H]
  const float       *dt_bias_ptr,              // [H]
  const nv_bfloat16 *b_ptr,                    // [total_T, H]
        float       *g_cu_ptr,                 // [total_T, H]
  const int64_t     *cu_seqlens_ptr,           // [N+1]
  const int32_t     *chunk_indices_ptr,        // [total_num_chunks, 2]
  const int32_t     *total_chunks_ptr
) {
  const int tid = threadIdx.x;
  const int warp_id = warp_uniform(tid / WARP_SIZE);
  const int lane_id = tid % WARP_SIZE;

  const int head_id = blockIdx.x;
  const int bid = blockIdx.y;
  const int total_chunks = total_chunks_ptr[0];

  extern __shared__ __align__(1024) char smem_ptr[];
  const uint32_t smem = __cvta_generic_to_shared(smem_ptr);

  const uint32_t Ai_smem       = smem + NUM_STAGES * STAGE_SIZE;
  const uint32_t U_smem        = Ai_smem + A_size;
  const uint32_t W_smem        = U_smem + V_size;
  const uint32_t beta_smem     = W_smem + K_size;
  const uint32_t g_cu_smem     = beta_smem + BT * (uint32_t)sizeof(float);
  const uint32_t g_raw_smem    = g_cu_smem + BT * (uint32_t)sizeof(float);
  const uint32_t g_scratch_smem = g_raw_smem + BT * (uint32_t)sizeof(float);

  const uint32_t tma_mbar = g_scratch_smem + 16 * (uint32_t)sizeof(float);
  const uint32_t kkt_mbar = tma_mbar + NUM_STAGES * 8;
  const uint32_t inv_mbar = kkt_mbar + NUM_STAGES * 8;
  const uint32_t mma_mbar = inv_mbar + NUM_STAGES * 8;
  const uint32_t epi_mbar = mma_mbar + NUM_STAGES * 8;

  const uint32_t taddr = epi_mbar + NUM_STAGES * 8;

  // KKT aliases U's low 64 cols. U eventually writes full 128 cols, overwriting.
  const uint32_t U_tmem  = 0;
  const uint32_t Ab_tmem = U_tmem + V_dim * NUM_STAGES;

  if (warp_id == 0) {
    if (elect_sync()) {
      for (int i = 0; i < NUM_STAGES; i++) {
        mbarrier_init(tma_mbar + i * 8, 1);
        mbarrier_init(kkt_mbar + i * 8, 1);
        mbarrier_init(inv_mbar + i * 8, 128);
        mbarrier_init(mma_mbar + i * 8, 1);
        mbarrier_init(epi_mbar + i * 8, 128);
      }
      fence_mbarrier_init();
    }
  }
  else if (warp_id == 1) {
    if (elect_sync()) {
      prefetch_tensormap(&V_tmap);
      prefetch_tensormap(&K_tmap);
    }
  }
  __syncthreads();

  if (warp_id == NUM_WARPS - 1) {
    // TMA warp
    if (elect_sync()) {
      int stage_id = 0;
      int parity = 1;

      const int k_head_id = head_id / (H / Hg);

      for (int global_chunk_id = bid; global_chunk_id < total_chunks; global_chunk_id += gridDim.y) {
        const int2 tmp = reinterpret_cast<const int2 *>(chunk_indices_ptr)[global_chunk_id];
        const int seq_id = tmp.x;
        const int chunk_id = tmp.y;
        const int bos = cu_seqlens_ptr[seq_id];

        const uint32_t A_smem = smem + stage_id * STAGE_SIZE;
        const uint32_t V_smem = A_smem + A_size;
        const uint32_t K_smem = V_smem + V_size;

        const int off_t = bos + chunk_id * BT;
        const int mbar = tma_mbar + stage_id * 8;

        mbarrier_wait(mma_mbar + stage_id * 8, parity);

        tma_load_4d(V_smem, &V_tmap, 0, off_t, 0, head_id, mbar, EVICT_FIRST);
        tma_load_4d(K_smem, &K_tmap, 0, off_t, 0, k_head_id, mbar);
        mbarrier_arrive_expect_tx(mbar, K_size + V_size);

        stage_id = (stage_id + 1) % NUM_STAGES;
        if (stage_id == 0)
          parity ^= 1;
      }
    }
  }
  else if (warp_id == NUM_WARPS - 2) {
    // MMA warp
    tcgen05_alloc(taddr, 512);

    if (elect_sync()) {
      int stage_id = 0;
      int tma_parity = 0;
      int epi_parity = 1;

      for (int global_chunk_id = bid; global_chunk_id < total_chunks; global_chunk_id += gridDim.y) {
        const uint32_t A_smem = smem + stage_id * STAGE_SIZE;
        const uint32_t V_smem = A_smem + A_size;
        const uint32_t K_smem = V_smem + V_size;

        const uint32_t this_U_tmem = U_tmem + V_dim * stage_id;
        const uint32_t this_W_tmem = this_U_tmem | (16U << 16U);
        const uint32_t this_Ab_tmem  = Ab_tmem + (BT / 2) * stage_id;
        const uint32_t this_Abg_tmem = this_Ab_tmem | (16U << 16U);
        // KKT aliases U slot
        const uint32_t this_kkt_tmem = this_U_tmem;

        // --- MMA #1: KKT = K @ K^T ---
        // Wait for TMA (K) ready AND for prev iter's U/W drained by epi.
        // epi_mbar guards the aliased U tmem cols.
        mbarrier_wait(tma_mbar + stage_id * 8, tma_parity);
        mbarrier_wait(epi_mbar + stage_id * 8, epi_parity);
        tcgen05_fence_after_thread_sync();

        {
          constexpr uint32_t kkt_idesc = make_tcgen05_idesc(BT, BT);
          constexpr uint64_t desc_base = (desc_encode(8 * 128) << 32ULL)
                                        | (1ULL << 46ULL) | (2ULL << 61ULL);

          for (int i = 0; i < K_dim / 64; i++)
            for (int j = 0; j < 64 / 16; j++) {
              const uint64_t k_desc = desc_base | ((K_smem + i * BT * 128 + j * 32) >> 4);
              const int enable_input = (i > 0) || (j > 0);
              tcgen05_mma(this_kkt_tmem, k_desc, k_desc, kkt_idesc, enable_input);
            }
        }
        tcgen05_commit(kkt_mbar + stage_id * 8);

        // --- MMA #2: U = Ab @ V, W = Abg @ K ---
        // Wait inv_mbar: inv has finished reading KKT AND produced Ab/Abg.
        mbarrier_wait(inv_mbar + stage_id * 8, tma_parity);
        tcgen05_fence_after_thread_sync();

        constexpr uint32_t u_idesc = make_tcgen05_idesc(BT, V_dim) | (1U << 16U);
        constexpr uint32_t w_idesc = make_tcgen05_idesc(BT, K_dim) | (1U << 16U);

        constexpr uint64_t sdesc_base = (desc_encode(BT * 128) << 16ULL)
                                      | (desc_encode(8 * 128) << 32ULL)
                                      | (1ULL << 46ULL) | (2ULL << 61ULL);

        for (int i = 0; i < BT / 16; i++) {
          const uint64_t v_desc = sdesc_base | ((V_smem + i * 16 * 128) >> 4U);
          const uint64_t k_desc = sdesc_base | ((K_smem + i * 16 * 128) >> 4U);
          tcgen05_mma_tmem(this_U_tmem, this_Ab_tmem + i * 8, v_desc, u_idesc, i > 0);
          tcgen05_mma_tmem(this_W_tmem, this_Abg_tmem + i * 8, k_desc, w_idesc, i > 0);
        }
        tcgen05_commit(mma_mbar + stage_id * 8);

        stage_id = (stage_id + 1) % NUM_STAGES;
        if (stage_id == 0) {
          tma_parity ^= 1;
          epi_parity ^= 1;
        }
      }
    }
  }
  else if (warp_id >= 4) {
    // inv warps (CUDA compute)
    int tid_ = tid % 128;
    int warp_id_ = warp_id % 4;

    int stage_id = 0;
    int kkt_parity = 0;
    int mma_parity = 0;

    float *beta_smem_ptr      = reinterpret_cast<float *>(smem_ptr + (beta_smem      - smem));
    float *g_cu_smem_ptr      = reinterpret_cast<float *>(smem_ptr + (g_cu_smem      - smem));
    float *g_raw_smem_ptr     = reinterpret_cast<float *>(smem_ptr + (g_raw_smem     - smem));
    float *g_scratch_smem_ptr = reinterpret_cast<float *>(smem_ptr + (g_scratch_smem - smem));

    const float A_log_val   = -__expf(A_log_ptr[head_id]);
    const float dt_bias_val = dt_bias_ptr[head_id];

    // zero Ai_smem top [48,64]
    for (int i = 0; i < 3; i++) {
      uint32_t zeros[4] = {};
      const uint32_t row = i * 16 + warp_id_ * 4 + (lane_id / 8);
      const uint32_t col = lane_id % 8;
      sts_b32x4_fuse(Ai_smem + row * 128 + col * 16, zeros);
    }

    for (int global_chunk_id = bid; global_chunk_id < total_chunks; global_chunk_id += gridDim.y) {
      const int2 tmp = reinterpret_cast<const int2 *>(chunk_indices_ptr)[global_chunk_id];
      const int seq_id = tmp.x;
      const int chunk_id = tmp.y;
      const int bos = cu_seqlens_ptr[seq_id];
      const int eos = cu_seqlens_ptr[seq_id + 1];

      const uint32_t A_smem = smem + stage_id * STAGE_SIZE;

      // -------- Phase 1: beta, g_cu --------
      const int my_row = warp_id_ * 16 + (lane_id % 16);
      const int off_t = bos + chunk_id * BT + my_row;
      const bool in_range = off_t < eos;
      const bool is_active = (lane_id < 16);

      float b_val = 0.0f, a_val = 0.0f;
      if (is_active && in_range) {
        b_val = __bfloat162float(b_ptr[off_t * H + head_id]);
        a_val = __bfloat162float(a_ptr[off_t * H + head_id]);
      }

      float beta_val = __frcp_rn(1.0f + __expf(-b_val));
      float g_val    = A_log_val * __logf(1.0f + __expf(a_val + dt_bias_val));
      if (!in_range) g_val = 0.0f;

      if (is_active) {
        beta_smem_ptr[my_row] = beta_val;
      }

      // parallel scan g among lower 16 lanes
      for (int i = 1; i < 16; i *= 2) {
        float lower = __shfl_up_sync(0xFFFF'FFFF, g_val, i);
        if ((lane_id % 16) >= i)
          g_val += lower;
      }
      if (is_active && (lane_id % 16) == 15) {
        g_scratch_smem_ptr[warp_id_] = g_val;
      }
      bar_sync<1>(128);

      if (is_active) {
        if (warp_id_ >= 1) g_val += g_scratch_smem_ptr[0];
        if (warp_id_ >= 2) g_val += g_scratch_smem_ptr[1];
        if (warp_id_ >= 3) g_val += g_scratch_smem_ptr[2];

        if (in_range) {
          g_cu_ptr[off_t * H + head_id] = g_val;
        }
        g_raw_smem_ptr[my_row] = g_val;
        g_cu_smem_ptr[my_row]  = in_range ? __expf(g_val) : 0.0f;
      }
      bar_sync<1>(128);

      // -------- Phase 2: wait KKT, mask, write A to A_smem --------
      if (warp_id_ == 0)
        mbarrier_wait(kkt_mbar + stage_id * 8, kkt_parity);
      bar_sync<1>(128);
      tcgen05_fence_after_thread_sync();

      // KKT at U_tmem + V_dim*stage_id (low 64 cols)
      const uint32_t kkt_tmem_addr = U_tmem + V_dim * stage_id;
      float kkt_lo[16], kkt_hi[16];
      tcgen05_ld<SHAPE::_16x256b, 4>(kkt_lo, 0, kkt_tmem_addr);
      tcgen05_ld<SHAPE::_16x256b, 4>(kkt_hi, 0, kkt_tmem_addr + 32);
      tcgen05_wait_ld();
      tcgen05_fence_before_thread_sync();

      const int row_base = warp_id_ * 16 + lane_id / 4;
      const int row_hi = row_base + 8;
      const int lane_col = lane_id % 4;
      const float beta_base = beta_smem_ptr[row_base];
      const float beta_hi = beta_smem_ptr[row_hi];
      const float g_base = g_raw_smem_ptr[row_base];
      const float g_hi = g_raw_smem_ptr[row_hi];
      const bool row_base_valid = bos + chunk_id * BT + row_base < eos;
      const bool row_hi_valid = bos + chunk_id * BT + row_hi < eos;

      auto store_pair = [&](int row, bool row_valid, float beta_row, float g_row,
                            int col, float v0, float v1) {
        if (!(row_valid && row > col && bos + chunk_id * BT + col < eos)) {
          v0 = 0.0f;
          v1 = 0.0f;
        } else {
          const float col_g0 = g_raw_smem_ptr[col];
          const float col_g1 = g_raw_smem_ptr[col + 1];
          v0 *= beta_row * __expf(g_row - col_g0);
          v1 *= beta_row * __expf(g_row - col_g1);
        }

        const int bank = col / 8;
        const int pair = (col & 7) / 2;
        const int phys_bank = bank ^ (row & 7);
        const uint32_t addr =
            A_smem + row * 128 + phys_bank * 16 + pair * 4;
        sts_b32_fuse(addr, fp32x2_to_bf16x2_fuse(v0, v1));
      };

      #pragma unroll
      for (int step_pair = 0; step_pair < 4; ++step_pair) {
        const int reg_base = step_pair * 4;
        const int col_lo = step_pair * 8 + 2 * lane_col;
        const int col_hi = col_lo + 32;

        store_pair(row_base, row_base_valid, beta_base, g_base, col_lo,
                   kkt_lo[reg_base + 0], kkt_lo[reg_base + 1]);
        store_pair(row_hi, row_hi_valid, beta_hi, g_hi, col_lo,
                   kkt_lo[reg_base + 2], kkt_lo[reg_base + 3]);
        store_pair(row_base, row_base_valid, beta_base, g_base, col_hi,
                   kkt_hi[reg_base + 0], kkt_hi[reg_base + 1]);
        store_pair(row_hi, row_hi_valid, beta_hi, g_hi, col_hi,
                   kkt_hi[reg_base + 2], kkt_hi[reg_base + 3]);
      }
      bar_sync<1>(128);

      // -------- Phase 3: Newton-Schulz inverse --------
      auto compute_offset = [&](uint32_t row16, uint32_t col16) {
        const uint32_t row = row16 * 16 + (lane_id % 16);
        const uint32_t col = (col16 * 2 + (lane_id / 16)) ^ (lane_id % 8);
        return row * 128 + col * 16;
      };

      auto set_diagonal_bf16 = [&](uint32_t *A) {
        if (lane_id % 9 == 0) {
          A[0] = (A[0] & 0xFFFF0000U) | 0x3F80U;
          A[3] = (A[3] & 0xFFFF0000U) | 0x3F80U;
        }
        else if (lane_id % 9 == 4) {
          A[0] = (A[0] & 0xFFFFU) | 0x3F800000U;
          A[3] = (A[3] & 0xFFFFU) | 0x3F800000U;
        }
      };

      uint32_t Ai[4], mma_B[4], M[4];
      float acc[8], zeros[4] = {}, Ai_f32[8];

      constexpr int NUM_NEWTON = 3;

      const uint32_t diag_addr = A_smem + compute_offset(warp_id_, warp_id_);
      ldmatrix<4>(Ai, diag_addr);
      for (int i = 0; i < 4; i++) Ai[i] ^= 0x80008000U;
      set_diagonal_bf16(Ai);
      for (int i = 0; i < 4; i++)
        bf16x2_to_fp32x2_fuse(Ai_f32 + i * 2, Ai[i]);

      ldmatrix_trans<4>(M, diag_addr);
      set_diagonal_bf16(M);
      for (int i = 0; i < 4; i++) M[i] ^= 0x80008000U;

      for (int i = 0; i < NUM_NEWTON; i++) {
        stmatrix<4>(diag_addr, Ai);
        __syncwarp();
        mma_bf16_fuse(acc + 0, Ai, M + 0, zeros);
        mma_bf16_fuse(acc + 4, Ai, M + 2, zeros);
        for (int j = 0; j < 4; j++)
          Ai[j] = fp32x2_to_bf16x2_fuse(acc[j * 2], acc[j * 2 + 1]);

        for (int j = 0; j < 8; j++) Ai_f32[j] *= 2.0f;
        ldmatrix_trans<4>(mma_B, diag_addr);
        mma_bf16_fuse(Ai_f32 + 0, Ai, mma_B + 0, Ai_f32 + 0);
        mma_bf16_fuse(Ai_f32 + 4, Ai, mma_B + 2, Ai_f32 + 4);
        for (int j = 0; j < 4; j++)
          Ai[j] = fp32x2_to_bf16x2_fuse(Ai_f32[j * 2], Ai_f32[j * 2 + 1]);
      }

      stmatrix<4>(Ai_smem + compute_offset(warp_id_, warp_id_), Ai);
      bar_sync<1>(128);

      // off-diagonal by 1
      if (warp_id_ > 0) {
        for (int i = 0; i < 4; i++) Ai[i] ^= 0x80008000U;
        ldmatrix_trans<4>(mma_B, A_smem + compute_offset(warp_id_, warp_id_ - 1));
        mma_bf16_fuse(acc + 0, Ai, mma_B + 0, zeros);
        mma_bf16_fuse(acc + 4, Ai, mma_B + 2, zeros);
        for (int i = 0; i < 4; i++)
          Ai[i] = fp32x2_to_bf16x2_fuse(acc[i * 2], acc[i * 2 + 1]);

        ldmatrix_trans<4>(mma_B, Ai_smem + compute_offset(warp_id_ - 1, warp_id_ - 1));
        mma_bf16_fuse(acc + 0, Ai, mma_B + 0, zeros);
        mma_bf16_fuse(acc + 4, Ai, mma_B + 2, zeros);
        for (int i = 0; i < 4; i++)
          Ai[i] = fp32x2_to_bf16x2_fuse(acc[i * 2], acc[i * 2 + 1]);

        stmatrix<4>(Ai_smem + compute_offset(warp_id_, warp_id_ - 1), Ai);
      }
      bar_sync<1>(128);

      // off-diagonal by 2
      if (warp_id_ < 2) {
        ldmatrix<4>(Ai, A_smem + compute_offset(warp_id_ + 2, warp_id_));
        ldmatrix_trans<4>(mma_B, Ai_smem + compute_offset(warp_id_, warp_id_));
        mma_bf16_fuse(acc + 0, Ai, mma_B + 0, zeros);
        mma_bf16_fuse(acc + 4, Ai, mma_B + 2, zeros);

        ldmatrix<4>(Ai, A_smem + compute_offset(warp_id_ + 2, warp_id_ + 1));
        ldmatrix_trans<4>(mma_B, Ai_smem + compute_offset(warp_id_ + 1, warp_id_));
        mma_bf16_fuse(acc + 0, Ai, mma_B + 0, acc + 0);
        mma_bf16_fuse(acc + 4, Ai, mma_B + 2, acc + 4);

        for (int i = 0; i < 4; i++)
          mma_B[i] = fp32x2_to_bf16x2_fuse(acc[i * 2], acc[i * 2 + 1]);
        stmatrix<4>(Ai_smem + compute_offset(warp_id_ + 2, warp_id_), mma_B);
        __syncwarp();

        ldmatrix<4>(Ai, Ai_smem + compute_offset(warp_id_ + 2, warp_id_ + 2));
        for (int i = 0; i < 4; i++) Ai[i] ^= 0x80008000U;
        ldmatrix_trans<4>(mma_B, Ai_smem + compute_offset(warp_id_ + 2, warp_id_));
        mma_bf16_fuse(acc + 0, Ai, mma_B + 0, zeros);
        mma_bf16_fuse(acc + 4, Ai, mma_B + 2, zeros);
        for (int i = 0; i < 4; i++)
          mma_B[i] = fp32x2_to_bf16x2_fuse(acc[i * 2], acc[i * 2 + 1]);
        stmatrix<4>(Ai_smem + compute_offset(warp_id_ + 2, warp_id_), mma_B);
      }
      bar_sync<1>(128);

      // off-diagonal by 3
      if (warp_id_ == 0) {
        ldmatrix<4>(Ai, A_smem + compute_offset(3, 0));
        ldmatrix_trans<4>(mma_B, Ai_smem + compute_offset(0, 0));
        mma_bf16_fuse(acc + 0, Ai, mma_B + 0, zeros);
        mma_bf16_fuse(acc + 4, Ai, mma_B + 2, zeros);

        ldmatrix<4>(Ai, A_smem + compute_offset(3, 1));
        ldmatrix_trans<4>(mma_B, Ai_smem + compute_offset(1, 0));
        mma_bf16_fuse(acc + 0, Ai, mma_B + 0, acc + 0);
        mma_bf16_fuse(acc + 4, Ai, mma_B + 2, acc + 4);

        ldmatrix<4>(Ai, A_smem + compute_offset(3, 2));
        ldmatrix_trans<4>(mma_B, Ai_smem + compute_offset(2, 0));
        mma_bf16_fuse(acc + 0, Ai, mma_B + 0, acc + 0);
        mma_bf16_fuse(acc + 4, Ai, mma_B + 2, acc + 4);

        for (int i = 0; i < 4; i++)
          mma_B[i] = fp32x2_to_bf16x2_fuse(acc[i * 2], acc[i * 2 + 1]);
        stmatrix<4>(Ai_smem + compute_offset(3, 0), mma_B);
        __syncwarp();

        ldmatrix<4>(Ai, Ai_smem + compute_offset(3, 3));
        for (int i = 0; i < 4; i++) Ai[i] ^= 0x80008000U;
        ldmatrix_trans<4>(mma_B, Ai_smem + compute_offset(3, 0));
        mma_bf16_fuse(acc + 0, Ai, mma_B + 0, zeros);
        mma_bf16_fuse(acc + 4, Ai, mma_B + 2, zeros);
        for (int i = 0; i < 4; i++)
          mma_B[i] = fp32x2_to_bf16x2_fuse(acc[i * 2], acc[i * 2 + 1]);
        stmatrix<4>(Ai_smem + compute_offset(3, 0), mma_B);
      }

      // -------- Phase 4: Compute Ab, Abg → tmem --------
      if (warp_id_ == 3)
        mbarrier_wait(mma_mbar + stage_id * 8, mma_parity ^ 1);
      bar_sync<1>(128);

      for (int i = 0; i < 64 / 16; i++) {
        uint32_t Ai2[4], Ab[4], Abg[4];
        float beta[4], g_cu[4];
        ldmatrix<4>(Ai2, Ai_smem + compute_offset(warp_id_, i));

        const int col = i * 16 + (lane_id % 4) * 2;
        lds_f32x2_fuse(beta + 0, beta_smem + (col + 0) * 4);
        lds_f32x2_fuse(beta + 2, beta_smem + (col + 8) * 4);
        lds_f32x2_fuse(g_cu + 0, g_cu_smem + (col + 0) * 4);
        lds_f32x2_fuse(g_cu + 2, g_cu_smem + (col + 8) * 4);

        float tmp[8];
        for (int j = 0; j < 4; j++) {
          bf16x2_to_fp32x2_fuse(tmp + j * 2, Ai2[j]);
          tmp[j * 2 + 0] *= beta[j / 2 * 2 + 0];
          tmp[j * 2 + 1] *= beta[j / 2 * 2 + 1];
          Ab[j] = fp32x2_to_bf16x2_fuse(tmp[j * 2], tmp[j * 2 + 1]);

          tmp[j * 2 + 0] *= g_cu[j / 2 * 2 + 0];
          tmp[j * 2 + 1] *= g_cu[j / 2 * 2 + 1];
          Abg[j] = fp32x2_to_bf16x2_fuse(tmp[j * 2], tmp[j * 2 + 1]);
        }

        const uint32_t this_Ab_tmem = Ab_tmem + (BT / 2) * stage_id + i * 8;
        tcgen05_st<SHAPE::_16x128b, 2>(warp_id_ * 32, this_Ab_tmem, Ab);
        tcgen05_st<SHAPE::_16x128b, 2>(warp_id_ * 32 + 16, this_Ab_tmem, Abg);
      }
      tcgen05_wait_st();
      tcgen05_fence_before_thread_sync();
      mbarrier_arrive(inv_mbar + stage_id * 8);

      stage_id = (stage_id + 1) % NUM_STAGES;
      if (stage_id == 0) {
        kkt_parity ^= 1;
        mma_parity ^= 1;
      }
    }
  }
  else {
    // epilogue warps
    int stage_id = 0;
    int parity = 0;

    for (int global_chunk_id = bid; global_chunk_id < total_chunks; global_chunk_id += gridDim.y) {
      const uint32_t this_U_tmem = U_tmem + V_dim * stage_id;

      if (warp_id == 0)
        mbarrier_wait(mma_mbar + stage_id * 8, parity);
      else if (warp_id == 1) {
        if (elect_sync())
          cp_async_bulk_wait_group_read<0>();
      }
      bar_sync<2>(128);
      tcgen05_fence_after_thread_sync();

      float u_tmp[V_dim / 2];
      tcgen05_ld<SHAPE::_16x256b, V_dim / 8>(u_tmp, warp_id * 32, this_U_tmem);
      for (int i = 0; i < V_dim / 64; i++)
        for (int j = 0; j < 64 / 16; j++) {
          uint32_t tmp[4];
          for (int k = 0; k < 4; k++)
            tmp[k] = fp32x2_to_bf16x2_fuse(u_tmp[i * 32 + j * 8 + k * 2],
                                      u_tmp[i * 32 + j * 8 + k * 2 + 1]);

          const int row = warp_id * 16 + (lane_id % 16);
          const int col = (j * 2 + (lane_id / 16)) ^ (lane_id % 8);
          stmatrix<4>(U_smem + i * BT * 128 + row * 128 + col * 16, tmp);
        }

      float w_tmp[K_dim / 2];
      tcgen05_ld<SHAPE::_16x256b, K_dim / 8>(w_tmp, warp_id * 32, this_U_tmem | (16U << 16U));
      tcgen05_fence_before_thread_sync();
      mbarrier_arrive(epi_mbar + stage_id * 8);
      for (int i = 0; i < K_dim / 64; i++)
        for (int j = 0; j < 64 / 16; j++) {
          uint32_t tmp[4];
          for (int k = 0; k < 4; k++)
            tmp[k] = fp32x2_to_bf16x2_fuse(w_tmp[i * 32 + j * 8 + k * 2],
                                      w_tmp[i * 32 + j * 8 + k * 2 + 1]);

          const int row = warp_id * 16 + (lane_id % 16);
          const int col = (j * 2 + (lane_id / 16)) ^ (lane_id % 8);
          stmatrix<4>(W_smem + i * BT * 128 + row * 128 + col * 16, tmp);
        }

      bar_sync<2>(128);
      asm volatile("fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster;");
      if (warp_id == 1 && elect_sync()) {
        tma_store_4d(&U_tmap, U_smem, 0, global_chunk_id * BT, 0, head_id);
        tma_store_4d(&W_tmap, W_smem, 0, global_chunk_id * BT, 0, head_id);
        cp_async_bulk_commit_group();
      }

      stage_id = (stage_id + 1) % NUM_STAGES;
      if (stage_id == 0)
        parity ^= 1;
    }
  }

  __syncthreads();
  if (warp_id == 0)
    tcgen05_dealloc(0, 512);
}

static
CUtensorMap encode_tma_kkt_fuse(void *ptr, uint64_t T, uint64_t H_, uint64_t dim) {
  CUtensorMap tmap;

  constexpr uint32_t rank = 4;
  uint64_t globalDim[rank] = {64, T, dim / 64, H_};
  uint64_t globalStrides[rank - 1] = {H_ * dim * sizeof(nv_bfloat16),
                                                                128,
                                          dim * sizeof(nv_bfloat16)};
  uint32_t boxDim[rank] = {64, BT, dim / 64, 1};
  uint32_t elementStrides[rank] = {1, 1, 1, 1};

  cuTensorMapEncodeTiled(
    &tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, ptr,
    globalDim, globalStrides, boxDim, elementStrides,
    CU_TENSOR_MAP_INTERLEAVE_NONE,
    CU_TENSOR_MAP_SWIZZLE_128B,
    CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return tmap;
}

void kkt_inv_uw_v1b(
  TensorView K,
  TensorView V,
  TensorView U,
  TensorView W,
  TensorView A_log,
  TensorView a,
  TensorView dt_bias,
  TensorView b,
  TensorView g_cu,
  TensorView cu_seqlens,
  TensorView chunk_indices,
  TensorView total_chunks
) {
  const int T = K.size(0);

  auto K_tmap = encode_tma_kkt_fuse(K.data_ptr(), T, Hg, K_dim);
  auto V_tmap = encode_tma_kkt_fuse(V.data_ptr(), T, H, V_dim);
  auto U_tmap = encode_tma_kkt_fuse(U.data_ptr(), U.size(0), H, V_dim);
  auto W_tmap = encode_tma_kkt_fuse(W.data_ptr(), W.size(0), H, K_dim);

  auto *A_log_ptr   = reinterpret_cast<const float *>(A_log.data_ptr());
  auto *a_ptr       = reinterpret_cast<const nv_bfloat16 *>(a.data_ptr());
  auto *dt_bias_ptr = reinterpret_cast<const float *>(dt_bias.data_ptr());
  auto *b_ptr       = reinterpret_cast<const nv_bfloat16 *>(b.data_ptr());
  auto *g_cu_ptr    = reinterpret_cast<float *>(g_cu.data_ptr());
  auto *cu_seqlens_ptr    = reinterpret_cast<int64_t *>(cu_seqlens.data_ptr());
  auto *chunk_indices_ptr = reinterpret_cast<int32_t *>(chunk_indices.data_ptr());
  auto *total_chunks_ptr  = reinterpret_cast<int32_t *>(total_chunks.data_ptr());

  constexpr int NUM_STAGES = 3;
  constexpr int smem_size = NUM_STAGES * STAGE_SIZE
                          + A_size + V_size + K_size
                          + 3 * BT * sizeof(float)
                          + 16 * sizeof(float)
                          + 5 * NUM_STAGES * 8
                          + 4;

  auto kernel = kkt_inv_uw_v1b_kernel_cutlass<NUM_STAGES>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

  const dim3 grid(H, 148 / H);
  kernel<<<grid, TB_SIZE, smem_size>>>(
    K_tmap, V_tmap, U_tmap, W_tmap,
    A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
    g_cu_ptr,
    cu_seqlens_ptr, chunk_indices_ptr, total_chunks_ptr);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(kkt_inv_uw_v1b, kkt_inv_uw_v1b);
