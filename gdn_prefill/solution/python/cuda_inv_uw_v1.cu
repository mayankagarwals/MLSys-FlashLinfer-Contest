#include <cuda_bf16.h>
#include <cudaTypedefs.h>
#include <cstdint>
#include "cuda_utils.h"

__host__ __device__ inline
int cdiv(int a, int b) { return (a + b - 1) / b; }

__device__ inline
uint32_t fp32x2_to_bf16x2(float a, float b) {
  uint32_t tmp;
  asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(tmp) : "f"(b), "f"(a));
  return tmp;
}
__device__ inline
void bf16x2_to_fp32x2(float *out, uint32_t data) {
  asm volatile("shl.b32 %0, %2, 16;\n"        // low 16-bit
               "and.b32 %1, %2, 0xFFFF0000;"  // high 16-bit
              : "=f"(out[0]), "=f"(out[1]) : "r"(data));
}

__device__ inline
void lds_f32x2(float *data, uint32_t addr) {
  asm volatile("ld.shared.v2.f32 {%0, %1}, [%2];" : "=f"(data[0]), "=f"(data[1]) : "r"(addr));
};

__device__ __forceinline__
void mma_bf16(float d[4], uint32_t a[4], uint32_t b[2], float c[4]) {
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
void inv_uw_v1_kernel_cutlass(
  const __grid_constant__ CUtensorMap A_tmap,  // [total_T, H, BT]
  const __grid_constant__ CUtensorMap K_tmap,  // [total_T, Hg, K_dim]
  const __grid_constant__ CUtensorMap V_tmap,  // [total_T, H, V_dim]
  const __grid_constant__ CUtensorMap U_tmap,  // [pad_T, H, V_dim]
  const __grid_constant__ CUtensorMap W_tmap,  // [pad_T, H, V_dim]
  const float       *beta_ptr,                 // [total_T, H]
  const float       *g_cu_ptr,                 // [total_T, H]
  const int64_t     *cu_seqlens_ptr,           // [N+1]
  const int32_t     *chunk_indices_ptr,        // [total_num_chunks, 2]
  const int32_t     *total_chunks_ptr          // [1]
) {
  const int tid = threadIdx.x;
  const int warp_id = warp_uniform(tid / WARP_SIZE);
  const int lane_id = tid % WARP_SIZE;

  const int head_id = blockIdx.x;
  const int bid = blockIdx.y;
  const int total_chunks = total_chunks_ptr[0];

  // set up smem
  extern __shared__ __align__(1024) char smem_ptr[];
  const uint32_t smem = __cvta_generic_to_shared(smem_ptr);

  const uint32_t Ai_smem   = smem + NUM_STAGES * STAGE_SIZE;
  const uint32_t U_smem    = Ai_smem + A_size;
  const uint32_t W_smem    = U_smem + V_size;
  const uint32_t beta_smem = W_smem + K_size;
  const uint32_t g_cu_smem = beta_smem + BT * (uint32_t)sizeof(float);

  const uint32_t tma_mbar = g_cu_smem + BT * (uint32_t)sizeof(float);
  const uint32_t inv_mbar = tma_mbar + NUM_STAGES * 8;
  const uint32_t mma_mbar = inv_mbar + NUM_STAGES * 8;
  const uint32_t epi_mbar = mma_mbar + NUM_STAGES * 8;

  const uint32_t taddr = epi_mbar + NUM_STAGES * 8;

  // set up tmem
  const uint32_t U_tmem  = 0;
  const uint32_t Ab_tmem = U_tmem + V_dim * NUM_STAGES;

  if (warp_id == 0) {
    // init mbar
    if (elect_sync()) {
      for (int i = 0; i < NUM_STAGES; i++) {
        mbarrier_init(tma_mbar + i * 8, 1);
        mbarrier_init(inv_mbar + i * 8, 128);
        mbarrier_init(mma_mbar + i * 8, 1);
        mbarrier_init(epi_mbar + i * 8, 128);
      }
      fence_mbarrier_init();
    }
  }
  else if (warp_id == 1) {
    // prefetch TMA descriptor
    if (elect_sync()) {
      prefetch_tensormap(&A_tmap);
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
        // const int eos = cu_seqlens_ptr[seq_id + 1];

        const uint32_t A_smem = smem + stage_id * STAGE_SIZE;
        const uint32_t V_smem = A_smem + A_size;
        const uint32_t K_smem = V_smem + V_size;

        const int off_t = bos + chunk_id * BT;
        const int mbar = tma_mbar + stage_id * 8;

        mbarrier_wait(mma_mbar + stage_id * 8, parity);

        // KV: [H, KV_dim/64, total_T, 64]
        tma_load_4d(A_smem, &A_tmap, 0, off_t, 0, head_id, mbar, EVICT_FIRST);
        tma_load_4d(V_smem, &V_tmap, 0, off_t, 0, head_id, mbar, EVICT_FIRST);
        tma_load_4d(K_smem, &K_tmap, 0, off_t, 0, k_head_id, mbar);
        mbarrier_arrive_expect_tx(mbar, STAGE_SIZE);

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
      int parity = 0;

      for (int global_chunk_id = bid; global_chunk_id < total_chunks; global_chunk_id += gridDim.y) {
        const uint32_t A_smem = smem + stage_id * STAGE_SIZE;
        const uint32_t V_smem = A_smem + A_size;
        const uint32_t K_smem = V_smem + V_size;

        // U uses lower 16 lanes, W uses higher 16 lanes
        const uint32_t this_U_tmem = U_tmem + V_dim * stage_id;
        const uint32_t this_W_tmem = this_U_tmem | (16U << 16U);
        const uint32_t this_Ab_tmem = Ab_tmem + (BT / 2) * stage_id;
        const uint32_t this_Abg_tmem = this_Ab_tmem | (16U << 16U);

        mbarrier_wait(epi_mbar + stage_id * 8, parity ^ 1);  // wait for acc tmem
        mbarrier_wait(inv_mbar + stage_id * 8, parity);  // wait for Ab and Abg
        tcgen05_fence_after_thread_sync();

        // U = (Ai * beta) @ V
        // W = (Ai * beta * g_cu) @ K
        constexpr uint32_t u_idesc = make_tcgen05_idesc(BT, V_dim) | (1U << 16U);  // transpose B
        constexpr uint32_t w_idesc = make_tcgen05_idesc(BT, K_dim) | (1U << 16U);  // transpose B

        // 128B swizzling
        constexpr uint64_t sdesc_base = (desc_encode(BT * 128) << 16ULL)  // LBO, ignored for K-major
                                      | (desc_encode(8 * 128) << 32ULL)   // SBO
                                      | (1ULL << 46ULL) | (2ULL << 61ULL);

        for (int i = 0; i < BT / 16; i++) {
          const uint64_t v_desc = sdesc_base | ((V_smem + i * 16 * 128) >> 4U);
          const uint64_t k_desc = sdesc_base | ((K_smem + i * 16 * 128) >> 4U);
          tcgen05_mma_tmem(this_U_tmem, this_Ab_tmem + i * 8, v_desc, u_idesc, i > 0);
          tcgen05_mma_tmem(this_W_tmem, this_Abg_tmem + i * 8, k_desc, w_idesc, i > 0);
        }
        tcgen05_commit(mma_mbar + stage_id * 8);

        stage_id = (stage_id + 1) % NUM_STAGES;
        if (stage_id == 0)
          parity ^= 1;
      }
    }
  }
  else if (warp_id >= 4) {
    // inv warps
    int tid_ = tid % 128;
    int warp_id_ = warp_id % 4;

    int stage_id = 0;
    int parity = 0;

    float *Ai_smem_ptr = reinterpret_cast<float *>(smem_ptr + (Ai_smem - smem));
    float *beta_smem_ptr = reinterpret_cast<float *>(smem_ptr + (beta_smem - smem));
    float *g_cu_smem_ptr = reinterpret_cast<float *>(smem_ptr + (g_cu_smem - smem));

    for (int global_chunk_id = bid; global_chunk_id < total_chunks; global_chunk_id += gridDim.y) {
      const int2 tmp = reinterpret_cast<const int2 *>(chunk_indices_ptr)[global_chunk_id];
      const int seq_id = tmp.x;
      const int chunk_id = tmp.y;
      const int bos = cu_seqlens_ptr[seq_id];
      const int eos = cu_seqlens_ptr[seq_id + 1];

      const uint32_t A_smem = smem + stage_id * STAGE_SIZE;
      const uint32_t V_smem = A_smem + A_size;
      const uint32_t K_smem = V_smem + V_size;

      // load beta and g_cu
      {
        const int local_t = tid_ % BT;
        const int t = bos + chunk_id * BT + local_t;
        if (tid_ < BT)
          beta_smem_ptr[local_t] = t < eos ? beta_ptr[t * H + head_id] : 0.0f;
        else
          g_cu_smem_ptr[local_t] = t < eos ? __expf(g_cu_ptr[t * H + head_id]) : 0.0f;
      }

      if (warp_id_ == 0)  // wait TMA
        mbarrier_wait(tma_mbar + stage_id * 8, parity);
      bar_sync<1>(128);

      // compute inverse
      //
      // Neumann series
      // init:
      //   An = A
      //   Ai = I-A
      // iterate:
      //   new_An = An @ An
      //   new_Ai = Ai @ (I + new_An)
      //
      // Newton-Schulz iteration
      // init
      //   Ai = I-A
      // iterate:
      //   MAi = (I+A) @ Ai
      //   new_Ai = Ai @ (2I - MAi)

      {
        // each warp compute inverse of 16x16 diagonal tiles
        // A smem layout: [BT/64, BT, 64] = [BT, 64]
        uint32_t Ai[4], An[4], mma_B[4];
        float acc[8], zeros[4] = {};

        // set diagonal of a [16,16] ldmatrix BF16 tile held by a warp to 1
        auto set_diagonal = [&](uint32_t *A) {
          if (lane_id % 9 == 0) {  // 0, 9, 18, 27
            A[0] |= 0x3F80U;  // top left [8,8] tile
            A[3] |= 0x3F80U;  // bottom right [8,8] tile
          }
          else if (lane_id % 9 == 4) {  // 4, 13, 22, 31
            A[0] |= 0x3F800000U;
            A[3] |= 0x3F800000U;
          }
        };

        // compute address for [16,16] ldmatrix BF16 tile
        // row16 and col16 is row and col for [16,16] tile within [64,64] tile.
        auto compute_addr = [&](uint32_t row16, uint32_t col16) {
          const uint32_t row = row16 * 16 + (lane_id % 16);
          const uint32_t col = (col16 * 2 + (lane_id / 16)) ^ (lane_id % 8);
          return A_smem + row * 128 + col * 16;
        };

        // init Ai
        const uint32_t diag_addr = compute_addr(warp_id_, warp_id_);
        ldmatrix<4>(Ai, diag_addr);  // A
        for (int i = 0; i < 4; i++)
          Ai[i] ^= 0x80008000U;  // flip sign bit i.e. -A
        set_diagonal(Ai);  // I-A

        for (int i = 0; i < 3; i++) {
          // new_An = An @ An
          ldmatrix<4>(An, diag_addr);
          ldmatrix_trans<4>(mma_B, diag_addr);
          mma_bf16(acc + 0, An, mma_B + 0, zeros);
          mma_bf16(acc + 4, An, mma_B + 2, zeros);

          // pack to BF16, then store back to smem
          for (int j = 0; j < 4; j++)
            mma_B[j] = fp32x2_to_bf16x2(acc[j * 2], acc[j * 2 + 1]);
          stmatrix<4>(diag_addr, mma_B);
          __syncwarp();  // do we need this?

          // new_Ai = Ai @ (I + new_An)
          // separate acc registers?
          ldmatrix_trans<4>(mma_B, diag_addr);
          set_diagonal(mma_B);  // I+An
          mma_bf16(acc + 0, Ai, mma_B + 0, zeros);
          mma_bf16(acc + 4, Ai, mma_B + 2, zeros);

          // pack to BF16
          for (int j = 0; j < 4; j++)
            Ai[j] = fp32x2_to_bf16x2(acc[j * 2], acc[j * 2 + 1]);
        }
        stmatrix<4>(diag_addr, Ai);
        bar_sync<1>(128);

        // compute inverse for off-diagonal tiles
        // [ I+A00                      ]
        // [   A10, I+A11               ]
        // [   A20,   A21, I+A22        ]
        // [   A30,   A31,   A32, I+A33 ]

        // off-diagonal by 1
        //   Ai10 = -Ai11 @ A10 @ Ai00
        //   Ai21 = -Ai22 @ A21 @ Ai11
        //   Ai32 = -Ai33 @ A32 @ Ai22
        if (warp_id_ > 0) {
          for (int i = 0; i < 4; i++)
            Ai[i] ^= 0x80008000U;  // flip sign bit i.e. -Ai

          // warp1 loads A10, warp2 loads A21, warp3 loads A32
          ldmatrix_trans<4>(mma_B, compute_addr(warp_id_, warp_id_ - 1));
          mma_bf16(acc + 0, Ai, mma_B + 0, zeros);
          mma_bf16(acc + 4, Ai, mma_B + 2, zeros);
          for (int i = 0; i < 4; i++)
            Ai[i] = fp32x2_to_bf16x2(acc[i * 2], acc[i * 2 + 1]);

          // warp1 loads Ai00, warp2 loads Ai11, warp3 loads Ai22
          ldmatrix_trans<4>(mma_B, compute_addr(warp_id_ - 1, warp_id_ - 1));
          mma_bf16(acc + 0, Ai, mma_B + 0, zeros);
          mma_bf16(acc + 4, Ai, mma_B + 2, zeros);
          for (int i = 0; i < 4; i++)
            Ai[i] = fp32x2_to_bf16x2(acc[i * 2], acc[i * 2 + 1]);

          // warp1 stores Ai10, warp2 stores Ai21, warp3 stores Ai32
          stmatrix<4>(compute_addr(warp_id_, warp_id_ - 1), Ai);
        }
        bar_sync<1>(128);

        // off-diagonal by 2
        //   Ai20 = -Ai22 @ (A20 @ Ai00 + A21 @ Ai10)
        //   Ai31 = -Ai33 @ (A31 @ Ai11 + A32 @ Ai21)
        if (warp_id_ < 2) {
          ldmatrix<4>(Ai, compute_addr(warp_id_ + 2, warp_id_));
          ldmatrix_trans<4>(mma_B, diag_addr);
          mma_bf16(acc + 0, Ai, mma_B + 0, zeros);
          mma_bf16(acc + 4, Ai, mma_B + 2, zeros);

          ldmatrix<4>(Ai, compute_addr(warp_id_ + 2, warp_id_ + 1));
          ldmatrix_trans<4>(mma_B, compute_addr(warp_id_ + 1, warp_id_));
          mma_bf16(acc + 0, Ai, mma_B + 0, acc + 0);
          mma_bf16(acc + 4, Ai, mma_B + 2, acc + 4);

          // NOTE: we can swap A and B operand to avoid transposing via smem
          for (int i = 0; i < 4; i++)
            mma_B[i] = fp32x2_to_bf16x2(acc[i * 2], acc[i * 2 + 1]);
          stmatrix<4>(compute_addr(warp_id_ + 2, warp_id_), mma_B);
          __syncwarp();

          ldmatrix<4>(Ai, compute_addr(warp_id_ + 2, warp_id_ + 2));
          for (int i = 0; i < 4; i++)
            Ai[i] ^= 0x80008000U;  // flip sign bit
          ldmatrix_trans<4>(mma_B, compute_addr(warp_id_ + 2, warp_id_));
          mma_bf16(acc + 0, Ai, mma_B + 0, zeros);
          mma_bf16(acc + 4, Ai, mma_B + 2, zeros);
          for (int i = 0; i < 4; i++)
            mma_B[i] = fp32x2_to_bf16x2(acc[i * 2], acc[i * 2 + 1]);
          stmatrix<4>(compute_addr(warp_id_ + 2, warp_id_), mma_B);
        }
        bar_sync<1>(128);

        // off-diagonal by 3
        //   Ai30 = -Ai33 @ (A30 @ Ai00 + A31 @ Ai10 + A32 @ Ai20)
        if (warp_id_ == 0) {
          ldmatrix<4>(Ai, compute_addr(3, 0));
          ldmatrix_trans<4>(mma_B, diag_addr);
          mma_bf16(acc + 0, Ai, mma_B + 0, zeros);
          mma_bf16(acc + 4, Ai, mma_B + 2, zeros);

          ldmatrix<4>(Ai, compute_addr(3, 1));
          ldmatrix_trans<4>(mma_B, compute_addr(1, 0));
          mma_bf16(acc + 0, Ai, mma_B + 0, acc + 0);
          mma_bf16(acc + 4, Ai, mma_B + 2, acc + 4);

          ldmatrix<4>(Ai, compute_addr(3, 2));
          ldmatrix_trans<4>(mma_B, compute_addr(2, 0));
          mma_bf16(acc + 0, Ai, mma_B + 0, acc + 0);
          mma_bf16(acc + 4, Ai, mma_B + 2, acc + 4);

          // NOTE: we can swap A and B operand to avoid transposing via smem
          for (int i = 0; i < 4; i++)
            mma_B[i] = fp32x2_to_bf16x2(acc[i * 2], acc[i * 2 + 1]);
          stmatrix<4>(compute_addr(3, 0), mma_B);
          __syncwarp();

          ldmatrix<4>(Ai, compute_addr(3, 3));
          for (int i = 0; i < 4; i++)
            Ai[i] ^= 0x80008000U;  // flip sign bit
          ldmatrix_trans<4>(mma_B, compute_addr(3, 0));
          mma_bf16(acc + 0, Ai, mma_B + 0, zeros);
          mma_bf16(acc + 4, Ai, mma_B + 2, zeros);
          for (int i = 0; i < 4; i++)
            mma_B[i] = fp32x2_to_bf16x2(acc[i * 2], acc[i * 2 + 1]);
          stmatrix<4>(compute_addr(3, 0), mma_B);
        }
        // bar_sync<1>(128);
      }

      // check that MMA has finished using Ab/Abg buffer
      if (warp_id_ == 3)
        mbarrier_wait(mma_mbar + stage_id * 8, parity ^ 1);
      bar_sync<1>(128);

      // compute Ab and Abg, then store to tmem
      // Ai smem layout: [64, 64]
      // each warp:      [16, 64]
      // each i iter:    [16, 16]
      for (int i = 0; i < 64 / 16; i++) {
        uint32_t Ai[4], Ab[4], Abg[4];
        float beta[4], g_cu[4];

        const int row = warp_id_ * 16 + (lane_id % 16);
        const int col = (i * 2 + (lane_id / 16)) ^ (lane_id % 8);
        ldmatrix<4>(Ai, A_smem + row * 128 + col * 16);

        const int scale_col = i * 16 + (lane_id % 4) * 2;
        lds_f32x2(beta + 0, beta_smem + scale_col * 4);
        lds_f32x2(beta + 2, beta_smem + scale_col * 4 + 8);
        lds_f32x2(g_cu + 0, g_cu_smem + scale_col * 4);
        lds_f32x2(g_cu + 2, g_cu_smem + scale_col * 4 + 8);

        float tmp[8];
        for (int j = 0; j < 4; j++) {
          bf16x2_to_fp32x2(tmp + j * 2, Ai[j]);

          tmp[j * 2 + 0] *= beta[j / 2 * 2 + 0];
          tmp[j * 2 + 1] *= beta[j / 2 * 2 + 1];
          Ab[j] = fp32x2_to_bf16x2(tmp[j * 2], tmp[j * 2 + 1]);

          tmp[j * 2 + 0] *= g_cu[j / 2 * 2 + 0];
          tmp[j * 2 + 1] *= g_cu[j / 2 * 2 + 1];
          Abg[j] = fp32x2_to_bf16x2(tmp[j * 2], tmp[j * 2 + 1]);
        }

        // U uses lower 16 lanes, W uses higher 16 lanes
        const uint32_t this_Ab_tmem = Ab_tmem + (BT / 2) * stage_id + i * 8;
        tcgen05_st<SHAPE::_16x128b, 2>(warp_id_ * 32, this_Ab_tmem, Ab);
        tcgen05_st<SHAPE::_16x128b, 2>(warp_id_ * 32 + 16, this_Ab_tmem, Abg);
      }
      tcgen05_wait_st();
      tcgen05_fence_before_thread_sync();
      mbarrier_arrive(inv_mbar + stage_id * 8);

      stage_id = (stage_id + 1) % NUM_STAGES;
      if (stage_id == 0)
        parity ^= 1;
    }
  }
  else {
    // epilogue warps
    int stage_id = 0;
    int parity = 0;

    for (int global_chunk_id = bid; global_chunk_id < total_chunks; global_chunk_id += gridDim.y) {
      const int2 tmp = reinterpret_cast<const int2 *>(chunk_indices_ptr)[global_chunk_id];
      const int seq_id = tmp.x;
      const int chunk_id = tmp.y;
      const int bos = cu_seqlens_ptr[seq_id];
      const int eos = cu_seqlens_ptr[seq_id + 1];

      const uint32_t this_U_tmem = U_tmem + V_dim * stage_id;

      // wait for U and W
      if (warp_id == 0)
        mbarrier_wait(mma_mbar + stage_id * 8, parity);
      else if (warp_id == 1) {
        if (elect_sync())
          cp_async_bulk_wait_group_read<0>();
      }
      bar_sync<2>(128);
      tcgen05_fence_after_thread_sync();

      // U smem layout: [V_dim/64, BT, 64]
      // per warp:      [V_dim/64, 16, 64]
      // per i iter:    [       1, 16, 64]
      // per j iter:    [       1, 16, 16]
      float u_tmp[V_dim / 2];
      tcgen05_ld<SHAPE::_16x256b, V_dim / 8>(u_tmp, warp_id * 32, this_U_tmem);
      for (int i = 0; i < V_dim / 64; i++)
        for (int j = 0; j < 64 / 16; j++) {
          // pack to bf16
          uint32_t tmp[4];
          for (int k = 0; k < 4; k++)
            tmp[k] = fp32x2_to_bf16x2(u_tmp[i * 32 + j * 8 + k * 2],
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
          // pack to bf16
          uint32_t tmp[4];
          for (int k = 0; k < 4; k++)
            tmp[k] = fp32x2_to_bf16x2(w_tmp[i * 32 + j * 8 + k * 2],
                                      w_tmp[i * 32 + j * 8 + k * 2 + 1]);

          const int row = warp_id * 16 + (lane_id % 16);
          const int col = (j * 2 + (lane_id / 16)) ^ (lane_id % 8);
          stmatrix<4>(W_smem + i * BT * 128 + row * 128 + col * 16, tmp);
        }

      // release to async proxy
      bar_sync<2>(128);
      asm volatile("fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster;");
      if (warp_id == 1 && elect_sync()) {
        // natural shape:  [pad_T, H, V_dim]
        // permuted shape: [H, V_dim/64, pad_T, 64]
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
CUtensorMap encode_tma(void *ptr, uint64_t T, uint64_t H, uint64_t dim) {
  CUtensorMap tmap;

  // natural shape:  [T, H, dim]
  // permuted shape: [H, dim/64, T, 64]
  constexpr uint32_t rank = 4;
  uint64_t globalDim[rank] = {64, T, dim / 64, H};
  uint64_t globalStrides[rank - 1] = {H * dim * sizeof(nv_bfloat16),
                                                                128,
                                          dim * sizeof(nv_bfloat16)};  // in bytes
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

void inv_uw_v1(
  TensorView A,
  TensorView K,
  TensorView V,
  TensorView U,
  TensorView W,
  TensorView beta,
  TensorView g_cu,
  TensorView cu_seqlens,
  TensorView chunk_indices,
  TensorView total_chunks
) {
  const int T = K.size(0);
  const int N = cu_seqlens.size(0) - 1;

  auto A_tmap = encode_tma(A.data_ptr(), T, H, BT);
  auto K_tmap = encode_tma(K.data_ptr(), T, Hg, K_dim);
  auto V_tmap = encode_tma(V.data_ptr(), T, H, V_dim);
  auto U_tmap = encode_tma(U.data_ptr(), U.size(0), H, V_dim);
  auto W_tmap = encode_tma(W.data_ptr(), W.size(0), H, K_dim);

  auto *u_ptr    = reinterpret_cast<nv_bfloat16 *>(U.data_ptr());
  auto *w_ptr    = reinterpret_cast<nv_bfloat16 *>(W.data_ptr());
  auto *beta_ptr = reinterpret_cast<const float *>(beta.data_ptr());
  auto *g_cu_ptr = reinterpret_cast<const float *>(g_cu.data_ptr());
  auto *cu_seqlens_ptr    = reinterpret_cast<int64_t *>(cu_seqlens.data_ptr());
  auto *chunk_indices_ptr = reinterpret_cast<int32_t *>(chunk_indices.data_ptr());
  auto *total_chunks_ptr  = reinterpret_cast<int32_t *>(total_chunks.data_ptr());

  constexpr int NUM_STAGES = 2;
  constexpr int smem_size = NUM_STAGES * STAGE_SIZE
                          + A_size + V_size + K_size  // Ai, U, W
                          + 2 * BT * sizeof(float)    // beta and g_cu
                          + 4 * NUM_STAGES * 8        // tma, inv, mma, epi mbar
                          + 4;                        // taddr

  auto kernel = inv_uw_v1_kernel_cutlass<NUM_STAGES>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

  const dim3 grid(H, 148 / H);
  kernel<<<grid, TB_SIZE, smem_size>>>(
    A_tmap, K_tmap, V_tmap, U_tmap, W_tmap, beta_ptr, g_cu_ptr,
    cu_seqlens_ptr, chunk_indices_ptr, total_chunks_ptr);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(inv_uw_v1, inv_uw_v1);
