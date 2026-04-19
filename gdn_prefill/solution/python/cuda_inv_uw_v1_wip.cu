// for 64x64 inv
// inv(I + A) = (I - A)(I + A^2)(I + A^4)(I + A^8)(I + A^16)(I + A^32)
//
// initialize:
//   An = A
//   Ai = I-A
// iter 1:
//   new_An = A^2 = An @ An
//   new_Ai = (I-A)(I+A^2) = Ai + Ai @ new_An
// iter 2:
//   new_An = A^4 = An @ An
//   new_Ai = (I-A)(I+A^2)(I+A^4) = Ai + Ai @ new_An
//
// ### algorithm ###
// load A (TMA)
//
// # initialize
// MMA  warp: wait TMA
// CUDA warp: wait TMA, compute Ai = I-A, store to tmem
//            (duplicate as Ai_in and Ai_acc)
//
// # repeats
// MMA  warp: issue new_An = An @ An (NOTE: this MMA doesn't need to wait)
// CUDA warp: wait MMA, store An to smem, copy Ai_acc to Ai_in
// MMA  warp: wait CUDA, issue new_Ai = Ai_acc + Ai_in @ new_An

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

constexpr int H = 8;
constexpr int Hg = 4;
constexpr int BT = 64;
constexpr int K_dim = 128;
constexpr int V_dim = 128;

constexpr int A_size = BT * BT * sizeof(float);
constexpr int K_size = BT * K_dim * sizeof(nv_bfloat16);
constexpr int V_size = BT * V_dim * sizeof(nv_bfloat16);
constexpr int STAGE_SIZE = A_size * 2 + K_size + V_size;  // we use 2 A buffer

constexpr int NUM_WARPS = 4 + 2;
constexpr int WARP_SIZE = 32;
constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;

template <int NUM_STAGES>
__global__
__block_size__((TB_SIZE, 1, 1))
void inv_uw_v1_kernel_cutlass(
  const __grid_constant__ CUtensorMap A_tmap,  // [total_T, H, BT]
  const __grid_constant__ CUtensorMap K_tmap,  // [total_T, Hg, K_dim]
  const __grid_constant__ CUtensorMap V_tmap,  // [total_T, H, V_dim]
        nv_bfloat16 *u_ptr,                    // [total_T, H, V_dim]
        nv_bfloat16 *w_ptr,                    // [total_T, H, K_dim]
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
  const int global_chunk_id = blockIdx.y;

  // early return
  const int total_chunks = total_chunks_ptr[0];
  if (global_chunk_id >= total_chunks) {
    return;
  }

  const int2 tmp = reinterpret_cast<const int2 *>(chunk_indices_ptr)[global_chunk_id];
  const int seq_id = tmp.x;
  const int chunk_id = tmp.y;
  const int bos = cu_seqlens_ptr[seq_id];
  const int eos = cu_seqlens_ptr[seq_id + 1];

  // set up smem
  extern __shared__ __align__(1024) char smem_ptr[];
  const uint32_t smem = __cvta_generic_to_shared(smem_ptr);

  const uint32_t beta_smem = smem + NUM_STAGES * STAGE_SIZE;
  const uint32_t g_cu_smem = beta_smem + BT * (uint32_t)sizeof(float);

  const uint32_t tma_full_mbar  = g_cu_smem + BT * (uint32_t)sizeof(float);
  const uint32_t tma_empty_mbar = tma_full_mbar + NUM_STAGES * 8;
  const uint32_t cuda_mbar      = tma_empty_mbar + NUM_STAGES * 8;
  const uint32_t mma_mbar       = cuda_mbar + 8;

  const uint32_t taddr = mma_mbar + 8;

  // set up tmem
  const uint32_t Ai_in_tmem = 0;
  const uint32_t Ai_acc_tmem = Ai_in_tmem + 64;
  const uint32_t An_acc_tmem = Ai_acc_tmem + 64;

  const uint32_t U_tmem   = 0;
  const uint32_t W_tmem   = U_tmem + V_dim;
  const uint32_t Ab_tmem  = W_tmem + K_dim;
  const uint32_t Abg_tmem = Ab_tmem + BT / 2;

  if (warp_id == 0) {
    // init mbar
    if (elect_sync()) {
      for (int i = 0; i < NUM_STAGES; i++) {
        mbarrier_init(tma_full_mbar + i * 8, 1);
        mbarrier_init(tma_empty_mbar + i * 8, 1);
      }
      mbarrier_init(cuda_mbar, 128);
      mbarrier_init(mma_mbar, 1);
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
      const uint32_t A_smem = smem;
      const uint32_t An_smem = A_smem + A_size;
      const uint32_t V_smem = An_smem + A_size;
      const uint32_t K_smem = V_smem + V_size;

      const int off_t = bos + chunk_id * BT;
      const int k_head_id = head_id / (H / Hg);

      // A:  [H, BT/32,     total_T, 32]
      // KV: [H, KV_dim/64, total_T, 64]
      tma_load_4d(A_smem, &A_tmap, 0, off_t, 0, head_id, tma_full_mbar);
      tma_load_4d(An_smem, &A_tmap, 0, off_t, 0, head_id, tma_full_mbar, EVICT_FIRST);  // init An with A
      tma_load_4d(V_smem, &V_tmap, 0, off_t, 0, head_id, tma_full_mbar, EVICT_FIRST);
      tma_load_4d(K_smem, &K_tmap, 0, off_t, 0, k_head_id, tma_full_mbar);
      mbarrier_arrive_expect_tx(tma_full_mbar, STAGE_SIZE);
    }
  }
  else if (warp_id == NUM_WARPS - 2) {
    // MMA warp
    tcgen05_alloc(taddr, 512);

    if (elect_sync()) {
      int parity = 0;

      const uint32_t A_smem = smem;
      const uint32_t An_smem = A_smem + A_size;
      const uint32_t V_smem = An_smem + A_size;
      const uint32_t K_smem = V_smem + V_size;

      mbarrier_wait(tma_full_mbar, 0);

      constexpr uint32_t tf32_idesc = (1U << 4U)   // dtype = FP32
                                    | (2U << 7U)   // atype = TF32
                                    | (2U << 10U)  // btype = TF32
                                    | (1U << 16U)  // transpose B
                                    | ((uint32_t)(BT >> 3U) << 17U)  // MMA_N
                                    | ((uint32_t)(BT >> 4U) << 24U); // MMA_M

      // 128B swizzling
      constexpr uint64_t sdesc_base = (desc_encode(BT * 128) << 16ULL)  // LBO, ignored for K-major
                                    | (desc_encode(8 * 128) << 32ULL)   // SBO
                                    | (1ULL << 46ULL) | (2ULL << 61ULL);

      // compute inverse via
      // (I - A)(I + A^2)(I + A^4)(I + A^8)(I + A^16)(I + A^32)
      // log2(64) - 1 = 5
      for (int aa = 0; aa < 5; aa++) {
        // 1st MMA: new An = An @ An
        //
        // A smem layout: [BT/32, BT, 32]
        // for A operand (K-major):
        //   i selects [1, BT, 32]
        //   j selects [1, BT,  8] (there's swizzling)
        // for B operatnd (MN-major):
        //   (i * 4 + j) selects [BT/32, 8, 32]
        for (int i = 0; i < BT / 32; i++)
          for (int j = 0; j < 32 / 8; j++) {
            const uint64_t a_desc = sdesc_base | ((An_smem + i * BT * 128 + j * 32) >> 4U);
            const uint64_t b_desc = sdesc_base | ((An_smem + (i * 4 + j) * 8 * 128) >> 4U);
            const int enable_input_d = i > 0 || j > 0;
            tcgen05_mma_tf32(An_acc_tmem, a_desc, b_desc, tf32_idesc, enable_input_d);
          }
        tcgen05_commit(mma_mbar);

        // 2nd MMA: new_Ai = Ai_acc + Ai_in @ new_An
        //
        // wait CUDA warps to prepare Ai_in and new_An
        // we need separate Ai_in and Ai_acc to avoid Ai input being
        // modified while issuing tcgen05.mma
        mbarrier_wait(cuda_mbar, parity);
        tcgen05_fence_after_thread_sync();
        parity ^= 1;

        for (int i = 0; i < BT / 8; i++) {
          const uint32_t a_tmem = Ai_in_tmem + i * 8;
          const uint64_t b_desc = sdesc_base | ((An_smem + i * 8 * 128) >> 4U);
          tcgen05_mma_tf32_tmem(Ai_acc_tmem, a_tmem, b_desc, tf32_idesc, 1);  // always enable input d
        }
      }

      // finish computing inverse
      tcgen05_commit(mma_mbar);

      // wait for Ab and Abg
      mbarrier_wait(cuda_mbar, parity);
      tcgen05_fence_after_thread_sync();
      parity ^= 1;

      // U = (Ai * beta) @ V
      {
        const uint32_t bf16_idesc = make_tcgen05_idesc(BT, V_dim) | (1U << 16U);  // transpose B
        for (int i = 0; i < BT / 16; i++) {
          const uint32_t a_tmem = Ab_tmem + i * 8;  // 8 columns = 32B
          const uint64_t b_desc = sdesc_base | ((V_smem + i * 16 * 128) >> 4U);
          tcgen05_mma_tmem(U_tmem, a_tmem, b_desc, bf16_idesc, i > 0);
        }
      }

      // W = (Ai * beta * g_cu) @ K
      {
        const uint32_t bf16_idesc = make_tcgen05_idesc(BT, K_dim) | (1U << 16U);  // transpose B
        for (int i = 0; i < BT / 16; i++) {
          const uint32_t a_tmem = Abg_tmem + i * 8;  // 8 columns = 32B
          const uint64_t b_desc = sdesc_base | ((K_smem + i * 16 * 128) >> 4U);
          tcgen05_mma_tmem(W_tmem, a_tmem, b_desc, bf16_idesc, i > 0);
        }
      }

      // finish U and W MMA
      tcgen05_commit(mma_mbar);
    }
  }
  else {
    // CUDA warps
    int parity = 0;

    if (warp_id == 0)
      mbarrier_wait(tma_full_mbar, 0);
    bar_sync<1>(128);

    // A0 is used to compute A-I
    // A1 is used to store An
    const uint32_t A_smem = smem;
    const uint32_t An_smem = A_smem + A_size;
    const uint32_t V_smem = An_smem + A_size;
    const uint32_t K_smem = V_smem + V_size;

    // compute A-I in smem
    // A smem layout: [BT/32, BT, 32]
    // since A is strictly lower triangular, we only need to fill
    // the diagonal with -1.
    if (tid < BT) {
      float *ptr = reinterpret_cast<float *>(smem_ptr + (A_smem - smem));
      ptr[(tid / 32) * BT * 32 + tid * 32 + (tid % 32)] = -1.0f;
    }
    bar_sync<1>(128);

    // store A-I to tmem as Ai
    // A smem layout: [BT/32, BT, 32]
    // each warp:     [BT/32, 16, 32]
    // each i iter:   [    1, 16, 32]
    // each j iter:   [    1, 16,  8] (ldmatrix)
    for (int i = 0; i < BT / 32; i++) {
      for (int j = 0; j < 32 / 8; j++) {
        float tmp[4];
        const uint32_t row = warp_id * 16 + (lane_id % 16);
        const uint32_t col = (j * 2 + (lane_id / 16)) ^ (lane_id % 8);
        const uint32_t addr = A_smem + i * BT * 128 + row * 128 + col * 16;
        ldmatrix<4>(tmp, addr);

        // store to both Ai_in and Ai_acc
        tcgen05_st<SHAPE::_16x128b, 2>(warp_id * 32, Ai_in_tmem + i * 32 + j * 8, tmp);
        tcgen05_st<SHAPE::_16x128b, 2>(warp_id * 32, Ai_acc_tmem + i * 32 + j * 8, tmp);
      }
    }

    // compute inverse via
    // (I - A)(I + A^2)(I + A^4)(I + A^8)(I + A^16)(I + A^32)
    // log2(64) - 1 = 5
    for (int aa = 0; aa < 5; aa++) {
      if (warp_id == 0)
        mbarrier_wait(mma_mbar, parity);
      bar_sync<1>(128);
      tcgen05_fence_after_thread_sync();
      parity ^= 1;

      // store new An from tmem->smem
      // An smem layout: [BT/32, BT, 32]
      // each warp:      [BT/32, 16, 32]
      // each i iter:    [    1, 16, 32]
      // each j iter:    [    1, 16,  8]
      for (int i = 0; i < BT / 32; i++)
        for (int j = 0; j < 32 / 8; j++) {
          float tmp[4];
          tcgen05_ld<SHAPE::_16x128b, 2>(tmp, warp_id * 32, An_acc_tmem + i * 32 + j * 8);

          const uint32_t row = warp_id * 16 + (lane_id % 16);
          const uint32_t col = (j * 2 + (lane_id / 16)) ^ (lane_id % 8);
          const uint32_t addr = An_smem + i * BT * 128 + row * 128 + col * 16;
          stmatrix<4>(addr, tmp);
        }
      // release to async proxy for tcgen05.mma
      asm volatile("fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster;");

      // copy Ai_acc to Ai_in (1st iteration doesn't need to do this)
      if (aa > 0) {
        float tmp[BT / 2];
        tcgen05_ld<SHAPE::_16x256b, BT / 8>(tmp, warp_id * 32, Ai_acc_tmem);
        tcgen05_wait_ld();
        tcgen05_st<SHAPE::_16x256b, BT / 8>(warp_id * 32, Ai_in_tmem, tmp);
        tcgen05_wait_st();
      }

      tcgen05_fence_before_thread_sync();
      mbarrier_arrive(cuda_mbar);
    }

    // load beta and g_cu to smem
    // NOTE: because we initialize Ai with (A-I) instead of (I-A),
    // the final result in -Ai.
    // to compensate, we negate beta here.
    float *beta_smem_ptr = reinterpret_cast<float *>(smem_ptr + (beta_smem - smem));
    float *g_cu_smem_ptr = reinterpret_cast<float *>(smem_ptr + (g_cu_smem - smem));
    if (tid < BT) {
      const int t = bos + chunk_id * BT + tid;
      beta_smem_ptr[tid] = t < eos ? -beta_ptr[t * H + head_id] : 0.0f;
    }
    else {
      const int t = bos + chunk_id * BT + (tid % BT);
      g_cu_smem_ptr[tid % BT] = t < eos ? __expf(g_cu_ptr[t * H + head_id]) : 0.0f;
    }
    // visibility is ensured by bar.sync later

    // wait for the final Ai
    if (warp_id == 0)
      mbarrier_wait(mma_mbar, parity);
    bar_sync<1>(128);
    tcgen05_fence_after_thread_sync();
    parity ^= 1;

    // compute Ab and Abg, then store to tmem
    // Ab/Abg smem layout: [BT, 64]
    // each warp:          [16, 64]
    // each i iter:        [16, 16]
    for (int i = 0; i < 64 / 16; i++) {
      float tmp[8];
      uint32_t tmp_bf16[4];
      tcgen05_ld<SHAPE::_16x256b, 2>(tmp, warp_id * 32, Ai_acc_tmem + i * 16);

      // compute Ab = Ai * beta
      const int local_t = i * 16 + (lane_id % 4) * 2;
      float2 beta[2];
      beta[0] = reinterpret_cast<float2 *>(beta_smem_ptr + (local_t + 0))[0];
      beta[1] = reinterpret_cast<float2 *>(beta_smem_ptr + (local_t + 8))[0];

      // each j iter: [16, 8] tile
      for (int j = 0; j < 2; j++) {
        tmp[j * 4 + 0] *= beta[j].x;
        tmp[j * 4 + 1] *= beta[j].y;
        tmp[j * 4 + 2] *= beta[j].x;
        tmp[j * 4 + 3] *= beta[j].y;
      }

      // pack Ab to BF16
      for (int j = 0; j < 4; j++)
        tmp_bf16[j] = fp32x2_to_bf16x2(tmp[j * 2], tmp[j * 2 + 1]);
      tcgen05_st<SHAPE::_16x128b, 2>(warp_id * 32, Ab_tmem + i * 8, tmp_bf16);

      // compute Abg = Ai * beta * g_cu
      float2 g_cu[2];
      g_cu[0] = reinterpret_cast<float2 *>(g_cu_smem_ptr + (local_t + 0))[0];
      g_cu[1] = reinterpret_cast<float2 *>(g_cu_smem_ptr + (local_t + 8))[0];

      // each j iter: [16, 8] tile
      for (int j = 0; j < 2; j++) {
        tmp[j * 4 + 0] *= g_cu[j].x;
        tmp[j * 4 + 1] *= g_cu[j].y;
        tmp[j * 4 + 2] *= g_cu[j].x;
        tmp[j * 4 + 3] *= g_cu[j].y;
      }

      // pack Abg to BF16
      // (Abg uses higher 16 lanes)
      for (int j = 0; j < 4; j++)
        tmp_bf16[j] = fp32x2_to_bf16x2(tmp[j * 2], tmp[j * 2 + 1]);
      tcgen05_st<SHAPE::_16x128b, 2>(warp_id * 32, Abg_tmem + i * 8, tmp_bf16);
    }
    tcgen05_wait_st();
    tcgen05_fence_before_thread_sync();
    mbarrier_arrive(cuda_mbar);

    // wait for U and W
    if (warp_id == 0)
      mbarrier_wait(mma_mbar, parity);
    bar_sync<1>(128);
    tcgen05_fence_after_thread_sync();
    parity ^= 1;

    // store U to gmem
    {
      float tmp[V_dim / 2];
      tcgen05_ld<SHAPE::_16x256b, V_dim / 8>(tmp, warp_id * 32, U_tmem);

      const int t = bos + chunk_id * BT + warp_id * 16 + (lane_id / 4);
      if (t < eos) {
        const int offset = ((t + 0) * H + head_id) * V_dim + (lane_id % 4) * 2;
        for (int i = 0; i < V_dim / 8; i++)
          reinterpret_cast<nv_bfloat162 *>(u_ptr + (offset + i * 8))[0] = __float22bfloat162_rn({tmp[i * 4 + 0], tmp[i * 4 + 1]});
      }
      if (t + 8 < eos) {
        const int offset = ((t + 8) * H + head_id) * V_dim + (lane_id % 4) * 2;
        for (int i = 0; i < V_dim / 8; i++)
          reinterpret_cast<nv_bfloat162 *>(u_ptr + (offset + i * 8))[0] = __float22bfloat162_rn({tmp[i * 4 + 2], tmp[i * 4 + 3]});
      }
    }

    // store W to gmem
    {
      float tmp[K_dim / 2];
      tcgen05_ld<SHAPE::_16x256b, K_dim / 8>(tmp, warp_id * 32, W_tmem);

      const int t = bos + chunk_id * BT + warp_id * 16 + (lane_id / 4);
      if (t < eos) {
        const int offset = ((t + 0) * H + head_id) * K_dim + (lane_id % 4) * 2;
        for (int i = 0; i < K_dim / 8; i++)
          reinterpret_cast<nv_bfloat162 *>(w_ptr + (offset + i * 8))[0] = __float22bfloat162_rn({tmp[i * 4 + 0], tmp[i * 4 + 1]});
      }
      if (t + 8 < eos) {
        const int offset = ((t + 8) * H + head_id) * K_dim + (lane_id % 4) * 2;
        for (int i = 0; i < K_dim / 8; i++)
          reinterpret_cast<nv_bfloat162 *>(w_ptr + (offset + i * 8))[0] = __float22bfloat162_rn({tmp[i * 4 + 2], tmp[i * 4 + 3]});
      }
    }
 
    bar_sync<1>(128);
    if (warp_id == 0)
      tcgen05_dealloc(0, 512);
  }
}

static
CUtensorMap encode_tma(void *ptr, uint64_t T, uint64_t H, uint64_t dim, CUtensorMapDataType dtype) {
  CUtensorMap tmap;

  int elem_width = 0;
  if (dtype == CU_TENSOR_MAP_DATA_TYPE_FLOAT32) elem_width = 4;
  else if (dtype == CU_TENSOR_MAP_DATA_TYPE_BFLOAT16) elem_width = 2;

  const int num_elems = 128 / elem_width;

  // natural shape:  [T, H, dim]
  // permuted shape: [H, dim/64, T, 64] (BF16)
  //                 [H, dim/32, T, 32] (FP32)
  constexpr uint32_t rank = 4;
  uint64_t globalDim[rank] = {num_elems, T, dim / num_elems, H};
  uint64_t globalStrides[rank - 1] = {H * dim * elem_width,
                                                       128,
                                          dim * elem_width};  // in bytes
  uint32_t boxDim[rank] = {num_elems, BT, dim / num_elems, 1};
  uint32_t elementStrides[rank] = {1, 1, 1, 1};

  cuTensorMapEncodeTiled(
    &tmap, dtype, rank, ptr,
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

  auto A_tmap = encode_tma(A.data_ptr(), T, H, BT, CU_TENSOR_MAP_DATA_TYPE_FLOAT32);
  auto K_tmap = encode_tma(K.data_ptr(), T, Hg, K_dim, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16);
  auto V_tmap = encode_tma(V.data_ptr(), T, H, V_dim, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16);

  auto *u_ptr    = reinterpret_cast<nv_bfloat16 *>(U.data_ptr());
  auto *w_ptr    = reinterpret_cast<nv_bfloat16 *>(W.data_ptr());
  auto *beta_ptr = reinterpret_cast<const float *>(beta.data_ptr());
  auto *g_cu_ptr = reinterpret_cast<const float *>(g_cu.data_ptr());
  auto *cu_seqlens_ptr    = reinterpret_cast<int64_t *>(cu_seqlens.data_ptr());
  auto *chunk_indices_ptr = reinterpret_cast<int32_t *>(chunk_indices.data_ptr());
  auto *total_chunks_ptr  = reinterpret_cast<int32_t *>(total_chunks.data_ptr());

  constexpr int NUM_STAGES = 1;
  constexpr int smem_size = NUM_STAGES * STAGE_SIZE
                          + 2 * BT * sizeof(float)  // beta and g_cu
                          + 2 * NUM_STAGES * 8      // tma full/empty mbar
                          + 2 * 8                   // cuda and mma mbar
                          + 4;                      // taddr

  auto kernel = inv_uw_v1_kernel_cutlass<NUM_STAGES>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

  const int max_chunks = (N - 1) + cdiv(T - (N - 1), BT);
  const dim3 grid(H, max_chunks);
  kernel<<<grid, TB_SIZE, smem_size>>>(
    A_tmap, K_tmap, V_tmap, u_ptr, w_ptr, beta_ptr, g_cu_ptr,
    cu_seqlens_ptr, chunk_indices_ptr, total_chunks_ptr);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(inv_uw_v1, inv_uw_v1);
