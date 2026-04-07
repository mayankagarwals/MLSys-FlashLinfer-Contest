// for global_chunk_id in range(pid, num_chunks, num_programs):
//   seq_id, chunk_id = chunk_indices[global_chunk_id]
//
//   # TMA warp
//   load k
//
//   # MMA warp
//   kkt = k @ k.T  - [BT, BT] = [64, 64]
//
//   # CUDA warp
//   load b, a, A_log, dt_bias
//   compute beta, g
//   compute g_cu
//   store beta, g_cu to gmem
//
//   compute Gamma = exp(g_cu - g_cu.T)
//
//   wait kkt MMA

#include "cuda_utils.h"

constexpr int H = 8;
constexpr int Hg = 4;
constexpr int BT = 64;
constexpr int K_dim = 128;
constexpr int V_dim = 128;

constexpr uint32_t K_size = BT * K_dim * sizeof(nv_bfloat16);

constexpr int NUM_WARPS = 4 + 2;
constexpr int WARP_SIZE = 32;
constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;

template <int NUM_STAGES>
__global__
__block_size__((TB_SIZE, 1, 1))
void kkt_v1_kernel_cutlass(
  const __grid_constant__ CUtensorMap K_tmap,  // [total_T, Hg, K_dim]
  const float       *A_log_ptr,                // [H]
  const nv_bfloat16 *a_ptr,                    // [total_T, H]
  const float       *dt_bias_ptr,              // [H]
  const nv_bfloat16 *b_ptr,                    // [total_T, H]
        float       *g_cu_ptr,                 // [total_T, H]
        float       *beta_ptr,                 // [total_T, H]
        float       *A_ptr,                    // [total_T, H, BT]
  const int64_t     *cu_seqlens_ptr,           // [N+1]
  const int32_t     *chunk_indices_ptr,        // [total_num_chunks, 2]
  int32_t total_num_chunks
) {
  const int tid = threadIdx.x;
  const int warp_id = warp_uniform(tid / WARP_SIZE);
  const int lane_id = tid % WARP_SIZE;

  const int bid = blockIdx.x;
  const int k_head_id = blockIdx.y;

  // set up smem
  extern __shared__ __align__(1024) char smem_ptr[];
  const uint32_t smem = __cvta_generic_to_shared(smem_ptr);

  // g has size [BT,2] (2 v heads per 1 k head)
  // we will partition into 4 warps, each hold [BT/4,2]
  // during parallel scan, we need to store sum of each warp
  // -> hence extra 4 * 2 * sizeof(float)
  const uint32_t g_tmp_size = (4 + BT) * 2 * sizeof(float);

  const uint32_t g_tmp_smem = smem + K_size * NUM_STAGES;
  float *g_tmp_smem_ptr = reinterpret_cast<float *>(smem_ptr + K_size * NUM_STAGES);

  const uint32_t tma_mbar_addr = g_tmp_smem + g_tmp_size;
  const uint32_t mma_mbar_addr = tma_mbar_addr + 8 * NUM_STAGES;
  const uint32_t epi_mbar_addr = mma_mbar_addr + 8 * NUM_STAGES;
  const uint32_t taddr = epi_mbar_addr + 8 * NUM_STAGES;

  if (warp_id == 0) {
    // init mbar
    if (elect_sync()) {
      for (int i = 0; i < NUM_STAGES; i++) {
        mbarrier_init(tma_mbar_addr + i * 8, 1);
        mbarrier_init(mma_mbar_addr + i * 8, 1);
        mbarrier_init(epi_mbar_addr + i * 8, 128);
      }
      fence_mbarrier_init();
    }
  }
  else if (warp_id == 1) {
    // prefetch TMA descriptor
    if (elect_sync())
      prefetch_tensormap(&K_tmap);
  }
  __syncthreads();

  if (warp_id == NUM_WARPS - 1) {
    // TMA warp
    if (elect_sync()) {
      int stage_id = 0;
      int mma_parity = 1; 

      for (int global_chunk_id = bid; global_chunk_id < total_num_chunks; global_chunk_id += gridDim.x) {
        int2 tmp = reinterpret_cast<const int2 *>(chunk_indices_ptr)[global_chunk_id];
        const int seq_id = tmp.x;
        const int chunk_id = tmp.y;
        const int bos = cu_seqlens_ptr[seq_id];

        const uint32_t K_smem = smem + stage_id * K_size;
        const int off_t = bos + chunk_id * BT;
        const uint32_t mbar_addr = tma_mbar_addr + stage_id * 8;

        mbarrier_wait(mma_mbar_addr + stage_id * 8, mma_parity);  // wait MMA to release buffer
        tma_load_4d(K_smem, &K_tmap, 0, off_t, 0, k_head_id, mbar_addr);  // [Hg, K_dim/64, total_T, 64]
        mbarrier_arrive_expect_tx(mbar_addr, K_size);

        stage_id = (stage_id + 1) % NUM_STAGES;
        if (stage_id == 0)
          mma_parity ^= 1;
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

      constexpr uint32_t idesc = make_tcgen05_idesc(BT, BT);

      for (int global_chunk_id = bid; global_chunk_id < total_num_chunks; global_chunk_id += gridDim.x) {
        // 128B swizzling
        constexpr uint64_t desc_base = (desc_encode(8 * 128) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
        const uint32_t K_smem = smem + stage_id * K_size;

        static_assert(BT * NUM_STAGES <= 512);
        const uint32_t acc_tmem = BT * stage_id;

        mbarrier_wait(tma_mbar_addr + stage_id * 8, tma_parity);  // wait for TMA data to arrive
        mbarrier_wait(epi_mbar_addr + stage_id * 8, epi_parity);  // wait for epilogue to release acc buffer
        tcgen05_fence_after_thread_sync();

        // i selects the [BT, 64] tile (increment by BT x 128B)
        // j selects the [BT, 16] tile (increment by 32B due to swizzling)
        for (int i = 0; i < K_dim / 64; i++)
          for (int j = 0; j < 64 / 16; j++) {
            const uint64_t k_desc = desc_base | ((K_smem + i * BT * 128 + j * 32) >> 4);
            const int enable_input_id = (i > 0) || (j > 0);
            tcgen05_mma(acc_tmem, k_desc, k_desc, idesc, enable_input_id);
          }
        tcgen05_commit(mma_mbar_addr + stage_id * 8);

        stage_id = (stage_id + 1) % NUM_STAGES;
        if (stage_id == 0) {
          tma_parity ^= 1;
          epi_parity ^= 1;
        }
      }
    }
  }
  else {
    // CUDA warps
    // for a and b, each threadblock loads [BT, H/Hg] = [64,2] tile
    // this is strided access so there is no good way to avoid uncoalesced access.
    //
    // due to how each warp/thread loads MMA result, for a and b:
    // - threadblock: [BT  , H/Hg] = [64,2]
    // - warp:        [BT/4, H/Hg] = [16,2]
    //
    // we can let lower  half warp handle head_id = k_head_id * 2 + 0
    //            higher half warp        head_id = k_head_id * 2 + 1

    static_assert(H / Hg == 2);
    const int head_id = k_head_id * 2 + (lane_id / 16);

    // A_log and dt_bias stays the same across for loop -> load once
    float A = -__expf(A_log_ptr[head_id]);
    float dt_bias = dt_bias_ptr[head_id];

    float kkt[BT];

    int stage_id = 0;
    int mma_parity = 0;

    float *g_warp_sum_ptr = g_tmp_smem_ptr;
    float *g_cu_smem_ptr = g_warp_sum_ptr + 4 * 2;

    for (int global_chunk_id = bid; global_chunk_id < total_num_chunks; global_chunk_id += gridDim.x) {
        int2 tmp = reinterpret_cast<const int2 *>(chunk_indices_ptr)[global_chunk_id];
        const int seq_id = tmp.x;
        const int chunk_id = tmp.y;
        const int bos = cu_seqlens_ptr[seq_id];
        const int eos = cu_seqlens_ptr[seq_id + 1];

        const int off_t = bos + chunk_id * BT + warp_id * 16 + (lane_id % 16);

        float b = __bfloat162float(b_ptr[off_t * H + head_id]);
        float a = __bfloat162float(a_ptr[off_t * H + head_id]);

        float beta = __frcp_rn(1.0f + __expf(-b));
        float g = A * __logf(1.0f + __expf(a + dt_bias));

        // store to gmem for future use
        if (off_t < eos)
          beta_ptr[off_t * H + head_id] = beta;

        // parallel scan among half warp (16)
        // illustrating for 4 lanes
        // lane  | lane0 | lane1 | lane2    | lane3
        // iter0 | a0    |    a1 |       a2 |          a3
        // iter1 | a0    | a0+a1 |    a1+a2 |       a2+a3
        // iter2 | a0    | a0+a1 | a0+a1+a2 | a0+a1+a2+a3
        for (int i = 1; i < 16; i *= 2) {
          float lower_g = __shfl_up_sync(0xFFFF'FFFF, g, i);  // g from lower lane
          if ((lane_id % 16) >= i)
            g += lower_g;
        }
        // store warp sum to smem. layout [4,2]
        if (lane_id % 16 == 15)
          g_warp_sum_ptr[warp_id * 2 + (lane_id / 16)] = g;
        bar_sync<1>(128);

        // add warp sum from lower warps
        // we finish doing g cumsum in registers
        if (warp_id >= 1) g += g_warp_sum_ptr[0 * 2 + (lane_id / 16)];
        if (warp_id >= 2) g += g_warp_sum_ptr[1 * 2 + (lane_id / 16)];
        if (warp_id >= 3) g += g_warp_sum_ptr[2 * 2 + (lane_id / 16)];

        // store to gmem for future use
        if (off_t < eos)
          g_cu_ptr[off_t * H + head_id] = g;

        // store to smem for computing Gamma. layout [2, BT]
        g_cu_smem_ptr[(lane_id / 16) * BT + (warp_id * 16 + (lane_id % 16))] = g;
        bar_sync<1>(128);

        // wait MMA
        if (warp_id == 0)
          mbarrier_wait(mma_mbar_addr + stage_id * 8, mma_parity);
        bar_sync<1>(128);
        tcgen05_fence_after_thread_sync();

        // load MMA result
        // only lower half warp contains result
        tcgen05_ld<SHAPE::_32x32b, 64>(kkt, 0, stage_id * BT);
        tcgen05_wait_ld();
        tcgen05_fence_before_thread_sync();
        mbarrier_arrive(epi_mbar_addr + stage_id * 8);

        // row/col within [BT,BT] tile
        const int row = warp_id * 16 + (lane_id % 16);
        for (int col = 0; col < BT; col++) {
          kkt[col] = __shfl_up_sync(0xFFFF'FFFF, kkt[col], 16);  // broadcast from lower half to upper half

          // strict lower mask + time mask
          if (row > col && bos + chunk_id * BT + col < eos) {
            kkt[col] *= beta * __expf(g - g_cu_smem_ptr[(lane_id / 16) * BT + col]);
          } else {
            kkt[col] = 0;
          }
        }

        // store A to gmem
        if (off_t < eos) {
          for (int i = 0; i < BT / 8; i++)
            stg_u32x8(A_ptr + (off_t * H * BT + head_id * BT + i * 8), kkt + i * 8);
        }

        stage_id = (stage_id + 1) % NUM_STAGES;
        if (stage_id == 0)
          mma_parity ^= 1;
    }
  }

  __syncthreads();
  if (warp_id == 0)
    tcgen05_dealloc(0, 512);
}

static
CUtensorMap encode_tma(void *ptr, uint64_t T, uint64_t H, uint64_t dim) {
  CUtensorMap tmap;

  // natural shape: [T, H, dim]
  // permuted shape: [H, dim/64, T, 64]
  constexpr uint32_t rank = 4;
  uint64_t globalDim[rank] = {64, T, dim / 64, H};
  uint64_t globalStrides[rank - 1] = {H * dim * sizeof(nv_bfloat16), 128, dim * sizeof(nv_bfloat16)};
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

void kkt_v1(
  TensorView K,
  TensorView A_log,
  TensorView a,
  TensorView dt_bias,
  TensorView b,
  TensorView g_cu,
  TensorView beta,
  TensorView A,
  TensorView cu_seqlens,
  TensorView chunk_indices,
  int total_num_chunks
) {
  const int T = K.size(0);
  const int Hg = K.size(1);

  auto K_tmap = encode_tma(K.data_ptr(), T, Hg, K_dim);

  auto *A_log_ptr         = reinterpret_cast<const float *>(A_log.data_ptr());
  auto *a_ptr             = reinterpret_cast<const nv_bfloat16 *>(a.data_ptr());
  auto *dt_bias_ptr       = reinterpret_cast<const float *>(dt_bias.data_ptr());
  auto *b_ptr             = reinterpret_cast<const nv_bfloat16 *>(b.data_ptr());
  auto *g_cu_ptr          = reinterpret_cast<float *>(g_cu.data_ptr());
  auto *beta_ptr          = reinterpret_cast<float *>(beta.data_ptr());
  auto *A_ptr             = reinterpret_cast<float *>(A.data_ptr());
  auto *cu_seqlens_ptr    = reinterpret_cast<const int64_t *>(cu_seqlens.data_ptr());
  auto *chunk_indices_ptr = reinterpret_cast<const int32_t *>(chunk_indices.data_ptr());

  constexpr int NUM_STAGES = 4;
  constexpr int smem_size = K_size * NUM_STAGES
                          + (4 + BT) * 2 * sizeof(float)  // scratchpad for g_cu
                          + 3 * NUM_STAGES * 8            // tma, mma, epi mbar for each stage
                          + 4;                            // taddr

  auto kernel = kkt_v1_kernel_cutlass<NUM_STAGES>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

  dim3 grid(148 / Hg, Hg);
  kernel<<<grid, TB_SIZE, smem_size>>>(
    K_tmap, A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
    g_cu_ptr, beta_ptr, A_ptr,
    cu_seqlens_ptr, chunk_indices_ptr, total_num_chunks);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(kkt_v1, kkt_v1);
