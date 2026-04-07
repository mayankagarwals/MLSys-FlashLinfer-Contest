#include "cuda_utils.h"

constexpr int H = 8;
constexpr int Hg = 4;
constexpr int BT = 64;
constexpr int K_dim = 128;
constexpr int V_dim = 128;
constexpr int BV = 64;

constexpr int NUM_CUDA_WARPS = 4;
constexpr int NUM_WARPS = NUM_CUDA_WARPS + 2;  // CUDA + MMA + TMA
constexpr int WARP_SIZE = 32;
constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;

constexpr uint32_t Q_tc_size = BT * K_dim * sizeof(nv_bfloat16);
constexpr uint32_t K_tc_size = BT * K_dim * sizeof(nv_bfloat16);
constexpr uint32_t Q_rm_size = BT * K_dim * sizeof(nv_bfloat16);
constexpr uint32_t H_rm_size = BV * K_dim * sizeof(nv_bfloat16);
constexpr uint32_t V_rm_size = BT * BV * sizeof(nv_bfloat16);
constexpr uint32_t O_size = BT * BV * sizeof(float);
constexpr uint32_t G_size = BT * sizeof(float);

template <int ROWS, int COLS>
__device__ __forceinline__ void load_bf16_tile_rowmajor(
    nv_bfloat16 *dst_ptr,
    const nv_bfloat16 *src_ptr,
    int src_row_stride,
    int valid_rows) {
  static_assert(COLS % 8 == 0);

  constexpr int VEC_ELEMS = 8;
  constexpr int VECS_PER_ROW = COLS / VEC_ELEMS;
  constexpr int TOTAL_VECS = ROWS * VECS_PER_ROW;

  auto *dst_vec = reinterpret_cast<uint4 *>(dst_ptr);
  auto *src_vec = reinterpret_cast<const uint4 *>(src_ptr);

  for (int vec_id = threadIdx.x; vec_id < TOTAL_VECS; vec_id += TB_SIZE) {
    const int row = vec_id / VECS_PER_ROW;
    const int col_vec = vec_id % VECS_PER_ROW;

    uint4 data = make_uint4(0, 0, 0, 0);
    if (row < valid_rows) {
      data = src_vec[row * (src_row_stride / VEC_ELEMS) + col_vec];
    }
    dst_vec[vec_id] = data;
  }
}

__global__ __block_size__((TB_SIZE, 1, 1)) void o_v1_kernel_cutlass(
    const __grid_constant__ CUtensorMap Q_tmap,  // [T, Hg, K_dim]
    const __grid_constant__ CUtensorMap K_tmap,  // [T, Hg, K_dim]
    const nv_bfloat16 *q_ptr,                    // [T, Hg, K_dim]
    const nv_bfloat16 *v_ptr,                    // [T, H, V_dim]
    const nv_bfloat16 *h_ptr,                    // [total_num_chunks, H, V_dim, K_dim]
    const float *g_cu_ptr,                       // [T, H]
    nv_bfloat16 *o_ptr,                          // [T, H, V_dim]
    const int64_t *cu_seqlens_ptr,               // [N+1]
    const int32_t *chunk_indices_ptr,            // [total_num_chunks, 2]
    float scale) {
  const int tid = threadIdx.x;
  const int warp_id = warp_uniform(tid / WARP_SIZE);
  const int lane_id = tid % WARP_SIZE;

  const int bid = blockIdx.x;
  const int head_id = blockIdx.y;
  const int global_chunk_id = bid >> 1;
  const int i_v = bid & 1;
  const int k_head_id = head_id / (H / Hg);

  int2 tmp = reinterpret_cast<const int2 *>(chunk_indices_ptr)[global_chunk_id];
  const int seq_id = tmp.x;
  const int chunk_id = tmp.y;
  const int bos = cu_seqlens_ptr[seq_id];
  const int eos = cu_seqlens_ptr[seq_id + 1];
  const int seqlen = eos - bos;
  const int chunk_bos = chunk_id * BT;
  const int valid_rows = max(0, min(BT, seqlen - chunk_bos));

  extern __shared__ __align__(1024) char smem_ptr[];
  const uint32_t smem = __cvta_generic_to_shared(smem_ptr);

  const uint32_t q_tc_smem = smem;
  const uint32_t k_tc_smem = q_tc_smem + Q_tc_size;
  const uint32_t q_rm_smem = k_tc_smem + K_tc_size;
  const uint32_t h_rm_smem = q_rm_smem + Q_rm_size;
  const uint32_t v_rm_smem = h_rm_smem + H_rm_size;
  const uint32_t o_smem = v_rm_smem + V_rm_size;
  const uint32_t g_smem = o_smem + O_size;
  const uint32_t q_mbar_addr = g_smem + G_size;
  const uint32_t k_mbar_addr = q_mbar_addr + 8;
  const uint32_t mma_mbar_addr = k_mbar_addr + 8;
  const uint32_t taddr = mma_mbar_addr + 8;

  auto *q_rm_smem_ptr =
      reinterpret_cast<nv_bfloat16 *>(smem_ptr + (q_rm_smem - smem));
  auto *h_rm_smem_ptr =
      reinterpret_cast<nv_bfloat16 *>(smem_ptr + (h_rm_smem - smem));
  auto *v_rm_smem_ptr =
      reinterpret_cast<nv_bfloat16 *>(smem_ptr + (v_rm_smem - smem));
  auto *o_smem_ptr = reinterpret_cast<float *>(smem_ptr + (o_smem - smem));
  auto *g_smem_ptr = reinterpret_cast<float *>(smem_ptr + (g_smem - smem));

  if (warp_id == 0 && elect_sync()) {
    mbarrier_init(q_mbar_addr, 1);
    mbarrier_init(k_mbar_addr, 1);
    mbarrier_init(mma_mbar_addr, 1);
    fence_mbarrier_init();
  }
  __syncthreads();

  const auto *q_gmem_ptr =
      q_ptr + (bos * Hg + k_head_id) * K_dim + chunk_bos * Hg * K_dim;
  const auto *v_gmem_ptr =
      v_ptr + (bos * H + head_id) * V_dim + chunk_bos * H * V_dim + i_v * BV;
  const auto *h_gmem_ptr =
      h_ptr + ((global_chunk_id * H + head_id) * V_dim + i_v * BV) * K_dim;
  const auto *g_gmem_ptr = g_cu_ptr + bos * H + head_id + chunk_bos * H;

  load_bf16_tile_rowmajor<BT, K_dim>(
      q_rm_smem_ptr, q_gmem_ptr, Hg * K_dim, valid_rows);
  load_bf16_tile_rowmajor<BV, K_dim>(
      h_rm_smem_ptr, h_gmem_ptr, K_dim, BV);
  load_bf16_tile_rowmajor<BT, BV>(
      v_rm_smem_ptr, v_gmem_ptr, H * V_dim, valid_rows);

  for (int row = tid; row < BT; row += TB_SIZE) {
    g_smem_ptr[row] = (row < valid_rows) ? g_gmem_ptr[row * H] : 0.0f;
  }
  __syncthreads();

  if (warp_id == NUM_WARPS - 1) {
    if (elect_sync()) {
      prefetch_tensormap(&Q_tmap);
      prefetch_tensormap(&K_tmap);
      const int off_t = bos + chunk_bos;
      tma_load_4d(q_tc_smem, &Q_tmap, 0, off_t, 0, k_head_id, q_mbar_addr);
      mbarrier_arrive_expect_tx(q_mbar_addr, Q_tc_size);
      tma_load_4d(k_tc_smem, &K_tmap, 0, off_t, 0, k_head_id, k_mbar_addr);
      mbarrier_arrive_expect_tx(k_mbar_addr, K_tc_size);
    }
  } else if (warp_id == NUM_WARPS - 2) {
    tcgen05_alloc(taddr, BT);

    if (elect_sync()) {
      mbarrier_wait(q_mbar_addr, 0);
      mbarrier_wait(k_mbar_addr, 0);
      tcgen05_fence_after_thread_sync();

      constexpr uint32_t idesc = make_tcgen05_idesc(BT, BT);
      constexpr uint64_t desc_base =
          (desc_encode(8 * 128) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);

      for (int i = 0; i < K_dim / 64; ++i) {
        for (int j = 0; j < 64 / 16; ++j) {
          const uint64_t q_desc =
              desc_base | ((q_tc_smem + i * BT * 128 + j * 32) >> 4);
          const uint64_t k_desc =
              desc_base | ((k_tc_smem + i * BT * 128 + j * 32) >> 4);
          tcgen05_mma(0, q_desc, k_desc, idesc, (i > 0) || (j > 0));
        }
      }
      tcgen05_commit(mma_mbar_addr);
    }
  } else {
    if (lane_id < 16) {
      const int row = warp_id * 16 + lane_id;
      const float gamma = __expf(g_smem_ptr[row]);

      for (int col = 0; col < BV; ++col) {
        float acc = 0.0f;
        if (row < valid_rows) {
          for (int kk = 0; kk < K_dim; ++kk) {
            const float q = __bfloat162float(q_rm_smem_ptr[row * K_dim + kk]);
            const float h = __bfloat162float(h_rm_smem_ptr[col * K_dim + kk]);
            acc += q * h;
          }
          acc *= gamma;
        }
        o_smem_ptr[row * BV + col] = acc;
      }
    }

    if (warp_id == 0 && elect_sync()) {
      mbarrier_wait(mma_mbar_addr, 0);
    }
    bar_sync<1>(NUM_CUDA_WARPS * WARP_SIZE);
    tcgen05_fence_after_thread_sync();

    float qk_row[BT];
    tcgen05_ld<SHAPE::_32x32b, BT>(qk_row, 0, 0);
    tcgen05_wait_ld();
    tcgen05_fence_before_thread_sync();

    if (lane_id < 16) {
      const int row = warp_id * 16 + lane_id;
      if (row < valid_rows) {
        alignas(16) nv_bfloat16 out[BV];
        for (int col = 0; col < BV; ++col) {
          float acc = o_smem_ptr[row * BV + col];
          for (int t = 0; t < valid_rows; ++t) {
            float a = 0.0f;
            if (row >= t) {
              a = qk_row[t] * __expf(g_smem_ptr[row] - g_smem_ptr[t]);
              a = __bfloat162float(__float2bfloat16_rn(a));
            }
            const float v =
                __bfloat162float(v_rm_smem_ptr[t * BV + col]);
            acc += a * v;
          }
          out[col] = __float2bfloat16_rn(acc * scale);
        }

        auto *out_ptr =
            o_ptr + ((bos + chunk_bos + row) * H + head_id) * V_dim + i_v * BV;
        auto *out_words = reinterpret_cast<uint4 *>(out_ptr);
        auto *src_words = reinterpret_cast<uint4 *>(out);
#pragma unroll
        for (int i = 0; i < BV / 8; ++i) {
          out_words[i] = src_words[i];
        }
      }
    }
  }

  __syncthreads();
  if (warp_id == 0) {
    tcgen05_dealloc(0, BT);
  }
}

static CUtensorMap encode_tma(void *ptr, uint64_t T, uint64_t H, uint64_t dim) {
  CUtensorMap tmap;

  constexpr uint32_t rank = 4;
  uint64_t globalDim[rank] = {64, T, dim / 64, H};
  uint64_t globalStrides[rank - 1] = {
      H * dim * sizeof(nv_bfloat16), 128, dim * sizeof(nv_bfloat16)};
  uint32_t boxDim[rank] = {64, BT, dim / 64, 1};
  uint32_t elementStrides[rank] = {1, 1, 1, 1};

  cuTensorMapEncodeTiled(
      &tmap,
      CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
      rank,
      ptr,
      globalDim,
      globalStrides,
      boxDim,
      elementStrides,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return tmap;
}

void o_v1(
    TensorView q,
    TensorView k,
    TensorView v,
    TensorView h,
    TensorView g_cu,
    TensorView o,
    TensorView cu_seqlens,
    TensorView chunk_indices,
    int total_num_chunks,
    float scale) {
  const int T = q.size(0);
  const int Hg_ = q.size(1);

  auto Q_tmap = encode_tma(q.data_ptr(), T, Hg_, K_dim);
  auto K_tmap = encode_tma(k.data_ptr(), T, Hg_, K_dim);

  auto *q_ptr = reinterpret_cast<const nv_bfloat16 *>(q.data_ptr());
  auto *v_ptr = reinterpret_cast<const nv_bfloat16 *>(v.data_ptr());
  auto *h_ptr = reinterpret_cast<const nv_bfloat16 *>(h.data_ptr());
  auto *g_cu_ptr = reinterpret_cast<const float *>(g_cu.data_ptr());
  auto *o_ptr = reinterpret_cast<nv_bfloat16 *>(o.data_ptr());
  auto *cu_seqlens_ptr = reinterpret_cast<const int64_t *>(cu_seqlens.data_ptr());
  auto *chunk_indices_ptr =
      reinterpret_cast<const int32_t *>(chunk_indices.data_ptr());

  constexpr int smem_size = Q_tc_size + K_tc_size + Q_rm_size + H_rm_size +
                            V_rm_size + O_size + G_size + 3 * 8 + 4;

  auto kernel = o_v1_kernel_cutlass;
  cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

  dim3 grid(total_num_chunks * (V_dim / BV), H);
  kernel<<<grid, TB_SIZE, smem_size>>>(
      Q_tmap,
      K_tmap,
      q_ptr,
      v_ptr,
      h_ptr,
      g_cu_ptr,
      o_ptr,
      cu_seqlens_ptr,
      chunk_indices_ptr,
      scale);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(o_v1, o_v1);
