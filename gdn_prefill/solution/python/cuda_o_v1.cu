#include "cuda_utils.h"

#include <cstdint>
#include <cuda_bf16.h>
#include <math.h>

//// Tile Config
constexpr uint32_t BLOCK_T = 64U;
constexpr uint32_t HEAD_DIM = 128U;
constexpr uint32_t VALUE_DIM = 128U;
constexpr uint32_t BLOCK_V = 64U;

//// Head Config
constexpr uint32_t NUM_OUTPUT_HEADS = 8U;
constexpr uint32_t NUM_QK_HEADS = 4U;
constexpr uint32_t HEADS_PER_QK_HEAD = NUM_OUTPUT_HEADS / NUM_QK_HEADS;

//// MMA Config
constexpr uint32_t MMA_M = BLOCK_T;
constexpr uint32_t MMA_N = BLOCK_T;
constexpr uint32_t MMA_K = 16U;
constexpr uint32_t NUM_MMA_STEPS = BLOCK_T / MMA_K;
constexpr uint32_t NUM_SWIZZLE_ATOMS = HEAD_DIM / BLOCK_T;
constexpr uint32_t BYTES_ONE_MMA = MMA_K * sizeof(nv_bfloat16);

//// Threads per Block
constexpr uint32_t NUM_CUDA_WARPS = 4U;
constexpr uint32_t TMA_WARP = NUM_CUDA_WARPS;
constexpr uint32_t MMA_WARP = NUM_CUDA_WARPS + 1U;
constexpr uint32_t NUM_WARPS = NUM_CUDA_WARPS + 2U;
constexpr uint32_t WARP_SIZE = 32U;
constexpr uint32_t NUM_THREADS = NUM_WARPS * WARP_SIZE;
constexpr uint32_t ROWS_PER_WARP = 16U;

//// Swizzle 128B, 1024 bit width atom
constexpr uint32_t TILE_ATOM = 8U;
constexpr uint32_t TILE_ATOM_ELEMS = TILE_ATOM * TILE_ATOM;
constexpr uint32_t SWIZZLE_HEIGHT = BLOCK_T;
constexpr uint32_t SWIZZLE_WIDTH = 128U / sizeof(nv_bfloat16);
constexpr uint32_t SWIZZLE_BYTES =
    SWIZZLE_HEIGHT * SWIZZLE_WIDTH * sizeof(nv_bfloat16);
constexpr uint32_t SWIZZLE_SBO = 8U * 128U;

//// Shared Memory Offsets in Bytes
constexpr uint32_t A_SMEM_SIZE = BLOCK_T * HEAD_DIM * sizeof(nv_bfloat16);
constexpr uint32_t B_SMEM_SIZE = BLOCK_T * HEAD_DIM * sizeof(nv_bfloat16);
constexpr uint32_t V_SMEM_SIZE = BLOCK_V * BLOCK_T * sizeof(nv_bfloat16);
constexpr uint32_t OUTPUT_SMEM_SIZE = BLOCK_T * BLOCK_V * sizeof(float);
constexpr uint32_t G_SMEM_SIZE = BLOCK_T * sizeof(float);
constexpr uint32_t OFFSET_MATRIX_A = 0U;
constexpr uint32_t OFFSET_MATRIX_B = OFFSET_MATRIX_A + A_SMEM_SIZE;
constexpr uint32_t OFFSET_V = OFFSET_MATRIX_B + B_SMEM_SIZE;
constexpr uint32_t OFFSET_OUTPUT = OFFSET_V + V_SMEM_SIZE;
constexpr uint32_t OFFSET_G = OFFSET_OUTPUT + OUTPUT_SMEM_SIZE;
constexpr uint32_t OFFSET_QKH_TMA_BAR = OFFSET_G + G_SMEM_SIZE;
constexpr uint32_t OFFSET_V_TMA_BAR = OFFSET_QKH_TMA_BAR + 8U;
constexpr uint32_t OFFSET_MMA_BAR = OFFSET_V_TMA_BAR + 8U;
constexpr uint32_t OFFSET_TMEM_ADDR = OFFSET_MMA_BAR + 8U;
constexpr uint32_t SMEM_SIZE = (OFFSET_TMEM_ADDR + 4U + 1023U) & ~1023U;

//// Tensor Memory
constexpr uint32_t MAX_COLUMNS = 128U;
constexpr uint32_t OUTPUT_TMEM_COL = BLOCK_T;

__device__ __forceinline__ int make_tile_layout_index(int tile_rows, int row,
                                                      int col) {
  return (col / TILE_ATOM) * (tile_rows * TILE_ATOM) +
         (row / TILE_ATOM) * TILE_ATOM_ELEMS + (row % TILE_ATOM) * TILE_ATOM +
         (col % TILE_ATOM);
}

static CUtensorMap encode_tma(void *ptr, uint64_t outer, uint64_t rows,
                              uint64_t cols) {
  CUtensorMap tmap;
  constexpr uint32_t rank = 4;
  uint64_t globalDim[rank] = {SWIZZLE_WIDTH, rows, cols / SWIZZLE_WIDTH, outer};
  uint64_t globalStrides[rank - 1] = {
      cols * sizeof(nv_bfloat16),
      SWIZZLE_WIDTH * sizeof(nv_bfloat16),
      rows * cols * sizeof(nv_bfloat16),
  };
  uint32_t boxDim[rank] = {SWIZZLE_WIDTH, BLOCK_T,
                           static_cast<uint32_t>(cols / SWIZZLE_WIDTH), 1};
  uint32_t elementStrides[rank] = {1, 1, 1, 1};

  cuTensorMapEncodeTiled(
      &tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, ptr, globalDim,
      globalStrides, boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return tmap;
}

static CUtensorMap encode_v_tma(void *ptr, uint64_t rows, uint64_t cols) {
  CUtensorMap tmap;
  init_tma_desc_3d(&tmap, reinterpret_cast<const __nv_bfloat16 *>(ptr), rows,
                   cols, BLOCK_V, BLOCK_T);
  return tmap;
}

template <int NUM_ATOMS>
__device__ __forceinline__ void mma_swizzled(uint32_t output_tmem,
                                             uint32_t matrix_a_smem,
                                             uint32_t matrix_b_smem) {
  constexpr uint32_t idesc = make_tcgen05_idesc(MMA_M, MMA_N);
  constexpr uint64_t desc_base =
      (desc_encode(SWIZZLE_SBO) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);

#pragma unroll
  for (int atom = 0; atom < NUM_ATOMS; ++atom) {
#pragma unroll
    for (int mi = 0; mi < NUM_MMA_STEPS; ++mi) {
      const uint64_t a_desc =
          desc_base |
          ((matrix_a_smem + atom * SWIZZLE_BYTES + mi * BYTES_ONE_MMA) >> 4);
      const uint64_t b_desc =
          desc_base |
          ((matrix_b_smem + atom * SWIZZLE_BYTES + mi * BYTES_ONE_MMA) >> 4);
      const int accum = (atom > 0) || (mi > 0);
      tcgen05_mma(output_tmem, a_desc, b_desc, idesc, accum);
    }
  }
}

__device__ __forceinline__ void mma_noswizzle_64x64(uint32_t output_tmem,
                                                    uint32_t matrix_a_smem,
                                                    uint32_t matrix_b_smem) {
  constexpr uint32_t idesc = make_tcgen05_idesc(MMA_M, MMA_N);
  constexpr uint32_t matrix_a_k_stride_bytes =
      BLOCK_T * MMA_K * sizeof(nv_bfloat16);
  constexpr uint32_t matrix_b_k_stride_bytes =
      BLOCK_V * MMA_K * sizeof(nv_bfloat16);
#pragma unroll
  for (int ki = 0; ki < NUM_MMA_STEPS; ++ki) {
    const uint32_t a_base = matrix_a_smem + ki * matrix_a_k_stride_bytes;
    const uint32_t b_base = matrix_b_smem + ki * matrix_b_k_stride_bytes;
    tcgen05_mma(output_tmem,
                make_tcgen05_desc_noswizzle(a_base, BLOCK_T,
                                            BLOCK_T * sizeof(nv_bfloat16)),
                make_tcgen05_desc_noswizzle(b_base, BLOCK_V,
                                            BLOCK_T * sizeof(nv_bfloat16)),
                idesc, ki > 0);
  }
}

__global__ __block_size__((NUM_THREADS, 1, 1)) void o_v1_kernel_swizzled(
    const __grid_constant__ CUtensorMap q_tmap,
    const __grid_constant__ CUtensorMap k_tmap,
    const __grid_constant__ CUtensorMap v_tmap,
    const __grid_constant__ CUtensorMap h_tmap, const float *g_chunks_ptr,
    nv_bfloat16 *o_ptr, const int64_t *cu_seqlens_ptr,
    const int32_t *chunk_indices_ptr, int32_t total_num_chunks, float scale) {
  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  const int v_tile = blockIdx.x;
  const int global_chunk_id = blockIdx.y;
  const int head_id = blockIdx.z;
  if (global_chunk_id >= total_num_chunks) {
    return;
  }

  const int q_head_id = head_id / HEADS_PER_QK_HEAD;
  const int k_head_id = head_id / HEADS_PER_QK_HEAD;

  int2 chunk_meta =
      reinterpret_cast<const int2 *>(chunk_indices_ptr)[global_chunk_id];
  const int seq_id = chunk_meta.x;
  const int chunk_id = chunk_meta.y;
  const int64_t bos = cu_seqlens_ptr[seq_id];
  const int64_t eos = cu_seqlens_ptr[seq_id + 1];
  const int64_t chunk_start = bos + static_cast<int64_t>(chunk_id) * BLOCK_T;
  const int64_t remaining = eos - chunk_start;
  const int chunk_len = remaining <= 0
                            ? 0
                            : (remaining < BLOCK_T ? static_cast<int>(remaining)
                                                   : static_cast<int>(BLOCK_T));
  const int v_start = v_tile * BLOCK_V;

  extern __shared__ __align__(1024) char smem_ptr[];
  const uint32_t smem = __cvta_generic_to_shared(smem_ptr);

  const uint32_t matrix_a_smem = smem + OFFSET_MATRIX_A;
  const uint32_t matrix_b_smem = smem + OFFSET_MATRIX_B;
  const uint32_t v_smem = smem + OFFSET_V;
  const uint32_t output_smem = smem + OFFSET_OUTPUT;
  const uint32_t g_smem = smem + OFFSET_G;
  const uint32_t qkh_tma_barrier = smem + OFFSET_QKH_TMA_BAR;
  const uint32_t v_tma_barrier = smem + OFFSET_V_TMA_BAR;
  const uint32_t mma_barrier = smem + OFFSET_MMA_BAR;
  const uint32_t tmem_alloc_smem = smem + OFFSET_TMEM_ADDR;

  float *output_smem_ptr =
      reinterpret_cast<float *>(smem_ptr + (output_smem - smem));
  float *g_smem_ptr = reinterpret_cast<float *>(smem_ptr + (g_smem - smem));

  if (warp_id == TMA_WARP) {
    if (elect_sync()) {
      mbarrier_init(qkh_tma_barrier, 1);
      mbarrier_init(v_tma_barrier, 1);
      mbarrier_init(mma_barrier, 1);
      prefetch_tensormap(&q_tmap);
      prefetch_tensormap(&k_tmap);
      prefetch_tensormap(&v_tmap);
      prefetch_tensormap(&h_tmap);
      fence_mbarrier_init();
    }
  } else if (warp_id == MMA_WARP) {
    tcgen05_alloc(tmem_alloc_smem, MAX_COLUMNS);
  }

  for (int i = tid; i < BLOCK_T; i += NUM_THREADS) {
    g_smem_ptr[i] =
        g_chunks_ptr[(global_chunk_id * NUM_OUTPUT_HEADS + head_id) * BLOCK_T +
                     i];
  }
  __syncthreads();

  constexpr int QK_TMA_PHASE = 0;
  constexpr int QH_TMA_PHASE = 1;
  constexpr int QK_MMA_PHASE = 0;
  constexpr int OV_MMA_PHASE = 1;
  constexpr int QH_MMA_PHASE = 0;
  const int q_outer = global_chunk_id * NUM_QK_HEADS + q_head_id;
  const int k_outer = global_chunk_id * NUM_QK_HEADS + k_head_id;
  const int h_outer = global_chunk_id * NUM_OUTPUT_HEADS + head_id;
  const int v_row =
      ((global_chunk_id * NUM_OUTPUT_HEADS + head_id) * (VALUE_DIM / BLOCK_V) +
       v_tile) *
      BLOCK_V;

  if (warp_id == TMA_WARP && elect_sync()) {
    tma_load_4d(matrix_a_smem, &q_tmap, 0, 0, 0, q_outer, qkh_tma_barrier);
    tma_load_4d(matrix_b_smem, &k_tmap, 0, 0, 0, k_outer, qkh_tma_barrier);
    tma_load_3d(v_smem, &v_tmap, 0, v_row, 0, v_tma_barrier);
    mbarrier_arrive_expect_tx(qkh_tma_barrier, A_SMEM_SIZE + B_SMEM_SIZE);
    mbarrier_arrive_expect_tx(v_tma_barrier, V_SMEM_SIZE);
  }
  if (warp_id == MMA_WARP && elect_sync()) {
    mbarrier_wait(qkh_tma_barrier, QK_TMA_PHASE);
    tcgen05_fence_after_thread_sync();
    mma_swizzled<NUM_SWIZZLE_ATOMS>(0, matrix_a_smem, matrix_b_smem);
    tcgen05_commit(mma_barrier);
  }

  nv_bfloat16 *attn_smem_ptr =
      reinterpret_cast<nv_bfloat16 *>(smem_ptr + (matrix_a_smem - smem));

  if (warp_id < NUM_CUDA_WARPS) {
    mbarrier_wait(mma_barrier, QK_MMA_PHASE);
    tcgen05_fence_after_thread_sync();
    float reg[BLOCK_T];
    tcgen05_ld<SHAPE::_32x32b, BLOCK_T>(reg, 0, 0);
    tcgen05_wait_ld();
    if (lane_id < ROWS_PER_WARP) {
      const int row = warp_id * ROWS_PER_WARP + lane_id;
#pragma unroll
      for (int c = 0; c < BLOCK_T; ++c) {
        if (row < chunk_len && c < chunk_len && c <= row) {
          reg[c] *= __expf(g_smem_ptr[row] - g_smem_ptr[c]);
        } else {
          reg[c] = 0.0f;
        }
        const int tile_idx = make_tile_layout_index(BLOCK_T, row, c);
        attn_smem_ptr[tile_idx] = __float2bfloat16_rn(reg[c]);
      }
    }
  }
  tcgen05_fence_before_thread_sync();
  __syncthreads();

  if (warp_id == MMA_WARP && elect_sync()) {
    mbarrier_wait(v_tma_barrier, 0);
    tcgen05_fence_after_thread_sync();
    mma_noswizzle_64x64(OUTPUT_TMEM_COL, matrix_a_smem, v_smem);
    tcgen05_commit(mma_barrier);
  }

  if (warp_id < NUM_CUDA_WARPS) {
    mbarrier_wait(mma_barrier, OV_MMA_PHASE);
    tcgen05_fence_after_thread_sync();
    float reg[BLOCK_V];
    tcgen05_ld<SHAPE::_32x32b, BLOCK_T>(reg, 0, OUTPUT_TMEM_COL);
    tcgen05_wait_ld();
    if (lane_id < ROWS_PER_WARP) {
      const int row = warp_id * ROWS_PER_WARP + lane_id;
#pragma unroll
      for (int col = 0; col < BLOCK_V; ++col) {
        output_smem_ptr[row * BLOCK_V + col] = reg[col];
      }
    }
  }

  if (warp_id == TMA_WARP) {
    mbarrier_wait(mma_barrier, OV_MMA_PHASE);
    tcgen05_fence_after_thread_sync();
  }
  if (warp_id == TMA_WARP && elect_sync()) {
    tma_load_4d(matrix_a_smem, &q_tmap, 0, 0, 0, q_outer, qkh_tma_barrier);
    tma_load_4d(matrix_b_smem, &h_tmap, 0, v_start, 0, h_outer,
                qkh_tma_barrier);
    mbarrier_arrive_expect_tx(qkh_tma_barrier, A_SMEM_SIZE + B_SMEM_SIZE);
  }

  if (warp_id == MMA_WARP && elect_sync()) {
    mbarrier_wait(qkh_tma_barrier, QH_TMA_PHASE);
    tcgen05_fence_after_thread_sync();
    mma_swizzled<NUM_SWIZZLE_ATOMS>(0, matrix_a_smem, matrix_b_smem);
    tcgen05_commit(mma_barrier);
  }

  if (warp_id < NUM_CUDA_WARPS) {
    mbarrier_wait(mma_barrier, QH_MMA_PHASE);
    tcgen05_fence_after_thread_sync();
    float reg[BLOCK_V];
    tcgen05_ld<SHAPE::_32x32b, BLOCK_T>(reg, 0, 0);
    tcgen05_wait_ld();
    if (lane_id < ROWS_PER_WARP) {
      const int row = warp_id * ROWS_PER_WARP + lane_id;
      if (row < chunk_len) {
        const float g = __expf(g_smem_ptr[row]);
        for (int col = 0; col < BLOCK_V; ++col) {
          const float value =
              scale * (output_smem_ptr[row * BLOCK_V + col] + reg[col] * g);
          o_ptr[((chunk_start + row) * NUM_OUTPUT_HEADS + head_id) * VALUE_DIM +
                v_start + col] = __float2bfloat16_rn(value);
        }
      }
    }
  }

  tcgen05_fence_before_thread_sync();
  __syncthreads();
  if (warp_id == MMA_WARP) {
    tcgen05_dealloc(0, MAX_COLUMNS);
  }
  return;
}

void o_v1(TensorView q_chunks, TensorView k_chunks, TensorView v_new_t,
          TensorView h, TensorView g_chunks, TensorView o,
          TensorView cu_seqlens, TensorView chunk_indices, int total_num_chunks,
          double scale) {
  auto q_tmap = encode_tma(q_chunks.data_ptr(), total_num_chunks * NUM_QK_HEADS,
                           BLOCK_T, HEAD_DIM);
  auto k_tmap = encode_tma(k_chunks.data_ptr(), total_num_chunks * NUM_QK_HEADS,
                           BLOCK_T, HEAD_DIM);
  auto v_tmap =
      encode_v_tma(v_new_t.data_ptr(),
                   total_num_chunks * NUM_OUTPUT_HEADS * VALUE_DIM, BLOCK_T);
  auto h_tmap = encode_tma(h.data_ptr(), total_num_chunks * NUM_OUTPUT_HEADS,
                           VALUE_DIM, HEAD_DIM);

  auto *g_chunks_ptr = reinterpret_cast<const float *>(g_chunks.data_ptr());
  auto *o_ptr = reinterpret_cast<nv_bfloat16 *>(o.data_ptr());
  auto *cu_seqlens_ptr =
      reinterpret_cast<const int64_t *>(cu_seqlens.data_ptr());
  auto *chunk_indices_ptr =
      reinterpret_cast<const int32_t *>(chunk_indices.data_ptr());

  auto kernel = o_v1_kernel_swizzled;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       SMEM_SIZE);

  dim3 grid(VALUE_DIM / BLOCK_V, total_num_chunks, NUM_OUTPUT_HEADS);
  kernel<<<grid, NUM_THREADS, SMEM_SIZE>>>(
      q_tmap, k_tmap, v_tmap, h_tmap, g_chunks_ptr, o_ptr, cu_seqlens_ptr,
      chunk_indices_ptr, total_num_chunks, static_cast<float>(scale));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(o_v1, o_v1);
