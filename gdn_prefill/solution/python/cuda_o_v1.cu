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
constexpr uint32_t COLS_PER_FRAGMENT = 32U;
constexpr uint32_t REGS_PER_FRAGMENT = 16U;
constexpr uint32_t ROW_PAIR_STRIDE = 8U;
constexpr uint32_t LANES_PER_ROW_GROUP = 4U;
constexpr uint32_t FRAGMENT_STEPS = 8U;
constexpr uint32_t FRAGMENT_PAIRS = FRAGMENT_STEPS / 2U;

//// Swizzle 128B, 1024 bit width atom
constexpr uint32_t TILE_ATOM = 8U;
constexpr uint32_t TILE_ATOM_ELEMS = TILE_ATOM * TILE_ATOM;
constexpr uint32_t SWIZZLE_HEIGHT = BLOCK_T;
constexpr uint32_t SWIZZLE_WIDTH = 128U / sizeof(nv_bfloat16);
constexpr uint32_t SWIZZLE_BYTES =
    SWIZZLE_HEIGHT * SWIZZLE_WIDTH * sizeof(nv_bfloat16);
constexpr uint32_t SWIZZLE_SBO = 8U * 128U;
constexpr uint32_t V_MMA_SBO = SWIZZLE_WIDTH * 8U * sizeof(nv_bfloat16);
constexpr uint32_t V_MMA_LBO = (BLOCK_T / 8U) * V_MMA_SBO;
constexpr uint32_t BYTES_ONE_MMA_MMAJOR =
    MMA_K * SWIZZLE_WIDTH * sizeof(nv_bfloat16);

//// Shared Memory Offsets in Bytes
constexpr uint32_t Q_SMEM_SIZE = BLOCK_T * HEAD_DIM * sizeof(nv_bfloat16);
constexpr uint32_t K_SMEM_SIZE = BLOCK_T * HEAD_DIM * sizeof(nv_bfloat16);
constexpr uint32_t V_SMEM_SIZE = BLOCK_V * BLOCK_T * sizeof(nv_bfloat16);
constexpr uint32_t H_SMEM_SIZE = BLOCK_T * HEAD_DIM * sizeof(nv_bfloat16);
constexpr uint32_t ATTN_SMEM_SIZE = BLOCK_T * BLOCK_T * sizeof(nv_bfloat16);
constexpr uint32_t OUTPUT_SMEM_SIZE = BLOCK_T * BLOCK_V * sizeof(float);
constexpr uint32_t G_SMEM_SIZE = BLOCK_T * sizeof(float);
constexpr uint32_t OFFSET_Q = 0U;
constexpr uint32_t OFFSET_K = OFFSET_Q + Q_SMEM_SIZE;
constexpr uint32_t OFFSET_V = OFFSET_K + K_SMEM_SIZE;
constexpr uint32_t OFFSET_H = OFFSET_V + V_SMEM_SIZE;
constexpr uint32_t OFFSET_ATTN = OFFSET_H + H_SMEM_SIZE;
constexpr uint32_t OFFSET_OUTPUT = OFFSET_ATTN + ATTN_SMEM_SIZE;
constexpr uint32_t OFFSET_G = OFFSET_OUTPUT + OUTPUT_SMEM_SIZE;
constexpr uint32_t OFFSET_QK_TMA_BAR = OFFSET_G + G_SMEM_SIZE;
constexpr uint32_t OFFSET_V_TMA_BAR = OFFSET_QK_TMA_BAR + 8U;
constexpr uint32_t OFFSET_H_TMA_BAR = OFFSET_V_TMA_BAR + 8U;
constexpr uint32_t OFFSET_MMA_BAR = OFFSET_H_TMA_BAR + 8U;
constexpr uint32_t OFFSET_ATTN_READY_BAR = OFFSET_MMA_BAR + 8U;
constexpr uint32_t OFFSET_QH_MMA_BAR = OFFSET_ATTN_READY_BAR + 8U;
constexpr uint32_t OFFSET_OV_MMA_BAR = OFFSET_QH_MMA_BAR + 8U;
constexpr uint32_t OFFSET_TMEM_ADDR = OFFSET_OV_MMA_BAR + 8U;
constexpr uint32_t SMEM_SIZE = (OFFSET_TMEM_ADDR + 4U + 1023U) & ~1023U;

//// Tensor Memory
constexpr uint32_t MAX_COLUMNS = 256U;
constexpr uint32_t OUTPUT_TMEM_COL = BLOCK_T;
constexpr uint32_t QH_TMEM_COL = 2U * BLOCK_T;

__device__ __forceinline__ uint32_t make_tile_layout_index(uint32_t tile_rows,
                                                           uint32_t row,
                                                           uint32_t col) {
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

static CUtensorMap encode_qk_tma(void *ptr, uint64_t num_tokens,
                                 uint64_t num_heads, uint64_t dim) {
  CUtensorMap tmap;
  constexpr uint32_t rank = 4;
  uint64_t globalDim[rank] = {SWIZZLE_WIDTH, num_tokens, dim / SWIZZLE_WIDTH,
                              num_heads};
  uint64_t globalStrides[rank - 1] = {
      num_heads * dim * sizeof(nv_bfloat16),
      SWIZZLE_WIDTH * sizeof(nv_bfloat16),
      dim * sizeof(nv_bfloat16),
  };
  uint32_t boxDim[rank] = {SWIZZLE_WIDTH, BLOCK_T,
                           static_cast<uint32_t>(dim / SWIZZLE_WIDTH), 1};
  uint32_t elementStrides[rank] = {1, 1, 1, 1};

  cuTensorMapEncodeTiled(
      &tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, ptr, globalDim,
      globalStrides, boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return tmap;
}

static CUtensorMap encode_v_tma(void *ptr, uint64_t num_tokens) {
  CUtensorMap tmap;
  constexpr uint32_t rank = 4;
  uint64_t globalDim[rank] = {SWIZZLE_WIDTH, num_tokens,
                              VALUE_DIM / SWIZZLE_WIDTH, NUM_OUTPUT_HEADS};
  uint64_t globalStrides[rank - 1] = {
      NUM_OUTPUT_HEADS * VALUE_DIM * sizeof(nv_bfloat16),
      SWIZZLE_WIDTH * sizeof(nv_bfloat16),
      VALUE_DIM * sizeof(nv_bfloat16),
  };
  uint32_t boxDim[rank] = {SWIZZLE_WIDTH, BLOCK_T, 1U, 1U};
  uint32_t elementStrides[rank] = {1, 1, 1, 1};

  cuTensorMapEncodeTiled(
      &tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, ptr, globalDim,
      globalStrides, boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return tmap;
}

__device__ __forceinline__ uint64_t make_tcgen05_desc_mmajor_v(
    uint32_t addr) {
  return desc_encode(addr) | (desc_encode(V_MMA_LBO) << 16ULL) |
         (desc_encode(V_MMA_SBO) << 32ULL) | (1ULL << 46ULL) |
         (2ULL << 61ULL);
}

template <int NUM_ATOMS>
__device__ __forceinline__ void mma_swizzled(uint32_t output_tmem,
                                             uint32_t matrix_a_smem,
                                             uint32_t matrix_b_smem) {
  constexpr uint32_t idesc = make_tcgen05_idesc(MMA_M, MMA_N);
  constexpr uint64_t desc_base =
      (desc_encode(SWIZZLE_SBO) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);

#pragma unroll
  for (uint32_t atom = 0; atom < static_cast<uint32_t>(NUM_ATOMS); ++atom) {
#pragma unroll
    for (uint32_t mi = 0; mi < NUM_MMA_STEPS; ++mi) {
      const uint64_t a_desc =
          desc_base |
          ((matrix_a_smem + atom * SWIZZLE_BYTES + mi * BYTES_ONE_MMA) >> 4);
      const uint64_t b_desc =
          desc_base |
          ((matrix_b_smem + atom * SWIZZLE_BYTES + mi * BYTES_ONE_MMA) >> 4);
      const uint32_t accum = (atom > 0U) || (mi > 0U);
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
  for (uint32_t ki = 0; ki < NUM_MMA_STEPS; ++ki) {
    const uint32_t a_base = matrix_a_smem + ki * matrix_a_k_stride_bytes;
    const uint32_t b_base = matrix_b_smem + ki * matrix_b_k_stride_bytes;
    tcgen05_mma(output_tmem,
                make_tcgen05_desc_noswizzle(a_base, BLOCK_T,
                                            BLOCK_T * sizeof(nv_bfloat16)),
                make_tcgen05_desc_noswizzle(b_base, BLOCK_V,
                                            BLOCK_T * sizeof(nv_bfloat16)),
                idesc, ki > 0U);
  }
}

__device__ __forceinline__ void mma_attn_v_mmajor_64x64(
    uint32_t output_tmem, uint32_t matrix_a_smem, uint32_t matrix_b_smem) {
  constexpr uint32_t idesc = make_tcgen05_idesc(MMA_M, MMA_N) | (1U << 16U);
  constexpr uint32_t matrix_a_k_stride_bytes =
      BLOCK_T * MMA_K * sizeof(nv_bfloat16);
#pragma unroll
  for (uint32_t ki = 0; ki < NUM_MMA_STEPS; ++ki) {
    const uint32_t a_base = matrix_a_smem + ki * matrix_a_k_stride_bytes;
    const uint32_t b_base = matrix_b_smem + ki * BYTES_ONE_MMA_MMAJOR;
    tcgen05_mma(output_tmem,
                make_tcgen05_desc_noswizzle(a_base, BLOCK_T,
                                            BLOCK_T * sizeof(nv_bfloat16)),
                make_tcgen05_desc_mmajor_v(b_base), idesc, ki > 0U);
  }
}

__device__ __forceinline__ void store_attn_column(
    nv_bfloat16 *attn_smem_ptr, const float *g_smem_ptr, const float *reg,
    uint32_t row_base, uint32_t row_hi, uint32_t row_base_limit,
    uint32_t row_hi_limit, uint32_t reg_row0, uint32_t reg_row1,
    uint32_t col) {
  float value_row0 = 0.0f;
  if (col < row_base_limit) {
    value_row0 = reg[reg_row0] * __expf(g_smem_ptr[row_base] - g_smem_ptr[col]);
  }

  float value_row1 = 0.0f;
  if (col < row_hi_limit) {
    value_row1 = reg[reg_row1] * __expf(g_smem_ptr[row_hi] - g_smem_ptr[col]);
  }

  attn_smem_ptr[make_tile_layout_index(BLOCK_T, row_base, col)] =
      __float2bfloat16_rn(value_row0);
  attn_smem_ptr[make_tile_layout_index(BLOCK_T, row_hi, col)] =
      __float2bfloat16_rn(value_row1);
}

__global__ __block_size__((NUM_THREADS, 1, 1)) void o_v1_kernel_swizzled(
    const __grid_constant__ CUtensorMap q_tmap,
    const __grid_constant__ CUtensorMap k_tmap,
    const __grid_constant__ CUtensorMap v_tmap,
    const __grid_constant__ CUtensorMap h_tmap, const float *g_cu_ptr,
    nv_bfloat16 *o_ptr, const int64_t *cu_seqlens_ptr,
    const int32_t *chunk_indices_ptr, int32_t total_num_chunks, float scale) {
  const uint32_t tid = threadIdx.x;
  const uint32_t warp_id = tid / WARP_SIZE;
  const uint32_t lane_id = tid % WARP_SIZE;

  const uint32_t v_tile = blockIdx.x;
  const uint32_t global_chunk_id = blockIdx.y;
  const uint32_t head_id = blockIdx.z;
  if (global_chunk_id >= static_cast<uint32_t>(total_num_chunks)) {
    return;
  }

  const uint32_t q_head_id = head_id / HEADS_PER_QK_HEAD;
  const uint32_t k_head_id = head_id / HEADS_PER_QK_HEAD;

  const int2 chunk_meta =
      reinterpret_cast<const int2 *>(chunk_indices_ptr)[global_chunk_id];
  const uint32_t seq_id = static_cast<uint32_t>(chunk_meta.x);
  const uint32_t chunk_id = static_cast<uint32_t>(chunk_meta.y);
  const int64_t bos = cu_seqlens_ptr[seq_id];
  const int64_t eos = cu_seqlens_ptr[seq_id + 1];
  const int64_t chunk_start = bos + static_cast<int64_t>(chunk_id) * BLOCK_T;
  const int64_t remaining = eos - chunk_start;
  const uint32_t chunk_len =
      remaining <= 0
          ? 0U
          : static_cast<uint32_t>(remaining < static_cast<int64_t>(BLOCK_T)
                                      ? remaining
                                      : static_cast<int64_t>(BLOCK_T));
  const uint32_t v_start = v_tile * BLOCK_V;

  extern __shared__ __align__(1024) char smem_ptr[];
  const uint32_t smem = __cvta_generic_to_shared(smem_ptr);

  const uint32_t q_smem = smem + OFFSET_Q;
  const uint32_t k_smem = smem + OFFSET_K;
  const uint32_t v_smem = smem + OFFSET_V;
  const uint32_t h_smem = smem + OFFSET_H;
  const uint32_t attn_smem = smem + OFFSET_ATTN;
  const uint32_t output_smem = smem + OFFSET_OUTPUT;
  const uint32_t g_smem = smem + OFFSET_G;
  const uint32_t qk_tma_barrier = smem + OFFSET_QK_TMA_BAR;
  const uint32_t v_tma_barrier = smem + OFFSET_V_TMA_BAR;
  const uint32_t h_tma_barrier = smem + OFFSET_H_TMA_BAR;
  const uint32_t mma_barrier = smem + OFFSET_MMA_BAR;
  const uint32_t attn_ready_barrier = smem + OFFSET_ATTN_READY_BAR;
  const uint32_t qh_mma_barrier = smem + OFFSET_QH_MMA_BAR;
  const uint32_t ov_mma_barrier = smem + OFFSET_OV_MMA_BAR;
  const uint32_t tmem_alloc_smem = smem + OFFSET_TMEM_ADDR;

  float *output_smem_ptr =
      reinterpret_cast<float *>(smem_ptr + (output_smem - smem));
  float *g_smem_ptr = reinterpret_cast<float *>(smem_ptr + (g_smem - smem));

  if (warp_id == TMA_WARP) {
    if (elect_sync()) {
      mbarrier_init(qk_tma_barrier, 1);
      mbarrier_init(v_tma_barrier, 1);
      mbarrier_init(h_tma_barrier, 1);
      mbarrier_init(mma_barrier, 1);
      mbarrier_init(attn_ready_barrier, NUM_CUDA_WARPS);
      mbarrier_init(qh_mma_barrier, 1);
      mbarrier_init(ov_mma_barrier, 1);
      prefetch_tensormap(&q_tmap);
      prefetch_tensormap(&k_tmap);
      prefetch_tensormap(&v_tmap);
      prefetch_tensormap(&h_tmap);
      fence_mbarrier_init();
    }
  } else if (warp_id == MMA_WARP) {
    tcgen05_alloc(tmem_alloc_smem, MAX_COLUMNS);
  }

  __syncthreads();

  constexpr uint32_t TMA_PHASE = 0U;
  constexpr uint32_t QK_MMA_PHASE = 0U;
  constexpr uint32_t OV_MMA_PHASE = 0U;
  constexpr uint32_t QH_MMA_PHASE = 0U;
  constexpr uint32_t ATTN_READY_PHASE = 0U;
  const uint32_t h_outer = global_chunk_id * NUM_OUTPUT_HEADS + head_id;
  const int32_t chunk_start_i32 = static_cast<int32_t>(chunk_start);

  if (warp_id == TMA_WARP && elect_sync()) {
    tma_load_4d(q_smem, &q_tmap, 0, chunk_start_i32, 0, q_head_id,
                qk_tma_barrier);
    tma_load_4d(k_smem, &k_tmap, 0, chunk_start_i32, 0, k_head_id,
                qk_tma_barrier);
    tma_load_4d(v_smem, &v_tmap, 0, chunk_start_i32, v_tile, head_id,
                v_tma_barrier);
    tma_load_4d(h_smem, &h_tmap, 0, v_start, 0, h_outer, h_tma_barrier);
    mbarrier_arrive_expect_tx(qk_tma_barrier, Q_SMEM_SIZE + K_SMEM_SIZE);
    mbarrier_arrive_expect_tx(v_tma_barrier, V_SMEM_SIZE);
    mbarrier_arrive_expect_tx(h_tma_barrier, H_SMEM_SIZE);
  }
  if (warp_id == MMA_WARP && elect_sync()) {
    mbarrier_wait(qk_tma_barrier, TMA_PHASE);
    tcgen05_fence_after_thread_sync();
    mma_swizzled<NUM_SWIZZLE_ATOMS>(0, q_smem, k_smem);
    tcgen05_commit(mma_barrier);
    mbarrier_wait(h_tma_barrier, TMA_PHASE);
    tcgen05_fence_after_thread_sync();
    mma_swizzled<NUM_SWIZZLE_ATOMS>(QH_TMEM_COL, q_smem, h_smem);
    tcgen05_commit(qh_mma_barrier);
    mbarrier_wait(v_tma_barrier, TMA_PHASE);
    mbarrier_wait(attn_ready_barrier, ATTN_READY_PHASE);
    tcgen05_fence_after_thread_sync();
    mma_attn_v_mmajor_64x64(OUTPUT_TMEM_COL, attn_smem, v_smem);
    tcgen05_commit(ov_mma_barrier);
  }

  if (warp_id < NUM_CUDA_WARPS) {
    for (uint32_t i = tid; i < BLOCK_T; i += NUM_CUDA_WARPS * WARP_SIZE) {
      float g_value = 0.0f;
      if (i < chunk_len) {
        const int64_t token_idx = chunk_start + static_cast<int64_t>(i);
        g_value = g_cu_ptr[token_idx * NUM_OUTPUT_HEADS + head_id];
      }
      g_smem_ptr[i] = g_value;
    }
    bar_sync<1>(NUM_CUDA_WARPS * WARP_SIZE);
  }

  nv_bfloat16 *attn_smem_ptr =
      reinterpret_cast<nv_bfloat16 *>(smem_ptr + (attn_smem - smem));

  if (warp_id < NUM_CUDA_WARPS) {
    mbarrier_wait(mma_barrier, QK_MMA_PHASE);
    tcgen05_fence_after_thread_sync();
    float reg_lo[REGS_PER_FRAGMENT];
    float reg_hi[REGS_PER_FRAGMENT];
    tcgen05_ld<SHAPE::_16x256b, 4>(reg_lo, 0, 0);
    tcgen05_ld<SHAPE::_16x256b, 4>(reg_hi, 0, COLS_PER_FRAGMENT);
    tcgen05_wait_ld();
    const uint32_t row_base =
        warp_id * ROWS_PER_WARP + lane_id / LANES_PER_ROW_GROUP;
    const uint32_t row_hi = row_base + ROW_PAIR_STRIDE;
    const uint32_t row_base_limit = row_base < chunk_len ? row_base + 1U : 0U;
    const uint32_t row_hi_limit = row_hi < chunk_len ? row_hi + 1U : 0U;
    const uint32_t lane_col = lane_id % LANES_PER_ROW_GROUP;
#pragma unroll
    for (uint32_t step = 0; step < FRAGMENT_STEPS; ++step) {
      const uint32_t col_in_fragment =
          (step / 2U) * 8U + 2U * lane_col + (step % 2U);
      const uint32_t col_lo = col_in_fragment;
      const uint32_t col_hi = col_in_fragment + COLS_PER_FRAGMENT;
      const uint32_t reg_row0 = (step / 2U) * 4U + (step % 2U);
      const uint32_t reg_row1 = reg_row0 + 2U;

      store_attn_column(attn_smem_ptr, g_smem_ptr, reg_lo, row_base, row_hi,
                        row_base_limit, row_hi_limit, reg_row0, reg_row1,
                        col_lo);
      store_attn_column(attn_smem_ptr, g_smem_ptr, reg_hi, row_base, row_hi,
                        row_base_limit, row_hi_limit, reg_row0, reg_row1,
                        col_hi);
    }
    fence_proxy_async_shared_cta();
    __syncwarp();
    if (elect_sync()) {
      mbarrier_arrive(attn_ready_barrier);
    }
  }

  if (warp_id < NUM_CUDA_WARPS) {
    mbarrier_wait(ov_mma_barrier, OV_MMA_PHASE);
    tcgen05_fence_after_thread_sync();
    float reg_lo[REGS_PER_FRAGMENT];
    float reg_hi[REGS_PER_FRAGMENT];
    tcgen05_ld<SHAPE::_16x256b, 4>(reg_lo, 0, OUTPUT_TMEM_COL);
    tcgen05_ld<SHAPE::_16x256b, 4>(reg_hi, 0,
                                   OUTPUT_TMEM_COL + COLS_PER_FRAGMENT);
    tcgen05_wait_ld();
    const uint32_t row_base =
        warp_id * ROWS_PER_WARP + lane_id / LANES_PER_ROW_GROUP;
    const uint32_t row_hi = row_base + ROW_PAIR_STRIDE;
    const uint32_t lane_col = lane_id % LANES_PER_ROW_GROUP;
    float *row_base_output_ptr = output_smem_ptr + row_base * BLOCK_V;
    float *row_hi_output_ptr = output_smem_ptr + row_hi * BLOCK_V;
#pragma unroll
    for (uint32_t step_pair = 0; step_pair < FRAGMENT_PAIRS; ++step_pair) {
      const uint32_t col_lo = step_pair * 8U + 2U * lane_col;
      const uint32_t col_hi = col_lo + COLS_PER_FRAGMENT;
      const uint32_t reg_row0 = step_pair * 4U;
      const uint32_t reg_row1 = reg_row0 + 2U;

      reinterpret_cast<float2 *>(row_base_output_ptr + col_lo)[0] =
          make_float2(reg_lo[reg_row0], reg_lo[reg_row0 + 1U]);
      reinterpret_cast<float2 *>(row_hi_output_ptr + col_lo)[0] =
          make_float2(reg_lo[reg_row1], reg_lo[reg_row1 + 1U]);
      reinterpret_cast<float2 *>(row_base_output_ptr + col_hi)[0] =
          make_float2(reg_hi[reg_row0], reg_hi[reg_row0 + 1U]);
      reinterpret_cast<float2 *>(row_hi_output_ptr + col_hi)[0] =
          make_float2(reg_hi[reg_row1], reg_hi[reg_row1 + 1U]);
    }
  }

  if (warp_id < NUM_CUDA_WARPS) {
    mbarrier_wait(qh_mma_barrier, QH_MMA_PHASE);
    tcgen05_fence_after_thread_sync();
    float reg_lo[REGS_PER_FRAGMENT];
    float reg_hi[REGS_PER_FRAGMENT];
    tcgen05_ld<SHAPE::_16x256b, 4>(reg_lo, 0, QH_TMEM_COL);
    tcgen05_ld<SHAPE::_16x256b, 4>(reg_hi, 0,
                                   QH_TMEM_COL + COLS_PER_FRAGMENT);
    tcgen05_wait_ld();
    const uint32_t row_base =
        warp_id * ROWS_PER_WARP + lane_id / LANES_PER_ROW_GROUP;
    const uint32_t row_hi = row_base + ROW_PAIR_STRIDE;
    const bool row_base_active = row_base < chunk_len;
    const bool row_hi_active = row_hi < chunk_len;
    const float g_row_base =
        row_base_active ? __expf(g_smem_ptr[row_base]) : 0.0f;
    const float g_row_hi = row_hi_active ? __expf(g_smem_ptr[row_hi]) : 0.0f;
    const uint32_t lane_col = lane_id % LANES_PER_ROW_GROUP;
    nv_bfloat16 *row_base_o_ptr = nullptr;
    nv_bfloat16 *row_hi_o_ptr = nullptr;
    if (row_base_active) {
      row_base_o_ptr =
          o_ptr +
          (((chunk_start + static_cast<int64_t>(row_base)) * NUM_OUTPUT_HEADS +
            head_id) *
               VALUE_DIM +
           v_start);
    }
    if (row_hi_active) {
      row_hi_o_ptr =
          o_ptr +
          (((chunk_start + static_cast<int64_t>(row_hi)) * NUM_OUTPUT_HEADS +
            head_id) *
               VALUE_DIM +
           v_start);
    }
#pragma unroll
    for (uint32_t step_pair = 0; step_pair < FRAGMENT_PAIRS; ++step_pair) {
      const uint32_t col_lo = step_pair * 8U + 2U * lane_col;
      const uint32_t col_hi = col_lo + COLS_PER_FRAGMENT;
      const uint32_t reg_row0 = step_pair * 4U;
      const uint32_t reg_row1 = reg_row0 + 2U;

      if (row_base_active) {
        const float value_lo_0 =
            scale * (output_smem_ptr[row_base * BLOCK_V + col_lo] +
                     reg_lo[reg_row0] * g_row_base);
        const float value_lo_1 =
            scale * (output_smem_ptr[row_base * BLOCK_V + col_lo + 1U] +
                     reg_lo[reg_row0 + 1U] * g_row_base);
        const float value_hi_0 =
            scale * (output_smem_ptr[row_base * BLOCK_V + col_hi] +
                     reg_hi[reg_row0] * g_row_base);
        const float value_hi_1 =
            scale * (output_smem_ptr[row_base * BLOCK_V + col_hi + 1U] +
                     reg_hi[reg_row0 + 1U] * g_row_base);
        reinterpret_cast<__nv_bfloat162 *>(row_base_o_ptr + col_lo)[0] =
            __floats2bfloat162_rn(value_lo_0, value_lo_1);
        reinterpret_cast<__nv_bfloat162 *>(row_base_o_ptr + col_hi)[0] =
            __floats2bfloat162_rn(value_hi_0, value_hi_1);
      }

      if (row_hi_active) {
        const float value_lo_0 =
            scale * (output_smem_ptr[row_hi * BLOCK_V + col_lo] +
                     reg_lo[reg_row1] * g_row_hi);
        const float value_lo_1 =
            scale * (output_smem_ptr[row_hi * BLOCK_V + col_lo + 1U] +
                     reg_lo[reg_row1 + 1U] * g_row_hi);
        const float value_hi_0 =
            scale * (output_smem_ptr[row_hi * BLOCK_V + col_hi] +
                     reg_hi[reg_row1] * g_row_hi);
        const float value_hi_1 =
            scale * (output_smem_ptr[row_hi * BLOCK_V + col_hi + 1U] +
                     reg_hi[reg_row1 + 1U] * g_row_hi);
        reinterpret_cast<__nv_bfloat162 *>(row_hi_o_ptr + col_lo)[0] =
            __floats2bfloat162_rn(value_lo_0, value_lo_1);
        reinterpret_cast<__nv_bfloat162 *>(row_hi_o_ptr + col_hi)[0] =
            __floats2bfloat162_rn(value_hi_0, value_hi_1);
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

void o_v1(TensorView q_chunks, TensorView k_chunks, TensorView v_new,
          TensorView h, TensorView g_cu, TensorView o, TensorView cu_seqlens,
          TensorView chunk_indices, int total_num_chunks, double scale) {
  const uint32_t total_num_tokens = static_cast<uint32_t>(q_chunks.size(0));
  const uint32_t total_num_chunks_u = static_cast<uint32_t>(total_num_chunks);
  const uint64_t total_num_output_tiles =
      static_cast<uint64_t>(total_num_chunks_u) * NUM_OUTPUT_HEADS;

  auto q_tmap = encode_qk_tma(q_chunks.data_ptr(), total_num_tokens,
                              NUM_QK_HEADS, HEAD_DIM);
  auto k_tmap = encode_qk_tma(k_chunks.data_ptr(), total_num_tokens,
                              NUM_QK_HEADS, HEAD_DIM);
  auto v_tmap = encode_v_tma(v_new.data_ptr(), total_num_tokens);
  auto h_tmap =
      encode_tma(h.data_ptr(), total_num_output_tiles, VALUE_DIM, HEAD_DIM);

  auto *g_cu_ptr = reinterpret_cast<const float *>(g_cu.data_ptr());
  auto *o_ptr = reinterpret_cast<nv_bfloat16 *>(o.data_ptr());
  auto *cu_seqlens_ptr =
      reinterpret_cast<const int64_t *>(cu_seqlens.data_ptr());
  auto *chunk_indices_ptr =
      reinterpret_cast<const int32_t *>(chunk_indices.data_ptr());

  auto kernel = o_v1_kernel_swizzled;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       SMEM_SIZE);

  dim3 grid(VALUE_DIM / BLOCK_V, total_num_chunks_u, NUM_OUTPUT_HEADS);
  kernel<<<grid, NUM_THREADS, SMEM_SIZE>>>(
      q_tmap, k_tmap, v_tmap, h_tmap, g_cu_ptr, o_ptr, cu_seqlens_ptr,
      chunk_indices_ptr, total_num_chunks, static_cast<float>(scale));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(o_v1, o_v1);
