#include "cuda_utils.h"

#include <cstdint>
#include <cuda_bf16.h>
#include <math.h>

//// Tile Config
constexpr uint32_t BLOCK_T = 64U;
constexpr uint32_t HEAD_DIM = 128U;
constexpr uint32_t VALUE_DIM = 128U;
constexpr uint32_t BLOCK_V = 128U;

//// Head Config
constexpr uint32_t NUM_OUTPUT_HEADS = 8U;
constexpr uint32_t NUM_QK_HEADS = 4U;
constexpr uint32_t HEADS_PER_QK_HEAD = NUM_OUTPUT_HEADS / NUM_QK_HEADS;
static_assert(HEADS_PER_QK_HEAD == 2U);

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
constexpr uint32_t CUDA_MMA_THREAD_COUNT = (NUM_CUDA_WARPS + 1U) * WARP_SIZE;
constexpr uint32_t ROWS_PER_WARP = 16U;
constexpr uint32_t COLS_PER_FRAGMENT = 32U;
constexpr uint32_t REGS_PER_FRAGMENT = 16U;
constexpr uint32_t ROW_PAIR_STRIDE = 8U;
constexpr uint32_t ROW_PAIR_OUTPUT_STRIDE =
    ROW_PAIR_STRIDE * NUM_OUTPUT_HEADS * VALUE_DIM;
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
constexpr uint32_t H_SWIZZLE_BYTES =
    BLOCK_V * SWIZZLE_WIDTH * sizeof(nv_bfloat16);
constexpr uint32_t SWIZZLE_SBO = 8U * 128U;
constexpr uint32_t V_MMA_SBO = SWIZZLE_WIDTH * 8U * sizeof(nv_bfloat16);
constexpr uint32_t V_MMA_LBO = (BLOCK_T / 8U) * V_MMA_SBO;
constexpr uint32_t BYTES_ONE_MMA_MMAJOR =
    MMA_K * SWIZZLE_WIDTH * sizeof(nv_bfloat16);
constexpr uint32_t QK_STAGE_COUNT = 2U;
constexpr uint32_t VH_STAGE_COUNT = HEADS_PER_QK_HEAD;
constexpr uint32_t ATTN_STAGE_COUNT = HEADS_PER_QK_HEAD;
constexpr uint32_t HEAD0_STAGE_INDEX = 0U;
constexpr uint32_t HEAD1_STAGE_INDEX = 1U;

//// Shared Memory Offsets in Bytes
constexpr uint32_t Q_SMEM_SIZE = BLOCK_T * HEAD_DIM * sizeof(nv_bfloat16);
constexpr uint32_t K_SMEM_SIZE = BLOCK_T * HEAD_DIM * sizeof(nv_bfloat16);
constexpr uint32_t Q_SMEM_BYTES = QK_STAGE_COUNT * Q_SMEM_SIZE;
constexpr uint32_t K_SMEM_BYTES = QK_STAGE_COUNT * K_SMEM_SIZE;
constexpr uint32_t V_SMEM_SIZE = BLOCK_V * BLOCK_T * sizeof(nv_bfloat16);
constexpr uint32_t H_SMEM_SIZE = BLOCK_V * HEAD_DIM * sizeof(nv_bfloat16);
constexpr uint32_t V_SMEM_BYTES = VH_STAGE_COUNT * V_SMEM_SIZE;
constexpr uint32_t H_SMEM_BYTES = VH_STAGE_COUNT * H_SMEM_SIZE;
constexpr uint32_t ATTN_SMEM_SIZE = BLOCK_T * BLOCK_T * sizeof(nv_bfloat16);
constexpr uint32_t ATTN_SMEM_BYTES = ATTN_STAGE_COUNT * ATTN_SMEM_SIZE;
constexpr uint32_t OUTPUT_SMEM_SIZE = 0U;
constexpr uint32_t QK_SPILL_SMEM_SIZE = 0U;
constexpr uint32_t G_STAGE_SMEM_SIZE = BLOCK_T * sizeof(float);
constexpr uint32_t G_RAW_SMEM_SIZE = HEADS_PER_QK_HEAD * G_STAGE_SMEM_SIZE;
constexpr uint32_t G_CENTER_SMEM_SIZE = HEADS_PER_QK_HEAD * sizeof(float);
constexpr uint32_t G_POS_SMEM_SIZE = HEADS_PER_QK_HEAD * G_STAGE_SMEM_SIZE;
constexpr uint32_t G_ALPHA_SMEM_SIZE = HEADS_PER_QK_HEAD * sizeof(float);
constexpr uint32_t G_NEG_SMEM_SIZE = HEADS_PER_QK_HEAD * G_STAGE_SMEM_SIZE;
constexpr uint32_t G_REDUCE_PARTIAL_SMEM_SIZE = NUM_CUDA_WARPS * sizeof(float);
constexpr uint32_t QK_TMA_BARRIER_BYTES = QK_STAGE_COUNT * 8U;
constexpr uint32_t VH_TMA_BARRIER_BYTES = VH_STAGE_COUNT * 8U;
constexpr uint32_t ATTN_READY_BARRIER_BYTES = ATTN_STAGE_COUNT * 8U;
constexpr uint32_t OFFSET_Q = 0U;
constexpr uint32_t OFFSET_K = OFFSET_Q + Q_SMEM_BYTES;
constexpr uint32_t OFFSET_V = OFFSET_K + K_SMEM_BYTES;
constexpr uint32_t OFFSET_H = OFFSET_V + V_SMEM_BYTES;
constexpr uint32_t OFFSET_ATTN = OFFSET_H + H_SMEM_BYTES;
constexpr uint32_t OFFSET_OUTPUT = OFFSET_ATTN + ATTN_SMEM_BYTES;
constexpr uint32_t OFFSET_QK_SPILL = OFFSET_OUTPUT + OUTPUT_SMEM_SIZE;
constexpr uint32_t OFFSET_G_RAW = OFFSET_QK_SPILL + QK_SPILL_SMEM_SIZE;
constexpr uint32_t OFFSET_G_CENTER = OFFSET_G_RAW + G_RAW_SMEM_SIZE;
constexpr uint32_t OFFSET_G_POS = OFFSET_G_CENTER + G_CENTER_SMEM_SIZE;
constexpr uint32_t OFFSET_G_ALPHA = OFFSET_G_POS + G_POS_SMEM_SIZE;
constexpr uint32_t OFFSET_G_NEG = OFFSET_G_ALPHA + G_ALPHA_SMEM_SIZE;
constexpr uint32_t OFFSET_G_REDUCE_MIN = OFFSET_G_NEG + G_NEG_SMEM_SIZE;
constexpr uint32_t OFFSET_G_REDUCE_MAX =
    OFFSET_G_REDUCE_MIN + G_REDUCE_PARTIAL_SMEM_SIZE;
constexpr uint32_t OFFSET_QK_TMA_BAR =
    OFFSET_G_REDUCE_MAX + G_REDUCE_PARTIAL_SMEM_SIZE;
constexpr uint32_t OFFSET_V_TMA_BAR = OFFSET_QK_TMA_BAR + QK_TMA_BARRIER_BYTES;
constexpr uint32_t OFFSET_H_TMA_BAR = OFFSET_V_TMA_BAR + VH_TMA_BARRIER_BYTES;
constexpr uint32_t OFFSET_MMA_BAR = OFFSET_H_TMA_BAR + VH_TMA_BARRIER_BYTES;
constexpr uint32_t OFFSET_ATTN_READY_BAR = OFFSET_MMA_BAR + 8U;
constexpr uint32_t OFFSET_HEAD0_RELEASE_BAR =
    OFFSET_ATTN_READY_BAR + ATTN_READY_BARRIER_BYTES;
constexpr uint32_t OFFSET_QH_MMA_BAR = OFFSET_HEAD0_RELEASE_BAR + 8U;
constexpr uint32_t OFFSET_OV_MMA_BAR = OFFSET_QH_MMA_BAR + 8U;
constexpr uint32_t OFFSET_TMEM_ADDR = OFFSET_OV_MMA_BAR + 8U;
constexpr uint32_t SMEM_SIZE = (OFFSET_TMEM_ADDR + 4U + 1023U) & ~1023U;

//// Tensor Memory
constexpr uint32_t MAX_COLUMNS = 512U;
constexpr uint32_t OUTPUT_TMEM_COL = BLOCK_T;
constexpr uint32_t QH_TMEM_COL = 3U * BLOCK_T;
constexpr CUtensorMapL2promotion TMA_L2_PROMOTION =
    CU_TENSOR_MAP_L2_PROMOTION_L2_256B;

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
  uint32_t boxDim[rank] = {SWIZZLE_WIDTH, BLOCK_V,
                           static_cast<uint32_t>(cols / SWIZZLE_WIDTH), 1};
  uint32_t elementStrides[rank] = {1, 1, 1, 1};

  cuTensorMapEncodeTiled(&tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, ptr,
                         globalDim, globalStrides, boxDim, elementStrides,
                         CU_TENSOR_MAP_INTERLEAVE_NONE,
                         CU_TENSOR_MAP_SWIZZLE_128B, TMA_L2_PROMOTION,
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

  cuTensorMapEncodeTiled(&tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, ptr,
                         globalDim, globalStrides, boxDim, elementStrides,
                         CU_TENSOR_MAP_INTERLEAVE_NONE,
                         CU_TENSOR_MAP_SWIZZLE_128B, TMA_L2_PROMOTION,
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
  uint32_t boxDim[rank] = {SWIZZLE_WIDTH, BLOCK_T, BLOCK_V / SWIZZLE_WIDTH, 1U};
  uint32_t elementStrides[rank] = {1, 1, 1, 1};

  cuTensorMapEncodeTiled(&tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, ptr,
                         globalDim, globalStrides, boxDim, elementStrides,
                         CU_TENSOR_MAP_INTERLEAVE_NONE,
                         CU_TENSOR_MAP_SWIZZLE_128B, TMA_L2_PROMOTION,
                         CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return tmap;
}

__device__ __forceinline__ uint64_t make_tcgen05_desc_mmajor_v(uint32_t addr) {
  return desc_encode(addr) | (desc_encode(V_MMA_LBO) << 16ULL) |
         (desc_encode(V_MMA_SBO) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
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

__device__ __forceinline__ void mma_swizzled_qh_64x128(uint32_t output_tmem,
                                                       uint32_t matrix_a_smem,
                                                       uint32_t matrix_b_smem) {
  constexpr uint32_t idesc = make_tcgen05_idesc(MMA_M, BLOCK_V);
  constexpr uint64_t desc_base =
      (desc_encode(SWIZZLE_SBO) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);

#pragma unroll
  for (uint32_t atom = 0; atom < static_cast<uint32_t>(NUM_SWIZZLE_ATOMS);
       ++atom) {
#pragma unroll
    for (uint32_t mi = 0; mi < NUM_MMA_STEPS; ++mi) {
      const uint64_t a_desc =
          desc_base |
          ((matrix_a_smem + atom * SWIZZLE_BYTES + mi * BYTES_ONE_MMA) >> 4);
      const uint64_t b_desc =
          desc_base |
          ((matrix_b_smem + atom * H_SWIZZLE_BYTES + mi * BYTES_ONE_MMA) >> 4);
      const uint32_t accum = (atom > 0U) || (mi > 0U);
      tcgen05_mma(output_tmem, a_desc, b_desc, idesc, accum);
    }
  }
}

__device__ __forceinline__ void
mma_attn_v_mmajor_64x128(uint32_t output_tmem, uint32_t matrix_a_smem,
                         uint32_t matrix_b_smem) {
  constexpr uint32_t idesc = make_tcgen05_idesc(MMA_M, BLOCK_V) | (1U << 16U);
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

__device__ __forceinline__ float scale_qk_if_active(float qk, float g_col,
                                                    uint32_t active) {
  if (active == 0U) {
    return 0.0f;
  }
  return qk * g_col;
}

__device__ __forceinline__ void store_bf162_shared(nv_bfloat16 *ptr,
                                                   __nv_bfloat162 value) {
  const uint32_t data = reinterpret_cast<const uint32_t &>(value);
  asm volatile("st.shared.b32 [%0], %1;" :: "r"(cvt_smem_ptr(ptr)), "r"(data)
               : "memory");
}

__device__ __forceinline__ void
store_attn_column_pair(nv_bfloat16 *attn_smem_ptr, const float *g_neg_smem_ptr,
                       float qk_row0_col0, float qk_row0_col1,
                       float qk_row1_col0, float qk_row1_col1,
                       uint32_t row_base, uint32_t row_hi,
                       uint32_t row_base_limit, uint32_t row_hi_limit,
                       uint32_t col) {
  const uint32_t col_hi = col + 1U;
  const float g_col0 = g_neg_smem_ptr[col];
  const float g_col1 = g_neg_smem_ptr[col_hi];
  const uint32_t row_base_idx = make_tile_layout_index(BLOCK_T, row_base, col);
  const uint32_t row_hi_idx = make_tile_layout_index(BLOCK_T, row_hi, col);

  store_bf162_shared(
      attn_smem_ptr + row_base_idx,
      __floats2bfloat162_rn(
          scale_qk_if_active(qk_row0_col0, g_col0, col < row_base_limit),
          scale_qk_if_active(qk_row0_col1, g_col1, col_hi < row_base_limit)));
  store_bf162_shared(
      attn_smem_ptr + row_hi_idx,
      __floats2bfloat162_rn(
          scale_qk_if_active(qk_row1_col0, g_col0, col < row_hi_limit),
          scale_qk_if_active(qk_row1_col1, g_col1, col_hi < row_hi_limit)));
}

__device__ __forceinline__ void load_qk_fragment(float *qk_reg_lo,
                                                 float *qk_reg_hi) {
  tcgen05_ld<SHAPE::_16x256b, 4>(qk_reg_lo, 0, 0);
  tcgen05_ld<SHAPE::_16x256b, 4>(qk_reg_hi, 0, COLS_PER_FRAGMENT);
  tcgen05_wait_ld();
}

__device__ __forceinline__ float warp_reduce_min(float value) {
#pragma unroll
  for (uint32_t offset = WARP_SIZE / 2U; offset > 0U; offset >>= 1U) {
    value = fminf(value, __shfl_down_sync(0xffffffffU, value, offset));
  }
  return value;
}

__device__ __forceinline__ float warp_reduce_max(float value) {
#pragma unroll
  for (uint32_t offset = WARP_SIZE / 2U; offset > 0U; offset >>= 1U) {
    value = fmaxf(value, __shfl_down_sync(0xffffffffU, value, offset));
  }
  return value;
}

__device__ __forceinline__ void reduce_g_center_per_head(
    const float *g_raw_smem_ptr, float *g_center_smem_ptr,
    float *g_reduce_min_smem_ptr, float *g_reduce_max_smem_ptr,
    uint32_t tid, uint32_t warp_id, uint32_t lane_id, uint32_t chunk_len) {
  const uint32_t head_offset = warp_id >> 1U;
  const uint32_t token_offset = ((warp_id & 1U) * WARP_SIZE) + lane_id;
  float g_min = INFINITY;
  float g_max = -INFINITY;
  if (token_offset < chunk_len) {
    const float g_value = g_raw_smem_ptr[head_offset * BLOCK_T + token_offset];
    g_min = g_value;
    g_max = g_value;
  }

  g_min = warp_reduce_min(g_min);
  g_max = warp_reduce_max(g_max);
  if (lane_id == 0U) {
    g_reduce_min_smem_ptr[warp_id] = g_min;
    g_reduce_max_smem_ptr[warp_id] = g_max;
  }
  bar_sync<1>(NUM_CUDA_WARPS * WARP_SIZE);

  if (tid < HEADS_PER_QK_HEAD) {
    float g_center = 0.0f;
    if (chunk_len > 0U) {
      const uint32_t warp_base = tid * 2U;
      const float head_min =
          fminf(g_reduce_min_smem_ptr[warp_base], g_reduce_min_smem_ptr[warp_base + 1U]);
      const float head_max =
          fmaxf(g_reduce_max_smem_ptr[warp_base], g_reduce_max_smem_ptr[warp_base + 1U]);
      g_center = 0.5f * (head_min + head_max);
    }
    g_center_smem_ptr[tid] = g_center;
  }
}

__device__ __forceinline__ void materialize_attn_stage_from_regs(
    char *smem_ptr, uint32_t smem, uint32_t attn_stage_smem,
    const float *qk_reg_lo, const float *qk_reg_hi, const float *g_neg_head_ptr,
    uint32_t row_base, uint32_t row_hi, uint32_t row_base_limit,
    uint32_t row_hi_limit, uint32_t lane_col) {
  nv_bfloat16 *attn_smem_ptr =
      reinterpret_cast<nv_bfloat16 *>(smem_ptr + (attn_stage_smem - smem));
#pragma unroll
  for (uint32_t step_pair = 0; step_pair < FRAGMENT_PAIRS; ++step_pair) {
    const uint32_t col_lo = step_pair * 8U + 2U * lane_col;
    const uint32_t col_hi = col_lo + COLS_PER_FRAGMENT;
    const uint32_t reg_base = step_pair * 4U;

    store_attn_column_pair(attn_smem_ptr, g_neg_head_ptr, qk_reg_lo[reg_base],
                           qk_reg_lo[reg_base + 1U], qk_reg_lo[reg_base + 2U],
                           qk_reg_lo[reg_base + 3U], row_base, row_hi,
                           row_base_limit, row_hi_limit, col_lo);
    store_attn_column_pair(attn_smem_ptr, g_neg_head_ptr, qk_reg_hi[reg_base],
                           qk_reg_hi[reg_base + 1U], qk_reg_hi[reg_base + 2U],
                           qk_reg_hi[reg_base + 3U], row_base, row_hi,
                           row_base_limit, row_hi_limit, col_hi);
  }
}

__device__ __forceinline__ __nv_bfloat162 combine_output_pair(
    float ov_0, float ov_1, float qh_0, float qh_1, float gp_row,
    float alpha, float scale) {
  const float row_scale = scale * gp_row;
  return __floats2bfloat162_rn(row_scale * __fmaf_rn(alpha, qh_0, ov_0),
                               row_scale * __fmaf_rn(alpha, qh_1, ov_1));
}

__device__ __forceinline__ void
tma_load_qk_stage(uint32_t q_stage_smem, uint32_t k_stage_smem,
                  const CUtensorMap *q_tmap, const CUtensorMap *k_tmap,
                  int32_t chunk_start_i32, uint32_t q_head_id,
                  uint32_t k_head_id, uint32_t qk_tma_barrier) {
  tma_load_4d(q_stage_smem, q_tmap, 0, chunk_start_i32, 0, q_head_id,
              qk_tma_barrier);
  tma_load_4d(k_stage_smem, k_tmap, 0, chunk_start_i32, 0, k_head_id,
              qk_tma_barrier);
  mbarrier_arrive_expect_tx(qk_tma_barrier, Q_SMEM_SIZE + K_SMEM_SIZE);
}

__device__ __forceinline__ void store_bf162_no_allocate(nv_bfloat16 *ptr,
                                                        __nv_bfloat162 value) {
  const uint32_t data = reinterpret_cast<const uint32_t &>(value);
  asm volatile("st.global.relaxed.cta.L1::no_allocate.u32 [%0], %1;" ::"l"(ptr),
               "r"(data));
}

__device__ __forceinline__ void
store_bf162_pair_no_allocate_if(nv_bfloat16 *ptr_0, __nv_bfloat162 value_0,
                                nv_bfloat16 *ptr_1, __nv_bfloat162 value_1,
                                uint32_t predicate) {
  const uint32_t data_0 = reinterpret_cast<const uint32_t &>(value_0);
  const uint32_t data_1 = reinterpret_cast<const uint32_t &>(value_1);
  asm volatile("{\n\t"
               ".reg .pred p;\n\t"
               "setp.ne.u32 p, %4, 0;\n\t"
               "@p st.global.relaxed.cta.L1::no_allocate.u32 [%0], %2;\n\t"
               "@p st.global.relaxed.cta.L1::no_allocate.u32 [%1], %3;\n\t"
               "}" ::"l"(ptr_0),
               "l"(ptr_1), "r"(data_0), "r"(data_1), "r"(predicate));
}

__global__ __block_size__((NUM_THREADS, 1, 1)) void o_v1_kernel_cutlass(
    const __grid_constant__ CUtensorMap q_tmap,
    const __grid_constant__ CUtensorMap k_tmap,
    const __grid_constant__ CUtensorMap v_tmap,
    const __grid_constant__ CUtensorMap h_tmap, const float *g_cu_ptr,
    nv_bfloat16 *o_ptr, const int64_t *cu_seqlens_ptr,
    const int32_t *chunk_indices_ptr, const int32_t *total_chunks_ptr,
    float scale) {
  const uint32_t tid = threadIdx.x;
  const uint32_t warp_id = tid / WARP_SIZE;
  const uint32_t lane_id = tid % WARP_SIZE;

  const uint32_t v_tile = blockIdx.x;
  const uint32_t qk_head_id = blockIdx.z;
  const uint32_t q_head_id = qk_head_id;
  const uint32_t k_head_id = qk_head_id;
  const uint32_t head_id_base = qk_head_id * HEADS_PER_QK_HEAD;
  const uint32_t v_start = v_tile * BLOCK_V;

  extern __shared__ __align__(1024) char smem_ptr[];
  const uint32_t smem = __cvta_generic_to_shared(smem_ptr);

  const uint32_t q_smem = smem + OFFSET_Q;
  const uint32_t k_smem = smem + OFFSET_K;
  const uint32_t v_smem = smem + OFFSET_V;
  const uint32_t h_smem = smem + OFFSET_H;
  const uint32_t attn_smem = smem + OFFSET_ATTN;
  const uint32_t qk_spill_smem = smem + OFFSET_QK_SPILL;
  const uint32_t g_raw_smem = smem + OFFSET_G_RAW;
  const uint32_t g_center_smem = smem + OFFSET_G_CENTER;
  const uint32_t g_pos_smem = smem + OFFSET_G_POS;
  const uint32_t g_alpha_smem = smem + OFFSET_G_ALPHA;
  const uint32_t g_neg_smem = smem + OFFSET_G_NEG;
  const uint32_t g_reduce_min_smem = smem + OFFSET_G_REDUCE_MIN;
  const uint32_t g_reduce_max_smem = smem + OFFSET_G_REDUCE_MAX;
  const uint32_t qk_tma_barriers = smem + OFFSET_QK_TMA_BAR;
  const uint32_t v_tma_barrier = smem + OFFSET_V_TMA_BAR;
  const uint32_t h_tma_barrier = smem + OFFSET_H_TMA_BAR;
  const uint32_t mma_barrier = smem + OFFSET_MMA_BAR;
  const uint32_t attn_ready_barriers = smem + OFFSET_ATTN_READY_BAR;
  const uint32_t head0_release_barrier = smem + OFFSET_HEAD0_RELEASE_BAR;
  const uint32_t qh_mma_barrier = smem + OFFSET_QH_MMA_BAR;
  const uint32_t ov_mma_barrier = smem + OFFSET_OV_MMA_BAR;
  const uint32_t tmem_alloc_smem = smem + OFFSET_TMEM_ADDR;

  float *g_raw_smem_ptr =
      reinterpret_cast<float *>(smem_ptr + (g_raw_smem - smem));
  float *g_center_smem_ptr =
      reinterpret_cast<float *>(smem_ptr + (g_center_smem - smem));
  float *g_pos_smem_ptr =
      reinterpret_cast<float *>(smem_ptr + (g_pos_smem - smem));
  float *g_alpha_smem_ptr =
      reinterpret_cast<float *>(smem_ptr + (g_alpha_smem - smem));
  float *g_neg_smem_ptr =
      reinterpret_cast<float *>(smem_ptr + (g_neg_smem - smem));
  float *g_reduce_min_smem_ptr =
      reinterpret_cast<float *>(smem_ptr + (g_reduce_min_smem - smem));
  float *g_reduce_max_smem_ptr =
      reinterpret_cast<float *>(smem_ptr + (g_reduce_max_smem - smem));

  if (warp_id == TMA_WARP) {
    if (elect_sync()) {
      mbarrier_init(qk_tma_barriers + 0U * 8U, 1);
      mbarrier_init(qk_tma_barriers + 1U * 8U, 1);
      mbarrier_init(v_tma_barrier + 0U * 8U, 1);
      mbarrier_init(v_tma_barrier + 1U * 8U, 1);
      mbarrier_init(h_tma_barrier + 0U * 8U, 1);
      mbarrier_init(h_tma_barrier + 1U * 8U, 1);
      mbarrier_init(mma_barrier, 1);
      mbarrier_init(attn_ready_barriers + 0U * 8U, NUM_CUDA_WARPS);
      mbarrier_init(attn_ready_barriers + 1U * 8U, NUM_CUDA_WARPS);
      mbarrier_init(head0_release_barrier, NUM_CUDA_WARPS);
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
  const bool cuda_warp = warp_id < NUM_CUDA_WARPS;
  uint32_t row_base = 0U;
  uint32_t row_hi = 0U;
  uint32_t lane_col = 0U;

  if (cuda_warp) {
    row_base = warp_id * ROWS_PER_WARP + lane_id / LANES_PER_ROW_GROUP;
    row_hi = row_base + ROW_PAIR_STRIDE;
    lane_col = lane_id % LANES_PER_ROW_GROUP;
  }

  const uint32_t total_num_chunks_u = static_cast<uint32_t>(*total_chunks_ptr);

  if (warp_id == TMA_WARP) {
    if (blockIdx.y < total_num_chunks_u) {
      const int2 first_chunk_meta =
          reinterpret_cast<const int2 *>(chunk_indices_ptr)[blockIdx.y];
      const uint32_t first_seq_id = static_cast<uint32_t>(first_chunk_meta.x);
      const uint32_t first_chunk_id = static_cast<uint32_t>(first_chunk_meta.y);
      const int32_t first_chunk_start_i32 = static_cast<int32_t>(
          cu_seqlens_ptr[first_seq_id] +
          static_cast<int64_t>(first_chunk_id) * static_cast<int64_t>(BLOCK_T));
      if (elect_sync()) {
        tma_load_qk_stage(q_smem, k_smem, &q_tmap, &k_tmap,
                          first_chunk_start_i32, q_head_id, k_head_id,
                          qk_tma_barriers);
        const int32_t first_v_chunk_start_i32 =
            static_cast<int32_t>(blockIdx.y * BLOCK_T);
        for (uint32_t head_offset = 0; head_offset < HEADS_PER_QK_HEAD;
             ++head_offset) {
          const uint32_t head_id = head_id_base + head_offset;
          const uint32_t h_outer = blockIdx.y * NUM_OUTPUT_HEADS + head_id;
          const uint32_t stage_index = head_offset;
          tma_load_4d(v_smem + stage_index * V_SMEM_SIZE, &v_tmap, 0,
                      first_v_chunk_start_i32, v_tile, head_id,
                      v_tma_barrier + stage_index * 8U);
          tma_load_4d(h_smem + stage_index * H_SMEM_SIZE, &h_tmap, 0, v_start,
                      0, h_outer, h_tma_barrier + stage_index * 8U);
          mbarrier_arrive_expect_tx(v_tma_barrier + stage_index * 8U,
                                    V_SMEM_SIZE);
          mbarrier_arrive_expect_tx(h_tma_barrier + stage_index * 8U,
                                    H_SMEM_SIZE);
        }
      }
    }

    for (uint32_t global_chunk_id = blockIdx.y, chunk_iter = 0U;
         global_chunk_id < total_num_chunks_u;
         global_chunk_id += gridDim.y, ++chunk_iter) {
      const int2 chunk_meta =
          reinterpret_cast<const int2 *>(chunk_indices_ptr)[global_chunk_id];
      const uint32_t seq_id = static_cast<uint32_t>(chunk_meta.x);
      const uint32_t chunk_id = static_cast<uint32_t>(chunk_meta.y);
      const int32_t chunk_start_i32 = static_cast<int32_t>(
          cu_seqlens_ptr[seq_id] +
          static_cast<int64_t>(chunk_id) * static_cast<int64_t>(BLOCK_T));
      const int32_t v_chunk_start_i32 =
          static_cast<int32_t>(global_chunk_id * BLOCK_T);
      const uint32_t qk_phase = chunk_iter & 1U;
      const uint32_t next_global_chunk_id = global_chunk_id + gridDim.y;
      const uint32_t next_qk_phase = qk_phase ^ 1U;
      const uint32_t next_qk_tma_barrier = qk_tma_barriers + next_qk_phase * 8U;
      const int32_t next_v_chunk_start_i32 =
          static_cast<int32_t>(next_global_chunk_id * BLOCK_T);

      if (elect_sync() && chunk_iter > 0U) {
        const uint32_t head1_id = head_id_base + HEAD1_STAGE_INDEX;
        const uint32_t head1_h_outer =
            global_chunk_id * NUM_OUTPUT_HEADS + head1_id;
        tma_load_4d(h_smem + HEAD1_STAGE_INDEX * H_SMEM_SIZE, &h_tmap, 0,
                    v_start, 0, head1_h_outer,
                    h_tma_barrier + HEAD1_STAGE_INDEX * 8U);
        tma_load_4d(v_smem + HEAD1_STAGE_INDEX * V_SMEM_SIZE, &v_tmap, 0,
                    v_chunk_start_i32, v_tile, head1_id,
                    v_tma_barrier + HEAD1_STAGE_INDEX * 8U);
        mbarrier_arrive_expect_tx(h_tma_barrier + HEAD1_STAGE_INDEX * 8U,
                                  H_SMEM_SIZE);
        mbarrier_arrive_expect_tx(v_tma_barrier + HEAD1_STAGE_INDEX * 8U,
                                  V_SMEM_SIZE);
      }

      if (elect_sync() && next_global_chunk_id < total_num_chunks_u) {
        const int2 next_chunk_meta = reinterpret_cast<const int2 *>(
            chunk_indices_ptr)[next_global_chunk_id];
        const uint32_t next_seq_id = static_cast<uint32_t>(next_chunk_meta.x);
        const uint32_t next_chunk_id = static_cast<uint32_t>(next_chunk_meta.y);
        const int32_t next_chunk_start_i32 = static_cast<int32_t>(
            cu_seqlens_ptr[next_seq_id] + static_cast<int64_t>(next_chunk_id) *
                                              static_cast<int64_t>(BLOCK_T));
        tma_load_qk_stage(q_smem + next_qk_phase * Q_SMEM_SIZE,
                          k_smem + next_qk_phase * K_SMEM_SIZE, &q_tmap,
                          &k_tmap, next_chunk_start_i32, q_head_id, k_head_id,
                          next_qk_tma_barrier);
      }

      if (elect_sync() && next_global_chunk_id < total_num_chunks_u) {
        const uint32_t next_head0_id = head_id_base + HEAD0_STAGE_INDEX;
        const uint32_t next_head0_h_outer =
            next_global_chunk_id * NUM_OUTPUT_HEADS + next_head0_id;
        mbarrier_wait(qh_mma_barrier, 0);
        tma_load_4d(h_smem + HEAD0_STAGE_INDEX * H_SMEM_SIZE, &h_tmap, 0,
                    v_start, 0, next_head0_h_outer,
                    h_tma_barrier + HEAD0_STAGE_INDEX * 8U);
        mbarrier_wait(ov_mma_barrier, 0);
        tma_load_4d(v_smem + HEAD0_STAGE_INDEX * V_SMEM_SIZE, &v_tmap, 0,
                    next_v_chunk_start_i32, v_tile, next_head0_id,
                    v_tma_barrier + HEAD0_STAGE_INDEX * 8U);
        mbarrier_arrive_expect_tx(h_tma_barrier + HEAD0_STAGE_INDEX * 8U,
                                  H_SMEM_SIZE);
        mbarrier_arrive_expect_tx(v_tma_barrier + HEAD0_STAGE_INDEX * 8U,
                                  V_SMEM_SIZE);
      }

      tcgen05_fence_before_thread_sync();
      __syncthreads();
    }
  } else if (warp_id == MMA_WARP) {
    for (uint32_t global_chunk_id = blockIdx.y, chunk_iter = 0U;
         global_chunk_id < total_num_chunks_u;
         global_chunk_id += gridDim.y, ++chunk_iter) {
      const uint32_t qk_phase = chunk_iter & 1U;
      const uint32_t q_stage_smem = q_smem + qk_phase * Q_SMEM_SIZE;
      const uint32_t k_stage_smem = k_smem + qk_phase * K_SMEM_SIZE;
      const uint32_t qk_tma_barrier = qk_tma_barriers + qk_phase * 8U;
      const uint32_t qk_tma_stage_phase = (chunk_iter / QK_STAGE_COUNT) & 1U;

      if (elect_sync()) {
        mbarrier_wait(qk_tma_barrier, TMA_PHASE ^ qk_tma_stage_phase);
        tcgen05_fence_after_thread_sync();
        mma_swizzled<NUM_SWIZZLE_ATOMS>(0, q_stage_smem, k_stage_smem);
        tcgen05_commit(mma_barrier);
      }

#pragma unroll
      for (uint32_t head_offset = 0; head_offset < HEADS_PER_QK_HEAD;
           ++head_offset) {
        const uint32_t head_phase = head_offset & 1U;
        const uint32_t vh_stage_phase = chunk_iter & 1U;
        const uint32_t v_stage_index = head_offset;
        const uint32_t v_stage_smem = v_smem + v_stage_index * V_SMEM_SIZE;
        const uint32_t h_stage_index = head_offset;
        const uint32_t h_stage_smem = h_smem + h_stage_index * H_SMEM_SIZE;
        const uint32_t attn_stage_smem =
            attn_smem + head_offset * ATTN_SMEM_SIZE;
        const uint32_t attn_stage_barrier =
            attn_ready_barriers + head_offset * 8U;
        const uint32_t v_stage_barrier = v_tma_barrier + v_stage_index * 8U;
        const uint32_t h_stage_barrier = h_tma_barrier + h_stage_index * 8U;

        if (elect_sync()) {
          if (head_offset > 0U) {
            mbarrier_wait(head0_release_barrier, qk_phase);
          }
          mbarrier_wait(h_stage_barrier, vh_stage_phase);
          tcgen05_fence_after_thread_sync();
          mma_swizzled_qh_64x128(QH_TMEM_COL, q_stage_smem, h_stage_smem);
          tcgen05_commit(qh_mma_barrier);
          mbarrier_wait(v_stage_barrier, vh_stage_phase);
          mbarrier_wait(attn_stage_barrier, vh_stage_phase);
          tcgen05_fence_after_thread_sync();
          mma_attn_v_mmajor_64x128(OUTPUT_TMEM_COL, attn_stage_smem,
                                   v_stage_smem);
          tcgen05_commit(ov_mma_barrier);
        }

        tcgen05_fence_before_thread_sync();
        if (head_offset + 1U == HEADS_PER_QK_HEAD) {
          __syncthreads();
        }
      }
    }
  } else {
    for (uint32_t global_chunk_id = blockIdx.y, chunk_iter = 0U;
         global_chunk_id < total_num_chunks_u;
         global_chunk_id += gridDim.y, ++chunk_iter) {
      const int2 chunk_meta =
          reinterpret_cast<const int2 *>(chunk_indices_ptr)[global_chunk_id];
      const uint32_t seq_id = static_cast<uint32_t>(chunk_meta.x);
      const uint32_t chunk_id = static_cast<uint32_t>(chunk_meta.y);
      const int64_t bos = cu_seqlens_ptr[seq_id];
      const int64_t eos = cu_seqlens_ptr[seq_id + 1];
      const int64_t chunk_start =
          bos + static_cast<int64_t>(chunk_id) * BLOCK_T;
      const int64_t remaining = eos - chunk_start;
      const uint32_t chunk_len =
          remaining <= 0
              ? 0U
              : static_cast<uint32_t>(remaining < static_cast<int64_t>(BLOCK_T)
                                          ? remaining
                                          : static_cast<int64_t>(BLOCK_T));
      const bool full_chunk = chunk_len == BLOCK_T;
      const uint32_t row_base_limit = row_base < chunk_len ? row_base + 1U : 0U;
      const uint32_t row_hi_limit = row_hi < chunk_len ? row_hi + 1U : 0U;
      const uint32_t qk_phase = chunk_iter & 1U;

      for (uint32_t i = tid; i < HEADS_PER_QK_HEAD * BLOCK_T;
           i += NUM_CUDA_WARPS * WARP_SIZE) {
        const uint32_t token_offset = i % BLOCK_T;
        float g_value = 0.0f;
        if (token_offset < chunk_len) {
          const uint32_t head_offset = i / BLOCK_T;
          const uint32_t head_id = head_id_base + head_offset;
          const int64_t token_idx =
              chunk_start + static_cast<int64_t>(token_offset);
          g_value = g_cu_ptr[token_idx * NUM_OUTPUT_HEADS + head_id];
        }
        g_raw_smem_ptr[i] = g_value;
      }
      bar_sync<1>(NUM_CUDA_WARPS * WARP_SIZE);

      reduce_g_center_per_head(g_raw_smem_ptr, g_center_smem_ptr,
                               g_reduce_min_smem_ptr, g_reduce_max_smem_ptr,
                               tid, warp_id, lane_id, chunk_len);
      bar_sync<1>(NUM_CUDA_WARPS * WARP_SIZE);

      if (tid < HEADS_PER_QK_HEAD) {
        g_alpha_smem_ptr[tid] = __expf(g_center_smem_ptr[tid]);
      }
      for (uint32_t i = tid; i < HEADS_PER_QK_HEAD * BLOCK_T;
           i += NUM_CUDA_WARPS * WARP_SIZE) {
        const uint32_t head_offset = i / BLOCK_T;
        const float g_shifted =
            g_raw_smem_ptr[i] - g_center_smem_ptr[head_offset];
        const float gp = __expf(g_shifted);
        g_pos_smem_ptr[i] = gp;
        g_neg_smem_ptr[i] = __frcp_rn(gp);
      }
      bar_sync<1>(NUM_CUDA_WARPS * WARP_SIZE);

      mbarrier_wait(mma_barrier, QK_MMA_PHASE ^ qk_phase);
      tcgen05_fence_after_thread_sync();

      float qk_reg_lo[REGS_PER_FRAGMENT];
      float qk_reg_hi[REGS_PER_FRAGMENT];
      load_qk_fragment(qk_reg_lo, qk_reg_hi);

      const float *head0_g_neg_ptr = g_neg_smem_ptr + HEAD0_STAGE_INDEX * BLOCK_T;
      materialize_attn_stage_from_regs(
          smem_ptr, smem, attn_smem + HEAD0_STAGE_INDEX * ATTN_SMEM_SIZE,
          qk_reg_lo, qk_reg_hi, head0_g_neg_ptr, row_base, row_hi,
          row_base_limit, row_hi_limit, lane_col);
      fence_proxy_async_shared_cta();
      __syncwarp();
      if (elect_sync()) {
        mbarrier_arrive(attn_ready_barriers + HEAD0_STAGE_INDEX * 8U);
      }

#pragma unroll
      for (uint32_t head_offset = 0; head_offset < HEADS_PER_QK_HEAD;
           ++head_offset) {
        const uint32_t head_phase = head_offset & 1U;
        const uint32_t head_id = head_id_base + head_offset;
        const float *gp_head_ptr = g_pos_smem_ptr + head_offset * BLOCK_T;
        const float gp_row_base = gp_head_ptr[row_base];
        const float gp_row_hi = gp_head_ptr[row_hi];
        const float alpha = g_alpha_smem_ptr[head_offset];

        if (head_offset + 1U < HEADS_PER_QK_HEAD) {
          const uint32_t next_head_offset = head_offset + 1U;
          const float *next_g_neg_ptr =
              g_neg_smem_ptr + next_head_offset * BLOCK_T;
          materialize_attn_stage_from_regs(
              smem_ptr, smem, attn_smem + next_head_offset * ATTN_SMEM_SIZE,
              qk_reg_lo, qk_reg_hi, next_g_neg_ptr, row_base, row_hi,
              row_base_limit, row_hi_limit, lane_col);
          fence_proxy_async_shared_cta();
          __syncwarp();
          if (elect_sync()) {
            mbarrier_arrive(attn_ready_barriers + next_head_offset * 8U);
          }
        }

        mbarrier_wait(ov_mma_barrier, head_phase);
        mbarrier_wait(qh_mma_barrier, head_phase);
        tcgen05_fence_after_thread_sync();

        if (full_chunk) {
          nv_bfloat16 *row_base_o_ptr =
              o_ptr + (((chunk_start + static_cast<int64_t>(row_base)) *
                            NUM_OUTPUT_HEADS +
                        head_id) *
                           VALUE_DIM +
                       v_start);
          nv_bfloat16 *row_hi_o_ptr = row_base_o_ptr + ROW_PAIR_OUTPUT_STRIDE;
#pragma unroll
          for (uint32_t fragment_pair = 0; fragment_pair < BLOCK_V / BLOCK_T;
               ++fragment_pair) {
            const uint32_t col_base = fragment_pair * BLOCK_T;
            float ov_reg_lo[REGS_PER_FRAGMENT];
            float ov_reg_hi[REGS_PER_FRAGMENT];
            float qh_reg_lo[REGS_PER_FRAGMENT];
            float qh_reg_hi[REGS_PER_FRAGMENT];
            tcgen05_ld<SHAPE::_16x256b, 4>(ov_reg_lo, 0,
                                           OUTPUT_TMEM_COL + col_base);
            tcgen05_ld<SHAPE::_16x256b, 4>(
                ov_reg_hi, 0, OUTPUT_TMEM_COL + col_base + COLS_PER_FRAGMENT);
            tcgen05_ld<SHAPE::_16x256b, 4>(qh_reg_lo, 0,
                                           QH_TMEM_COL + col_base);
            tcgen05_ld<SHAPE::_16x256b, 4>(
                qh_reg_hi, 0, QH_TMEM_COL + col_base + COLS_PER_FRAGMENT);
            tcgen05_wait_ld();
#pragma unroll
            for (uint32_t step_pair = 0; step_pair < FRAGMENT_PAIRS;
                 ++step_pair) {
              const uint32_t col_lo = col_base + step_pair * 8U + 2U * lane_col;
              const uint32_t col_hi = col_lo + COLS_PER_FRAGMENT;
              const uint32_t reg_row0 = step_pair * 4U;
              const uint32_t reg_row1 = reg_row0 + 2U;

              store_bf162_no_allocate(
                  row_base_o_ptr + col_lo,
                  combine_output_pair(
                      ov_reg_lo[reg_row0], ov_reg_lo[reg_row0 + 1U],
                      qh_reg_lo[reg_row0], qh_reg_lo[reg_row0 + 1U],
                      gp_row_base, alpha, scale));
              store_bf162_no_allocate(
                  row_base_o_ptr + col_hi,
                  combine_output_pair(
                      ov_reg_hi[reg_row0], ov_reg_hi[reg_row0 + 1U],
                      qh_reg_hi[reg_row0], qh_reg_hi[reg_row0 + 1U],
                      gp_row_base, alpha, scale));
              store_bf162_no_allocate(
                  row_hi_o_ptr + col_lo,
                  combine_output_pair(
                      ov_reg_lo[reg_row1], ov_reg_lo[reg_row1 + 1U],
                      qh_reg_lo[reg_row1], qh_reg_lo[reg_row1 + 1U],
                      gp_row_hi, alpha, scale));
              store_bf162_no_allocate(
                  row_hi_o_ptr + col_hi,
                  combine_output_pair(
                      ov_reg_hi[reg_row1], ov_reg_hi[reg_row1 + 1U],
                      qh_reg_hi[reg_row1], qh_reg_hi[reg_row1 + 1U],
                      gp_row_hi, alpha, scale));
            }
          }
        } else {
          const uint32_t row_base_active = row_base < chunk_len;
          const uint32_t row_hi_active = row_hi < chunk_len;
          nv_bfloat16 *row_base_o_ptr =
              o_ptr + (((chunk_start + static_cast<int64_t>(row_base)) *
                            NUM_OUTPUT_HEADS +
                        head_id) *
                           VALUE_DIM +
                       v_start);
          nv_bfloat16 *row_hi_o_ptr = row_base_o_ptr + ROW_PAIR_OUTPUT_STRIDE;
#pragma unroll
          for (uint32_t fragment_pair = 0; fragment_pair < BLOCK_V / BLOCK_T;
               ++fragment_pair) {
            const uint32_t col_base = fragment_pair * BLOCK_T;
            float ov_reg_lo[REGS_PER_FRAGMENT];
            float ov_reg_hi[REGS_PER_FRAGMENT];
            float qh_reg_lo[REGS_PER_FRAGMENT];
            float qh_reg_hi[REGS_PER_FRAGMENT];
            tcgen05_ld<SHAPE::_16x256b, 4>(ov_reg_lo, 0,
                                           OUTPUT_TMEM_COL + col_base);
            tcgen05_ld<SHAPE::_16x256b, 4>(
                ov_reg_hi, 0, OUTPUT_TMEM_COL + col_base + COLS_PER_FRAGMENT);
            tcgen05_ld<SHAPE::_16x256b, 4>(qh_reg_lo, 0,
                                           QH_TMEM_COL + col_base);
            tcgen05_ld<SHAPE::_16x256b, 4>(
                qh_reg_hi, 0, QH_TMEM_COL + col_base + COLS_PER_FRAGMENT);
            tcgen05_wait_ld();
#pragma unroll
            for (uint32_t step_pair = 0; step_pair < FRAGMENT_PAIRS;
                 ++step_pair) {
              const uint32_t col_lo = col_base + step_pair * 8U + 2U * lane_col;
              const uint32_t col_hi = col_lo + COLS_PER_FRAGMENT;
              const uint32_t reg_row0 = step_pair * 4U;
              const uint32_t reg_row1 = reg_row0 + 2U;

              store_bf162_pair_no_allocate_if(
                  row_base_o_ptr + col_lo,
                  combine_output_pair(
                      ov_reg_lo[reg_row0], ov_reg_lo[reg_row0 + 1U],
                      qh_reg_lo[reg_row0], qh_reg_lo[reg_row0 + 1U],
                      gp_row_base, alpha, scale),
                  row_base_o_ptr + col_hi,
                  combine_output_pair(
                      ov_reg_hi[reg_row0], ov_reg_hi[reg_row0 + 1U],
                      qh_reg_hi[reg_row0], qh_reg_hi[reg_row0 + 1U],
                      gp_row_base, alpha, scale),
                  row_base_active);
              store_bf162_pair_no_allocate_if(
                  row_hi_o_ptr + col_lo,
                  combine_output_pair(
                      ov_reg_lo[reg_row1], ov_reg_lo[reg_row1 + 1U],
                      qh_reg_lo[reg_row1], qh_reg_lo[reg_row1 + 1U],
                      gp_row_hi, alpha, scale),
                  row_hi_o_ptr + col_hi,
                  combine_output_pair(
                      ov_reg_hi[reg_row1], ov_reg_hi[reg_row1 + 1U],
                      qh_reg_hi[reg_row1], qh_reg_hi[reg_row1 + 1U],
                      gp_row_hi, alpha, scale),
                  row_hi_active);
            }
          }
        }

        tcgen05_fence_before_thread_sync();
        if (head_offset + 1U < HEADS_PER_QK_HEAD) {
          __syncwarp();
          if (elect_sync()) {
            mbarrier_arrive(head0_release_barrier);
          }
        } else {
          __syncthreads();
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

void launch_o_v1(TensorView q_chunks, TensorView k_chunks, TensorView v_new,
                 TensorView h, TensorView g_cu, TensorView o,
                 TensorView cu_seqlens, TensorView chunk_indices,
                 TensorView total_chunks, double scale) {
  const uint32_t total_num_tokens = static_cast<uint32_t>(q_chunks.size(0));
  const uint32_t total_num_chunks_capacity = static_cast<uint32_t>(v_new.size(0));
  if (total_num_chunks_capacity == 0U) {
    return;
  }

  const uint32_t v_num_tokens = total_num_chunks_capacity * BLOCK_T;
  const uint64_t total_num_output_tiles =
      static_cast<uint64_t>(h.size(0)) * NUM_OUTPUT_HEADS;

  auto q_tmap = encode_qk_tma(q_chunks.data_ptr(), total_num_tokens,
                              NUM_QK_HEADS, HEAD_DIM);
  auto k_tmap = encode_qk_tma(k_chunks.data_ptr(), total_num_tokens,
                              NUM_QK_HEADS, HEAD_DIM);
  auto v_tmap = encode_v_tma(v_new.data_ptr(), v_num_tokens);
  auto h_tmap =
      encode_tma(h.data_ptr(), total_num_output_tiles, VALUE_DIM, HEAD_DIM);

  auto *g_cu_ptr = reinterpret_cast<const float *>(g_cu.data_ptr());
  auto *o_ptr = reinterpret_cast<nv_bfloat16 *>(o.data_ptr());
  auto *cu_seqlens_ptr =
      reinterpret_cast<const int64_t *>(cu_seqlens.data_ptr());
  auto *chunk_indices_ptr =
      reinterpret_cast<const int32_t *>(chunk_indices.data_ptr());
  auto *total_chunks_ptr =
      reinterpret_cast<const int32_t *>(total_chunks.data_ptr());

  auto kernel = o_v1_kernel_cutlass;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       SMEM_SIZE);

  int active_blocks_per_sm = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks_per_sm, kernel,
                                                NUM_THREADS, SMEM_SIZE);
  int device_id = 0;
  cudaGetDevice(&device_id);
  cudaDeviceProp device_prop;
  cudaGetDeviceProperties(&device_prop, device_id);
  const uint32_t blocks_per_chunk = (VALUE_DIM / BLOCK_V) * NUM_QK_HEADS;
  const uint32_t resident_blocks =
      active_blocks_per_sm > 0
          ? static_cast<uint32_t>(active_blocks_per_sm) *
                static_cast<uint32_t>(device_prop.multiProcessorCount)
          : blocks_per_chunk;
  uint32_t grid_y = 2U * (resident_blocks / blocks_per_chunk);
  if (grid_y == 0U) {
    grid_y = 1U;
  }
  grid_y = grid_y > 1U ? (grid_y + 1U) / 2U : 1U;
  if (grid_y > total_num_chunks_capacity) {
    grid_y = total_num_chunks_capacity;
  }

  dim3 grid(VALUE_DIM / BLOCK_V, grid_y, NUM_QK_HEADS);
  kernel<<<grid, NUM_THREADS, SMEM_SIZE>>>(
      q_tmap, k_tmap, v_tmap, h_tmap, g_cu_ptr, o_ptr, cu_seqlens_ptr,
      chunk_indices_ptr, total_chunks_ptr, static_cast<float>(scale));
}

void o_v1(TensorView q_chunks, TensorView k_chunks, TensorView v_new,
          TensorView h, TensorView g_cu, TensorView o, TensorView cu_seqlens,
          TensorView chunk_indices, TensorView total_chunks, double scale) {
  launch_o_v1(q_chunks, k_chunks, v_new, h, g_cu, o, cu_seqlens, chunk_indices,
              total_chunks, scale);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(o_v1, o_v1);
