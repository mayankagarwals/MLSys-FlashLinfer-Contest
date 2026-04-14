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
constexpr uint32_t H_RING_STAGE_COUNT = 2U;
constexpr uint32_t V_RING_STAGE_COUNT = 2U;
constexpr uint32_t H_STAGE_COUNT = HEADS_PER_QK_HEAD * H_RING_STAGE_COUNT;
constexpr uint32_t V_STAGE_COUNT = HEADS_PER_QK_HEAD * V_RING_STAGE_COUNT;
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
constexpr uint32_t V_SMEM_BYTES = V_STAGE_COUNT * V_SMEM_SIZE;
constexpr uint32_t H_SMEM_BYTES = H_STAGE_COUNT * H_SMEM_SIZE;
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
constexpr uint32_t V_TMA_BARRIER_BYTES = V_STAGE_COUNT * 8U;
constexpr uint32_t H_TMA_BARRIER_BYTES = H_STAGE_COUNT * 8U;
constexpr uint32_t ATTN_READY_BARRIER_BYTES = ATTN_STAGE_COUNT * 8U;
constexpr uint32_t QH_MMA_BARRIER_BYTES = H_STAGE_COUNT * 8U;
constexpr uint32_t OV_MMA_BARRIER_BYTES = V_STAGE_COUNT * 8U;
constexpr uint32_t QH_RELEASE_BARRIER_BYTES = H_STAGE_COUNT * 8U;
constexpr uint32_t HEAD_MMA_BARRIER_BYTES = HEADS_PER_QK_HEAD * 8U;
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
constexpr uint32_t OFFSET_H_TMA_BAR = OFFSET_V_TMA_BAR + V_TMA_BARRIER_BYTES;
constexpr uint32_t OFFSET_MMA_BAR = OFFSET_H_TMA_BAR + H_TMA_BARRIER_BYTES;
constexpr uint32_t OFFSET_QK_STAGE_RELEASE_BAR = OFFSET_MMA_BAR + 8U;
constexpr uint32_t OFFSET_QK_CONSUMED_BAR =
    OFFSET_QK_STAGE_RELEASE_BAR + QK_TMA_BARRIER_BYTES;
constexpr uint32_t OFFSET_ATTN_READY_BAR = OFFSET_QK_CONSUMED_BAR + 8U;
constexpr uint32_t OFFSET_ATTN_RELEASE_BAR =
    OFFSET_ATTN_READY_BAR + ATTN_READY_BARRIER_BYTES;
constexpr uint32_t OFFSET_QH_MMA_BAR =
    OFFSET_ATTN_RELEASE_BAR + ATTN_READY_BARRIER_BYTES;
constexpr uint32_t OFFSET_OV_MMA_BAR =
    OFFSET_QH_MMA_BAR + QH_MMA_BARRIER_BYTES;
constexpr uint32_t OFFSET_QH_RELEASE_BAR =
    OFFSET_OV_MMA_BAR + OV_MMA_BARRIER_BYTES;
constexpr uint32_t OFFSET_OUTPUT_RELEASE_BAR =
    OFFSET_QH_RELEASE_BAR + QH_RELEASE_BARRIER_BYTES;
constexpr uint32_t OFFSET_TMEM_ADDR =
    OFFSET_OUTPUT_RELEASE_BAR + HEAD_MMA_BARRIER_BYTES;
constexpr uint32_t SMEM_SIZE = (OFFSET_TMEM_ADDR + 4U + 1023U) & ~1023U;

//// Tensor Memory
constexpr uint32_t MAX_COLUMNS = 512U;
constexpr uint32_t QK_TMEM_COL = 0U;
constexpr uint32_t OUTPUT_TMEM_COLS = HEADS_PER_QK_HEAD * BLOCK_T;
constexpr uint32_t QH_TMEM_COLS =
    H_RING_STAGE_COUNT * HEADS_PER_QK_HEAD * BLOCK_T;
static_assert(BLOCK_T + OUTPUT_TMEM_COLS + QH_TMEM_COLS <= MAX_COLUMNS);
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

__device__ __forceinline__ uint32_t get_output_tmem_col(uint32_t head_offset) {
  return BLOCK_T + head_offset * BLOCK_T;
}

__device__ __forceinline__ uint32_t get_qh_tmem_col(uint32_t head_offset,
                                                    uint32_t h_stage) {
  return 3U * BLOCK_T + h_stage * HEADS_PER_QK_HEAD * BLOCK_T +
         head_offset * BLOCK_T;
}

__device__ __forceinline__ uint32_t get_h_stage_index(uint32_t head_offset,
                                                      uint32_t h_stage) {
  return head_offset * H_RING_STAGE_COUNT + h_stage;
}

__device__ __forceinline__ uint32_t get_v_stage_index(uint32_t head_offset,
                                                      uint32_t v_stage) {
  return head_offset * V_RING_STAGE_COUNT + v_stage;
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
  asm volatile("st.shared.b32 [%0], %1;" ::"r"(cvt_smem_ptr(ptr)), "r"(data)
               : "memory");
}

__device__ __forceinline__ __nv_bfloat162 load_bf162_shared(
    const nv_bfloat16 *ptr) {
  uint32_t data;
  asm volatile("ld.shared.b32 %0, [%1];"
               : "=r"(data)
               : "r"(cvt_smem_ptr(ptr))
               : "memory");
  return reinterpret_cast<const __nv_bfloat162 &>(data);
}

__device__ __forceinline__ void store_attn_column_pair(
    nv_bfloat16 *attn_smem_ptr, const float *g_neg_smem_ptr, float qk_row0_col0,
    float qk_row0_col1, float qk_row1_col0, float qk_row1_col1,
    uint32_t row_base, uint32_t row_hi, uint32_t row_base_limit,
    uint32_t row_hi_limit, uint32_t col) {
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
    float *g_reduce_min_smem_ptr, float *g_reduce_max_smem_ptr, uint32_t tid,
    uint32_t warp_id, uint32_t lane_id, uint32_t chunk_len) {
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
      const float head_min = fminf(g_reduce_min_smem_ptr[warp_base],
                                   g_reduce_min_smem_ptr[warp_base + 1U]);
      const float head_max = fmaxf(g_reduce_max_smem_ptr[warp_base],
                                   g_reduce_max_smem_ptr[warp_base + 1U]);
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

__device__ __forceinline__ __nv_bfloat162
combine_output_pair(float ov_0, float ov_1, float qh_0, float qh_1,
                    float gp_row, float alpha, float scale) {
  const float row_scale = scale * gp_row;
  return __floats2bfloat162_rn(row_scale * __fmaf_rn(alpha, qh_0, ov_0),
                               row_scale * __fmaf_rn(alpha, qh_1, ov_1));
}

__device__ __forceinline__ __nv_bfloat162
combine_output_pair_prescaled_qh(float ov_0, float ov_1, float qh_0,
                                 float qh_1, float gp_row, float scale) {
  const float row_scale = scale * gp_row;
  return __floats2bfloat162_rn(row_scale * (qh_0 + ov_0),
                               row_scale * (qh_1 + ov_1));
}

__device__ __forceinline__ void store_bf162_no_allocate(nv_bfloat16 *ptr,
                                                        __nv_bfloat162 value);
__device__ __forceinline__ void
store_bf162_pair_no_allocate_if(nv_bfloat16 *ptr_0, __nv_bfloat162 value_0,
                                nv_bfloat16 *ptr_1, __nv_bfloat162 value_1,
                                uint32_t predicate);

__device__ __forceinline__ void
load_tmem_fragment_pair(uint32_t tmem_col, uint32_t col_base, float *reg_lo,
                        float *reg_hi) {
  tcgen05_ld<SHAPE::_16x256b, 4>(reg_lo, 0, tmem_col + col_base);
  tcgen05_ld<SHAPE::_16x256b, 4>(reg_hi, 0, tmem_col + col_base +
                                                COLS_PER_FRAGMENT);
  tcgen05_wait_ld();
}

__device__ __forceinline__ void
load_output_fragment_pair(uint32_t output_tmem_col, uint32_t qh_tmem_col,
                          uint32_t col_base, float *ov_reg_lo, float *ov_reg_hi,
                          float *qh_reg_lo, float *qh_reg_hi) {
  tcgen05_ld<SHAPE::_16x256b, 4>(ov_reg_lo, 0, output_tmem_col + col_base);
  tcgen05_ld<SHAPE::_16x256b, 4>(
      ov_reg_hi, 0, output_tmem_col + col_base + COLS_PER_FRAGMENT);
  tcgen05_ld<SHAPE::_16x256b, 4>(qh_reg_lo, 0, qh_tmem_col + col_base);
  tcgen05_ld<SHAPE::_16x256b, 4>(qh_reg_hi, 0,
                                 qh_tmem_col + col_base + COLS_PER_FRAGMENT);
  tcgen05_wait_ld();
}

__device__ __forceinline__ void
stage_qh_row_pair_to_shared(uint32_t qh_tmem_col,
                            nv_bfloat16 *row_base_output_smem_ptr,
                            nv_bfloat16 *row_hi_output_smem_ptr,
                            uint32_t lane_col, float alpha) {
#pragma unroll
  for (uint32_t fragment_pair = 0; fragment_pair < BLOCK_V / BLOCK_T;
       ++fragment_pair) {
    const uint32_t col_base = fragment_pair * BLOCK_T;
    float qh_reg_lo[REGS_PER_FRAGMENT];
    float qh_reg_hi[REGS_PER_FRAGMENT];
    load_tmem_fragment_pair(qh_tmem_col, col_base, qh_reg_lo, qh_reg_hi);

#pragma unroll
    for (uint32_t step_pair = 0; step_pair < FRAGMENT_PAIRS; ++step_pair) {
      const uint32_t col_lo = col_base + step_pair * 8U + 2U * lane_col;
      const uint32_t col_hi = col_lo + COLS_PER_FRAGMENT;
      const uint32_t reg_row0 = step_pair * 4U;
      const uint32_t reg_row1 = reg_row0 + 2U;
      const __nv_bfloat162 row_base_lo = __floats2bfloat162_rn(
          alpha * qh_reg_lo[reg_row0], alpha * qh_reg_lo[reg_row0 + 1U]);
      const __nv_bfloat162 row_base_hi = __floats2bfloat162_rn(
          alpha * qh_reg_hi[reg_row0], alpha * qh_reg_hi[reg_row0 + 1U]);
      const __nv_bfloat162 row_hi_lo = __floats2bfloat162_rn(
          alpha * qh_reg_lo[reg_row1], alpha * qh_reg_lo[reg_row1 + 1U]);
      const __nv_bfloat162 row_hi_hi = __floats2bfloat162_rn(
          alpha * qh_reg_hi[reg_row1], alpha * qh_reg_hi[reg_row1 + 1U]);
      store_bf162_shared(row_base_output_smem_ptr + col_lo, row_base_lo);
      store_bf162_shared(row_base_output_smem_ptr + col_hi, row_base_hi);
      store_bf162_shared(row_hi_output_smem_ptr + col_lo, row_hi_lo);
      store_bf162_shared(row_hi_output_smem_ptr + col_hi, row_hi_hi);
    }
  }
}

__device__ __forceinline__ void
stage_output_row_pair_to_shared(uint32_t output_tmem_col,
                                uint32_t qh_tmem_col,
                                nv_bfloat16 *row_base_output_smem_ptr,
                                nv_bfloat16 *row_hi_output_smem_ptr,
                                uint32_t lane_col, float gp_row_base,
                                float gp_row_hi, float alpha, float scale) {
#pragma unroll
  for (uint32_t fragment_pair = 0; fragment_pair < BLOCK_V / BLOCK_T;
       ++fragment_pair) {
    const uint32_t col_base = fragment_pair * BLOCK_T;
    float ov_reg_lo[REGS_PER_FRAGMENT];
    float ov_reg_hi[REGS_PER_FRAGMENT];
    float qh_reg_lo[REGS_PER_FRAGMENT];
    float qh_reg_hi[REGS_PER_FRAGMENT];
    load_output_fragment_pair(output_tmem_col, qh_tmem_col, col_base, ov_reg_lo,
                              ov_reg_hi, qh_reg_lo, qh_reg_hi);

#pragma unroll
    for (uint32_t step_pair = 0; step_pair < FRAGMENT_PAIRS; ++step_pair) {
      const uint32_t col_lo = col_base + step_pair * 8U + 2U * lane_col;
      const uint32_t col_hi = col_lo + COLS_PER_FRAGMENT;
      const uint32_t reg_row0 = step_pair * 4U;
      const uint32_t reg_row1 = reg_row0 + 2U;
      const __nv_bfloat162 row_base_lo = combine_output_pair(
          ov_reg_lo[reg_row0], ov_reg_lo[reg_row0 + 1U], qh_reg_lo[reg_row0],
          qh_reg_lo[reg_row0 + 1U], gp_row_base, alpha, scale);
      const __nv_bfloat162 row_base_hi = combine_output_pair(
          ov_reg_hi[reg_row0], ov_reg_hi[reg_row0 + 1U], qh_reg_hi[reg_row0],
          qh_reg_hi[reg_row0 + 1U], gp_row_base, alpha, scale);
      const __nv_bfloat162 row_hi_lo = combine_output_pair(
          ov_reg_lo[reg_row1], ov_reg_lo[reg_row1 + 1U], qh_reg_lo[reg_row1],
          qh_reg_lo[reg_row1 + 1U], gp_row_hi, alpha, scale);
      const __nv_bfloat162 row_hi_hi = combine_output_pair(
          ov_reg_hi[reg_row1], ov_reg_hi[reg_row1 + 1U], qh_reg_hi[reg_row1],
          qh_reg_hi[reg_row1 + 1U], gp_row_hi, alpha, scale);
      store_bf162_shared(row_base_output_smem_ptr + col_lo, row_base_lo);
      store_bf162_shared(row_base_output_smem_ptr + col_hi, row_base_hi);
      store_bf162_shared(row_hi_output_smem_ptr + col_lo, row_hi_lo);
      store_bf162_shared(row_hi_output_smem_ptr + col_hi, row_hi_hi);
    }
  }
}

template <bool FULL_CHUNK>
__device__ __forceinline__ void
store_output_row_pair_from_tmem_and_shared(
    uint32_t output_tmem_col, const nv_bfloat16 *row_base_output_smem_ptr,
    const nv_bfloat16 *row_hi_output_smem_ptr, nv_bfloat16 *row_base_o_ptr,
    nv_bfloat16 *row_hi_o_ptr, uint32_t lane_col, uint32_t row_base_active,
    uint32_t row_hi_active, float gp_row_base, float gp_row_hi, float scale) {
#pragma unroll
  for (uint32_t fragment_pair = 0; fragment_pair < BLOCK_V / BLOCK_T;
       ++fragment_pair) {
    const uint32_t col_base = fragment_pair * BLOCK_T;
    float ov_reg_lo[REGS_PER_FRAGMENT];
    float ov_reg_hi[REGS_PER_FRAGMENT];
    load_tmem_fragment_pair(output_tmem_col, col_base, ov_reg_lo, ov_reg_hi);

#pragma unroll
    for (uint32_t step_pair = 0; step_pair < FRAGMENT_PAIRS; ++step_pair) {
      const uint32_t col_lo = col_base + step_pair * 8U + 2U * lane_col;
      const uint32_t col_hi = col_lo + COLS_PER_FRAGMENT;
      const uint32_t reg_row0 = step_pair * 4U;
      const uint32_t reg_row1 = reg_row0 + 2U;
      const float2 qh_row_base_lo =
          __bfloat1622float2(load_bf162_shared(row_base_output_smem_ptr + col_lo));
      const float2 qh_row_base_hi =
          __bfloat1622float2(load_bf162_shared(row_base_output_smem_ptr + col_hi));
      const float2 qh_row_hi_lo =
          __bfloat1622float2(load_bf162_shared(row_hi_output_smem_ptr + col_lo));
      const float2 qh_row_hi_hi =
          __bfloat1622float2(load_bf162_shared(row_hi_output_smem_ptr + col_hi));
      const __nv_bfloat162 row_base_lo = combine_output_pair_prescaled_qh(
          ov_reg_lo[reg_row0], ov_reg_lo[reg_row0 + 1U], qh_row_base_lo.x,
          qh_row_base_lo.y, gp_row_base, scale);
      const __nv_bfloat162 row_base_hi = combine_output_pair_prescaled_qh(
          ov_reg_hi[reg_row0], ov_reg_hi[reg_row0 + 1U], qh_row_base_hi.x,
          qh_row_base_hi.y, gp_row_base, scale);
      const __nv_bfloat162 row_hi_lo = combine_output_pair_prescaled_qh(
          ov_reg_lo[reg_row1], ov_reg_lo[reg_row1 + 1U], qh_row_hi_lo.x,
          qh_row_hi_lo.y, gp_row_hi, scale);
      const __nv_bfloat162 row_hi_hi = combine_output_pair_prescaled_qh(
          ov_reg_hi[reg_row1], ov_reg_hi[reg_row1 + 1U], qh_row_hi_hi.x,
          qh_row_hi_hi.y, gp_row_hi, scale);

      if constexpr (FULL_CHUNK) {
        store_bf162_no_allocate(row_base_o_ptr + col_lo, row_base_lo);
        store_bf162_no_allocate(row_base_o_ptr + col_hi, row_base_hi);
        store_bf162_no_allocate(row_hi_o_ptr + col_lo, row_hi_lo);
        store_bf162_no_allocate(row_hi_o_ptr + col_hi, row_hi_hi);
      } else {
        store_bf162_pair_no_allocate_if(row_base_o_ptr + col_lo, row_base_lo,
                                        row_base_o_ptr + col_hi, row_base_hi,
                                        row_base_active);
        store_bf162_pair_no_allocate_if(row_hi_o_ptr + col_lo, row_hi_lo,
                                        row_hi_o_ptr + col_hi, row_hi_hi,
                                        row_hi_active);
      }
    }
  }
}

template <bool FULL_CHUNK>
__device__ __forceinline__ void
store_output_row_pair_from_tmem(uint32_t output_tmem_col,
                                uint32_t qh_tmem_col,
                                nv_bfloat16 *row_base_o_ptr,
                                nv_bfloat16 *row_hi_o_ptr, uint32_t lane_col,
                                uint32_t row_base_active,
                                uint32_t row_hi_active, float gp_row_base,
                                float gp_row_hi, float alpha, float scale) {
#pragma unroll
  for (uint32_t fragment_pair = 0; fragment_pair < BLOCK_V / BLOCK_T;
       ++fragment_pair) {
    const uint32_t col_base = fragment_pair * BLOCK_T;
    float ov_reg_lo[REGS_PER_FRAGMENT];
    float ov_reg_hi[REGS_PER_FRAGMENT];
    float qh_reg_lo[REGS_PER_FRAGMENT];
    float qh_reg_hi[REGS_PER_FRAGMENT];
    load_output_fragment_pair(output_tmem_col, qh_tmem_col, col_base, ov_reg_lo,
                              ov_reg_hi, qh_reg_lo, qh_reg_hi);

#pragma unroll
    for (uint32_t step_pair = 0; step_pair < FRAGMENT_PAIRS; ++step_pair) {
      const uint32_t col_lo = col_base + step_pair * 8U + 2U * lane_col;
      const uint32_t col_hi = col_lo + COLS_PER_FRAGMENT;
      const uint32_t reg_row0 = step_pair * 4U;
      const uint32_t reg_row1 = reg_row0 + 2U;
      const __nv_bfloat162 row_base_lo = combine_output_pair(
          ov_reg_lo[reg_row0], ov_reg_lo[reg_row0 + 1U], qh_reg_lo[reg_row0],
          qh_reg_lo[reg_row0 + 1U], gp_row_base, alpha, scale);
      const __nv_bfloat162 row_base_hi = combine_output_pair(
          ov_reg_hi[reg_row0], ov_reg_hi[reg_row0 + 1U], qh_reg_hi[reg_row0],
          qh_reg_hi[reg_row0 + 1U], gp_row_base, alpha, scale);
      const __nv_bfloat162 row_hi_lo = combine_output_pair(
          ov_reg_lo[reg_row1], ov_reg_lo[reg_row1 + 1U], qh_reg_lo[reg_row1],
          qh_reg_lo[reg_row1 + 1U], gp_row_hi, alpha, scale);
      const __nv_bfloat162 row_hi_hi = combine_output_pair(
          ov_reg_hi[reg_row1], ov_reg_hi[reg_row1 + 1U], qh_reg_hi[reg_row1],
          qh_reg_hi[reg_row1 + 1U], gp_row_hi, alpha, scale);

      if constexpr (FULL_CHUNK) {
        store_bf162_no_allocate(row_base_o_ptr + col_lo, row_base_lo);
        store_bf162_no_allocate(row_base_o_ptr + col_hi, row_base_hi);
        store_bf162_no_allocate(row_hi_o_ptr + col_lo, row_hi_lo);
        store_bf162_no_allocate(row_hi_o_ptr + col_hi, row_hi_hi);
      } else {
        store_bf162_pair_no_allocate_if(row_base_o_ptr + col_lo, row_base_lo,
                                        row_base_o_ptr + col_hi, row_base_hi,
                                        row_base_active);
        store_bf162_pair_no_allocate_if(row_hi_o_ptr + col_lo, row_hi_lo,
                                        row_hi_o_ptr + col_hi, row_hi_hi,
                                        row_hi_active);
      }
    }
  }
}

template <bool FULL_CHUNK>
__device__ __forceinline__ void
store_output_row_pair_from_shared(const nv_bfloat16 *row_base_output_smem_ptr,
                                  const nv_bfloat16 *row_hi_output_smem_ptr,
                                  nv_bfloat16 *row_base_o_ptr,
                                  nv_bfloat16 *row_hi_o_ptr, uint32_t lane_col,
                                  uint32_t row_base_active,
                                  uint32_t row_hi_active) {
#pragma unroll
  for (uint32_t fragment_pair = 0; fragment_pair < BLOCK_V / BLOCK_T;
       ++fragment_pair) {
    const uint32_t col_base = fragment_pair * BLOCK_T;

#pragma unroll
    for (uint32_t step_pair = 0; step_pair < FRAGMENT_PAIRS; ++step_pair) {
      const uint32_t col_lo = col_base + step_pair * 8U + 2U * lane_col;
      const uint32_t col_hi = col_lo + COLS_PER_FRAGMENT;
      const __nv_bfloat162 row_base_lo =
          load_bf162_shared(row_base_output_smem_ptr + col_lo);
      const __nv_bfloat162 row_base_hi =
          load_bf162_shared(row_base_output_smem_ptr + col_hi);
      const __nv_bfloat162 row_hi_lo =
          load_bf162_shared(row_hi_output_smem_ptr + col_lo);
      const __nv_bfloat162 row_hi_hi =
          load_bf162_shared(row_hi_output_smem_ptr + col_hi);

      if constexpr (FULL_CHUNK) {
        store_bf162_no_allocate(row_base_o_ptr + col_lo, row_base_lo);
        store_bf162_no_allocate(row_base_o_ptr + col_hi, row_base_hi);
        store_bf162_no_allocate(row_hi_o_ptr + col_lo, row_hi_lo);
        store_bf162_no_allocate(row_hi_o_ptr + col_hi, row_hi_hi);
      } else {
        store_bf162_pair_no_allocate_if(row_base_o_ptr + col_lo, row_base_lo,
                                        row_base_o_ptr + col_hi, row_base_hi,
                                        row_base_active);
        store_bf162_pair_no_allocate_if(row_hi_o_ptr + col_lo, row_hi_lo,
                                        row_hi_o_ptr + col_hi, row_hi_hi,
                                        row_hi_active);
      }
    }
  }
}

__device__ __forceinline__ void
tma_load_qk_stage(uint32_t q_stage_smem, uint32_t k_stage_smem,
                  const CUtensorMap *q_tmap, const CUtensorMap *k_tmap,
                  int32_t chunk_start_i32, uint32_t q_head_id,
                  uint32_t k_head_id, uint32_t qk_tma_barrier) {
  tma_load_4d(q_stage_smem, q_tmap, 0, chunk_start_i32, 0, q_head_id,
              qk_tma_barrier, EVICT_FIRST);
  tma_load_4d(k_stage_smem, k_tmap, 0, chunk_start_i32, 0, k_head_id,
              qk_tma_barrier, EVICT_FIRST);
  mbarrier_arrive_expect_tx(qk_tma_barrier, Q_SMEM_SIZE + K_SMEM_SIZE);
}

__device__ __forceinline__ bool mbarrier_try_wait_once(uint32_t mbar_addr,
                                                       uint32_t phase) {
  uint32_t ready;
  constexpr uint32_t ticks = 1U;
  asm volatile(
      "{\n\t"
      ".reg .pred P1;\n\t"
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%1], %2, "
      "%3;\n\t"
      "selp.u32 %0, 1, 0, P1;\n\t"
      "}"
      : "=r"(ready)
      : "r"(mbar_addr), "r"(phase), "r"(ticks));
  return ready != 0U;
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

__global__ __block_size__((NUM_THREADS, 1, 1)) void o_v2_kernel_cutlass(
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
  const uint32_t output_smem = smem + OFFSET_OUTPUT;
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
  const uint32_t qk_stage_release_barriers = smem + OFFSET_QK_STAGE_RELEASE_BAR;
  const uint32_t qk_consumed_barrier = smem + OFFSET_QK_CONSUMED_BAR;
  const uint32_t attn_ready_barriers = smem + OFFSET_ATTN_READY_BAR;
  const uint32_t attn_release_barriers = smem + OFFSET_ATTN_RELEASE_BAR;
  const uint32_t qh_mma_barrier = smem + OFFSET_QH_MMA_BAR;
  const uint32_t ov_mma_barrier = smem + OFFSET_OV_MMA_BAR;
  const uint32_t qh_release_barriers = smem + OFFSET_QH_RELEASE_BAR;
  const uint32_t output_release_barriers = smem + OFFSET_OUTPUT_RELEASE_BAR;
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
  nv_bfloat16 *output_smem_ptr =
      reinterpret_cast<nv_bfloat16 *>(smem_ptr + (output_smem - smem));
  (void)output_smem_ptr;
  (void)qk_spill_smem;

  if (warp_id == TMA_WARP) {
    if (elect_sync()) {
#pragma unroll
      for (uint32_t qk_stage = 0; qk_stage < QK_STAGE_COUNT; ++qk_stage) {
        mbarrier_init(qk_tma_barriers + qk_stage * 8U, 1);
        mbarrier_init(qk_stage_release_barriers + qk_stage * 8U, 1);
      }
#pragma unroll
      for (uint32_t v_stage_index = 0; v_stage_index < V_STAGE_COUNT;
           ++v_stage_index) {
        mbarrier_init(v_tma_barrier + v_stage_index * 8U, 1);
      }
#pragma unroll
      for (uint32_t h_stage_index = 0; h_stage_index < H_STAGE_COUNT;
           ++h_stage_index) {
        mbarrier_init(h_tma_barrier + h_stage_index * 8U, 1);
      }
      mbarrier_init(mma_barrier, 1);
      mbarrier_init(qk_consumed_barrier, NUM_CUDA_WARPS);
#pragma unroll
      for (uint32_t attn_stage = 0; attn_stage < ATTN_STAGE_COUNT;
           ++attn_stage) {
        mbarrier_init(attn_ready_barriers + attn_stage * 8U, NUM_CUDA_WARPS);
        mbarrier_init(attn_release_barriers + attn_stage * 8U, 1);
      }
#pragma unroll
      for (uint32_t h_stage_index = 0; h_stage_index < H_STAGE_COUNT;
           ++h_stage_index) {
        mbarrier_init(qh_mma_barrier + h_stage_index * 8U, 1);
      }
#pragma unroll
      for (uint32_t v_stage_index = 0; v_stage_index < V_STAGE_COUNT;
           ++v_stage_index) {
        mbarrier_init(ov_mma_barrier + v_stage_index * 8U, 1);
      }
#pragma unroll
      for (uint32_t h_stage_index = 0; h_stage_index < H_STAGE_COUNT;
           ++h_stage_index) {
        mbarrier_init(qh_release_barriers + h_stage_index * 8U,
                      NUM_CUDA_WARPS);
      }
      mbarrier_init(output_release_barriers + 0U * 8U, NUM_CUDA_WARPS);
      mbarrier_init(output_release_barriers + 1U * 8U, NUM_CUDA_WARPS);
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
    uint32_t global_chunk_id = blockIdx.y;

    if (global_chunk_id < total_num_chunks_u && elect_sync()) {
      const int2 chunk_meta =
          reinterpret_cast<const int2 *>(chunk_indices_ptr)[global_chunk_id];
      const uint32_t seq_id = static_cast<uint32_t>(chunk_meta.x);
      const uint32_t chunk_id = static_cast<uint32_t>(chunk_meta.y);
      const int32_t chunk_start_i32 = static_cast<int32_t>(
          cu_seqlens_ptr[seq_id] +
          static_cast<int64_t>(chunk_id) * static_cast<int64_t>(BLOCK_T));
      const int32_t v_chunk_start_i32 =
          static_cast<int32_t>(global_chunk_id * BLOCK_T);

      tma_load_qk_stage(q_smem, k_smem, &q_tmap, &k_tmap, chunk_start_i32,
                        q_head_id, k_head_id, qk_tma_barriers);

#pragma unroll
      for (uint32_t head_offset = 0; head_offset < HEADS_PER_QK_HEAD;
           ++head_offset) {
        const uint32_t head_id = head_id_base + head_offset;
        const uint32_t h_outer = global_chunk_id * NUM_OUTPUT_HEADS + head_id;
        const uint32_t v_stage_index = get_v_stage_index(head_offset, 0U);
        const uint32_t h_stage_index = get_h_stage_index(head_offset, 0U);
        tma_load_4d(v_smem + v_stage_index * V_SMEM_SIZE, &v_tmap, 0,
                    v_chunk_start_i32, v_tile, head_id,
                    v_tma_barrier + v_stage_index * 8U, EVICT_FIRST);
        tma_load_4d(h_smem + h_stage_index * H_SMEM_SIZE, &h_tmap, 0, v_start,
                    0, h_outer, h_tma_barrier + h_stage_index * 8U,
                    EVICT_FIRST);
        mbarrier_arrive_expect_tx(v_tma_barrier + v_stage_index * 8U,
                                  V_SMEM_SIZE);
        mbarrier_arrive_expect_tx(h_tma_barrier + h_stage_index * 8U,
                                  H_SMEM_SIZE);
      }
    }

    if (elect_sync()) {
      uint32_t next_qk_global_chunk_id = global_chunk_id + gridDim.y;
      uint32_t next_h_global_chunk_id = next_qk_global_chunk_id;
      uint32_t next_v_global_chunk_id = next_qk_global_chunk_id;
      uint32_t next_qk_chunk_iter = 1U;
      uint32_t next_h_chunk_iter = 1U;
      uint32_t next_v_chunk_iter = 1U;

      for (; next_qk_chunk_iter < QK_STAGE_COUNT &&
             next_qk_global_chunk_id < total_num_chunks_u;
           ++next_qk_chunk_iter, next_qk_global_chunk_id += gridDim.y) {
        const int2 chunk_meta = reinterpret_cast<const int2 *>(
            chunk_indices_ptr)[next_qk_global_chunk_id];
        const uint32_t seq_id = static_cast<uint32_t>(chunk_meta.x);
        const uint32_t chunk_id = static_cast<uint32_t>(chunk_meta.y);
        const uint32_t qk_stage = next_qk_chunk_iter % QK_STAGE_COUNT;
        const int32_t chunk_start_i32 = static_cast<int32_t>(
            cu_seqlens_ptr[seq_id] +
            static_cast<int64_t>(chunk_id) * static_cast<int64_t>(BLOCK_T));

        tma_load_qk_stage(q_smem + qk_stage * Q_SMEM_SIZE,
                          k_smem + qk_stage * K_SMEM_SIZE, &q_tmap, &k_tmap,
                          chunk_start_i32, q_head_id, k_head_id,
                          qk_tma_barriers + qk_stage * 8U);
      }

      for (; next_h_chunk_iter < H_RING_STAGE_COUNT &&
             next_h_global_chunk_id < total_num_chunks_u;
           ++next_h_chunk_iter, next_h_global_chunk_id += gridDim.y) {
        const int2 chunk_meta = reinterpret_cast<const int2 *>(
            chunk_indices_ptr)[next_h_global_chunk_id];
        const uint32_t h_stage = next_h_chunk_iter % H_RING_STAGE_COUNT;

#pragma unroll
        for (uint32_t head_offset = 0; head_offset < HEADS_PER_QK_HEAD;
             ++head_offset) {
          const uint32_t head_id = head_id_base + head_offset;
          const uint32_t h_outer =
              next_h_global_chunk_id * NUM_OUTPUT_HEADS + head_id;
          const uint32_t h_stage_index =
              get_h_stage_index(head_offset, h_stage);
          tma_load_4d(h_smem + h_stage_index * H_SMEM_SIZE, &h_tmap, 0,
                      v_start, 0, h_outer,
                      h_tma_barrier + h_stage_index * 8U, EVICT_FIRST);
          mbarrier_arrive_expect_tx(h_tma_barrier + h_stage_index * 8U,
                                    H_SMEM_SIZE);
        }
      }

      for (; next_v_chunk_iter < V_RING_STAGE_COUNT &&
             next_v_global_chunk_id < total_num_chunks_u;
           ++next_v_chunk_iter, next_v_global_chunk_id += gridDim.y) {
        const uint32_t v_stage = next_v_chunk_iter % V_RING_STAGE_COUNT;
        const int32_t v_chunk_start_i32 =
            static_cast<int32_t>(next_v_global_chunk_id * BLOCK_T);

#pragma unroll
        for (uint32_t head_offset = 0; head_offset < HEADS_PER_QK_HEAD;
             ++head_offset) {
          const uint32_t head_id = head_id_base + head_offset;
          const uint32_t v_stage_index =
              get_v_stage_index(head_offset, v_stage);
          tma_load_4d(v_smem + v_stage_index * V_SMEM_SIZE, &v_tmap, 0,
                      v_chunk_start_i32, v_tile, head_id,
                      v_tma_barrier + v_stage_index * 8U, EVICT_FIRST);
          mbarrier_arrive_expect_tx(v_tma_barrier + v_stage_index * 8U,
                                    V_SMEM_SIZE);
        }
      }

      while (next_qk_global_chunk_id < total_num_chunks_u ||
             next_h_global_chunk_id < total_num_chunks_u ||
             next_v_global_chunk_id < total_num_chunks_u) {
        bool progressed = false;

        if (next_qk_global_chunk_id < total_num_chunks_u) {
          const uint32_t qk_stage = next_qk_chunk_iter % QK_STAGE_COUNT;
          const uint32_t qk_stage_release_phase =
              ((next_qk_chunk_iter / QK_STAGE_COUNT) & 1U) ^ 1U;

          if (mbarrier_try_wait_once(qk_stage_release_barriers +
                                         qk_stage * 8U,
                                     qk_stage_release_phase)) {
            const int2 chunk_meta = reinterpret_cast<const int2 *>(
                chunk_indices_ptr)[next_qk_global_chunk_id];
            const uint32_t seq_id = static_cast<uint32_t>(chunk_meta.x);
            const uint32_t chunk_id = static_cast<uint32_t>(chunk_meta.y);
            const int32_t chunk_start_i32 = static_cast<int32_t>(
                cu_seqlens_ptr[seq_id] +
                static_cast<int64_t>(chunk_id) * static_cast<int64_t>(BLOCK_T));

            tma_load_qk_stage(q_smem + qk_stage * Q_SMEM_SIZE,
                              k_smem + qk_stage * K_SMEM_SIZE, &q_tmap,
                              &k_tmap, chunk_start_i32, q_head_id, k_head_id,
                              qk_tma_barriers + qk_stage * 8U);

            next_qk_global_chunk_id += gridDim.y;
            ++next_qk_chunk_iter;
            progressed = true;
          }
        }

        if (next_h_global_chunk_id < total_num_chunks_u) {
          const uint32_t h_stage = next_h_chunk_iter % H_RING_STAGE_COUNT;
          const uint32_t h_stage_reuse_phase =
              ((next_h_chunk_iter / H_RING_STAGE_COUNT) & 1U) ^ 1U;

          bool h_ready = true;
#pragma unroll
          for (uint32_t head_offset = 0; head_offset < HEADS_PER_QK_HEAD;
               ++head_offset) {
            const uint32_t h_stage_index =
                get_h_stage_index(head_offset, h_stage);
            h_ready = h_ready &&
                      mbarrier_try_wait_once(
                          qh_mma_barrier + h_stage_index * 8U,
                          h_stage_reuse_phase);
          }

          if (h_ready) {
            const int2 chunk_meta = reinterpret_cast<const int2 *>(
                chunk_indices_ptr)[next_h_global_chunk_id];

#pragma unroll
            for (uint32_t head_offset = 0; head_offset < HEADS_PER_QK_HEAD;
                 ++head_offset) {
              const uint32_t head_id = head_id_base + head_offset;
              const uint32_t h_outer =
                  next_h_global_chunk_id * NUM_OUTPUT_HEADS + head_id;
              const uint32_t h_stage_index =
                  get_h_stage_index(head_offset, h_stage);
              tma_load_4d(h_smem + h_stage_index * H_SMEM_SIZE, &h_tmap, 0,
                          v_start, 0, h_outer,
                          h_tma_barrier + h_stage_index * 8U, EVICT_FIRST);
              mbarrier_arrive_expect_tx(h_tma_barrier + h_stage_index * 8U,
                                        H_SMEM_SIZE);
            }

            next_h_global_chunk_id += gridDim.y;
            ++next_h_chunk_iter;
            progressed = true;
          }
        }

        if (next_v_global_chunk_id < total_num_chunks_u) {
          const uint32_t v_stage = next_v_chunk_iter % V_RING_STAGE_COUNT;
          const uint32_t v_stage_reuse_phase =
              ((next_v_chunk_iter / V_RING_STAGE_COUNT) & 1U) ^ 1U;

          bool v_ready = true;
#pragma unroll
          for (uint32_t head_offset = 0; head_offset < HEADS_PER_QK_HEAD;
               ++head_offset) {
            const uint32_t v_stage_index =
                get_v_stage_index(head_offset, v_stage);
            v_ready = v_ready &&
                      mbarrier_try_wait_once(
                          ov_mma_barrier + v_stage_index * 8U,
                          v_stage_reuse_phase);
          }

          if (v_ready) {
            const int32_t v_chunk_start_i32 =
                static_cast<int32_t>(next_v_global_chunk_id * BLOCK_T);

#pragma unroll
            for (uint32_t head_offset = 0; head_offset < HEADS_PER_QK_HEAD;
                 ++head_offset) {
              const uint32_t head_id = head_id_base + head_offset;
              const uint32_t v_stage_index =
                  get_v_stage_index(head_offset, v_stage);
              tma_load_4d(v_smem + v_stage_index * V_SMEM_SIZE, &v_tmap, 0,
                          v_chunk_start_i32, v_tile, head_id,
                          v_tma_barrier + v_stage_index * 8U, EVICT_FIRST);
              mbarrier_arrive_expect_tx(v_tma_barrier + v_stage_index * 8U,
                                        V_SMEM_SIZE);
            }

            next_v_global_chunk_id += gridDim.y;
            ++next_v_chunk_iter;
            progressed = true;
          }
        }

        if (!progressed) {
          if (next_qk_global_chunk_id < total_num_chunks_u) {
            const uint32_t qk_stage = next_qk_chunk_iter % QK_STAGE_COUNT;
            const uint32_t qk_stage_release_phase =
                ((next_qk_chunk_iter / QK_STAGE_COUNT) & 1U) ^ 1U;
            if (!mbarrier_try_wait_once(qk_stage_release_barriers +
                                            qk_stage * 8U,
                                        qk_stage_release_phase)) {
              mbarrier_wait(qk_stage_release_barriers + qk_stage * 8U,
                            qk_stage_release_phase);
            }
          } else if (next_h_global_chunk_id < total_num_chunks_u) {
            const uint32_t h_stage = next_h_chunk_iter % H_RING_STAGE_COUNT;
            const uint32_t h_stage_reuse_phase =
                ((next_h_chunk_iter / H_RING_STAGE_COUNT) & 1U) ^ 1U;

#pragma unroll
            for (uint32_t head_offset = 0; head_offset < HEADS_PER_QK_HEAD;
                 ++head_offset) {
              const uint32_t h_stage_index =
                  get_h_stage_index(head_offset, h_stage);
              if (!mbarrier_try_wait_once(qh_mma_barrier +
                                              h_stage_index * 8U,
                                          h_stage_reuse_phase)) {
                mbarrier_wait(qh_mma_barrier + h_stage_index * 8U,
                              h_stage_reuse_phase);
                break;
              }
            }
          } else if (next_v_global_chunk_id < total_num_chunks_u) {
            const uint32_t v_stage = next_v_chunk_iter % V_RING_STAGE_COUNT;
            const uint32_t v_stage_reuse_phase =
                ((next_v_chunk_iter / V_RING_STAGE_COUNT) & 1U) ^ 1U;

#pragma unroll
            for (uint32_t head_offset = 0; head_offset < HEADS_PER_QK_HEAD;
                 ++head_offset) {
              const uint32_t v_stage_index =
                  get_v_stage_index(head_offset, v_stage);
              if (!mbarrier_try_wait_once(ov_mma_barrier +
                                              v_stage_index * 8U,
                                          v_stage_reuse_phase)) {
                mbarrier_wait(ov_mma_barrier + v_stage_index * 8U,
                              v_stage_reuse_phase);
                break;
              }
            }
          }
        }
      }
    }
  } else if (warp_id == MMA_WARP) {
    uint32_t global_chunk_id = blockIdx.y;

    if (global_chunk_id < total_num_chunks_u && elect_sync()) {
      mbarrier_wait(qk_tma_barriers, TMA_PHASE);
      tcgen05_fence_after_thread_sync();
      mma_swizzled<NUM_SWIZZLE_ATOMS>(QK_TMEM_COL, q_smem, k_smem);
      tcgen05_commit(mma_barrier);

#pragma unroll
      for (uint32_t head_offset = 0; head_offset < HEADS_PER_QK_HEAD;
           ++head_offset) {
        const uint32_t h_stage_index = get_h_stage_index(head_offset, 0U);
        const uint32_t h_stage_smem = h_smem + h_stage_index * H_SMEM_SIZE;
        const uint32_t h_stage_barrier = h_tma_barrier + h_stage_index * 8U;

        mbarrier_wait(h_stage_barrier, 0U);
        tcgen05_fence_after_thread_sync();
        mma_swizzled_qh_64x128(get_qh_tmem_col(head_offset, 0U), q_smem,
                               h_stage_smem);
        tcgen05_commit(qh_mma_barrier + h_stage_index * 8U);
      }

      mbarrier_arrive(qk_stage_release_barriers);

#pragma unroll
      for (uint32_t head_offset = 0; head_offset < HEADS_PER_QK_HEAD;
           ++head_offset) {
        const uint32_t v_stage_index = get_v_stage_index(head_offset, 0U);
        const uint32_t v_stage_smem = v_smem + v_stage_index * V_SMEM_SIZE;
        const uint32_t attn_stage_index = head_offset;
        const uint32_t attn_stage_smem =
            attn_smem + attn_stage_index * ATTN_SMEM_SIZE;
        const uint32_t attn_stage_barrier =
            attn_ready_barriers + attn_stage_index * 8U;
        const uint32_t v_stage_barrier = v_tma_barrier + v_stage_index * 8U;

        mbarrier_wait(v_stage_barrier, 0U);
        mbarrier_wait(attn_stage_barrier, 0U);
        tcgen05_fence_after_thread_sync();
        mma_attn_v_mmajor_64x128(get_output_tmem_col(head_offset),
                                 attn_stage_smem, v_stage_smem);
        tcgen05_commit(ov_mma_barrier + v_stage_index * 8U);
        mbarrier_arrive(attn_release_barriers + attn_stage_index * 8U);
      }

    }

    global_chunk_id += gridDim.y;
    for (uint32_t chunk_iter = 1U; global_chunk_id < total_num_chunks_u;
         global_chunk_id += gridDim.y, ++chunk_iter) {
      const uint32_t chunk_parity = chunk_iter & 1U;
      const uint32_t h_stage = chunk_iter % H_RING_STAGE_COUNT;
      const uint32_t h_stage_phase =
          (chunk_iter / H_RING_STAGE_COUNT) & 1U;
      const uint32_t v_stage = chunk_iter % V_RING_STAGE_COUNT;
      const uint32_t v_stage_phase =
          (chunk_iter / V_RING_STAGE_COUNT) & 1U;
      const uint32_t qk_stage = chunk_iter % QK_STAGE_COUNT;
      const uint32_t q_stage_smem = q_smem + qk_stage * Q_SMEM_SIZE;
      const uint32_t k_stage_smem = k_smem + qk_stage * K_SMEM_SIZE;
      const uint32_t qk_tma_barrier = qk_tma_barriers + qk_stage * 8U;
      const uint32_t qk_tma_stage_phase = (chunk_iter / QK_STAGE_COUNT) & 1U;

      if (elect_sync()) {
        mbarrier_wait(qk_tma_barrier, TMA_PHASE ^ qk_tma_stage_phase);
        tcgen05_fence_after_thread_sync();
        mma_swizzled<NUM_SWIZZLE_ATOMS>(QK_TMEM_COL, q_stage_smem,
                                        k_stage_smem);
        tcgen05_commit(mma_barrier);
      }

#pragma unroll
      for (uint32_t head_offset = 0; head_offset < HEADS_PER_QK_HEAD;
           ++head_offset) {
        const uint32_t h_stage_index =
            get_h_stage_index(head_offset, h_stage);
        const uint32_t h_stage_smem = h_smem + h_stage_index * H_SMEM_SIZE;
        const uint32_t h_stage_barrier = h_tma_barrier + h_stage_index * 8U;

        if (elect_sync()) {
          if (chunk_iter >= H_RING_STAGE_COUNT) {
            mbarrier_wait(qh_release_barriers + h_stage_index * 8U,
                          h_stage_phase ^ 1U);
          }
          mbarrier_wait(h_stage_barrier, h_stage_phase);
          tcgen05_fence_after_thread_sync();
          mma_swizzled_qh_64x128(get_qh_tmem_col(head_offset, h_stage),
                                 q_stage_smem, h_stage_smem);
          tcgen05_commit(qh_mma_barrier + h_stage_index * 8U);
        }
      }

      if (elect_sync()) {
        mbarrier_arrive(qk_stage_release_barriers + qk_stage * 8U);
      }

#pragma unroll
      for (uint32_t head_offset = 0; head_offset < HEADS_PER_QK_HEAD;
           ++head_offset) {
        const uint32_t v_stage_index =
            get_v_stage_index(head_offset, v_stage);
        const uint32_t v_stage_smem = v_smem + v_stage_index * V_SMEM_SIZE;
        const uint32_t attn_stage_smem =
            attn_smem + head_offset * ATTN_SMEM_SIZE;
        const uint32_t attn_stage_barrier =
            attn_ready_barriers + head_offset * 8U;
        const uint32_t v_stage_barrier = v_tma_barrier + v_stage_index * 8U;

        if (elect_sync()) {
          mbarrier_wait(output_release_barriers + head_offset * 8U,
                        chunk_parity ^ 1U);
          mbarrier_wait(v_stage_barrier, v_stage_phase);
          mbarrier_wait(attn_stage_barrier, chunk_parity);
          tcgen05_fence_after_thread_sync();
          mma_attn_v_mmajor_64x128(get_output_tmem_col(head_offset),
                                   attn_stage_smem, v_stage_smem);
          tcgen05_commit(ov_mma_barrier + v_stage_index * 8U);
          mbarrier_arrive(attn_release_barriers + head_offset * 8U);
        }
      }

      if (elect_sync()) {
        mbarrier_wait(qk_consumed_barrier, chunk_parity);
      }
    }
  } else {
    uint32_t global_chunk_id = blockIdx.y;

    if (global_chunk_id < total_num_chunks_u) {
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
      const uint32_t qk_phase = 0U;

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
      tcgen05_fence_before_thread_sync();
      __syncwarp();
      if (elect_sync()) {
        mbarrier_arrive(qk_consumed_barrier);
      }

#pragma unroll
      for (uint32_t head_offset = 0; head_offset < HEADS_PER_QK_HEAD;
           ++head_offset) {
        const uint32_t attn_stage_index = head_offset;
        const uint32_t attn_stage_smem =
            attn_smem + attn_stage_index * ATTN_SMEM_SIZE;
        const float *g_neg_head_ptr = g_neg_smem_ptr + head_offset * BLOCK_T;

        materialize_attn_stage_from_regs(smem_ptr, smem, attn_stage_smem,
                                         qk_reg_lo, qk_reg_hi, g_neg_head_ptr,
                                         row_base, row_hi, row_base_limit,
                                         row_hi_limit, lane_col);
        fence_proxy_async_shared_cta();
        __syncwarp();
        if (elect_sync()) {
          mbarrier_arrive(attn_ready_barriers + attn_stage_index * 8U);
        }
      }

#pragma unroll
      for (uint32_t head_offset = 0; head_offset < HEADS_PER_QK_HEAD;
           ++head_offset) {
        const uint32_t h_stage_index = get_h_stage_index(head_offset, 0U);
        const uint32_t v_stage_index = get_v_stage_index(head_offset, 0U);
        const uint32_t head_id = head_id_base + head_offset;
        const uint32_t output_tmem_col = get_output_tmem_col(head_offset);
        const uint32_t qh_tmem_col = get_qh_tmem_col(head_offset, 0U);
        const float *gp_head_ptr = g_pos_smem_ptr + head_offset * BLOCK_T;
        const float gp_row_base = gp_head_ptr[row_base];
        const float gp_row_hi = gp_head_ptr[row_hi];
        const float alpha = g_alpha_smem_ptr[head_offset];

        mbarrier_wait(ov_mma_barrier + v_stage_index * 8U, 0U);
        mbarrier_wait(qh_mma_barrier + h_stage_index * 8U, 0U);
        tcgen05_fence_after_thread_sync();

        nv_bfloat16 *row_base_o_ptr =
            o_ptr + (((chunk_start + static_cast<int64_t>(row_base)) *
                          NUM_OUTPUT_HEADS +
                      head_id) *
                         VALUE_DIM +
                     v_start);
        nv_bfloat16 *row_hi_o_ptr = row_base_o_ptr + ROW_PAIR_OUTPUT_STRIDE;

        if (full_chunk) {
          store_output_row_pair_from_tmem<true>(
              output_tmem_col, qh_tmem_col, row_base_o_ptr, row_hi_o_ptr,
              lane_col, 0U, 0U, gp_row_base, gp_row_hi, alpha, scale);
        } else {
          const uint32_t row_base_active = row_base < chunk_len;
          const uint32_t row_hi_active = row_hi < chunk_len;
          store_output_row_pair_from_tmem<false>(
              output_tmem_col, qh_tmem_col, row_base_o_ptr, row_hi_o_ptr,
              lane_col, row_base_active, row_hi_active, gp_row_base,
              gp_row_hi, alpha, scale);
        }

        tcgen05_fence_before_thread_sync();
        __syncwarp();
        if (elect_sync()) {
          mbarrier_arrive(qh_release_barriers + h_stage_index * 8U);
          mbarrier_arrive(output_release_barriers + head_offset * 8U);
        }
      }

      global_chunk_id += gridDim.y;
    }

    for (uint32_t chunk_iter = 1U; global_chunk_id < total_num_chunks_u;
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
      const uint32_t h_stage = chunk_iter % H_RING_STAGE_COUNT;
      const uint32_t h_stage_phase =
          (chunk_iter / H_RING_STAGE_COUNT) & 1U;
      const uint32_t v_stage = chunk_iter % V_RING_STAGE_COUNT;
      const uint32_t v_stage_phase =
          (chunk_iter / V_RING_STAGE_COUNT) & 1U;

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
      tcgen05_fence_before_thread_sync();
      __syncwarp();
      if (elect_sync()) {
        mbarrier_arrive(qk_consumed_barrier);
      }

#pragma unroll
      for (uint32_t head_offset = 0; head_offset < HEADS_PER_QK_HEAD;
           ++head_offset) {
        const uint32_t attn_stage_smem =
            attn_smem + head_offset * ATTN_SMEM_SIZE;
        const float *g_neg_head_ptr = g_neg_smem_ptr + head_offset * BLOCK_T;

        mbarrier_wait(attn_release_barriers + head_offset * 8U, qk_phase ^ 1U);

        materialize_attn_stage_from_regs(smem_ptr, smem, attn_stage_smem,
                                         qk_reg_lo, qk_reg_hi, g_neg_head_ptr,
                                         row_base, row_hi, row_base_limit,
                                         row_hi_limit, lane_col);
        fence_proxy_async_shared_cta();
        __syncwarp();
        if (elect_sync()) {
          mbarrier_arrive(attn_ready_barriers + head_offset * 8U);
        }
      }

#pragma unroll
      for (uint32_t head_offset = 0; head_offset < HEADS_PER_QK_HEAD;
           ++head_offset) {
        const uint32_t h_stage_index =
            get_h_stage_index(head_offset, h_stage);
        const uint32_t v_stage_index =
            get_v_stage_index(head_offset, v_stage);
        const uint32_t head_id = head_id_base + head_offset;
        const uint32_t output_tmem_col = get_output_tmem_col(head_offset);
        const uint32_t qh_tmem_col = get_qh_tmem_col(head_offset, h_stage);
        const float *gp_head_ptr = g_pos_smem_ptr + head_offset * BLOCK_T;
        const float gp_row_base = gp_head_ptr[row_base];
        const float gp_row_hi = gp_head_ptr[row_hi];
        const float alpha = g_alpha_smem_ptr[head_offset];

        mbarrier_wait(ov_mma_barrier + v_stage_index * 8U, v_stage_phase);
        mbarrier_wait(qh_mma_barrier + h_stage_index * 8U, h_stage_phase);
        tcgen05_fence_after_thread_sync();

        nv_bfloat16 *row_base_o_ptr =
            o_ptr + (((chunk_start + static_cast<int64_t>(row_base)) *
                          NUM_OUTPUT_HEADS +
                      head_id) *
                         VALUE_DIM +
                     v_start);
        nv_bfloat16 *row_hi_o_ptr = row_base_o_ptr + ROW_PAIR_OUTPUT_STRIDE;

        if (full_chunk) {
          store_output_row_pair_from_tmem<true>(
              output_tmem_col, qh_tmem_col, row_base_o_ptr, row_hi_o_ptr,
              lane_col, 0U, 0U, gp_row_base, gp_row_hi, alpha, scale);
        } else {
          const uint32_t row_base_active = row_base < chunk_len;
          const uint32_t row_hi_active = row_hi < chunk_len;
          store_output_row_pair_from_tmem<false>(
              output_tmem_col, qh_tmem_col, row_base_o_ptr, row_hi_o_ptr,
              lane_col, row_base_active, row_hi_active, gp_row_base,
              gp_row_hi, alpha, scale);
        }

        tcgen05_fence_before_thread_sync();
        __syncwarp();
        if (elect_sync()) {
          mbarrier_arrive(qh_release_barriers + h_stage_index * 8U);
          mbarrier_arrive(output_release_barriers + head_offset * 8U);
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

void launch_o_v2(TensorView q_chunks, TensorView k_chunks, TensorView v_new,
                 TensorView h, TensorView g_cu, TensorView o,
                 TensorView cu_seqlens, TensorView chunk_indices,
                 TensorView total_chunks, double scale) {
  const uint32_t total_num_tokens = static_cast<uint32_t>(q_chunks.size(0));
  const uint32_t total_num_chunks_capacity =
      static_cast<uint32_t>(v_new.size(0));
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

  auto kernel = o_v2_kernel_cutlass;
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

void o_v2(TensorView q_chunks, TensorView k_chunks, TensorView v_new,
          TensorView h, TensorView g_cu, TensorView o, TensorView cu_seqlens,
          TensorView chunk_indices, TensorView total_chunks, double scale) {
  launch_o_v2(q_chunks, k_chunks, v_new, h, g_cu, o, cu_seqlens, chunk_indices,
              total_chunks, scale);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(o_v2, o_v2);
