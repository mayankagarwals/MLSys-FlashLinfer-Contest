#include "cuda_utils.h"

#include <cstdint>
#include <cuda_bf16.h>
#include <math.h>

constexpr uint32_t BLOCK_T = 64U;
constexpr uint32_t HEAD_DIM = 128U;
constexpr uint32_t VALUE_DIM = 128U;
constexpr uint32_t BLOCK_V = 128U;

constexpr uint32_t NUM_OUTPUT_HEADS = 8U;
constexpr uint32_t NUM_QK_HEADS = 4U;
constexpr uint32_t HEADS_PER_QK_HEAD = NUM_OUTPUT_HEADS / NUM_QK_HEADS;
static_assert(HEADS_PER_QK_HEAD == 2U);

constexpr uint32_t NUM_QK_STAGES = 1U;
constexpr uint32_t NUM_H_STAGES = 2U;
constexpr uint32_t NUM_V_STAGES = 2U;
constexpr uint32_t INITIAL_QK_PREFETCH_STAGES = 1U;
constexpr uint32_t INITIAL_H_PREFETCH_STAGES = 1U;
constexpr uint32_t INITIAL_V_PREFETCH_STAGES = 1U;
constexpr uint32_t NUM_CUDA_WARPS = 4U;
constexpr uint32_t TMA_WARP = NUM_CUDA_WARPS;
constexpr uint32_t QK_MMA_WARP = TMA_WARP + 1U;
constexpr uint32_t QH_MMA_WARP = TMA_WARP + 2U;
constexpr uint32_t NUM_WARPS = NUM_CUDA_WARPS + 3U;
constexpr uint32_t WARP_SIZE = 32U;
constexpr uint32_t NUM_THREADS = NUM_WARPS * WARP_SIZE;

constexpr uint32_t ROWS_PER_WARP = 16U;
constexpr uint32_t COLS_PER_FRAGMENT = 32U;
constexpr uint32_t REGS_PER_FRAGMENT = 16U;
constexpr uint32_t ROW_PAIR_STRIDE = 8U;
constexpr uint32_t ROW_PAIR_OUTPUT_STRIDE =
    ROW_PAIR_STRIDE * NUM_OUTPUT_HEADS * VALUE_DIM;
constexpr uint32_t LANES_PER_ROW_GROUP = 4U;
constexpr uint32_t FRAGMENT_STEPS = 8U;
constexpr uint32_t FRAGMENT_PAIRS = FRAGMENT_STEPS / 2U;

constexpr uint32_t MMA_M = BLOCK_T;
constexpr uint32_t MMA_N = BLOCK_T;
constexpr uint32_t MMA_K = 16U;
constexpr uint32_t NUM_MMA_STEPS = BLOCK_T / MMA_K;
constexpr uint32_t BYTES_ONE_MMA = MMA_K * sizeof(nv_bfloat16);

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
constexpr uint32_t NUM_QK_ATOMS = HEAD_DIM / SWIZZLE_WIDTH;
constexpr uint32_t NUM_H_ATOMS = HEAD_DIM / SWIZZLE_WIDTH;
constexpr uint32_t NUM_V_ATOMS = VALUE_DIM / SWIZZLE_WIDTH;
constexpr uint32_t NUM_SWIZZLE_ATOMS = NUM_QK_ATOMS;
static_assert(NUM_QK_ATOMS == 2U);
static_assert(NUM_H_ATOMS == 2U);
static_assert(NUM_V_ATOMS == 2U);

constexpr uint32_t align_up(uint32_t value, uint32_t align) {
  return (value + align - 1U) & ~(align - 1U);
}

constexpr uint32_t Q_STAGE_SMEM_SIZE = BLOCK_T * HEAD_DIM * sizeof(nv_bfloat16);
constexpr uint32_t K_STAGE_SMEM_SIZE = BLOCK_T * HEAD_DIM * sizeof(nv_bfloat16);
constexpr uint32_t V_STAGE_SMEM_SIZE =
    BLOCK_T * VALUE_DIM * sizeof(nv_bfloat16);
constexpr uint32_t H_STAGE_SMEM_SIZE = BLOCK_V * HEAD_DIM * sizeof(nv_bfloat16);
constexpr uint32_t TMA_STAGE_BYTES = Q_STAGE_SMEM_SIZE + K_STAGE_SMEM_SIZE +
                                     V_STAGE_SMEM_SIZE + H_STAGE_SMEM_SIZE;

constexpr uint32_t G_RAW_SMEM_BYTES = BLOCK_T * sizeof(float);
constexpr uint32_t G_CENTER_SMEM_BYTES = sizeof(float);
constexpr uint32_t G_POS_SMEM_BYTES = BLOCK_T * sizeof(float);
constexpr uint32_t G_ALPHA_SMEM_BYTES = sizeof(float);
constexpr uint32_t G_NEG_SMEM_BYTES = BLOCK_T * sizeof(float);
constexpr uint32_t G_REDUCE_PARTIAL_SMEM_BYTES = NUM_CUDA_WARPS * sizeof(float);
constexpr uint32_t Q_SMEM_BYTES = NUM_QK_STAGES * Q_STAGE_SMEM_SIZE;
constexpr uint32_t K_SMEM_BYTES = NUM_QK_STAGES * K_STAGE_SMEM_SIZE;
constexpr uint32_t V_SMEM_BYTES = NUM_V_STAGES * V_STAGE_SMEM_SIZE;
constexpr uint32_t H_SMEM_BYTES = NUM_H_STAGES * H_STAGE_SMEM_SIZE;
constexpr uint32_t QK_TMA_BARRIER_BYTES = NUM_QK_STAGES * NUM_QK_ATOMS * 8U;
constexpr uint32_t H_TMA_BARRIER_BYTES = NUM_H_STAGES * NUM_H_ATOMS * 8U;
constexpr uint32_t V_TMA_BARRIER_BYTES = NUM_V_STAGES * NUM_V_ATOMS * 8U;
constexpr uint32_t QK_READY_BARRIER_BYTES = NUM_QK_STAGES * 8U;
constexpr uint32_t QH_READY_BARRIER_BYTES = NUM_QK_STAGES * 8U;
constexpr uint32_t QK_REUSE_BARRIER_BYTES = NUM_QK_STAGES * 8U;
constexpr uint32_t H_REUSE_BARRIER_BYTES = NUM_H_STAGES * 8U;
constexpr uint32_t ATTN_READY_BARRIER_BYTES = NUM_QK_STAGES * 8U;
constexpr uint32_t OV_READY_BARRIER_BYTES = NUM_V_STAGES * 8U;
constexpr uint32_t V_REUSE_BARRIER_BYTES = NUM_V_STAGES * 8U;

constexpr uint32_t MAX_TMEM_COLUMNS = 512U;
constexpr uint32_t QK_TMEM_COL = 0U;
constexpr uint32_t OUTPUT_TMEM_COL = BLOCK_T;
constexpr uint32_t QH_TMEM_COL = 3U * BLOCK_T;
constexpr uint32_t ATTN_TMEM_COL = 5U * BLOCK_T;
constexpr CUtensorMapL2promotion TMA_L2_PROMOTION =
    CU_TENSOR_MAP_L2_PROMOTION_L2_256B;

constexpr uint32_t OFFSET_Q = 0U;
constexpr uint32_t OFFSET_K = align_up(OFFSET_Q + Q_SMEM_BYTES, 1024U);
constexpr uint32_t OFFSET_V = align_up(OFFSET_K + K_SMEM_BYTES, 1024U);
constexpr uint32_t OFFSET_H = align_up(OFFSET_V + V_SMEM_BYTES, 1024U);
constexpr uint32_t OFFSET_G_RAW = align_up(OFFSET_H + H_SMEM_BYTES, 16U);
constexpr uint32_t OFFSET_G_CENTER =
    align_up(OFFSET_G_RAW + G_RAW_SMEM_BYTES, 16U);
constexpr uint32_t OFFSET_G_POS =
    align_up(OFFSET_G_CENTER + G_CENTER_SMEM_BYTES, 16U);
constexpr uint32_t OFFSET_G_ALPHA =
    align_up(OFFSET_G_POS + G_POS_SMEM_BYTES, 16U);
constexpr uint32_t OFFSET_G_NEG =
    align_up(OFFSET_G_ALPHA + G_ALPHA_SMEM_BYTES, 16U);
constexpr uint32_t OFFSET_G_REDUCE_MIN =
    align_up(OFFSET_G_NEG + G_NEG_SMEM_BYTES, 16U);
constexpr uint32_t OFFSET_G_REDUCE_MAX =
    align_up(OFFSET_G_REDUCE_MIN + G_REDUCE_PARTIAL_SMEM_BYTES, 16U);
constexpr uint32_t OFFSET_QK_TMA_BAR =
    align_up(OFFSET_G_REDUCE_MAX + G_REDUCE_PARTIAL_SMEM_BYTES, 8U);
constexpr uint32_t OFFSET_H_TMA_BAR = OFFSET_QK_TMA_BAR + QK_TMA_BARRIER_BYTES;
constexpr uint32_t OFFSET_V_TMA_BAR = OFFSET_H_TMA_BAR + H_TMA_BARRIER_BYTES;
constexpr uint32_t OFFSET_QK_READY_BAR =
    OFFSET_V_TMA_BAR + V_TMA_BARRIER_BYTES;
constexpr uint32_t OFFSET_QH_READY_BAR =
    OFFSET_QK_READY_BAR + QK_READY_BARRIER_BYTES;
constexpr uint32_t OFFSET_QK_REUSE_BAR =
    OFFSET_QH_READY_BAR + QH_READY_BARRIER_BYTES;
constexpr uint32_t OFFSET_H_REUSE_BAR =
    OFFSET_QK_REUSE_BAR + QK_REUSE_BARRIER_BYTES;
constexpr uint32_t OFFSET_ATTN_READY_BAR =
    OFFSET_H_REUSE_BAR + H_REUSE_BARRIER_BYTES;
constexpr uint32_t OFFSET_OV_READY_BAR =
    OFFSET_ATTN_READY_BAR + ATTN_READY_BARRIER_BYTES;
constexpr uint32_t OFFSET_V_REUSE_BAR =
    OFFSET_OV_READY_BAR + OV_READY_BARRIER_BYTES;
constexpr uint32_t OFFSET_QK_TMEM_RELEASE_BAR =
    OFFSET_V_REUSE_BAR + V_REUSE_BARRIER_BYTES;
constexpr uint32_t OFFSET_OUTPUT_TMEM_RELEASE_BAR =
    OFFSET_QK_TMEM_RELEASE_BAR + 8U;
constexpr uint32_t OFFSET_ATTN_TMEM_RELEASE_BAR =
    OFFSET_OUTPUT_TMEM_RELEASE_BAR + 8U;
constexpr uint32_t OFFSET_TMEM_ADDR =
    align_up(OFFSET_ATTN_TMEM_RELEASE_BAR + 8U, 4U);
constexpr uint32_t SMEM_SIZE = align_up(OFFSET_TMEM_ADDR + 4U, 1024U);

static CUtensorMap encode_h_atom_tma(void *ptr, uint64_t outer) {
  CUtensorMap tmap;
  constexpr uint32_t rank = 4;
  uint64_t global_dim[rank] = {SWIZZLE_WIDTH, VALUE_DIM,
                               HEAD_DIM / SWIZZLE_WIDTH, outer};
  uint64_t global_strides[rank - 1] = {
      HEAD_DIM * sizeof(nv_bfloat16),
      SWIZZLE_WIDTH * sizeof(nv_bfloat16),
      VALUE_DIM * HEAD_DIM * sizeof(nv_bfloat16),
  };
  uint32_t box_dim[rank] = {SWIZZLE_WIDTH, BLOCK_V, 1U, 1U};
  uint32_t element_strides[rank] = {1, 1, 1, 1};

  cuTensorMapEncodeTiled(&tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, ptr,
                         global_dim, global_strides, box_dim, element_strides,
                         CU_TENSOR_MAP_INTERLEAVE_NONE,
                         CU_TENSOR_MAP_SWIZZLE_128B, TMA_L2_PROMOTION,
                         CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return tmap;
}

static CUtensorMap encode_qk_atom_tma(void *ptr, uint64_t num_tokens,
                                      uint64_t num_heads, uint64_t dim) {
  CUtensorMap tmap;
  constexpr uint32_t rank = 4;
  uint64_t global_dim[rank] = {SWIZZLE_WIDTH, num_tokens, dim / SWIZZLE_WIDTH,
                               num_heads};
  uint64_t global_strides[rank - 1] = {
      num_heads * dim * sizeof(nv_bfloat16),
      SWIZZLE_WIDTH * sizeof(nv_bfloat16),
      dim * sizeof(nv_bfloat16),
  };
  uint32_t box_dim[rank] = {SWIZZLE_WIDTH, BLOCK_T, 1U, 1U};
  uint32_t element_strides[rank] = {1, 1, 1, 1};

  cuTensorMapEncodeTiled(&tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, ptr,
                         global_dim, global_strides, box_dim, element_strides,
                         CU_TENSOR_MAP_INTERLEAVE_NONE,
                         CU_TENSOR_MAP_SWIZZLE_128B, TMA_L2_PROMOTION,
                         CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return tmap;
}

static CUtensorMap encode_v_atom_tma(void *ptr, uint64_t num_tokens) {
  CUtensorMap tmap;
  constexpr uint32_t rank = 4;
  uint64_t global_dim[rank] = {SWIZZLE_WIDTH, num_tokens,
                               VALUE_DIM / SWIZZLE_WIDTH, NUM_OUTPUT_HEADS};
  uint64_t global_strides[rank - 1] = {
      NUM_OUTPUT_HEADS * VALUE_DIM * sizeof(nv_bfloat16),
      SWIZZLE_WIDTH * sizeof(nv_bfloat16),
      VALUE_DIM * sizeof(nv_bfloat16),
  };
  uint32_t box_dim[rank] = {SWIZZLE_WIDTH, BLOCK_T, 1U, 1U};
  uint32_t element_strides[rank] = {1, 1, 1, 1};

  cuTensorMapEncodeTiled(&tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, ptr,
                         global_dim, global_strides, box_dim, element_strides,
                         CU_TENSOR_MAP_INTERLEAVE_NONE,
                         CU_TENSOR_MAP_SWIZZLE_128B, TMA_L2_PROMOTION,
                         CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return tmap;
}

__device__ __forceinline__ uint64_t make_tcgen05_desc_mmajor_v(uint32_t addr) {
  return desc_encode(addr) | (desc_encode(V_MMA_LBO) << 16ULL) |
         (desc_encode(V_MMA_SBO) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
}

__device__ __forceinline__ void
tma_load_qk_atom(uint32_t q_smem_atom, uint32_t k_smem_atom,
                 const CUtensorMap *q_tmap, const CUtensorMap *k_tmap,
                 int32_t chunk_start_i32, uint32_t atom_id, uint32_t qk_head_id,
                 uint32_t barrier, uint64_t cache_policy = EVICT_NORMAL) {
  mbarrier_arrive_expect_tx(barrier, 2U * SWIZZLE_BYTES);
  tma_load_4d(q_smem_atom, q_tmap, 0, chunk_start_i32, atom_id, qk_head_id,
              barrier, cache_policy);
  tma_load_4d(k_smem_atom, k_tmap, 0, chunk_start_i32, atom_id, qk_head_id,
              barrier, cache_policy);
}

__device__ __forceinline__ void
tma_load_h_atom(uint32_t h_smem_atom, const CUtensorMap *h_tmap,
                uint32_t atom_id, uint32_t h_outer, uint32_t barrier,
                uint64_t cache_policy = EVICT_NORMAL) {
  mbarrier_arrive_expect_tx(barrier, H_SWIZZLE_BYTES);
  tma_load_4d(h_smem_atom, h_tmap, 0, 0, atom_id, h_outer, barrier,
              cache_policy);
}

__device__ __forceinline__ void
tma_load_v_atom(uint32_t v_smem_atom, const CUtensorMap *v_tmap,
                int32_t v_chunk_start_i32, uint32_t atom_id, uint32_t head_id,
                uint32_t barrier, uint64_t cache_policy = EVICT_NORMAL) {
  mbarrier_arrive_expect_tx(barrier, SWIZZLE_BYTES);
  tma_load_4d(v_smem_atom, v_tmap, 0, v_chunk_start_i32, atom_id, head_id,
              barrier, cache_policy);
}

__device__ __forceinline__ uint32_t ring_stage(uint32_t producer_slot,
                                               uint32_t num_stages) {
  return producer_slot % num_stages;
}

__device__ __forceinline__ uint32_t ring_reuse_phase(uint32_t producer_slot,
                                                     uint32_t num_stages) {
  const uint32_t reused_slot = producer_slot - num_stages;
  return (reused_slot / num_stages) & 1U;
}

__device__ __forceinline__ void mma_swizzled_atom(uint32_t output_tmem,
                                                  uint32_t matrix_a_smem,
                                                  uint32_t matrix_b_smem,
                                                  uint32_t accum) {
  constexpr uint32_t idesc = make_tcgen05_idesc(MMA_M, MMA_N);
  constexpr uint64_t desc_base =
      (desc_encode(SWIZZLE_SBO) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);

#pragma unroll
  for (uint32_t mi = 0; mi < NUM_MMA_STEPS; ++mi) {
    const uint64_t a_desc =
        desc_base | ((matrix_a_smem + mi * BYTES_ONE_MMA) >> 4);
    const uint64_t b_desc =
        desc_base | ((matrix_b_smem + mi * BYTES_ONE_MMA) >> 4);
    tcgen05_mma(output_tmem, a_desc, b_desc, idesc, accum || (mi > 0U));
  }
}

__device__ __forceinline__ void
mma_swizzled_qh_atom_64x128(uint32_t output_tmem, uint32_t matrix_a_smem,
                            uint32_t matrix_b_smem, uint32_t accum) {
  constexpr uint32_t idesc = make_tcgen05_idesc(MMA_M, BLOCK_V);
  constexpr uint64_t desc_base =
      (desc_encode(SWIZZLE_SBO) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);

#pragma unroll
  for (uint32_t mi = 0; mi < NUM_MMA_STEPS; ++mi) {
    const uint64_t a_desc =
        desc_base | ((matrix_a_smem + mi * BYTES_ONE_MMA) >> 4);
    const uint64_t b_desc =
        desc_base | ((matrix_b_smem + mi * BYTES_ONE_MMA) >> 4);
    tcgen05_mma(output_tmem, a_desc, b_desc, idesc, accum || (mi > 0U));
  }
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

__device__ __forceinline__ void mma_attn_v_tmem_64x128(uint32_t output_tmem,
                                                       uint32_t matrix_a_tmem,
                                                       uint32_t matrix_b_smem) {
  constexpr uint32_t idesc = make_tcgen05_idesc(MMA_M, BLOCK_V) | (1U << 16U);

#pragma unroll
  for (uint32_t ki = 0; ki < NUM_MMA_STEPS; ++ki) {
    const uint32_t a_tmem = matrix_a_tmem + ki * 8U;
    const uint32_t b_base = matrix_b_smem + ki * BYTES_ONE_MMA_MMAJOR;
    tcgen05_mma_tmem(output_tmem, a_tmem, make_tcgen05_desc_mmajor_v(b_base),
                     idesc, ki > 0U);
  }
}

__device__ __forceinline__ float scale_qk_if_active(float qk, float g_col,
                                                    uint32_t active) {
  return active ? qk * g_col : 0.0f;
}

__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  uint32_t packed;
  asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
               : "=r"(packed)
               : "f"(hi), "f"(lo));
  return packed;
}

__device__ __forceinline__ void load_qk_fragment(float *qk_reg_lo,
                                                 float *qk_reg_hi) {
  tcgen05_ld<SHAPE::_16x256b, 4>(qk_reg_lo, 0, QK_TMEM_COL);
  tcgen05_ld<SHAPE::_16x256b, 4>(qk_reg_hi, 0, QK_TMEM_COL + COLS_PER_FRAGMENT);
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

__device__ __forceinline__ void
reduce_g_center(const float *g_raw_ptr, float *g_center_ptr,
                float *g_reduce_min_ptr, float *g_reduce_max_ptr,
                uint32_t warp_id, uint32_t lane_id, uint32_t chunk_len) {
  const uint32_t token_offset = warp_id * 16U + (lane_id & 15U);
  float g_min = INFINITY;
  float g_max = -INFINITY;
  if (lane_id < 16U && token_offset < chunk_len) {
    const float g_value = g_raw_ptr[token_offset];
    g_min = g_value;
    g_max = g_value;
  }

  g_min = warp_reduce_min(g_min);
  g_max = warp_reduce_max(g_max);
  if (lane_id == 0U) {
    g_reduce_min_ptr[warp_id] = g_min;
    g_reduce_max_ptr[warp_id] = g_max;
  }
  bar_sync<1>(NUM_CUDA_WARPS * WARP_SIZE);

  if (warp_id == 0U && lane_id == 0U) {
    float g_center = 0.0f;
    if (chunk_len > 0U) {
      float head_min = g_reduce_min_ptr[0];
      float head_max = g_reduce_max_ptr[0];
#pragma unroll
      for (uint32_t i = 1; i < NUM_CUDA_WARPS; ++i) {
        head_min = fminf(head_min, g_reduce_min_ptr[i]);
        head_max = fmaxf(head_max, g_reduce_max_ptr[i]);
      }
      g_center = 0.5f * (head_min + head_max);
    }
    g_center_ptr[0] = g_center;
  }
}

__device__ __forceinline__ void
materialize_attn_tmem_from_regs(const float *qk_reg_lo, const float *qk_reg_hi,
                                const float *g_neg_ptr, uint32_t row_start,
                                uint32_t row_base_limit, uint32_t row_hi_limit,
                                uint32_t lane_id, uint32_t lane_col) {
  (void)row_start;
  (void)lane_id;
#pragma unroll
  for (uint32_t step_pair = 0; step_pair < FRAGMENT_PAIRS; ++step_pair) {
    const uint32_t col_lo = step_pair * 8U + 2U * lane_col;
    const uint32_t col_hi = col_lo + COLS_PER_FRAGMENT;
    const uint32_t reg_base = step_pair * 4U;

    uint32_t attn_regs_lo[2];
    attn_regs_lo[0] = pack_bf16x2(
        scale_qk_if_active(qk_reg_lo[reg_base + 0U], g_neg_ptr[col_lo],
                           col_lo < row_base_limit),
        scale_qk_if_active(qk_reg_lo[reg_base + 1U], g_neg_ptr[col_lo + 1U],
                           col_lo + 1U < row_base_limit));
    attn_regs_lo[1] = pack_bf16x2(
        scale_qk_if_active(qk_reg_lo[reg_base + 2U], g_neg_ptr[col_lo],
                           col_lo < row_hi_limit),
        scale_qk_if_active(qk_reg_lo[reg_base + 3U], g_neg_ptr[col_lo + 1U],
                           col_lo + 1U < row_hi_limit));
    tcgen05_st<SHAPE::_16x128b, 1>(0U, ATTN_TMEM_COL + step_pair * 4U,
                                   attn_regs_lo);

    uint32_t attn_regs_hi[2];
    attn_regs_hi[0] = pack_bf16x2(
        scale_qk_if_active(qk_reg_hi[reg_base + 0U], g_neg_ptr[col_hi],
                           col_hi < row_base_limit),
        scale_qk_if_active(qk_reg_hi[reg_base + 1U], g_neg_ptr[col_hi + 1U],
                           col_hi + 1U < row_base_limit));
    attn_regs_hi[1] = pack_bf16x2(
        scale_qk_if_active(qk_reg_hi[reg_base + 2U], g_neg_ptr[col_hi],
                           col_hi < row_hi_limit),
        scale_qk_if_active(qk_reg_hi[reg_base + 3U], g_neg_ptr[col_hi + 1U],
                           col_hi + 1U < row_hi_limit));
    tcgen05_st<SHAPE::_16x128b, 1>(0U, ATTN_TMEM_COL + 16U + step_pair * 4U,
                                   attn_regs_hi);
  }
  tcgen05_wait_st();
}

__device__ __forceinline__ __nv_bfloat162
combine_output_pair(float ov_0, float ov_1, float qh_0, float qh_1,
                    float gp_row, float alpha, float scale) {
  const float row_scale = scale * gp_row;
  return __floats2bfloat162_rn(row_scale * __fmaf_rn(alpha, qh_0, ov_0),
                               row_scale * __fmaf_rn(alpha, qh_1, ov_1));
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

__device__ __forceinline__ void
load_output_fragment_pair(uint32_t col_base, float *ov_reg_lo, float *ov_reg_hi,
                          float *qh_reg_lo, float *qh_reg_hi) {
  tcgen05_ld<SHAPE::_16x256b, 4>(ov_reg_lo, 0, OUTPUT_TMEM_COL + col_base);
  tcgen05_ld<SHAPE::_16x256b, 4>(
      ov_reg_hi, 0, OUTPUT_TMEM_COL + col_base + COLS_PER_FRAGMENT);
  tcgen05_ld<SHAPE::_16x256b, 4>(qh_reg_lo, 0, QH_TMEM_COL + col_base);
  tcgen05_ld<SHAPE::_16x256b, 4>(qh_reg_hi, 0,
                                 QH_TMEM_COL + col_base + COLS_PER_FRAGMENT);
  tcgen05_wait_ld();
}

template <bool FULL_CHUNK>
__device__ __forceinline__ void
store_output_row_pair(nv_bfloat16 *row_base_o_ptr, nv_bfloat16 *row_hi_o_ptr,
                      uint32_t lane_col, uint32_t row_base_active,
                      uint32_t row_hi_active, float gp_row_base,
                      float gp_row_hi, float alpha, float scale,
                      uint32_t warp_elected,
                      uint32_t output_tmem_release_barrier) {
  constexpr uint32_t NUM_OUTPUT_FRAGMENT_PAIRS = BLOCK_V / BLOCK_T;
#pragma unroll
  for (uint32_t fragment_pair = 0; fragment_pair < NUM_OUTPUT_FRAGMENT_PAIRS;
       ++fragment_pair) {
    const uint32_t col_base = fragment_pair * BLOCK_T;
    float ov_reg_lo[REGS_PER_FRAGMENT];
    float ov_reg_hi[REGS_PER_FRAGMENT];
    float qh_reg_lo[REGS_PER_FRAGMENT];
    float qh_reg_hi[REGS_PER_FRAGMENT];
    load_output_fragment_pair(col_base, ov_reg_lo, ov_reg_hi, qh_reg_lo,
                              qh_reg_hi);
    if constexpr (NUM_OUTPUT_FRAGMENT_PAIRS == 1U) {
      tcgen05_fence_before_thread_sync();
      if (warp_elected) {
        mbarrier_arrive(output_tmem_release_barrier);
      }
    } else if (fragment_pair + 1U == NUM_OUTPUT_FRAGMENT_PAIRS) {
      tcgen05_fence_before_thread_sync();
      if (warp_elected) {
        mbarrier_arrive(output_tmem_release_barrier);
      }
    }

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

__device__ __forceinline__ void
issue_tma_stage_qk(uint32_t q_stage_smem, uint32_t k_stage_smem,
                   const CUtensorMap *q_tmap, const CUtensorMap *k_tmap,
                   const int64_t *cu_seqlens_ptr,
                   const int32_t *chunk_indices_ptr, uint32_t global_chunk_id,
                   uint32_t qk_head_id, uint32_t qk_tma_barriers) {
  const int2 chunk_meta =
      reinterpret_cast<const int2 *>(chunk_indices_ptr)[global_chunk_id];
  const uint32_t seq_id = static_cast<uint32_t>(chunk_meta.x);
  const uint32_t chunk_id = static_cast<uint32_t>(chunk_meta.y);
  const int32_t chunk_start_i32 = static_cast<int32_t>(
      cu_seqlens_ptr[seq_id] +
      static_cast<int64_t>(chunk_id) * static_cast<int64_t>(BLOCK_T));

#pragma unroll
  for (uint32_t atom = 0; atom < NUM_QK_ATOMS; ++atom) {
    tma_load_qk_atom(q_stage_smem + atom * SWIZZLE_BYTES,
                     k_stage_smem + atom * SWIZZLE_BYTES, q_tmap, k_tmap,
                     chunk_start_i32, atom, qk_head_id,
                     qk_tma_barriers + atom * 8U, EVICT_LAST);
  }
}

__device__ __forceinline__ void issue_tma_stage_h(uint32_t h_stage_smem,
                                                  const CUtensorMap *h_tmap,
                                                  uint32_t global_chunk_id,
                                                  uint32_t head_id,
                                                  uint32_t h_tma_barriers) {
  const uint32_t h_outer = global_chunk_id * NUM_OUTPUT_HEADS + head_id;

#pragma unroll
  for (uint32_t atom = 0; atom < NUM_H_ATOMS; ++atom) {
    tma_load_h_atom(h_stage_smem + atom * H_SWIZZLE_BYTES, h_tmap, atom,
                    h_outer, h_tma_barriers + atom * 8U, EVICT_FIRST);
  }
}

__device__ __forceinline__ void issue_tma_stage_v(uint32_t v_stage_smem,
                                                  const CUtensorMap *v_tmap,
                                                  uint32_t global_chunk_id,
                                                  uint32_t head_id,
                                                  uint32_t v_tma_barriers) {
  const int32_t v_chunk_start_i32 =
      static_cast<int32_t>(global_chunk_id * BLOCK_T);

#pragma unroll
  for (uint32_t atom = 0; atom < NUM_V_ATOMS; ++atom) {
    tma_load_v_atom(v_stage_smem + atom * SWIZZLE_BYTES, v_tmap,
                    v_chunk_start_i32, atom, head_id,
                    v_tma_barriers + atom * 8U, EVICT_FIRST);
  }
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
  const uint32_t warp_elected = elect_sync();

  const uint32_t stripe_id = blockIdx.x;
  const uint32_t head_id = blockIdx.y;
  const uint32_t qk_head_id = head_id / HEADS_PER_QK_HEAD;
  const uint32_t total_num_chunks_u = static_cast<uint32_t>(*total_chunks_ptr);

  extern __shared__ __align__(1024) char smem_ptr[];
  const uint32_t smem = __cvta_generic_to_shared(smem_ptr);

  const uint32_t q_smem = smem + OFFSET_Q;
  const uint32_t k_smem = smem + OFFSET_K;
  const uint32_t v_smem = smem + OFFSET_V;
  const uint32_t h_smem = smem + OFFSET_H;
  const uint32_t g_raw_smem = smem + OFFSET_G_RAW;
  const uint32_t g_center_smem = smem + OFFSET_G_CENTER;
  const uint32_t g_pos_smem = smem + OFFSET_G_POS;
  const uint32_t g_alpha_smem = smem + OFFSET_G_ALPHA;
  const uint32_t g_neg_smem = smem + OFFSET_G_NEG;
  const uint32_t g_reduce_min_smem = smem + OFFSET_G_REDUCE_MIN;
  const uint32_t g_reduce_max_smem = smem + OFFSET_G_REDUCE_MAX;
  const uint32_t qk_tma_barriers = smem + OFFSET_QK_TMA_BAR;
  const uint32_t h_tma_barriers = smem + OFFSET_H_TMA_BAR;
  const uint32_t v_tma_barriers = smem + OFFSET_V_TMA_BAR;
  const uint32_t qk_ready_barriers = smem + OFFSET_QK_READY_BAR;
  const uint32_t qh_ready_barriers = smem + OFFSET_QH_READY_BAR;
  const uint32_t qk_reuse_barriers = smem + OFFSET_QK_REUSE_BAR;
  const uint32_t h_reuse_barriers = smem + OFFSET_H_REUSE_BAR;
  const uint32_t attn_ready_barriers = smem + OFFSET_ATTN_READY_BAR;
  const uint32_t ov_ready_barriers = smem + OFFSET_OV_READY_BAR;
  const uint32_t v_reuse_barriers = smem + OFFSET_V_REUSE_BAR;
  const uint32_t qk_tmem_release_barrier = smem + OFFSET_QK_TMEM_RELEASE_BAR;
  const uint32_t output_tmem_release_barrier =
      smem + OFFSET_OUTPUT_TMEM_RELEASE_BAR;
  const uint32_t attn_tmem_release_barrier =
      smem + OFFSET_ATTN_TMEM_RELEASE_BAR;
  const uint32_t tmem_alloc_smem = smem + OFFSET_TMEM_ADDR;

  float *g_raw_ptr = reinterpret_cast<float *>(smem_ptr + (g_raw_smem - smem));
  float *g_center_ptr =
      reinterpret_cast<float *>(smem_ptr + (g_center_smem - smem));
  float *g_pos_ptr = reinterpret_cast<float *>(smem_ptr + (g_pos_smem - smem));
  float *g_alpha_ptr =
      reinterpret_cast<float *>(smem_ptr + (g_alpha_smem - smem));
  float *g_neg_ptr = reinterpret_cast<float *>(smem_ptr + (g_neg_smem - smem));
  float *g_reduce_min_ptr =
      reinterpret_cast<float *>(smem_ptr + (g_reduce_min_smem - smem));
  float *g_reduce_max_ptr =
      reinterpret_cast<float *>(smem_ptr + (g_reduce_max_smem - smem));

  if (warp_id == TMA_WARP) {
    if (warp_elected) {
#pragma unroll
      for (uint32_t stage = 0; stage < NUM_QK_STAGES; ++stage) {
        mbarrier_init(qk_ready_barriers + stage * 8U, 1);
        mbarrier_init(qh_ready_barriers + stage * 8U, 1);
        mbarrier_init(qk_reuse_barriers + stage * 8U, 2);
        mbarrier_init(attn_ready_barriers + stage * 8U, NUM_CUDA_WARPS);
#pragma unroll
        for (uint32_t atom = 0; atom < NUM_QK_ATOMS; ++atom) {
          mbarrier_init(qk_tma_barriers + (stage * NUM_QK_ATOMS + atom) * 8U,
                        1);
        }
      }
#pragma unroll
      for (uint32_t stage = 0; stage < NUM_H_STAGES; ++stage) {
        mbarrier_init(h_reuse_barriers + stage * 8U, 1);
#pragma unroll
        for (uint32_t atom = 0; atom < NUM_H_ATOMS; ++atom) {
          mbarrier_init(h_tma_barriers + (stage * NUM_H_ATOMS + atom) * 8U, 1);
        }
      }
#pragma unroll
      for (uint32_t stage = 0; stage < NUM_V_STAGES; ++stage) {
        mbarrier_init(ov_ready_barriers + stage * 8U, 1);
        mbarrier_init(v_reuse_barriers + stage * 8U, 1);
#pragma unroll
        for (uint32_t atom = 0; atom < NUM_V_ATOMS; ++atom) {
          mbarrier_init(v_tma_barriers + (stage * NUM_V_ATOMS + atom) * 8U, 1);
        }
      }
      mbarrier_init(qk_tmem_release_barrier, NUM_CUDA_WARPS);
      mbarrier_init(output_tmem_release_barrier, NUM_CUDA_WARPS);
      mbarrier_init(attn_tmem_release_barrier, 1);
      prefetch_tensormap(&q_tmap);
      prefetch_tensormap(&k_tmap);
      prefetch_tensormap(&v_tmap);
      prefetch_tensormap(&h_tmap);
      fence_mbarrier_init();
    }
  } else if (warp_id == QK_MMA_WARP) {
    tcgen05_alloc(tmem_alloc_smem, MAX_TMEM_COLUMNS);
  }

  __syncthreads();

  const bool cuda_warp = warp_id < NUM_CUDA_WARPS;
  uint32_t row_base = 0U;
  uint32_t row_hi = 0U;
  uint32_t lane_col = 0U;
  if (cuda_warp) {
    row_base = warp_id * ROWS_PER_WARP + lane_id / LANES_PER_ROW_GROUP;
    row_hi = row_base + ROW_PAIR_STRIDE;
    lane_col = lane_id % LANES_PER_ROW_GROUP;
  }

  if (warp_id == TMA_WARP) {
    if (warp_elected) {
      uint32_t qk_prod_slot = 0U;
      uint32_t next_qk_global_chunk_id = stripe_id;
      while (qk_prod_slot < INITIAL_QK_PREFETCH_STAGES &&
             next_qk_global_chunk_id < total_num_chunks_u) {
        const uint32_t qk_stage = ring_stage(qk_prod_slot, NUM_QK_STAGES);
        issue_tma_stage_qk(q_smem + qk_stage * Q_STAGE_SMEM_SIZE,
                           k_smem + qk_stage * K_STAGE_SMEM_SIZE, &q_tmap,
                           &k_tmap, cu_seqlens_ptr, chunk_indices_ptr,
                           next_qk_global_chunk_id, qk_head_id,
                           qk_tma_barriers + qk_stage * NUM_QK_ATOMS * 8U);
        next_qk_global_chunk_id += gridDim.x;
        ++qk_prod_slot;
      }

      uint32_t h_prod_slot = 0U;
      uint32_t next_h_global_chunk_id = stripe_id;
      while (h_prod_slot < INITIAL_H_PREFETCH_STAGES &&
             next_h_global_chunk_id < total_num_chunks_u) {
        const uint32_t h_stage = ring_stage(h_prod_slot, NUM_H_STAGES);
        issue_tma_stage_h(h_smem + h_stage * H_STAGE_SMEM_SIZE, &h_tmap,
                          next_h_global_chunk_id, head_id,
                          h_tma_barriers + h_stage * NUM_H_ATOMS * 8U);
        next_h_global_chunk_id += gridDim.x;
        ++h_prod_slot;
      }

      uint32_t v_prod_slot = 0U;
      uint32_t next_v_global_chunk_id = stripe_id;
      while (v_prod_slot < INITIAL_V_PREFETCH_STAGES &&
             next_v_global_chunk_id < total_num_chunks_u) {
        const uint32_t v_stage = ring_stage(v_prod_slot, NUM_V_STAGES);
        issue_tma_stage_v(v_smem + v_stage * V_STAGE_SMEM_SIZE, &v_tmap,
                          next_v_global_chunk_id, head_id,
                          v_tma_barriers + v_stage * NUM_V_ATOMS * 8U);
        next_v_global_chunk_id += gridDim.x;
        ++v_prod_slot;
      }

      for (uint32_t global_chunk_id = stripe_id, chunk_iter = 0U;
           global_chunk_id < total_num_chunks_u;
           global_chunk_id += gridDim.x, ++chunk_iter) {
        if (next_qk_global_chunk_id < total_num_chunks_u) {
          const uint32_t next_qk_stage = ring_stage(qk_prod_slot, NUM_QK_STAGES);
          if (qk_prod_slot >= NUM_QK_STAGES) {
            mbarrier_wait(qk_reuse_barriers + next_qk_stage * 8U,
                          ring_reuse_phase(qk_prod_slot, NUM_QK_STAGES));
          }
          issue_tma_stage_qk(
              q_smem + next_qk_stage * Q_STAGE_SMEM_SIZE,
              k_smem + next_qk_stage * K_STAGE_SMEM_SIZE, &q_tmap, &k_tmap,
              cu_seqlens_ptr, chunk_indices_ptr, next_qk_global_chunk_id,
              qk_head_id, qk_tma_barriers + next_qk_stage * NUM_QK_ATOMS * 8U);
          next_qk_global_chunk_id += gridDim.x;
          ++qk_prod_slot;
        }

        if (next_h_global_chunk_id < total_num_chunks_u) {
          const uint32_t next_h_stage = ring_stage(h_prod_slot, NUM_H_STAGES);
          if (h_prod_slot >= NUM_H_STAGES) {
            mbarrier_wait(h_reuse_barriers + next_h_stage * 8U,
                          ring_reuse_phase(h_prod_slot, NUM_H_STAGES));
          }
          issue_tma_stage_h(h_smem + next_h_stage * H_STAGE_SMEM_SIZE, &h_tmap,
                            next_h_global_chunk_id, head_id,
                            h_tma_barriers + next_h_stage * NUM_H_ATOMS * 8U);
          next_h_global_chunk_id += gridDim.x;
          ++h_prod_slot;
        }

        if (next_v_global_chunk_id < total_num_chunks_u) {
          const uint32_t next_v_stage = ring_stage(v_prod_slot, NUM_V_STAGES);
          if (v_prod_slot >= NUM_V_STAGES) {
            mbarrier_wait(v_reuse_barriers + next_v_stage * 8U,
                          ring_reuse_phase(v_prod_slot, NUM_V_STAGES));
          }
          issue_tma_stage_v(v_smem + next_v_stage * V_STAGE_SMEM_SIZE, &v_tmap,
                            next_v_global_chunk_id, head_id,
                            v_tma_barriers + next_v_stage * NUM_V_ATOMS * 8U);
          next_v_global_chunk_id += gridDim.x;
          ++v_prod_slot;
        }
      }
    }
  } else if (warp_id == QK_MMA_WARP) {
    for (uint32_t global_chunk_id = stripe_id, chunk_iter = 0U;
         global_chunk_id < total_num_chunks_u;
         global_chunk_id += gridDim.x, ++chunk_iter) {
      const uint32_t qk_stage = chunk_iter % NUM_QK_STAGES;
      const uint32_t qk_stage_phase = (chunk_iter / NUM_QK_STAGES) & 1U;
      const uint32_t v_stage = chunk_iter % NUM_V_STAGES;
      const uint32_t v_stage_phase = (chunk_iter / NUM_V_STAGES) & 1U;
      const uint32_t q_stage_smem = q_smem + qk_stage * Q_STAGE_SMEM_SIZE;
      const uint32_t k_stage_smem = k_smem + qk_stage * K_STAGE_SMEM_SIZE;
      const uint32_t v_stage_smem = v_smem + v_stage * V_STAGE_SMEM_SIZE;
      const uint32_t qk_tma_stage =
          qk_tma_barriers + qk_stage * NUM_QK_ATOMS * 8U;
      const uint32_t v_tma_stage = v_tma_barriers + v_stage * NUM_V_ATOMS * 8U;

      if (chunk_iter > 0U) {
        mbarrier_wait(qk_tmem_release_barrier, 1U ^ (chunk_iter & 1U));
      }
      mbarrier_wait(qk_tma_stage + 0U * 8U, qk_stage_phase);
      tcgen05_fence_after_thread_sync();
      if (warp_elected) {
        mma_swizzled_atom(QK_TMEM_COL, q_stage_smem + 0U * SWIZZLE_BYTES,
                          k_stage_smem + 0U * SWIZZLE_BYTES, 0U);
      }
      mbarrier_wait(qk_tma_stage + 1U * 8U, qk_stage_phase);
      tcgen05_fence_after_thread_sync();
      if (warp_elected) {
        mma_swizzled_atom(QK_TMEM_COL, q_stage_smem + 1U * SWIZZLE_BYTES,
                          k_stage_smem + 1U * SWIZZLE_BYTES, 1U);
        tcgen05_commit(qk_ready_barriers + qk_stage * 8U);
        mbarrier_arrive(qk_reuse_barriers + qk_stage * 8U);
      }
      mbarrier_wait(v_tma_stage + 0U * 8U, v_stage_phase);
      mbarrier_wait(v_tma_stage + 1U * 8U, v_stage_phase);
      mbarrier_wait(attn_ready_barriers + qk_stage * 8U, qk_stage_phase);
      tcgen05_fence_after_thread_sync();
      if (chunk_iter > 0U) {
        mbarrier_wait(output_tmem_release_barrier, 1U ^ (chunk_iter & 1U));
      }
      if (warp_elected) {
        mma_attn_v_tmem_64x128(OUTPUT_TMEM_COL, ATTN_TMEM_COL, v_stage_smem);
        tcgen05_commit(ov_ready_barriers + v_stage * 8U);
        mbarrier_arrive(v_reuse_barriers + v_stage * 8U);
      }
      tcgen05_fence_before_thread_sync();
      if (warp_elected) {
        mbarrier_arrive(attn_tmem_release_barrier);
      }
      mbarrier_wait(ov_ready_barriers + v_stage * 8U, v_stage_phase);
    }
  } else if (warp_id == QH_MMA_WARP) {
    for (uint32_t global_chunk_id = stripe_id, chunk_iter = 0U;
         global_chunk_id < total_num_chunks_u;
         global_chunk_id += gridDim.x, ++chunk_iter) {
      const uint32_t qk_stage = chunk_iter % NUM_QK_STAGES;
      const uint32_t qk_stage_phase = (chunk_iter / NUM_QK_STAGES) & 1U;
      const uint32_t h_stage = chunk_iter % NUM_H_STAGES;
      const uint32_t h_stage_phase = (chunk_iter / NUM_H_STAGES) & 1U;
      const uint32_t q_stage_smem = q_smem + qk_stage * Q_STAGE_SMEM_SIZE;
      const uint32_t h_stage_smem = h_smem + h_stage * H_STAGE_SMEM_SIZE;
      const uint32_t qk_tma_stage =
          qk_tma_barriers + qk_stage * NUM_QK_ATOMS * 8U;
      const uint32_t h_tma_stage = h_tma_barriers + h_stage * NUM_H_ATOMS * 8U;

      mbarrier_wait(qk_tma_stage + 0U * 8U, qk_stage_phase);
      mbarrier_wait(h_tma_stage + 0U * 8U, h_stage_phase);
      tcgen05_fence_after_thread_sync();
      if (chunk_iter > 0U) {
        mbarrier_wait(output_tmem_release_barrier, 1U ^ (chunk_iter & 1U));
      }
      if (warp_elected) {
        mma_swizzled_qh_atom_64x128(QH_TMEM_COL,
                                    q_stage_smem + 0U * SWIZZLE_BYTES,
                                    h_stage_smem + 0U * H_SWIZZLE_BYTES, 0U);
      }
      mbarrier_wait(qk_tma_stage + 1U * 8U, qk_stage_phase);
      mbarrier_wait(h_tma_stage + 1U * 8U, h_stage_phase);
      tcgen05_fence_after_thread_sync();
      if (warp_elected) {
        mma_swizzled_qh_atom_64x128(QH_TMEM_COL,
                                    q_stage_smem + 1U * SWIZZLE_BYTES,
                                    h_stage_smem + 1U * H_SWIZZLE_BYTES, 1U);
        tcgen05_commit(qh_ready_barriers + qk_stage * 8U);
        mbarrier_arrive(qk_reuse_barriers + qk_stage * 8U);
        mbarrier_arrive(h_reuse_barriers + h_stage * 8U);
      }
    }
  } else {
    for (uint32_t global_chunk_id = stripe_id, chunk_iter = 0U;
         global_chunk_id < total_num_chunks_u;
         global_chunk_id += gridDim.x, ++chunk_iter) {
      const int2 chunk_meta =
          reinterpret_cast<const int2 *>(chunk_indices_ptr)[global_chunk_id];
      const uint32_t seq_id = static_cast<uint32_t>(chunk_meta.x);
      const uint32_t chunk_id = static_cast<uint32_t>(chunk_meta.y);
      const int64_t bos = cu_seqlens_ptr[seq_id];
      const int64_t eos = cu_seqlens_ptr[seq_id + 1];
      const int64_t chunk_start =
          bos + static_cast<int64_t>(chunk_id) * static_cast<int64_t>(BLOCK_T);
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
      const uint32_t qk_stage = chunk_iter % NUM_QK_STAGES;
      const uint32_t qk_stage_phase = (chunk_iter / NUM_QK_STAGES) & 1U;
      const uint32_t v_stage = chunk_iter % NUM_V_STAGES;
      const uint32_t v_stage_phase = (chunk_iter / NUM_V_STAGES) & 1U;
      for (uint32_t i = tid; i < BLOCK_T; i += NUM_CUDA_WARPS * WARP_SIZE) {
        float g_value = 0.0f;
        if (i < chunk_len) {
          const int64_t token_idx = chunk_start + static_cast<int64_t>(i);
          g_value = g_cu_ptr[token_idx * NUM_OUTPUT_HEADS + head_id];
        }
        g_raw_ptr[i] = g_value;
      }
      bar_sync<1>(NUM_CUDA_WARPS * WARP_SIZE);

      reduce_g_center(g_raw_ptr, g_center_ptr, g_reduce_min_ptr,
                      g_reduce_max_ptr, warp_id, lane_id, chunk_len);
      bar_sync<1>(NUM_CUDA_WARPS * WARP_SIZE);

      if (tid == 0U) {
        g_alpha_ptr[0] = __expf(g_center_ptr[0]);
      }
      for (uint32_t i = tid; i < BLOCK_T; i += NUM_CUDA_WARPS * WARP_SIZE) {
        const float g_shifted = g_raw_ptr[i] - g_center_ptr[0];
        const float gp = __expf(g_shifted);
        g_pos_ptr[i] = gp;
        g_neg_ptr[i] = __frcp_rn(gp);
      }
      bar_sync<1>(NUM_CUDA_WARPS * WARP_SIZE);

      mbarrier_wait(qk_ready_barriers + qk_stage * 8U, qk_stage_phase);
      tcgen05_fence_after_thread_sync();

      float qk_reg_lo[REGS_PER_FRAGMENT];
      float qk_reg_hi[REGS_PER_FRAGMENT];
      load_qk_fragment(qk_reg_lo, qk_reg_hi);
      tcgen05_fence_before_thread_sync();
      if (warp_elected) {
        mbarrier_arrive(qk_tmem_release_barrier);
      }
      if (chunk_iter > 0U) {
        mbarrier_wait(attn_tmem_release_barrier, 1U ^ (chunk_iter & 1U));
      }
      materialize_attn_tmem_from_regs(qk_reg_lo, qk_reg_hi, g_neg_ptr,
                                      warp_id * ROWS_PER_WARP, row_base_limit,
                                      row_hi_limit, lane_id, lane_col);
      if (warp_elected) {
        mbarrier_arrive(attn_ready_barriers + qk_stage * 8U);
      }

      mbarrier_wait(qh_ready_barriers + qk_stage * 8U, qk_stage_phase);
      mbarrier_wait(ov_ready_barriers + v_stage * 8U, v_stage_phase);
      tcgen05_fence_after_thread_sync();

      const float gp_row_base = g_pos_ptr[row_base];
      const float gp_row_hi = g_pos_ptr[row_hi];
      const float alpha = g_alpha_ptr[0];

      nv_bfloat16 *row_base_o_ptr =
          o_ptr +
          ((chunk_start + static_cast<int64_t>(row_base)) * NUM_OUTPUT_HEADS +
           head_id) *
              VALUE_DIM;
      nv_bfloat16 *row_hi_o_ptr = row_base_o_ptr + ROW_PAIR_OUTPUT_STRIDE;

      if (full_chunk) {
        store_output_row_pair<true>(row_base_o_ptr, row_hi_o_ptr, lane_col, 0U,
                                    0U, gp_row_base, gp_row_hi, alpha, scale,
                                    warp_elected,
                                    output_tmem_release_barrier);
      } else {
        const uint32_t row_base_active = row_base < chunk_len;
        const uint32_t row_hi_active = row_hi < chunk_len;
        store_output_row_pair<false>(row_base_o_ptr, row_hi_o_ptr, lane_col,
                                     row_base_active, row_hi_active,
                                     gp_row_base, gp_row_hi, alpha, scale,
                                     warp_elected,
                                     output_tmem_release_barrier);
      }
    }
  }

  tcgen05_fence_before_thread_sync();
  __syncthreads();
  if (warp_id == QK_MMA_WARP) {
    tcgen05_dealloc(0, MAX_TMEM_COLUMNS);
  }
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

  auto q_tmap = encode_qk_atom_tma(q_chunks.data_ptr(), total_num_tokens,
                                   NUM_QK_HEADS, HEAD_DIM);
  auto k_tmap = encode_qk_atom_tma(k_chunks.data_ptr(), total_num_tokens,
                                   NUM_QK_HEADS, HEAD_DIM);
  auto v_tmap =
      encode_v_atom_tma(v_new.data_ptr(), total_num_chunks_capacity * BLOCK_T);
  auto h_tmap = encode_h_atom_tma(
      h.data_ptr(), static_cast<uint64_t>(h.size(0)) * NUM_OUTPUT_HEADS);

  auto *g_cu_ptr = reinterpret_cast<const float *>(g_cu.data_ptr());
  auto *o_ptr = reinterpret_cast<nv_bfloat16 *>(o.data_ptr());
  auto *cu_seqlens_ptr =
      reinterpret_cast<const int64_t *>(cu_seqlens.data_ptr());
  auto *chunk_indices_ptr =
      reinterpret_cast<const int32_t *>(chunk_indices.data_ptr());
  auto *total_chunks_ptr =
      reinterpret_cast<const int32_t *>(total_chunks.data_ptr());

  int device_id = 0;
  cudaGetDevice(&device_id);
  cudaDeviceProp device_prop;
  cudaGetDeviceProperties(&device_prop, device_id);
  TVM_FFI_CHECK(device_prop.sharedMemPerBlockOptin >= SMEM_SIZE, RuntimeError)
      << "o_v2 requires " << SMEM_SIZE
      << " bytes of dynamic shared memory, but the device only supports "
      << device_prop.sharedMemPerBlockOptin << " bytes per block";

  auto kernel = o_v2_kernel_cutlass;
  const cudaError_t attr_err = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE);
  if (attr_err != cudaSuccess) {
    TVM_FFI_THROW(RuntimeError)
        << "Failed to set o_v2 dynamic shared memory attribute: "
        << cudaGetErrorString(attr_err);
  }

  int active_blocks_per_sm = 0;
  const cudaError_t occ_err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks_per_sm, kernel, NUM_THREADS, SMEM_SIZE);
  if (occ_err != cudaSuccess) {
    TVM_FFI_THROW(RuntimeError)
        << "Failed to query o_v2 occupancy: " << cudaGetErrorString(occ_err);
  }

  const uint32_t resident_blocks =
      active_blocks_per_sm > 0
          ? static_cast<uint32_t>(active_blocks_per_sm) *
                static_cast<uint32_t>(device_prop.multiProcessorCount)
          : NUM_OUTPUT_HEADS;
  uint32_t grid_x = resident_blocks / NUM_OUTPUT_HEADS;
  if (grid_x == 0U) {
    grid_x = 1U;
  }
  if (grid_x > total_num_chunks_capacity) {
    grid_x = total_num_chunks_capacity;
  }

  dim3 grid(grid_x, NUM_OUTPUT_HEADS);
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
