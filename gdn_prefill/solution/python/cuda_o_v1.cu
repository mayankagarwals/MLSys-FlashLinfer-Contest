#include "cuda_utils.h"

#include <cstdint>
#include <cuda_bf16.h>
#include <math.h>

/*
Pipeline sketch for `o_v1`

  Participants
    TMA  : producer warp, fills SMEM stages
    MMA  : single elected thread issues tcgen05.mma / tcgen05.commit
    CUDA : four consumer warps, materialize ATTN and drain TMEM to GMEM

  Storage
    SMEM :
      - q/k ring: 2 stages
      - h ring:   1 stage per output head within the qk-head group
      - v ring:   1 stage per output head within the qk-head group
    TMEM :
      [   0 ..  63] QK
      [  64 .. 127] ATTN0
      [ 128 .. 191] ATTN1
      [ 192 .. 319] OUT   (shared by head0 and head1)
      [ 320 .. 447] QH0
      head1 QH aliases [QK | ATTN0] = [0 .. 127]

  Important aliasing rule
    The head1 QH build reuses the same TMEM columns that hold QK and ATTN0.
    That alias is only legal after all of the following are true:
      1. CUDA has finished reading QK and arrived
         `qk_tmem_release_barrier`
      2. MMA has finished head0 QH and head0 OV for the current chunk phase and
         both per-head commits have completed
         (`qh_mma_barriers[head0]` and `ov_mma_barriers[head0]`)
      3. The previous chunk has finished draining head1 OUT+QH and arrived
         `output_tmem_release_barrier`

  Resource lifetime rule
    A TMEM buffer is considered free when its last TMEM consumer has finished,
    not when the later GMEM store finishes. CUDA reads OUT/QH into registers
    first, then stores to GMEM after the TMEM read-release barrier arrives.

  Steady-state flow for one chunk

  1. TMA makes the SMEM stages visible.
     - q/k uses `qk_tma_barriers[qk_stage]`
     - h uses `h_tma_barriers[head]`
     - v uses `v_tma_barriers[head]`

  2. MMA builds QK in TMEM from q/k.
     - Commit: `qk_mma_barrier`
     - CUDA waits on `qk_mma_barrier`, reads QK to registers, then arrives
       `qk_tmem_release_barrier`

  3. CUDA materializes ATTN0 from the QK registers.
     - ATTN0 becomes visible via `attn_ready_barriers[head0]`
     - In parallel, MMA builds QH0 from q + h0 and commits
       `qh_mma_barriers[head0]`

  4. MMA consumes ATTN0 + v0 into the shared OUT bank.
     - Commit: `ov_mma_barriers[head0]`
     - While that is in flight, CUDA materializes ATTN1 from the same QK regs
       and arrives `attn_ready_barriers[head1]`

  5. MMA builds QH1 in the aliased [QK | ATTN0] region.
     Gate before the alias is reused:
       - `qk_tmem_release_barrier`
       - head0 `qh_mma_barriers[head0]` for the current `qk_phase`
       - head0 `ov_mma_barriers[head0]` for the current `qk_phase`
     Releases caused by this transition:
       - h0/v0 SMEM stages are returned through `h_reuse_barriers[head0]` and
         `v_reuse_barriers[head0]`
       - once QH1 is complete, h1 and the q/k SMEM stage are returned through
         `h_reuse_barriers[head1]` and `qk_reuse_barriers[qk_stage]`

  6. CUDA drains head0 by reading OUT + QH0 into registers.
     - CUDA waits on head0 `ov_mma_barriers[head0]` and
       `qh_mma_barriers[head0]` for the current `qk_phase`
     - On the final TMEM read, CUDA arrives `head0_release_barrier`
     - Only after that may head1 OV reuse the shared OUT bank

  7. MMA consumes ATTN1 + v1 into OUT, then CUDA drains head1 from OUT + QH1.
     - MMA waits on `head0_release_barrier` before launching head1 OV
     - Once head1 OV is complete, v1 is returned through
       `v_reuse_barriers[head1]`
     - On the final head1 OUT+QH TMEM read, CUDA arrives
       `output_tmem_release_barrier`

  Cross-chunk consequence
    `output_tmem_release_barrier` is the barrier that makes the next chunk safe:
      - MMA waits on it before rebuilding the next chunk's QK
      - CUDA waits on it before rebuilding the next chunk's ATTN0
    This is required because the next chunk's [QK | ATTN0] overlaps the prior
    chunk's head1 QH alias.
*/

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
constexpr uint32_t OUTPUT_FRAGMENT_PAIRS = BLOCK_V / BLOCK_T;

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
constexpr uint32_t G_STAGE_SMEM_SIZE = BLOCK_T * sizeof(float);
constexpr uint32_t G_RAW_SMEM_SIZE = HEADS_PER_QK_HEAD * G_STAGE_SMEM_SIZE;
constexpr uint32_t G_CENTER_SMEM_SIZE = HEADS_PER_QK_HEAD * sizeof(float);
constexpr uint32_t G_POS_SMEM_SIZE = HEADS_PER_QK_HEAD * G_STAGE_SMEM_SIZE;
constexpr uint32_t G_ALPHA_SMEM_SIZE = HEADS_PER_QK_HEAD * sizeof(float);
constexpr uint32_t G_NEG_SMEM_SIZE = HEADS_PER_QK_HEAD * G_STAGE_SMEM_SIZE;
constexpr uint32_t G_REDUCE_PARTIAL_SMEM_SIZE = NUM_CUDA_WARPS * sizeof(float);
constexpr uint32_t QK_TMA_BARRIER_BYTES = QK_STAGE_COUNT * 8U;
constexpr uint32_t VH_TMA_BARRIER_BYTES = VH_STAGE_COUNT * 8U;
constexpr uint32_t QK_REUSE_BARRIER_BYTES = QK_STAGE_COUNT * 8U;
constexpr uint32_t VH_REUSE_BARRIER_BYTES = VH_STAGE_COUNT * 8U;
constexpr uint32_t ATTN_READY_BARRIER_BYTES = HEADS_PER_QK_HEAD * 8U;
constexpr uint32_t VH_MMA_BARRIER_BYTES = HEADS_PER_QK_HEAD * 8U;
constexpr uint32_t OFFSET_Q = 0U;
constexpr uint32_t OFFSET_K = OFFSET_Q + Q_SMEM_BYTES;
constexpr uint32_t OFFSET_V = OFFSET_K + K_SMEM_BYTES;
constexpr uint32_t OFFSET_H = OFFSET_V + V_SMEM_BYTES;
constexpr uint32_t OFFSET_G_RAW = OFFSET_H + H_SMEM_BYTES;
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
constexpr uint32_t OFFSET_QK_REUSE_BAR =
    OFFSET_H_TMA_BAR + VH_TMA_BARRIER_BYTES;
constexpr uint32_t OFFSET_H_REUSE_BAR =
    OFFSET_QK_REUSE_BAR + QK_REUSE_BARRIER_BYTES;
constexpr uint32_t OFFSET_V_REUSE_BAR =
    OFFSET_H_REUSE_BAR + VH_REUSE_BARRIER_BYTES;
constexpr uint32_t OFFSET_MMA_BAR = OFFSET_V_REUSE_BAR + VH_REUSE_BARRIER_BYTES;
constexpr uint32_t OFFSET_ATTN_READY_BAR = OFFSET_MMA_BAR + 8U;
constexpr uint32_t OFFSET_HEAD0_RELEASE_BAR =
    OFFSET_ATTN_READY_BAR + ATTN_READY_BARRIER_BYTES;
constexpr uint32_t OFFSET_QH_MMA_BAR = OFFSET_HEAD0_RELEASE_BAR + 8U;
constexpr uint32_t OFFSET_OV_MMA_BAR = OFFSET_QH_MMA_BAR + VH_MMA_BARRIER_BYTES;
constexpr uint32_t OFFSET_QK_TMEM_RELEASE_BAR = OFFSET_OV_MMA_BAR + VH_MMA_BARRIER_BYTES;
constexpr uint32_t OFFSET_OUTPUT_TMEM_RELEASE_BAR =
    OFFSET_QK_TMEM_RELEASE_BAR + 8U;
constexpr uint32_t OFFSET_TMEM_ADDR = OFFSET_OUTPUT_TMEM_RELEASE_BAR + 8U;
constexpr uint32_t SMEM_SIZE = (OFFSET_TMEM_ADDR + 4U + 1023U) & ~1023U;

//// Tensor Memory
constexpr uint32_t MAX_COLUMNS = 512U;
constexpr uint32_t QK_TMEM_COL = 0U;
constexpr uint32_t ATTN_TMEM_BASE_COL = BLOCK_T;
constexpr uint32_t ATTN_TMEM_HEAD_STRIDE = BLOCK_T;
constexpr uint32_t OUTPUT_TMEM_COL = 3U * BLOCK_T;
constexpr uint32_t HEAD0_QH_TMEM_COL = 5U * BLOCK_T;
constexpr uint32_t HEAD1_QH_TMEM_COL = QK_TMEM_COL;
static_assert(ATTN_TMEM_BASE_COL == QK_TMEM_COL + BLOCK_T);
static_assert(ATTN_TMEM_BASE_COL + HEADS_PER_QK_HEAD * ATTN_TMEM_HEAD_STRIDE ==
              OUTPUT_TMEM_COL);
static_assert(HEAD0_QH_TMEM_COL + 2U * BLOCK_T <= MAX_COLUMNS);
constexpr CUtensorMapL2promotion TMA_L2_PROMOTION =
    CU_TENSOR_MAP_L2_PROMOTION_L2_256B;

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
  if (active == 0U) {
    return 0.0f;
  }
  return qk * g_col;
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

__device__ __forceinline__ uint32_t qh_tmem_col_for_head(uint32_t head_offset) {
  return head_offset == HEAD0_STAGE_INDEX ? HEAD0_QH_TMEM_COL
                                          : HEAD1_QH_TMEM_COL;
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

__device__ __forceinline__ void materialize_attn_tmem_from_regs(
    uint32_t attn_tmem_col, const float *qk_reg_lo, const float *qk_reg_hi,
    const float *g_neg_head_ptr, uint32_t row_base_limit, uint32_t row_hi_limit,
    uint32_t lane_col) {
#pragma unroll
  for (uint32_t step_pair = 0; step_pair < FRAGMENT_PAIRS; ++step_pair) {
    const uint32_t col_lo = step_pair * 8U + 2U * lane_col;
    const uint32_t col_hi = col_lo + COLS_PER_FRAGMENT;
    const uint32_t reg_base = step_pair * 4U;

    uint32_t attn_regs_lo[2];
    attn_regs_lo[0] = pack_bf16x2(
        scale_qk_if_active(qk_reg_lo[reg_base + 0U], g_neg_head_ptr[col_lo],
                           col_lo < row_base_limit),
        scale_qk_if_active(qk_reg_lo[reg_base + 1U],
                           g_neg_head_ptr[col_lo + 1U],
                           col_lo + 1U < row_base_limit));
    attn_regs_lo[1] = pack_bf16x2(
        scale_qk_if_active(qk_reg_lo[reg_base + 2U], g_neg_head_ptr[col_lo],
                           col_lo < row_hi_limit),
        scale_qk_if_active(qk_reg_lo[reg_base + 3U],
                           g_neg_head_ptr[col_lo + 1U],
                           col_lo + 1U < row_hi_limit));
    tcgen05_st<SHAPE::_16x128b, 1>(0U, attn_tmem_col + step_pair * 4U,
                                   attn_regs_lo);

    uint32_t attn_regs_hi[2];
    attn_regs_hi[0] = pack_bf16x2(
        scale_qk_if_active(qk_reg_hi[reg_base + 0U], g_neg_head_ptr[col_hi],
                           col_hi < row_base_limit),
        scale_qk_if_active(qk_reg_hi[reg_base + 1U],
                           g_neg_head_ptr[col_hi + 1U],
                           col_hi + 1U < row_base_limit));
    attn_regs_hi[1] = pack_bf16x2(
        scale_qk_if_active(qk_reg_hi[reg_base + 2U], g_neg_head_ptr[col_hi],
                           col_hi < row_hi_limit),
        scale_qk_if_active(qk_reg_hi[reg_base + 3U],
                           g_neg_head_ptr[col_hi + 1U],
                           col_hi + 1U < row_hi_limit));
    tcgen05_st<SHAPE::_16x128b, 1>(0U, attn_tmem_col + 16U + step_pair * 4U,
                                   attn_regs_hi);
  }
}

__device__ __forceinline__ __nv_bfloat162
combine_output_pair(float ov_0, float ov_1, float qh_0, float qh_1,
                    float gp_row, float alpha, float scale) {
  const float row_scale = scale * gp_row;
  return __floats2bfloat162_rn(row_scale * __fmaf_rn(alpha, qh_0, ov_0),
                               row_scale * __fmaf_rn(alpha, qh_1, ov_1));
}

__device__ __forceinline__ void store_bf162_no_allocate(nv_bfloat16 *ptr,
                                                        __nv_bfloat162 value);
__device__ __forceinline__ void
store_bf162_pair_no_allocate_if(nv_bfloat16 *ptr_0, __nv_bfloat162 value_0,
                                nv_bfloat16 *ptr_1, __nv_bfloat162 value_1,
                                uint32_t predicate);

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

template <bool FULL_CHUNK>
__device__ __forceinline__ void store_output_fragment_pair(
    nv_bfloat16 *row_base_o_ptr, nv_bfloat16 *row_hi_o_ptr, uint32_t col_base,
    uint32_t lane_col, uint32_t row_base_active, uint32_t row_hi_active,
    const float *ov_reg_lo, const float *ov_reg_hi, const float *qh_reg_lo,
    const float *qh_reg_hi, float gp_row_base, float gp_row_hi, float alpha,
    float scale) {
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

__device__ __forceinline__ uint32_t ring_stage(uint32_t producer_slot,
                                               uint32_t num_stages) {
  return producer_slot % num_stages;
}

__device__ __forceinline__ uint32_t ring_reuse_phase(uint32_t producer_slot,
                                                     uint32_t num_stages) {
  const uint32_t reused_slot = producer_slot - num_stages;
  return (reused_slot / num_stages) & 1U;
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
  const uint32_t warp_elected = elect_sync();

  const uint32_t v_tile = blockIdx.x;
  const uint32_t qk_head_id = blockIdx.z;
  const uint32_t head_id_base = qk_head_id * HEADS_PER_QK_HEAD;
  const uint32_t v_start = v_tile * BLOCK_V;

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
  const uint32_t v_tma_barriers = smem + OFFSET_V_TMA_BAR;
  const uint32_t h_tma_barriers = smem + OFFSET_H_TMA_BAR;
  const uint32_t qk_reuse_barriers = smem + OFFSET_QK_REUSE_BAR;
  const uint32_t h_reuse_barriers = smem + OFFSET_H_REUSE_BAR;
  const uint32_t v_reuse_barriers = smem + OFFSET_V_REUSE_BAR;
  const uint32_t qk_mma_barrier = smem + OFFSET_MMA_BAR;
  const uint32_t attn_ready_barriers = smem + OFFSET_ATTN_READY_BAR;
  const uint32_t head0_release_barrier = smem + OFFSET_HEAD0_RELEASE_BAR;
  const uint32_t qh_mma_barriers = smem + OFFSET_QH_MMA_BAR;
  const uint32_t ov_mma_barriers = smem + OFFSET_OV_MMA_BAR;
  const uint32_t qk_tmem_release_barrier = smem + OFFSET_QK_TMEM_RELEASE_BAR;
  const uint32_t output_tmem_release_barrier =
      smem + OFFSET_OUTPUT_TMEM_RELEASE_BAR;
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
    if (warp_elected) {
#pragma unroll
      for (uint32_t qk_stage = 0; qk_stage < QK_STAGE_COUNT; ++qk_stage) {
        mbarrier_init(qk_tma_barriers + qk_stage * 8U, 1);
        mbarrier_init(qk_reuse_barriers + qk_stage * 8U, 1);
      }
#pragma unroll
      for (uint32_t head_offset = 0; head_offset < HEADS_PER_QK_HEAD;
           ++head_offset) {
        mbarrier_init(v_tma_barriers + head_offset * 8U, 1);
        mbarrier_init(h_tma_barriers + head_offset * 8U, 1);
        mbarrier_init(h_reuse_barriers + head_offset * 8U, 1);
        mbarrier_init(v_reuse_barriers + head_offset * 8U, 1);
        mbarrier_init(attn_ready_barriers + head_offset * 8U, NUM_CUDA_WARPS);
      }
      mbarrier_init(qk_mma_barrier, 1);
      mbarrier_init(head0_release_barrier, NUM_CUDA_WARPS);
      for (uint32_t head_offset = 0; head_offset < HEADS_PER_QK_HEAD;
           ++head_offset) {
        mbarrier_init(qh_mma_barriers + head_offset * 8U, 1);
        mbarrier_init(ov_mma_barriers + head_offset * 8U, 1);
      }
      mbarrier_init(qk_tmem_release_barrier, NUM_CUDA_WARPS);
      mbarrier_init(output_tmem_release_barrier, NUM_CUDA_WARPS);
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
    if (blockIdx.y < total_num_chunks_u && warp_elected) {
      const int2 first_chunk_meta =
          reinterpret_cast<const int2 *>(chunk_indices_ptr)[blockIdx.y];
      const uint32_t first_seq_id = static_cast<uint32_t>(first_chunk_meta.x);
      const uint32_t first_chunk_id = static_cast<uint32_t>(first_chunk_meta.y);
      const int32_t first_chunk_start_i32 = static_cast<int32_t>(
          cu_seqlens_ptr[first_seq_id] +
          static_cast<int64_t>(first_chunk_id) * static_cast<int64_t>(BLOCK_T));
      // Step 1: seed the first resident chunk's q/k stage and both per-head
      // h/v stages so MMA/CUDA can enter steady state without an initial gap.
      tma_load_qk_stage(q_smem, k_smem, &q_tmap, &k_tmap, first_chunk_start_i32,
                        qk_head_id, qk_head_id, qk_tma_barriers);
      const int32_t first_v_chunk_start_i32 =
          static_cast<int32_t>(blockIdx.y * BLOCK_T);
      for (uint32_t head_offset = 0; head_offset < HEADS_PER_QK_HEAD;
           ++head_offset) {
        const uint32_t head_id = head_id_base + head_offset;
        const uint32_t h_outer = blockIdx.y * NUM_OUTPUT_HEADS + head_id;
        tma_load_4d(v_smem + head_offset * V_SMEM_SIZE, &v_tmap, 0,
                    first_v_chunk_start_i32, v_tile, head_id,
                    v_tma_barriers + head_offset * 8U);
        tma_load_4d(h_smem + head_offset * H_SMEM_SIZE, &h_tmap, 0, v_start, 0,
                    h_outer, h_tma_barriers + head_offset * 8U);
        mbarrier_arrive_expect_tx(v_tma_barriers + head_offset * 8U,
                                  V_SMEM_SIZE);
        mbarrier_arrive_expect_tx(h_tma_barriers + head_offset * 8U,
                                  H_SMEM_SIZE);
      }
      uint32_t qk_prod_slot = 1U;

      for (uint32_t global_chunk_id = blockIdx.y, chunk_iter = 0U;
           global_chunk_id < total_num_chunks_u;
           global_chunk_id += gridDim.y, ++chunk_iter) {
        const uint32_t stage_phase = chunk_iter & 1U;
        const uint32_t next_global_chunk_id = global_chunk_id + gridDim.y;
        if (next_global_chunk_id >= total_num_chunks_u) {
          continue;
        }

        // Step 1 overlap: keep the next chunk resident in SMEM while the
        // current chunk is consuming TMEM. Each ring slot is refilled only
        // after the corresponding reuse barrier says the old contents are dead.
        const int2 next_chunk_meta = reinterpret_cast<const int2 *>(
            chunk_indices_ptr)[next_global_chunk_id];
        const uint32_t next_seq_id = static_cast<uint32_t>(next_chunk_meta.x);
        const uint32_t next_chunk_id = static_cast<uint32_t>(next_chunk_meta.y);
        const int32_t next_chunk_start_i32 = static_cast<int32_t>(
            cu_seqlens_ptr[next_seq_id] + static_cast<int64_t>(next_chunk_id) *
                                              static_cast<int64_t>(BLOCK_T));
        const int32_t next_v_chunk_start_i32 =
            static_cast<int32_t>(next_global_chunk_id * BLOCK_T);
        const uint32_t next_h_outer_base =
            next_global_chunk_id * NUM_OUTPUT_HEADS;
        const uint32_t next_qk_stage = ring_stage(qk_prod_slot, QK_STAGE_COUNT);
        if (qk_prod_slot >= QK_STAGE_COUNT) {
          mbarrier_wait(qk_reuse_barriers + next_qk_stage * 8U,
                        ring_reuse_phase(qk_prod_slot, QK_STAGE_COUNT));
        }
        tma_load_qk_stage(q_smem + next_qk_stage * Q_SMEM_SIZE,
                          k_smem + next_qk_stage * K_SMEM_SIZE, &q_tmap,
                          &k_tmap, next_chunk_start_i32, qk_head_id, qk_head_id,
                          qk_tma_barriers + next_qk_stage * 8U);
        ++qk_prod_slot;

        const uint32_t next_head0_id = head_id_base + HEAD0_STAGE_INDEX;
        mbarrier_wait(h_reuse_barriers + HEAD0_STAGE_INDEX * 8U, stage_phase);
        tma_load_4d(h_smem + HEAD0_STAGE_INDEX * H_SMEM_SIZE, &h_tmap, 0,
                    v_start, 0, next_h_outer_base + next_head0_id,
                    h_tma_barriers + HEAD0_STAGE_INDEX * 8U);
        mbarrier_arrive_expect_tx(h_tma_barriers + HEAD0_STAGE_INDEX * 8U,
                                  H_SMEM_SIZE);
        mbarrier_wait(v_reuse_barriers + HEAD0_STAGE_INDEX * 8U, stage_phase);
        tma_load_4d(v_smem + HEAD0_STAGE_INDEX * V_SMEM_SIZE, &v_tmap, 0,
                    next_v_chunk_start_i32, v_tile, next_head0_id,
                    v_tma_barriers + HEAD0_STAGE_INDEX * 8U);
        mbarrier_arrive_expect_tx(v_tma_barriers + HEAD0_STAGE_INDEX * 8U,
                                  V_SMEM_SIZE);

        const uint32_t next_head1_id = head_id_base + HEAD1_STAGE_INDEX;
        mbarrier_wait(h_reuse_barriers + HEAD1_STAGE_INDEX * 8U, stage_phase);
        tma_load_4d(h_smem + HEAD1_STAGE_INDEX * H_SMEM_SIZE, &h_tmap, 0,
                    v_start, 0, next_h_outer_base + next_head1_id,
                    h_tma_barriers + HEAD1_STAGE_INDEX * 8U);
        mbarrier_arrive_expect_tx(h_tma_barriers + HEAD1_STAGE_INDEX * 8U,
                                  H_SMEM_SIZE);
        mbarrier_wait(v_reuse_barriers + HEAD1_STAGE_INDEX * 8U, stage_phase);
        tma_load_4d(v_smem + HEAD1_STAGE_INDEX * V_SMEM_SIZE, &v_tmap, 0,
                    next_v_chunk_start_i32, v_tile, next_head1_id,
                    v_tma_barriers + HEAD1_STAGE_INDEX * 8U);
        mbarrier_arrive_expect_tx(v_tma_barriers + HEAD1_STAGE_INDEX * 8U,
                                  V_SMEM_SIZE);
      }
    }
  } else if (warp_id == MMA_WARP) {
    for (uint32_t global_chunk_id = blockIdx.y, chunk_iter = 0U;
         global_chunk_id < total_num_chunks_u;
         global_chunk_id += gridDim.y, ++chunk_iter) {
      const uint32_t qk_stage = chunk_iter % QK_STAGE_COUNT;
      const uint32_t qk_phase = chunk_iter & 1U;
      const uint32_t q_stage_smem = q_smem + qk_stage * Q_SMEM_SIZE;
      const uint32_t k_stage_smem = k_smem + qk_stage * K_SMEM_SIZE;
      const uint32_t qk_tma_barrier = qk_tma_barriers + qk_stage * 8U;
      const uint32_t qk_tma_stage_phase = (chunk_iter / QK_STAGE_COUNT) & 1U;

      if (chunk_iter > 0U) {
        // Cross-chunk alias gate: the next chunk's QK rebuild touches cols
        // [0..63], which still hold the prior chunk's head1 QH alias until the
        // prior chunk finishes its final OUT+QH TMEM read.
        mbarrier_wait(output_tmem_release_barrier, 1U ^ qk_phase);
      }

      if (warp_elected) {
        // Step 2: build QK in TMEM once the q/k stage is visible.
        mbarrier_wait(qk_tma_barrier, qk_tma_stage_phase);
        tcgen05_fence_after_thread_sync();
        mma_swizzled<NUM_SWIZZLE_ATOMS>(QK_TMEM_COL, q_stage_smem,
                                        k_stage_smem);
        tcgen05_commit(qk_mma_barrier);
      }

#pragma unroll
      for (uint32_t head_offset = 0; head_offset < HEADS_PER_QK_HEAD;
           ++head_offset) {
        const uint32_t v_stage_smem = v_smem + head_offset * V_SMEM_SIZE;
        const uint32_t h_stage_smem = h_smem + head_offset * H_SMEM_SIZE;
        const uint32_t attn_stage_tmem =
            ATTN_TMEM_BASE_COL + head_offset * ATTN_TMEM_HEAD_STRIDE;
        const uint32_t qh_stage_tmem = qh_tmem_col_for_head(head_offset);
        const uint32_t attn_stage_barrier =
            attn_ready_barriers + head_offset * 8U;
        const uint32_t v_stage_barrier = v_tma_barriers + head_offset * 8U;
        const uint32_t h_stage_barrier = h_tma_barriers + head_offset * 8U;
        const uint32_t qh_stage_barrier = qh_mma_barriers + head_offset * 8U;
        const uint32_t ov_stage_barrier = ov_mma_barriers + head_offset * 8U;

        if (warp_elected) {
          if (head_offset == HEAD1_STAGE_INDEX) {
            // Step 5 gate: head1 QH writes into cols [0..127], reusing this
            // chunk's [QK | ATTN0] region. Wait until:
            //   1. CUDA has finished reading QK from TMEM
            //   2. head0 QH build has completed
            //   3. head0 OV into the shared OUT bank has completed
            // Only then is it safe to reuse those TMEM columns and return the
            // head0 h/v SMEM stages to the TMA producer.
            mbarrier_wait(qk_tmem_release_barrier, qk_phase);
            mbarrier_wait(ov_mma_barriers + HEAD0_STAGE_INDEX * 8U, qk_phase);
            mbarrier_wait(qh_mma_barriers + HEAD0_STAGE_INDEX * 8U, qk_phase);
            tcgen05_fence_before_thread_sync();
            mbarrier_arrive(h_reuse_barriers + HEAD0_STAGE_INDEX * 8U);
            mbarrier_arrive(v_reuse_barriers + HEAD0_STAGE_INDEX * 8U);
          }
          // Step 3 / 5: build this head's QH tile once its h stage is visible.
          mbarrier_wait(h_stage_barrier, qk_phase);
          tcgen05_fence_after_thread_sync();
          mma_swizzled_qh_64x128(qh_stage_tmem, q_stage_smem, h_stage_smem);
          tcgen05_commit(qh_stage_barrier);
          mbarrier_wait(v_stage_barrier, qk_phase);
          mbarrier_wait(attn_stage_barrier, qk_phase);
          if (head_offset == HEAD1_STAGE_INDEX) {
            // Step 5 release: once the QH1 build itself has completed, the
            // q-stage input and h1 SMEM input are no longer needed. CUDA will
            // later read QH1 from TMEM, but it will not touch those SMEM stages.
            mbarrier_wait(qh_stage_barrier, qk_phase);
            tcgen05_fence_before_thread_sync();
            mbarrier_arrive(h_reuse_barriers + HEAD1_STAGE_INDEX * 8U);
            mbarrier_arrive(qk_reuse_barriers + qk_stage * 8U);
            // Step 6 gate: head0 and head1 share the OUT bank, so head1 OV
            // cannot start until CUDA has finished the final head0 OUT+QH read.
            mbarrier_wait(head0_release_barrier, qk_phase);
          }
          // Step 4 / 7: once both ATTN and V are visible, consume them into the
          // shared OUT bank for this head.
          tcgen05_fence_after_thread_sync();
          mma_attn_v_tmem_64x128(OUTPUT_TMEM_COL, attn_stage_tmem,
                                 v_stage_smem);
          tcgen05_commit(ov_stage_barrier);
          if (head_offset + 1U == HEADS_PER_QK_HEAD) {
            // Step 7 release: after head1 OV completes, no later MMA reuses v1.
            mbarrier_wait(ov_stage_barrier, qk_phase);
            tcgen05_fence_before_thread_sync();
            mbarrier_arrive(v_reuse_barriers + HEAD1_STAGE_INDEX * 8U);
          }
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

      // Step 2: wait for the QK MMA commit, then read QK from TMEM into CUDA
      // registers. After this point CUDA no longer needs QK in TMEM, so this
      // satisfies the QK-consumer side of the later head1 alias gate.
      mbarrier_wait(qk_mma_barrier, QK_MMA_PHASE ^ qk_phase);
      tcgen05_fence_after_thread_sync();

      float qk_reg_lo[REGS_PER_FRAGMENT];
      float qk_reg_hi[REGS_PER_FRAGMENT];
      load_qk_fragment(qk_reg_lo, qk_reg_hi);
      tcgen05_fence_before_thread_sync();
      if (warp_elected) {
        mbarrier_arrive(qk_tmem_release_barrier);
      }
      if (chunk_iter > 0U) {
        // Cross-chunk alias gate: cols [64..127] are ATTN0 for this chunk, but
        // they were head1 QH for the prior chunk. Wait until the prior chunk's
        // final OUT+QH TMEM read has completed before overwriting that region.
        mbarrier_wait(output_tmem_release_barrier, 1U ^ qk_phase);
        tcgen05_fence_after_thread_sync();
      }

      // Step 3: materialize ATTN0 from the resident QK registers and publish it
      // for MMA via `attn_ready_barriers[head0]`.
      materialize_attn_tmem_from_regs(
          ATTN_TMEM_BASE_COL + HEAD0_STAGE_INDEX * ATTN_TMEM_HEAD_STRIDE,
          qk_reg_lo, qk_reg_hi, g_neg_smem_ptr + HEAD0_STAGE_INDEX * BLOCK_T,
          row_base_limit, row_hi_limit, lane_col);
      tcgen05_wait_st();
      tcgen05_fence_before_thread_sync();
      if (warp_elected) {
        mbarrier_arrive(attn_ready_barriers + HEAD0_STAGE_INDEX * 8U);
      }

      // Step 4: materialize ATTN1 from the same QK registers while MMA can
      // already be working on head0.
      materialize_attn_tmem_from_regs(
          ATTN_TMEM_BASE_COL + HEAD1_STAGE_INDEX * ATTN_TMEM_HEAD_STRIDE,
          qk_reg_lo, qk_reg_hi, g_neg_smem_ptr + HEAD1_STAGE_INDEX * BLOCK_T,
          row_base_limit, row_hi_limit, lane_col);
      tcgen05_wait_st();
      tcgen05_fence_before_thread_sync();
      if (warp_elected) {
        mbarrier_arrive(attn_ready_barriers + HEAD1_STAGE_INDEX * 8U);
      }

#pragma unroll
      for (uint32_t head_offset = 0; head_offset < HEADS_PER_QK_HEAD;
           ++head_offset) {
        const uint32_t head_id = head_id_base + head_offset;
        const uint32_t qh_stage_tmem = qh_tmem_col_for_head(head_offset);
        const uint32_t read_release_barrier =
            head_offset + 1U < HEADS_PER_QK_HEAD ? head0_release_barrier
                                                 : output_tmem_release_barrier;
        const uint32_t qh_stage_barrier = qh_mma_barriers + head_offset * 8U;
        const uint32_t ov_stage_barrier = ov_mma_barriers + head_offset * 8U;
        const float *gp_head_ptr = g_pos_smem_ptr + head_offset * BLOCK_T;
        const float gp_row_base = gp_head_ptr[row_base];
        const float gp_row_hi = gp_head_ptr[row_hi];
        const float alpha = g_alpha_smem_ptr[head_offset];

        // Step 6 / 7: wait until both QH and OUT are ready for this head, then
        // drain TMEM to registers and finally store the result to GMEM.
        mbarrier_wait(ov_stage_barrier, qk_phase);
        mbarrier_wait(qh_stage_barrier, qk_phase);
        tcgen05_fence_after_thread_sync();

        nv_bfloat16 *row_base_o_ptr =
            o_ptr + (((chunk_start + static_cast<int64_t>(row_base)) *
                          NUM_OUTPUT_HEADS +
                      head_id) *
                         VALUE_DIM +
                     v_start);
        nv_bfloat16 *row_hi_o_ptr = row_base_o_ptr + ROW_PAIR_OUTPUT_STRIDE;

        if (full_chunk) {
#pragma unroll
          for (uint32_t fragment_pair = 0;
               fragment_pair < OUTPUT_FRAGMENT_PAIRS; ++fragment_pair) {
            const uint32_t col_base = fragment_pair * BLOCK_T;
            float ov_reg_lo[REGS_PER_FRAGMENT];
            float ov_reg_hi[REGS_PER_FRAGMENT];
            float qh_reg_lo[REGS_PER_FRAGMENT];
            float qh_reg_hi[REGS_PER_FRAGMENT];
            load_output_fragment_pair(OUTPUT_TMEM_COL, qh_stage_tmem, col_base,
                                      ov_reg_lo, ov_reg_hi, qh_reg_lo,
                                      qh_reg_hi);
            if (fragment_pair + 1U == OUTPUT_FRAGMENT_PAIRS) {
              // Step 6 / 7 read-release: after the final OUT+QH fragment has
              // been read into registers, this head's TMEM storage is dead.
              // The following GMEM stores use only registers.
              tcgen05_fence_before_thread_sync();
              __syncwarp();
              if (warp_elected) {
                mbarrier_arrive(read_release_barrier);
              }
            }
            store_output_fragment_pair<true>(
                row_base_o_ptr, row_hi_o_ptr, col_base, lane_col, 0U, 0U,
                ov_reg_lo, ov_reg_hi, qh_reg_lo, qh_reg_hi, gp_row_base,
                gp_row_hi, alpha, scale);
          }
        } else {
          const uint32_t row_base_active = row_base < chunk_len;
          const uint32_t row_hi_active = row_hi < chunk_len;
#pragma unroll
          for (uint32_t fragment_pair = 0;
               fragment_pair < OUTPUT_FRAGMENT_PAIRS; ++fragment_pair) {
            const uint32_t col_base = fragment_pair * BLOCK_T;
            float ov_reg_lo[REGS_PER_FRAGMENT];
            float ov_reg_hi[REGS_PER_FRAGMENT];
            float qh_reg_lo[REGS_PER_FRAGMENT];
            float qh_reg_hi[REGS_PER_FRAGMENT];
            load_output_fragment_pair(OUTPUT_TMEM_COL, qh_stage_tmem, col_base,
                                      ov_reg_lo, ov_reg_hi, qh_reg_lo,
                                      qh_reg_hi);
            if (fragment_pair + 1U == OUTPUT_FRAGMENT_PAIRS) {
              // Step 6 / 7 read-release: after the final OUT+QH fragment has
              // been read into registers, this head's TMEM storage is dead.
              // The following GMEM stores use only registers.
              tcgen05_fence_before_thread_sync();
              __syncwarp();
              if (warp_elected) {
                mbarrier_arrive(read_release_barrier);
              }
            }
            store_output_fragment_pair<false>(
                row_base_o_ptr, row_hi_o_ptr, col_base, lane_col,
                row_base_active, row_hi_active, ov_reg_lo, ov_reg_hi, qh_reg_lo,
                qh_reg_hi, gp_row_base, gp_row_hi, alpha, scale);
          }
        }
      }
    }
  }

  tcgen05_fence_before_thread_sync();
  __syncthreads();
  if (warp_id == MMA_WARP) {
    tcgen05_dealloc(0, MAX_COLUMNS);
  }
}

void launch_o_v1(TensorView q_chunks, TensorView k_chunks, TensorView v_new,
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
  auto q_tmap = encode_qk_tma(q_chunks.data_ptr(), total_num_tokens,
                              NUM_QK_HEADS, HEAD_DIM);
  auto k_tmap = encode_qk_tma(k_chunks.data_ptr(), total_num_tokens,
                              NUM_QK_HEADS, HEAD_DIM);
  auto v_tmap = encode_v_tma(v_new.data_ptr(), v_num_tokens);
  auto h_tmap = encode_tma(h.data_ptr(),
                           static_cast<uint64_t>(h.size(0)) * NUM_OUTPUT_HEADS,
                           VALUE_DIM, HEAD_DIM);

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
