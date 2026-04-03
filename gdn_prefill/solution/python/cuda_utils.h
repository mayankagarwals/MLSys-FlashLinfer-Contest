/*
 * Shared helpers for GDN prefill CUDA kernels.
 *
 * Contains:
 *   - Tensor shape/type validation (TVM FFI)
 *   - tcgen05 (Blackwell SM_100a) tensor core utilities
 *   - TMA (Tensor Memory Access) helpers
 *
 * Prefill layout (packed variable-length sequences):
 *   q:          [total_seq_len, num_q_heads=4, head_size=128]    bf16
 *   k:          [total_seq_len, num_k_heads=4, head_size=128]    bf16
 *   v:          [total_seq_len, num_v_heads=8, head_size=128]    bf16
 *   state:      [num_seqs, num_v_heads=8, head_size=128, head_size=128]  fp32 (optional)
 *   A_log:      [num_v_heads=8]                                  fp32
 *   a:          [total_seq_len, num_v_heads=8]                   bf16
 *   dt_bias:    [num_v_heads=8]                                  fp32
 *   b:          [total_seq_len, num_v_heads=8]                   bf16
 *   cu_seqlens: [num_seqs + 1]                                   int64
 *   scale:      scalar                                           fp32
 *   output:     [total_seq_len, num_v_heads=8, head_size=128]    bf16
 *   new_state:  [num_seqs, num_v_heads=8, head_size=128, head_size=128]  fp32
 */
#pragma once

#include "tvm_ffi_utils.h"
#include <cuda_bf16.h>
#include <cstdint>
#include <cuda.h>  // For CUtensorMap

// ═══════════════════════════════════════════════════════════════════
// Tensor validation helpers
// ═══════════════════════════════════════════════════════════════════

namespace gdn_prefill {

constexpr int kHeadSize = 128;
constexpr int64_t kNumQHeads = 4;
constexpr int64_t kNumKHeads = 4;
constexpr int64_t kNumVHeads = 8;

inline void
ValidateShapesAndTypes(const TensorView &q, const TensorView &k,
                       const TensorView &v, const TensorView &A_log,
                       const TensorView &a, const TensorView &dt_bias,
                       const TensorView &b, const TensorView &cu_seqlens,
                       const TensorView &output, const TensorView &new_state) {
  CHECK_INPUT(q);
  CHECK_INPUT(k);
  CHECK_INPUT(v);
  CHECK_INPUT(A_log);
  CHECK_INPUT(a);
  CHECK_INPUT(dt_bias);
  CHECK_INPUT(b);
  CHECK_INPUT(cu_seqlens);
  CHECK_INPUT(output);
  CHECK_INPUT(new_state);

  CHECK_DIM(3, q);       // [T, Hq, K]
  CHECK_DIM(3, k);       // [T, Hk, K]
  CHECK_DIM(3, v);       // [T, Hv, V]
  CHECK_DIM(1, A_log);   // [Hv]
  CHECK_DIM(2, a);       // [T, Hv]
  CHECK_DIM(1, dt_bias); // [Hv]
  CHECK_DIM(2, b);       // [T, Hv]
  CHECK_DIM(1, cu_seqlens);
  CHECK_DIM(3, output);  // [T, Hv, V]
  CHECK_DIM(4, new_state); // [N, Hv, V, K]

  const int64_t T = q.size(0);
  const int64_t num_seqs = cu_seqlens.size(0) - 1;

  TVM_FFI_CHECK(num_seqs > 0, ValueError) << "must have at least one sequence";
  TVM_FFI_CHECK(q.size(1) == kNumQHeads && q.size(2) == kHeadSize, ValueError)
      << "q shape mismatch";
  TVM_FFI_CHECK(k.size(0) == T && k.size(1) == kNumKHeads &&
                    k.size(2) == kHeadSize,
                ValueError)
      << "k shape mismatch";
  TVM_FFI_CHECK(v.size(0) == T && v.size(1) == kNumVHeads &&
                    v.size(2) == kHeadSize,
                ValueError)
      << "v shape mismatch";
  TVM_FFI_CHECK(a.size(0) == T && a.size(1) == kNumVHeads, ValueError)
      << "a shape mismatch";
  TVM_FFI_CHECK(b.size(0) == T && b.size(1) == kNumVHeads, ValueError)
      << "b shape mismatch";

  TVM_FFI_CHECK(q.dtype() == dl_bfloat16, TypeError) << "q must be bfloat16";
  TVM_FFI_CHECK(k.dtype() == dl_bfloat16, TypeError) << "k must be bfloat16";
  TVM_FFI_CHECK(v.dtype() == dl_bfloat16, TypeError) << "v must be bfloat16";
  TVM_FFI_CHECK(a.dtype() == dl_bfloat16, TypeError) << "a must be bfloat16";
  TVM_FFI_CHECK(b.dtype() == dl_bfloat16, TypeError) << "b must be bfloat16";
  TVM_FFI_CHECK(A_log.dtype() == dl_float32, TypeError);
  TVM_FFI_CHECK(dt_bias.dtype() == dl_float32, TypeError);
  TVM_FFI_CHECK(output.dtype() == dl_bfloat16, TypeError);
  TVM_FFI_CHECK(new_state.dtype() == dl_float32, TypeError);

  TVM_FFI_CHECK(new_state.size(0) == num_seqs && new_state.size(1) == kNumVHeads &&
                    new_state.size(2) == kHeadSize && new_state.size(3) == kHeadSize,
                ValueError)
      << "new_state shape mismatch";

  CHECK_DEVICE(q, k);
  CHECK_DEVICE(q, v);
  CHECK_DEVICE(q, A_log);
  CHECK_DEVICE(q, a);
  CHECK_DEVICE(q, dt_bias);
  CHECK_DEVICE(q, b);
  CHECK_DEVICE(q, output);
  CHECK_DEVICE(q, new_state);
}

inline void ValidateState(const TensorView &state, int64_t num_seqs) {
  CHECK_INPUT(state);
  CHECK_DIM(4, state);
  TVM_FFI_CHECK(state.dtype() == dl_float32, TypeError);
  TVM_FFI_CHECK(state.size(0) == num_seqs && state.size(1) == kNumVHeads &&
                    state.size(2) == kHeadSize && state.size(3) == kHeadSize,
                ValueError)
      << "state shape mismatch";
}

} // namespace gdn_prefill

// ═══════════════════════════════════════════════════════════════════
// TMA (Tensor Memory Access) helpers
// ═══════════════════════════════════════════════════════════════════

// 2D TMA descriptor (host-side creation)
inline void init_tma_desc_2d(
    CUtensorMap *tmap,
    const __nv_bfloat16 *ptr,
    uint64_t rows, uint64_t cols,
    uint32_t box_rows, uint32_t box_cols
) {
  constexpr uint32_t rank = 2;
  uint64_t globalDim[rank] = {cols, rows};
  uint64_t globalStrides[rank - 1] = {cols * sizeof(__nv_bfloat16)};
  uint32_t boxDim[rank] = {box_cols, box_rows};
  uint32_t elementStrides[rank] = {1, 1};

  typedef CUresult (*FnType)(CUtensorMap *, CUtensorMapDataType, cuuint32_t,
      void *, const cuuint64_t *, const cuuint64_t *, const cuuint32_t *,
      const cuuint32_t *, CUtensorMapInterleave, CUtensorMapSwizzle,
      CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
  FnType pfn = nullptr;
  cudaGetDriverEntryPoint("cuTensorMapEncodeTiled",
                          (void **)&pfn, cudaEnableDefault);
  pfn(tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
      rank, (void *)ptr, globalDim, globalStrides,
      boxDim, elementStrides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

// 3D TMA for no-swizzle tile layout
inline void init_tma_desc_3d(
    CUtensorMap *tmap,
    const __nv_bfloat16 *ptr,
    uint64_t rows, uint64_t cols,
    uint32_t box_rows, uint32_t box_cols
) {
  constexpr uint32_t rank = 3;
  uint64_t globalDim[rank] = {8, rows, cols / 8};
  uint64_t globalStrides[rank - 1] = {cols * sizeof(__nv_bfloat16), 16};
  uint32_t boxDim[rank] = {8, box_rows, box_cols / 8};
  uint32_t elementStrides[rank] = {1, 1, 1};

  typedef CUresult (*FnType)(CUtensorMap *, CUtensorMapDataType, cuuint32_t,
      void *, const cuuint64_t *, const cuuint64_t *, const cuuint32_t *,
      const cuuint32_t *, CUtensorMapInterleave, CUtensorMapSwizzle,
      CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
  FnType pfn = nullptr;
  cudaGetDriverEntryPoint("cuTensorMapEncodeTiled",
                          (void **)&pfn, cudaEnableDefault);
  pfn(tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
      rank, (void *)ptr, globalDim, globalStrides,
      boxDim, elementStrides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);
}

// TMA device-side loads
__device__ __forceinline__
void tma_load_2d(uint32_t smem_addr, const void *tmap_ptr,
                 int coord_x, int coord_y, uint32_t mbar_addr) {
  asm volatile(
    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
    "[%0], [%1, {%2, %3}], [%4];"
    :: "r"(smem_addr), "l"(tmap_ptr), "r"(coord_x), "r"(coord_y),
       "r"(mbar_addr) : "memory");
}

__device__ __forceinline__
void tma_load_3d(uint32_t smem_addr, const void *tmap_ptr,
                 int coord_x, int coord_y, int coord_z, uint32_t mbar_addr) {
  asm volatile(
    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes "
    "[%0], [%1, {%2, %3, %4}], [%5];"
    :: "r"(smem_addr), "l"(tmap_ptr), "r"(coord_x), "r"(coord_y), "r"(coord_z),
       "r"(mbar_addr) : "memory");
}

// ═══════════════════════════════════════════════════════════════════
// Shared memory address helper
// ═══════════════════════════════════════════════════════════════════

__device__ __forceinline__ uint32_t cvt_smem_ptr(const void *ptr) {
  uint32_t addr;
  asm volatile("{ .reg .u64 u64addr;\n\t"
               "  cvta.to.shared.u64 u64addr, %1;\n\t"
               "  cvt.u32.u64 %0, u64addr;\n\t"
               "}" : "=r"(addr) : "l"(ptr));
  return addr;
}

// ═══════════════════════════════════════════════════════════════════
// tcgen05 descriptor encoding
// ═══════════════════════════════════════════════════════════════════

__device__ __forceinline__
constexpr uint64_t desc_encode(uint64_t x) {
  return (x & 0x3FFFFULL) >> 4ULL;
}

// Build shared memory descriptor for tcgen05.mma (no-swizzle, tile layout)
__device__ __forceinline__
uint64_t make_tcgen05_desc_noswizzle(uint32_t addr, int height, int sbo = 128) {
  const int LBO = height * 16;
  return desc_encode(addr) | (desc_encode(LBO) << 16ULL)
       | (desc_encode(sbo) << 32ULL) | (1ULL << 46ULL);
}

// ═══════════════════════════════════════════════════════════════════
// Warp election
// ═══════════════════════════════════════════════════════════════════

__device__ __forceinline__
uint32_t elect_sync() {
  uint32_t pred = 0;
  asm volatile(
    "{\n\t"
    ".reg .pred %%px;\n\t"
    "elect.sync _|%%px, %1;\n\t"
    "@%%px mov.s32 %0, 1;\n\t"
    "}"
    : "+r"(pred) : "r"(0xFFFFFFFF));
  return pred;
}

// ═══════════════════════════════════════════════════════════════════
// mbarrier helpers
// ═══════════════════════════════════════════════════════════════════

__device__ __forceinline__
void mbarrier_init(uint32_t mbar_addr, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbar_addr), "r"(count));
}

__device__ __forceinline__
void mbarrier_wait(uint32_t mbar_addr, int phase) {
  uint32_t ticks = 0x989680;
  asm volatile(
    "{\n\t"
    ".reg .pred P1;\n\t"
    "LAB_WAIT:\n\t"
    "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t"
    "@P1 bra.uni DONE;\n\t"
    "bra.uni LAB_WAIT;\n\t"
    "DONE:\n\t"
    "}"
    :: "r"(mbar_addr), "r"(phase), "r"(ticks));
}

// ═══════════════════════════════════════════════════════════════════
// tcgen05 tensor memory management
// ═══════════════════════════════════════════════════════════════════

__device__ __forceinline__
void tcgen05_alloc(uint32_t smem_result_addr, int columns) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
              :: "r"(smem_result_addr), "r"(columns));
}

__device__ __forceinline__
void tcgen05_dealloc(int taddr, int columns) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
              :: "r"(taddr), "r"(columns));
}

__device__ __forceinline__
void tcgen05_relinquish_alloc_permit() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
}

// ═══════════════════════════════════════════════════════════════════
// tcgen05.mma
// ═══════════════════════════════════════════════════════════════════

// Instruction descriptor: bf16 inputs → fp32 accumulator
template <int BLOCK_M, int BLOCK_N>
__device__ __forceinline__
constexpr uint32_t make_tcgen05_idesc() {
  return (1U << 4U)                           // dtype = FP32
       | (1U << 7U)                           // atype = BF16
       | (1U << 10U)                          // btype = BF16
       | ((uint32_t)(BLOCK_N >> 3U) << 17U)   // MMA_N
       | ((uint32_t)(BLOCK_M >> 4U) << 24U);  // MMA_M
}

__device__ __forceinline__
void tcgen05_mma(int taddr, uint64_t a_desc, uint64_t b_desc,
                 uint32_t i_desc, int enable_input_d) {
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "setp.ne.b32 p, %4, 0;\n\t"
    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t"
    "}"
    :: "r"(taddr), "l"(a_desc), "l"(b_desc), "r"(i_desc), "r"(enable_input_d));
}

// ═══════════════════════════════════════════════════════════════════
// tcgen05 commit, load, fence
// ═══════════════════════════════════════════════════════════════════

__device__ __forceinline__
void tcgen05_commit(uint32_t mbar_addr) {
  asm volatile(
    "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
    :: "r"(mbar_addr) : "memory");
}

__device__ __forceinline__
void tcgen05_ld_32x8(float (&out)[8], int taddr, int row, int col) {
  int addr = taddr + (row << 16) + col;
  asm volatile(
    "tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
    : "=f"(out[0]), "=f"(out[1]), "=f"(out[2]), "=f"(out[3]),
      "=f"(out[4]), "=f"(out[5]), "=f"(out[6]), "=f"(out[7])
    : "r"(addr));
  asm volatile("tcgen05.wait::ld.sync.aligned;");
}

__device__ __forceinline__
void tcgen05_fence() {
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}
