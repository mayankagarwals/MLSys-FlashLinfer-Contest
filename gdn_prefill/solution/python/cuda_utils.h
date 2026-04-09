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
#include <cudaTypedefs.h>

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

// https://github.com/NVIDIA/cutlass/blob/v4.3.2/include/cute/arch/copy_sm90_desc.hpp#L193-L197
constexpr uint64_t EVICT_NORMAL = 0x1000000000000000;
constexpr uint64_t EVICT_FIRST  = 0x12F0000000000000;
constexpr uint64_t EVICT_LAST   = 0x14F0000000000000;

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

  cuTensorMapEncodeTiled(tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
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

  cuTensorMapEncodeTiled(tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
      rank, (void *)ptr, globalDim, globalStrides,
      boxDim, elementStrides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);
}

__device__ inline
void prefetch_tensormap(const void *tmap_ptr) {
  asm volatile("prefetch.tensormap [%0];" :: "l"(tmap_ptr));
}

// TMA device-side loads
__device__ __forceinline__
void tma_load_2d(uint32_t smem_addr, const void *tmap_ptr,
                 int x, int y,
                 uint32_t mbar_addr, uint64_t cache_policy = EVICT_NORMAL) {
  asm volatile(
    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint "
    "[%0], [%1, {%2, %3}], [%4], %5;"
    :: "r"(smem_addr), "l"(tmap_ptr), "r"(x), "r"(y),
       "r"(mbar_addr), "l"(cache_policy) : "memory");
}

__device__ __forceinline__
void tma_load_3d(uint32_t smem_addr, const void *tmap_ptr,
                 int x, int y, int z,
                 uint32_t mbar_addr, uint64_t cache_policy = EVICT_NORMAL) {
  asm volatile(
    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint "
    "[%0], [%1, {%2, %3, %4}], [%5], %6;"
    :: "r"(smem_addr), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
       "r"(mbar_addr), "l"(cache_policy) : "memory");
}

__device__ __forceinline__
void tma_load_4d(uint32_t smem_addr, const void *tmap_ptr,
                 int x, int y, int z, int w,
                 uint32_t mbar_addr, uint64_t cache_policy = EVICT_NORMAL) {
  asm volatile(
    "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint "
    "[%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
    :: "r"(smem_addr), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w),
       "r"(mbar_addr), "l"(cache_policy) : "memory");
}

// TMA device-side stores
__device__ __forceinline__
void tma_store_2d(const void *tmap_ptr, uint32_t smem_addr, 
                 int x, int y,
                 uint64_t cache_policy = EVICT_NORMAL) {
  asm volatile(
    "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group.L2::cache_hint "
    "[%0, {%1, %2}], [%3], %4;"
    :: "l"(tmap_ptr), "r"(x), "r"(y),
       "r"(smem_addr), "l"(cache_policy) : "memory");
}

__device__ __forceinline__
void tma_store_3d(const void *tmap_ptr, uint32_t smem_addr, 
                 int x, int y, int z,
                 uint64_t cache_policy = EVICT_NORMAL) {
  asm volatile(
    "cp.async.bulk.tensor.3d.global.shared::cta.bulk_group.L2::cache_hint "
    "[%0, {%1, %2, %3}], [%4], %5;"
    :: "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
       "r"(smem_addr), "l"(cache_policy) : "memory");
}

__device__ __forceinline__
void tma_store_4d(const void *tmap_ptr, uint32_t smem_addr, 
                 int x, int y, int z, int w,
                 uint64_t cache_policy = EVICT_NORMAL) {
  asm volatile(
    "cp.async.bulk.tensor.4d.global.shared::cta.bulk_group.L2::cache_hint "
    "[%0, {%1, %2, %3, %4}], [%5], %6;"
    :: "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w),
       "r"(smem_addr), "l"(cache_policy) : "memory");
}

__device__ __forceinline__
void cp_async_bulk_commit_group() {
  asm volatile("cp.async.bulk.commit_group;");
}

template <int N>
__device__ __forceinline__
void cp_async_bulk_wait_group_read() {
  asm volatile("cp.async.bulk.wait_group.read %0;" :: "n"(N));
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
void fence_mbarrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;");
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

__device__ __forceinline__
void mbarrier_arrive(uint32_t mbar_addr) {
  asm volatile("mbarrier.arrive.relaxed.cta.shared::cta.b64 _, [%0];" :: "r"(mbar_addr) : "memory");
}

__device__ __forceinline__
void mbarrier_arrive_expect_tx(uint32_t mbar_addr, int size) {
  asm volatile(
    "mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64 _, [%0], %1;"
    :: "r"(mbar_addr), "r"(size) : "memory");
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
void tcgen05_dealloc(uint32_t taddr, int columns) {
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
__device__ __forceinline__
constexpr uint32_t make_tcgen05_idesc(int MMA_M, int MMA_N) {
  return (1U << 4U)   // dtype = FP32
       | (1U << 7U)   // atype = BF16
       | (1U << 10U)  // btype = BF16
       | ((uint32_t)(MMA_N >> 3U) << 17U)
       | ((uint32_t)(MMA_M >> 4U) << 24U);
}

__device__ __forceinline__
void tcgen05_mma(uint32_t taddr, uint64_t a_desc, uint64_t b_desc,
                 uint32_t i_desc, int enable_input_d) {
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "setp.ne.b32 p, %4, 0;\n\t"
    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t"
    "}"
    :: "r"(taddr), "l"(a_desc), "l"(b_desc), "r"(i_desc), "r"(enable_input_d));
}

__device__ __forceinline__
void tcgen05_mma_tmem(uint32_t taddr, uint32_t a_tmem, uint64_t b_desc,
                      uint32_t i_desc, int enable_input_d) {
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "setp.ne.b32 p, %4, 0;\n\t"
    "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n\t"
    "}"
    :: "r"(taddr), "r"(a_tmem), "l"(b_desc), "r"(i_desc), "r"(enable_input_d));
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

template <auto> struct ptx_str{};

enum class SHAPE { _32x32b, _16x128b, _16x256b, _16x32bx2 };
template<> struct ptx_str<SHAPE::_32x32b>  { static constexpr char value[] = ".32x32b"; };
template<> struct ptx_str<SHAPE::_16x128b> { static constexpr char value[] = ".16x128b"; };
template<> struct ptx_str<SHAPE::_16x256b> { static constexpr char value[] = ".16x256b"; };
template<> struct ptx_str<SHAPE::_16x32bx2>{ static constexpr char value[] = ".16x32bx2"; };

// each 32x32b tile uses 1 register per thread
// each 16x256b tile uses 4 registers per thread
template <SHAPE shape> struct _regs_per_tile{};
template<> struct _regs_per_tile<SHAPE::_32x32b>  { static constexpr int value = 1; };
template<> struct _regs_per_tile<SHAPE::_16x128b> { static constexpr int value = 2; };
template<> struct _regs_per_tile<SHAPE::_16x256b> { static constexpr int value = 4; };
template<> struct _regs_per_tile<SHAPE::_16x32bx2>{ static constexpr int value = 1; };

template <SHAPE shape, int NUM>
__device__ inline
void tcgen05_ld(void *tmp_, uint32_t row, uint32_t col) {
  float *tmp = reinterpret_cast<float *>(tmp_);
  uint32_t addr = (row << 16u) | col;

  constexpr int NUM_REGS = _regs_per_tile<shape>::value * NUM;

  if constexpr (NUM_REGS == 1)
  asm volatile("tcgen05.ld.sync.aligned%2.x%3.b32 {%0}, [%1];"
              : "=f"(tmp[0])
              : "r"(addr), "C"(ptx_str<shape>::value), "n"(NUM));
  if constexpr (NUM_REGS == 2)
  asm volatile("tcgen05.ld.sync.aligned%3.x%4.b32 {%0, %1}, [%2];"
              : "=f"(tmp[0]), "=f"(tmp[1])
              : "r"(addr), "C"(ptx_str<shape>::value), "n"(NUM));
  if constexpr (NUM_REGS == 4)
  asm volatile("tcgen05.ld.sync.aligned%5.x%6.b32 "
              "{%0, %1, %2, %3}, [%4];"
              : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3])
              : "r"(addr), "C"(ptx_str<shape>::value), "n"(NUM));
  if constexpr (NUM_REGS == 8)
  asm volatile("tcgen05.ld.sync.aligned%9.x%10.b32 "
              "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7}, [%8];"
              : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3]), "=f"(tmp[4]), "=f"(tmp[5]), "=f"(tmp[6]), "=f"(tmp[7])
              : "r"(addr), "C"(ptx_str<shape>::value), "n"(NUM));
  if constexpr (NUM_REGS == 16)
  asm volatile("tcgen05.ld.sync.aligned%17.x%18.b32 "
              "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
              "  %8,  %9, %10, %11, %12, %13, %14, %15}, [%16];"
              : "=f"(tmp[ 0]), "=f"(tmp[ 1]), "=f"(tmp[ 2]), "=f"(tmp[ 3]), "=f"(tmp[ 4]), "=f"(tmp[ 5]), "=f"(tmp[ 6]), "=f"(tmp[ 7]),
                "=f"(tmp[ 8]), "=f"(tmp[ 9]), "=f"(tmp[10]), "=f"(tmp[11]), "=f"(tmp[12]), "=f"(tmp[13]), "=f"(tmp[14]), "=f"(tmp[15])
              : "r"(addr), "C"(ptx_str<shape>::value), "n"(NUM));
  if constexpr (NUM_REGS == 32)
  asm volatile("tcgen05.ld.sync.aligned%33.x%34.b32 "
              "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
              "  %8,  %9, %10, %11, %12, %13, %14, %15, "
              " %16, %17, %18, %19, %20, %21, %22, %23, "
              " %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
              : "=f"(tmp[ 0]), "=f"(tmp[ 1]), "=f"(tmp[ 2]), "=f"(tmp[ 3]), "=f"(tmp[ 4]), "=f"(tmp[ 5]), "=f"(tmp[ 6]), "=f"(tmp[ 7]),
                "=f"(tmp[ 8]), "=f"(tmp[ 9]), "=f"(tmp[10]), "=f"(tmp[11]), "=f"(tmp[12]), "=f"(tmp[13]), "=f"(tmp[14]), "=f"(tmp[15]),
                "=f"(tmp[16]), "=f"(tmp[17]), "=f"(tmp[18]), "=f"(tmp[19]), "=f"(tmp[20]), "=f"(tmp[21]), "=f"(tmp[22]), "=f"(tmp[23]),
                "=f"(tmp[24]), "=f"(tmp[25]), "=f"(tmp[26]), "=f"(tmp[27]), "=f"(tmp[28]), "=f"(tmp[29]), "=f"(tmp[30]), "=f"(tmp[31])
              : "r"(addr), "C"(ptx_str<shape>::value), "n"(NUM));
  if constexpr (NUM_REGS == 64)
  asm volatile("tcgen05.ld.sync.aligned%65.x%66.b32 "
              "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
              "  %8,  %9, %10, %11, %12, %13, %14, %15, "
              " %16, %17, %18, %19, %20, %21, %22, %23, "
              " %24, %25, %26, %27, %28, %29, %30, %31, "
              " %32, %33, %34, %35, %36, %37, %38, %39, "
              " %40, %41, %42, %43, %44, %45, %46, %47, "
              " %48, %49, %50, %51, %52, %53, %54, %55, "
              " %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
              : "=f"(tmp[ 0]), "=f"(tmp[ 1]), "=f"(tmp[ 2]), "=f"(tmp[ 3]), "=f"(tmp[ 4]), "=f"(tmp[ 5]), "=f"(tmp[ 6]), "=f"(tmp[ 7]),
                "=f"(tmp[ 8]), "=f"(tmp[ 9]), "=f"(tmp[10]), "=f"(tmp[11]), "=f"(tmp[12]), "=f"(tmp[13]), "=f"(tmp[14]), "=f"(tmp[15]),
                "=f"(tmp[16]), "=f"(tmp[17]), "=f"(tmp[18]), "=f"(tmp[19]), "=f"(tmp[20]), "=f"(tmp[21]), "=f"(tmp[22]), "=f"(tmp[23]),
                "=f"(tmp[24]), "=f"(tmp[25]), "=f"(tmp[26]), "=f"(tmp[27]), "=f"(tmp[28]), "=f"(tmp[29]), "=f"(tmp[30]), "=f"(tmp[31]),
                "=f"(tmp[32]), "=f"(tmp[33]), "=f"(tmp[34]), "=f"(tmp[35]), "=f"(tmp[36]), "=f"(tmp[37]), "=f"(tmp[38]), "=f"(tmp[39]),
                "=f"(tmp[40]), "=f"(tmp[41]), "=f"(tmp[42]), "=f"(tmp[43]), "=f"(tmp[44]), "=f"(tmp[45]), "=f"(tmp[46]), "=f"(tmp[47]),
                "=f"(tmp[48]), "=f"(tmp[49]), "=f"(tmp[50]), "=f"(tmp[51]), "=f"(tmp[52]), "=f"(tmp[53]), "=f"(tmp[54]), "=f"(tmp[55]),
                "=f"(tmp[56]), "=f"(tmp[57]), "=f"(tmp[58]), "=f"(tmp[59]), "=f"(tmp[60]), "=f"(tmp[61]), "=f"(tmp[62]), "=f"(tmp[63])
              : "r"(addr), "C"(ptx_str<shape>::value), "n"(NUM));
  if constexpr (NUM_REGS == 128)
  asm volatile("tcgen05.ld.sync.aligned%129.x%130.b32 "
              "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
              "  %8,  %9, %10, %11, %12, %13, %14, %15, "
              " %16, %17, %18, %19, %20, %21, %22, %23, "
              " %24, %25, %26, %27, %28, %29, %30, %31, "
              " %32, %33, %34, %35, %36, %37, %38, %39, "
              " %40, %41, %42, %43, %44, %45, %46, %47, "
              " %48, %49, %50, %51, %52, %53, %54, %55, "
              " %56, %57, %58, %59, %60, %61, %62, %63, "
              " %64, %65, %66, %67, %68, %69, %70, %71, "
              " %72, %73, %74, %75, %76, %77, %78, %79, "
              " %80, %81, %82, %83, %84, %85, %86, %87, "
              " %88, %89, %90, %91, %92, %93, %94, %95, "
              " %96, %97, %98, %99,%100,%101,%102,%103, "
              "%104,%105,%106,%107,%108,%109,%110,%111, "
              "%112,%113,%114,%115,%116,%117,%118,%119, "
              "%120,%121,%122,%123,%124,%125,%126,%127}, [%128];"
              : "=f"(tmp[ 0]), "=f"(tmp[ 1]), "=f"(tmp[ 2]), "=f"(tmp[ 3]), "=f"(tmp[ 4]), "=f"(tmp[ 5]), "=f"(tmp[ 6]), "=f"(tmp[ 7]),
                "=f"(tmp[ 8]), "=f"(tmp[ 9]), "=f"(tmp[10]), "=f"(tmp[11]), "=f"(tmp[12]), "=f"(tmp[13]), "=f"(tmp[14]), "=f"(tmp[15]),
                "=f"(tmp[16]), "=f"(tmp[17]), "=f"(tmp[18]), "=f"(tmp[19]), "=f"(tmp[20]), "=f"(tmp[21]), "=f"(tmp[22]), "=f"(tmp[23]),
                "=f"(tmp[24]), "=f"(tmp[25]), "=f"(tmp[26]), "=f"(tmp[27]), "=f"(tmp[28]), "=f"(tmp[29]), "=f"(tmp[30]), "=f"(tmp[31]),
                "=f"(tmp[32]), "=f"(tmp[33]), "=f"(tmp[34]), "=f"(tmp[35]), "=f"(tmp[36]), "=f"(tmp[37]), "=f"(tmp[38]), "=f"(tmp[39]),
                "=f"(tmp[40]), "=f"(tmp[41]), "=f"(tmp[42]), "=f"(tmp[43]), "=f"(tmp[44]), "=f"(tmp[45]), "=f"(tmp[46]), "=f"(tmp[47]),
                "=f"(tmp[48]), "=f"(tmp[49]), "=f"(tmp[50]), "=f"(tmp[51]), "=f"(tmp[52]), "=f"(tmp[53]), "=f"(tmp[54]), "=f"(tmp[55]),
                "=f"(tmp[56]), "=f"(tmp[57]), "=f"(tmp[58]), "=f"(tmp[59]), "=f"(tmp[60]), "=f"(tmp[61]), "=f"(tmp[62]), "=f"(tmp[63]),
                "=f"(tmp[64]), "=f"(tmp[65]), "=f"(tmp[66]), "=f"(tmp[67]), "=f"(tmp[68]), "=f"(tmp[69]), "=f"(tmp[70]), "=f"(tmp[71]),
                "=f"(tmp[72]), "=f"(tmp[73]), "=f"(tmp[74]), "=f"(tmp[75]), "=f"(tmp[76]), "=f"(tmp[77]), "=f"(tmp[78]), "=f"(tmp[79]),
                "=f"(tmp[80]), "=f"(tmp[81]), "=f"(tmp[82]), "=f"(tmp[83]), "=f"(tmp[84]), "=f"(tmp[85]), "=f"(tmp[86]), "=f"(tmp[87]),
                "=f"(tmp[88]), "=f"(tmp[89]), "=f"(tmp[90]), "=f"(tmp[91]), "=f"(tmp[92]), "=f"(tmp[93]), "=f"(tmp[94]), "=f"(tmp[95]),
                "=f"(tmp[96]), "=f"(tmp[97]), "=f"(tmp[98]), "=f"(tmp[99]), "=f"(tmp[100]),"=f"(tmp[101]),"=f"(tmp[102]),"=f"(tmp[103]),
                "=f"(tmp[104]),"=f"(tmp[105]),"=f"(tmp[106]),"=f"(tmp[107]),"=f"(tmp[108]),"=f"(tmp[109]),"=f"(tmp[110]),"=f"(tmp[111]),
                "=f"(tmp[112]),"=f"(tmp[113]),"=f"(tmp[114]),"=f"(tmp[115]),"=f"(tmp[116]),"=f"(tmp[117]),"=f"(tmp[118]),"=f"(tmp[119]),
                "=f"(tmp[120]),"=f"(tmp[121]),"=f"(tmp[122]),"=f"(tmp[123]),"=f"(tmp[124]),"=f"(tmp[125]),"=f"(tmp[126]),"=f"(tmp[127])
              : "r"(addr), "C"(ptx_str<shape>::value), "n"(NUM));
}


template <SHAPE shape, int NUM>
__device__ inline
void tcgen05_st(uint32_t row, uint32_t col, const void *tmp_) {
  const float *tmp = reinterpret_cast<const float *>(tmp_);
  uint32_t addr = (row << 16u) | col;

  constexpr int NUM_REGS = _regs_per_tile<shape>::value * NUM;

  if constexpr (NUM_REGS == 1)
  asm volatile("tcgen05.st.sync.aligned%2.x%3.b32 [%1], {%0};"
              :: "f"(tmp[0]),
              "r"(addr), "C"(ptx_str<shape>::value), "n"(NUM));
  if constexpr (NUM_REGS == 2)
  asm volatile("tcgen05.st.sync.aligned%3.x%4.b32 [%2], {%0, %1};"
              :: "f"(tmp[0]), "f"(tmp[1]),
              "r"(addr), "C"(ptx_str<shape>::value), "n"(NUM));
  if constexpr (NUM_REGS == 4)
  asm volatile("tcgen05.st.sync.aligned%5.x%6.b32 [%4], {%0, %1, %2, %3};"
              :: "f"(tmp[0]), "f"(tmp[1]), "f"(tmp[2]), "f"(tmp[3]),
              "r"(addr), "C"(ptx_str<shape>::value), "n"(NUM));
  if constexpr (NUM_REGS == 8)
  asm volatile("tcgen05.st.sync.aligned%9.x%10.b32 [%8], "
              "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7};"
              :: "f"(tmp[0]), "f"(tmp[1]), "f"(tmp[2]), "f"(tmp[3]), "f"(tmp[4]), "f"(tmp[5]), "f"(tmp[6]), "f"(tmp[7]),
              "r"(addr), "C"(ptx_str<shape>::value), "n"(NUM));
  if constexpr (NUM_REGS == 16)
  asm volatile("tcgen05.st.sync.aligned%17.x%18.b32 [%16], "
              "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
              "  %8,  %9, %10, %11, %12, %13, %14, %15};"
              :: "f"(tmp[ 0]), "f"(tmp[ 1]), "f"(tmp[ 2]), "f"(tmp[ 3]), "f"(tmp[ 4]), "f"(tmp[ 5]), "f"(tmp[ 6]), "f"(tmp[ 7]),
                 "f"(tmp[ 8]), "f"(tmp[ 9]), "f"(tmp[10]), "f"(tmp[11]), "f"(tmp[12]), "f"(tmp[13]), "f"(tmp[14]), "f"(tmp[15]),
              "r"(addr), "C"(ptx_str<shape>::value), "n"(NUM));
  if constexpr (NUM_REGS == 32)
  asm volatile("tcgen05.st.sync.aligned%33.x%34.b32 [%32], "
              "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
              "  %8,  %9, %10, %11, %12, %13, %14, %15, "
              " %16, %17, %18, %19, %20, %21, %22, %23, "
              " %24, %25, %26, %27, %28, %29, %30, %31};"
              :: "f"(tmp[ 0]), "f"(tmp[ 1]), "f"(tmp[ 2]), "f"(tmp[ 3]), "f"(tmp[ 4]), "f"(tmp[ 5]), "f"(tmp[ 6]), "f"(tmp[ 7]),
                 "f"(tmp[ 8]), "f"(tmp[ 9]), "f"(tmp[10]), "f"(tmp[11]), "f"(tmp[12]), "f"(tmp[13]), "f"(tmp[14]), "f"(tmp[15]),
                 "f"(tmp[16]), "f"(tmp[17]), "f"(tmp[18]), "f"(tmp[19]), "f"(tmp[20]), "f"(tmp[21]), "f"(tmp[22]), "f"(tmp[23]),
                 "f"(tmp[24]), "f"(tmp[25]), "f"(tmp[26]), "f"(tmp[27]), "f"(tmp[28]), "f"(tmp[29]), "f"(tmp[30]), "f"(tmp[31]),
              "r"(addr), "C"(ptx_str<shape>::value), "n"(NUM));
  if constexpr (NUM_REGS == 64)
  asm volatile("tcgen05.st.sync.aligned%65.x%66.b32 [%64], "
              "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
              "  %8,  %9, %10, %11, %12, %13, %14, %15, "
              " %16, %17, %18, %19, %20, %21, %22, %23, "
              " %24, %25, %26, %27, %28, %29, %30, %31, "
              " %32, %33, %34, %35, %36, %37, %38, %39, "
              " %40, %41, %42, %43, %44, %45, %46, %47, "
              " %48, %49, %50, %51, %52, %53, %54, %55, "
              " %56, %57, %58, %59, %60, %61, %62, %63};"
              :: "f"(tmp[ 0]), "f"(tmp[ 1]), "f"(tmp[ 2]), "f"(tmp[ 3]), "f"(tmp[ 4]), "f"(tmp[ 5]), "f"(tmp[ 6]), "f"(tmp[ 7]),
                 "f"(tmp[ 8]), "f"(tmp[ 9]), "f"(tmp[10]), "f"(tmp[11]), "f"(tmp[12]), "f"(tmp[13]), "f"(tmp[14]), "f"(tmp[15]),
                 "f"(tmp[16]), "f"(tmp[17]), "f"(tmp[18]), "f"(tmp[19]), "f"(tmp[20]), "f"(tmp[21]), "f"(tmp[22]), "f"(tmp[23]),
                 "f"(tmp[24]), "f"(tmp[25]), "f"(tmp[26]), "f"(tmp[27]), "f"(tmp[28]), "f"(tmp[29]), "f"(tmp[30]), "f"(tmp[31]),
                 "f"(tmp[32]), "f"(tmp[33]), "f"(tmp[34]), "f"(tmp[35]), "f"(tmp[36]), "f"(tmp[37]), "f"(tmp[38]), "f"(tmp[39]),
                 "f"(tmp[40]), "f"(tmp[41]), "f"(tmp[42]), "f"(tmp[43]), "f"(tmp[44]), "f"(tmp[45]), "f"(tmp[46]), "f"(tmp[47]),
                 "f"(tmp[48]), "f"(tmp[49]), "f"(tmp[50]), "f"(tmp[51]), "f"(tmp[52]), "f"(tmp[53]), "f"(tmp[54]), "f"(tmp[55]),
                 "f"(tmp[56]), "f"(tmp[57]), "f"(tmp[58]), "f"(tmp[59]), "f"(tmp[60]), "f"(tmp[61]), "f"(tmp[62]), "f"(tmp[63]),
              "r"(addr), "C"(ptx_str<shape>::value), "n"(NUM));
  if constexpr (NUM_REGS == 128)
  asm volatile("tcgen05.st.sync.aligned%129.x%130.b32 [%128], "
              "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
              "  %8,  %9, %10, %11, %12, %13, %14, %15, "
              " %16, %17, %18, %19, %20, %21, %22, %23, "
              " %24, %25, %26, %27, %28, %29, %30, %31, "
              " %32, %33, %34, %35, %36, %37, %38, %39, "
              " %40, %41, %42, %43, %44, %45, %46, %47, "
              " %48, %49, %50, %51, %52, %53, %54, %55, "
              " %56, %57, %58, %59, %60, %61, %62, %63, "
              " %64, %65, %66, %67, %68, %69, %70, %71, "
              " %72, %73, %74, %75, %76, %77, %78, %79, "
              " %80, %81, %82, %83, %84, %85, %86, %87, "
              " %88, %89, %90, %91, %92, %93, %94, %95, "
              " %96, %97, %98, %99,%100,%101,%102,%103, "
              "%104,%105,%106,%107,%108,%109,%110,%111, "
              "%112,%113,%114,%115,%116,%117,%118,%119, "
              "%120,%121,%122,%123,%124,%125,%126,%127};"
              :: "f"(tmp[ 0]), "f"(tmp[ 1]), "f"(tmp[ 2]), "f"(tmp[ 3]), "f"(tmp[ 4]), "f"(tmp[ 5]), "f"(tmp[ 6]), "f"(tmp[ 7]),
                 "f"(tmp[ 8]), "f"(tmp[ 9]), "f"(tmp[10]), "f"(tmp[11]), "f"(tmp[12]), "f"(tmp[13]), "f"(tmp[14]), "f"(tmp[15]),
                 "f"(tmp[16]), "f"(tmp[17]), "f"(tmp[18]), "f"(tmp[19]), "f"(tmp[20]), "f"(tmp[21]), "f"(tmp[22]), "f"(tmp[23]),
                 "f"(tmp[24]), "f"(tmp[25]), "f"(tmp[26]), "f"(tmp[27]), "f"(tmp[28]), "f"(tmp[29]), "f"(tmp[30]), "f"(tmp[31]),
                 "f"(tmp[32]), "f"(tmp[33]), "f"(tmp[34]), "f"(tmp[35]), "f"(tmp[36]), "f"(tmp[37]), "f"(tmp[38]), "f"(tmp[39]),
                 "f"(tmp[40]), "f"(tmp[41]), "f"(tmp[42]), "f"(tmp[43]), "f"(tmp[44]), "f"(tmp[45]), "f"(tmp[46]), "f"(tmp[47]),
                 "f"(tmp[48]), "f"(tmp[49]), "f"(tmp[50]), "f"(tmp[51]), "f"(tmp[52]), "f"(tmp[53]), "f"(tmp[54]), "f"(tmp[55]),
                 "f"(tmp[56]), "f"(tmp[57]), "f"(tmp[58]), "f"(tmp[59]), "f"(tmp[60]), "f"(tmp[61]), "f"(tmp[62]), "f"(tmp[63]),
                 "f"(tmp[64]), "f"(tmp[65]), "f"(tmp[66]), "f"(tmp[67]), "f"(tmp[68]), "f"(tmp[69]), "f"(tmp[70]), "f"(tmp[71]),
                 "f"(tmp[72]), "f"(tmp[73]), "f"(tmp[74]), "f"(tmp[75]), "f"(tmp[76]), "f"(tmp[77]), "f"(tmp[78]), "f"(tmp[79]),
                 "f"(tmp[80]), "f"(tmp[81]), "f"(tmp[82]), "f"(tmp[83]), "f"(tmp[84]), "f"(tmp[85]), "f"(tmp[86]), "f"(tmp[87]),
                 "f"(tmp[88]), "f"(tmp[89]), "f"(tmp[90]), "f"(tmp[91]), "f"(tmp[92]), "f"(tmp[93]), "f"(tmp[94]), "f"(tmp[95]),
                 "f"(tmp[96]), "f"(tmp[97]), "f"(tmp[98]), "f"(tmp[99]), "f"(tmp[100]),"f"(tmp[101]),"f"(tmp[102]),"f"(tmp[103]),
                 "f"(tmp[104]),"f"(tmp[105]),"f"(tmp[106]),"f"(tmp[107]),"f"(tmp[108]),"f"(tmp[109]),"f"(tmp[110]),"f"(tmp[111]),
                 "f"(tmp[112]),"f"(tmp[113]),"f"(tmp[114]),"f"(tmp[115]),"f"(tmp[116]),"f"(tmp[117]),"f"(tmp[118]),"f"(tmp[119]),
                 "f"(tmp[120]),"f"(tmp[121]),"f"(tmp[122]),"f"(tmp[123]),"f"(tmp[124]),"f"(tmp[125]),"f"(tmp[126]),"f"(tmp[127])
              "r"(addr), "C"(ptx_str<shape>::value), "n"(NUM));
}

// Specialized tcgen05 ld/st for 16x32bx2 shape (requires ncols argument in PTX).
// addr = pre-composed TMEM address (taddr + tmem_offset).
// Loads/stores 16 tiles × 1 reg/tile = 16 uint32 registers, ncols=16.
__device__ __forceinline__
void tcgen05_ld_16x32bx2(uint32_t (&out)[16], int addr) {
  asm volatile(
    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32 "
    "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16 + 0], 16;"
    : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3]),
      "=r"(out[4]), "=r"(out[5]), "=r"(out[6]), "=r"(out[7]),
      "=r"(out[8]), "=r"(out[9]), "=r"(out[10]), "=r"(out[11]),
      "=r"(out[12]), "=r"(out[13]), "=r"(out[14]), "=r"(out[15])
    : "r"(addr));
  asm volatile("tcgen05.wait::ld.sync.aligned;");
}

__device__ __forceinline__
void tcgen05_st_16x32bx2(int addr, const uint32_t (&in)[16]) {
  asm volatile(
    "{\n\t.reg .pred p;\n\tmov.pred p, -1;\n\t"
    "@p tcgen05.st.sync.aligned.16x32bx2.x16.b32 [%0 + 0], 16, "
    "{%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};\n\t}"
    :: "r"(addr),
       "r"(in[0]), "r"(in[1]), "r"(in[2]), "r"(in[3]),
       "r"(in[4]), "r"(in[5]), "r"(in[6]), "r"(in[7]),
       "r"(in[8]), "r"(in[9]), "r"(in[10]), "r"(in[11]),
       "r"(in[12]), "r"(in[13]), "r"(in[14]), "r"(in[15])
    : "memory");
  asm volatile("tcgen05.wait::st.sync.aligned;");
}

__device__ __forceinline__ void tcgen05_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;"); }
__device__ __forceinline__ void tcgen05_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;"); }

__device__ __forceinline__
void tcgen05_fence_before_thread_sync() {
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

__device__ __forceinline__
void tcgen05_fence_after_thread_sync() {
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

template <uint32_t barrier>
__device__ __forceinline__
void bar_sync(uint32_t count) {
  asm volatile("bar.sync %0, %1;" :: "n"(barrier), "r"(count));
}

template <int num>
__device__ __forceinline__
void ldmatrix(uint32_t *data, uint32_t addr) {
  static_assert(num == 1 || num == 2 || num == 4);
  if constexpr (num == 1)
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];"
                : "=r"(data[0])
                : "r"(addr));
  else if constexpr (num == 2)
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
                : "=r"(data[0]), "=r"(data[1])
                : "r"(addr));
  else if constexpr (num == 4)
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                : "=r"(data[0]), "=r"(data[1]), "=r"(data[2]), "=r"(data[3])
                : "r"(addr));
}

template <int num>
__device__ __forceinline__
void ldmatrix_trans(uint32_t *data, uint32_t addr) {
  static_assert(num == 1 || num == 2 || num == 4);
  if constexpr (num == 1)
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];"
                : "=r"(data[0])
                : "r"(addr));
  else if constexpr (num == 2)
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
                : "=r"(data[0]), "=r"(data[1])
                : "r"(addr));
  else if constexpr (num == 4)
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
                : "=r"(data[0]), "=r"(data[1]), "=r"(data[2]), "=r"(data[3])
                : "r"(addr));
}

template <int num>
__device__ __forceinline__
void stmatrix(uint32_t addr, uint32_t *data) {
  static_assert(num == 1 || num == 2 || num == 4);
  if constexpr (num == 1)
    asm volatile("stmatrix.sync.aligned.m8n8.x1.shared.b16 [%1], {%0};"
                :: "r"(data[0]), "r"(addr));
  else if constexpr (num == 2)
    asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%2], {%0, %1};"
                :: "r"(data[0]), "r"(data[1]), "r"(addr));
  else if constexpr (num == 4)
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%4], {%0, %1, %2, %3};"
                :: "r"(data[0]), "r"(data[1]), "r"(data[2]), "r"(data[3]), "r"(addr));
}

template <int num>
__device__ __forceinline__
void stmatrix_trans(uint32_t addr, uint32_t *data) {
  static_assert(num == 1 || num == 2 || num == 4);
  if constexpr (num == 1)
    asm volatile("stmatrix.sync.aligned.m8n8.x1.trans.shared.b16 [%1], {%0};"
                :: "r"(data[0]), "r"(addr));
  else if constexpr (num == 2)
    asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%2], {%0, %1};"
                :: "r"(data[0]), "r"(data[1]), "r"(addr));
  else if constexpr (num == 4)
    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%4], {%0, %1, %2, %3};"
                :: "r"(data[0]), "r"(data[1]), "r"(data[2]), "r"(data[3]), "r"(addr));
}

__device__ inline
void ldg_u32x8(void *data_, const void *ptr) {
  uint32_t *data = reinterpret_cast<uint32_t *>(data_);
  asm volatile(
    "ld.global.v8.f32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
    : "=r"(data[0]), "=r"(data[1]), "=r"(data[2]), "=r"(data[3]),
      "=r"(data[4]), "=r"(data[5]), "=r"(data[6]), "=r"(data[7])
    : "l"(ptr));
}

__device__ inline
void stg_u32x8(void *ptr, const void *data_) {
  const uint32_t *data = reinterpret_cast<const uint32_t *>(data_);
  asm volatile(
    "st.global.v8.u32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
    :: "l"(ptr),
      "r"(data[0]), "r"(data[1]), "r"(data[2]), "r"(data[3]),
      "r"(data[4]), "r"(data[5]), "r"(data[6]), "r"(data[7]));
}

// using .L1::no_allocate, meant for epilogue stores (no read again).
__device__ inline
void stg_u32x8_fast(void *ptr, const void *data_) {
  const uint32_t *data = reinterpret_cast<const uint32_t *>(data_);
  asm volatile(
    "st.global.relaxed.cta.L1::no_allocate.v8.u32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
    :: "l"(ptr),
      "r"(data[0]), "r"(data[1]), "r"(data[2]), "r"(data[3]),
      "r"(data[4]), "r"(data[5]), "r"(data[6]), "r"(data[7]));
}

template <typename T>
__device__ inline
T warp_uniform(T x) { return __shfl_sync(0xFFFF'FFFF, x, 0); }

// ═══════════════════════════════════════════════════════════════════
// Shared memory address conversion
// ═══════════════════════════════════════════════════════════════════

__device__ __forceinline__ uint32_t cvt_smem_ptr(const void *ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// ═══════════════════════════════════════════════════════════════════
// mma.sync m16n8k16 (bf16 inputs → fp32 accumulator)
// ═══════════════════════════════════════════════════════════════════

__device__ __forceinline__
void mma_m16n8k16_bf16(
    float &d0, float &d1, float &d2, float &d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3) {
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
    : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
      "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

// mma.sync m16n8k8 (tf32 inputs → fp32 accumulator)
__device__ __forceinline__
void mma_m16n8k8_tf32(
    float &d0, float &d1, float &d2, float &d3,
    float a0, float a1, float a2, float a3,
    float b0, float b1,
    float c0, float c1, float c2, float c3) {
  asm volatile(
    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
    : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
    : "f"(a0), "f"(a1), "f"(a2), "f"(a3), "f"(b0), "f"(b1),
      "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

// ═══════════════════════════════════════════════════════════════════
// cp.async (legacy, non-bulk)
// ═══════════════════════════════════════════════════════════════════

__device__ __forceinline__
void cp_async_cg_128(uint32_t smem_addr, const void *gmem_ptr, bool pred) {
  uint32_t p = pred ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 0x10, %2;"
    :: "r"(smem_addr), "l"(gmem_ptr), "r"(p) : "memory");
}
