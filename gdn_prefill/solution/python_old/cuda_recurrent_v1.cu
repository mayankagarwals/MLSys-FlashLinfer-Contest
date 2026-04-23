/*
 * GDN Prefill — Recurrent CUDA Kernel (v1)
 *
 * Token-by-token recurrent processing, identical math to gdn_decode_kernel_7
 * but adapted for the prefill layout with variable-length sequences.
 *
 * Key differences from decode kernel (gdn_decode_kernel_7.cu):
 *
 * 1. Tensor layout
 *    Decode: q/k [B, 1, Hq/Hk, K], v [B, 1, Hv, V], a/b [B, 1, Hv]  (4D/3D, T=1)
 *    Prefill: q/k [T, Hq/Hk, K], v [T, Hv, V], a/b [T, Hv]  (3D/2D, packed)
 *    Prefill adds cu_seqlens [N+1] to mark sequence boundaries.
 *
 * 2. Grid indexing
 *    Decode: block_linear → (batch_idx, hv_idx, tile_idx), batch_idx is constant.
 *    Prefill: block_linear → (seq_idx, hv_idx, tile_idx), seq_idx is constant.
 *
 * 3. Token loop
 *    Decode: no loop — processes exactly one token per block.
 *    Prefill: loops over all tokens in [cu_seqlens[seq_idx], cu_seqlens[seq_idx+1]).
 *
 * 4. Address computation
 *    Decode: hv_base = batch_idx * kNumVHeads + hv_idx (constant per block).
 *            q_base/k_base use batch_idx (constant).
 *    Prefill: hv_base = t * kNumVHeads + hv_idx (changes each iteration).
 *             q_base/k_base use token index t (changes each iteration).
 *
 * 5. State management
 *    Decode: state_vec is `const float4` (immutable, one token). Writes updated_vec
 *            to new_state immediately after the state update.
 *    Prefill: state_vec is `float4` (mutable). Updated in-place each iteration and
 *             carried across the loop in registers. Written to new_state only ONCE
 *             after the loop ends — avoids (seq_len - 1) unnecessary global writes.
 *
 * 6. Optional initial state
 *    Decode: state is always required (non-null).
 *    Prefill: state can be nullptr (first-ever sequence) → zero-initialize.
 *
 * 7. State tensor indexing
 *    Decode: state_row_base from v_offset = (batch_idx * Hv + hv_idx) * V + v_idx.
 *    Prefill: state_row_base from (seq_idx * Hv + hv_idx) * V + v_idx.
 *             Same formula but indexed by sequence, not by token.
 */

#include "cuda_utils.h"
#include "tvm_ffi_utils.h"

#include <cstdint>
#include <cuda_bf16.h>
#include <math.h>

namespace {

constexpr int kHeadSize = gdn_prefill::kHeadSize;
constexpr int64_t kNumQHeads = gdn_prefill::kNumQHeads;
constexpr int64_t kNumKHeads = gdn_prefill::kNumKHeads;
constexpr int64_t kNumVHeads = gdn_prefill::kNumVHeads;

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 4;
constexpr int kRowsPerBlock = kWarpsPerBlock;
constexpr int kNumThreads = kWarpSize * kWarpsPerBlock;
constexpr int kNumVTiles = kHeadSize / kRowsPerBlock;
constexpr int kElemsPerLane = kHeadSize / kWarpSize;

constexpr int64_t kQGroupSize = kNumVHeads / kNumQHeads;
constexpr int64_t kKGroupSize = kNumVHeads / kNumKHeads;
constexpr unsigned kFullWarpMask = 0xffffffffu;

static_assert(kHeadSize == 128, "expects head size 128");
static_assert(kNumThreads == 128, "expects 128 threads per block");
static_assert(kHeadSize % kWarpSize == 0,
              "head size must be divisible by warp size");
static_assert(kRowsPerBlock == 4, "expects four rows per block");
static_assert(kHeadSize % kRowsPerBlock == 0,
              "head size must be divisible by rows per block");
static_assert(kElemsPerLane == 4, "expects four elements per lane");

__device__ __forceinline__ float SoftplusStable(float x) {
  const float abs_x = fabsf(x);
  return log1pf(expf(-abs_x)) + fmaxf(x, 0.0f);
}

__device__ __forceinline__ float Sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float ComputeGdnScalars(float negated_exp_A_log,
                                                   __nv_bfloat16 a_val,
                                                   float dt_bias_val) {
  const float x = __bfloat162float(a_val) + dt_bias_val;
  const float softplus_x = SoftplusStable(x);
  return expf(negated_exp_A_log * softplus_x);
}

__device__ __forceinline__ float WarpAllReduceSum(float value) {
#pragma unroll
  for (int mask = kWarpSize / 2; mask > 0; mask >>= 1) {
    value += __shfl_xor_sync(kFullWarpMask, value, mask);
  }
  return value;
}

__device__ __forceinline__ float4
LoadBf16x4GlobalNc(const __nv_bfloat16 *__restrict__ ptr) {
  float4 out;
  asm volatile("{\n\t"
               ".reg .b16 h<4>;\n\t"
               "ld.global.nc.L1::evict_first.v4.b16 {h0, h1, h2, h3}, [%4];\n\t"
               "cvt.rn.f32.bf16 %0, h0;\n\t"
               "cvt.rn.f32.bf16 %1, h1;\n\t"
               "cvt.rn.f32.bf16 %2, h2;\n\t"
               "cvt.rn.f32.bf16 %3, h3;\n\t"
               "}\n"
               : "=f"(out.x), "=f"(out.y), "=f"(out.z), "=f"(out.w)
               : "l"(ptr));
  return out;
}

__device__ __forceinline__ void
StoreF32x4RelaxedNoAllocate(float *addr, const float4 &value) {
  asm volatile(
      "st.relaxed.cta.global.L1::no_allocate.v4.f32 [%0], {%1, %2, %3, %4};"
      :
      : "l"(addr), "f"(value.x), "f"(value.y), "f"(value.z), "f"(value.w));
}

__global__ void GdnPrefillRecurrentKernel(
    const __nv_bfloat16 *__restrict__ q,          // [T, Hq=4, K=128]   bf16
    const __nv_bfloat16 *__restrict__ k,          // [T, Hk=4, K=128]   bf16
    const __nv_bfloat16 *__restrict__ v,          // [T, Hv=8, V=128]   bf16
    const float *__restrict__ state,              // [N, Hv, V, K]      fp32 or nullptr
    const float *__restrict__ A_log,              // [Hv=8]             fp32
    const __nv_bfloat16 *__restrict__ a,          // [T, Hv=8]          bf16
    const float *__restrict__ dt_bias,            // [Hv=8]             fp32
    const __nv_bfloat16 *__restrict__ b,          // [T, Hv=8]          bf16
    const int64_t *__restrict__ cu_seqlens,       // [N+1]              int64
    float scale,                                  // scalar             fp32
    __nv_bfloat16 *__restrict__ output,           // [T, Hv=8, V=128]   bf16
    float *__restrict__ new_state,                // [N, Hv, V, K]      fp32
    int64_t num_seqs) {

  const int64_t block_linear = static_cast<int64_t>(blockIdx.x);
  const int64_t tile_idx = block_linear % kNumVTiles;
  const int64_t bh = block_linear / kNumVTiles;

  const int64_t seq_idx = bh / kNumVHeads;
  const int64_t hv_idx = bh % kNumVHeads;
  const float negated_exp_A_log = -expf(A_log[hv_idx]);

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & (kWarpSize - 1);

  const int64_t v_idx = tile_idx * kRowsPerBlock + warp_id;

  const int64_t q_head = hv_idx / kQGroupSize;
  const int64_t k_head = hv_idx / kKGroupSize;

  const int64_t state_row_base =
      ((seq_idx * kNumVHeads + hv_idx) * kHeadSize + v_idx) * kHeadSize;

  float4 state_vec;
  if (state != nullptr) {
    state_vec =
        reinterpret_cast<const float4 *>(state + state_row_base)[lane];
  } else {
    state_vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  }

  const int64_t seq_start = cu_seqlens[seq_idx];
  const int64_t seq_end = cu_seqlens[seq_idx + 1];
  const int kk_base = lane * kElemsPerLane;

  for (int64_t t = seq_start; t < seq_end; t++) {
    const int64_t hv_base = t * kNumVHeads + hv_idx;

    const int64_t q_base = (t * kNumQHeads + q_head) * kHeadSize;
    const int64_t k_base = (t * kNumKHeads + k_head) * kHeadSize;
    const float4 q_vec = LoadBf16x4GlobalNc(q + q_base + kk_base);
    const float4 k_vec = LoadBf16x4GlobalNc(k + k_base + kk_base);

    const int64_t v_offset = hv_base * kHeadSize + v_idx;
    const float v_scalar = __bfloat162float(v[v_offset]);
    const float beta = Sigmoid(__bfloat162float(b[hv_base]));

    const float g =
        ComputeGdnScalars(negated_exp_A_log, a[hv_base], dt_bias[hv_idx]);

    float4 old_state_vec;
    old_state_vec.x = g * state_vec.x;
    old_state_vec.y = g * state_vec.y;
    old_state_vec.z = g * state_vec.z;
    old_state_vec.w = g * state_vec.w;

    float old_v_partial = k_vec.x * old_state_vec.x;
    old_v_partial = fmaf(k_vec.y, old_state_vec.y, old_v_partial);
    old_v_partial = fmaf(k_vec.z, old_state_vec.z, old_v_partial);
    old_v_partial = fmaf(k_vec.w, old_state_vec.w, old_v_partial);

    const float old_v = WarpAllReduceSum(old_v_partial);
    const float delta = beta * (v_scalar - old_v);

    state_vec.x = fmaf(k_vec.x, delta, old_state_vec.x);
    state_vec.y = fmaf(k_vec.y, delta, old_state_vec.y);
    state_vec.z = fmaf(k_vec.z, delta, old_state_vec.z);
    state_vec.w = fmaf(k_vec.w, delta, old_state_vec.w);

    float out_partial = q_vec.x * state_vec.x;
    out_partial = fmaf(q_vec.y, state_vec.y, out_partial);
    out_partial = fmaf(q_vec.z, state_vec.z, out_partial);
    out_partial = fmaf(q_vec.w, state_vec.w, out_partial);

    const float out_acc = WarpAllReduceSum(out_partial);
    if (lane == 0) {
      output[v_offset] = __float2bfloat16_rn(scale * out_acc);
    }
  }

  float *addr = new_state + state_row_base + lane * kElemsPerLane;
  StoreF32x4RelaxedNoAllocate(addr, state_vec);
}

__host__ __forceinline__ float ResolveScale(double scale) {
  float scale_f = static_cast<float>(scale);
  if (scale_f == 0.0f) {
    scale_f = 1.0f / sqrtf(static_cast<float>(kHeadSize));
  }
  return scale_f;
}

void RunGdnPrefillRecurrentV1(
    TensorView q, TensorView k, TensorView v,
    ffi::Optional<TensorView> state_opt, TensorView A_log, TensorView a,
    TensorView dt_bias, TensorView b, TensorView cu_seqlens, double scale,
    TensorView output, TensorView new_state) {

  gdn_prefill::ValidateShapesAndTypes(q, k, v, A_log, a, dt_bias, b,
                                      cu_seqlens, output, new_state);

  const int64_t num_seqs = cu_seqlens.size(0) - 1;

  const float *state_ptr = nullptr;
  if (state_opt.has_value()) {
    TensorView state = state_opt.value();
    gdn_prefill::ValidateState(state, num_seqs);
    CHECK_DEVICE(q, state);
    state_ptr = static_cast<const float *>(state.data_ptr());
  }

  const float scale_f = ResolveScale(scale);
  ffi::CUDADeviceGuard guard(q.device().device_id);
  const cudaStream_t stream = get_cuda_stream(q.device());

  const dim3 grid(num_seqs * kNumVHeads * kNumVTiles, 1, 1);
  GdnPrefillRecurrentKernel<<<grid, kNumThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(q.data_ptr()),
      static_cast<const __nv_bfloat16 *>(k.data_ptr()),
      static_cast<const __nv_bfloat16 *>(v.data_ptr()),
      state_ptr,
      static_cast<const float *>(A_log.data_ptr()),
      static_cast<const __nv_bfloat16 *>(a.data_ptr()),
      static_cast<const float *>(dt_bias.data_ptr()),
      static_cast<const __nv_bfloat16 *>(b.data_ptr()),
      static_cast<const int64_t *>(cu_seqlens.data_ptr()),
      scale_f,
      static_cast<__nv_bfloat16 *>(output.data_ptr()),
      static_cast<float *>(new_state.data_ptr()),
      num_seqs);

  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    TVM_FFI_THROW(RuntimeError)
        << "GdnPrefillRecurrentKernel launch failed: "
        << cudaGetErrorString(err);
  }
}

} // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(gdn_prefill_recurrent_v1,
                               RunGdnPrefillRecurrentV1);
