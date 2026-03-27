#include "gdn_decode_utils.h"
#include "tvm_ffi_utils.h"

#include <cstdint>
#include <cuda_bf16.h>
#include <math.h>

namespace {

constexpr int kHeadSize = gdn_decode::kHeadSize;
constexpr int64_t kNumQHeads = gdn_decode::kNumQHeads;
constexpr int64_t kNumKHeads = gdn_decode::kNumKHeads;
constexpr int64_t kNumVHeads = gdn_decode::kNumVHeads;

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
    const __nv_bfloat16 *__restrict__ q,
    const __nv_bfloat16 *__restrict__ k,
    const __nv_bfloat16 *__restrict__ v,
    const float *__restrict__ state,
    const float *__restrict__ A_log,
    const __nv_bfloat16 *__restrict__ a,
    const float *__restrict__ dt_bias,
    const __nv_bfloat16 *__restrict__ b,
    // DIFF: extra params for variable-length sequences (decode has neither)
    const int64_t *__restrict__ cu_seqlens,
    float scale,
    __nv_bfloat16 *__restrict__ output,
    float *__restrict__ new_state,
    // DIFF: decode doesn't need this — grid is B * Hv * tiles
    int64_t num_seqs) {

  const int64_t block_linear = static_cast<int64_t>(blockIdx.x);
  const int64_t tile_idx = block_linear % kNumVTiles;
  const int64_t bh = block_linear / kNumVTiles;

  // DIFF: decode has batch_idx here. Same math, just renamed because we're
  // iterating over variable-length sequences instead of fixed batch elements.
  const int64_t seq_idx = bh / kNumVHeads;
  const int64_t hv_idx = bh % kNumVHeads;
  const float negated_exp_A_log = -expf(A_log[hv_idx]);

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & (kWarpSize - 1);

  const int64_t v_idx = tile_idx * kRowsPerBlock + warp_id;

  const int64_t q_head = hv_idx / kQGroupSize;
  const int64_t k_head = hv_idx / kKGroupSize;

  // DIFF: decode computes state_row_base from v_offset = hv_base * kHeadSize + v_idx,
  // then state_row_base = v_offset * kHeadSize. Same formula, but here we use
  // seq_idx (constant for the whole block) instead of batch_idx, because the
  // state tensor is [N, Hv, V, K] indexed by sequence, not by token.
  const int64_t state_row_base =
      ((seq_idx * kNumVHeads + hv_idx) * kHeadSize + v_idx) * kHeadSize;

  // DIFF: decode declares `const float4 state_vec` (immutable, one token).
  // Prefill declares `float4 state_vec` (mutable) because it's updated each
  // iteration and carried across the loop in registers.
  float4 state_vec;
  // DIFF: decode always has state (required). Prefill allows nullptr for the
  // first-ever sequence with no prior state — zero-initialize in that case.
  if (state != nullptr) {
    state_vec =
        reinterpret_cast<const float4 *>(state + state_row_base)[lane];
  } else {
    state_vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  }

  // DIFF: decode has no loop and no cu_seqlens. It processes exactly one token
  // per block. Prefill uses cu_seqlens to find the token range for this
  // sequence, then loops over all of them.
  const int64_t seq_start = cu_seqlens[seq_idx];
  const int64_t seq_end = cu_seqlens[seq_idx + 1];
  const int kk_base = lane * kElemsPerLane;

  // DIFF: the entire for-loop is new. Decode's per-token body runs once;
  // prefill wraps the same body in this loop.
  for (int64_t t = seq_start; t < seq_end; t++) {
    // DIFF: decode computes hv_base = batch_idx * kNumVHeads + hv_idx (constant).
    // Prefill: t replaces batch_idx since layout is [T, H, ...] not [B, H, ...].
    // hv_base changes each iteration because t changes.
    const int64_t hv_base = t * kNumVHeads + hv_idx;

    // DIFF: decode uses batch_idx in q_base/k_base (constant).
    // Prefill uses t (changes per iteration). Same formula otherwise.
    const int64_t q_base = (t * kNumQHeads + q_head) * kHeadSize;
    const int64_t k_base = (t * kNumKHeads + k_head) * kHeadSize;
    const float4 q_vec = LoadBf16x4GlobalNc(q + q_base + kk_base);
    const float4 k_vec = LoadBf16x4GlobalNc(k + k_base + kk_base);

    // DIFF: decode computes v_offset once before the body (constant).
    // Prefill recomputes it each iteration because hv_base changes with t.
    const int64_t v_offset = hv_base * kHeadSize + v_idx;
    const float v_scalar = __bfloat162float(v[v_offset]);
    const float beta = Sigmoid(__bfloat162float(b[hv_base]));

    const float g =
        ComputeGdnScalars(negated_exp_A_log, a[hv_base], dt_bias[hv_idx]);

    // --- Everything below is identical to decode ---

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

    // DIFF: decode writes into a separate `updated_vec`, then stores it to
    // new_state immediately. Prefill writes back into `state_vec` so the
    // updated state carries forward to the next iteration in registers.
    state_vec.x = fmaf(k_vec.x, delta, old_state_vec.x);
    state_vec.y = fmaf(k_vec.y, delta, old_state_vec.y);
    state_vec.z = fmaf(k_vec.z, delta, old_state_vec.z);
    state_vec.w = fmaf(k_vec.w, delta, old_state_vec.w);

    // DIFF: decode computes output from `updated_vec`.
    // Prefill uses `state_vec` (same data, just a different variable name).
    float out_partial = q_vec.x * state_vec.x;
    out_partial = fmaf(q_vec.y, state_vec.y, out_partial);
    out_partial = fmaf(q_vec.z, state_vec.z, out_partial);
    out_partial = fmaf(q_vec.w, state_vec.w, out_partial);

    const float out_acc = WarpAllReduceSum(out_partial);
    if (lane == 0) {
      output[v_offset] = __float2bfloat16_rn(scale * out_acc);
    }
  }

  // DIFF: decode writes new_state right after the state update (mid-kernel).
  // Prefill writes only ONCE here after the loop ends — only the final state
  // matters. This avoids (seq_len - 1) unnecessary global memory writes.
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

  CHECK_DIM(3, q);
  CHECK_DIM(3, k);
  CHECK_DIM(3, v);
  CHECK_DIM(1, A_log);
  CHECK_DIM(2, a);
  CHECK_DIM(1, dt_bias);
  CHECK_DIM(2, b);
  CHECK_DIM(1, cu_seqlens);
  CHECK_DIM(3, output);
  CHECK_DIM(4, new_state);

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

  const float *state_ptr = nullptr;
  if (state_opt.has_value()) {
    TensorView state = state_opt.value();
    CHECK_INPUT(state);
    CHECK_DIM(4, state);
    TVM_FFI_CHECK(state.dtype() == dl_float32, TypeError);
    TVM_FFI_CHECK(state.size(0) == num_seqs && state.size(1) == kNumVHeads &&
                      state.size(2) == kHeadSize && state.size(3) == kHeadSize,
                  ValueError)
        << "state shape mismatch";
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
