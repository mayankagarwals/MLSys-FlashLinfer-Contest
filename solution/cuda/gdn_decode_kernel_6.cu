#include "gdn_decode_utils.h"
#include "tvm_ffi_utils.h"

#include <cstdint>
#include <cuda_bf16.h>
#include <math.h>

namespace {

// v6: Same as v5 with __stcs streaming store for new_state.
// Bypasses L2 cache on state writes (~1MB write-once data),
// freeing L2 capacity for concurrent state reads by other blocks.

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

static_assert(kHeadSize == 128, "kernel_6 expects head size 128");
static_assert(kNumThreads == 128, "kernel_6 expects 128 threads per block");
static_assert(kHeadSize % kWarpSize == 0,
              "head size must be divisible by warp size");
static_assert(kRowsPerBlock == 4, "kernel_6 expects four rows per block");
static_assert(kHeadSize % kRowsPerBlock == 0,
              "head size must be divisible by rows per block");
static_assert(kElemsPerLane == 4, "kernel_6 expects four elements per lane");

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

__global__ void GdnDecodeKernel6(
    const __nv_bfloat16 *__restrict__ q, const __nv_bfloat16 *__restrict__ k,
    const __nv_bfloat16 *__restrict__ v, const float *__restrict__ state,
    const float *__restrict__ A_log, const __nv_bfloat16 *__restrict__ a,
    const float *__restrict__ dt_bias, const __nv_bfloat16 *__restrict__ b,
    float scale, __nv_bfloat16 *__restrict__ output,
    float *__restrict__ new_state) {

  const int64_t block_linear = static_cast<int64_t>(blockIdx.x);
  const int64_t tile_idx = block_linear % kNumVTiles;
  const int64_t bh = block_linear / kNumVTiles;

  const int64_t batch_idx = bh / kNumVHeads;
  const int64_t hv_idx = bh % kNumVHeads;
  const float negated_exp_A_log = -expf(A_log[hv_idx]);

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & (kWarpSize - 1);

  const int64_t v_idx = tile_idx * kRowsPerBlock + warp_id;

  const int64_t q_head = hv_idx / kQGroupSize;
  const int64_t k_head = hv_idx / kKGroupSize;

  const int64_t q_base = (batch_idx * kNumQHeads + q_head) * kHeadSize;
  const int64_t k_base = (batch_idx * kNumKHeads + k_head) * kHeadSize;
  const int64_t hv_base = (batch_idx * kNumVHeads + hv_idx);
  const float beta = Sigmoid(__bfloat162float(b[hv_base]));

  const int64_t v_offset = hv_base * kHeadSize + v_idx;
  const int64_t state_row_base = v_offset * kHeadSize;
  const float v_scalar = v[v_offset];

  const float4 state_vec =
      reinterpret_cast<const float4 *>(state + state_row_base)[lane];

  const int kk_base = lane * kElemsPerLane;
  float4 q_vec;
  q_vec.x = __bfloat162float(q[q_base + kk_base + 0]);
  q_vec.y = __bfloat162float(q[q_base + kk_base + 1]);
  q_vec.z = __bfloat162float(q[q_base + kk_base + 2]);
  q_vec.w = __bfloat162float(q[q_base + kk_base + 3]);

  float4 k_vec;
  k_vec.x = __bfloat162float(k[k_base + kk_base + 0]);
  k_vec.y = __bfloat162float(k[k_base + kk_base + 1]);
  k_vec.z = __bfloat162float(k[k_base + kk_base + 2]);
  k_vec.w = __bfloat162float(k[k_base + kk_base + 3]);

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

  float4 updated_vec;
  updated_vec.x = fmaf(k_vec.x, delta, old_state_vec.x);
  updated_vec.y = fmaf(k_vec.y, delta, old_state_vec.y);
  updated_vec.z = fmaf(k_vec.z, delta, old_state_vec.z);
  updated_vec.w = fmaf(k_vec.w, delta, old_state_vec.w);

  // Streaming store: bypass L2 cache for write-once state data
  __stcs(reinterpret_cast<float4 *>(new_state + state_row_base +
                                     lane * kElemsPerLane),
         updated_vec);

  float out_partial = q_vec.x * updated_vec.x;
  out_partial = fmaf(q_vec.y, updated_vec.y, out_partial);
  out_partial = fmaf(q_vec.z, updated_vec.z, out_partial);
  out_partial = fmaf(q_vec.w, updated_vec.w, out_partial);

  const float out_acc = WarpAllReduceSum(out_partial);
  if (lane == 0) {
    output[v_offset] = __float2bfloat16_rn(scale * out_acc);
  }
}

__host__ __forceinline__ float ResolveScale(double scale) {
  float scale_f = static_cast<float>(scale);
  if (scale_f == 0.0f) {
    scale_f = 1.0f / sqrtf(static_cast<float>(kHeadSize));
  }
  return scale_f;
}

void LaunchGdnDecodeKernel6(const __nv_bfloat16 *__restrict__ q_ptr,
                            const __nv_bfloat16 *__restrict__ k_ptr,
                            const __nv_bfloat16 *__restrict__ v_ptr,
                            const float *__restrict__ state_ptr,
                            const float *__restrict__ A_log_ptr,
                            const __nv_bfloat16 *__restrict__ a_ptr,
                            const float *__restrict__ dt_bias_ptr,
                            const __nv_bfloat16 *__restrict__ b_ptr,
                            float scale_f,
                            __nv_bfloat16 *__restrict__ output_ptr,
                            float *__restrict__ new_state_ptr, int64_t B,
                            cudaStream_t stream) {
  TVM_FFI_CHECK(B > 0, ValueError) << "batch size must be positive";

  const dim3 grid(B * kNumVHeads * kNumVTiles, 1, 1);
  GdnDecodeKernel6<<<grid, kNumThreads, 0, stream>>>(
      q_ptr, k_ptr, v_ptr, state_ptr, A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
      scale_f, output_ptr, new_state_ptr);
}

void RunGdnDecodeKernel6(TensorView q, TensorView k, TensorView v,
                         TensorView state, TensorView A_log, TensorView a,
                         TensorView dt_bias, TensorView b, double scale,
                         TensorView output, TensorView new_state) {
  gdn_decode::ValidateShapesAndTypes(q, k, v, state, A_log, a, dt_bias, b,
                                     output, new_state);

  const int64_t B = q.size(0);
  const float scale_f = ResolveScale(scale);

  ffi::CUDADeviceGuard guard(q.device().device_id);
  const cudaStream_t stream = get_cuda_stream(q.device());

  const float *state_ptr = static_cast<const float *>(state.data_ptr());

  const __nv_bfloat16 *q_ptr = static_cast<const __nv_bfloat16 *>(q.data_ptr());
  const __nv_bfloat16 *k_ptr = static_cast<const __nv_bfloat16 *>(k.data_ptr());
  const __nv_bfloat16 *v_ptr = static_cast<const __nv_bfloat16 *>(v.data_ptr());
  const float *A_log_ptr = static_cast<const float *>(A_log.data_ptr());
  const __nv_bfloat16 *a_ptr = static_cast<const __nv_bfloat16 *>(a.data_ptr());
  const float *dt_bias_ptr = static_cast<const float *>(dt_bias.data_ptr());
  const __nv_bfloat16 *b_ptr = static_cast<const __nv_bfloat16 *>(b.data_ptr());
  __nv_bfloat16 *output_ptr = static_cast<__nv_bfloat16 *>(output.data_ptr());
  float *new_state_ptr = static_cast<float *>(new_state.data_ptr());

  LaunchGdnDecodeKernel6(q_ptr, k_ptr, v_ptr, state_ptr, A_log_ptr, a_ptr,
                         dt_bias_ptr, b_ptr, scale_f, output_ptr, new_state_ptr,
                         B, stream);

  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    TVM_FFI_THROW(RuntimeError)
        << "GdnDecodeKernel6 launch failed: " << cudaGetErrorString(err);
  }
}

} // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(gdn_decode_v6, RunGdnDecodeKernel6);
