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

constexpr int64_t kQGroupSize = kNumVHeads / kNumQHeads;
constexpr int64_t kKGroupSize = kNumVHeads / kNumKHeads;
constexpr unsigned kFullWarpMask = 0xffffffffu;

static_assert(kHeadSize == 128, "kernel_2 expects head size 128");
static_assert(kNumThreads == 128, "kernel_2 expects 128 threads per block");
static_assert(kHeadSize % kWarpSize == 0,
              "head size must be divisible by warp size");
static_assert(kRowsPerBlock == 4, "kernel_2 expects four rows per block");
static_assert(kHeadSize % kRowsPerBlock == 0,
              "head size must be divisible by rows per block");

struct GdnScalars {
  float g;
  float beta;
};

__device__ __forceinline__ float SoftplusStable(float x) {
  const float abs_x = fabsf(x);
  return log1pf(expf(-abs_x)) + fmaxf(x, 0.0f);
}

__device__ __forceinline__ float Sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ GdnScalars ComputeGdnScalars(float A_log_val,
                                                        __nv_bfloat16 a_val,
                                                        float dt_bias_val,
                                                        __nv_bfloat16 b_val) {
  const float x = __bfloat162float(a_val) + dt_bias_val;
  const float softplus_x = SoftplusStable(x);

  GdnScalars out;
  out.g = expf(-expf(A_log_val) * softplus_x);
  out.beta = Sigmoid(__bfloat162float(b_val));
  return out;
}

__device__ __forceinline__ float WarpAllReduceSum(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(kFullWarpMask, value, offset);
  }
  return __shfl_sync(kFullWarpMask, value, 0);
}

__device__ __forceinline__ void
StoreBf16Predicated(__nv_bfloat16 *ptr, __nv_bfloat16 value, int predicate) {
  union {
    __nv_bfloat16 bf16;
    unsigned short u16;
  } bits;
  bits.bf16 = value;

  asm volatile("{\n\t"
               ".reg .pred p;\n\t"
               "setp.ne.s32 p, %2, 0;\n\t"
               "@p st.global.u16 [%0], %1;\n\t"
               "}\n"
               :
               : "l"(ptr), "h"(bits.u16), "r"(predicate)
               : "memory");
}

__global__ void GdnDecodeKernel2(const __nv_bfloat16 *q, const __nv_bfloat16 *k,
                                 const __nv_bfloat16 *v, const float *state,
                                 const float *A_log, const __nv_bfloat16 *a,
                                 const float *dt_bias, const __nv_bfloat16 *b,
                                 float scale, __nv_bfloat16 *output,
                                 float *new_state) {
  const int64_t block_linear = static_cast<int64_t>(blockIdx.x);
  const int64_t tile_idx = block_linear % kNumVTiles;
  const int64_t bh = block_linear / kNumVTiles;

  const int64_t batch_idx = bh / kNumVHeads;
  const int64_t hv_idx = bh % kNumVHeads;

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & (kWarpSize - 1);

  const int64_t v_idx = tile_idx * kRowsPerBlock + warp_id;

  const int64_t q_head = hv_idx / kQGroupSize;
  const int64_t k_head = hv_idx / kKGroupSize;

  const int64_t q_base = (batch_idx * kNumQHeads + q_head) * kHeadSize;
  const int64_t k_base = (batch_idx * kNumKHeads + k_head) * kHeadSize;
  const int64_t hv_base = (batch_idx * kNumVHeads + hv_idx);

  __shared__ float s_q[kHeadSize];
  __shared__ float s_k[kHeadSize];

  s_q[tid] = __bfloat162float(q[q_base + tid]);
  s_k[tid] = __bfloat162float(k[k_base + tid]);

  __syncthreads();

  const GdnScalars scalars =
      ComputeGdnScalars(A_log[hv_idx], a[hv_base], dt_bias[hv_idx], b[hv_base]);
  const float g = scalars.g;
  const float beta = scalars.beta;

  const int64_t v_offset = hv_base * kHeadSize + v_idx;
  const int64_t state_row_base = v_offset * kHeadSize;

  const float v_scalar = __bfloat162float(v[v_offset]);

  float old_v_partial = 0.0f;
#pragma unroll
  for (int kk = lane; kk < kHeadSize; kk += kWarpSize) {
    const float old_state = g * state[state_row_base + kk];
    old_v_partial += s_k[kk] * old_state;
  }

  const float old_v = WarpAllReduceSum(old_v_partial);
  const float new_v = beta * v_scalar + (1.0f - beta) * old_v;
  const float delta = new_v - old_v;

  float out_partial = 0.0f;
#pragma unroll
  for (int kk = lane; kk < kHeadSize; kk += kWarpSize) {
    const float old_state = g * state[state_row_base + kk];
    const float updated = old_state + s_k[kk] * delta;
    new_state[state_row_base + kk] = updated;
    out_partial += s_q[kk] * updated;
  }

  const float out_acc = WarpAllReduceSum(out_partial);
  const int lane_is_leader = static_cast<int>(lane == 0);
  StoreBf16Predicated(output + v_offset, __float2bfloat16_rn(scale * out_acc),
                      lane_is_leader);
}

__host__ __forceinline__ float ResolveScale(double scale) {
  float scale_f = static_cast<float>(scale);
  if (scale_f == 0.0f) {
    scale_f = 1.0f / sqrtf(static_cast<float>(kHeadSize));
  }
  return scale_f;
}

void LaunchGdnDecodeKernel2(const __nv_bfloat16 *q_ptr,
                            const __nv_bfloat16 *k_ptr,
                            const __nv_bfloat16 *v_ptr, const float *state_ptr,
                            const float *A_log_ptr, const __nv_bfloat16 *a_ptr,
                            const float *dt_bias_ptr,
                            const __nv_bfloat16 *b_ptr, float scale_f,
                            __nv_bfloat16 *output_ptr, float *new_state_ptr,
                            int64_t B, cudaStream_t stream) {
  TVM_FFI_CHECK(B > 0, ValueError) << "batch size must be positive";

  const dim3 grid(B * kNumVHeads * kNumVTiles, 1, 1);
  GdnDecodeKernel2<<<grid, kNumThreads, 0, stream>>>(
      q_ptr, k_ptr, v_ptr, state_ptr, A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
      scale_f, output_ptr, new_state_ptr);
}

void RunGdnDecodeKernel2(TensorView q, TensorView k, TensorView v,
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

  LaunchGdnDecodeKernel2(q_ptr, k_ptr, v_ptr, state_ptr, A_log_ptr, a_ptr,
                         dt_bias_ptr, b_ptr, scale_f, output_ptr, new_state_ptr,
                         B, stream);

  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    TVM_FFI_THROW(RuntimeError)
        << "GdnDecodeKernel2 launch failed: " << cudaGetErrorString(err);
  }
}

} // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(gdn_decode_v2, RunGdnDecodeKernel2);
