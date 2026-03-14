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

constexpr int kRowsPerBlock = 4;
constexpr int kNumThreads = kRowsPerBlock;
constexpr int kNumVTiles = kHeadSize / kRowsPerBlock;

constexpr int64_t kQGroupSize = kNumVHeads / kNumQHeads;
constexpr int64_t kKGroupSize = kNumVHeads / kNumKHeads;
constexpr DLDataType dl_int64 = DLDataType{kDLInt, 64, 1};

static_assert(kHeadSize == 128, "prefill kernel expects head size 128");
static_assert(kRowsPerBlock == 4, "prefill kernel expects four rows per block");
static_assert(kNumThreads == 4, "prefill kernel uses one thread per row");

inline void ValidatePrefillShapesAndTypes(const TensorView &q, const TensorView &k,
                                          const TensorView &v,
                                          const TensorView &state,
                                          const TensorView &A_log,
                                          const TensorView &a,
                                          const TensorView &dt_bias,
                                          const TensorView &b,
                                          const TensorView &cu_seqlens,
                                          const TensorView &output,
                                          const TensorView &new_state) {
  CHECK_INPUT(q);
  CHECK_INPUT(k);
  CHECK_INPUT(v);
  CHECK_INPUT(state);
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
  CHECK_DIM(4, state);
  CHECK_DIM(1, A_log);
  CHECK_DIM(2, a);
  CHECK_DIM(1, dt_bias);
  CHECK_DIM(2, b);
  CHECK_DIM(1, cu_seqlens);
  CHECK_DIM(3, output);
  CHECK_DIM(4, new_state);

  CHECK_DEVICE(q, k);
  CHECK_DEVICE(q, v);
  CHECK_DEVICE(q, state);
  CHECK_DEVICE(q, A_log);
  CHECK_DEVICE(q, a);
  CHECK_DEVICE(q, dt_bias);
  CHECK_DEVICE(q, b);
  CHECK_DEVICE(q, cu_seqlens);
  CHECK_DEVICE(q, output);
  CHECK_DEVICE(q, new_state);

  TVM_FFI_CHECK(q.dtype() == dl_bfloat16, TypeError) << "q must be bfloat16";
  TVM_FFI_CHECK(k.dtype() == dl_bfloat16, TypeError) << "k must be bfloat16";
  TVM_FFI_CHECK(v.dtype() == dl_bfloat16, TypeError) << "v must be bfloat16";
  TVM_FFI_CHECK(state.dtype() == dl_float32, TypeError)
      << "state must be float32";
  TVM_FFI_CHECK(A_log.dtype() == dl_float32, TypeError)
      << "A_log must be float32";
  TVM_FFI_CHECK(a.dtype() == dl_bfloat16, TypeError) << "a must be bfloat16";
  TVM_FFI_CHECK(dt_bias.dtype() == dl_float32, TypeError)
      << "dt_bias must be float32";
  TVM_FFI_CHECK(b.dtype() == dl_bfloat16, TypeError) << "b must be bfloat16";
  TVM_FFI_CHECK(cu_seqlens.dtype() == dl_int64, TypeError)
      << "cu_seqlens must be int64";
  TVM_FFI_CHECK(output.dtype() == dl_bfloat16, TypeError)
      << "output must be bfloat16";
  TVM_FFI_CHECK(new_state.dtype() == dl_float32, TypeError)
      << "new_state must be float32";

  const int64_t total_seq_len = q.size(0);

  TVM_FFI_CHECK(q.size(1) == kNumQHeads && q.size(2) == kHeadSize, ValueError)
      << "q must have shape [T, 4, 128]";
  TVM_FFI_CHECK(k.size(0) == total_seq_len && k.size(1) == kNumKHeads &&
                    k.size(2) == kHeadSize,
                ValueError)
      << "k must have shape [T, 4, 128]";
  TVM_FFI_CHECK(v.size(0) == total_seq_len && v.size(1) == kNumVHeads &&
                    v.size(2) == kHeadSize,
                ValueError)
      << "v must have shape [T, 8, 128]";

  TVM_FFI_CHECK(A_log.size(0) == kNumVHeads, ValueError)
      << "A_log must have shape [8]";
  TVM_FFI_CHECK(dt_bias.size(0) == kNumVHeads, ValueError)
      << "dt_bias must have shape [8]";
  TVM_FFI_CHECK(a.size(0) == total_seq_len && a.size(1) == kNumVHeads,
                ValueError)
      << "a must have shape [T, 8]";
  TVM_FFI_CHECK(b.size(0) == total_seq_len && b.size(1) == kNumVHeads,
                ValueError)
      << "b must have shape [T, 8]";

  TVM_FFI_CHECK(cu_seqlens.size(0) >= 1, ValueError)
      << "cu_seqlens must have at least one entry";
  const int64_t num_seqs = cu_seqlens.size(0) - 1;

  TVM_FFI_CHECK(state.size(0) == num_seqs && state.size(1) == kNumVHeads &&
                    state.size(2) == kHeadSize && state.size(3) == kHeadSize,
                ValueError)
      << "state must have shape [N, 8, 128, 128]";
  TVM_FFI_CHECK(new_state.size(0) == num_seqs &&
                    new_state.size(1) == kNumVHeads &&
                    new_state.size(2) == kHeadSize &&
                    new_state.size(3) == kHeadSize,
                ValueError)
      << "new_state must have shape [N, 8, 128, 128]";
  TVM_FFI_CHECK(output.size(0) == total_seq_len &&
                    output.size(1) == kNumVHeads &&
                    output.size(2) == kHeadSize,
                ValueError)
      << "output must have shape [T, 8, 128]";
}

__device__ __forceinline__ float SoftplusStable(float x) {
  const float abs_x = fabsf(x);
  return log1pf(expf(-abs_x)) + fmaxf(x, 0.0f);
}

__device__ __forceinline__ double SoftplusStable(double x) {
  const double abs_x = fabs(x);
  return log1p(exp(-abs_x)) + fmax(x, 0.0);
}

__device__ __forceinline__ double Sigmoid(double x) {
  return 1.0 / (1.0 + exp(-x));
}

__device__ __forceinline__ double ComputeGdnScalars(double negated_exp_A_log,
                                                    __nv_bfloat16 a_val,
                                                    float dt_bias_val) {
  const double x = static_cast<double>(__bfloat162float(a_val)) +
                   static_cast<double>(dt_bias_val);
  const double softplus_x = SoftplusStable(x);
  return exp(negated_exp_A_log * softplus_x);
}

__global__ void GdnPrefillKernel1(
    const __nv_bfloat16 *__restrict__ q, const __nv_bfloat16 *__restrict__ k,
    const __nv_bfloat16 *__restrict__ v, const float *__restrict__ state,
    const float *__restrict__ A_log, const __nv_bfloat16 *__restrict__ a,
    const float *__restrict__ dt_bias, const __nv_bfloat16 *__restrict__ b,
    const int64_t *__restrict__ cu_seqlens, float scale,
    __nv_bfloat16 *__restrict__ output, float *__restrict__ new_state) {
  const int64_t block_linear = static_cast<int64_t>(blockIdx.x);
  const int64_t tile_idx = block_linear % kNumVTiles;
  const int64_t sh = block_linear / kNumVTiles;

  const int64_t seq_idx = sh / kNumVHeads;
  const int64_t hv_idx = sh % kNumVHeads;
  const int row_local = threadIdx.x;
  const int64_t v_idx = tile_idx * kRowsPerBlock + row_local;

  const int64_t q_head = hv_idx / kQGroupSize;
  const int64_t k_head = hv_idx / kKGroupSize;
  const int64_t seq_start = cu_seqlens[seq_idx];
  const int64_t seq_end = cu_seqlens[seq_idx + 1];
  const double negated_exp_A_log =
      -exp(static_cast<double>(A_log[hv_idx]));

  __shared__ float s_state[kRowsPerBlock][kHeadSize];

  const int64_t state_row_base =
      ((((seq_idx * kNumVHeads) + hv_idx) * kHeadSize) + v_idx) * kHeadSize;
  for (int kk = 0; kk < kHeadSize; ++kk) {
    s_state[row_local][kk] = state[state_row_base + kk];
  }
  __syncthreads();

  for (int64_t tok = seq_start; tok < seq_end; ++tok) {
    const int64_t q_base = ((tok * kNumQHeads) + q_head) * kHeadSize;
    const int64_t k_base = ((tok * kNumKHeads) + k_head) * kHeadSize;
    const int64_t hv_base = (tok * kNumVHeads) + hv_idx;
    const int64_t out_offset = (hv_base * kHeadSize) + v_idx;

    const double g =
        ComputeGdnScalars(negated_exp_A_log, a[hv_base], dt_bias[hv_idx]);
    const double beta =
        Sigmoid(static_cast<double>(__bfloat162float(b[hv_base])));
    const double v_scalar = static_cast<double>(__bfloat162float(v[out_offset]));

    double old_v = 0.0;
    for (int kk = 0; kk < kHeadSize; ++kk) {
      const double old_state = g * static_cast<double>(s_state[row_local][kk]);
      s_state[row_local][kk] = static_cast<float>(old_state);
      old_v += static_cast<double>(__bfloat162float(k[k_base + kk])) * old_state;
    }

    const double delta = beta * (v_scalar - old_v);

    double out_acc = 0.0;
    for (int kk = 0; kk < kHeadSize; ++kk) {
      const double updated = static_cast<double>(s_state[row_local][kk]) +
                             static_cast<double>(__bfloat162float(k[k_base + kk])) *
                                 delta;
      s_state[row_local][kk] = static_cast<float>(updated);
      out_acc += static_cast<double>(__bfloat162float(q[q_base + kk])) * updated;
    }

    output[out_offset] =
        __float2bfloat16_rn(static_cast<float>(static_cast<double>(scale) * out_acc));
  }

  __syncthreads();
  for (int kk = 0; kk < kHeadSize; ++kk) {
    new_state[state_row_base + kk] = s_state[row_local][kk];
  }
}

void LaunchGdnPrefillKernel1(const __nv_bfloat16 *__restrict__ q_ptr,
                             const __nv_bfloat16 *__restrict__ k_ptr,
                             const __nv_bfloat16 *__restrict__ v_ptr,
                             const float *__restrict__ state_ptr,
                             const float *__restrict__ A_log_ptr,
                             const __nv_bfloat16 *__restrict__ a_ptr,
                             const float *__restrict__ dt_bias_ptr,
                             const __nv_bfloat16 *__restrict__ b_ptr,
                             const int64_t *__restrict__ cu_seqlens_ptr,
                             float scale_f,
                             __nv_bfloat16 *__restrict__ output_ptr,
                             float *__restrict__ new_state_ptr,
                             int64_t num_seqs, cudaStream_t stream) {
  TVM_FFI_CHECK(num_seqs > 0, ValueError) << "num_seqs must be positive";

  const dim3 grid(num_seqs * kNumVHeads * kNumVTiles, 1, 1);
  GdnPrefillKernel1<<<grid, kNumThreads, 0, stream>>>(
      q_ptr, k_ptr, v_ptr, state_ptr, A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
      cu_seqlens_ptr, scale_f, output_ptr, new_state_ptr);
}

void RunGdnPrefillKernel1(TensorView q, TensorView k, TensorView v,
                          TensorView state, TensorView A_log, TensorView a,
                          TensorView dt_bias, TensorView b,
                          TensorView cu_seqlens, double scale,
                          TensorView output, TensorView new_state) {
  ValidatePrefillShapesAndTypes(q, k, v, state, A_log, a, dt_bias, b,
                                cu_seqlens, output, new_state);

  const int64_t num_seqs = cu_seqlens.size(0) - 1;
  const float scale_f = gdn_decode::ResolveScale(scale);

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
  const int64_t *cu_seqlens_ptr =
      static_cast<const int64_t *>(cu_seqlens.data_ptr());
  __nv_bfloat16 *output_ptr = static_cast<__nv_bfloat16 *>(output.data_ptr());
  float *new_state_ptr = static_cast<float *>(new_state.data_ptr());

  LaunchGdnPrefillKernel1(q_ptr, k_ptr, v_ptr, state_ptr, A_log_ptr, a_ptr,
                          dt_bias_ptr, b_ptr, cu_seqlens_ptr, scale_f,
                          output_ptr, new_state_ptr, num_seqs, stream);

  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    TVM_FFI_THROW(RuntimeError)
        << "GdnPrefillKernel1 launch failed: " << cudaGetErrorString(err);
  }
}

} // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(gdn_prefill_1, RunGdnPrefillKernel1);
