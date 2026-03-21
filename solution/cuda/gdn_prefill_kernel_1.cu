#include "tvm_ffi_utils.h"

#include <cstdint>
#include <cuda_bf16.h>
#include <math.h>

namespace {

constexpr int kHeadSize = 128;
constexpr int kBlockT = 16;
constexpr int kNumThreads = 128;
constexpr float kKeyScale = 0.08838834764831845f;
constexpr int64_t kNumQHeads = 4;
constexpr int64_t kNumKHeads = 4;
constexpr int64_t kNumVHeads = 8;
constexpr int64_t kQGroupSize = kNumVHeads / kNumQHeads;
constexpr int64_t kKGroupSize = kNumVHeads / kNumKHeads;
constexpr DLDataType dl_int64 = DLDataType{kDLInt, 64, 1};

static_assert(kHeadSize == kNumThreads,
              "prefill kernel expects one thread per output row");

__device__ __forceinline__ float SoftplusStable(float x) {
  const float abs_x = fabsf(x);
  return log1pf(expf(-abs_x)) + fmaxf(x, 0.0f);
}

__device__ __forceinline__ float Sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ double DotRowF64(const float *a, const float *b) {
  double acc = 0.0;
#pragma unroll 8
  for (int d = 0; d < kHeadSize; ++d) {
    acc += static_cast<double>(a[d]) * static_cast<double>(b[d]);
  }
  return acc;
}

inline void ValidatePrefillShapesAndTypes(const TensorView &q,
                                          const TensorView &k,
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
  const int64_t q_heads = q.size(1);
  const int64_t q_dim = q.size(2);

  TVM_FFI_CHECK(k.size(0) == total_seq_len, ValueError)
      << "k total_seq_len must match q";
  TVM_FFI_CHECK(v.size(0) == total_seq_len, ValueError)
      << "v total_seq_len must match q";
  TVM_FFI_CHECK(k.size(1) == kNumKHeads, ValueError) << "k must have 4 heads";
  TVM_FFI_CHECK(v.size(1) == kNumVHeads, ValueError) << "v must have 8 heads";
  TVM_FFI_CHECK(q_heads == kNumQHeads, ValueError) << "q must have 4 heads";
  TVM_FFI_CHECK(k.size(2) == kHeadSize, ValueError)
      << "k head size must be 128";
  TVM_FFI_CHECK(v.size(2) == kHeadSize, ValueError)
      << "v head size must be 128";
  TVM_FFI_CHECK(q_dim == kHeadSize, ValueError)
      << "q head size must be 128";

  TVM_FFI_CHECK(a.size(0) == total_seq_len && a.size(1) == kNumVHeads,
                ValueError)
      << "a must have shape [T, HV]";
  TVM_FFI_CHECK(b.size(0) == total_seq_len && b.size(1) == kNumVHeads,
                ValueError)
      << "b must have shape [T, HV]";
  TVM_FFI_CHECK(A_log.size(0) == kNumVHeads, ValueError)
      << "A_log must have shape [HV]";
  TVM_FFI_CHECK(dt_bias.size(0) == kNumVHeads, ValueError)
      << "dt_bias must have shape [HV]";

  const int64_t num_seqs = cu_seqlens.size(0) - 1;
  TVM_FFI_CHECK(num_seqs > 0, ValueError) << "cu_seqlens must have length >= 2";
  TVM_FFI_CHECK(state.size(0) == num_seqs && state.size(1) == kNumVHeads &&
                    state.size(2) == kHeadSize && state.size(3) == kHeadSize,
                ValueError)
      << "state must have shape [N, HV, 128, 128]";
  TVM_FFI_CHECK(new_state.size(0) == num_seqs && new_state.size(1) == kNumVHeads &&
                    new_state.size(2) == kHeadSize &&
                    new_state.size(3) == kHeadSize,
                ValueError)
      << "new_state must have shape [N, HV, 128, 128]";
  TVM_FFI_CHECK(output.size(0) == total_seq_len && output.size(1) == kNumVHeads &&
                    output.size(2) == kHeadSize,
                ValueError)
      << "output must have shape [T, HV, 128]";
}

__global__ void GdnPrefillKernel1(
    const __nv_bfloat16 *__restrict__ q, const __nv_bfloat16 *__restrict__ k,
    const __nv_bfloat16 *__restrict__ v, const float *__restrict__ state,
    const float *__restrict__ A_log, const __nv_bfloat16 *__restrict__ a,
    const float *__restrict__ dt_bias, const __nv_bfloat16 *__restrict__ b,
    const int64_t *__restrict__ cu_seqlens, float scale,
    __nv_bfloat16 *__restrict__ output, float *__restrict__ new_state) {
  const int64_t block_linear = static_cast<int64_t>(blockIdx.x);
  const int64_t seq_idx = block_linear / kNumVHeads;
  const int64_t hv_idx = block_linear % kNumVHeads;
  const int row = threadIdx.x;

  const int64_t q_head = hv_idx / kQGroupSize;
  const int64_t k_head = hv_idx / kKGroupSize;
  const int64_t seq_start = cu_seqlens[seq_idx];
  const int64_t seq_end = cu_seqlens[seq_idx + 1];
  if (seq_end <= seq_start) {
    return;
  }

  __shared__ float s_q[kBlockT][kHeadSize];
  __shared__ float s_k[kBlockT][kHeadSize];
  __shared__ float s_v[kBlockT][kHeadSize];
  __shared__ float s_logg[kBlockT];
  __shared__ float s_beta[kBlockT];
  __shared__ float s_log_cu[kBlockT];
  __shared__ float s_cu_gate[kBlockT];
  __shared__ float s_qk[kBlockT][kBlockT];
  __shared__ float s_kkt[kBlockT][kBlockT];
  __shared__ float s_gamma[kBlockT][kBlockT];
  __shared__ float s_ikk[kBlockT][kBlockT];
  __shared__ float s_inv_ikk[kBlockT][kBlockT];
  __shared__ float s_rhs[kBlockT][kHeadSize];
  __shared__ float s_new_v[kBlockT][kHeadSize];

  const int64_t state_row_base =
      ((((seq_idx * kNumVHeads) + hv_idx) * kHeadSize) + row) * kHeadSize;
  float row_buf[kHeadSize];
  for (int d = 0; d < kHeadSize; ++d) {
    row_buf[d] = state[state_row_base + d];
  }

  const float exp_A = expf(A_log[hv_idx]);

  for (int64_t t0 = seq_start; t0 < seq_end; t0 += kBlockT) {
    const int tile_n =
        static_cast<int>(((seq_end - t0) < kBlockT) ? (seq_end - t0) : kBlockT);

    for (int idx = threadIdx.x; idx < tile_n * kHeadSize; idx += kNumThreads) {
      const int t = idx / kHeadSize;
      const int d = idx % kHeadSize;
      s_q[t][d] = __bfloat162float(
          q[((t0 + t) * kNumQHeads + q_head) * kHeadSize + d]);
      s_k[t][d] = kKeyScale * __bfloat162float(
          k[((t0 + t) * kNumKHeads + k_head) * kHeadSize + d]);
      s_v[t][d] = __bfloat162float(
          v[((t0 + t) * kNumVHeads + hv_idx) * kHeadSize + d]);
    }

    if (threadIdx.x < tile_n) {
      const int t = threadIdx.x;
      const float x =
          __bfloat162float(a[(t0 + t) * kNumVHeads + hv_idx]) + dt_bias[hv_idx];
      s_logg[t] = -exp_A * SoftplusStable(x);
      s_beta[t] =
          Sigmoid(__bfloat162float(b[(t0 + t) * kNumVHeads + hv_idx]));
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      double acc = 0.0;
      for (int i = 0; i < tile_n; ++i) {
        acc += static_cast<double>(s_logg[i]);
        const double cu = exp(acc);
        s_log_cu[i] = static_cast<float>(acc);
        s_cu_gate[i] = static_cast<float>(cu);
      }
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < tile_n * tile_n; idx += kNumThreads) {
      const int i = idx / tile_n;
      const int j = idx % tile_n;
      double qk_acc = 0.0;
      double kk_acc = 0.0;
#pragma unroll 8
      for (int d = 0; d < kHeadSize; ++d) {
        qk_acc += static_cast<double>(s_q[i][d]) * static_cast<double>(s_k[j][d]);
        kk_acc += static_cast<double>(s_k[i][d]) * static_cast<double>(s_k[j][d]);
      }
      s_qk[i][j] = static_cast<float>(qk_acc);
      s_kkt[i][j] = static_cast<float>(kk_acc);
      s_gamma[i][j] = expf(s_log_cu[i] - s_log_cu[j]);
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < tile_n * tile_n; idx += kNumThreads) {
      const int i = idx / tile_n;
      const int j = idx % tile_n;
      float val = 0.0f;
      if (j < i) {
        val = s_beta[i] * s_gamma[i][j] * s_kkt[i][j];
      }
      if (i == j) {
        val += 1.0f;
      }
      s_ikk[i][j] = val;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      // Match the Blackwell kernel's blockwise decomposition:
      // build inv(I + strict_lower(beta * Gamma * K K^T)) first, then apply it to
      // beta * (V - gamma * K S_old).
      for (int row_i = 0; row_i < tile_n; ++row_i) {
        for (int col_j = 0; col_j < tile_n; ++col_j) {
          if (col_j > row_i) {
            s_inv_ikk[row_i][col_j] = 0.0f;
            continue;
          }
          double rhs = (row_i == col_j) ? 1.0 : 0.0;
          for (int kk = 0; kk < row_i; ++kk) {
            rhs -= static_cast<double>(s_ikk[row_i][kk]) *
                   static_cast<double>(s_inv_ikk[kk][col_j]);
          }
          rhs /= static_cast<double>(s_ikk[row_i][row_i]);
          s_inv_ikk[row_i][col_j] = static_cast<float>(rhs);
        }
      }
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < tile_n * kHeadSize; idx += kNumThreads) {
      const int i = idx / kHeadSize;
      const int d = idx % kHeadSize;
      s_rhs[i][d] = static_cast<float>(DotRowF64(s_k[i], row_buf));
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < tile_n * kHeadSize; idx += kNumThreads) {
      const int i = idx / kHeadSize;
      const int d = idx % kHeadSize;
      const double rhs =
          static_cast<double>(s_beta[i]) *
          (static_cast<double>(s_v[i][d]) -
           static_cast<double>(s_cu_gate[i]) * static_cast<double>(s_rhs[i][d]));
      s_rhs[i][d] = static_cast<float>(rhs);
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < tile_n * kHeadSize; idx += kNumThreads) {
      const int i = idx / kHeadSize;
      const int d = idx % kHeadSize;
      double new_v = 0.0;
      for (int j = 0; j < tile_n; ++j) {
        new_v += static_cast<double>(s_inv_ikk[i][j]) *
                 static_cast<double>(s_rhs[j][d]);
      }
      s_new_v[i][d] = static_cast<float>(new_v);
    }
    __syncthreads();

    for (int i = 0; i < tile_n; ++i) {
      double o_intra = 0.0;
      for (int j = 0; j <= i; ++j) {
        o_intra += static_cast<double>(s_qk[i][j]) *
                   static_cast<double>(s_gamma[i][j]) *
                   static_cast<double>(s_new_v[j][row]);
      }
      const double o_inter =
          static_cast<double>(s_cu_gate[i]) * DotRowF64(s_q[i], row_buf);
      const double out_val = o_inter + o_intra;
      output[((t0 + i) * kNumVHeads + hv_idx) * kHeadSize + row] =
          __float2bfloat16_rn(static_cast<float>(static_cast<double>(scale) *
                                                 out_val));
    }

    for (int d = 0; d < kHeadSize; ++d) {
      double next_val =
          static_cast<double>(s_cu_gate[tile_n - 1]) * static_cast<double>(row_buf[d]);
      for (int j = 0; j < tile_n; ++j) {
        next_val += static_cast<double>(s_gamma[tile_n - 1][j]) *
                    static_cast<double>(s_new_v[j][row]) *
                    static_cast<double>(s_k[j][d]);
      }
      row_buf[d] = static_cast<float>(next_val);
    }
    __syncthreads();
  }

  for (int d = 0; d < kHeadSize; ++d) {
    new_state[state_row_base + d] = row_buf[d];
  }
}

__host__ __forceinline__ float ResolveScale(double scale) {
  float scale_f = static_cast<float>(scale);
  if (scale_f == 0.0f) {
    scale_f = 1.0f / sqrtf(static_cast<float>(kHeadSize));
  }
  return scale_f;
}

void LaunchGdnPrefillKernel1(
    const __nv_bfloat16 *__restrict__ q_ptr,
    const __nv_bfloat16 *__restrict__ k_ptr,
    const __nv_bfloat16 *__restrict__ v_ptr,
    const float *__restrict__ state_ptr,
    const float *__restrict__ A_log_ptr,
    const __nv_bfloat16 *__restrict__ a_ptr,
    const float *__restrict__ dt_bias_ptr,
    const __nv_bfloat16 *__restrict__ b_ptr,
    const int64_t *__restrict__ cu_seqlens_ptr, float scale_f,
    __nv_bfloat16 *__restrict__ output_ptr, float *__restrict__ new_state_ptr,
    int64_t num_seqs, int64_t total_seq_len, cudaStream_t stream) {
  TVM_FFI_CHECK(num_seqs > 0, ValueError) << "num_seqs must be positive";
  TVM_FFI_CHECK(total_seq_len >= 0, ValueError)
      << "total_seq_len must be non-negative";

  const size_t output_bytes = static_cast<size_t>(total_seq_len) * kNumVHeads *
                              kHeadSize * sizeof(__nv_bfloat16);
  const size_t state_bytes = static_cast<size_t>(num_seqs) * kNumVHeads *
                             kHeadSize * kHeadSize * sizeof(float);
  const cudaError_t output_clear_err =
      cudaMemsetAsync(output_ptr, 0, output_bytes, stream);
  TVM_FFI_CHECK(output_clear_err == cudaSuccess, RuntimeError)
      << "Failed to clear output: " << cudaGetErrorString(output_clear_err);
  const cudaError_t state_clear_err =
      cudaMemsetAsync(new_state_ptr, 0, state_bytes, stream);
  TVM_FFI_CHECK(state_clear_err == cudaSuccess, RuntimeError)
      << "Failed to clear new_state: " << cudaGetErrorString(state_clear_err);

  const dim3 grid(num_seqs * kNumVHeads, 1, 1);
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

  const int64_t total_seq_len = q.size(0);
  const int64_t num_seqs = cu_seqlens.size(0) - 1;
  const float scale_f = ResolveScale(scale);

  ffi::CUDADeviceGuard guard(q.device().device_id);
  const cudaStream_t stream = get_cuda_stream(q.device());

  const __nv_bfloat16 *q_ptr = static_cast<const __nv_bfloat16 *>(q.data_ptr());
  const __nv_bfloat16 *k_ptr = static_cast<const __nv_bfloat16 *>(k.data_ptr());
  const __nv_bfloat16 *v_ptr = static_cast<const __nv_bfloat16 *>(v.data_ptr());
  const float *state_ptr = static_cast<const float *>(state.data_ptr());
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
                          output_ptr, new_state_ptr, num_seqs, total_seq_len,
                          stream);

  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    TVM_FFI_THROW(RuntimeError)
        << "GdnPrefillKernel1 launch failed: " << cudaGetErrorString(err);
  }
}

} // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(gdn_prefill_1, RunGdnPrefillKernel1);
