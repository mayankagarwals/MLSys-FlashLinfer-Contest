/*
 * Shared helpers for GDN prefill CUDA kernels.
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
