/*
 * Shared helpers for GDN decode CUDA kernels.
 */
#pragma once

#include "tvm_ffi_utils.h"

namespace gdn_decode {

constexpr int kHeadSize = 128;

inline void
ValidateShapesAndTypes(const TensorView &q, const TensorView &k,
                       const TensorView &v, const TensorView &state,
                       const TensorView &A_log, const TensorView &a,
                       const TensorView &dt_bias, const TensorView &b,
                       const TensorView &output, const TensorView &new_state) {
  CHECK_INPUT(q);
  CHECK_INPUT(k);
  CHECK_INPUT(v);
  CHECK_INPUT(A_log);
  CHECK_INPUT(a);
  CHECK_INPUT(dt_bias);
  CHECK_INPUT(b);
  CHECK_INPUT(output);
  CHECK_INPUT(new_state);

  CHECK_DIM(4, q);
  CHECK_DIM(4, k);
  CHECK_DIM(4, v);
  CHECK_DIM(1, A_log);
  CHECK_DIM(3, a);
  CHECK_DIM(1, dt_bias);
  CHECK_DIM(3, b);
  CHECK_DIM(4, output);
  CHECK_DIM(4, new_state);

  CHECK_INPUT(state);
  CHECK_DIM(4, state);
  CHECK_DEVICE(q, state);

  CHECK_DEVICE(q, k);
  CHECK_DEVICE(q, v);
  CHECK_DEVICE(q, A_log);
  CHECK_DEVICE(q, a);
  CHECK_DEVICE(q, dt_bias);
  CHECK_DEVICE(q, b);
  CHECK_DEVICE(q, output);
  CHECK_DEVICE(q, new_state);

  TVM_FFI_CHECK(q.dtype() == dl_bfloat16, TypeError) << "q must be bfloat16";
  TVM_FFI_CHECK(k.dtype() == dl_bfloat16, TypeError) << "k must be bfloat16";
  TVM_FFI_CHECK(v.dtype() == dl_bfloat16, TypeError) << "v must be bfloat16";
  TVM_FFI_CHECK(a.dtype() == dl_bfloat16, TypeError) << "a must be bfloat16";
  TVM_FFI_CHECK(b.dtype() == dl_bfloat16, TypeError) << "b must be bfloat16";
  TVM_FFI_CHECK(A_log.dtype() == dl_float32, TypeError)
      << "A_log must be float32";
  TVM_FFI_CHECK(dt_bias.dtype() == dl_float32, TypeError)
      << "dt_bias must be float32";
  TVM_FFI_CHECK(output.dtype() == dl_bfloat16, TypeError)
      << "output must be bfloat16";
  TVM_FFI_CHECK(new_state.dtype() == dl_float32, TypeError)
      << "new_state must be float32";
  TVM_FFI_CHECK(state.dtype() == dl_float32, TypeError)
      << "state must be float32";

  int64_t B = q.size(0);
  int64_t T = q.size(1);
  int64_t Hq = q.size(2);
  int64_t K = q.size(3);

  int64_t Bk = k.size(0);
  int64_t Tk = k.size(1);
  int64_t Hk = k.size(2);
  int64_t Kk = k.size(3);

  int64_t Bv = v.size(0);
  int64_t Tv = v.size(1);
  int64_t HV = v.size(2);
  int64_t V = v.size(3);

  TVM_FFI_CHECK(B == Bk && B == Bv, ValueError) << "batch mismatch among q/k/v";
  TVM_FFI_CHECK(T == Tk && T == Tv, ValueError)
      << "sequence length mismatch among q/k/v";
  TVM_FFI_CHECK(K == Kk, ValueError) << "head size mismatch between q and k";
  TVM_FFI_CHECK(T == 1, ValueError) << "decode requires T=1";
  TVM_FFI_CHECK(K == kHeadSize, ValueError) << "head size K must be 128";
  TVM_FFI_CHECK(V == kHeadSize, ValueError) << "value size V must be 128";
  TVM_FFI_CHECK(Hq > 0 && Hk > 0, ValueError) << "head counts must be positive";
  TVM_FFI_CHECK(HV % Hq == 0, ValueError) << "HV must be divisible by Hq";
  TVM_FFI_CHECK(HV % Hk == 0, ValueError) << "HV must be divisible by Hk";

  TVM_FFI_CHECK(a.size(0) == B && a.size(1) == T && a.size(2) == HV, ValueError)
      << "a must have shape [B, T, HV]";
  TVM_FFI_CHECK(b.size(0) == B && b.size(1) == T && b.size(2) == HV, ValueError)
      << "b must have shape [B, T, HV]";
  TVM_FFI_CHECK(A_log.size(0) == HV, ValueError)
      << "A_log must have shape [HV]";
  TVM_FFI_CHECK(dt_bias.size(0) == HV, ValueError)
      << "dt_bias must have shape [HV]";

  TVM_FFI_CHECK(output.size(0) == B && output.size(1) == T &&
                    output.size(2) == HV && output.size(3) == V,
                ValueError)
      << "output must have shape [B, T, HV, V]";

  TVM_FFI_CHECK(new_state.size(0) == B && new_state.size(1) == HV &&
                    new_state.size(2) == V && new_state.size(3) == K,
                ValueError)
      << "new_state must have shape [B, HV, V, K]";

  TVM_FFI_CHECK(state.size(0) == B && state.size(1) == HV &&
                    state.size(2) == V && state.size(3) == K,
                ValueError)
      << "state must have shape [B, HV, V, K]";
}

} // namespace gdn_decode
