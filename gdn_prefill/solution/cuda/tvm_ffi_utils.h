/*
 * Utility helpers for TVM FFI CUDA kernels.
 */
#pragma once

#include <cuda_runtime.h>

#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/cuda/device_guard.h>
#include <tvm/ffi/tvm_ffi.h>

namespace ffi = tvm::ffi;
using ffi::Optional;
using ffi::Tensor;
using ffi::TensorView;

#define CHECK_CUDA(x)                                                          \
  TVM_FFI_CHECK((x).device().device_type == kDLCUDA, ValueError)               \
      << #x " must be a CUDA tensor"

#define CHECK_CONTIGUOUS(x)                                                    \
  TVM_FFI_CHECK((x).IsContiguous(), ValueError) << #x " must be contiguous"

#define CHECK_INPUT(x)                                                         \
  do {                                                                         \
    CHECK_CUDA(x);                                                             \
    CHECK_CONTIGUOUS(x);                                                       \
  } while (0)

#define CHECK_DIM(d, x)                                                        \
  TVM_FFI_CHECK((x).ndim() == (d), ValueError)                                 \
      << #x " must be a " #d "D " << "tensor"

#define CHECK_DEVICE(a, b)                                                     \
  do {                                                                         \
    TVM_FFI_CHECK((a).device().device_type == (b).device().device_type,        \
                  ValueError)                                                  \
        << #a " and " #b " must be on the same device type";                   \
    TVM_FFI_CHECK((a).device().device_id == (b).device().device_id,            \
                  ValueError)                                                  \
        << #a " and " #b " must be on the same device";                        \
  } while (0)

constexpr DLDataType dl_bfloat16 = DLDataType{kDLBfloat, 16, 1};
constexpr DLDataType dl_float32 = DLDataType{kDLFloat, 32, 1};

inline cudaStream_t get_cuda_stream(DLDevice device) {
  return static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(device.device_type, device.device_id));
}
