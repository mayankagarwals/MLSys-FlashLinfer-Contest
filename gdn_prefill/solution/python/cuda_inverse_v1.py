import ctypes
import os
ctypes.CDLL("libcudart.so", mode=ctypes.RTLD_GLOBAL)

import torch
from pathlib import Path
from torch import Tensor

os.environ["TVM_FFI_CUDA_ARCH_LIST"] = "10.0a"

import tvm_ffi

CURRENT_DIR = Path(__file__).parent

lib_path = tvm_ffi.cpp.build(
    name="gdn_prefill_inverse_v1",
    cuda_files=[str(CURRENT_DIR / "cuda_inverse_v1.cu")],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
    extra_ldflags=["-lcuda"],
)

mod = tvm_ffi.load_module(lib_path)
_kernel = mod.inverse_v1


def run(
    k: Tensor, v: Tensor, w_out: Tensor, u_out: Tensor,
    A: Tensor, beta: Tensor, g_cu: Tensor,
    cu_seqlens: Tensor, chunk_indices: Tensor, total_chunks_ptr: Tensor,
    upper_bound_chunks: int,
):
    """Run CUDA inverse kernel. Signature: A, k, v, w, u, beta, g_cu, cu_seqlens, chunk_indices, total_chunks, upper_bound_chunks"""
    _kernel(A, k, v, w_out, u_out, beta, g_cu,
            cu_seqlens, chunk_indices, total_chunks_ptr,
            upper_bound_chunks)
