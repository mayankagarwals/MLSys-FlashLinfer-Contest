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
    name="gdn_prefill_cuda_v4",
    cuda_files=[str(CURRENT_DIR / "cuda_parallel_v4.cu")],
    extra_cflags=["-O3"],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "-lineinfo",
    ],
    extra_ldflags=["-lcuda"],
)

mod = tvm_ffi.load_module(lib_path)
_kernel = mod.gdn_prefill_tcgen05


def run(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    state: Tensor,
    A_log: Tensor,
    a: Tensor,
    dt_bias: Tensor,
    b: Tensor,
    cu_seqlens: Tensor,
    scale: float,
):
    output = torch.empty_like(v)
    new_state = torch.empty_like(state, dtype=torch.float32)
    _kernel(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, output, new_state)
    return output, new_state
