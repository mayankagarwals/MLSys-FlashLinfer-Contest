import ctypes
ctypes.CDLL("libcudart.so", mode=ctypes.RTLD_GLOBAL)

import torch
from pathlib import Path
from torch import Tensor

import tvm_ffi
import tvm_ffi.cpp.extension as _ext

# Monkey-patch to force sm_100a (needed for tcgen05 instructions on Blackwell)
_orig_get_cuda_target = _ext._get_cuda_target
_ext._get_cuda_target = lambda: "-gencode=arch=compute_100a,code=sm_100a"

CURRENT_DIR = Path(__file__).parent

lib_path = tvm_ffi.cpp.build(
    name="gdn_prefill_cuda_v3",
    cuda_files=[str(CURRENT_DIR / "cuda_parallel_v3.cu")],
    extra_cflags=["-O3"],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "-lineinfo",
    ],
    extra_ldflags=["-lcuda"],
)

# Restore original
_ext._get_cuda_target = _orig_get_cuda_target

mod = tvm_ffi.load_module(lib_path)
_kernel = mod.gdn_prefill_tcgen05

# Try chunk_v5 (CUDA kkt + Triton), fallback to triton_v4
import sys
sys.path.insert(0, str(CURRENT_DIR))
_fast_chunk_run = None
try:
    from chunk_v5 import run as _fast_chunk_run
except Exception:
    try:
        from triton_v4 import run as _fast_chunk_run
    except Exception:
        pass


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
    T, H, V_dim = v.shape
    N = state.shape[0]

    # Use chunk_v5/triton for large workloads where it's faster
    if _fast_chunk_run is not None and T >= 4096:
        return _fast_chunk_run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)

    output = torch.empty(T, H, V_dim, device=v.device, dtype=v.dtype)
    new_state = torch.empty(N, H, V_dim, V_dim, device=state.device, dtype=torch.float32)
    _kernel(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, output, new_state)
    return output, new_state
