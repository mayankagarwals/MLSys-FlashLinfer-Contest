"""Hybrid dispatcher: CUDA for small workloads, Triton for large."""
import ctypes
ctypes.CDLL("libcudart.so", mode=ctypes.RTLD_GLOBAL)

import torch
from torch import Tensor
from pathlib import Path

# Import both kernels
from cuda_parallel_v3 import run as cuda_run
from triton_v4 import run as triton_run

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
    T = q.shape[0]
    if T >= 4096:
        return triton_run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)
    else:
        return cuda_run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)
