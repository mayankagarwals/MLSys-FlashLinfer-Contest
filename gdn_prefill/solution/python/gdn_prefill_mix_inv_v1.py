import ctypes

ctypes.CDLL("libcudart.so", mode=ctypes.RTLD_GLOBAL)

import torch
from torch import Tensor

from .cuda_recurrent_v1 import run as cuda_recurrent_v1
from .chunk_v11 import run as chunk_v11
from .cuda_parallel_v4 import run as cuda_v4


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
    N = cu_seqlens.shape[0] - 1

    if T >= 525:
        return chunk_v11(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)

    if T >= 64 or (N == 1 and T >= 46):
        return cuda_v4(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)

    o = torch.empty_like(v)
    new_state = torch.empty_like(state)
    cuda_recurrent_v1(
        q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, o, new_state
    )
    return o, new_state
