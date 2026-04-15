import ctypes

# get weird errors without this
ctypes.CDLL("libcudart.so", mode=ctypes.RTLD_GLOBAL)

import torch
from torch import Tensor

from .cuda_recurrent_v1 import run as cuda_recurrent_v1
from .chunk_v8c import run as chunk_v8c
from .chunk_v8b import run as chunk_v8b
from .cuda_parallel_v4 import run as cuda_v4


def run(
    q: Tensor,  # (total_seqlen, num_q_heads, head_dim)
    k: Tensor,  # (total_seqlen, num_k_heads, head_dim)
    v: Tensor,  # (total_seqlen, num_v_heads, head_dim)
    state: Tensor,  # (num_seqs, num_v_heads, head_dim, head_dim)
    A_log: Tensor,  # (num_v_heads)
    a: Tensor,  # (total_seqlen, num_v_heads)
    dt_bias: Tensor,  # (num_v_heads)
    b: Tensor,  # (total_seqlen, num_v_heads)
    cu_seqlens: Tensor,  # (num_seqlens + 1)
    scale: float,
):
    T = q.shape[0]
    N = cu_seqlens.shape[0] - 1

    # chunk_v8 pipeline for T>=525
    # v8c (Triton H, BV=16) wins for low N — creates 8x more blocks for better SM utilization
    # v8b (CUDA tcgen05 H) wins for high N — tcgen05 MMA is faster per chunk
    if T >= 525:
        use_v8c = (N <= 2 and T >= 600) or (N == 3 and T >= 1500) or (N <= 5 and T >= 3000)
        if use_v8c:
            return chunk_v8c(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)
        else:
            return chunk_v8b(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)

    # CUDA v4 for medium workloads
    if T >= 64 or (N == 1 and T >= 46):
        return cuda_v4(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)

    # CUDA recurrent for tiny/small workloads
    o = torch.empty_like(v)
    new_state = torch.empty_like(state)
    cuda_recurrent_v1(
        q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, o, new_state
    )
    return o, new_state
