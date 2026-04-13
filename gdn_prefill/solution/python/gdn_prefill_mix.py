import ctypes

# get weird errors without this
ctypes.CDLL("libcudart.so", mode=ctypes.RTLD_GLOBAL)

import torch
from torch import Tensor

from .cuda_recurrent_v1 import run as cuda_recurrent_v1
from .chunk_v6c import run as chunk_v6c
from .chunk_v7b import run as chunk_v7b
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

    # chunk_v7b for T>=525 (chunk pipeline beats v4 FusedPrep)
    # v6c (Triton H) for very large T with N<=2 where CUDA H has poor SM utilization
    if T >= 525:
        if N <= 2 and T >= 2500:
            return chunk_v6c(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)
        else:
            return chunk_v7b(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)

    # CUDA v4 for medium workloads — use CUDA graph for launch overhead reduction
    if T >= 64 or (N == 1 and T >= 46):
        return cuda_v4(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)

    # CUDA recurrent for tiny workloads
    o = torch.empty_like(v)
    new_state = torch.empty_like(state)
    cuda_recurrent_v1(
        q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, o, new_state
    )
    return o, new_state
