import ctypes

# get weird errors without this
ctypes.CDLL("libcudart.so", mode=ctypes.RTLD_GLOBAL)

import torch
from torch import Tensor

from .cuda_recurrent_v1 import run as cuda_recurrent_v1
from .chunk_v6c import run as chunk_v6c
from .chunk_v7b import run as chunk_v7b
from .cuda_parallel_v5 import run as cuda_v5


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

    # chunk pipeline for T>=525 (Triton prep — must match reference precision)
    # v6c (Triton H) for N<=2 T>=600, v7b (CUDA H) for N>2 or T<600
    if T >= 525:
        if N <= 2 and T >= 600:
            return chunk_v6c(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)
        else:
            return chunk_v7b(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)

    # CUDA v5 for medium workloads (T>=67 for N=1, T>=64 for N>=3)
    # Recurrent is faster for T<64 N>=2 and T<67 N=1 (profiled threshold)
    if T >= 64 and (N >= 3 or T >= 67):
        return cuda_v5(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)

    # CUDA recurrent for tiny/small workloads
    o = torch.empty_like(v)
    new_state = torch.empty_like(state)
    cuda_recurrent_v1(
        q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, o, new_state
    )
    return o, new_state
