import ctypes

# get weird errors without this
ctypes.CDLL("libcudart.so", mode=ctypes.RTLD_GLOBAL)

import torch
from torch import Tensor

from .cuda_recurrent_v1 import run as cuda_recurrent_v1
from .chunk_v5 import run as chunk_v5
from .cuda_parallel_v3 import run as cuda_v3


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

    # chunk impl for large workloads
    if T >= 4096:
        return chunk_v5(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)

    # CUDA v3 for small/medium workloads (2x faster than Triton for T<256)
    return cuda_v3(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)
