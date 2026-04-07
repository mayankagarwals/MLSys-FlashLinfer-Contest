import ctypes

# get weird errors without this
ctypes.CDLL("libcudart.so", mode=ctypes.RTLD_GLOBAL)

import torch
from torch import Tensor

from .cuda_recurrent_v1 import run as cuda_recurrent_v1
from .chunk_v5 import run as chunk_v5


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

    # chunk impl
    if T >= 128:
        return chunk_v5(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)

    # recurrent impl
    o = torch.empty_like(v)
    new_state = torch.empty_like(state)
    cuda_recurrent_v1(
        q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, o, new_state
    )
    return o, new_state
