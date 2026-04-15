import ctypes

# get weird errors without this
ctypes.CDLL("libcudart.so", mode=ctypes.RTLD_GLOBAL)

import torch
from torch import Tensor

from .cuda_recurrent_v1 import run as cuda_recurrent_v1
from .chunk_v8 import run as chunk_v8
from .chunk_v9 import run as chunk_v9
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

    # Measured fast paths on the official workload set:
    # - chunk_v9 wins for the large-T/high-N region
    # - chunk_v9 also wins in the narrow midrange window around T~1k with N>=3
    # - below the chunked regime, chunk_v9 is also a clean win for 134<=T<525
    if (T >= 3999 and N >= 13) or (900 <= T <= 1100 and N >= 3):
        return chunk_v9(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)

    if 134 <= T < 525:
        return chunk_v9(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)

    # chunk_v8 pipeline for the remaining T>=525 workloads
    if T >= 525:
        return chunk_v8(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)

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
