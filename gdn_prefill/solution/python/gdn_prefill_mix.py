import ctypes

# get weird errors without this
ctypes.CDLL("libcudart.so", mode=ctypes.RTLD_GLOBAL)

import torch
from torch import Tensor

from .cuda_recurrent_v1 import run as cuda_recurrent_v1
from .chunk_v6 import run as chunk_v6
from .cuda_parallel_v4 import run as cuda_v4

_rec_cache = {}  # Cache output tensors for recurrent kernel


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
    N = cu_seqlens.shape[0] - 1  # num_seqs

    # chunk_v6 (CUDA kkt + Triton) for large workloads
    if T >= 1024:
        return chunk_v6(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)

    # CUDA v4 chunk kernel for medium workloads
    # Also faster than recurrent for single-seq workloads with T>=46
    if T >= 64 or (N == 1 and T >= 46):
        return cuda_v4(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)

    # CUDA recurrent for tiny workloads (cached output tensors)
    rec_key = (T, v.shape[1], v.shape[2], state.shape[0])
    if rec_key not in _rec_cache:
        _rec_cache[rec_key] = (torch.empty_like(v), torch.empty_like(state))
    o, new_state = _rec_cache[rec_key]
    cuda_recurrent_v1(
        q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, o, new_state
    )
    return o, new_state
