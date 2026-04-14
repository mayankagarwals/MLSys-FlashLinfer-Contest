# chunk_v7b_v5: Triton prep (KKT + inverse) + CUDA H+O (h_kernel_v5 + OOutputKernel)
# Saves Python inter-kernel gap between H and O launches vs chunk_v7b.

from pathlib import Path

import torch
import triton
from torch import Tensor

from . import chunk_v7 as _chunk_v7
from .chunk_v6c import (
    merge_16x16_to_64x64_inverse_kernel_v2,
)
from .cuda_parallel_v5 import run_h_and_o

mod = _chunk_v7.mod


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
    T, Hg, K_dim = k.shape
    N, H, V_dim, _ = state.shape

    BT = 64

    upper_bound_chunks = (N - 1) + triton.cdiv(T - (N - 1), BT)
    chunk_offsets = q.new_empty(N + 1, dtype=torch.int32)
    chunk_indices = q.new_empty((upper_bound_chunks, 2), dtype=torch.int32)
    total_chunks_ptr = chunk_offsets[N:]

    # Combined: compute chunk metadata + K@K.T + gating -> A, g_cu, beta
    g_cu = torch.empty_like(a, dtype=torch.float32)
    beta = torch.empty_like(b, dtype=torch.float32)
    A = torch.empty(T, H, BT, device=k.device, dtype=torch.float32)
    mod.kkt_v1b_with_meta(
        k, A_log, a, dt_bias, b, g_cu, beta, A,
        cu_seqlens, chunk_indices, chunk_offsets,
    )

    # Inverse + W/U computation (Triton — matches reference precision)
    u = torch.empty_like(v)
    w = k.new_empty(T, H, K_dim)
    merge_16x16_to_64x64_inverse_kernel_v2[(upper_bound_chunks, H)](
        k, v, w, u, A, beta, g_cu,
        cu_seqlens, chunk_indices, total_chunks_ptr,
        H=H, Hg=Hg, K_dim=K_dim, V_dim=V_dim, BT=BT, num_warps=2,
    )

    # Combined H+O in single C++ call (h_kernel_v5 + OOutputKernel)
    o, final_state = run_h_and_o(
        q, k, w, u, g_cu, state, cu_seqlens, chunk_offsets, scale
    )

    return o, final_state
