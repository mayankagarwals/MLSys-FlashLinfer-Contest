# chunk_v7c: v4 FusedPrep (KKT+inverse combined) + CUDA H + Triton O
# Eliminates A intermediate tensor (16MB) and one kernel launch

from pathlib import Path
import torch
import triton
from torch import Tensor

from . import chunk_v7 as _chunk_v7

chunk_fwd_kernel_o = _chunk_v7.chunk_fwd_kernel_o
mod = _chunk_v7.mod  # CUDA H kernel

# Import v4 module for FusedPrep
from .cuda_parallel_v4 import mod as v4_mod


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
    T, Hg, K_dim = k.shape
    N, H, V_dim, _ = state.shape
    BT = 64

    upper_bound_chunks = (N - 1) + triton.cdiv(T - (N - 1), BT)
    chunk_offsets = q.new_empty(N + 1, dtype=torch.int32)
    chunk_indices = q.new_empty((upper_bound_chunks, 2), dtype=torch.int32)
    total_chunks_ptr = chunk_offsets[N:]

    # Fused KKT + inverse + W/U: eliminates A intermediate tensor
    g_cu = torch.empty_like(a, dtype=torch.float32)
    u = torch.empty_like(v)
    w = k.new_empty(T, H, K_dim)
    v4_mod.run_fused_prep(
        k, v, A_log, a, dt_bias, b, cu_seqlens,
        g_cu, w, u, chunk_indices, chunk_offsets,
    )

    # CUDA H kernel (same as chunk_v7b)
    h = k.new_empty(upper_bound_chunks, H, V_dim, K_dim)
    final_state = torch.empty_like(state, dtype=torch.float32)
    v_new = q.new_empty(upper_bound_chunks, BT, H, V_dim)

    mod.h_v1(
        k, u, w, v_new, g_cu, h, state, final_state,
        cu_seqlens, chunk_offsets, None,
    )

    # Triton O kernel (same as chunk_v7b)
    o = torch.empty_like(v)
    BV = 64
    chunk_fwd_kernel_o[(upper_bound_chunks, H)](
        q, k, v_new, h, g_cu, o,
        cu_seqlens, chunk_indices, total_chunks_ptr,
        scale=scale, H=H, Hg=Hg, K_dim=K_dim, V_dim=V_dim, BT=BT, BV=BV,
        num_warps=4,
    )

    return o, final_state
