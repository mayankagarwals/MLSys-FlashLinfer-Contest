# chunk_v7b but using V-tiled H kernel for better SM utilization on low-N workloads
from pathlib import Path
import torch
import triton
from torch import Tensor
from . import chunk_v7 as _chunk_v7
from .chunk_v6c import merge_16x16_to_64x64_inverse_kernel_v2

chunk_fwd_kernel_o = _chunk_v7.chunk_fwd_kernel_o
mod = _chunk_v7.mod

def run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    T, Hg, K_dim = k.shape
    N, H, V_dim, _ = state.shape
    BT = 64
    upper_bound_chunks = (N - 1) + triton.cdiv(T - (N - 1), BT)
    chunk_offsets = q.new_empty(N + 1, dtype=torch.int32)
    chunk_indices = q.new_empty((upper_bound_chunks, 2), dtype=torch.int32)
    total_chunks_ptr = chunk_offsets[N:]
    g_cu = torch.empty_like(a, dtype=torch.float32)
    beta = torch.empty_like(b, dtype=torch.float32)
    A = torch.empty(T, H, BT, device=k.device, dtype=torch.float32)
    mod.kkt_v1b_with_meta(k, A_log, a, dt_bias, b, g_cu, beta, A, cu_seqlens, chunk_indices, chunk_offsets)
    u = torch.empty_like(v)
    w = k.new_empty(T, H, K_dim)
    merge_16x16_to_64x64_inverse_kernel_v2[(upper_bound_chunks, H)](
        k, v, w, u, A, beta, g_cu, cu_seqlens, chunk_indices, total_chunks_ptr,
        H=H, Hg=Hg, K_dim=K_dim, V_dim=V_dim, BT=BT, num_warps=2)
    h = k.new_empty(upper_bound_chunks, H, V_dim, K_dim)
    final_state = torch.empty_like(state, dtype=torch.float32)
    v_new = q.new_empty(upper_bound_chunks, BT, H, V_dim)
    # Use V-tiled H kernel (2x more blocks for better SM utilization)
    mod.h_v1_vtile(k, u, w, v_new, g_cu, h, state, final_state, cu_seqlens, chunk_offsets, None)
    o = torch.empty_like(v)
    BV = 64
    chunk_fwd_kernel_o[(upper_bound_chunks, H)](
        q, k, v_new, h, g_cu, o, cu_seqlens, chunk_indices, total_chunks_ptr,
        scale=scale, H=H, Hg=Hg, K_dim=K_dim, V_dim=V_dim, BT=BT, BV=BV, num_warps=4)
    return o, final_state
