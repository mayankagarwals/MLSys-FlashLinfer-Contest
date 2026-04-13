# chunk_v9: chunk pipeline with fused H+O CUDA kernel
# Uses: kkt_v1b_with_meta → inverse (Triton) → fused_ho (CUDA)
# Eliminates h and v_new intermediate tensors (60MB for T=8192)

import os
from pathlib import Path

import torch
import triton
from torch import Tensor

os.environ["TVM_FFI_CUDA_ARCH_LIST"] = "10.0a"

import tvm_ffi

CURRENT_DIR = Path(__file__).parent

from .chunk_v6c import (
    merge_16x16_to_64x64_inverse_kernel_v2,
    mod as kkt_mod,
)
from .triton_fused_ho import fused_ho_kernel


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

    # Combined: compute chunk metadata + K@K.T + gating → A, g_cu, beta
    g_cu = torch.empty_like(a, dtype=torch.float32)
    beta = torch.empty_like(b, dtype=torch.float32)
    A = torch.empty(T, H, BT, device=k.device, dtype=torch.float32)
    kkt_mod.kkt_v1b_with_meta(
        k, A_log, a, dt_bias, b, g_cu, beta, A,
        cu_seqlens, chunk_indices, chunk_offsets,
    )

    # Inverse + W/U computation (Triton)
    u = torch.empty_like(v)
    w = k.new_empty(T, H, K_dim)
    merge_16x16_to_64x64_inverse_kernel_v2[(upper_bound_chunks, H)](
        k, v, w, u, A, beta, g_cu,
        cu_seqlens, chunk_indices, total_chunks_ptr,
        H=H, Hg=Hg, K_dim=K_dim, V_dim=V_dim, BT=BT,
        num_warps=2,
    )

    # Fused H+O Triton kernel — V-tiled, processes all chunks sequentially
    o = torch.empty_like(v)
    new_state = torch.empty_like(state, dtype=torch.float32)
    BV = 64  # V-tile size (halves h state to [64, 128] = 32KB)
    grid = (N * H, V_dim // BV)
    fused_ho_kernel[grid](
        q, k, w, u, g_cu,
        state, cu_seqlens,
        o, new_state,
        scale,
        H=H, Hg=Hg, K_dim=K_dim, V_dim=V_dim, BT=BT, BV=BV,
        num_warps=4,
    )

    return o, new_state
