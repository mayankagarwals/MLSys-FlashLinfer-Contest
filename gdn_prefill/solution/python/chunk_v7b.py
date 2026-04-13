# from chunk_v7, but use chunk_v6c inverse kernel

from pathlib import Path

import torch
import triton
from torch import Tensor

from . import chunk_v7 as _chunk_v7
from .chunk_v6c import (
    compute_chunks_kernel,
    merge_16x16_to_64x64_inverse_kernel_v2,
)

chunk_fwd_kernel_o = _chunk_v7.chunk_fwd_kernel_o
export_trace = _chunk_v7.export_trace
mod = _chunk_v7.mod

_FLAG = None


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

    # prepare chunk metadata
    BT = 64

    # Triton version
    # flag for grid sync
    global _FLAG
    if _FLAG is None:
        _FLAG = q.new_zeros(1, dtype=torch.int32)

    # we allocate more than enough for chunk_indices so that we don't need to know
    # the value of total_num_chunks before calling the kernel.
    upper_bound_chunks = (N - 1) + triton.cdiv(T - (N - 1), BT)
    num_chunks = q.new_empty(N, dtype=torch.int32)
    chunk_offsets = q.new_empty(N + 1, dtype=torch.int32)
    chunk_indices = q.new_empty((upper_bound_chunks, 2), dtype=torch.int32)
    compute_chunks_kernel[(N,)](
        cu_seqlens,
        num_chunks,
        chunk_offsets,
        chunk_indices,
        _FLAG,
        N=N,
        BT=BT,
        # max N is 57 -> max BLOCK_SIZE is 64, still very small
        BLOCK_SIZE=triton.next_power_of_2(N),
    )
    total_chunks_ptr = chunk_offsets[N:]

    # this kernel does multiple things:
    # - compute K @ K.T
    # - compute g and its chunk local cumsum
    # - compute beta
    # - compute strictLower(beta * Gamma * (K @ K.T))
    g_cu = torch.empty_like(a, dtype=torch.float32)
    beta = torch.empty_like(b, dtype=torch.float32)
    A = torch.empty(T, H, BT, device=k.device, dtype=torch.float32)
    mod.kkt_v1b(
        k,
        A_log,
        a,
        dt_bias,
        b,
        g_cu,
        beta,
        A,
        cu_seqlens,
        chunk_indices,
        total_chunks_ptr,
    )

    # - compute Ai = inverse(I + strictTriu(A))
    # - obtain WY representation: U = Ai @ V and W = (Ai * g_cu) @ K
    u = torch.empty_like(v)
    w = k.new_empty(T, H, K_dim)
    merge_16x16_to_64x64_inverse_kernel_v2[(upper_bound_chunks, H)](
        k,
        v,
        w,
        u,
        A,
        beta,
        g_cu,
        cu_seqlens,
        chunk_indices,
        total_chunks_ptr,
        H=H,
        Hg=Hg,
        K_dim=K_dim,
        V_dim=V_dim,
        BT=BT,
        num_warps=2,
    )

    # padded v_new so we can use TMA store for v_new in H kernel
    h = k.new_empty(upper_bound_chunks, H, V_dim, K_dim)
    final_state = torch.empty_like(state, dtype=torch.float32)
    v_new = q.new_empty(upper_bound_chunks, BT, H, V_dim)

    # uncomment to enable profiling
    profiler = None
    # profiler = torch.zeros(148, 10, 1 + 1000 * 2, dtype=torch.int64, device="cuda")
    mod.h_v1(
        k,
        u,
        w,
        v_new,
        g_cu,
        h,
        state,
        final_state,
        cu_seqlens,
        chunk_offsets,
        profiler,
    )

    if profiler is not None:
        export_trace(profiler, Path("trace.json.gz"))

    o = torch.empty_like(v)

    # we only need separate o kernel if h kernel is too small?
    BV = 64
    chunk_fwd_kernel_o[(upper_bound_chunks, H)](
        q,
        k,
        v_new,
        h,
        g_cu,
        o,
        cu_seqlens,
        chunk_indices,
        total_chunks_ptr,
        scale=scale,
        H=H,
        Hg=Hg,
        K_dim=K_dim,
        V_dim=V_dim,
        BT=BT,
        BV=BV,
        num_warps=4,
    )

    return o, final_state
