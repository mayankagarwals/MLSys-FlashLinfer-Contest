# from triton_v4, but replace kkt kernel with CUDA version

import os
from pathlib import Path

import torch
import triton
import tvm_ffi
from torch import Tensor

from .triton_v4 import (
    chunk_fwd_kernel_o,
    chunk_gated_delta_rule_fwd_kernel_h,
    compute_chunks_kernel,
    merge_16x16_to_64x64_inverse_kernel,
)

os.environ["TVM_FFI_CUDA_ARCH_LIST"] = "10.0a"

CURRENT_DIR = Path(__file__).parent

lib_path = tvm_ffi.cpp.build(
    name="gdn_prefill_cuda",
    cuda_files=[
        str(CURRENT_DIR / "cuda_kkt_v1.cu"),
        str(CURRENT_DIR / "cuda_h_v1.cu"),
    ],
    extra_cflags=["-O3"],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "-lineinfo",
    ],
    extra_ldflags=["-lcuda"],
)

mod = tvm_ffi.load_module(lib_path)


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

    # # PyTorch version
    # num_chunks = triton.cdiv(cu_seqlens.diff(1), BT)  # for each sequence
    # chunk_offsets = F.pad(num_chunks, (1, 0)).cumsum(0)

    # # 1st value is sequence ID, 2nd value is chunk_id within that sequence
    # indices = torch.cat([torch.arange(n) for n in num_chunks.tolist()])
    # chunk_indices = torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1)
    # chunk_indices = chunk_indices.to(cu_seqlens.device, non_blocking=True)
    # total_num_chunks = chunk_indices.shape[0]

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
    total_num_chunks = chunk_offsets[-1].item()  # CUDA sync

    # this kernel does multiple things:
    # - compute K @ K.T
    # - compute g and its chunk local cumsum
    # - compute beta
    # - compute strictLower(beta * Gamma * (K @ K.T))
    # NOTE: transpose g_cu and beta to make them T-contiguous?
    g_cu = torch.empty_like(a, dtype=torch.float32)
    beta = torch.empty_like(b, dtype=torch.float32)
    A = torch.empty(T, H, BT, device=k.device, dtype=torch.float32)
    mod.kkt_v1(
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
        total_num_chunks,
    )

    # - compute Ai = inverse(I + strictTriu(A))
    # - obtain WY representation: U = Ai @ V and W = (Ai * g_cu) @ K
    Ai = torch.empty_like(A, dtype=k.dtype)  # BF16
    u = torch.empty_like(v)
    w = k.new_empty(T, H, K_dim)
    merge_16x16_to_64x64_inverse_kernel[(total_num_chunks, H)](
        k,
        v,
        w,
        u,
        A,
        Ai,
        beta,
        g_cu,
        cu_seqlens,
        chunk_indices,
        H=H,
        Hg=Hg,
        K_dim=K_dim,
        V_dim=V_dim,
        BT=BT,
        DOT_PRECISION="tf32x3",  # using tf32 may cause NaN
        num_warps=2,
    )

    h = k.new_empty(total_num_chunks, H, V_dim, K_dim)
    final_state = torch.empty_like(state, dtype=torch.float32)
    v_new = torch.empty_like(u)

    # reduce BV to increase no. of SMs used.
    # helpful when N * H is small.
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
    )
    # BV = 16
    # grid = (triton.cdiv(V_dim, BV), N * H)
    # chunk_gated_delta_rule_fwd_kernel_h[grid](
    #     k,
    #     u,
    #     w,
    #     v_new,
    #     g_cu,
    #     h,
    #     state,
    #     final_state,
    #     cu_seqlens,
    #     chunk_offsets,
    #     H=H,
    #     Hg=Hg,
    #     K_dim=K_dim,
    #     V_dim=V_dim,
    #     BT=BT,
    #     BV=BV,
    #     num_warps=4,
    #     num_stages=3,
    # )

    o = torch.empty_like(v)

    # we only need separate o kernel if h kernel is too small?
    BV = 64
    grid = (triton.cdiv(V_dim, BV), total_num_chunks, H)
    chunk_fwd_kernel_o[grid](
        q,
        k,
        v_new,
        h,
        g_cu,
        o,
        cu_seqlens,
        chunk_indices,
        scale=scale,
        H=H,
        Hg=Hg,
        K_dim=K_dim,
        V_dim=V_dim,
        BT=BT,
        BV=BV,
        num_warps=8,
    )

    return o, final_state
