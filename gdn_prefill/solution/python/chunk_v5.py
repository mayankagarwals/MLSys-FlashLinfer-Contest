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
_cv5_cache = {}  # Cache intermediate tensors to avoid per-call torch.empty overhead


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
    meta_key = (N, upper_bound_chunks)
    if meta_key not in _cv5_cache:
        _cv5_cache[meta_key] = {
            'num_chunks': q.new_empty(N, dtype=torch.int32),
            'chunk_offsets': q.new_empty(N + 1, dtype=torch.int32),
            'chunk_indices': q.new_empty((upper_bound_chunks, 2), dtype=torch.int32),
        }
    num_chunks = _cv5_cache[meta_key]['num_chunks']
    chunk_offsets = _cv5_cache[meta_key]['chunk_offsets']
    chunk_indices = _cv5_cache[meta_key]['chunk_indices']
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
    # Cache intermediate tensors to avoid per-call allocation overhead
    cache_key = (T, N, H, Hg, K_dim, V_dim, total_num_chunks)
    if cache_key not in _cv5_cache:
        dev = k.device
        _cv5_cache[cache_key] = {
            'g_cu': torch.empty(T, H, dtype=torch.float32, device=dev),
            'beta': torch.empty(T, H, dtype=torch.float32, device=dev),
            'A': torch.empty(T, H, BT, dtype=torch.float32, device=dev),
            'Ai': torch.empty(T, H, BT, dtype=k.dtype, device=dev),
            'u': torch.empty(T, H, V_dim, dtype=v.dtype, device=dev),
            'w': torch.empty(T, H, K_dim, dtype=k.dtype, device=dev),
            'h': torch.empty(total_num_chunks, H, V_dim, K_dim, dtype=k.dtype, device=dev),
            'final_state': torch.empty(N, H, V_dim, V_dim, dtype=torch.float32, device=dev),
            'v_new': torch.empty(T, H, V_dim, dtype=v.dtype, device=dev),
            'o': torch.empty(T, H, V_dim, dtype=v.dtype, device=dev),
        }
    c = _cv5_cache[cache_key]
    g_cu = c['g_cu']
    beta = c['beta']
    A = c['A']
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
    Ai = c['Ai']
    u = c['u']
    w = c['w']
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

    h = c['h']
    final_state = c['final_state']
    v_new = c['v_new']

    # reduce BV to increase no. of SMs used.
    # helpful when N * H is small.
    BV = 16
    grid = (triton.cdiv(V_dim, BV), N * H)
    chunk_gated_delta_rule_fwd_kernel_h[grid](
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
        H=H,
        Hg=Hg,
        K_dim=K_dim,
        V_dim=V_dim,
        BT=BT,
        BV=BV,
        num_warps=4,
        num_stages=3,
    )

    o = c['o']

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
