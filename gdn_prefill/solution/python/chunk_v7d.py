# chunk_v7d: v7b with o_inter pre-computed during H kernel
# Phase 1: Use separate Triton kernel for o_inter (correctness validation)
# Phase 2: Fuse into CUDA H kernel once correctness is validated

from pathlib import Path
import torch
import triton
import triton.language as tl
from torch import Tensor

from . import chunk_v7 as _chunk_v7
from .chunk_v6c import merge_16x16_to_64x64_inverse_kernel_v2

chunk_fwd_kernel_o = _chunk_v7.chunk_fwd_kernel_o
export_trace = _chunk_v7.export_trace
mod = _chunk_v7.mod


@triton.jit
def compute_o_inter_kernel(
    q_ptr, h_ptr, g_cu_ptr, o_ptr,
    cu_seqlens_ptr, chunk_indices_ptr, total_chunks_ptr,
    scale,
    H: tl.constexpr, Hg: tl.constexpr,
    K_dim: tl.constexpr, V_dim: tl.constexpr,
    BT: tl.constexpr, BV: tl.constexpr,
):
    """Compute o_inter = q @ h^T * exp(g) * scale for each chunk."""
    global_chunk_id = tl.program_id(0)
    head_id = tl.program_id(1)

    if global_chunk_id >= tl.load(total_chunks_ptr).to(tl.int32):
        return

    seq_id = tl.load(chunk_indices_ptr + global_chunk_id * 2).to(tl.int32)
    chunk_id = tl.load(chunk_indices_ptr + (global_chunk_id * 2 + 1)).to(tl.int32)
    bos = tl.load(cu_seqlens_ptr + seq_id).to(tl.int32)
    eos = tl.load(cu_seqlens_ptr + (seq_id + 1)).to(tl.int32)
    seqlen = eos - bos

    q_ptr += (bos * Hg + head_id // (H // Hg)) * K_dim
    o_ptr += (bos * H + head_id) * V_dim
    h_ptr += (global_chunk_id * H + head_id) * V_dim * K_dim
    g_cu_ptr += bos * H + head_id

    offs_t = chunk_id * BT + tl.arange(0, BT)[:, None]
    offs_k = tl.arange(0, K_dim)[None, :]
    mask_t = offs_t < seqlen

    q = tl.load(q_ptr + (offs_t * (Hg * K_dim) + offs_k), mask=mask_t, other=0.0)
    t_1d = chunk_id * BT + tl.arange(0, BT)
    g_cu = tl.load(g_cu_ptr + t_1d * H, mask=t_1d < seqlen, other=0.0)
    exp_g = tl.exp(g_cu)

    for i_v in tl.static_range(V_dim // BV):
        offs_v = i_v * BV + tl.arange(0, BV)[:, None]
        offs_v_block = i_v * BV + tl.arange(0, BV)[None, :]
        h = tl.load(h_ptr + (offs_v * K_dim + offs_k))
        o_inter = tl.dot(q, h.T) * exp_g[:, None] * scale
        # Store as fp32 to scratch buffer (o_ptr points to fp32 tensor)
        tl.store(o_ptr + (offs_t * (H * V_dim) + offs_v_block), o_inter, mask=mask_t)


@triton.jit
def add_o_intra_kernel(
    q_ptr, k_ptr, v_ptr, g_cu_ptr, scratch_ptr, o_ptr,
    cu_seqlens_ptr, chunk_indices_ptr, total_chunks_ptr,
    scale,
    H: tl.constexpr, Hg: tl.constexpr,
    K_dim: tl.constexpr, V_dim: tl.constexpr,
    BT: tl.constexpr, BV: tl.constexpr,
):
    """Compute o = (o_inter_fp32 + A @ v_new) * scale, store as bf16."""
    global_chunk_id = tl.program_id(0)
    head_id = tl.program_id(1)

    if global_chunk_id >= tl.load(total_chunks_ptr).to(tl.int32):
        return

    seq_id = tl.load(chunk_indices_ptr + global_chunk_id * 2).to(tl.int32)
    chunk_id = tl.load(chunk_indices_ptr + (global_chunk_id * 2 + 1)).to(tl.int32)
    bos = tl.load(cu_seqlens_ptr + seq_id).to(tl.int32)
    eos = tl.load(cu_seqlens_ptr + (seq_id + 1)).to(tl.int32)
    seqlen = eos - bos

    q_ptr += (bos * Hg + head_id // (H // Hg)) * K_dim
    k_ptr += (bos * Hg + head_id // (H // Hg)) * K_dim
    # v_new uses chunk-indexed layout: (ub, BT, H, V_dim)
    v_ptr += ((global_chunk_id - chunk_id) * BT * H + head_id) * V_dim
    scratch_ptr += (bos * H + head_id) * V_dim  # fp32 o_inter scratch (token-indexed)
    o_ptr += (bos * H + head_id) * V_dim         # bf16 output (token-indexed)
    g_cu_ptr += bos * H + head_id

    offs_t = chunk_id * BT + tl.arange(0, BT)[:, None]
    offs_k = tl.arange(0, K_dim)[None, :]
    mask_t = offs_t < seqlen

    q = tl.load(q_ptr + (offs_t * (Hg * K_dim) + offs_k), mask=mask_t, other=0.0)
    k = tl.load(k_ptr + (offs_t * (Hg * K_dim) + offs_k), mask=mask_t, other=0.0)
    t_1d = chunk_id * BT + tl.arange(0, BT)
    g_cu = tl.load(g_cu_ptr + t_1d * H, mask=t_1d < seqlen, other=0.0)

    A = tl.dot(q, k.T) * tl.exp(g_cu[:, None] - g_cu[None, :])
    A = tl.where(tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :], A, 0.0)

    for i_v in tl.static_range(V_dim // BV):
        offs_v_block = i_v * BV + tl.arange(0, BV)[None, :]
        v = tl.load(v_ptr + (offs_t * (H * V_dim) + offs_v_block), mask=mask_t, other=0.0)
        # Load o_inter from fp32 scratch
        o_inter = tl.load(scratch_ptr + (offs_t * (H * V_dim) + offs_v_block), mask=mask_t, other=0.0)
        # o = o_inter + A @ v_new (both parts already include their scaling via o_inter having scale=1.0)
        o_intra = tl.dot(A.to(v.dtype), v)
        o = (o_inter + o_intra) * scale
        tl.store(o_ptr + (offs_t * (H * V_dim) + offs_v_block), o.to(tl.bfloat16), mask=mask_t)


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

    mod.h_v1(k, u, w, v_new, g_cu, h, state, final_state, cu_seqlens, chunk_offsets, None)

    # Split O into: o_inter (q@h^T) + o_intra (A@v_new)
    # Use fp32 scratch for o_inter to avoid bf16 precision loss
    o_scratch = torch.empty(T, H, V_dim, device=v.device, dtype=torch.float32)

    # Step 1: o_inter = q @ h^T * exp(g) (fp32, no scale yet)
    BV = 64
    compute_o_inter_kernel[(upper_bound_chunks, H)](
        q, h, g_cu, o_scratch, cu_seqlens, chunk_indices, total_chunks_ptr,
        scale=1.0, H=H, Hg=Hg, K_dim=K_dim, V_dim=V_dim, BT=BT, BV=BV, num_warps=4)

    # Step 2: o = (o_inter + A @ v_new) * scale
    o = torch.empty_like(v)
    add_o_intra_kernel[(upper_bound_chunks, H)](
        q, k, v_new, g_cu, o_scratch, o, cu_seqlens, chunk_indices, total_chunks_ptr,
        scale=scale, H=H, Hg=Hg, K_dim=K_dim, V_dim=V_dim, BT=BT, BV=BV, num_warps=4)

    return o, final_state
