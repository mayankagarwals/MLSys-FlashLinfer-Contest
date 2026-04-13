# chunk_v7f: Algorithmic reformulation — P@h^T + A@u
#
# Math: o = scale * (q@h^T*exp(g) + A@v_new)
#       where v_new = u - w@h^T
#       = scale * (q*exp(g)@h^T + A@u - (A@w)@h^T)
#       = scale * ((q*exp(g) - A@w)@h^T + A@u)
#       = scale * (P@h^T + A@u)
#       where P = q*exp(g) - A@w
#
# Key: P@h^T uses same tcgen05 tiling as w@h^T in the H kernel
# And A@u doesn't need h at all!
#
# Pipeline:
# 1. KKT + metadata
# 2. Inverse (computes u, w)
# 3. P kernel (computes P = q*exp(g) - A@w for each chunk)
# 4. H kernel with P: computes h recurrence using w, AND P@h^T → o_inter
#    (P replaces w in the second MMA pass)
# 5. O kernel: computes A@u + o_inter → output
#
# For Phase 1 (validation): use separate P kernel + existing H + split O

from pathlib import Path
import torch
import triton
import triton.language as tl
from torch import Tensor

from . import chunk_v7 as _chunk_v7
from .chunk_v6c import merge_16x16_to_64x64_inverse_kernel_v2
from .triton_compute_p import compute_p_kernel
from .triton_o_intra_u import o_intra_u_kernel

chunk_fwd_kernel_o = _chunk_v7.chunk_fwd_kernel_o
mod = _chunk_v7.mod


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

    # Step 1: KKT + metadata
    g_cu = torch.empty_like(a, dtype=torch.float32)
    beta = torch.empty_like(b, dtype=torch.float32)
    A = torch.empty(T, H, BT, device=k.device, dtype=torch.float32)
    mod.kkt_v1b_with_meta(k, A_log, a, dt_bias, b, g_cu, beta, A, cu_seqlens, chunk_indices, chunk_offsets)

    # Step 2: Inverse (u, w)
    u = torch.empty_like(v)
    w = k.new_empty(T, H, K_dim)
    merge_16x16_to_64x64_inverse_kernel_v2[(upper_bound_chunks, H)](
        k, v, w, u, A, beta, g_cu, cu_seqlens, chunk_indices, total_chunks_ptr,
        H=H, Hg=Hg, K_dim=K_dim, V_dim=V_dim, BT=BT, num_warps=2)

    # Step 3: Compute P = q*exp(g) - A_causal@w
    p = k.new_empty(T, H, K_dim)  # same layout as w
    compute_p_kernel[(upper_bound_chunks, H)](
        q, k, w, g_cu, p, cu_seqlens, chunk_indices, total_chunks_ptr,
        H=H, Hg=Hg, K_dim=K_dim, BT=BT, num_warps=4)

    # Step 4: H kernel (unchanged — computes h, v_new, final_state)
    h = k.new_empty(upper_bound_chunks, H, V_dim, K_dim)
    final_state = torch.empty_like(state, dtype=torch.float32)
    v_new = q.new_empty(upper_bound_chunks, BT, H, V_dim)
    mod.h_v1(k, u, w, v_new, g_cu, h, state, final_state, cu_seqlens, chunk_offsets, None)

    # Step 5a: Compute o_inter = P @ h^T (using the ORIGINAL O kernel structure)
    # Since P has the SAME layout as w (and q), we can use the O kernel
    # with P as the "q" input and skip the A computation
    # For Phase 1: use a simple Triton kernel for P@h^T
    o_inter = torch.empty(T, H, V_dim, device=k.device, dtype=torch.bfloat16)
    _compute_ph_kernel[(upper_bound_chunks, H)](
        p, h, o_inter, cu_seqlens, chunk_indices, total_chunks_ptr,
        H=H, Hg=Hg, K_dim=K_dim, V_dim=V_dim, BT=BT, BV=64, num_warps=4)

    # Step 5b: O kernel: o = (o_inter + A@u) * scale
    o = torch.empty_like(v)
    o_intra_u_kernel[(upper_bound_chunks, H)](
        q, k, u, g_cu, o_inter, o, cu_seqlens, chunk_indices, total_chunks_ptr,
        scale=scale, H=H, Hg=Hg, K_dim=K_dim, V_dim=V_dim, BT=BT, BV=64, num_warps=4)

    return o, final_state


@triton.jit
def _compute_ph_kernel(
    p_ptr, h_ptr, o_inter_ptr,
    cu_seqlens_ptr, chunk_indices_ptr, total_chunks_ptr,
    H: tl.constexpr, Hg: tl.constexpr,
    K_dim: tl.constexpr, V_dim: tl.constexpr,
    BT: tl.constexpr, BV: tl.constexpr,
):
    """Compute o_inter = P @ h^T for each chunk. P has layout [T, H, K_dim]."""
    global_chunk_id = tl.program_id(0)
    head_id = tl.program_id(1)

    if global_chunk_id >= tl.load(total_chunks_ptr).to(tl.int32):
        return

    seq_id = tl.load(chunk_indices_ptr + global_chunk_id * 2).to(tl.int32)
    chunk_id = tl.load(chunk_indices_ptr + (global_chunk_id * 2 + 1)).to(tl.int32)
    bos = tl.load(cu_seqlens_ptr + seq_id).to(tl.int32)
    eos = tl.load(cu_seqlens_ptr + (seq_id + 1)).to(tl.int32)
    seqlen = eos - bos

    p_ptr += (bos * H + head_id) * K_dim
    o_inter_ptr += (bos * H + head_id) * V_dim
    h_ptr += (global_chunk_id * H + head_id) * V_dim * K_dim

    offs_t = chunk_id * BT + tl.arange(0, BT)[:, None]
    offs_k = tl.arange(0, K_dim)[None, :]
    mask_t = offs_t < seqlen

    # Load P for this chunk
    p = tl.load(p_ptr + (offs_t * (H * K_dim) + offs_k), mask=mask_t, other=0.0)

    # Compute P @ h^T for each V tile
    for i_v in tl.static_range(V_dim // BV):
        offs_v = i_v * BV + tl.arange(0, BV)[:, None]
        offs_v_block = i_v * BV + tl.arange(0, BV)[None, :]
        h = tl.load(h_ptr + (offs_v * K_dim + offs_k))
        o_inter = tl.dot(p, h.T)  # [BT, BV] — bf16 MMA, same as original O kernel's q@h^T
        tl.store(o_inter_ptr + (offs_t * (H * V_dim) + offs_v_block), o_inter.to(tl.bfloat16), mask=mask_t)
