# Simplified O kernel: o = (o_inter + A @ u) * scale
# o_inter was precomputed by H kernel as P @ h^T (stored as bf16)
# This kernel only needs q, k, u, g_cu — NO h!

import triton
import triton.language as tl


@triton.jit
def o_intra_u_kernel(
    q_ptr, k_ptr, u_ptr, g_cu_ptr,
    o_inter_ptr,   # [T, H, V_dim] bf16 — P@h^T from H kernel
    o_ptr,         # [T, H, V_dim] bf16 — final output
    cu_seqlens_ptr, chunk_indices_ptr, total_chunks_ptr,
    scale,
    H: tl.constexpr, Hg: tl.constexpr,
    K_dim: tl.constexpr, V_dim: tl.constexpr,
    BT: tl.constexpr, BV: tl.constexpr,
):
    """o = (o_inter + A @ u) * scale

    A = lower_tri(q@k^T * exp(g_diff))
    o_inter already contains P @ h^T (precomputed by fused H kernel)
    """
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
    u_ptr += (bos * H + head_id) * V_dim
    o_inter_ptr += (bos * H + head_id) * V_dim
    o_ptr += (bos * H + head_id) * V_dim
    g_cu_ptr += bos * H + head_id

    offs_t = chunk_id * BT + tl.arange(0, BT)[:, None]
    offs_k = tl.arange(0, K_dim)[None, :]
    mask_t = offs_t < seqlen

    q = tl.load(q_ptr + (offs_t * (Hg * K_dim) + offs_k), mask=mask_t, other=0.0)
    k = tl.load(k_ptr + (offs_t * (Hg * K_dim) + offs_k), mask=mask_t, other=0.0)
    t_1d = chunk_id * BT + tl.arange(0, BT)
    g_cu = tl.load(g_cu_ptr + t_1d * H, mask=t_1d < seqlen, other=0.0)

    # Compute A_causal
    A = tl.dot(q, k.T) * tl.exp(g_cu[:, None] - g_cu[None, :])
    A = tl.where(tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :], A, 0.0)

    # For each V tile
    for i_v in tl.static_range(V_dim // BV):
        offs_v_block = i_v * BV + tl.arange(0, BV)[None, :]

        # Load u (token-indexed, same layout as v)
        u = tl.load(u_ptr + (offs_t * (H * V_dim) + offs_v_block), mask=mask_t, other=0.0)

        # Load o_inter (bf16, token-indexed)
        o_inter = tl.load(o_inter_ptr + (offs_t * (H * V_dim) + offs_v_block), mask=mask_t, other=0.0)

        # o = (o_inter + A @ u) * scale
        o_intra = tl.dot(A.to(u.dtype), u)
        o = (o_inter.to(tl.float32) + o_intra) * scale

        tl.store(o_ptr + (offs_t * (H * V_dim) + offs_v_block), o.to(tl.bfloat16), mask=mask_t)
