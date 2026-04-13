# triton_fused_ho.py: Fused H+O kernel with precomputed A_causal
# Two kernels:
# 1. precompute_attn_kernel: Q@K^T * gating → A_causal (one per chunk/head)
# 2. fused_ho_kernel: H recurrence + O output using precomputed A_causal (V-tiled)

import triton
import triton.language as tl


@triton.jit
def precompute_attn_kernel(
    q_ptr, k_ptr, g_cu_ptr, attn_ptr,
    cu_seqlens_ptr, chunk_indices_ptr, total_chunks_ptr,
    H: tl.constexpr, Hg: tl.constexpr, K_dim: tl.constexpr,
    BT: tl.constexpr,
):
    """Precompute A_causal = lower(Q@K^T * exp(g_i - g_j)) for each chunk."""
    chunk_id = tl.program_id(0)
    head_id = tl.program_id(1)

    if chunk_id >= tl.load(total_chunks_ptr).to(tl.int32):
        return

    seq_id = tl.load(chunk_indices_ptr + chunk_id * 2).to(tl.int32)
    local_chunk = tl.load(chunk_indices_ptr + (chunk_id * 2 + 1)).to(tl.int32)
    bos = tl.load(cu_seqlens_ptr + seq_id).to(tl.int32)
    eos = tl.load(cu_seqlens_ptr + (seq_id + 1)).to(tl.int32)
    seqlen = eos - bos

    cstart = bos + local_chunk * BT

    q_base = q_ptr + (cstart * Hg + head_id // (H // Hg)) * K_dim
    k_base = k_ptr + (cstart * Hg + head_id // (H // Hg)) * K_dim
    g_base = g_cu_ptr + cstart * H + head_id

    offs_t = local_chunk * BT + tl.arange(0, BT)[:, None]
    offs_k = tl.arange(0, K_dim)[None, :]
    mask_t = offs_t < seqlen

    q = tl.load(q_base + (offs_t * Hg * K_dim + offs_k), mask=mask_t, other=0.0)
    k = tl.load(k_base + (offs_t * Hg * K_dim + offs_k), mask=mask_t, other=0.0)
    t_1d = local_chunk * BT + tl.arange(0, BT)
    g_cu = tl.load(g_base + t_1d * H, mask=t_1d < seqlen, other=0.0)

    A = tl.dot(q, k.T) * tl.exp(g_cu[:, None] - g_cu[None, :])
    A = tl.where(tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :], A, 0.0)

    # Store A_causal as bf16 [chunk_id, head_id, BT, BT]
    attn_base = attn_ptr + (chunk_id * H + head_id) * BT * BT
    offs_out = tl.arange(0, BT)[:, None] * BT + tl.arange(0, BT)[None, :]
    mask_out = (tl.arange(0, BT)[:, None] < seqlen - local_chunk * BT)
    tl.store(attn_base + offs_out, A.to(tl.bfloat16), mask=mask_out)


@triton.jit
def fused_ho_kernel(
    q_ptr, k_ptr, w_ptr, u_ptr, g_cu_ptr, attn_ptr,
    state_ptr, cu_seqlens_ptr, chunk_offsets_ptr,
    output_ptr, new_state_ptr,
    scale,
    H: tl.constexpr, Hg: tl.constexpr,
    K_dim: tl.constexpr, V_dim: tl.constexpr,
    BT: tl.constexpr, BV: tl.constexpr,
):
    """Fused H+O with precomputed A_causal. V-tiled, processes all chunks sequentially."""
    pid = tl.program_id(0)
    v_tile = tl.program_id(1)
    head_id = pid % H
    seq_id = pid // H

    bos = tl.load(cu_seqlens_ptr + seq_id).to(tl.int32)
    eos = tl.load(cu_seqlens_ptr + seq_id + 1).to(tl.int32)
    seqlen = eos - bos
    num_chunks = (seqlen + BT - 1) // BT

    k_head = head_id // (H // Hg)
    q_head = head_id // (H // Hg)
    v_start = v_tile * BV

    # Read chunk_offset for this sequence
    chunk_offset = tl.load(chunk_offsets_ptr + seq_id).to(tl.int32)

    # Initialize h state [BV, K_dim] fp32
    offs_v = v_start + tl.arange(0, BV)[:, None]
    offs_k = tl.arange(0, K_dim)[None, :]
    offs_bv = tl.arange(0, BV)[None, :]

    if state_ptr is not None:
        h_base = state_ptr + (seq_id * H + head_id) * V_dim * K_dim
        h = tl.load(h_base + offs_v * K_dim + offs_k).to(tl.float32)
    else:
        h = tl.zeros((BV, K_dim), dtype=tl.float32)

    for chunk_id in range(num_chunks):
        cstart = bos + chunk_id * BT
        clen = tl.minimum(BT, seqlen - chunk_id * BT)

        offs_t = tl.arange(0, BT)[:, None]
        mask_t = offs_t < clen

        # Load per-chunk data
        w = tl.load(w_ptr + ((cstart + offs_t) * H * K_dim + head_id * K_dim + offs_k),
                     mask=mask_t, other=0.0)
        k = tl.load(k_ptr + ((cstart + offs_t) * Hg * K_dim + k_head * K_dim + offs_k),
                     mask=mask_t, other=0.0)
        u = tl.load(u_ptr + ((cstart + offs_t) * H * V_dim + head_id * V_dim + offs_bv),
                     mask=mask_t, other=0.0)
        q = tl.load(q_ptr + ((cstart + offs_t) * Hg * K_dim + q_head * K_dim + offs_k),
                     mask=mask_t, other=0.0)

        t_1d = tl.arange(0, BT)
        g_cu = tl.load(g_cu_ptr + (cstart + t_1d) * H + head_id, mask=t_1d < clen, other=0.0)

        # Compute A_causal inline (matching reference exactly)
        A_causal = tl.dot(q, tl.trans(k)) * tl.exp(g_cu[:, None] - g_cu[None, :])
        A_causal = tl.where(tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :], A_causal, 0.0)

        # Step 1: v_new = U - W @ H^T
        h_bf16 = h.to(tl.bfloat16)
        wh = tl.dot(w, tl.trans(h_bf16))
        v_new = u.to(tl.float32) - wh

        # Step 2: o_inter = Q @ H^T * exp(g)
        qh = tl.dot(q, tl.trans(h_bf16))
        exp_g = tl.exp(g_cu)
        o_inter = qh * exp_g[:, None]

        # Step 3: o_intra = A_causal @ v_new (bf16 MMA matching reference)
        o_intra = tl.dot(A_causal.to(tl.bfloat16), v_new.to(tl.bfloat16))

        # Output
        o = (o_inter + o_intra) * scale
        o_base = output_ptr + (cstart * H + head_id) * V_dim + v_start
        tl.store(o_base + offs_t * H * V_dim + offs_bv, o.to(tl.bfloat16), mask=mask_t)

        # Step 4: Update h
        last_idx = tl.minimum(clen - 1, BT - 1)
        g_last = tl.sum(tl.where(t_1d == last_idx, g_cu, 0.0))
        alpha = tl.exp(g_last)
        v_new_scaled = v_new * tl.exp(g_last - g_cu)[:, None]
        h = h * alpha + tl.dot(tl.trans(v_new_scaled.to(k.dtype)), k).to(tl.float32)

    # Store final state
    ns_base = new_state_ptr + (seq_id * H + head_id) * V_dim * K_dim
    tl.store(ns_base + offs_v * K_dim + offs_k, h)
