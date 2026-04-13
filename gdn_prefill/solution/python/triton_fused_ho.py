# triton_fused_ho.py: Fused H+O kernel in Triton
# Keeps h state [V, K] in registers across chunks.
# Eliminates ~60MB intermediate memory (h, v_new global tensors).

import triton
import triton.language as tl


@triton.jit
def fused_ho_kernel(
    q_ptr, k_ptr, w_ptr, u_ptr, g_cu_ptr,
    state_ptr, cu_seqlens_ptr,
    output_ptr, new_state_ptr,
    scale,
    H: tl.constexpr, Hg: tl.constexpr,
    K_dim: tl.constexpr, V_dim: tl.constexpr,
    BT: tl.constexpr, BV: tl.constexpr,
):
    """
    One TB per (seq, head, v_tile). Processes ALL chunks sequentially.
    h state [BV, K_dim] kept in registers across the loop.
    BV < V_dim tiles the V dimension across TBs.
    """
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

    # Initialize h state [BV, K_dim] fp32 (V-tiled)
    offs_v = v_start + tl.arange(0, BV)[:, None]  # [BV, 1]
    offs_k = tl.arange(0, K_dim)[None, :]  # [1, K]

    if state_ptr is not None:
        h_base = state_ptr + (seq_id * H + head_id) * V_dim * K_dim
        h = tl.load(h_base + offs_v * K_dim + offs_k).to(tl.float32)
    else:
        h = tl.zeros((BV, K_dim), dtype=tl.float32)

    # Main chunk loop
    for chunk_id in range(num_chunks):
        cstart = bos + chunk_id * BT
        clen = tl.minimum(BT, seqlen - chunk_id * BT)

        offs_t = tl.arange(0, BT)[:, None]  # [BT, 1]
        mask_t = offs_t < clen

        # Load data for this chunk
        w_base = w_ptr + (cstart * H + head_id) * K_dim
        w = tl.load(w_base + offs_t * H * K_dim + offs_k, mask=mask_t, other=0.0)  # [BT, K]

        k_base = k_ptr + (cstart * Hg + k_head) * K_dim
        k = tl.load(k_base + offs_t * Hg * K_dim + offs_k, mask=mask_t, other=0.0)  # [BT, K]

        u_base = u_ptr + (cstart * H + head_id) * V_dim + v_start
        offs_bv = tl.arange(0, BV)[None, :]
        u = tl.load(u_base + offs_t * H * V_dim + offs_bv, mask=mask_t, other=0.0)  # [BT, BV]

        g_base = g_cu_ptr + cstart * H + head_id
        offs_t_1d = tl.arange(0, BT)
        g_cu = tl.load(g_base + offs_t_1d * H, mask=offs_t_1d < clen, other=0.0)  # [BT]

        # ═══ Step 1: v_new = U - W @ H^T ═══
        # W[BT, K] @ H^T[K, V] → [BT, V]
        h_bf16 = h.to(tl.bfloat16)
        wh = tl.dot(w, tl.trans(h_bf16))  # [BT, BV]
        v_new = u.to(tl.float32) - wh  # [BT, BV]

        # ═══ Step 2: Compute output o ═══
        # o_inter = Q @ H^T * exp(g)
        q_base = q_ptr + (cstart * Hg + q_head) * K_dim
        q = tl.load(q_base + offs_t * Hg * K_dim + offs_k, mask=mask_t, other=0.0)  # [BT, K]

        qh = tl.dot(q, tl.trans(h_bf16))  # [BT, BV]
        exp_g = tl.exp(g_cu)
        o_inter = qh * exp_g[:, None]

        A = tl.dot(q, tl.trans(k)) * tl.exp(g_cu[:, None] - g_cu[None, :])
        A = tl.where(tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :], A, 0.0)

        o_intra = tl.dot(A.to(v_new.dtype), v_new)  # [BT, BV]
        o = (o_inter + o_intra) * scale

        o_base = output_ptr + (cstart * H + head_id) * V_dim + v_start
        tl.store(o_base + offs_t * H * V_dim + offs_bv, o.to(tl.bfloat16), mask=mask_t)

        # ═══ Step 3: Update h ═══
        last_idx = tl.minimum(clen - 1, BT - 1)
        g_last = tl.sum(tl.where(offs_t_1d == last_idx, g_cu, 0.0))
        alpha = tl.exp(g_last)
        v_new_scaled = v_new * tl.exp(g_last - g_cu)[:, None]

        h = h * alpha + tl.dot(tl.trans(v_new_scaled.to(k.dtype)), k).to(tl.float32)

    # Store final state (V-tiled)
    ns_base = new_state_ptr + (seq_id * H + head_id) * V_dim * K_dim
    tl.store(ns_base + offs_v * K_dim + offs_k, h)

    # Note: v_new.dtype in dot A.to(v_new.dtype) gives fp32 since v_new is fp32.
    # The A is fp32 from the gating, and v_new is fp32. tl.dot(fp32, fp32) uses tf32.
    # To match the reference (bf16 MMA): cast both to bf16 before dot.
