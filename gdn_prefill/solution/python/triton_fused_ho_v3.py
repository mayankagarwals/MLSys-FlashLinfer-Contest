# Fused H+O kernel v3 — tf32 precision for h_update (compounding operation)
# Uses BV as constexpr, supports BV=32 or BV=64
import triton
import triton.language as tl

@triton.jit
def fused_ho_kernel_v3(
    q_ptr, k_ptr, w_ptr, u_ptr, g_cu_ptr,
    state_ptr, cu_seqlens_ptr,
    output_ptr, new_state_ptr,
    scale,
    H: tl.constexpr, Hg: tl.constexpr,
    K_dim: tl.constexpr, V_dim: tl.constexpr,
    BT: tl.constexpr, BV: tl.constexpr,
):
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

    offs_v = v_start + tl.arange(0, BV)[:, None]
    offs_k = tl.arange(0, K_dim)[None, :]
    offs_bv = tl.arange(0, BV)[None, :]

    # Load initial h state [BV, K_dim] in fp32
    if state_ptr is not None:
        h = tl.load(state_ptr + (seq_id * H + head_id) * V_dim * K_dim + offs_v * K_dim + offs_k).to(tl.float32)
    else:
        h = tl.zeros((BV, K_dim), dtype=tl.float32)

    for chunk_id in range(num_chunks):
        cstart = bos + chunk_id * BT
        clen = tl.minimum(BT, seqlen - chunk_id * BT)

        offs_t = tl.arange(0, BT)[:, None]
        mask_t = offs_t < clen
        t_1d = tl.arange(0, BT)
        mask_1d = t_1d < clen

        # === Stage 1: Load w, u, compute v_new = u - w @ h^T ===
        w = tl.load(w_ptr + ((cstart + offs_t) * H * K_dim + head_id * K_dim + offs_k), mask=mask_t, other=0.0)
        u = tl.load(u_ptr + ((cstart + offs_t) * H * V_dim + head_id * V_dim + offs_bv), mask=mask_t, other=0.0)

        # v_new computation: use tf32 for closer match to fp32 reference
        wh = tl.dot(w.to(tl.float32), tl.trans(h), input_precision="tf32")  # [BT, BV]
        v_new = u.to(tl.float32) - wh  # fp32

        # === Stage 2: Load q, k, g_cu, compute output ===
        k_chunk = tl.load(k_ptr + ((cstart + offs_t) * Hg * K_dim + k_head * K_dim + offs_k), mask=mask_t, other=0.0)
        q_chunk = tl.load(q_ptr + ((cstart + offs_t) * Hg * K_dim + q_head * K_dim + offs_k), mask=mask_t, other=0.0)
        g_cu = tl.load(g_cu_ptr + (cstart + t_1d) * H + head_id, mask=mask_1d, other=0.0)

        # o_inter = q @ h^T * exp(g) — tf32 for precision
        qh = tl.dot(q_chunk.to(tl.float32), tl.trans(h), input_precision="tf32")  # [BT, BV]
        o_inter = qh * tl.exp(g_cu)[:, None]

        # A_causal = lower_tri(q @ k^T * exp(g_diff))
        A = tl.dot(q_chunk, tl.trans(k_chunk)) * tl.exp(g_cu[:, None] - g_cu[None, :])
        A = tl.where(tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :], A, 0.0)

        # o_intra = A @ v_new — bf16 like separate O kernel
        o_intra = tl.dot(A.to(tl.bfloat16), v_new.to(tl.bfloat16))
        o = (o_inter + o_intra) * scale

        tl.store(output_ptr + (cstart * H + head_id) * V_dim + v_start + offs_t * H * V_dim + offs_bv, o.to(tl.bfloat16), mask=mask_t)

        # === Stage 3: Update h (tf32 for compounding precision) ===
        last_idx = tl.minimum(clen - 1, BT - 1)
        g_last = tl.sum(tl.where(t_1d == last_idx, g_cu, 0.0))

        # Explicitly mask padding
        v_new_masked = tl.where(mask_t, v_new, 0.0)
        v_new_scaled = v_new_masked * tl.exp(g_last - g_cu)[:, None]
        v_new_scaled = tl.where(mask_t, v_new_scaled, 0.0)

        # tf32 h update — closer to fp32 reference, reduces compounding error
        h = h * tl.exp(g_last) + tl.dot(tl.trans(v_new_scaled), k_chunk.to(tl.float32), input_precision="tf32")

    # Store final state
    tl.store(new_state_ptr + (seq_id * H + head_id) * V_dim * K_dim + offs_v * K_dim + offs_k, h)
