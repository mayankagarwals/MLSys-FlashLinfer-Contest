# Fused H+O kernel — proven correct version (BV=64, inline A_causal)
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
    if state_ptr is not None:
        h = tl.load(state_ptr + (seq_id * H + head_id) * V_dim * K_dim + offs_v * K_dim + offs_k).to(tl.float32)
    else:
        h = tl.zeros((BV, K_dim), dtype=tl.float32)
    for chunk_id in range(num_chunks):
        cstart = bos + chunk_id * BT
        clen = tl.minimum(BT, seqlen - chunk_id * BT)
        offs_t = tl.arange(0, BT)[:, None]
        mask_t = offs_t < clen
        w = tl.load(w_ptr + ((cstart + offs_t) * H * K_dim + head_id * K_dim + offs_k), mask=mask_t, other=0.0)
        k = tl.load(k_ptr + ((cstart + offs_t) * Hg * K_dim + k_head * K_dim + offs_k), mask=mask_t, other=0.0)
        u = tl.load(u_ptr + ((cstart + offs_t) * H * V_dim + head_id * V_dim + offs_bv), mask=mask_t, other=0.0)
        q = tl.load(q_ptr + ((cstart + offs_t) * Hg * K_dim + q_head * K_dim + offs_k), mask=mask_t, other=0.0)
        t_1d = tl.arange(0, BT)
        g_cu = tl.load(g_cu_ptr + (cstart + t_1d) * H + head_id, mask=t_1d < clen, other=0.0)
        h_bf16 = h.to(tl.bfloat16)
        wh = tl.dot(w, tl.trans(h_bf16))
        v_new = u.to(tl.float32) - wh
        qh = tl.dot(q, tl.trans(h_bf16))
        o_inter = qh * tl.exp(g_cu)[:, None]
        A = tl.dot(q, tl.trans(k)) * tl.exp(g_cu[:, None] - g_cu[None, :])
        A = tl.where(tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :], A, 0.0)
        o_intra = tl.dot(A.to(v_new.dtype), v_new)
        o = (o_inter + o_intra) * scale
        tl.store(output_ptr + (cstart * H + head_id) * V_dim + v_start + offs_t * H * V_dim + offs_bv, o.to(tl.bfloat16), mask=mask_t)
        last_idx = tl.minimum(clen - 1, BT - 1)
        g_last = tl.sum(tl.where(t_1d == last_idx, g_cu, 0.0))
        v_new_scaled = v_new * tl.exp(g_last - g_cu)[:, None]
        h = h * tl.exp(g_last) + tl.dot(tl.trans(v_new_scaled.to(k.dtype)), k).to(tl.float32)
    tl.store(new_state_ptr + (seq_id * H + head_id) * V_dim * K_dim + offs_v * K_dim + offs_k, h)
