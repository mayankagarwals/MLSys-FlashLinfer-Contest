# Compute P[t] = q[t]*exp(g[t]) - A_causal[t,:] @ w for each chunk
# P is [BT, K_dim] per chunk — same shape as w
# P @ h^T replaces q@h^T*exp(g) in the fused H+O pipeline

import triton
import triton.language as tl


@triton.jit
def compute_p_kernel(
    q_ptr, k_ptr, w_ptr, g_cu_ptr, p_ptr,
    cu_seqlens_ptr, chunk_indices_ptr, total_chunks_ptr,
    H: tl.constexpr, Hg: tl.constexpr,
    K_dim: tl.constexpr,
    BT: tl.constexpr,
):
    """Compute P[t, k] = q[t, k]*exp(g[t]) - sum_{j<=t} A[t,j]*w[j, k]

    where A[t,j] = q[t]@k[j]^T * exp(g[t]-g[j]) for j<=t, 0 otherwise.

    Grid: (upper_bound_chunks, H)
    Output P has same layout as w: [T, H, K_dim] bf16
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

    q_head = head_id // (H // Hg)
    q_ptr += (bos * Hg + q_head) * K_dim
    k_ptr += (bos * Hg + q_head) * K_dim
    w_ptr += (bos * H + head_id) * K_dim
    p_ptr += (bos * H + head_id) * K_dim
    g_cu_ptr += bos * H + head_id

    offs_t = chunk_id * BT + tl.arange(0, BT)[:, None]
    offs_k = tl.arange(0, K_dim)[None, :]
    mask_t = offs_t < seqlen

    # Load q, k, w for this chunk
    q = tl.load(q_ptr + (offs_t * (Hg * K_dim) + offs_k), mask=mask_t, other=0.0)
    k = tl.load(k_ptr + (offs_t * (Hg * K_dim) + offs_k), mask=mask_t, other=0.0)
    w = tl.load(w_ptr + (offs_t * (H * K_dim) + offs_k), mask=mask_t, other=0.0)
    t_1d = chunk_id * BT + tl.arange(0, BT)
    g_cu = tl.load(g_cu_ptr + t_1d * H, mask=t_1d < seqlen, other=0.0)

    # Compute A_causal = lower_tri(q @ k^T * exp(g_diff))
    A = tl.dot(q, k.T) * tl.exp(g_cu[:, None] - g_cu[None, :])
    A = tl.where(tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :], A, 0.0)

    # Compute Q_scaled = q * exp(g)
    Q_scaled = q.to(tl.float32) * tl.exp(g_cu)[:, None]

    # Compute C = A @ w (causal attention applied to w)
    C = tl.dot(A.to(w.dtype), w)  # [BT, K_dim]

    # P = Q_scaled - C
    P = Q_scaled - C.to(tl.float32)

    # Store P in same layout as w: [T, H, K_dim]
    tl.store(p_ptr + (offs_t * (H * K_dim) + offs_k), P.to(tl.bfloat16), mask=mask_t)
