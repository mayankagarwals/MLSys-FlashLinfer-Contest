"""
V2: Triton 
"""

import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def _alloc_fn(size: int, alignment: int, stream: int | None):
    return torch.empty(size, device="cuda", dtype=torch.int8)


triton.set_allocator(_alloc_fn)


def _matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Float32 matmul for numerical stability (matches the contest reference)."""
    return a.float() @ b.float()


# === [Triton replacement point -- Stage 1: gate precompute] ==================
def _compute_gate_and_beta(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """g = exp(-exp(A_log) * softplus(a + dt_bias)),  beta = sigmoid(b)."""
    x = a.float() + dt_bias.float()  # [T, HV]
    g = -torch.exp(A_log.float()) * F.softplus(x)  # [T, HV]
    beta = torch.sigmoid(b.float())  # [T, HV]
    return g, beta


@triton.jit
def _unit_lower_inverse(A_orig, BT: tl.constexpr):
    """(I + A)^{-1}: tf32 Neumann on 16x16 + tf32x3 Newton-Schulz refinement."""
    idx = tl.arange(0, BT)
    m_I = tl.where(idx[:, None] == idx[None, :], 1.0, 0.0)

    # (I + A)^{-1} = (I - A)(I + A^2)(I + A^4)(I + A^8) for BT=16.
    A = A_orig
    Ai = m_I - A
    A = tl.dot(A, A)
    Ai = tl.dot(Ai, m_I + A)
    A = tl.dot(A, A)
    Ai = tl.dot(Ai, m_I + A)
    A = tl.dot(A, A)
    Ai = tl.dot(Ai, m_I + A)

    # Newton-Schulz: Ai <- Ai @ (2I - (I+A) @ Ai), squares error E -> E^2.
    MAi = Ai + tl.dot(A_orig, Ai, input_precision="tf32x3")
    Ai = tl.dot(Ai, 2.0 * m_I - MAi, input_precision="tf32x3")
    return Ai



@triton.jit
def _recurrent_sequence_kernel_1(
    k_HK,  # [seq_len, HV, K]
    v_HV,  # [seq_len, HV, V]
    g_H,  # [seq_len, HV]
    beta_H,  # [seq_len, HV]
    seq_len,
    u, 
    w,
    NUM_HEADS: tl.constexpr,
    BT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    head = tl.program_id(axis=0)
    chunk_idx = tl.program_id(axis=1)

    start = chunk_idx * BT

    seq = start + tl.arange(0, BT)          # [BT]

    offsets = seq*NUM_HEADS + head
    mask = seq < seq_len
    G = tl.load(g_H + offsets, mask = mask, other=0.0) # [BT]
    G = tl.cumsum(G, axis=0)  # [BT] in log space

    B = tl.load(beta_H + offsets, mask = mask, other=0.0)# [BT]

    dim = tl.arange(0, HEAD_DIM)            # [128]     
    offsets = (
        seq[:, None] * NUM_HEADS * HEAD_DIM
        + head * HEAD_DIM
        + dim[None, :]
    )  # [BT, HEAD_DIM]
    mask = seq[:, None] < seq_len 

    V = tl.load(v_HV + offsets, mask = mask, other = 0.0).to(tl.float32)# [BT, V]
    K = tl.load(k_HK + offsets, mask = mask, other = 0.0).to(tl.float32) # [BT, K]
    

    Gamma = tl.exp(G[:, None] - G[None, :]) # [N, N]
    C = tl.dot(K, tl.trans(K)) # [N, N]
    C = Gamma * C # [N, N] (pointwise mul)
    C = B[:, None] * C # [N, N] (broadcast)

    row = tl.arange(0, BT)[:, None]  # [BT, 1]
    col = tl.arange(0, BT)[None, :]  # [1, BT]
    lower_tri_mask = row > col 
    C = tl.where(lower_tri_mask, C, 0.0)

    # L is strictly lower; T = (I + L)^{-1} diag(B)
    L = C
    T = _unit_lower_inverse(L, BT=BT)
    T = T * B[None, :]

    U = tl.dot(T , V)  # [N, V]
    W = tl.dot((T * tl.exp(G[None, :])) , K) # [N, K]

    tl.store(u + offsets, U, mask)
    tl.store(w + offsets, W, mask)




@triton.jit
def _recurrent_sequence_kernel_2(
    q_HK,  # [seq_len, HV, K]   (q broadcast to HV heads)
    k_HK,  # [seq_len, HV, K]
    g_H,  # [seq_len, HV]
    state_HKV,  # [HV, K, V]  (k-first internal layout)
    scale,
    out, # [seq_len, HV, K], 
    seq_len,
    u,
    w,
    chunk_state_out_ptr,
    v_p_ptr,
    NUM_CHUNKS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    BT: tl.constexpr, 
    HEAD_DIM: tl.constexpr,
    BV: tl.constexpr
):
    head = tl.program_id(axis=0)
    v_offset_id = tl.program_id(axis=1)
    
    # assert in main code that BV divides HEAD_DIM so we don't need to worry about masking
    bv_offsets =  v_offset_id*BV + tl.arange(0, BV)
    state_head_offsets = head*(HEAD_DIM * HEAD_DIM) + (tl.arange(0, HEAD_DIM)[:, None])*HEAD_DIM + bv_offsets[None, :]
    S_in = tl.load(state_HKV + state_head_offsets) # [HEAD_DIM, BV]
    
    for i in range(NUM_CHUNKS):

        start = i*BT

        seq = start + tl.arange(0, BT)          # [BT]

        offsets = seq*NUM_HEADS + head
        mask = seq < seq_len
        G = tl.load(g_H + offsets, mask = mask, other=0.0) # [BT]
        G = tl.cumsum(G, axis=0)  # [BT] in log space


        dim = tl.arange(0, HEAD_DIM)            # [128]     
        k_offsets = (
            seq[:, None] * NUM_HEADS * HEAD_DIM
            + head * HEAD_DIM
            + dim[None, :]
        )  # [BT, HEAD_DIM]
        v_offsets = (
            seq[:, None] * NUM_HEADS * HEAD_DIM
            + head * HEAD_DIM
            + bv_offsets[None, :]
        )  # [BT, BV]
        mask = seq[:, None] < seq_len 

        K = tl.load(k_HK + k_offsets, mask = mask, other = 0.0).to(tl.float32) # [BT, K]
        U = tl.load(u + v_offsets, mask = mask, other = 0.0).to(tl.float32) # [BT, BV]
        W = tl.load(w + k_offsets, mask = mask, other = 0.0).to(tl.float32) # [BT, K]

        V_p = U - tl.dot(W , S_in) #[N , BV]
        tl.store(v_p_ptr + v_offsets, V_p, mask) # [N, BV]

        state_chunk_out_offsets = i*(NUM_HEADS*HEAD_DIM*HEAD_DIM) + state_head_offsets
        tl.store(chunk_state_out_ptr + state_chunk_out_offsets, S_in)

        chunk_len = tl.minimum(BT, seq_len - start)
        # Triton 1D tensors do not support dynamic or scalar indexing (G[i]).
        last_G = tl.sum(tl.where(tl.arange(0, BT) == chunk_len - 1, G, 0.0))
        S_out = (
            tl.exp(last_G) * S_in + 
            tl.dot(tl.trans(K) , (V_p * tl.exp(last_G - G[:, None])))
        )  # [K, V]

        S_in = S_out
    
    tl.store(state_HKV + state_head_offsets, S_in)
        

@triton.jit
def _recurrent_sequence_kernel_3(
    q_HK,  # [seq_len, HV, K]   (q broadcast to HV heads)
    k_HK,  # [seq_len, HV, K]
    g_H,  # [seq_len, HV]
    state_HKV,  # [HV, K, V]  (k-first internal layout)
    scale,
    out, # [seq_len, HV, K], 
    seq_len,
    chunk_state_out_ptr,
    v_p_ptr,
    NUM_HEADS: tl.constexpr,
    BT: tl.constexpr, 
    HEAD_DIM: tl.constexpr,
    BV: tl.constexpr
):
    head = tl.program_id(axis=0)
    chunk_idx = tl.program_id(axis=1)

    state_head_offsets = head*(HEAD_DIM * HEAD_DIM) + (tl.arange(0, HEAD_DIM)[:, None])*HEAD_DIM + tl.arange(0, HEAD_DIM)[None, :]
    chunk_state_out_offsets = chunk_idx*(NUM_HEADS * HEAD_DIM * HEAD_DIM) + state_head_offsets
    chunk_state_out = tl.load(chunk_state_out_ptr + chunk_state_out_offsets)# [HEAD_DIM, HEAD_DIM]

    start = chunk_idx*BT

    seq = start + tl.arange(0, BT)          # [BT]

    offsets = seq*NUM_HEADS + head
    mask = seq < seq_len
    G = tl.load(g_H + offsets, mask = mask, other=0.0) # [BT]
    G = tl.cumsum(G, axis=0)  # [BT] in log space
    Gamma = tl.exp(G[:, None] - G[None, :]) # [N, N]


    dim = tl.arange(0, HEAD_DIM)            # [128]     
    offsets = (
        seq[:, None] * NUM_HEADS * HEAD_DIM
        + head * HEAD_DIM
        + dim[None, :]
    )  # [BT, HEAD_DIM]
    mask = seq[:, None] < seq_len 

    K = tl.load(k_HK + offsets, mask = mask, other = 0.0).to(tl.float32) # [BT, K]
    Q = tl.load(q_HK + offsets, mask = mask, other = 0.0).to(tl.float32) # [BT, K]

    row = tl.arange(0, BT)[:, None]  # [BT, 1]
    col = tl.arange(0, BT)[None, :]  # [1, BT]
    M = tl.where(row >= col, 1.0, 0.0)      # [CHUNK_LEN, CHUNK_LEN]        
    M_p = M * Gamma # [N, N]

    V_p = tl.load(v_p_ptr + offsets, mask)
    O = tl.exp(G[:, None]) * tl.dot(Q, chunk_state_out) + tl.dot(
        (tl.dot(Q, tl.trans(K)) * M_p), V_p
    ) # [N, V]

    scaled_O = (scale * O).to(tl.bfloat16)
    
    tl.store(out + offsets, scaled_O, mask)



# === [Triton replacement point -- Stages 2-4: per-sequence delta rule] =======
def _recurrent_sequence(
    q_HK: torch.Tensor,  # [seq_len, HV, K]   (q broadcast to HV heads)
    k_HK: torch.Tensor,  # [seq_len, HV, K]
    v_HV: torch.Tensor,  # [seq_len, HV, V]
    g_H: torch.Tensor,  # [seq_len, HV]
    beta_H: torch.Tensor,  # [seq_len, HV]
    state_HKV: torch.Tensor,  # [HV, K, V]  (k-first internal layout)
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One sequence of the gated delta rule. Returns (output[seq_len,HV,V], state)."""
    seq_len, num_heads, head_dim = v_HV.shape
    out = q_HK.new_zeros(seq_len, num_heads, head_dim)
    BT: int = 16
    BV: int = 32
    assert head_dim % BV == 0
    num_chunks = (seq_len + BT - 1)//BT
    num_v_blocks = head_dim//BV


    w = torch.empty(seq_len, num_heads, head_dim, device = k_HK.device, dtype = torch.float32)
    u = torch.empty(seq_len, num_heads, head_dim, device = v_HV.device, dtype = torch.float32)

    chunk_state_out_ptr = torch.zeros(num_chunks, num_heads, head_dim, head_dim, device = k_HK.device, dtype = torch.float32)
    v_p_ptr = torch.empty(seq_len, num_heads, head_dim, device = k_HK.device, dtype = torch.float32)

    grid = lambda meta: (meta['NUM_HEADS'], num_chunks)

    _recurrent_sequence_kernel_1[grid]( k_HK, v_HV, 
    g_H, beta_H, seq_len, u, w, NUM_HEADS = num_heads, BT = BT, HEAD_DIM = head_dim)



    grid = lambda meta: (meta['NUM_HEADS'], num_v_blocks)

    _recurrent_sequence_kernel_2[grid](q_HK, k_HK, 
    g_H, state_HKV, scale, out, seq_len, u, w, chunk_state_out_ptr, v_p_ptr, NUM_CHUNKS = num_chunks, NUM_HEADS = num_heads, BT = BT, HEAD_DIM = head_dim, BV = BV)

    grid = lambda meta: (meta['NUM_HEADS'], num_chunks)
    
    _recurrent_sequence_kernel_3[grid](q_HK, k_HK, 
    g_H, state_HKV, scale, out, seq_len, chunk_state_out_ptr, v_p_ptr, NUM_HEADS = num_heads, BT = BT, HEAD_DIM = head_dim, BV = BV)



    return out, state_HKV

@torch.no_grad()
def run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    """Gated DeltaNet prefill (k-last state layout [H, V, K]).

    Args / shapes match the contest definition gdn_prefill_qk4_v8_d128_k_last:
      q: bf16 [T, Hq=4, K=128]   k: bf16 [T, Hk=4, K=128]   v: bf16 [T, Hv=8, V=128]
      state: f32 [num_seqs, Hv=8, V=128, K=128] (optional)
      A_log: f32 [Hv]   dt_bias: f32 [Hv]   a: bf16 [T, Hv]   b: bf16 [T, Hv]
      cu_seqlens: int64 [num_seqs+1]   scale: f32 (scalar, optional)
    Returns (output: bf16 [T, Hv, V], new_state: f32 [num_seqs, Hv, V, K]).
    """
    total_seq_len, num_q_heads, head_size = q.shape
    num_v_heads = v.shape[1]
    num_k_heads = k.shape[1]
    num_sab_heads = max(num_q_heads, num_v_heads)
    num_seqs = cu_seqlens.size(0) - 1
    device = q.device

    assert num_q_heads == 4
    assert num_k_heads == 4
    assert num_v_heads == 8
    assert head_size == 128

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(head_size)

    g, beta = _compute_gate_and_beta(A_log, a, dt_bias, b)

    # GQA: broadcast q/k heads up to the number of v heads.
    q_exp = q.repeat_interleave(num_v_heads // num_q_heads, dim=1)  # [T, HV, K]
    k_exp = k.repeat_interleave(num_v_heads // num_k_heads, dim=1)  # [T, HV, K]

    output = torch.zeros(
        (total_seq_len, num_sab_heads, head_size), dtype=torch.bfloat16, device=device
    )
    new_state = torch.zeros(
        (num_seqs, num_sab_heads, head_size, head_size),
        dtype=torch.float32,
        device=device,
    )

    for seq_idx in range(num_seqs):
        seq_start = int(cu_seqlens[seq_idx].item())
        seq_end = int(cu_seqlens[seq_idx + 1].item())
        seq_len = seq_end - seq_start
        if seq_len <= 0:
            continue

        if state is not None:
            # [H, V, K] (k-last, as stored) -> [H, K, V] internal (k-first).
            state_HKV = state[seq_idx].clone().float().transpose(-1, -2).contiguous()
        else:
            state_HKV = torch.zeros(
                (num_sab_heads, head_size, head_size),
                dtype=torch.float32,
                device=device,
            )

        out_seq, state_HKV = _recurrent_sequence(
            q_exp[seq_start:seq_end],
            k_exp[seq_start:seq_end],
            v[seq_start:seq_end],
            g[seq_start:seq_end],
            beta[seq_start:seq_end],
            state_HKV,
            scale,
        )
        output[seq_start:seq_end] = out_seq
        new_state[seq_idx] = state_HKV.transpose(-1, -2)  # [H,K,V] -> [H,V,K]

    return output, new_state
