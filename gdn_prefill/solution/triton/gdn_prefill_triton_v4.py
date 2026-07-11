"""
V2: Triton 
"""

import math

import torch
import triton
import triton.language as tl


def _alloc_fn(size: int, alignment: int, stream: int | None):
    return torch.empty(size, device="cuda", dtype=torch.int8)


triton.set_allocator(_alloc_fn)


def _matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Float32 matmul for numerical stability (matches the contest reference)."""
    return a.float() @ b.float()


@triton.jit
def _shift_block_rows(X, distance: tl.constexpr):
    """Return Y[i] = X[i - distance], with out-of-range rows set to zero."""
    dst = tl.arange(0, 4)[:, None, None, None]
    src = tl.arange(0, 4)[None, :, None, None]
    return tl.sum(tl.where(src + distance == dst, X[None, :, :, :], 0.0), axis=1)


@triton.jit
def _block_subdiagonal(X, distance: tl.constexpr):
    """Extract X[i, i - distance], keeping i as a four-element batch axis."""
    row = tl.arange(0, 4)[:, None, None, None]
    col = tl.arange(0, 4)[None, :, None, None]
    return tl.sum(tl.where(row == col + distance, X, 0.0), axis=1)


@triton.jit
def _unit_lower_inverse_16(A):
    """Invert four batched 16x16 unit-lower-triangular blocks."""
    idx = tl.arange(0, 16)
    I = tl.where(idx[:, None] == idx[None, :], 1.0, 0.0)
    I = tl.broadcast_to(I[None, :, :], (4, 16, 16))

    # Exact in real arithmetic because every strictly lower 16x16 A has A^16 = 0.
    power = A
    Ai = I - power
    for _ in tl.static_range(3):
        power = tl.dot(power, power)
        Ai = tl.dot(Ai, I + power)

    # Repair tensor-core rounding. Each step squares the inverse residual.
    for _ in tl.static_range(3):
        MAi = Ai + tl.dot(A, Ai, input_precision="tf32x3")
        Ai = tl.dot(Ai, 2.0 * I - MAi, input_precision="tf32x3")
    return Ai


@triton.jit
def _unit_lower_inverse(A, BT: tl.constexpr):
    """Invert a 64x64 unit-lower-triangular matrix as 4x4 blocks of 16x16."""
    tl.static_assert(BT == 64)

    # [64,64] -> [block_row, block_col, row_in_block, col_in_block]
    A_blocks = tl.reshape(A, (4, 16, 4, 16))
    A_blocks = tl.permute(A_blocks, (0, 2, 1, 3))

    A0 = _block_subdiagonal(A_blocks, distance=0)
    A1 = _block_subdiagonal(A_blocks, distance=1)
    A2 = _block_subdiagonal(A_blocks, distance=2)
    A3 = _block_subdiagonal(A_blocks, distance=3)

    # Diagonal of B = (I + A)^-1.
    B0 = _unit_lower_inverse_16(A0)

    # First, second, and third inverse subdiagonals, batched by block row i.
    B1 = -tl.dot(B0, tl.dot(A1, _shift_block_rows(B0, distance=1)))
    B2 = -tl.dot(
        B0,
        tl.dot(A2, _shift_block_rows(B0, distance=2))
        + tl.dot(A1, _shift_block_rows(B1, distance=1)),
    )
    B3 = -tl.dot(
        B0,
        tl.dot(A3, _shift_block_rows(B0, distance=3))
        + tl.dot(A2, _shift_block_rows(B1, distance=2))
        + tl.dot(A1, _shift_block_rows(B2, distance=1)),
    )

    # Place the four computed subdiagonals back into a 4x4 block matrix.
    row = tl.arange(0, 4)[:, None, None, None]
    col = tl.arange(0, 4)[None, :, None, None]
    B_blocks = tl.where(row == col, B0[:, None, :, :], 0.0)
    B_blocks += tl.where(row == col + 1, B1[:, None, :, :], 0.0)
    B_blocks += tl.where(row == col + 2, B2[:, None, :, :], 0.0)
    B_blocks += tl.where(row == col + 3, B3[:, None, :, :], 0.0)

    # [block_row, block_col, row, col] -> [64,64]
    B = tl.permute(B_blocks, (0, 2, 1, 3))
    return tl.reshape(B, (BT, BT))



@triton.jit
def _recurrent_sequence_kernel_1(
    k_HK,  # [seq_len, Hqk, K]
    v_HV,  # [seq_len, HV, V]
    A_log,  # [HV]
    a,  # [seq_len, HV]
    dt_bias,  # [HV]
    b,  # [seq_len, HV]
    g_cu,  # output: chunk-local cumsum(g), [seq_len, HV]
    seq_len,
    u, 
    w,
    NUM_HEADS: tl.constexpr,
    NUM_QK_HEADS: tl.constexpr,
    BT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    head = tl.program_id(axis=0)
    chunk_idx = tl.program_id(axis=1)

    start = chunk_idx * BT

    seq = start + tl.arange(0, BT)          # [BT]

    token_head_offsets = seq * NUM_HEADS + head
    token_mask = seq < seq_len

    # Fuse the former PyTorch gate/beta precompute into the KKT kernel.
    x = tl.load(a + token_head_offsets, mask=token_mask, other=0.0).to(tl.float32)
    x += tl.load(dt_bias + head).to(tl.float32)
    softplus_x = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(x)))
    g = -tl.exp(tl.load(A_log + head).to(tl.float32)) * softplus_x
    g = tl.where(token_mask, g, 0.0)
    G = tl.cumsum(g, axis=0)
    tl.store(g_cu + token_head_offsets, G, mask=token_mask)

    b_value = tl.load(b + token_head_offsets, mask=token_mask, other=0.0).to(tl.float32)
    B = 1.0 / (1.0 + tl.exp(-b_value))

    dim = tl.arange(0, HEAD_DIM)
    v_offsets = (
        seq[:, None] * NUM_HEADS * HEAD_DIM
        + head * HEAD_DIM
        + dim[None, :]
    )

    # Two value heads share one q/k head: 0,1 -> 0; 2,3 -> 1; ...
    qk_head = head // (NUM_HEADS // NUM_QK_HEADS)
    k_offsets = (
        seq[:, None] * NUM_QK_HEADS * HEAD_DIM
        + qk_head * HEAD_DIM
        + dim[None, :]
    )
    matrix_mask = seq[:, None] < seq_len

    V = tl.load(v_HV + v_offsets, mask=matrix_mask, other=0.0).to(tl.float32)
    K = tl.load(k_HK + k_offsets, mask=matrix_mask, other=0.0).to(tl.float32)
    
    row = tl.arange(0, BT)[:, None]  # [BT, 1]
    col = tl.arange(0, BT)[None, :]  # [1, BT]
    lower = row >= col

    Gamma = tl.exp(tl.where(lower, G[:, None] - G[None, :], -float("inf"))) # [N, N]
    C = tl.dot(K, tl.trans(K)) # [N, N]
    C = Gamma * C # [N, N] (pointwise mul)
    C = B[:, None] * C # [N, N] (broadcast)


    lower_tri_mask = row > col 
    C = tl.where(lower_tri_mask, C, 0.0)

    # L is strictly lower; T = (I + L)^{-1} diag(B)
    L = C
    T = _unit_lower_inverse(L, BT=BT)
    T = T * B[None, :]

    U = tl.dot(T , V)  # [N, V]
    W = tl.dot((T * tl.exp(G[None, :])) , K) # [N, K]

    tl.store(u + v_offsets, U, matrix_mask)
    tl.store(w + v_offsets, W, matrix_mask)




@triton.jit
def _recurrent_sequence_kernel_2(
    k_HK,  # [seq_len, Hqk, K]
    g_cu,  # chunk-local cumsum(g), [seq_len, HV]
    state_HKV,  # [HV, K, V]  (k-first internal layout)
    seq_len,
    u,
    w,
    chunk_state_out_ptr,
    v_p_ptr,
    NUM_CHUNKS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_QK_HEADS: tl.constexpr,
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
        G = tl.load(g_cu + offsets, mask=mask, other=0.0)


        dim = tl.arange(0, HEAD_DIM)            # [128]     
        qk_head = head // (NUM_HEADS // NUM_QK_HEADS)
        k_offsets = (
            seq[:, None] * NUM_QK_HEADS * HEAD_DIM
            + qk_head * HEAD_DIM
            + dim[None, :]
        )
        w_offsets = (
            seq[:, None] * NUM_HEADS * HEAD_DIM
            + head * HEAD_DIM
            + dim[None, :]
        )
        v_offsets = (
            seq[:, None] * NUM_HEADS * HEAD_DIM
            + head * HEAD_DIM
            + bv_offsets[None, :]
        )  # [BT, BV]
        mask = seq[:, None] < seq_len 

        K = tl.load(k_HK + k_offsets, mask = mask, other = 0.0).to(tl.float32) # [BT, K]
        U = tl.load(u + v_offsets, mask = mask, other = 0.0).to(tl.float32) # [BT, BV]
        W = tl.load(w + w_offsets, mask=mask, other=0.0).to(tl.float32)

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
    q_HK,  # [seq_len, Hqk, K]
    k_HK,  # [seq_len, Hqk, K]
    g_cu,  # chunk-local cumsum(g), [seq_len, HV]
    state_HKV,  # [HV, K, V]  (k-first internal layout)
    scale,
    out, # [seq_len, HV, K], 
    seq_len,
    chunk_state_out_ptr,
    v_p_ptr,
    NUM_HEADS: tl.constexpr,
    NUM_QK_HEADS: tl.constexpr,
    BT: tl.constexpr, 
    HEAD_DIM: tl.constexpr,
    BV: tl.constexpr
):
    head = tl.program_id(axis=0)
    chunk_idx = tl.program_id(axis=1)
    v_offset_id = tl.program_id(axis=2)

    bv_offsets =  v_offset_id*BV + tl.arange(0, BV)
    state_head_offsets = head*(HEAD_DIM * HEAD_DIM) + (tl.arange(0, HEAD_DIM)[:, None])*HEAD_DIM + bv_offsets[None, :]
    chunk_state_out_offsets = chunk_idx*(NUM_HEADS * HEAD_DIM * HEAD_DIM) + state_head_offsets
    chunk_state_out = tl.load(chunk_state_out_ptr + chunk_state_out_offsets)# [HEAD_DIM, BV]

    start = chunk_idx*BT

    seq = start + tl.arange(0, BT)          # [BT]

    offsets = seq*NUM_HEADS + head
    mask = seq < seq_len
    G = tl.load(g_cu + offsets, mask=mask, other=0.0)


    dim = tl.arange(0, HEAD_DIM)            # [128]     
    qk_head = head // (NUM_HEADS // NUM_QK_HEADS)
    qk_offsets = (
        seq[:, None] * NUM_QK_HEADS * HEAD_DIM
        + qk_head * HEAD_DIM
        + dim[None, :]
    )
    v_offsets = (
        seq[:, None] * NUM_HEADS * HEAD_DIM
        + head * HEAD_DIM
        + bv_offsets[None, :]
    )  # [BT, BV]
    mask = seq[:, None] < seq_len 

    K = tl.load(k_HK + qk_offsets, mask=mask, other=0.0).to(tl.float32)
    Q = tl.load(q_HK + qk_offsets, mask=mask, other=0.0).to(tl.float32)

    row = tl.arange(0, BT)[:, None]  # [BT, 1]
    col = tl.arange(0, BT)[None, :]  # [1, BT]
    lower = row >= col

    M = tl.where(row >= col, 1.0, 0.0)      # [CHUNK_LEN, CHUNK_LEN]    

    # Do this masking to avoid exponentiating large values which we are going to find in upper diagonal
    Gamma = tl.exp(tl.where(lower, G[:, None] - G[None, :], -float("inf"))) # [N, N]
    
    M_p = M * Gamma # [N, N]

    V_p = tl.load(v_p_ptr + v_offsets, mask)
    O = tl.exp(G[:, None]) * tl.dot(Q, chunk_state_out) + tl.dot(
        (tl.dot(Q, tl.trans(K)) * M_p), V_p
    ) # [N, V]

    scaled_O = (scale * O).to(tl.bfloat16)
    
    tl.store(out + v_offsets, scaled_O, mask)



# === [Triton replacement point -- Stages 2-4: per-sequence delta rule] =======
def _recurrent_sequence(
    q_HK: torch.Tensor,  # [seq_len, Hqk, K]
    k_HK: torch.Tensor,  # [seq_len, Hqk, K]
    v_HV: torch.Tensor,  # [seq_len, HV, V]
    A_log: torch.Tensor,  # [HV]
    a: torch.Tensor,  # [seq_len, HV]
    dt_bias: torch.Tensor,  # [HV]
    b: torch.Tensor,  # [seq_len, HV]
    state_HKV: torch.Tensor,  # [HV, K, V]  (k-first internal layout)
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One sequence of the gated delta rule. Returns (output[seq_len,HV,V], state)."""
    seq_len, num_heads, head_dim = v_HV.shape
    num_qk_heads = k_HK.shape[1]
    assert q_HK.shape[1] == num_qk_heads
    assert num_heads % num_qk_heads == 0
    out = q_HK.new_zeros(seq_len, num_heads, head_dim)
    BT: int = 64
    BV: int = 32
    assert head_dim % BV == 0
    num_chunks = (seq_len + BT - 1)//BT
    num_v_blocks = head_dim//BV


    w = torch.empty(seq_len, num_heads, head_dim, device = k_HK.device, dtype = torch.float32)
    u = torch.empty(seq_len, num_heads, head_dim, device = v_HV.device, dtype = torch.float32)
    g_cu = torch.empty(seq_len, num_heads, device=v_HV.device, dtype=torch.float32)

    chunk_state_out_ptr = torch.zeros(num_chunks, num_heads, head_dim, head_dim, device = k_HK.device, dtype = torch.float32)
    v_p_ptr = torch.empty(seq_len, num_heads, head_dim, device = k_HK.device, dtype = torch.float32)

    grid = lambda meta: (meta['NUM_HEADS'], num_chunks)

    _recurrent_sequence_kernel_1[grid](
        k_HK, v_HV, A_log, a, dt_bias, b, g_cu, seq_len, u, w,
        NUM_HEADS=num_heads, NUM_QK_HEADS=num_qk_heads, BT=BT, HEAD_DIM=head_dim,
    )



    grid = lambda meta: (meta['NUM_HEADS'], num_v_blocks)

    _recurrent_sequence_kernel_2[grid](
        k_HK, g_cu, state_HKV, seq_len, u, w, chunk_state_out_ptr, v_p_ptr,
        NUM_CHUNKS=num_chunks, NUM_HEADS=num_heads, NUM_QK_HEADS=num_qk_heads,
        BT=BT, HEAD_DIM=head_dim, BV=BV,
    )

    grid = lambda meta: (meta['NUM_HEADS'], num_chunks, num_v_blocks)
    
    _recurrent_sequence_kernel_3[grid](
        q_HK, k_HK, g_cu, state_HKV, scale, out, seq_len,
        chunk_state_out_ptr, v_p_ptr, NUM_HEADS=num_heads,
        NUM_QK_HEADS=num_qk_heads, BT=BT, HEAD_DIM=head_dim, BV=BV,
    )



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
            q[seq_start:seq_end],
            k[seq_start:seq_end],
            v[seq_start:seq_end],
            A_log,
            a[seq_start:seq_end],
            dt_bias,
            b[seq_start:seq_end],
            state_HKV,
            scale,
        )
        output[seq_start:seq_end] = out_seq
        new_state[seq_idx] = state_HKV.transpose(-1, -2)  # [H,K,V] -> [H,V,K]

    return output, new_state