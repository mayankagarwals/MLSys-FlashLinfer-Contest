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

_FLAG = None


@triton.jit
def _prep_meta_kernel(
    cu_seqlens_ptr,
    num_chunks_ptr,
    chunk_offsets_ptr,
    chunk_indices_ptr,
    flag_ptr,
    N,
    BT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    FILL_BLOCK: tl.constexpr = 128,
):
    pid = tl.program_id(0)

    if pid == 0:
        offsets = tl.arange(0, BLOCK_SIZE)
        bos = tl.load(cu_seqlens_ptr + offsets, offsets < N + 1)
        eos = tl.load(cu_seqlens_ptr + (offsets + 1), offsets < N)
        seqlens = eos - bos

        num_chunks = tl.cdiv(seqlens, BT)
        chunk_offsets = tl.cumsum(num_chunks, axis=0)

        tl.store(num_chunks_ptr + offsets, num_chunks, offsets < N)
        tl.store(chunk_offsets_ptr + (offsets + 1), chunk_offsets, offsets < N)
        tl.store(chunk_offsets_ptr, 0)

        tl.atomic_add(flag_ptr, 1, sem="release", scope="gpu")
    else:
        while tl.atomic_add(flag_ptr, 0, sem="acquire", scope="gpu") == 0:
            pass

    seq_id = pid
    num_chunk = tl.load(num_chunks_ptr + seq_id)
    chunk_offset = tl.load(chunk_offsets_ptr + seq_id)

    for i in range(tl.cdiv(num_chunk, FILL_BLOCK)):
        x = i * FILL_BLOCK + tl.arange(0, FILL_BLOCK)
        data = tl.join(seq_id, x)

        offs = (chunk_offset + x[:, None]) * 2 + tl.arange(0, 2)
        tl.store(chunk_indices_ptr + offs, data, mask=x[:, None] < num_chunk)

    before = tl.atomic_add(flag_ptr, 1)
    if before == tl.num_programs(0):
        tl.store(flag_ptr, 0)


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
    g_cu,  # [seq_len, HV]
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

    qk_head = head // (NUM_HEADS // NUM_QK_HEADS)
    k_offsets = (
        seq[:, None] * NUM_QK_HEADS * HEAD_DIM
        + qk_head * HEAD_DIM
        + dim[None, :]
    )
    matrix_mask = seq[:, None] < seq_len

    V = tl.load(v_HV + v_offsets, mask=matrix_mask, other=0.0).to(tl.float32)
    K = tl.load(k_HK + k_offsets, mask=matrix_mask, other=0.0).to(tl.float32)

    row = tl.arange(0, BT)[:, None]
    col = tl.arange(0, BT)[None, :]
    lower = row >= col

    Gamma = tl.exp(tl.where(lower, G[:, None] - G[None, :], -float("inf")))
    C = tl.dot(K, tl.trans(K))
    C = Gamma * C
    C = B[:, None] * C

    lower_tri_mask = row > col
    C = tl.where(lower_tri_mask, C, 0.0)

    L = C
    T = _unit_lower_inverse(L, BT=BT)
    T = T * B[None, :]

    U = tl.dot(T, V)
    W = tl.dot((T * tl.exp(G[None, :])), K)

    tl.store(u + v_offsets, U, matrix_mask)
    tl.store(w + v_offsets, W, matrix_mask)


@triton.jit
def _recurrent_sequence_kernel_2(
    k_HK,  # [seq_len, Hqk, K]
    g_cu,  # [seq_len, HV]
    state_in_HVK,  # [HV, V, K]
    state_out_HVK,  # [HV, V, K]
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
    BV: tl.constexpr,
):
    head = tl.program_id(axis=0)
    v_offset_id = tl.program_id(axis=1)

    bv_offsets = v_offset_id * BV + tl.arange(0, BV)
    state_hkv_offsets = (
        head * (HEAD_DIM * HEAD_DIM)
        + (tl.arange(0, HEAD_DIM)[:, None]) * HEAD_DIM
        + bv_offsets[None, :]
    )
    state_hvk_offsets = (
        head * (HEAD_DIM * HEAD_DIM)
        + bv_offsets[None, :] * HEAD_DIM
        + tl.arange(0, HEAD_DIM)[:, None]
    )
    S_in = tl.load(state_in_HVK + state_hkv_offsets)

    for i in range(NUM_CHUNKS):
        start = i * BT
        seq = start + tl.arange(0, BT)

        offsets = seq * NUM_HEADS + head
        mask = seq < seq_len
        G = tl.load(g_cu + offsets, mask=mask, other=0.0)

        dim = tl.arange(0, HEAD_DIM)
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
        )
        mask = seq[:, None] < seq_len

        K = tl.load(k_HK + k_offsets, mask=mask, other=0.0).to(tl.float32)
        U = tl.load(u + v_offsets, mask=mask, other=0.0).to(tl.float32)
        W = tl.load(w + w_offsets, mask=mask, other=0.0).to(tl.float32)

        V_p = U - tl.dot(W, S_in)
        tl.store(v_p_ptr + v_offsets, V_p, mask)

        state_chunk_out_offsets = (
            i * (NUM_HEADS * HEAD_DIM * HEAD_DIM) + state_hkv_offsets
        )
        tl.store(chunk_state_out_ptr + state_chunk_out_offsets, S_in)

        chunk_len = tl.minimum(BT, seq_len - start)
        last_G = tl.sum(tl.where(tl.arange(0, BT) == chunk_len - 1, G, 0.0))
        S_out = tl.exp(last_G) * S_in + tl.dot(
            tl.trans(K), (V_p * tl.exp(last_G - G[:, None]))
        )

        S_in = S_out

    tl.store(state_out_HVK + state_hvk_offsets, S_in)


@triton.jit
def _recurrent_sequence_kernel_3(
    q_HK,  # [seq_len, Hqk, K]
    k_HK,  # [seq_len, Hqk, K]
    g_cu,  # [seq_len, HV]
    scale,
    out,  # [seq_len, HV, V]
    seq_len,
    chunk_state_out_ptr,
    v_p_ptr,
    NUM_HEADS: tl.constexpr,
    NUM_QK_HEADS: tl.constexpr,
    BT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BV: tl.constexpr,
):
    head = tl.program_id(axis=0)
    chunk_idx = tl.program_id(axis=1)
    v_offset_id = tl.program_id(axis=2)

    bv_offsets = v_offset_id * BV + tl.arange(0, BV)
    state_head_offsets = (
        head * (HEAD_DIM * HEAD_DIM)
        + (tl.arange(0, HEAD_DIM)[:, None]) * HEAD_DIM
        + bv_offsets[None, :]
    )
    chunk_state_out_offsets = (
        chunk_idx * (NUM_HEADS * HEAD_DIM * HEAD_DIM) + state_head_offsets
    )
    chunk_state_out = tl.load(chunk_state_out_ptr + chunk_state_out_offsets)

    start = chunk_idx * BT
    seq = start + tl.arange(0, BT)

    offsets = seq * NUM_HEADS + head
    mask = seq < seq_len
    G = tl.load(g_cu + offsets, mask=mask, other=0.0)

    dim = tl.arange(0, HEAD_DIM)
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
    )
    mask = seq[:, None] < seq_len

    K = tl.load(k_HK + qk_offsets, mask=mask, other=0.0).to(tl.float32)
    Q = tl.load(q_HK + qk_offsets, mask=mask, other=0.0).to(tl.float32)

    row = tl.arange(0, BT)[:, None]
    col = tl.arange(0, BT)[None, :]
    lower = row >= col

    M = tl.where(row >= col, 1.0, 0.0)
    Gamma = tl.exp(tl.where(lower, G[:, None] - G[None, :], -float("inf")))
    M_p = M * Gamma

    V_p = tl.load(v_p_ptr + v_offsets, mask)
    O = tl.exp(G[:, None]) * tl.dot(Q, chunk_state_out) + tl.dot(
        (tl.dot(Q, tl.trans(K)) * M_p), V_p
    )

    scaled_O = (scale * O).to(tl.bfloat16)
    tl.store(out + v_offsets, scaled_O, mask)


@triton.jit
def _batched_recurrent_sequence_kernel_1(
    k_HK,  # [T, Hqk, K]
    v_HV,  # [T, HV, V]
    A_log,  # [HV]
    a,  # [T, HV]
    dt_bias,  # [HV]
    b,  # [T, HV]
    g_cu,  # [T, HV]
    u,
    w,
    cu_seqlens,
    chunk_indices,
    NUM_HEADS: tl.constexpr,
    NUM_QK_HEADS: tl.constexpr,
    BT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    head = tl.program_id(axis=0)
    global_chunk_id = tl.program_id(axis=1)

    seq_id = tl.load(chunk_indices + global_chunk_id * 2).to(tl.int32)
    local_chunk_idx = tl.load(chunk_indices + global_chunk_id * 2 + 1).to(tl.int32)
    bos = tl.load(cu_seqlens + seq_id).to(tl.int32)
    eos = tl.load(cu_seqlens + seq_id + 1).to(tl.int32)
    start = bos + local_chunk_idx * BT
    seq = start + tl.arange(0, BT)

    token_head_offsets = seq * NUM_HEADS + head
    token_mask = seq < eos

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
    matrix_mask = token_mask[:, None]

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
def _batched_recurrent_sequence_kernel_2(
    k_HK,  # [T, Hqk, K]
    g_cu,  # [T, HV]
    state_in_HVK,  # [N, HV, V, K]
    state_out_HVK,  # [N, HV, V, K]
    u,
    w,
    chunk_state_out_ptr,
    v_p_ptr,
    cu_seqlens,
    chunk_offsets,
    NUM_HEADS: tl.constexpr,
    NUM_QK_HEADS: tl.constexpr,
    BT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BV: tl.constexpr,
):
    head = tl.program_id(axis=0)
    v_offset_id = tl.program_id(axis=1)
    seq_id = tl.program_id(axis=2)
    bos = tl.load(cu_seqlens + seq_id).to(tl.int32)
    eos = tl.load(cu_seqlens + seq_id + 1).to(tl.int32)
    chunk_offset = tl.load(chunk_offsets + seq_id).to(tl.int32)
    num_chunks = (
        tl.load(chunk_offsets + seq_id + 1).to(tl.int32) - chunk_offset
    )

    # assert in main code that BV divides HEAD_DIM so we don't need to worry about masking
    bv_offsets =  v_offset_id*BV + tl.arange(0, BV)
    state_seq_offset = seq_id * NUM_HEADS * HEAD_DIM * HEAD_DIM
    state_hkv_offsets = state_seq_offset + head*(HEAD_DIM * HEAD_DIM) + (tl.arange(0, HEAD_DIM)[:, None])*HEAD_DIM + bv_offsets[None, :]
    state_hvk_offsets = state_seq_offset + head*(HEAD_DIM * HEAD_DIM) + bv_offsets[None, :]*HEAD_DIM + tl.arange(0, HEAD_DIM)[:, None]
    S_in = tl.load(state_in_HVK + state_hvk_offsets) # [HEAD_DIM, BV]
    
    for i in range(num_chunks):

        start = bos + i*BT

        seq = start + tl.arange(0, BT)          # [BT]

        offsets = seq*NUM_HEADS + head
        mask = seq < eos
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
        mask = seq[:, None] < eos

        K = tl.load(k_HK + k_offsets, mask = mask, other = 0.0).to(tl.float32) # [BT, K]
        U = tl.load(u + v_offsets, mask = mask, other = 0.0).to(tl.float32) # [BT, BV]
        W = tl.load(w + w_offsets, mask=mask, other=0.0).to(tl.float32)

        V_p = U - tl.dot(W , S_in) #[N , BV]
        tl.store(v_p_ptr + v_offsets, V_p, mask) # [N, BV]

        chunk_state_head_offsets = head*(HEAD_DIM * HEAD_DIM) + (tl.arange(0, HEAD_DIM)[:, None])*HEAD_DIM + bv_offsets[None, :]
        state_chunk_out_offsets = (chunk_offset + i)*(NUM_HEADS*HEAD_DIM*HEAD_DIM) + chunk_state_head_offsets
        tl.store(chunk_state_out_ptr + state_chunk_out_offsets, S_in)

        chunk_len = tl.minimum(BT, eos - start)
        # Triton 1D tensors do not support dynamic or scalar indexing (G[i]).
        last_G = tl.sum(tl.where(tl.arange(0, BT) == chunk_len - 1, G, 0.0))
        S_out = (
            tl.exp(last_G) * S_in + 
            tl.dot(tl.trans(K) , (V_p * tl.exp(last_G - G[:, None])))
        )  # [K, V]

        S_in = S_out
    
    tl.store(state_out_HVK + state_hvk_offsets, S_in)
        

@triton.jit
def _batched_recurrent_sequence_kernel_3(
    q_HK,  # [T, Hqk, K]
    k_HK,  # [T, Hqk, K]
    g_cu,  # [T, HV]
    scale,
    out,  # [T, HV, V]
    chunk_state_out_ptr,
    v_p_ptr,
    cu_seqlens,
    chunk_indices,
    NUM_HEADS: tl.constexpr,
    NUM_QK_HEADS: tl.constexpr,
    BT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BV: tl.constexpr,
):
    head = tl.program_id(axis=0)
    global_chunk_id = tl.program_id(axis=1)
    v_offset_id = tl.program_id(axis=2)

    seq_id = tl.load(chunk_indices + global_chunk_id * 2).to(tl.int32)
    local_chunk_idx = tl.load(chunk_indices + global_chunk_id * 2 + 1).to(tl.int32)
    bos = tl.load(cu_seqlens + seq_id).to(tl.int32)
    eos = tl.load(cu_seqlens + seq_id + 1).to(tl.int32)
    start = bos + local_chunk_idx * BT

    bv_offsets = v_offset_id * BV + tl.arange(0, BV)
    state_head_offsets = (
        head * (HEAD_DIM * HEAD_DIM)
        + (tl.arange(0, HEAD_DIM)[:, None]) * HEAD_DIM
        + bv_offsets[None, :]
    )
    chunk_state_out_offsets = (
        global_chunk_id * (NUM_HEADS * HEAD_DIM * HEAD_DIM) + state_head_offsets
    )
    chunk_state_out = tl.load(chunk_state_out_ptr + chunk_state_out_offsets)

    seq = start + tl.arange(0, BT)
    offsets = seq * NUM_HEADS + head
    mask = seq < eos
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
    mask = seq[:, None] < eos

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


def _recurrent_sequence(
    q_HK: torch.Tensor,
    k_HK: torch.Tensor,
    v_HV: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    state_HVK: torch.Tensor,
    state_out_HVK: torch.Tensor,
    scale: float,
    out: torch.Tensor,
) -> None:
    """Run one sequence into caller-owned output and state storage."""
    seq_len, num_heads, head_dim = v_HV.shape
    num_qk_heads = k_HK.shape[1]
    BT = 64
    BV = 32
    num_chunks = (seq_len + BT - 1) // BT
    num_v_blocks = head_dim // BV

    w = torch.empty(
        seq_len, num_heads, head_dim, device=k_HK.device, dtype=torch.float32
    )
    u = torch.empty(
        seq_len, num_heads, head_dim, device=v_HV.device, dtype=torch.float32
    )
    g_cu = torch.empty(seq_len, num_heads, device=v_HV.device, dtype=torch.float32)
    chunk_state_out_ptr = torch.empty(
        num_chunks,
        num_heads,
        head_dim,
        head_dim,
        device=k_HK.device,
        dtype=torch.float32,
    )
    v_p_ptr = torch.empty(
        seq_len, num_heads, head_dim, device=k_HK.device, dtype=torch.float32
    )

    k_meta = dict(
        NUM_HEADS=num_heads,
        NUM_QK_HEADS=num_qk_heads,
        BT=BT,
        HEAD_DIM=head_dim,
    )

    _recurrent_sequence_kernel_1[(num_heads, num_chunks)](
        k_HK,
        v_HV,
        A_log,
        a,
        dt_bias,
        b,
        g_cu,
        seq_len,
        u,
        w,
        **k_meta,
    )
    _recurrent_sequence_kernel_2[(num_heads, num_v_blocks)](
        k_HK,
        g_cu,
        state_HVK,
        state_out_HVK,
        seq_len,
        u,
        w,
        chunk_state_out_ptr,
        v_p_ptr,
        NUM_CHUNKS=num_chunks,
        BV=BV,
        **k_meta,
    )
    _recurrent_sequence_kernel_3[(num_heads, num_chunks, num_v_blocks)](
        q_HK,
        k_HK,
        g_cu,
        scale,
        out,
        seq_len,
        chunk_state_out_ptr,
        v_p_ptr,
        BV=BV,
        **k_meta,
    )



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

    # The common single-sequence case spans the complete packed input. Avoid
    # synchronizing cu_seqlens to the CPU and return the kernel outputs directly.
    if num_seqs == 1:
        if state is not None:
            state_HVK = state[0]
        else:
            state_HVK = torch.zeros(
                (num_sab_heads, head_size, head_size),
                dtype=torch.float32,
                device=device,
            )

        output = torch.empty(
            (total_seq_len, num_sab_heads, head_size),
            dtype=torch.bfloat16,
            device=device,
        )
        state_out_HVK = torch.empty_like(state_HVK)
        _recurrent_sequence(
            q,
            k,
            v,
            A_log,
            a,
            dt_bias,
            b,
            state_HVK,
            state_out_HVK,
            scale,
            output,
        )
        return output, state_out_HVK.unsqueeze(0)

    BT = 64
    BV = 32
    num_v_blocks = head_size // BV

    # needed because if we try calculating exact chunk count per seq we will need to sync the 
    # gpu tensor which we want to avoid 
    upper_bound_chunks = (num_seqs - 1) + triton.cdiv(
        total_seq_len - (num_seqs - 1), BT
    )

    global _FLAG
    if _FLAG is None:
        _FLAG = q.new_zeros(1, dtype=torch.int32)

    num_chunks = q.new_empty(num_seqs, dtype=torch.int32)
    chunk_offsets = q.new_empty(num_seqs + 1, dtype=torch.int32)
    chunk_indices = q.new_empty((upper_bound_chunks, 2), dtype=torch.int32)
    _prep_meta_kernel[(num_seqs,)](
        cu_seqlens,
        num_chunks,
        chunk_offsets,
        chunk_indices,
        _FLAG,
        N=num_seqs,
        BT=BT,
        BLOCK_SIZE=triton.next_power_of_2(num_seqs),
    )
    total_num_chunks = int(chunk_offsets[-1].item())

    if state is None:
        state = torch.zeros(
            (num_seqs, num_sab_heads, head_size, head_size),
            dtype=torch.float32,
            device=device,
        )

    output = torch.empty_like(v)
    new_state = torch.empty_like(state)
    u = torch.empty(
        total_seq_len, num_v_heads, head_size, device=device, dtype=torch.float32
    )
    w = torch.empty_like(u)
    g_cu = torch.empty(total_seq_len, num_v_heads, device=device, dtype=torch.float32)
    chunk_state_out = torch.empty(
        total_num_chunks,
        num_v_heads,
        head_size,
        head_size,
        device=device,
        dtype=torch.float32,
    )
    v_p = torch.empty_like(u)

    k1_meta = dict(
        NUM_HEADS=num_v_heads,
        NUM_QK_HEADS=num_k_heads,
        BT=BT,
        HEAD_DIM=head_size,
    )
    k23_meta = dict(
        NUM_HEADS=num_v_heads,
        NUM_QK_HEADS=num_k_heads,
        BT=BT,
        HEAD_DIM=head_size,
        BV=BV,
    )

    _batched_recurrent_sequence_kernel_1[(num_v_heads, total_num_chunks)](
        k, v, A_log, a, dt_bias, b, g_cu, u, w, cu_seqlens, chunk_indices, **k1_meta
    )
    _batched_recurrent_sequence_kernel_2[(num_v_heads, num_v_blocks, num_seqs)](
        k,
        g_cu,
        state,
        new_state,
        u,
        w,
        chunk_state_out,
        v_p,
        cu_seqlens,
        chunk_offsets,
        **k23_meta,
    )
    _batched_recurrent_sequence_kernel_3[(num_v_heads, total_num_chunks, num_v_blocks)](
        q,
        k,
        g_cu,
        scale,
        output,
        chunk_state_out,
        v_p,
        cu_seqlens,
        chunk_indices,
        **k23_meta,
    )
    return output, new_state