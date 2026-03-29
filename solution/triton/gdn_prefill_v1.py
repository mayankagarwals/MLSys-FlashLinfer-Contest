# https://github.com/vllm-project/vllm/blob/v0.17.0/vllm/model_executor/layers/fla/ops/chunk.py

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra import libdevice


def alloc_fn(size: int, alignment: int, stream: int | None):
    return torch.empty(size, device="cuda", dtype=torch.int8)


triton.set_allocator(alloc_fn)


@triton.autotune(
    configs=[triton.Config(dict(), num_warps=num_warps) for num_warps in [2, 4, 8]],
    key=["H", "K_dim", "BT"],
)
@triton.jit
def chunk_scaled_dot_kkt_fwd_kernel(
    k_ptr,  # [T, Hg, K_dim]
    A_log_ptr,  # [H]
    a_ptr,  # [T, H]
    dt_bias_ptr,  # [H]
    b_ptr,  # [T, H]
    g_cu_ptr,  # [T, H]
    beta_ptr,  # [T, H]
    A_ptr,
    cu_seqlens_ptr,
    chunk_indices_ptr,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K_dim: tl.constexpr,
    BT: tl.constexpr,
):
    global_chunk_id = tl.program_id(0)
    k_head_id = tl.program_id(1)

    seq_id = tl.load(chunk_indices_ptr + global_chunk_id * 2).to(tl.int32)
    chunk_id = tl.load(chunk_indices_ptr + global_chunk_id * 2 + 1).to(tl.int32)

    bos = tl.load(cu_seqlens_ptr + seq_id).to(tl.int32)
    eos = tl.load(cu_seqlens_ptr + seq_id + 1).to(tl.int32)
    seqlen = eos - bos

    # this creates tensormap on device (or does it?) -> might not be good
    k_desc = tl.make_tensor_descriptor(
        k_ptr + (bos * Hg * K_dim + k_head_id * K_dim),
        [seqlen, K_dim],
        [Hg * K_dim, 1],
        [BT, K_dim],
    )
    A_desc = tl.make_tensor_descriptor(
        A_ptr + bos * H * BT,
        [seqlen, H, BT],
        [H * BT, BT, 1],
        [BT, 1, BT],
    )

    k = k_desc.load([chunk_id * BT, 0])
    A = tl.dot(k, k.T)  # [BT, BT]

    # each K head corresponds to (H // Hg) V heads
    # NOTE: we can load all (H // Hg) at the same time?
    for i in range(H // Hg):
        head_id = k_head_id * (H // Hg) + i

        # issue all loads
        offs_t = bos + chunk_id * BT + tl.arange(0, BT)
        b = tl.load(b_ptr + (offs_t * H + head_id)).to(tl.float32)
        a = tl.load(a_ptr + (offs_t * H + head_id)).to(tl.float32)
        A_log = tl.load(A_log_ptr + head_id).to(tl.float32)
        dt_bias = tl.load(dt_bias_ptr + head_id).to(tl.float32)

        # compute g and beta
        beta = libdevice.rcp_rn(1.0 + tl.exp(-b))  # sigmoid
        g = -tl.exp(A_log) * tl.log(1.0 + tl.exp(a + dt_bias))
        g_cu = tl.cumsum(g, axis=0)

        # store for future use
        tl.store(beta_ptr + (offs_t * H + head_id), beta, mask=offs_t < bos + seqlen)
        tl.store(g_cu_ptr + (offs_t * H + head_id), g_cu, mask=offs_t < bos + seqlen)

        # apply beta and gamma
        A_ = A * beta[:, None]
        A_ = A_ * tl.exp(g_cu[:, None] - g_cu[None, :])

        offs_t = chunk_id * BT + tl.arange(0, BT)
        mask_t = offs_t < seqlen
        mask_A = (offs_t[:, None] > offs_t[None, :]) & (mask_t[:, None] & mask_t)
        A_ = tl.where(mask_A, A_, 0)

        A_desc.store([chunk_id * BT, head_id, 0], A_.reshape(BT, 1, BT))


# concat along the first dim
@triton.jit
def _concat_2d_dim0(A, B):
    return tl.join(A, B).permute(2, 0, 1).reshape(A.shape[0] * 2, A.shape[1])


# concat along the last dim
@triton.jit
def _concat_2d_dim1(A, B):
    return tl.join(A, B).permute(0, 2, 1).reshape(A.shape[0], A.shape[1] * 2)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4, 5]
    ],
    key=["H", "K_dim", "V_dim", "BT"],
)
@triton.jit
def merge_16x16_to_64x64_inverse_kernel(
    k_ptr,
    v_ptr,
    w_ptr,
    u_ptr,
    A_ptr,
    Ai_ptr,
    beta_ptr,
    g_cu_ptr,
    cu_seqlens_ptr,
    chunk_indices_ptr,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K_dim: tl.constexpr,
    V_dim: tl.constexpr,
    BT: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    global_chunk_id = tl.program_id(0)
    head_id = tl.program_id(1)

    seq_id = tl.load(chunk_indices_ptr + global_chunk_id * 2).to(tl.int32)
    chunk_id = tl.load(chunk_indices_ptr + global_chunk_id * 2 + 1).to(tl.int32)

    bos = tl.load(cu_seqlens_ptr + seq_id).to(tl.int32)
    eos = tl.load(cu_seqlens_ptr + seq_id + 1).to(tl.int32)
    seqlen = eos - bos

    # compute inverse
    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    desc = tl.make_tensor_descriptor(
        A_ptr + (bos * H + head_id) * BT, [seqlen, BT], [H * BT, 1], [16, 16]
    )
    desc_o = tl.make_tensor_descriptor(
        Ai_ptr + (bos * H + head_id) * BT, [seqlen, BT], [H * BT, 1], [16, 16]
    )
    Ai_11 = desc.load([chunk_id * BT + 0, 0]).to(tl.float32)
    Ai_22 = desc.load([chunk_id * BT + 16, 16]).to(tl.float32)
    Ai_33 = desc.load([chunk_id * BT + 32, 32]).to(tl.float32)
    Ai_44 = desc.load([chunk_id * BT + 48, 48]).to(tl.float32)

    # [16, 16]
    Ai_11 = -tl.where(m_A, Ai_11, 0)
    Ai_22 = -tl.where(m_A, Ai_22, 0)
    Ai_33 = -tl.where(m_A, Ai_33, 0)
    Ai_44 = -tl.where(m_A, Ai_44, 0)

    for i in range(2, min(16, seqlen - chunk_id * BT)):
        a_11 = -tl.load(A_ptr + (bos + chunk_id * BT + i) * H * BT + head_id * BT + o_i)
        a_11 += tl.sum(a_11[:, None] * Ai_11, 0)
        Ai_11 = tl.where((o_i == i)[:, None], a_11, Ai_11)
    for i in range(16 + 2, min(32, seqlen - chunk_id * BT)):
        a_22 = -tl.load(
            A_ptr + (bos + chunk_id * BT + i) * H * BT + head_id * BT + o_i + 16
        )
        a_22 += tl.sum(a_22[:, None] * Ai_22, 0)
        Ai_22 = tl.where((o_i == i - 16)[:, None], a_22, Ai_22)
    for i in range(32 + 2, min(48, seqlen - chunk_id * BT)):
        a_33 = -tl.load(
            A_ptr + (bos + chunk_id * BT + i) * H * BT + head_id * BT + o_i + 32
        )
        a_33 += tl.sum(a_33[:, None] * Ai_33, 0)
        Ai_33 = tl.where((o_i == i - 32)[:, None], a_33, Ai_33)
    for i in range(48 + 2, min(64, seqlen - chunk_id * BT)):
        a_44 = -tl.load(
            A_ptr + (bos + chunk_id * BT + i) * H * BT + head_id * BT + o_i + 48
        )
        a_44 += tl.sum(a_44[:, None] * Ai_44, 0)
        Ai_44 = tl.where((o_i == i - 48)[:, None], a_44, Ai_44)
    Ai_11 += m_I
    Ai_22 += m_I
    Ai_33 += m_I
    Ai_44 += m_I

    A_21 = desc.load([chunk_id * BT + 16, 0]).to(tl.float32)
    A_31 = desc.load([chunk_id * BT + 32, 0]).to(tl.float32)
    A_32 = desc.load([chunk_id * BT + 32, 16]).to(tl.float32)
    A_41 = desc.load([chunk_id * BT + 48, 0]).to(tl.float32)
    A_42 = desc.load([chunk_id * BT + 48, 16]).to(tl.float32)
    A_43 = desc.load([chunk_id * BT + 48, 32]).to(tl.float32)

    Ai_21 = -tl.dot(
        tl.dot(Ai_22, A_21, input_precision=DOT_PRECISION),
        Ai_11,
        input_precision=DOT_PRECISION,
    )
    Ai_32 = -tl.dot(
        tl.dot(Ai_33, A_32, input_precision=DOT_PRECISION),
        Ai_22,
        input_precision=DOT_PRECISION,
    )
    Ai_43 = -tl.dot(
        tl.dot(Ai_44, A_43, input_precision=DOT_PRECISION),
        Ai_33,
        input_precision=DOT_PRECISION,
    )

    Ai_31 = -tl.dot(
        Ai_33,
        tl.dot(A_31, Ai_11, input_precision=DOT_PRECISION)
        + tl.dot(A_32, Ai_21, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )
    Ai_42 = -tl.dot(
        Ai_44,
        tl.dot(A_42, Ai_22, input_precision=DOT_PRECISION)
        + tl.dot(A_43, Ai_32, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )
    Ai_41 = -tl.dot(
        Ai_44,
        tl.dot(A_41, Ai_11, input_precision=DOT_PRECISION)
        + tl.dot(A_42, Ai_21, input_precision=DOT_PRECISION)
        + tl.dot(A_43, Ai_31, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )

    # this sometimes produces NaN for some reasons...
    # zeros_16x16 = tl.zeros((16, 16), dtype=tl.float32)
    # zeros_16x32 = tl.zeros((16, 32), dtype=tl.float32)
    # Ai_1 = _concat_2d_dim1(_concat_2d_dim1(Ai_11, zeros_16x16), zeros_16x32)
    # Ai_2 = _concat_2d_dim1(_concat_2d_dim1(Ai_21, Ai_22), zeros_16x32)
    # Ai_3 = _concat_2d_dim1(_concat_2d_dim1(Ai_31, Ai_32), _concat_2d_dim1(Ai_33, zeros_16x16))
    # Ai_4 = _concat_2d_dim1(_concat_2d_dim1(Ai_41, Ai_42), _concat_2d_dim1(Ai_43, Ai_44))
    # Ai = _concat_2d_dim0(_concat_2d_dim0(Ai_1, Ai_2), _concat_2d_dim0(Ai_3, Ai_4))

    # NOTE: we can move zeros store at the start of the program
    zero16x16 = tl.zeros((16, 16), dtype=tl.float32)

    desc_o.store([chunk_id * BT + 0, 0], Ai_11)
    desc_o.store([chunk_id * BT + 0, 16], zero16x16)
    desc_o.store([chunk_id * BT + 0, 32], zero16x16)
    desc_o.store([chunk_id * BT + 0, 48], zero16x16)

    desc_o.store([chunk_id * BT + 16, 0], Ai_21)
    desc_o.store([chunk_id * BT + 16, 16], Ai_22)
    desc_o.store([chunk_id * BT + 16, 32], zero16x16)
    desc_o.store([chunk_id * BT + 16, 48], zero16x16)

    desc_o.store([chunk_id * BT + 32, 0], Ai_31)
    desc_o.store([chunk_id * BT + 32, 16], Ai_32)
    desc_o.store([chunk_id * BT + 32, 32], Ai_33)
    desc_o.store([chunk_id * BT + 32, 48], zero16x16)

    desc_o.store([chunk_id * BT + 48, 0], Ai_41)
    desc_o.store([chunk_id * BT + 48, 16], Ai_42)
    desc_o.store([chunk_id * BT + 48, 32], Ai_43)
    desc_o.store([chunk_id * BT + 48, 48], Ai_44)

    # syncthreads to make stores visible within a threadblock
    tl.debug_barrier()

    # compute WY representation
    # issue all loads ASAP
    # NOTE: we remove some masking, which might not be valid. sometimes we got NaN -> investigate.
    offs_t = bos + chunk_id * BT + tl.arange(0, BT)[:, None]
    v_ptrs = v_ptr + (offs_t * H * V_dim + head_id * V_dim + tl.arange(0, V_dim))
    v = tl.load(v_ptrs, mask=offs_t < bos + seqlen, other=0.0)  # [BT, V_dim]

    k_head_id = head_id // (H // Hg)
    k_ptrs = k_ptr + (offs_t * Hg * K_dim + k_head_id * K_dim + tl.arange(0, K_dim))
    k = tl.load(k_ptrs, mask=offs_t < bos + seqlen, other=0.0)  # [BT, K_dim]

    Ai_ptrs = Ai_ptr + (offs_t * H * BT + head_id * BT + tl.arange(0, BT))
    Ai = tl.load(Ai_ptrs, mask=offs_t < bos + seqlen, other=0.0)  # [BT, BT]

    offs_t = bos + chunk_id * BT + tl.arange(0, BT)
    beta = tl.load(
        beta_ptr + (offs_t * H + head_id), mask=offs_t < bos + seqlen, other=0.0
    )  # [BT]
    g_cu = tl.load(
        g_cu_ptr + (offs_t * H + head_id), mask=offs_t < bos + seqlen, other=0.0
    )  # [BT]

    # U = (Ai * beta) @ V
    Ab = Ai * beta
    u = tl.dot(Ab.to(v.dtype), v)

    offs_t = bos + chunk_id * BT + tl.arange(0, BT)[:, None]
    u_ptrs = u_ptr + (offs_t * H * V_dim + head_id * V_dim + tl.arange(0, V_dim))
    tl.store(u_ptrs, u, mask=offs_t < bos + seqlen)

    # W = (Ai * beta * g_cu) @ K
    Abg = Ab * tl.exp(g_cu)
    w = tl.dot(Abg.to(k.dtype), k)

    offs_t = bos + chunk_id * BT + tl.arange(0, BT)[:, None]
    w_ptrs = w_ptr + (offs_t * H * K_dim + head_id * K_dim + tl.arange(0, K_dim))
    tl.store(w_ptrs, w, mask=offs_t < bos + seqlen)


@triton.autotune(
    configs=[
        triton.Config({"BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [2, 3, 4]
        for BV in [16, 32, 64, 128]
    ],
    key=["H", "K_dim", "V_dim", "BT"],
)
@triton.jit
def chunk_gated_delta_rule_fwd_kernel_h(
    k_ptr,
    v_ptr,
    w_ptr,
    v_new_ptr,
    g_cu_ptr,
    h_ptr,
    h0_ptr,
    ht_ptr,
    cu_seqlens_ptr,
    chunk_offsets_ptr,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K_dim: tl.constexpr,
    V_dim: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_nh = tl.program_id(1)
    seq_id = i_nh // H
    head_id = i_nh % H

    bos = tl.load(cu_seqlens_ptr + seq_id).to(tl.int32)
    eos = tl.load(cu_seqlens_ptr + seq_id + 1).to(tl.int32)
    T = eos - bos
    NT = tl.cdiv(T, BT)
    boh = tl.load(chunk_offsets_ptr + seq_id).to(tl.int32)

    # calculate offset
    h_ptr += ((boh * H + head_id) * V_dim * K_dim).to(tl.int64)
    v_ptr += ((bos * H + head_id) * V_dim).to(tl.int64)
    k_ptr += ((bos * Hg + head_id // (H // Hg)) * K_dim).to(tl.int64)
    w_ptr += ((bos * H + head_id) * K_dim).to(tl.int64)
    v_new_ptr += ((bos * H + head_id) * V_dim).to(tl.int64)

    stride_v = H * V_dim
    stride_h = H * V_dim * K_dim
    stride_k = Hg * K_dim
    stride_w = H * K_dim

    h0_ptr = h0_ptr + i_nh * V_dim * K_dim
    ht_ptr = ht_ptr + i_nh * V_dim * K_dim

    # load initial state
    h0_ptrs = tl.make_block_ptr(
        h0_ptr, (V_dim, K_dim), (K_dim, 1), (i_v * BV, 0), (BV, K_dim), (1, 0)
    )
    h = tl.load(h0_ptrs, boundary_check=(0, 1)).to(tl.float32)  # [BV, K_dim]

    # main recurrence
    for chunk_id in range(NT):
        # save intermediate state for o computation
        h_ptrs = tl.make_block_ptr(
            h_ptr + chunk_id * stride_h,
            (V_dim, K_dim),
            (K_dim, 1),
            (i_v * BV, 0),
            (BV, K_dim),
            (1, 0),
        )
        tl.store(h_ptrs, h.to(h_ptrs.dtype.element_ty), boundary_check=(0, 1))

        # issue all loads first
        w_ptrs = tl.make_block_ptr(
            w_ptr, (T, K_dim), (stride_w, 1), (chunk_id * BT, 0), (BT, K_dim), (1, 0)
        )
        w = tl.load(w_ptrs, boundary_check=(0, 1))  # [BT, K_dim]

        v_ptrs = tl.make_block_ptr(
            v_ptr,
            (T, V_dim),
            (stride_v, 1),
            (chunk_id * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        v = tl.load(v_ptrs, boundary_check=(0, 1))

        k_ptrs = tl.make_block_ptr(
            k_ptr, (T, K_dim), (stride_k, 1), (chunk_id * BT, 0), (BT, K_dim), (1, 0)
        )
        k = tl.load(k_ptrs, boundary_check=(0, 1))  # [BT, K_dim]

        last_idx = min((chunk_id + 1) * BT, T) - 1
        g_cu_last = tl.load(g_cu_ptr + bos * H + last_idx * H + head_id)
        g_cu_ptrs = tl.make_block_ptr(
            g_cu_ptr + bos * H + head_id, (T,), (H,), (chunk_id * BT,), (BT,), (0,)
        )
        g_cu = tl.load(g_cu_ptrs, boundary_check=(0,))

        # computation
        v_new = v - tl.dot(w, h.to(w.dtype).T)  # [BT, BV]

        # save new value for o computation
        v_new_ptrs = tl.make_block_ptr(
            v_new_ptr,
            (T, V_dim),
            (stride_v, 1),
            (chunk_id * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        tl.store(v_new_ptrs, v_new.to(v_ptrs.dtype.element_ty), boundary_check=(0, 1))

        # apply g
        mask_t = (chunk_id * BT + tl.arange(0, BT)) < T
        v_new = v_new * tl.where(mask_t, tl.exp(g_cu_last - g_cu), 0)[:, None]
        h *= tl.exp(g_cu_last)

        # update state
        h = tl.dot(v_new.to(k.dtype).T, k, acc=h)

    # epilogue
    ht_ptrs = tl.make_block_ptr(
        ht_ptr, (V_dim, K_dim), (K_dim, 1), (i_v * BV, 0), (BV, K_dim), (1, 0)
    )
    tl.store(ht_ptrs, h.to(ht_ptrs.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({"BK": BK, "BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64, 128]
        for BV in [32, 64, 128]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K_dim", "V_dim", "BT"],
)
@triton.jit
def chunk_fwd_kernel_o(
    q_ptr,
    k_ptr,
    v_ptr,
    h_ptr,
    g_cu_ptr,
    o_ptr,
    cu_seqlens_ptr,
    chunk_indices_ptr,
    scale,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K_dim: tl.constexpr,
    V_dim: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_t = tl.program_id(1)
    i_h = tl.program_id(2)

    i_tg = i_t
    i_n = tl.load(chunk_indices_ptr + i_t * 2).to(tl.int32)
    i_t = tl.load(chunk_indices_ptr + i_t * 2 + 1).to(tl.int32)

    bos = tl.load(cu_seqlens_ptr + i_n).to(tl.int32)
    eos = tl.load(cu_seqlens_ptr + i_n + 1).to(tl.int32)
    T = eos - bos

    # offset calculation
    q_ptr += (bos * Hg + i_h // (H // Hg)) * K_dim
    k_ptr += (bos * Hg + i_h // (H // Hg)) * K_dim
    v_ptr += (bos * H + i_h) * V_dim
    o_ptr += (bos * H + i_h) * V_dim
    h_ptr += (i_tg * H + i_h).to(tl.int64) * V_dim * K_dim

    o = tl.zeros([BT, BV], dtype=tl.float32)
    A = tl.zeros([BT, BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K_dim, BK)):
        q_ptrs = tl.make_block_ptr(
            q_ptr, (T, K_dim), (Hg * K_dim, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        k_ptrs = tl.make_block_ptr(
            k_ptr, (T, K_dim), (Hg * K_dim, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        h_ptrs = tl.make_block_ptr(
            h_ptr, (V_dim, K_dim), (K_dim, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0)
        )

        q = tl.load(q_ptrs, boundary_check=(0, 1))  # [BT, BK]
        k = tl.load(k_ptrs, boundary_check=(0, 1))  # [BT, BK]
        h = tl.load(h_ptrs, boundary_check=(0, 1))  # [BV, BK]

        o = tl.dot(q, h.T, acc=o)  # [BT, BV]
        A = tl.dot(q, k.T, acc=A)  # [BT, BT]

    # apply g
    g_cu_ptr += bos * H + i_h
    g_cu_ptrs = tl.make_block_ptr(g_cu_ptr, (T,), (H,), (i_t * BT,), (BT,), (0,))
    g_cu = tl.load(g_cu_ptrs, boundary_check=(0,))
    o = o * tl.exp(g_cu)[:, None]
    A = A * tl.exp(g_cu[:, None] - g_cu[None, :])

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    A = tl.where(m_A, A, 0)

    v_ptrs = tl.make_block_ptr(
        v_ptr, (T, V_dim), (H * V_dim, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
    )
    o_ptrs = tl.make_block_ptr(
        o_ptr, (T, V_dim), (H * V_dim, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
    )
    v = tl.load(v_ptrs, boundary_check=(0, 1))

    o = tl.dot(A.to(v.dtype), v, acc=o) * scale
    tl.store(o_ptrs, o.to(o_ptrs.dtype.element_ty), boundary_check=(0, 1))


def run(
    q: Tensor,  # (total_seqlen, num_q_heads, head_dim)
    k: Tensor,  # (total_seqlen, num_k_heads, head_dim)
    v: Tensor,  # (total_seqlen, num_v_heads, head_dim)
    state: Tensor,  # (num_seqs, num_v_heads, head_dim, head_dim)
    A_log: Tensor,  # (num_v_heads)
    a: Tensor,  # (total_seqlen, num_v_heads)
    dt_bias: Tensor,  # (num_v_heads)
    b: Tensor,  # (total_seqlen, num_v_heads)
    cu_seqlens: Tensor,  # (num_seqlens + 1)
    scale: float,
):
    T, Hg, K_dim = k.shape
    N, H, V_dim, _ = state.shape

    # prepare chunk metadata
    # TODO: use int32
    # NOTE: this causes CUDA sync. to avoid it, we need to use "padded" chunk indices layout,
    # which requires rewrite of all subsequent kernels.
    BT = 64
    num_chunks = triton.cdiv(cu_seqlens.diff(1), BT)  # for each sequence
    chunk_offsets = torch.cat([cu_seqlens.new_tensor([0]), num_chunks]).cumsum(0)

    # 1st value is sequence ID, 2nd value is chunk_id within that sequence
    indices = torch.cat([torch.arange(n) for n in num_chunks.tolist()])
    chunk_indices = torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(
        cu_seqlens
    )
    total_num_chunks = chunk_indices.shape[0]

    # this kernel does multiple things:
    # - compute K @ K.T
    # - compute g and its chunk local cumsum
    # - compute beta
    # - compute strictLower(beta * Gamma * (K @ K.T))
    g_cu = torch.empty_like(a, dtype=torch.float32)
    beta = torch.empty_like(b, dtype=torch.float32)
    A = torch.empty(T, H, BT, device=k.device, dtype=torch.float32)
    chunk_scaled_dot_kkt_fwd_kernel[(total_num_chunks, Hg)](
        k,
        A_log,
        a,
        dt_bias,
        b,
        g_cu,
        beta,
        A,
        cu_seqlens,
        chunk_indices,
        H=H,
        Hg=Hg,
        K_dim=K_dim,
        BT=BT,
    )

    # - compute Ai = inverse(I + strictTriu(A))
    # - obtain WY representation: U = Ai @ V and W = (Ai * g_cu) @ K
    Ai = torch.empty_like(A, dtype=k.dtype)  # BF16
    u = torch.empty_like(v)
    w = k.new_empty(T, H, K_dim)
    merge_16x16_to_64x64_inverse_kernel[(total_num_chunks, H)](
        k,
        v,
        w,
        u,
        A,
        Ai,
        beta,
        g_cu,
        cu_seqlens,
        chunk_indices,
        H=H,
        Hg=Hg,
        K_dim=K_dim,
        V_dim=V_dim,
        BT=BT,
        DOT_PRECISION="ieee",
    )

    h = k.new_empty(total_num_chunks, H, V_dim, K_dim)
    final_state = torch.empty_like(state, dtype=torch.float32)
    v_new = torch.empty_like(u)

    # reduce BV to increase no. of SMs used.
    # helpful when N * H is small.
    def grid(meta):
        return (triton.cdiv(V_dim, meta["BV"]), N * H)

    chunk_gated_delta_rule_fwd_kernel_h[grid](
        k,
        u,
        w,
        v_new,
        g_cu,
        h,
        state,
        final_state,
        cu_seqlens,
        chunk_offsets,
        H=H,
        Hg=Hg,
        K_dim=K_dim,
        V_dim=V_dim,
        BT=BT,
    )

    o = torch.empty_like(v)

    # we only need separate o kernel if h kernel is too small?
    def grid(meta):
        return (triton.cdiv(V_dim, meta["BV"]), total_num_chunks, H)

    chunk_fwd_kernel_o[grid](
        q,
        k,
        v_new,
        h,
        g_cu,
        o,
        cu_seqlens,
        chunk_indices,
        scale=scale,
        H=H,
        Hg=Hg,
        K_dim=K_dim,
        V_dim=V_dim,
        BT=BT,
    )

    return o, final_state
