# https://github.com/vllm-project/vllm/blob/v0.17.0/vllm/model_executor/layers/fla/ops/chunk.py

import torch
import triton
import triton.language as tl
from torch import Tensor


def alloc_fn(size: int, alignment: int, stream: int | None):
    return torch.empty(size, device="cuda", dtype=torch.int8)


triton.set_allocator(alloc_fn)


@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8]],
    key=["H", "BT"],
)
@triton.jit
def chunk_local_cumsum_scalar_kernel(
    A_log_ptr,  # [H]
    a_ptr,  # [T, H]
    dt_bias_ptr,  # [H]
    o_ptr,
    cu_seqlens_ptr,
    chunk_indices_ptr,
    H: tl.constexpr,
    BT: tl.constexpr,
):
    global_chunk_id = tl.program_id(0)
    head_id = tl.program_id(1)

    seq_id = tl.load(chunk_indices_ptr + global_chunk_id * 2).to(tl.int32)
    chunk_id = tl.load(chunk_indices_ptr + global_chunk_id * 2 + 1).to(tl.int32)

    bos = tl.load(cu_seqlens_ptr + seq_id).to(tl.int32)
    eos = tl.load(cu_seqlens_ptr + seq_id + 1).to(tl.int32)
    seqlen = eos - bos

    # NOTE: strided load. maybe each pid should handle all heads.
    offs = chunk_id * BT + tl.arange(0, BT)
    a_ptrs = a_ptr + ((bos + offs) * H + head_id)
    a = tl.load(a_ptrs).to(tl.float32)  # no mask

    A_log = tl.load(A_log_ptr + head_id).to(tl.float32)
    dt_bias = tl.load(dt_bias_ptr + head_id).to(tl.float32)

    g = -tl.exp(A_log) * tl.log(1.0 + tl.exp(a + dt_bias))
    o = tl.cumsum(g, axis=0)

    offs = chunk_id * BT + tl.arange(0, BT)
    o_ptrs = o_ptr + ((bos + offs) * H + head_id)
    tl.store(o_ptrs, o, mask=offs < seqlen)


@triton.autotune(
    configs=[triton.Config(dict(), num_warps=num_warps) for num_warps in [2, 4, 8]],
    key=["H", "K", "BT"],
)
@triton.jit()
def chunk_scaled_dot_kkt_fwd_kernel(
    k_ptr,  # [total_seqlen, Hg, K]
    beta_ptr,  # [total_seqlen, H]
    g_cu_ptr,  # [total_seqlen, H]
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

    # don't need masking for loads, since we will apply masking for store
    offs_t = bos + chunk_id * BT + tl.arange(0, BT)
    offs_k = tl.arange(0, K_dim)
    k_ptrs = k_ptr + (offs_t[:, None] * Hg * K_dim + k_head_id * K_dim + offs_k)
    k = tl.load(k_ptrs)  # [BT, K]
    A = tl.dot(k, k.T)  # [BT, BT]

    for head_id in range(k_head_id * (H // Hg), (k_head_id + 1) * (H // Hg)):
        # load beta and g
        offs_t = bos + chunk_id * BT + tl.arange(0, BT)
        beta = tl.load(beta_ptr + (offs_t * H + head_id))
        g_cu = tl.load(g_cu_ptr + (offs_t * H + head_id))

        # apply beta and gamma
        A_ = A * beta[:, None]
        A_ = A_ * tl.exp(g_cu[:, None] - g_cu[None, :])

        offs_t = chunk_id * BT + tl.arange(0, BT)
        mask_t = offs_t < seqlen
        mask_A = (offs_t[:, None] > offs_t[None, :]) & (mask_t[:, None] & mask_t)
        A_ = tl.where(mask_A, A_, 0)
        A_ptrs = tl.make_block_ptr(
            A_ptr + (bos * H + head_id) * BT,
            (seqlen, BT),
            (BT * H, 1),
            (chunk_id * BT, 0),
            (BT, BT),
            (1, 0),
        )
        tl.store(A_ptrs, A_, boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4, 5]
    ],
    key=["H", "BT"],
)
@triton.jit
def merge_16x16_to_64x64_inverse_kernel(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    H: tl.constexpr,
    BT: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    i_t = tl.program_id(0)
    i_h = tl.program_id(1)

    i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
    i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)

    bos = tl.load(cu_seqlens + i_n).to(tl.int32)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos

    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    A += (bos * H + i_h) * BT
    Ai += (bos * H + i_h) * BT

    desc = tl.make_tensor_descriptor(A, [T, BT], [H * BT, 1], [16, 16])
    desc_o = tl.make_tensor_descriptor(Ai, [T, BT], [H * BT, 1], [16, 16])
    b_Ai_11 = desc.load([i_t * BT + 0, 0]).to(tl.float32)
    b_Ai_22 = desc.load([i_t * BT + 16, 16]).to(tl.float32)
    b_Ai_33 = desc.load([i_t * BT + 32, 32]).to(tl.float32)
    b_Ai_44 = desc.load([i_t * BT + 48, 48]).to(tl.float32)

    # [16, 16]
    b_Ai_11 = -tl.where(m_A, b_Ai_11, 0)
    b_Ai_22 = -tl.where(m_A, b_Ai_22, 0)
    b_Ai_33 = -tl.where(m_A, b_Ai_33, 0)
    b_Ai_44 = -tl.where(m_A, b_Ai_44, 0)

    for i in range(2, min(16, T - i_t * BT)):
        b_a_11 = -tl.load(A + (i_t * BT + i) * H * BT + o_i)
        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
        b_Ai_11 = tl.where((o_i == i)[:, None], b_a_11, b_Ai_11)
    for i in range(16 + 2, min(32, T - i_t * BT)):
        b_a_22 = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 16)
        b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
        b_Ai_22 = tl.where((o_i == i - 16)[:, None], b_a_22, b_Ai_22)
    for i in range(32 + 2, min(48, T - i_t * BT)):
        b_a_33 = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 32)
        b_a_33 += tl.sum(b_a_33[:, None] * b_Ai_33, 0)
        b_Ai_33 = tl.where((o_i == i - 32)[:, None], b_a_33, b_Ai_33)
    for i in range(48 + 2, min(64, T - i_t * BT)):
        b_a_44 = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 48)
        b_a_44 += tl.sum(b_a_44[:, None] * b_Ai_44, 0)
        b_Ai_44 = tl.where((o_i == i - 48)[:, None], b_a_44, b_Ai_44)
    b_Ai_11 += m_I
    b_Ai_22 += m_I
    b_Ai_33 += m_I
    b_Ai_44 += m_I

    b_A_21 = desc.load([i_t * BT + 16, 0]).to(tl.float32)
    b_A_31 = desc.load([i_t * BT + 32, 0]).to(tl.float32)
    b_A_32 = desc.load([i_t * BT + 32, 16]).to(tl.float32)
    b_A_41 = desc.load([i_t * BT + 48, 0]).to(tl.float32)
    b_A_42 = desc.load([i_t * BT + 48, 16]).to(tl.float32)
    b_A_43 = desc.load([i_t * BT + 48, 32]).to(tl.float32)

    b_Ai_21 = -tl.dot(
        tl.dot(b_Ai_22, b_A_21, input_precision=DOT_PRECISION),
        b_Ai_11,
        input_precision=DOT_PRECISION,
    )
    b_Ai_32 = -tl.dot(
        tl.dot(b_Ai_33, b_A_32, input_precision=DOT_PRECISION),
        b_Ai_22,
        input_precision=DOT_PRECISION,
    )
    b_Ai_43 = -tl.dot(
        tl.dot(b_Ai_44, b_A_43, input_precision=DOT_PRECISION),
        b_Ai_33,
        input_precision=DOT_PRECISION,
    )

    b_Ai_31 = -tl.dot(
        b_Ai_33,
        tl.dot(b_A_31, b_Ai_11, input_precision=DOT_PRECISION)
        + tl.dot(b_A_32, b_Ai_21, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )
    b_Ai_42 = -tl.dot(
        b_Ai_44,
        tl.dot(b_A_42, b_Ai_22, input_precision=DOT_PRECISION)
        + tl.dot(b_A_43, b_Ai_32, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )
    b_Ai_41 = -tl.dot(
        b_Ai_44,
        tl.dot(b_A_41, b_Ai_11, input_precision=DOT_PRECISION)
        + tl.dot(b_A_42, b_Ai_21, input_precision=DOT_PRECISION)
        + tl.dot(b_A_43, b_Ai_31, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )

    desc_o.store(
        [i_t * BT + 0, 0], b_Ai_11.to(desc_o.dtype, fp_downcast_rounding="rtne")
    )
    desc_o.store(
        [i_t * BT + 16, 16], b_Ai_22.to(desc_o.dtype, fp_downcast_rounding="rtne")
    )
    desc_o.store(
        [i_t * BT + 32, 32], b_Ai_33.to(desc_o.dtype, fp_downcast_rounding="rtne")
    )
    desc_o.store(
        [i_t * BT + 48, 48], b_Ai_44.to(desc_o.dtype, fp_downcast_rounding="rtne")
    )
    desc_o.store(
        [i_t * BT + 16, 0], b_Ai_21.to(desc_o.dtype, fp_downcast_rounding="rtne")
    )
    desc_o.store(
        [i_t * BT + 32, 0], b_Ai_31.to(desc_o.dtype, fp_downcast_rounding="rtne")
    )
    desc_o.store(
        [i_t * BT + 32, 16], b_Ai_32.to(desc_o.dtype, fp_downcast_rounding="rtne")
    )
    desc_o.store(
        [i_t * BT + 48, 0], b_Ai_41.to(desc_o.dtype, fp_downcast_rounding="rtne")
    )
    desc_o.store(
        [i_t * BT + 48, 16], b_Ai_42.to(desc_o.dtype, fp_downcast_rounding="rtne")
    )
    desc_o.store(
        [i_t * BT + 48, 32], b_Ai_43.to(desc_o.dtype, fp_downcast_rounding="rtne")
    )


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "V", "BT", "BK", "BV"],
)
@triton.jit
def recompute_w_u_fwd_kernel(
    k,
    v,
    beta,
    w,
    u,
    A,
    g,
    cu_seqlens,
    chunk_indices,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_t = tl.program_id(0)
    i_h = tl.program_id(1)

    i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
    i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)

    bos = tl.load(cu_seqlens + i_n).to(tl.int32)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos

    p_beta = tl.make_block_ptr(
        beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
    )
    p_g = tl.make_block_ptr(g + (bos * H + i_h), (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_g = tl.exp(tl.load(p_g, boundary_check=(0,)))

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        p_u = tl.make_block_ptr(
            u + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + (bos * Hg + i_h // (H // Hg)) * K,
            (T, K),
            (Hg * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_w = tl.make_block_ptr(
            w + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = (b_k * b_beta[:, None] * b_g[:, None]).to(b_k.dtype)
        b_w = tl.dot(b_A, b_kb)
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({"BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [2, 3, 4]
        for BV in [32, 64]
    ],
    key=["H", "K", "V", "BT"],
)
@triton.jit
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
    k,
    v,
    w,
    v_new,
    g,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_nh = tl.program_id(1)
    i_n = i_nh // H
    i_h = i_nh % H

    bos = tl.load(cu_seqlens + i_n).to(tl.int32)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos
    NT = tl.cdiv(T, BT)
    boh = tl.load(chunk_offsets + i_n).to(tl.int32)

    # [BV, BK]
    b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
    b_h2 = tl.zeros([BV, 64], dtype=tl.float32)

    # calculate offset
    h += ((boh * H + i_h) * V * K).to(tl.int64)
    v += ((bos * H + i_h) * V).to(tl.int64)
    k += ((bos * Hg + i_h // (H // Hg)) * K).to(tl.int64)
    w += ((bos * H + i_h) * K).to(tl.int64)
    v_new += ((bos * H + i_h) * V).to(tl.int64)

    stride_v = H * V
    stride_h = H * V * K
    stride_k = Hg * K
    stride_w = H * K

    h0 = h0 + i_nh * V * K
    ht = ht + i_nh * V * K

    # load initial state
    p_h0_1 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
    b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
    p_h0_2 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
    b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)

    # main recurrence
    for i_t in range(NT):
        p_h1 = tl.make_block_ptr(
            h + i_t * stride_h, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0)
        )
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
        p_h2 = tl.make_block_ptr(
            h + i_t * stride_h, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0)
        )
        tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))

        p_w = tl.make_block_ptr(
            w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, 64), (1, 0)
        )
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))
        p_w = tl.make_block_ptr(
            w, (T, K), (stride_w, 1), (i_t * BT, 64), (BT, 64), (1, 0)
        )
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v += tl.dot(b_w, tl.trans(b_h2).to(b_w.dtype))

        p_v = tl.make_block_ptr(
            v, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v

        # save new value
        p_v = tl.make_block_ptr(
            v_new, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        tl.store(p_v, b_v.to(p_v.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t + 1) * BT, T) - 1

        # apply g
        m_t = (i_t * BT + tl.arange(0, BT)) < T
        b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
        p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_v = b_v * tl.where(m_t, tl.exp(b_g_last - b_g), 0)[:, None]
        b_g_last = tl.exp(b_g_last)
        b_h1 *= b_g_last
        b_h2 *= b_g_last

        b_v = b_v.to(k.dtype.element_ty)

        p_k = tl.make_block_ptr(
            k, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1)
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h1 += tl.trans(tl.dot(b_k, b_v))
        p_k = tl.make_block_ptr(
            k, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1)
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h2 += tl.trans(tl.dot(b_k, b_v))

    # epilogue
    p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
    tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
    p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
    tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({"BK": BK, "BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64, 128]
        for BV in [32, 64, 128]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "V", "BT"],
)
@triton.jit
def chunk_fwd_kernel_o(
    q,
    k,
    v,
    h,
    g,
    o,
    cu_seqlens,
    chunk_indices,
    scale,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_t = tl.program_id(1)
    i_h = tl.program_id(2)

    i_tg = i_t
    i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
    i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)

    bos = tl.load(cu_seqlens + i_n).to(tl.int32)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos

    # offset calculation
    q += (bos * Hg + i_h // (H // Hg)) * K
    k += (bos * Hg + i_h // (H // Hg)) * K
    v += (bos * H + i_h) * V
    o += (bos * H + i_h) * V
    h += (i_tg * H + i_h).to(tl.int64) * V * K

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(
            q, (T, K), (Hg * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        p_k = tl.make_block_ptr(
            k, (K, T), (1, Hg * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1)
        )
        p_h = tl.make_block_ptr(
            h, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0)
        )
        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BV, BK]
        b_h = tl.load(p_h, boundary_check=(0, 1))

        # [BT, BK] @ [BK, BV] -> [BT, BV]
        b_o += tl.dot(b_q, tl.trans(b_h))
        # [BT, BK] @ [BK, BT] -> [BT, BT]
        b_A += tl.dot(b_q, b_k)

    # apply g
    g += bos * H + i_h
    p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_g = tl.load(p_g, boundary_check=(0,))
    b_o = b_o * tl.exp(b_g)[:, None]
    b_A = b_A * tl.exp(b_g[:, None] - b_g[None, :])

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)

    p_v = tl.make_block_ptr(
        v, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
    )
    p_o = tl.make_block_ptr(
        o, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
    )
    b_v = tl.load(p_v, boundary_check=(0, 1))

    # to fix mma -> mma layout conversion
    # already solved by triton v3.2 or higher
    b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


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

    # beta
    beta = b.float().sigmoid()

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

    # chunk local cumsum for g
    g_cu = torch.empty_like(a, dtype=torch.float32)
    chunk_local_cumsum_scalar_kernel[(total_num_chunks, H)](
        A_log,
        a,
        dt_bias,
        g_cu,
        cu_seqlens,
        chunk_indices,
        H=H,
        BT=BT,
    )

    # obtain WY representation. u is actually the new v.
    # compute strictLower(beta * Gamma * (K @ K.T))
    A = torch.empty(T, H, BT, device=k.device, dtype=torch.float32)
    chunk_scaled_dot_kkt_fwd_kernel[(total_num_chunks, Hg)](
        k,
        beta,
        g_cu,
        A,
        cu_seqlens,
        chunk_indices,
        H=H,
        Hg=Hg,
        K_dim=K_dim,
        BT=BT,
    )

    # compute inverse of I + strictTriu(A)
    Ai = torch.zeros_like(A, dtype=k.dtype)
    merge_16x16_to_64x64_inverse_kernel[(total_num_chunks, H)](
        A=A,
        Ai=Ai,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        H=H,
        BT=BT,
        DOT_PRECISION="ieee",
    )
    A = Ai

    u = torch.empty_like(v)
    w = k.new_empty(T, H, K_dim)
    recompute_w_u_fwd_kernel[(total_num_chunks, H)](
        k=k,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=A,
        g=g_cu,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        H=H,
        Hg=Hg,
        K=K_dim,
        V=V_dim,
        BT=BT,
        BK=64,
        BV=64,
    )

    h = k.new_empty(total_num_chunks, H, V_dim, K_dim)
    final_state = torch.empty_like(state, dtype=torch.float32)
    v_new = torch.empty_like(u)

    def grid(meta):
        return (triton.cdiv(V_dim, meta["BV"]), N * H)

    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g_cu,
        h=h,
        h0=state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        H=H,
        Hg=Hg,
        K=K_dim,
        V=V_dim,
        BT=BT,
    )

    o = torch.empty_like(v)

    def grid(meta):
        return (triton.cdiv(V_dim, meta["BV"]), total_num_chunks, H)

    chunk_fwd_kernel_o[grid](
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g_cu,
        o=o,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        H=H,
        Hg=Hg,
        K=K_dim,
        V=V_dim,
        BT=BT,
    )

    return o, final_state
