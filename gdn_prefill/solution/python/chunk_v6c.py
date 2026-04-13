# chunk_v6c: chunk_v6b + bf16 diagonal inverse experiment

import os
from pathlib import Path

import torch
import triton
import triton.language as tl
import tvm_ffi
from torch import Tensor

from .triton_v4 import (
    chunk_gated_delta_rule_fwd_kernel_h,
)

os.environ["TVM_FFI_CUDA_ARCH_LIST"] = "10.0a"

CURRENT_DIR = Path(__file__).parent

lib_path = tvm_ffi.cpp.build(
    name="gdn_prefill_cuda_v6c",
    cuda_files=[
        str(CURRENT_DIR / "cuda_kkt_v1b.cu"),
    ],
    extra_cflags=["-O3"],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "-lineinfo",
    ],
    extra_ldflags=["-lcuda"],
)

mod = tvm_ffi.load_module(lib_path)


@triton.jit
def _unit_lower_inverse_16x16_bf16_corr1(A_orig):
    o_i = tl.arange(0, 16)
    m_I = tl.where(o_i[:, None] == o_i[None, :], 1.0, 0.0)
    m_I_bf16 = m_I.to(tl.bfloat16)

    # 2-iteration Neumann: covers A through A^4 exactly, missing A^5..A^15.
    # Correction step recovers most of the remaining error.
    # Saves 2 bf16 dots (4 mma) per block vs 3-iteration version.
    A = A_orig.to(tl.bfloat16)
    Ai = m_I_bf16 - A

    A_pow = tl.dot(A, A)
    A_pow_bf16 = A_pow.to(tl.bfloat16)
    Ai = tl.dot(Ai, m_I_bf16 + A_pow_bf16)

    A_pow = tl.dot(A_pow_bf16, A_pow_bf16)
    A_pow_bf16 = A_pow.to(tl.bfloat16)
    Ai = tl.dot(Ai.to(tl.bfloat16), m_I_bf16 + A_pow_bf16)

    # Correction: compute residual R = (I+A)@Ai - I, then Ai = Ai @ (I - R)
    MAi = Ai + tl.dot(A_orig, Ai, input_precision="tf32")
    R = MAi - m_I
    Ai = tl.dot(Ai, m_I - R, input_precision="tf32")

    return Ai


@triton.jit
def merge_16x16_to_64x64_inverse_kernel_v2(
    k_ptr,
    v_ptr,
    w_ptr,
    u_ptr,
    A_ptr,
    beta_ptr,
    g_cu_ptr,
    cu_seqlens_ptr,
    chunk_indices_ptr,
    total_chunks_ptr,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K_dim: tl.constexpr,
    V_dim: tl.constexpr,
    BT: tl.constexpr,
):
    global_chunk_id = tl.program_id(0)
    head_id = tl.program_id(1)

    if global_chunk_id >= tl.load(total_chunks_ptr).to(tl.int32):
        return

    seq_id = tl.load(chunk_indices_ptr + global_chunk_id * 2).to(tl.int32)
    chunk_id = tl.load(chunk_indices_ptr + (global_chunk_id * 2 + 1)).to(tl.int32)
    bos = tl.load(cu_seqlens_ptr + seq_id).to(tl.int32)
    eos = tl.load(cu_seqlens_ptr + (seq_id + 1)).to(tl.int32)
    seqlen = eos - bos

    A_ptr += bos * H * BT + head_id * BT

    offs_t = chunk_id * BT + tl.arange(0, 16)[:, None]
    offsets = offs_t * H * BT + tl.arange(0, 16)

    A_11 = tl.load(
        A_ptr + (offsets + (0 * H * BT + 0)), mask=offs_t < seqlen - 0, other=0.0
    )
    A_22 = tl.load(
        A_ptr + (offsets + (16 * H * BT + 16)), mask=offs_t < seqlen - 16, other=0.0
    )
    A_33 = tl.load(
        A_ptr + (offsets + (32 * H * BT + 32)), mask=offs_t < seqlen - 32, other=0.0
    )
    A_44 = tl.load(
        A_ptr + (offsets + (48 * H * BT + 48)), mask=offs_t < seqlen - 48, other=0.0
    )

    Ai_11 = _unit_lower_inverse_16x16_bf16_corr1(A_11)
    Ai_22 = _unit_lower_inverse_16x16_bf16_corr1(A_22)
    Ai_33 = _unit_lower_inverse_16x16_bf16_corr1(A_33)
    Ai_44 = _unit_lower_inverse_16x16_bf16_corr1(A_44)

    # Early bf16 roundtrip on diagonal blocks — enables faster bf16 MMA for off-diagonal
    Ai_11 = Ai_11.to(tl.bfloat16).to(tl.float32)
    Ai_22 = Ai_22.to(tl.bfloat16).to(tl.float32)
    Ai_33 = Ai_33.to(tl.bfloat16).to(tl.float32)
    Ai_44 = Ai_44.to(tl.bfloat16).to(tl.float32)

    A_21 = tl.load(A_ptr + (offsets + (16 * H * BT + 0)), mask=offs_t < seqlen - 16)
    A_31 = tl.load(A_ptr + (offsets + (32 * H * BT + 0)), mask=offs_t < seqlen - 32)
    A_32 = tl.load(A_ptr + (offsets + (32 * H * BT + 16)), mask=offs_t < seqlen - 32)
    A_41 = tl.load(A_ptr + (offsets + (48 * H * BT + 0)), mask=offs_t < seqlen - 48)
    A_42 = tl.load(A_ptr + (offsets + (48 * H * BT + 16)), mask=offs_t < seqlen - 48)
    A_43 = tl.load(A_ptr + (offsets + (48 * H * BT + 32)), mask=offs_t < seqlen - 48)

    # Level 0: bf16 MMA (faster — diagonal inputs already bf16-rounded)
    tmp = tl.dot(Ai_22.to(tl.bfloat16), A_21.to(tl.bfloat16))
    Ai_21 = -tl.dot(tmp.to(tl.bfloat16), Ai_11.to(tl.bfloat16))
    tmp = tl.dot(Ai_33.to(tl.bfloat16), A_32.to(tl.bfloat16))
    Ai_32 = -tl.dot(tmp.to(tl.bfloat16), Ai_22.to(tl.bfloat16))
    tmp = tl.dot(Ai_44.to(tl.bfloat16), A_43.to(tl.bfloat16))
    Ai_43 = -tl.dot(tmp.to(tl.bfloat16), Ai_33.to(tl.bfloat16))
    # Level 1: bf16 MMA
    tmp = tl.dot(A_31.to(tl.bfloat16), Ai_11.to(tl.bfloat16))
    tmp = tl.dot(A_32.to(tl.bfloat16), Ai_21.to(tl.bfloat16), acc=tmp)
    Ai_31 = -tl.dot(Ai_33.to(tl.bfloat16), tmp.to(tl.bfloat16))
    tmp = tl.dot(A_42.to(tl.bfloat16), Ai_22.to(tl.bfloat16))
    tmp = tl.dot(A_43.to(tl.bfloat16), Ai_32.to(tl.bfloat16), acc=tmp)
    Ai_42 = -tl.dot(Ai_44.to(tl.bfloat16), tmp.to(tl.bfloat16))
    # Level 2: tf32x3 for deepest chain (precision-sensitive)
    tmp = tl.dot(A_41, Ai_11, input_precision="tf32x3")
    tmp = tl.dot(A_42, Ai_21, acc=tmp, input_precision="tf32x3")
    tmp = tl.dot(A_43, Ai_31, acc=tmp, input_precision="tf32x3")
    Ai_41 = -tl.dot(Ai_44, tmp, input_precision="tf32x3")

    # bf16 roundtrip on off-diagonal blocks (diagonal already done above)
    Ai_21 = Ai_21.to(tl.bfloat16).to(tl.float32)
    Ai_31 = Ai_31.to(tl.bfloat16).to(tl.float32)
    Ai_32 = Ai_32.to(tl.bfloat16).to(tl.float32)
    Ai_41 = Ai_41.to(tl.bfloat16).to(tl.float32)
    Ai_42 = Ai_42.to(tl.bfloat16).to(tl.float32)
    Ai_43 = Ai_43.to(tl.bfloat16).to(tl.float32)

    k_ptr += bos * Hg * K_dim + head_id // (H // Hg) * K_dim
    v_ptr += bos * H * V_dim + head_id * V_dim
    w_ptr += bos * H * K_dim + head_id * K_dim
    u_ptr += bos * H * V_dim + head_id * V_dim
    beta_ptr += bos * H + head_id
    g_cu_ptr += bos * H + head_id

    offs_16 = tl.arange(0, 16)
    offs_v = tl.arange(0, V_dim)
    offs_k = tl.arange(0, K_dim)
    t_base = chunk_id * BT

    t0 = t_base + offs_16
    m0 = t0 < seqlen
    t1 = t_base + 16 + offs_16
    m1 = t1 < seqlen
    t2 = t_base + 32 + offs_16
    m2 = t2 < seqlen
    t3 = t_base + 48 + offs_16
    m3 = t3 < seqlen

    v0 = tl.load(
        v_ptr + (t0[:, None] * H * V_dim + offs_v[None, :]), mask=m0[:, None], other=0.0
    )
    v1 = tl.load(
        v_ptr + (t1[:, None] * H * V_dim + offs_v[None, :]), mask=m1[:, None], other=0.0
    )
    v2 = tl.load(
        v_ptr + (t2[:, None] * H * V_dim + offs_v[None, :]), mask=m2[:, None], other=0.0
    )
    v3 = tl.load(
        v_ptr + (t3[:, None] * H * V_dim + offs_v[None, :]), mask=m3[:, None], other=0.0
    )
    k0 = tl.load(
        k_ptr + (t0[:, None] * Hg * K_dim + offs_k[None, :]),
        mask=m0[:, None],
        other=0.0,
    )
    k1 = tl.load(
        k_ptr + (t1[:, None] * Hg * K_dim + offs_k[None, :]),
        mask=m1[:, None],
        other=0.0,
    )
    k2 = tl.load(
        k_ptr + (t2[:, None] * Hg * K_dim + offs_k[None, :]),
        mask=m2[:, None],
        other=0.0,
    )
    k3 = tl.load(
        k_ptr + (t3[:, None] * Hg * K_dim + offs_k[None, :]),
        mask=m3[:, None],
        other=0.0,
    )
    b0 = tl.load(beta_ptr + t0 * H, mask=m0, other=0.0)
    b1 = tl.load(beta_ptr + t1 * H, mask=m1, other=0.0)
    b2 = tl.load(beta_ptr + t2 * H, mask=m2, other=0.0)
    b3 = tl.load(beta_ptr + t3 * H, mask=m3, other=0.0)
    g0 = tl.load(g_cu_ptr + t0 * H, mask=m0, other=0.0)
    g1 = tl.load(g_cu_ptr + t1 * H, mask=m1, other=0.0)
    g2 = tl.load(g_cu_ptr + t2 * H, mask=m2, other=0.0)
    g3 = tl.load(g_cu_ptr + t3 * H, mask=m3, other=0.0)
    eg0 = tl.exp(g0)
    eg1 = tl.exp(g1)
    eg2 = tl.exp(g2)
    eg3 = tl.exp(g3)

    Ab_00 = Ai_11 * b0
    u0 = tl.dot(Ab_00.to(v0.dtype), v0)
    w0 = tl.dot((Ab_00 * eg0).to(k0.dtype), k0)
    Ab_10 = Ai_21 * b0
    Ab_11 = Ai_22 * b1
    u1 = tl.dot(Ab_10.to(v0.dtype), v0)
    u1 = tl.dot(Ab_11.to(v1.dtype), v1, acc=u1)
    w1 = tl.dot((Ab_10 * eg0).to(k0.dtype), k0)
    w1 = tl.dot((Ab_11 * eg1).to(k1.dtype), k1, acc=w1)
    Ab_20 = Ai_31 * b0
    Ab_21 = Ai_32 * b1
    Ab_22 = Ai_33 * b2
    u2 = tl.dot(Ab_20.to(v0.dtype), v0)
    u2 = tl.dot(Ab_21.to(v1.dtype), v1, acc=u2)
    u2 = tl.dot(Ab_22.to(v2.dtype), v2, acc=u2)
    w2 = tl.dot((Ab_20 * eg0).to(k0.dtype), k0)
    w2 = tl.dot((Ab_21 * eg1).to(k1.dtype), k1, acc=w2)
    w2 = tl.dot((Ab_22 * eg2).to(k2.dtype), k2, acc=w2)
    Ab_30 = Ai_41 * b0
    Ab_31 = Ai_42 * b1
    Ab_32 = Ai_43 * b2
    Ab_33 = Ai_44 * b3
    u3 = tl.dot(Ab_30.to(v0.dtype), v0)
    u3 = tl.dot(Ab_31.to(v1.dtype), v1, acc=u3)
    u3 = tl.dot(Ab_32.to(v2.dtype), v2, acc=u3)
    u3 = tl.dot(Ab_33.to(v3.dtype), v3, acc=u3)
    w3 = tl.dot((Ab_30 * eg0).to(k0.dtype), k0)
    w3 = tl.dot((Ab_31 * eg1).to(k1.dtype), k1, acc=w3)
    w3 = tl.dot((Ab_32 * eg2).to(k2.dtype), k2, acc=w3)
    w3 = tl.dot((Ab_33 * eg3).to(k3.dtype), k3, acc=w3)

    tl.store(u_ptr + (t0[:, None] * H * V_dim + offs_v[None, :]), u0, mask=m0[:, None])
    tl.store(u_ptr + (t1[:, None] * H * V_dim + offs_v[None, :]), u1, mask=m1[:, None])
    tl.store(u_ptr + (t2[:, None] * H * V_dim + offs_v[None, :]), u2, mask=m2[:, None])
    tl.store(u_ptr + (t3[:, None] * H * V_dim + offs_v[None, :]), u3, mask=m3[:, None])
    tl.store(w_ptr + (t0[:, None] * H * K_dim + offs_k[None, :]), w0, mask=m0[:, None])
    tl.store(w_ptr + (t1[:, None] * H * K_dim + offs_k[None, :]), w1, mask=m1[:, None])
    tl.store(w_ptr + (t2[:, None] * H * K_dim + offs_k[None, :]), w2, mask=m2[:, None])
    tl.store(w_ptr + (t3[:, None] * H * K_dim + offs_k[None, :]), w3, mask=m3[:, None])


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
    total_chunks_ptr,
    scale,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K_dim: tl.constexpr,
    V_dim: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
):
    global_chunk_id = tl.program_id(0)
    head_id = tl.program_id(1)

    if global_chunk_id >= tl.load(total_chunks_ptr).to(tl.int32):
        return

    seq_id = tl.load(chunk_indices_ptr + global_chunk_id * 2).to(tl.int32)
    chunk_id = tl.load(chunk_indices_ptr + (global_chunk_id * 2 + 1)).to(tl.int32)
    bos = tl.load(cu_seqlens_ptr + seq_id).to(tl.int32)
    eos = tl.load(cu_seqlens_ptr + (seq_id + 1)).to(tl.int32)
    seqlen = eos - bos

    q_ptr += (bos * Hg + head_id // (H // Hg)) * K_dim
    k_ptr += (bos * Hg + head_id // (H // Hg)) * K_dim
    v_ptr += (bos * H + head_id) * V_dim
    o_ptr += (bos * H + head_id) * V_dim
    h_ptr += (global_chunk_id * H + head_id) * V_dim * K_dim
    g_cu_ptr += bos * H + head_id

    offs_t = chunk_id * BT + tl.arange(0, BT)[:, None]
    offs_k = tl.arange(0, K_dim)[None, :]
    mask_t = offs_t < seqlen

    q = tl.load(q_ptr + (offs_t * (Hg * K_dim) + offs_k), mask=mask_t, other=0.0)
    k = tl.load(k_ptr + (offs_t * (Hg * K_dim) + offs_k), mask=mask_t, other=0.0)
    t_1d = chunk_id * BT + tl.arange(0, BT)
    g_cu = tl.load(g_cu_ptr + t_1d * H, mask=t_1d < seqlen, other=0.0)

    A = tl.dot(q, k.T) * tl.exp(g_cu[:, None] - g_cu[None, :])
    A = tl.where(tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :], A, 0.0)
    exp_g = tl.exp(g_cu)

    for i_v in tl.static_range(V_dim // BV):
        offs_v = i_v * BV + tl.arange(0, BV)[:, None]
        offs_v_block = i_v * BV + tl.arange(0, BV)[None, :]
        h = tl.load(h_ptr + (offs_v * K_dim + offs_k))
        v = tl.load(
            v_ptr + (offs_t * (H * V_dim) + offs_v_block), mask=mask_t, other=0.0
        )
        o = tl.dot(q, h.T) * exp_g[:, None]
        o = tl.dot(A.to(v.dtype), v, acc=o) * scale
        tl.store(o_ptr + (offs_t * (H * V_dim) + offs_v_block), o, mask=mask_t)


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

    BT = 64

    upper_bound_chunks = (N - 1) + triton.cdiv(T - (N - 1), BT)
    chunk_offsets = q.new_empty(N + 1, dtype=torch.int32)
    chunk_indices = q.new_empty((upper_bound_chunks, 2), dtype=torch.int32)
    total_chunks_ptr = chunk_offsets[N:]

    g_cu = torch.empty_like(a, dtype=torch.float32)
    beta = torch.empty_like(b, dtype=torch.float32)
    A = torch.empty(T, H, BT, device=k.device, dtype=torch.float32)
    mod.kkt_v1b_with_meta(
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
        chunk_offsets,
    )

    u = torch.empty_like(v)
    w = k.new_empty(T, H, K_dim)
    merge_16x16_to_64x64_inverse_kernel_v2[(upper_bound_chunks, H)](
        k,
        v,
        w,
        u,
        A,
        beta,
        g_cu,
        cu_seqlens,
        chunk_indices,
        total_chunks_ptr,
        H=H,
        Hg=Hg,
        K_dim=K_dim,
        V_dim=V_dim,
        BT=BT,
        num_warps=2,
    )

    h = k.new_empty(upper_bound_chunks, H, V_dim, K_dim)
    final_state = torch.empty_like(state, dtype=torch.float32)
    v_new = torch.empty_like(u)

    BV = 16
    grid = (triton.cdiv(V_dim, BV), N * H)
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
        BV=BV,
        num_warps=4,
        num_stages=3,
    )

    o = torch.empty_like(v)
    BV = 64
    chunk_fwd_kernel_o[(upper_bound_chunks, H)](
        q,
        k,
        v_new,
        h,
        g_cu,
        o,
        cu_seqlens,
        chunk_indices,
        total_chunks_ptr,
        scale=scale,
        H=H,
        Hg=Hg,
        K_dim=K_dim,
        V_dim=V_dim,
        BT=BT,
        BV=BV,
        num_warps=4,
    )

    return o, final_state
