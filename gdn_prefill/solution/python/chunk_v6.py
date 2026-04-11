# chunk_v6: improved inverse kernel (block-wise W/U + bf16 roundtrip + tf32)
# Imports unchanged kernels from triton_v4; only the merge/inverse kernel is new.

import os
from pathlib import Path

import torch
import triton
import triton.language as tl
import tvm_ffi
from torch import Tensor

from .triton_v4 import (
    chunk_fwd_kernel_o,
    chunk_gated_delta_rule_fwd_kernel_h,
    compute_chunks_kernel,
)

os.environ["TVM_FFI_CUDA_ARCH_LIST"] = "10.0a"

CURRENT_DIR = Path(__file__).parent

lib_path = tvm_ffi.cpp.build(
    name="gdn_prefill_cuda",
    cuda_files=[
        str(CURRENT_DIR / "cuda_kkt_v1.cu"),
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


# ── New inverse kernel ────────────────────────────────────────────


@triton.jit
def _unit_lower_inverse_16x16(A_orig, DOT_PRECISION: tl.constexpr):
    """Compute (I + A)^{-1} for strict lower triangular A.

    Precision strategy:
    - The 6-dot Neumann series uses DOT_PRECISION="tf32" (1 MMA pass per dot, fast).
      On Blackwell, Triton's tf32 compiles to HMMA.1688 (m16n8k8), which has 11-bit
      mantissa. After 6 chained dots, error accumulates to ~6 * 2^{-11} ≈ 0.003.
      This is close to bf16 precision (~2^{-8} ≈ 0.004), so some elements round to
      wrong bf16 values. Using tf32 alone fails on some workloads.

    - Newton-Schulz refinement corrects this. Given approximate inverse Ai with
      relative error E (~10^{-3}):
          Ai_new = Ai @ (2I - M @ Ai)    where M = I + A
      The new error is E^2 (~10^{-6}), well below bf16's ~10^{-2.4} threshold.
      The 2 refinement dots use tf32x3 (3 MMA passes, 23-bit precision) so they
      don't re-introduce tf32-level error.

    Note: CUDA wmma avoids this issue because ptxas compiles wmma.m16n16k8 to
    HMMA.1684 (K=4), giving finer accumulation. Triton emits mma.sync which maps
    to HMMA.1688 (K=8), hence the need for Newton-Schulz.

    Total: 6 tf32 dots + 2 tf32x3 dots = 12 MMA passes per diagonal block,
    vs 18 MMA passes for full tf32x3. ~33% fewer passes.
    """
    o_i = tl.arange(0, 16)
    m_I = tl.where(o_i[:, None] == o_i[None, :], 1.0, 0.0)

    # Fast tf32 inverse: (I + A)^-1 = (I - A)(I + A^2)(I + A^4)(I + A^8)
    A = A_orig
    Ai = m_I - A
    A = tl.dot(A, A, input_precision=DOT_PRECISION)
    Ai = tl.dot(Ai, m_I + A, input_precision=DOT_PRECISION)
    A = tl.dot(A, A, input_precision=DOT_PRECISION)
    Ai = tl.dot(Ai, m_I + A, input_precision=DOT_PRECISION)
    A = tl.dot(A, A, input_precision=DOT_PRECISION)
    Ai = tl.dot(Ai, m_I + A, input_precision=DOT_PRECISION)

    # Newton-Schulz refinement: squares the error (E -> E^2)
    MAi = Ai + tl.dot(A_orig, Ai, input_precision="tf32x3")  # M @ Ai = (I + A) @ Ai
    Ai = tl.dot(Ai, 2.0 * m_I - MAi, input_precision="tf32x3")  # Ai @ (2I - M @ Ai)

    return Ai


@triton.jit
def merge_16x16_to_64x64_inverse_kernel_v2(
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
    chunk_id = tl.load(chunk_indices_ptr + (global_chunk_id * 2 + 1)).to(tl.int32)

    bos = tl.load(cu_seqlens_ptr + seq_id).to(tl.int32)
    eos = tl.load(cu_seqlens_ptr + (seq_id + 1)).to(tl.int32)
    seqlen = eos - bos

    # compute inverse
    A_ptr += bos * H * BT + head_id * BT
    Ai_ptr += bos * H * BT + head_id * BT

    offs_t = chunk_id * BT + tl.arange(0, 16)[:, None]
    offsets = offs_t * H * BT + tl.arange(0, 16)
    A_11 = tl.load(A_ptr + (offsets + (0 * H * BT + 0)), mask=offs_t < seqlen - 0, other=0.0)
    A_22 = tl.load(A_ptr + (offsets + (16 * H * BT + 16)), mask=offs_t < seqlen - 16, other=0.0)
    A_33 = tl.load(A_ptr + (offsets + (32 * H * BT + 32)), mask=offs_t < seqlen - 32, other=0.0)
    A_44 = tl.load(A_ptr + (offsets + (48 * H * BT + 48)), mask=offs_t < seqlen - 48, other=0.0)

    Ai_11 = _unit_lower_inverse_16x16(A_11, DOT_PRECISION=DOT_PRECISION)
    Ai_22 = _unit_lower_inverse_16x16(A_22, DOT_PRECISION=DOT_PRECISION)
    Ai_33 = _unit_lower_inverse_16x16(A_33, DOT_PRECISION=DOT_PRECISION)
    Ai_44 = _unit_lower_inverse_16x16(A_44, DOT_PRECISION=DOT_PRECISION)

    offs_t = chunk_id * BT + tl.arange(0, 16)[:, None]
    offsets = offs_t * H * BT + tl.arange(0, 16)
    A_21 = tl.load(A_ptr + (offsets + (16 * H * BT + 0)), mask=offs_t < seqlen - 16)
    A_31 = tl.load(A_ptr + (offsets + (32 * H * BT + 0)), mask=offs_t < seqlen - 32)
    A_32 = tl.load(A_ptr + (offsets + (32 * H * BT + 16)), mask=offs_t < seqlen - 32)
    A_41 = tl.load(A_ptr + (offsets + (48 * H * BT + 0)), mask=offs_t < seqlen - 48)
    A_42 = tl.load(A_ptr + (offsets + (48 * H * BT + 16)), mask=offs_t < seqlen - 48)
    A_43 = tl.load(A_ptr + (offsets + (48 * H * BT + 32)), mask=offs_t < seqlen - 48)

    # Off-diagonal blocks use tf32x3: up to 4 chained dots can accumulate ~4*2^{-11} ≈ 0.002
    # error with tf32, which is borderline for bf16. tf32x3 keeps it safe.
    tmp = tl.dot(Ai_22, A_21, input_precision="tf32x3")
    Ai_21 = -tl.dot(tmp, Ai_11, input_precision="tf32x3")

    tmp = tl.dot(Ai_33, A_32, input_precision="tf32x3")
    Ai_32 = -tl.dot(tmp, Ai_22, input_precision="tf32x3")

    tmp = tl.dot(Ai_44, A_43, input_precision="tf32x3")
    Ai_43 = -tl.dot(tmp, Ai_33, input_precision="tf32x3")

    tmp = tl.dot(A_31, Ai_11, input_precision="tf32x3")
    tmp = tl.dot(A_32, Ai_21, acc=tmp, input_precision="tf32x3")
    Ai_31 = -tl.dot(Ai_33, tmp, input_precision="tf32x3")

    tmp = tl.dot(A_42, Ai_22, input_precision="tf32x3")
    tmp = tl.dot(A_43, Ai_32, acc=tmp, input_precision="tf32x3")
    Ai_42 = -tl.dot(Ai_44, tmp, input_precision="tf32x3")

    tmp = tl.dot(A_41, Ai_11, input_precision="tf32x3")
    tmp = tl.dot(A_42, Ai_21, acc=tmp, input_precision="tf32x3")
    tmp = tl.dot(A_43, Ai_31, acc=tmp, input_precision="tf32x3")
    Ai_41 = -tl.dot(Ai_44, tmp, input_precision="tf32x3")

    # === Block-wise W/U: compute directly from register-resident Ai blocks ===
    #
    # Original chunk_v5 flow: store Ai to global (bf16) → debug_barrier → reload [64,64] → single big dot
    # New flow: skip the global roundtrip, compute W/U from the 10 Ai blocks directly.
    #
    # bf16 roundtrip (.to(bf16).to(f32)): the original path stored Ai as bf16 to global memory,
    # which truncated precision. We replicate that truncation here so the downstream W/U dots
    # see identical bf16 values. Without this, 2 workloads fail at the atol=1e-2 boundary.
    #
    # acc= chaining (below): each row-block's W/U is computed as a sum of [16,16]@[16,dim] dots
    # using tl.dot(..., acc=prev). This feeds the MMA accumulator across dots, matching the
    # accumulation order of the original single [64,64]@[64,dim] dot.
    Ai_11 = Ai_11.to(tl.bfloat16).to(tl.float32)
    Ai_21 = Ai_21.to(tl.bfloat16).to(tl.float32)
    Ai_22 = Ai_22.to(tl.bfloat16).to(tl.float32)
    Ai_31 = Ai_31.to(tl.bfloat16).to(tl.float32)
    Ai_32 = Ai_32.to(tl.bfloat16).to(tl.float32)
    Ai_33 = Ai_33.to(tl.bfloat16).to(tl.float32)
    Ai_41 = Ai_41.to(tl.bfloat16).to(tl.float32)
    Ai_42 = Ai_42.to(tl.bfloat16).to(tl.float32)
    Ai_43 = Ai_43.to(tl.bfloat16).to(tl.float32)
    Ai_44 = Ai_44.to(tl.bfloat16).to(tl.float32)

    # Offset pointers for W/U computation
    k_ptr += bos * Hg * K_dim + head_id // (H // Hg) * K_dim
    v_ptr += bos * H * V_dim + head_id * V_dim
    w_ptr += bos * H * K_dim + head_id * K_dim
    u_ptr += bos * H * V_dim + head_id * V_dim
    beta_ptr += bos * H + head_id
    g_cu_ptr += bos * H + head_id

    # Load per-block v, k, beta, g_cu (16-row blocks)
    offs_16 = tl.arange(0, 16)
    offs_v = tl.arange(0, V_dim)
    offs_k = tl.arange(0, K_dim)
    t_base = chunk_id * BT

    t0 = t_base + offs_16; m0 = t0 < seqlen
    t1 = t_base + 16 + offs_16; m1 = t1 < seqlen
    t2 = t_base + 32 + offs_16; m2 = t2 < seqlen
    t3 = t_base + 48 + offs_16; m3 = t3 < seqlen

    v0 = tl.load(v_ptr + (t0[:, None] * H * V_dim + offs_v[None, :]), mask=m0[:, None], other=0.0)
    v1 = tl.load(v_ptr + (t1[:, None] * H * V_dim + offs_v[None, :]), mask=m1[:, None], other=0.0)
    v2 = tl.load(v_ptr + (t2[:, None] * H * V_dim + offs_v[None, :]), mask=m2[:, None], other=0.0)
    v3 = tl.load(v_ptr + (t3[:, None] * H * V_dim + offs_v[None, :]), mask=m3[:, None], other=0.0)

    k0 = tl.load(k_ptr + (t0[:, None] * Hg * K_dim + offs_k[None, :]), mask=m0[:, None], other=0.0)
    k1 = tl.load(k_ptr + (t1[:, None] * Hg * K_dim + offs_k[None, :]), mask=m1[:, None], other=0.0)
    k2 = tl.load(k_ptr + (t2[:, None] * Hg * K_dim + offs_k[None, :]), mask=m2[:, None], other=0.0)
    k3 = tl.load(k_ptr + (t3[:, None] * Hg * K_dim + offs_k[None, :]), mask=m3[:, None], other=0.0)

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

    # Row block 0: only Ai_11 contributes (lower-triangular)
    Ab_00 = Ai_11 * b0
    u0 = tl.dot(Ab_00.to(v0.dtype), v0)
    w0 = tl.dot((Ab_00 * eg0).to(k0.dtype), k0)

    # Row block 1: Ai_21 @ block0 + Ai_22 @ block1
    Ab_10 = Ai_21 * b0
    Ab_11 = Ai_22 * b1
    u1 = tl.dot(Ab_10.to(v0.dtype), v0)
    u1 = tl.dot(Ab_11.to(v1.dtype), v1, acc=u1)
    w1 = tl.dot((Ab_10 * eg0).to(k0.dtype), k0)
    w1 = tl.dot((Ab_11 * eg1).to(k1.dtype), k1, acc=w1)

    # Row block 2: Ai_31..Ai_33
    Ab_20 = Ai_31 * b0
    Ab_21 = Ai_32 * b1
    Ab_22 = Ai_33 * b2
    u2 = tl.dot(Ab_20.to(v0.dtype), v0)
    u2 = tl.dot(Ab_21.to(v1.dtype), v1, acc=u2)
    u2 = tl.dot(Ab_22.to(v2.dtype), v2, acc=u2)
    w2 = tl.dot((Ab_20 * eg0).to(k0.dtype), k0)
    w2 = tl.dot((Ab_21 * eg1).to(k1.dtype), k1, acc=w2)
    w2 = tl.dot((Ab_22 * eg2).to(k2.dtype), k2, acc=w2)

    # Row block 3: all 4 column blocks
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

    # Store u and w
    tl.store(u_ptr + (t0[:, None] * H * V_dim + offs_v[None, :]), u0, mask=m0[:, None])
    tl.store(u_ptr + (t1[:, None] * H * V_dim + offs_v[None, :]), u1, mask=m1[:, None])
    tl.store(u_ptr + (t2[:, None] * H * V_dim + offs_v[None, :]), u2, mask=m2[:, None])
    tl.store(u_ptr + (t3[:, None] * H * V_dim + offs_v[None, :]), u3, mask=m3[:, None])

    tl.store(w_ptr + (t0[:, None] * H * K_dim + offs_k[None, :]), w0, mask=m0[:, None])
    tl.store(w_ptr + (t1[:, None] * H * K_dim + offs_k[None, :]), w1, mask=m1[:, None])
    tl.store(w_ptr + (t2[:, None] * H * K_dim + offs_k[None, :]), w2, mask=m2[:, None])
    tl.store(w_ptr + (t3[:, None] * H * K_dim + offs_k[None, :]), w3, mask=m3[:, None])


# ── Run function ──────────────────────────────────────────────────


_FLAG = None


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

    global _FLAG
    if _FLAG is None:
        _FLAG = q.new_zeros(1, dtype=torch.int32)

    upper_bound_chunks = (N - 1) + triton.cdiv(T - (N - 1), BT)
    num_chunks = q.new_empty(N, dtype=torch.int32)
    chunk_offsets = q.new_empty(N + 1, dtype=torch.int32)
    chunk_indices = q.new_empty((upper_bound_chunks, 2), dtype=torch.int32)
    compute_chunks_kernel[(N,)](
        cu_seqlens,
        num_chunks,
        chunk_offsets,
        chunk_indices,
        _FLAG,
        N=N,
        BT=BT,
        BLOCK_SIZE=triton.next_power_of_2(N),
    )
    total_num_chunks = chunk_offsets[-1].item()  # CUDA sync

    g_cu = torch.empty_like(a, dtype=torch.float32)
    beta = torch.empty_like(b, dtype=torch.float32)
    A = torch.empty(T, H, BT, device=k.device, dtype=torch.float32)
    mod.kkt_v1(
        k, A_log, a, dt_bias, b, g_cu, beta, A,
        cu_seqlens, chunk_indices, total_num_chunks,
    )

    Ai = torch.empty_like(A, dtype=k.dtype)  # BF16
    u = torch.empty_like(v)
    w = k.new_empty(T, H, K_dim)
    merge_16x16_to_64x64_inverse_kernel_v2[(total_num_chunks, H)](
        k, v, w, u, A, Ai, beta, g_cu,
        cu_seqlens, chunk_indices,
        H=H, Hg=Hg, K_dim=K_dim, V_dim=V_dim, BT=BT,
        DOT_PRECISION="tf32",  # tf32 for diagonal inverse speed; Newton-Schulz corrects precision
        num_warps=2,
    )

    h = k.new_empty(total_num_chunks, H, V_dim, K_dim)
    final_state = torch.empty_like(state, dtype=torch.float32)
    v_new = torch.empty_like(u)

    BV = 16
    grid = (triton.cdiv(V_dim, BV), N * H)
    chunk_gated_delta_rule_fwd_kernel_h[grid](
        k, u, w, v_new, g_cu, h, state, final_state,
        cu_seqlens, chunk_offsets,
        H=H, Hg=Hg, K_dim=K_dim, V_dim=V_dim, BT=BT, BV=BV,
        num_warps=4, num_stages=3,
    )

    o = torch.empty_like(v)

    BV = 64
    grid = (triton.cdiv(V_dim, BV), total_num_chunks, H)
    chunk_fwd_kernel_o[grid](
        q, k, v_new, h, g_cu, o,
        cu_seqlens, chunk_indices,
        scale=scale,
        H=H, Hg=Hg, K_dim=K_dim, V_dim=V_dim, BT=BT, BV=BV,
        num_warps=8,
    )

    return o, final_state
