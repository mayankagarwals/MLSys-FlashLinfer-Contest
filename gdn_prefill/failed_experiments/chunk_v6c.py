# chunk_v6c: EXPERIMENTAL — direct 64×64 inverse (reference only, not used in production)
# Uses single [64,64] Neumann series instead of hierarchical 16×16 blocks.
# Kept as reference for comparison.
# Also evaluated 32x32, also slower than 16x16.

import os
from pathlib import Path

import torch
import triton
import triton.language as tl
import tvm_ffi
from torch import Tensor

from .triton_v4 import (
    chunk_gated_delta_rule_fwd_kernel_h,
    compute_chunks_kernel,
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


# ── Direct 64×64 inverse kernel (experimental reference) ─────────


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

    Note: CUDA wmma also uses single-pass tf32 and happens to pass tests for
    these workloads, but this is not guaranteed for all inputs. Newton-Schulz
    provides a provable precision guarantee (error E -> E^2).

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
    k_ptr, v_ptr, w_ptr, u_ptr, A_ptr,
    beta_ptr, g_cu_ptr, cu_seqlens_ptr, chunk_indices_ptr,
    total_chunks_ptr,
    H: tl.constexpr, Hg: tl.constexpr, K_dim: tl.constexpr,
    V_dim: tl.constexpr, BT: tl.constexpr, DOT_PRECISION: tl.constexpr,
):
    global_chunk_id = tl.program_id(0)
    head_id = tl.program_id(1)

    # Early-exit for excess blocks (grid uses upper_bound_chunks, not total_num_chunks)
    if global_chunk_id >= tl.load(total_chunks_ptr).to(tl.int32):
        return

    seq_id   = tl.load(chunk_indices_ptr + global_chunk_id * 2).to(tl.int32)
    chunk_id = tl.load(chunk_indices_ptr + (global_chunk_id * 2 + 1)).to(tl.int32)

    bos      = tl.load(cu_seqlens_ptr + seq_id).to(tl.int32)
    eos      = tl.load(cu_seqlens_ptr + (seq_id + 1)).to(tl.int32)
    seqlen   = eos - bos

    # === Direct 64×64 Neumann series: (I+A)^{-1} = (I-A)(I+A²)...(I+A³²) ===
    # Simpler than hierarchical 16×16 (no off-diagonal blocks, no block-wise W/U),
    # but slower due to cubic compute growth (14 [64,64] dots vs 48 [16,16] dots).
    A_ptr += bos * H * BT + head_id * BT
    offs_bt = chunk_id * BT + tl.arange(0, BT)
    mask_bt = offs_bt < seqlen
    A = tl.load(A_ptr + (offs_bt[:, None] * H * BT + tl.arange(0, BT)[None, :]),
                mask=mask_bt[:, None], other=0.0)

    o_i = tl.arange(0, BT)
    m_I = tl.where(o_i[:, None] == o_i[None, :], 1.0, 0.0)

    # 10 tf32 dots: (I-A)(I+A²)(I+A⁴)(I+A⁸)(I+A¹⁶)(I+A³²)
    A_orig = A
    Ai = m_I - A
    A = tl.dot(A, A, input_precision=DOT_PRECISION)
    Ai = tl.dot(Ai, m_I + A, input_precision=DOT_PRECISION)
    A = tl.dot(A, A, input_precision=DOT_PRECISION)
    Ai = tl.dot(Ai, m_I + A, input_precision=DOT_PRECISION)
    A = tl.dot(A, A, input_precision=DOT_PRECISION)
    Ai = tl.dot(Ai, m_I + A, input_precision=DOT_PRECISION)
    A = tl.dot(A, A, input_precision=DOT_PRECISION)
    Ai = tl.dot(Ai, m_I + A, input_precision=DOT_PRECISION)
    A = tl.dot(A, A, input_precision=DOT_PRECISION)
    Ai = tl.dot(Ai, m_I + A, input_precision=DOT_PRECISION)

    # 2× Newton-Schulz: 10 chained tf32 dots need 2 iterations (E→E²→E⁴)
    MAi = Ai + tl.dot(A_orig, Ai, input_precision="tf32x3")
    Ai = tl.dot(Ai, 2.0 * m_I - MAi, input_precision="tf32x3")
    MAi = Ai + tl.dot(A_orig, Ai, input_precision="tf32x3")
    Ai = tl.dot(Ai, 2.0 * m_I - MAi, input_precision="tf32x3")

    # bf16 roundtrip: matches reference's fp32→bf16→MMA pattern
    Ai = Ai.to(tl.bfloat16).to(tl.float32)

    # Single [64,64] W/U dots (no block-wise needed)
    k_ptr += bos * Hg * K_dim + head_id // (H // Hg) * K_dim
    v_ptr += bos * H * V_dim + head_id * V_dim
    w_ptr += bos * H * K_dim + head_id * K_dim
    u_ptr += bos * H * V_dim + head_id * V_dim
    beta_ptr += bos * H + head_id
    g_cu_ptr += bos * H + head_id

    beta = tl.load(beta_ptr + offs_bt * H, mask=mask_bt, other=0.0)
    g_cu = tl.load(g_cu_ptr + offs_bt * H, mask=mask_bt, other=0.0)
    v_full = tl.load(v_ptr + (offs_bt[:, None] * H * V_dim + tl.arange(0, V_dim)[None, :]),
                     mask=mask_bt[:, None], other=0.0)
    k_full = tl.load(k_ptr + (offs_bt[:, None] * Hg * K_dim + tl.arange(0, K_dim)[None, :]),
                     mask=mask_bt[:, None], other=0.0)

    Ab = Ai * beta
    u = tl.dot(Ab.to(v_full.dtype), v_full)
    offs_bt2 = chunk_id * BT + tl.arange(0, BT)[:, None]
    tl.store(u_ptr + (offs_bt2 * H * V_dim + tl.arange(0, V_dim)[None, :]), u, mask=offs_bt2 < seqlen)

    Abg = Ab * tl.exp(g_cu)
    w = tl.dot(Abg.to(k_full.dtype), k_full)
    tl.store(w_ptr + (offs_bt2 * H * K_dim + tl.arange(0, K_dim)[None, :]), w, mask=offs_bt2 < seqlen)


# ── Fused O-kernel (from v6b: V-dim loop inside kernel, single grid dim) ───


@triton.jit
def chunk_fwd_kernel_o(
    q_ptr, k_ptr, v_ptr, h_ptr, g_cu_ptr, o_ptr,
    cu_seqlens_ptr, chunk_indices_ptr, total_chunks_ptr, scale,
    H: tl.constexpr, Hg: tl.constexpr, K_dim: tl.constexpr,
    V_dim: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr,
):
    global_chunk_id = tl.program_id(0)
    head_id         = tl.program_id(1)

    if global_chunk_id >= tl.load(total_chunks_ptr).to(tl.int32):
        return

    seq_id   = tl.load(chunk_indices_ptr + global_chunk_id * 2).to(tl.int32)
    chunk_id = tl.load(chunk_indices_ptr + (global_chunk_id * 2 + 1)).to(tl.int32)
    bos      = tl.load(cu_seqlens_ptr + seq_id).to(tl.int32)
    eos      = tl.load(cu_seqlens_ptr + (seq_id + 1)).to(tl.int32)
    seqlen   = eos - bos

    q_ptr    += (bos * Hg + head_id // (H // Hg)) * K_dim
    k_ptr    += (bos * Hg + head_id // (H // Hg)) * K_dim
    v_ptr    += (bos * H + head_id) * V_dim
    o_ptr    += (bos * H + head_id) * V_dim
    h_ptr    += (global_chunk_id * H + head_id) * V_dim * K_dim
    g_cu_ptr += bos * H + head_id

    offs_t  = chunk_id * BT + tl.arange(0, BT)[:, None]
    offs_k  = tl.arange(0, K_dim)[None, :]
    mask_t  = offs_t < seqlen

    q    = tl.load(q_ptr + (offs_t * (Hg * K_dim) + offs_k), mask=mask_t, other=0.0)
    k    = tl.load(k_ptr + (offs_t * (Hg * K_dim) + offs_k), mask=mask_t, other=0.0)
    t_1d = chunk_id * BT + tl.arange(0, BT)
    g_cu = tl.load(g_cu_ptr + t_1d * H, mask=t_1d < seqlen, other=0.0)

    # Precompute A and exp_g once (shared across V-tiles)
    A     = tl.dot(q, k.T) * tl.exp(g_cu[:, None] - g_cu[None, :])
    A     = tl.where(tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :], A, 0.0)
    exp_g = tl.exp(g_cu)

    # Fused V-dim loop (v6b optimization: eliminates 3D grid, reduces kernel launch overhead)
    for i_v in tl.static_range(V_dim // BV):
        offs_v       = i_v * BV + tl.arange(0, BV)[:, None]
        offs_v_block = i_v * BV + tl.arange(0, BV)[None, :]
        h = tl.load(h_ptr + (offs_v * K_dim + offs_k))
        v = tl.load(v_ptr + (offs_t * (H * V_dim) + offs_v_block), mask=mask_t, other=0.0)
        o = tl.dot(q, h.T) * exp_g[:, None]
        o = tl.dot(A.to(v.dtype), v, acc=o) * scale
        tl.store(o_ptr + (offs_t * (H * V_dim) + offs_v_block), o, mask=mask_t)


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
    num_chunks    = q.new_empty(N, dtype=torch.int32)
    chunk_offsets = q.new_empty(N + 1, dtype=torch.int32)
    chunk_indices = q.new_zeros((upper_bound_chunks, 2), dtype=torch.int32)
    compute_chunks_kernel[(N,)](
        cu_seqlens, num_chunks, chunk_offsets, chunk_indices, _FLAG,
        N=N, BT=BT, BLOCK_SIZE=triton.next_power_of_2(N),
    )
    # No .item() sync — pass GPU pointer directly; kernels early-exit for excess blocks
    total_chunks_ptr = chunk_offsets[N:]

    g_cu = torch.empty_like(a, dtype=torch.float32)
    beta = torch.empty_like(b, dtype=torch.float32)
    A    = torch.empty(T, H, BT, device=k.device, dtype=torch.float32)
    mod.kkt_v1b(k, A_log, a, dt_bias, b, g_cu, beta, A,
                cu_seqlens, chunk_indices, total_chunks_ptr)

    u = torch.empty_like(v)
    w = k.new_empty(T, H, K_dim)
    merge_16x16_to_64x64_inverse_kernel_v2[(upper_bound_chunks, H)](
        k, v, w, u, A, beta, g_cu,
        cu_seqlens, chunk_indices, total_chunks_ptr,
        H=H, Hg=Hg, K_dim=K_dim, V_dim=V_dim, BT=BT,
        DOT_PRECISION="tf32",  # direct 64x64 inverse (experimental, slower than 16x16)
        num_warps=4,
    )

    h           = k.new_empty(upper_bound_chunks, H, V_dim, K_dim)
    final_state = torch.empty_like(state, dtype=torch.float32)
    v_new       = torch.empty_like(u)

    BV = 16
    grid = (triton.cdiv(V_dim, BV), N * H)
    chunk_gated_delta_rule_fwd_kernel_h[grid](
        k, u, w, v_new, g_cu, h, state, final_state,
        cu_seqlens, chunk_offsets,
        H=H, Hg=Hg, K_dim=K_dim, V_dim=V_dim, BT=BT, BV=BV,
        num_warps=4, num_stages=3,
    )

    o  = torch.empty_like(v)
    BV = 64
    # Fused o-kernel: V-dim loop inside kernel, 2D grid (no 3D grid overhead)
    chunk_fwd_kernel_o[(upper_bound_chunks, H)](
        q, k, v_new, h, g_cu, o,
        cu_seqlens, chunk_indices, total_chunks_ptr,
        scale=scale,
        H=H, Hg=Hg, K_dim=K_dim, V_dim=V_dim, BT=BT, BV=BV,
        num_warps=8,
    )

    return o, final_state
