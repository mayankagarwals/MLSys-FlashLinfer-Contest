# from triton_v6b, but replace H kernel with CUDA version

import os
from pathlib import Path

import torch
import triton
import triton.language as tl
import tvm_ffi
from torch import Tensor

from .chunk_v6b import (
    compute_chunks_kernel,
    merge_16x16_to_64x64_inverse_kernel_v2,
)

os.environ["TVM_FFI_CUDA_ARCH_LIST"] = "10.0a"

CURRENT_DIR = Path(__file__).parent

lib_path = tvm_ffi.cpp.build(
    name="gdn_prefill_cuda",
    cuda_files=[
        str(CURRENT_DIR / "cuda_kkt_v1b.cu"),
        str(CURRENT_DIR / "cuda_h_v1.cu"),
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
    v_ptr += ((global_chunk_id - chunk_id) * BT * H + head_id) * V_dim
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


def export_trace(profiler: Tensor, path: Path):
    import gzip
    import json

    num_sms, num_warps, _ = profiler.shape

    TAGS = [
        "START",
        "SETUP",
        "WAIT_MMA",
        "WAIT_TMA",
        "WAIT_H0",
        "WAIT_WH_IN",
        "WAIT_VK_IN",
        "WAIT_WH_MMA",
        "WAIT_VK_MMA",
        "ISSUE_TMA",
        "ISSUE_WH_MMA",
        "ISSUE_VK_MMA",
        # H warps
        "COMPUTE_H_SCALE",
        "PROCESS_H",
        "PROCESS_SCALED_H",
        # V warps
        "COMPUTE_V_SCALE",
        "PROCESS_V",
        "PROCESS_SCALED_V",
        "END",
    ]

    events = []
    profiler_data = profiler.tolist()
    for sm_id in range(num_sms):
        for warp_id in range(num_warps):
            data = profiler_data[sm_id][warp_id]
            cnt = data[0]

            if cnt == 0:
                continue

            start = 0
            for i in range(cnt):
                tag, ts = data[1 + i * 2 : 1 + (i + 1) * 2]

                # skip START tag
                # NOTE: there might be more than 1 start tag
                if tag > 0:
                    evt = dict(
                        name=TAGS[tag],
                        ph="X",
                        ts=start,
                        dur=ts - start,
                        pid=sm_id,
                        tid=sm_id + warp_id,
                    )
                    events.append(evt)

                start = ts

    offset = min([evt["ts"] for evt in events])
    for evt in events:
        evt["ts"] -= offset

    path.parent.mkdir(exist_ok=True)
    trace = dict(traceEvents=events)
    gzip.open(path, "w").write(json.dumps(trace).encode("utf-8"))


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

    # prepare chunk metadata
    BT = 64

    # # PyTorch version
    # num_chunks = triton.cdiv(cu_seqlens.diff(1), BT)  # for each sequence
    # chunk_offsets = F.pad(num_chunks, (1, 0)).cumsum(0)

    # # 1st value is sequence ID, 2nd value is chunk_id within that sequence
    # indices = torch.cat([torch.arange(n) for n in num_chunks.tolist()])
    # chunk_indices = torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1)
    # chunk_indices = chunk_indices.to(cu_seqlens.device, non_blocking=True)
    # total_num_chunks = chunk_indices.shape[0]

    # Triton version
    # flag for grid sync
    global _FLAG
    if _FLAG is None:
        _FLAG = q.new_zeros(1, dtype=torch.int32)

    # we allocate more than enough for chunk_indices so that we don't need to know
    # the value of total_num_chunks before calling the kernel.
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
        # max N is 57 -> max BLOCK_SIZE is 64, still very small
        BLOCK_SIZE=triton.next_power_of_2(N),
    )
    total_chunks_ptr = chunk_offsets[N:]

    # this kernel does multiple things:
    # - compute K @ K.T
    # - compute g and its chunk local cumsum
    # - compute beta
    # - compute strictLower(beta * Gamma * (K @ K.T))
    # NOTE: transpose g_cu and beta to make them T-contiguous?
    g_cu = torch.empty_like(a, dtype=torch.float32)
    beta = torch.empty_like(b, dtype=torch.float32)
    A = torch.empty(T, H, BT, device=k.device, dtype=torch.float32)
    mod.kkt_v1b(
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
        total_chunks_ptr,
    )

    # - compute Ai = inverse(I + strictTriu(A))
    # - obtain WY representation: U = Ai @ V and W = (Ai * g_cu) @ K
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
        DOT_PRECISION="tf32",
        num_warps=2,
    )

    # padded v_new so we can use TMA store for v_new in H kernel
    h = k.new_empty(upper_bound_chunks, H, V_dim, K_dim)
    final_state = torch.empty_like(state, dtype=torch.float32)
    v_new = q.new_empty(upper_bound_chunks, BT, H, V_dim)

    # uncomment to enable profiling
    profiler = None
    # profiler = torch.zeros(148, 10, 1 + 1000 * 2, dtype=torch.int64, device="cuda")
    mod.h_v1(
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
        profiler,
    )

    if profiler is not None:
        export_trace(profiler, Path("trace.json.gz"))

    o = torch.empty_like(v)

    # we only need separate o kernel if h kernel is too small?
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
        num_warps=8,
    )

    return o, final_state
