# from chunk_v7, but use chunk_v6c inverse kernel

import os
from pathlib import Path

import torch
import triton
import tvm_ffi
from torch import Tensor

os.environ["TVM_FFI_CUDA_ARCH_LIST"] = "10.0a"

CURRENT_DIR = Path(__file__).parent


lib_path = tvm_ffi.cpp.build(
    name="gdn_prefill_cuda",
    cuda_files=[
        str(CURRENT_DIR / "cuda_prep_meta_v2.cu"),
        str(CURRENT_DIR / "cuda_kkt_v1b.cu"),
        str(CURRENT_DIR / "cuda_inv_uw_v0.cu"),
        str(CURRENT_DIR / "cuda_h_v2.cu"),
        str(CURRENT_DIR / "cuda_o_v1.cu"),
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
    T, _, K_dim = k.shape
    N, H, V_dim, _ = state.shape

    BT = 64

    upper_bound_chunks = (N - 1) + triton.cdiv(T - (N - 1), BT)
    chunk_offsets = q.new_empty(N + 1, dtype=torch.int32)
    chunk_indices = q.new_empty((upper_bound_chunks, 2), dtype=torch.int32)
    total_chunks_ptr = chunk_offsets[N:]

    pad_T = upper_bound_chunks * BT

    mod.prep_meta_v2(cu_seqlens, chunk_indices, chunk_offsets)

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
    # padded UW so we can use TMA store
    pad_T = upper_bound_chunks * BT
    u = q.new_empty(pad_T, H, V_dim)
    w = q.new_empty(pad_T, H, V_dim)
    mod.inv_uw_v0(
        A,
        k,
        v,
        u,
        w,
        beta,
        g_cu,
        cu_seqlens,
        chunk_indices,
        total_chunks_ptr,
    )

    # padded v_new so we can use TMA store for v_new in H kernel
    h = k.new_empty(upper_bound_chunks, H, V_dim, K_dim)
    final_state = torch.empty_like(state, dtype=torch.float32)
    v_new = q.new_empty(pad_T, H, V_dim)
    v_new_chunks = v_new.view(upper_bound_chunks, BT, H, V_dim)

    # uncomment to enable profiling
    profiler = None
    # profiler = torch.zeros(148, 10, 1 + 1000 * 2, dtype=torch.int64, device="cuda")
    mod.h_v2(
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

    # if profiler is not None:
    #     export_trace(profiler, Path("trace.json.gz"))

    o = torch.empty_like(v)
    mod.o_v1(
        q,
        k,
        v_new_chunks,
        h,
        g_cu,
        o,
        cu_seqlens,
        chunk_indices,
        total_chunks_ptr,
        scale,
    )

    return o, final_state
