# from chunk_v11, but fuses kkt_v1b + inv_uw_v1 into kkt_inv_uw_v1
# (avoids materializing A in gmem)

import os
from pathlib import Path

import torch
import triton
import tvm_ffi
from torch import Tensor

os.environ["TVM_FFI_CUDA_ARCH_LIST"] = "10.0a"

CURRENT_DIR = Path(__file__).parent


lib_path = tvm_ffi.cpp.build(
    name="gdn_prefill_cuda_v12",
    cuda_files=[
        str(CURRENT_DIR / "cuda_prep_meta_v2.cu"),
        str(CURRENT_DIR / "cuda_kkt_inv_uw_v1.cu"),
        str(CURRENT_DIR / "cuda_h_v2b.cu"),
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
    q: Tensor,
    k: Tensor,
    v: Tensor,
    state: Tensor,
    A_log: Tensor,
    a: Tensor,
    dt_bias: Tensor,
    b: Tensor,
    cu_seqlens: Tensor,
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

    # fused KKT + inverse + UW (no A in gmem)
    u = q.new_empty(pad_T, H, V_dim)
    w = q.new_empty(pad_T, H, V_dim)
    mod.kkt_inv_uw_v1(
        k,
        v,
        u,
        w,
        A_log,
        a,
        dt_bias,
        b,
        g_cu,
        cu_seqlens,
        chunk_indices,
        total_chunks_ptr,
    )

    h = k.new_empty(upper_bound_chunks, H, V_dim, K_dim)
    final_state = torch.empty_like(state, dtype=torch.float32)
    v_new = q.new_empty(pad_T, H, V_dim)
    v_new_chunks = v_new.view(upper_bound_chunks, BT, H, V_dim)

    profiler = None
    mod.h_v2b(
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
