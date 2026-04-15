import os
from pathlib import Path

import torch
import triton
import tvm_ffi
from torch import Tensor

from . import chunk_v8 as _chunk_v8

os.environ["TVM_FFI_CUDA_ARCH_LIST"] = "10.0a"

CURRENT_DIR = Path(__file__).parent

lib_path = tvm_ffi.cpp.build(
    name="gdn_prefill_chunk_v9_ov2_cuda_o_v2_v4",
    cuda_files=[str(CURRENT_DIR / "cuda_o_v2.cu")],
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
    T, Hg, K_dim = k.shape
    N, H, V_dim, _ = state.shape

    BT = 64

    upper_bound_chunks = (N - 1) + triton.cdiv(T - (N - 1), BT)
    chunk_offsets = q.new_empty(N + 1, dtype=torch.int32)
    chunk_indices = q.new_empty((upper_bound_chunks, 2), dtype=torch.int32)
    total_chunks_ptr = chunk_offsets[N:]

    pad_T = upper_bound_chunks * BT

    _chunk_v8.mod.prep_meta(cu_seqlens, chunk_indices, chunk_offsets)

    g_cu = torch.empty_like(a, dtype=torch.float32)
    beta = torch.empty_like(b, dtype=torch.float32)
    A = torch.empty(T, H, BT, device=k.device, dtype=torch.float32)
    _chunk_v8.mod.kkt_v1b(
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

    u = torch.empty_like(v)
    w = k.new_empty(T, H, K_dim)
    _chunk_v8.merge_16x16_to_64x64_inverse_kernel_v2[(upper_bound_chunks, H)](
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
    v_new = v.new_empty(pad_T, H, V_dim)
    v_new_chunks = v_new.view(upper_bound_chunks, BT, H, V_dim)

    profiler = None
    _chunk_v8.mod.h_v1(
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
    mod.o_v2(
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
