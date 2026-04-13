# chunk_v7e: v7b with fused H+O kernel (ho_v1)
# The ho_v1 kernel runs the H recurrence AND computes o_inter in a second pass.
# The O kernel only needs to compute o_intra = A @ v_new and add to o_inter.

import os
from pathlib import Path
import torch
import triton
import triton.language as tl
from torch import Tensor

os.environ["TVM_FFI_CUDA_ARCH_LIST"] = "10.0a"
import tvm_ffi

CURRENT_DIR = Path(__file__).parent

# Build the fused HO kernel
ho_lib_path = tvm_ffi.cpp.build(
    name="gdn_prefill_cuda_ho",
    cuda_files=[
        str(CURRENT_DIR / "cuda_kkt_v1b.cu"),
        str(CURRENT_DIR / "cuda_ho_v1.cu"),
    ],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
    extra_ldflags=["-lcuda"],
)
ho_mod = tvm_ffi.load_module(ho_lib_path)

from .chunk_v6c import merge_16x16_to_64x64_inverse_kernel_v2
from .chunk_v7d import add_o_intra_kernel


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

    # KKT + metadata
    g_cu = torch.empty_like(a, dtype=torch.float32)
    beta = torch.empty_like(b, dtype=torch.float32)
    A = torch.empty(T, H, BT, device=k.device, dtype=torch.float32)
    ho_mod.kkt_v1b_with_meta(k, A_log, a, dt_bias, b, g_cu, beta, A, cu_seqlens, chunk_indices, chunk_offsets)

    # Inverse + W/U
    u = torch.empty_like(v)
    w = k.new_empty(T, H, K_dim)
    merge_16x16_to_64x64_inverse_kernel_v2[(upper_bound_chunks, H)](
        k, v, w, u, A, beta, g_cu, cu_seqlens, chunk_indices, total_chunks_ptr,
        H=H, Hg=Hg, K_dim=K_dim, V_dim=V_dim, BT=BT, num_warps=2)

    # Fused H + o_inter computation
    h = k.new_empty(upper_bound_chunks, H, V_dim, K_dim)
    final_state = torch.empty_like(state, dtype=torch.float32)
    v_new = q.new_empty(upper_bound_chunks, BT, H, V_dim)
    o_inter = torch.empty(T, H, V_dim, device=k.device, dtype=torch.float32)

    ho_mod.ho_v1(
        q, k, u, w, v_new, g_cu, h, state, final_state,
        o_inter, cu_seqlens, chunk_offsets, scale,
    )

    # O kernel: only o_intra = A @ v_new, added to o_inter
    o = torch.empty_like(v)
    BV = 64
    add_o_intra_kernel[(upper_bound_chunks, H)](
        q, k, v_new, g_cu, o_inter, o,
        cu_seqlens, chunk_indices, total_chunks_ptr,
        scale=scale, H=H, Hg=Hg, K_dim=K_dim, V_dim=V_dim, BT=BT, BV=BV, num_warps=4,
    )

    return o, final_state
