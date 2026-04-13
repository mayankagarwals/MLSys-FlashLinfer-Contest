# chunk_v9: kkt + inverse + fused H+O (Triton, BV=64, inline A_causal)
import os
from pathlib import Path
import torch, triton
from torch import Tensor
os.environ["TVM_FFI_CUDA_ARCH_LIST"] = "10.0a"
import tvm_ffi
CURRENT_DIR = Path(__file__).parent
from .chunk_v6c import merge_16x16_to_64x64_inverse_kernel_v2, mod as kkt_mod
from .triton_fused_ho import fused_ho_kernel_v2 as fused_ho_kernel

def run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    T, Hg, K_dim = k.shape; N, H, V_dim, _ = state.shape; BT = 64
    ub = (N - 1) + triton.cdiv(T - (N - 1), BT)
    co = q.new_empty(N + 1, dtype=torch.int32)
    ci = q.new_empty((ub, 2), dtype=torch.int32)
    tc = co[N:]
    g_cu = torch.empty_like(a, dtype=torch.float32)
    beta = torch.empty_like(b, dtype=torch.float32)
    A = torch.empty(T, H, BT, device=k.device, dtype=torch.float32)
    kkt_mod.kkt_v1b_with_meta(k, A_log, a, dt_bias, b, g_cu, beta, A, cu_seqlens, ci, co)
    u = torch.empty_like(v); w = k.new_empty(T, H, K_dim)
    merge_16x16_to_64x64_inverse_kernel_v2[(ub, H)](
        k, v, w, u, A, beta, g_cu, cu_seqlens, ci, tc,
        H=H, Hg=Hg, K_dim=K_dim, V_dim=V_dim, BT=BT, num_warps=2)
    o = torch.empty_like(v)
    ns = torch.empty_like(state, dtype=torch.float32)
    BV = 64
    fused_ho_kernel[(N * H, V_dim // BV)](
        q, k, w, u, g_cu, state, cu_seqlens, o, ns, scale,
        H=H, Hg=Hg, K_dim=K_dim, V_dim=V_dim, BT=BT, BV=BV, num_warps=4)
    return o, ns
