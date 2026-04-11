"""
gdn_decode_v11_mix: same structure as v10 but uses v11_1 for B=48 and B=64.

Dispatch:
  B < 8       → v7 CUDA  (small batch, same as v10)
  8 <= B < 48 → FlashInfer CuTe baseline (same as v10)
  B >= 48     → v11_1 CUDA (new: tanh sigmoid, write-through, all-lanes v_scalar/g/beta)

Changes in v11_1 vs v9 (the kernel used by v10 for B>=48):
  1. Sigmoid: expf-based → tanh.approx.f32  (SM75+, 2^-11 max err, ~2x faster)
  2. q/k loads: removed L1::evict_first hint (better L1 reuse)
  3. new_state stores: st.global.v4.f32 → st.global.wt.v4.f32 (write-through, frees L2)
  4. g/beta: lane-0 compute + 2 shfl_sync → all-lanes compute (elim 2 broadcasts/CTA)
  5. v_scalar: lane-0 load + shfl_sync → all-lanes uniform read (elim 1 broadcast/iter)
"""
import torch

from .gdn_decode_baseline import run as baseline_run
from .gdn_decode_cuda_kernel_7 import run as cuda_kernel_7_run
from .gdn_decode_cuda_kernel_11_1 import run as cuda_kernel_11_1_run


def run(q, k, v, state, A_log, a, dt_bias, b, scale):
    B = q.shape[0]

    if B >= 48:
        # v11_1: our optimized pipelined kernel (beats v9 by ~3-5% for B=48,64)
        output = torch.empty_like(v)
        new_state = torch.empty_like(state)
        cuda_kernel_11_1_run(
            q, k, v, state, A_log, a, dt_bias, b, scale, output, new_state,
        )
        return output, new_state

    if B >= 8:
        # CuTe baseline (same as v10, best for B=16,32)
        return baseline_run(q, k, v, state, A_log, a, dt_bias, b, scale)

    # B < 8: v7 small-batch kernel (same as v10)
    output = torch.empty_like(v)
    new_state = torch.empty_like(state)
    cuda_kernel_7_run(
        q, k, v, state, A_log, a, dt_bias, b, scale, output, new_state,
    )
    return output, new_state
