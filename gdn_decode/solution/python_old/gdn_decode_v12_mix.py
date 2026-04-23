"""
gdn_decode_v12: same structure as v10 but uses baseline with tuned num_blocks_per_state for B=8 and above.

Dispatch:
  B < 8  → v7 CUDA  (small batch, same as v10)
  B >= 8 → FlashInfer CuTe baseline  with tuned num_blocks_per_state

Changes in v12 vs v11_1
  1. Use FlashInfer CuTe baseline with tuned num_blocks_per_state
"""
import ctypes

# get weird errors without this
ctypes.CDLL("libcudart.so", mode=ctypes.RTLD_GLOBAL)


import torch

from .gdn_decode_baseline import run as baseline_run
from .gdn_decode_cuda_kernel_7 import run as cuda_kernel_7_run
from .gdn_decode_cuda_kernel_11_1 import run as cuda_kernel_11_1_run


def run(q, k, v, state, A_log, a, dt_bias, b, scale):
    B = q.shape[0]

    if B >= 8:
        # CuTe baseline with tuned num_blocks_per_state
        return baseline_run(q, k, v, state, A_log, a, dt_bias, b, scale)

    # B < 8: v7 small-batch kernel (same as v10)
    output = torch.empty_like(v)
    new_state = torch.empty_like(state)
    cuda_kernel_7_run(
        q, k, v, state, A_log, a, dt_bias, b, scale, output, new_state,
    )
    return output, new_state
