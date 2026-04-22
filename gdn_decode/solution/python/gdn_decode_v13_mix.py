"""
gdn_decode_v13: same as v12 but uses small batch kernel that directly read/write to GMEM for B<8.

Dispatch:
  B < 8  → FlashInfer CuTe small batch kernel that directly read/write to GMEM
  B >= 8 → FlashInfer CuTe baseline  with tuned num_blocks_per_state

Changes in v13 vs v12
  1. Use FlashInfer CuTe small batch kernel
  2. Tune the baseline FlashInfer CuTe kernel
"""
import ctypes

# get weird errors without this
ctypes.CDLL("libcudart.so", mode=ctypes.RTLD_GLOBAL)


from .gdn_decode_baseline import run as baseline_run
from .gdn_decode_v13_asm_small_batch import run as v13_asm_small_batch_run


def run(q, k, v, state, A_log, a, dt_bias, b, scale):
    B = q.shape[0]

    if B >= 8:
        # CuTe baseline with tuned num_blocks_per_state
        return baseline_run(q, k, v, state, A_log, a, dt_bias, b, scale)
    return v13_asm_small_batch_run(q, k, v, state, A_log, a, dt_bias, b, scale)
