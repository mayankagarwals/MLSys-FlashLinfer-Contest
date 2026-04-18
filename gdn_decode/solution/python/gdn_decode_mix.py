import ctypes

ctypes.CDLL("libcudart.so", mode=ctypes.RTLD_GLOBAL)

import torch

from .gdn_decode_baseline import run as baseline_run
from .gdn_decode_cuda_kernel_7 import run as cuda_kernel_7_run
from .gdn_decode_kernel_large import run as large_kernel_run
from .gdn_decode_kernel_b32_pipe2 import run as b32_kernel_run


def run(q, k, v, state, A_log, a, dt_bias, b, scale):
    B = q.shape[0]

    if B >= 48:
        output = torch.empty_like(v)
        new_state = torch.empty_like(state)
        large_kernel_run(
            q, k, v, state, A_log, a, dt_bias, b, scale, output, new_state,
        )
        return output, new_state

    if B >= 16:
        output = torch.empty_like(v)
        new_state = torch.empty_like(state)
        b32_kernel_run(
            q, k, v, state, A_log, a, dt_bias, b, scale, output, new_state,
        )
        return output, new_state

    if B >= 8:
        return baseline_run(q, k, v, state, A_log, a, dt_bias, b, scale)

    output = torch.empty_like(v)
    new_state = torch.empty_like(state)
    cuda_kernel_7_run(
        q, k, v, state, A_log, a, dt_bias, b, scale, output, new_state,
    )
    return output, new_state
