from torch import Tensor

import ctypes

# get weird errors without this
ctypes.CDLL("libcudart.so", mode=ctypes.RTLD_GLOBAL)


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
    T = q.shape[0]

    # On the official T>=525 slice, ov1 is the better O-kernel in the large-T regime,
    # while ov2 wins in the smaller midrange window that we route here from mix.
    if T >= 3999:
        from .chunk_v9_ov1 import run as chunk_v9_ov1

        return chunk_v9_ov1(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)

    from .chunk_v9_ov2 import run as chunk_v9_ov2

    return chunk_v9_ov2(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)
