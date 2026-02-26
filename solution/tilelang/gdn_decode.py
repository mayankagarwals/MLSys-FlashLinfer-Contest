import math
from typing import Optional, Tuple, Union

import torch
import tilelang
import tilelang.language as T

_BV = 8


def _resolve_scale(scale: Optional[Union[float, torch.Tensor]], k: int) -> float:
    if scale is None:
        return 1.0 / math.sqrt(k)
    if isinstance(scale, torch.Tensor):
        if scale.numel() != 1:
            raise ValueError(
                f"Expected scalar scale tensor, got shape={tuple(scale.shape)}"
            )
        scale = float(scale.item())
    else:
        scale = float(scale)
    if scale == 0.0:
        return 1.0 / math.sqrt(k)
    return scale


@tilelang.jit(out_idx=[-2, -1], pass_configs={"tl.disable_tma_lower": True})
def gdn_decode(B, H, HV, K, V, BV, scale):
    in_dtype = "bfloat16"
    acc_dtype = "float32"
    out_dtype = "bfloat16"
    group_size = HV // H
    NV = T.ceildiv(V, BV)

    @T.prim_func
    def main(
        q: T.Tensor([B, 1, H, K], in_dtype),  # type: ignore
        k: T.Tensor([B, 1, H, K], in_dtype),  # type: ignore
        v: T.Tensor([B, 1, HV, V], in_dtype),  # type: ignore
        h0: T.Tensor([B, HV, V, K], acc_dtype),  # type: ignore
        A_log: T.Tensor([HV], acc_dtype),  # type: ignore
        a: T.Tensor([B, 1, HV], in_dtype),  # type: ignore
        dt_bias: T.Tensor([HV], acc_dtype),  # type: ignore
        b: T.Tensor([B, 1, HV], in_dtype),  # type: ignore
        o: T.Tensor([B, 1, HV, V], out_dtype),  # type: ignore
        ht: T.Tensor([B, HV, V, K], acc_dtype),  # type: ignore
    ):
        with T.Kernel(NV, HV, B, threads=128) as (i_v, i_hv, i_b):
            i_h = i_hv // group_size

            q_local = T.alloc_fragment([K], acc_dtype)
            k_local = T.alloc_fragment([K], acc_dtype)
            v_local = T.alloc_fragment([BV], acc_dtype)
            h_local = T.alloc_fragment([BV, K], acc_dtype)
            hk_prod = T.alloc_fragment([BV, K], acc_dtype)
            hq_prod = T.alloc_fragment([BV, K], acc_dtype)
            hk_sum = T.alloc_fragment([BV], acc_dtype)
            o_local = T.alloc_fragment([BV], acc_dtype)

            for kk in T.Parallel(K):
                q_local[kk] = q[i_b, 0, i_h, kk].astype(acc_dtype) * scale
                k_local[kk] = k[i_b, 0, i_h, kk].astype(acc_dtype)

            for vv in T.Parallel(BV):
                v_idx = i_v * BV + vv
                v_local[vv] = v[i_b, 0, i_hv, v_idx].astype(acc_dtype)

            for vv, kk in T.Parallel(BV, K):
                v_idx = i_v * BV + vv
                h_local[vv, kk] = h0[i_b, i_hv, v_idx, kk]

            A_neg = -T.exp(A_log[i_hv])
            x = a[i_b, 0, i_hv].astype(acc_dtype) + dt_bias[i_hv]
            g = T.exp(A_neg * T.log(1.0 + T.exp(x)))
            beta = T.sigmoid(b[i_b, 0, i_hv].astype(acc_dtype))

            for vv, kk in T.Parallel(BV, K):
                h_local[vv, kk] = h_local[vv, kk] * g
                hk_prod[vv, kk] = h_local[vv, kk] * k_local[kk]
            T.reduce_sum(hk_prod, hk_sum, dim=1)

            for vv in T.Parallel(BV):
                v_local[vv] = beta * (v_local[vv] - hk_sum[vv])

            for vv, kk in T.Parallel(BV, K):
                h_local[vv, kk] = h_local[vv, kk] + k_local[kk] * v_local[vv]
                hq_prod[vv, kk] = h_local[vv, kk] * q_local[kk]
            T.reduce_sum(hq_prod, o_local, dim=1)

            for vv in T.Parallel(BV):
                v_idx = i_v * BV + vv
                o[i_b, 0, i_hv, v_idx] = o_local[vv].astype(out_dtype)

            for vv, kk in T.Parallel(BV, K):
                v_idx = i_v * BV + vv
                ht[i_b, i_hv, v_idx, kk] = h_local[vv, kk]

    return main


@torch.no_grad()
def run(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: Optional[torch.Tensor],
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    scale: Optional[Union[float, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K = q.shape
    _, _, HV, V = v.shape

    if T != 1:
        raise ValueError(f"Decode expects sequence length T=1, got T={T}")
    if HV % H != 0:
        raise ValueError(f"Expected HV % H == 0, got HV={HV}, H={H}")
    if V % _BV != 0:
        raise ValueError(f"Expected V divisible by {_BV}, got V={V}")

    scale_value = _resolve_scale(scale, K)

    kernel = gdn_decode(B, H, HV, K, V, _BV, scale_value)
    output, new_state = kernel(q, k, v, state, A_log, a, dt_bias, b)
    return output, new_state
