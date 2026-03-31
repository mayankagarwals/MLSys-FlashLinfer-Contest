# modified from https://github.com/fla-org/flash-linear-attention/blob/v0.4.1/fla/ops/gated_delta_rule/fused_recurrent.py
# convert V-last layout to K-last layout.
# we also remove unnecessary code under the challenge's constraints.
#
# [build]
# language = "triton"
# entry_point = "gdn_decode_v1.py::run"
# destination_passing_style = false

import torch
import triton
import triton.language as tl


@triton.jit
def softplus(x):
    return tl.math.log(1 + tl.math.exp(x))


@triton.jit
def kernel(
    q,  # [B, 1, H, K]
    k,  # [B, 1, H, K]
    v,  # [B, 1, HV, V]
    A_log,  # [HV]
    a,  # [B, 1, HV]
    dt_bias,  # [HV]
    b,  # [B, 1, HV]
    o,  # [B, 1, HV, V]
    h0,  # [B, HV, V, K]
    ht,  # [B, HV, V, K]
    scale,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_hv = tl.program_id(1)
    i_n = tl.program_id(2)
    i_h = i_hv // (HV // H)
    i_nh = i_n * HV + i_hv

    bos = i_n
    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v

    b_A_neg = -tl.math.exp(tl.load(A_log + i_hv).to(tl.float32))
    b_dt_bias = tl.load(dt_bias + i_hv).to(tl.float32)

    p_a = a + bos * HV + i_hv
    p_b = b + bos * HV + i_hv
    p_o = o + (bos * HV + i_hv) * V + o_v

    p_h0 = h0 + i_nh * K * V + o_v[:, None] * K + o_k[None, :]  # [BV, BK]
    b_h = tl.load(p_h0).to(tl.float32)

    b_q = tl.load(p_q).to(tl.float32)  # [BK]
    b_k = tl.load(p_k).to(tl.float32)  # [BK]
    b_v = tl.load(p_v).to(tl.float32)  # [BV]

    b_q = b_q * scale
    b_beta = tl.sigmoid(tl.load(p_b).to(tl.float32))

    # apply gating
    b_x = tl.load(p_a).to(tl.float32) + b_dt_bias
    b_g = tl.math.exp(b_A_neg * softplus(b_x))
    b_h *= b_g

    b_v = b_beta * (b_v - tl.sum(b_h * b_k[None, :], 1))
    b_h += b_k[None, :] * b_v[:, None]

    # [BV]
    b_o = tl.sum(b_h * b_q[None, :], 1)
    tl.store(p_o, b_o)

    p_ht = ht + i_nh * K * V + o_v[:, None] * K + o_k[None, :]
    tl.store(p_ht, b_h.to(p_ht.dtype.element_ty))


def run(q, k, v, state, A_log, a, dt_bias, b, scale):
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]

    o = torch.empty_like(v)
    final_state = q.new_empty(B, HV, V, K, dtype=torch.float32)

    BV = 4
    NV = triton.cdiv(V, BV)

    grid = (NV, HV, B)
    kernel[grid](
        q=q,
        k=k,
        v=v,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        o=o,
        h0=state,
        ht=final_state,
        scale=scale,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BV=BV,
        num_warps=1,  # improve occupancy?
    )
    return o, final_state


if __name__ == "__main__":
    B, T, H, HV, K, V = 1, 1, 4, 8, 128, 128
    q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, T, HV, V, dtype=torch.bfloat16, device="cuda")
    state = torch.randn(B, HV, V, K, dtype=torch.float32, device="cuda")
    A_log = torch.randn(HV, dtype=torch.float32, device="cuda")
    a = torch.randn(B, T, HV, dtype=torch.bfloat16, device="cuda")
    dt_bias = torch.randn(HV, dtype=torch.float32, device="cuda")
    b = torch.randn(B, T, HV, dtype=torch.bfloat16, device="cuda")
    scale = K**-0.5

    run(q, k, v, state, A_log, a, dt_bias, b, scale)
