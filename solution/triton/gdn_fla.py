# modified from https://github.com/fla-org/flash-linear-attention/blob/v0.4.1/fla/ops/gated_delta_rule/fused_recurrent.py
# convert V-last layout to K-last layout.
# we also remove unnecessary code under the challenge's constraints.
#
# [build]
# language = "triton"
# entry_point = "gdn_fla.py::decode"
# destination_passing_style = false

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice


@triton.jit
def fused_recurrent_gated_delta_rule_fwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    h0,
    ht,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    bos, eos = i_n * T, i_n * T + T
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v

    p_g = g + bos * HV + i_hv
    if IS_BETA_HEADWISE:
        p_beta = beta + bos * HV + i_hv
    else:
        p_beta = beta + (bos * HV + i_hv) * V + o_v

    p_o = o + (bos * HV + i_hv) * V + o_v

    p_h0 = h0 + i_nh * K * V + o_v[:, None] * K + o_k[None, :]  # [BV, BK]
    b_h = tl.load(p_h0).to(tl.float32)

    for _ in range(0, T):
        b_q = tl.load(p_q).to(tl.float32)  # [BK]
        b_k = tl.load(p_k).to(tl.float32)  # [BK]
        b_v = tl.load(p_v).to(tl.float32)  # [BV]

        b_q = b_q * scale
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta).to(tl.float32)
        else:
            # b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
            b_beta = tl.load(p_beta).to(tl.float32)

        # [BK, BV]
        b_g = tl.load(p_g).to(tl.float32)
        b_h *= tldevice.fast_expf(b_g)

        b_v = b_beta * (b_v - tl.sum(b_h * b_k[None, :], 1))
        b_h += b_k[None, :] * b_v[:, None]

        # [BV]
        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty))

        p_q += H * K
        p_k += H * K
        p_v += HV * V

        p_g += HV
        p_beta += HV * (1 if IS_BETA_HEADWISE else V)
        p_o += HV * V

    p_ht = ht + i_nh * K * V + o_v[:, None] * K + o_k[None, :]
    tl.store(p_ht, b_h.to(p_ht.dtype.element_ty))


def fused_recurrent_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B
    BK = triton.next_power_of_2(K)
    BV = min(8, triton.next_power_of_2(V))
    NV = triton.cdiv(V, BV)

    o = torch.empty_like(v)
    final_state = q.new_empty(N, HV, V, K, dtype=torch.float32)

    grid = (NV, N * HV)
    fused_recurrent_gated_delta_rule_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        o=o,
        h0=initial_state,
        ht=final_state,
        scale=scale,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        IS_BETA_HEADWISE=beta.ndim != v.ndim,
        num_warps=1,
        num_stages=3,
    )
    return o, final_state


def decode(q, k, v, state, A_log, a, dt_bias, b, scale):
    x = a.float() + dt_bias.float()  # [B, 1, HV]
    g = -torch.exp(A_log.float()) * F.softplus(x)  # [B, 1, HV]
    beta = torch.sigmoid(b.float())  # [B, 1, HV]
    return fused_recurrent_gated_delta_rule_fwd(
        q,
        k,
        v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=state,
    )


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

    decode(q, k, v, state, A_log, a, dt_bias, b, scale)
