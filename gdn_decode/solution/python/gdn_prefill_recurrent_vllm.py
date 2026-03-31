import math

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fla.ops.fused_recurrent import (
    fused_recurrent_gated_delta_rule,
)


@torch.no_grad()
def run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    T, Hq, Kdim = q.shape
    _, Hk, _ = k.shape
    _, Hv, Vdim = v.shape
    N = cu_seqlens.numel() - 1

    assert Hq == 4
    assert Hk == 4
    assert Hv == 8
    assert Kdim == 128
    assert Vdim == 128
    assert a.shape == (T, Hv)
    assert b.shape == (T, Hv)
    assert A_log.shape == (Hv,)
    assert dt_bias.shape == (Hv,)
    assert int(cu_seqlens[-1].item()) == T

    if state is not None:
        assert state.shape == (N, Hv, Vdim, Kdim)

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(Kdim)

    x = a.float() + dt_bias.float().unsqueeze(0)
    g = -torch.exp(A_log.float()).unsqueeze(0) * F.softplus(x)
    beta = torch.sigmoid(b.float())
    cu_seqlens = cu_seqlens.to(dtype=torch.long, device=q.device).contiguous()

    if state is None:
        initial_state = torch.zeros(
            (N, Hv, Vdim, Kdim), dtype=torch.float32, device=q.device
        )
    else:
        initial_state = state.contiguous().clone()

    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    max_seq_len = int(seq_lens.max().item()) if N > 0 else 0
    ssm_state_indices = torch.full(
        (N, max_seq_len), -1, dtype=torch.long, device=q.device
    )
    nonempty = seq_lens > 0
    if nonempty.any():
        seq_ids = torch.arange(N, device=q.device, dtype=torch.long)[nonempty]
        ssm_state_indices[nonempty, 0] = seq_ids
        ssm_state_indices[nonempty, seq_lens[nonempty] - 1] = seq_ids

    output, final_state = fused_recurrent_gated_delta_rule(
        q=q.unsqueeze(0),
        k=k.unsqueeze(0),
        v=v.unsqueeze(0),
        g=g.unsqueeze(0),
        beta=beta.unsqueeze(0),
        scale=scale,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        inplace_final_state=True,
        use_qk_l2norm_in_kernel=False,
    )

    output = output.squeeze(0).to(torch.bfloat16)
    new_state = final_state.float()
    if (seq_lens <= 0).any():
        new_state = new_state.clone()
        new_state[seq_lens <= 0] = 0

    return output, new_state
