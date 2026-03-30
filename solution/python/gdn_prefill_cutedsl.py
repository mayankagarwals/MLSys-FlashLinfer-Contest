import torch
import torch.nn.functional as F

from .gdn import chunk_gated_delta_rule


def run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    x = a.float() + dt_bias.float()
    # Gate values in log-space: log(exp(-exp(A_log) * softplus(x))) = -exp(A_log) * softplus(x)
    g = -torch.exp(A_log.float()) * F.softplus(x)
    beta = torch.sigmoid(b.float())

    varlen = cu_seqlens is not None and q.dim() == 3
    if varlen:
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)

    output, new_state = chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=None,
        initial_state=state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
    )

    if varlen:
        output = output.squeeze(0)

    return output, new_state
