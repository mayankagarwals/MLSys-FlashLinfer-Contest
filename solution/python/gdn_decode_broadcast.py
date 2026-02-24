import math
import torch
import torch.nn.functional as F


@torch.compile(mode="max-autotune-no-cudagraphs", dynamic=False)
@torch.no_grad()
def run(q, k, v, state, A_log, a, dt_bias, b, scale):
    """
    Gated Delta Net decode reference implementation (k-last layout).

    State layout: [B, H, V, K] (k-last, K dimension at the end)

    Gate computation:
    g = exp(-exp(A_log) * softplus(a + dt_bias))
    beta = sigmoid(b)

    Delta rule update:
    state_new = g * state_old + k^T @ (beta * v + (1-beta) * k @ state_old) - k^T @ (k @ state_old)
              = g * state_old + k^T @ (beta * (v - k @ state_old))
    output = scale * q @ state_new
    """
    B, T, num_q_heads, K = q.shape
    _, _, num_k_heads, _ = k.shape
    _, _, num_v_heads, V = v.shape
    num_heads = num_v_heads
    device = q.device

    assert K == 128 and V == 128
    assert T == 1
    assert num_v_heads % num_q_heads == 0
    assert num_v_heads % num_k_heads == 0

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(K)

    # Compute g and beta from raw parameters
    x = a.float() + dt_bias.float()  # [B, 1, HV]
    g = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))  # [B, 1, HV]
    beta = torch.sigmoid(b.float())  # [B, 1, HV]

    q_f32 = q.squeeze(1).float() * scale  # [B, Hq, K]
    k_f32 = k.squeeze(1).float()  # [B, Hk, K]
    v_f32 = v.squeeze(1).float()  # [B, Hv, V]
    g_f32 = g.squeeze(1).float()  # [B, Hv]
    beta_f32 = beta.squeeze(1).float()  # [B, Hv]

    if state is not None:
        state_f32 = state.float()  # [B, Hv, V, K]
    else:
        state_f32 = torch.zeros(B, num_heads, V, K, dtype=torch.float32, device=device)

    # Expand Q/K heads to V-head granularity for GVA-style setups.
    q_exp = q_f32.repeat_interleave(num_v_heads // num_q_heads, dim=1)
    k_exp = k_f32.repeat_interleave(num_v_heads // num_k_heads, dim=1)

    # apply gating to the old state
    old_state = state_f32 * g_f32[:, :, None, None]  # [B, Hv, V, K]

    # predict v from k using the old state
    old_v = (k_exp[:, :, None, :] @ old_state.transpose(2, 3)).squeeze(2)  # [B, Hv, V]

    # update the state
    v_diff = (v_f32 - old_v) * beta_f32[:, :, None]  #[B, Hv, V]
    new_state = old_state + v_diff[:, :, :, None] * k_exp[:, :, None, :]  # [B, Hv, V, K]

    # project q using the new state
    output = (q_exp[:, :, None, :] @ new_state.transpose(2, 3)).squeeze(2)

    output = output.unsqueeze(1).to(torch.bfloat16)  # [B,1,Hv,V]
    return output, new_state
