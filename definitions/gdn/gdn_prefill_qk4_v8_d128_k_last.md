# gdn_prefill_qk4_v8_d128_k_last

## Summary

- Source JSON: `gdn/gdn_prefill_qk4_v8_d128_k_last.json`
- Operation Type: `gdn`
- Description: Gated Delta Net prefill with GVA configuration and k-last state layout. The state is in k-last layout [N, H, V, K]. Captured from Qwen3 Next linear attention layers (TP=4).
- Tags: `stage:prefill`, `status:verified`, `model:qwen3-next`, `layout:k-last`

## Axes

| Axis | Type | Value | Description |
| --- | --- | --- | --- |
| `total_seq_len` | `var` | `-` |  |
| `num_seqs` | `var` | `-` |  |
| `num_q_heads` | `const` | `4` | Number of query heads (same as key heads in GVA mode, TP=4, 16/4=4). |
| `num_k_heads` | `const` | `4` | Number of key heads (TP=4, 16/4=4). |
| `num_v_heads` | `const` | `8` | Number of value heads (GVA: more value heads than query heads, TP=4, 32/4=8). |
| `head_size` | `const` | `128` |  |
| `len_cu_seqlens` | `var` | `-` | Length of cu_seqlens array (num_seqs + 1). |

## Constraints

- `len_cu_seqlens == num_seqs + 1`
- `total_seq_len == cu_seqlens[-1].item()`

## Inputs

| Name | Shape | Dtype | Optional | Description |
| --- | --- | --- | --- | --- |
| `q` | `["total_seq_len", "num_q_heads", "head_size"]` | `bfloat16` | `no` | Query tensor. |
| `k` | `["total_seq_len", "num_k_heads", "head_size"]` | `bfloat16` | `no` | Key tensor. |
| `v` | `["total_seq_len", "num_v_heads", "head_size"]` | `bfloat16` | `no` | Value tensor. |
| `state` | `["num_seqs", "num_v_heads", "head_size", "head_size"]` | `float32` | `yes` | Recurrent state in k-last layout [N, H, V, K]. |
| `A_log` | `["num_v_heads"]` | `float32` | `no` | Log decay parameter (learnable). Used to compute g = exp(-exp(A_log) * softplus(a + dt_bias)). |
| `a` | `["total_seq_len", "num_v_heads"]` | `bfloat16` | `no` | Input-dependent decay from projection. |
| `dt_bias` | `["num_v_heads"]` | `float32` | `no` | Decay bias (learnable). Added to 'a' before softplus. |
| `b` | `["total_seq_len", "num_v_heads"]` | `bfloat16` | `no` | Update gate input from projection. beta = sigmoid(b). |
| `cu_seqlens` | `["len_cu_seqlens"]` | `int64` | `no` | Cumulative sequence lengths for variable-length batching. |
| `scale` | `scalar` | `float32` | `no` | Scale factor. Default is 1/sqrt(head_size). |

## Outputs

| Name | Shape | Dtype | Optional | Description |
| --- | --- | --- | --- | --- |
| `output` | `["total_seq_len", "num_v_heads", "head_size"]` | `bfloat16` | `no` | Attention output. Shape follows num_v_heads in GVA mode. |
| `new_state` | `["num_seqs", "num_v_heads", "head_size", "head_size"]` | `float32` | `no` | Updated recurrent state in k-last layout [N, H, V, K]. |

## Reference

```python
import math
import torch
import torch.nn.functional as F


def matmul(a: torch.Tensor, b: torch.Tensor):
    """Float32 matmul for numerical stability."""
    return a.float() @ b.float()


@torch.no_grad()
def run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    """
    Gated Delta Net prefill reference implementation (k-last layout).
    
    State layout: [H, V, K] (k-last, K dimension at the end)
    
    Gate computation:
    g = exp(-exp(A_log) * softplus(a + dt_bias))
    beta = sigmoid(b)
    
    Delta rule update:
    state_new = g * state_old + k^T @ (beta * v + (1-beta) * k @ state_old) - k^T @ (k @ state_old)
    output = scale * q @ state_new
    """
    total_seq_len, num_q_heads, head_size = q.shape
    num_v_heads = v.shape[1]
    num_k_heads = k.shape[1]
    num_sab_heads = max(num_q_heads, num_v_heads)
    num_seqs = cu_seqlens.size(0) - 1
    device = q.device

    assert num_q_heads == 4
    assert num_k_heads == 4
    assert num_v_heads == 8
    assert head_size == 128

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(head_size)

    # Compute g and beta from raw parameters
    x = a.float() + dt_bias.float()  # [total_seq_len, HV]
    g = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))  # [total_seq_len, HV]
    beta = torch.sigmoid(b.float())  # [total_seq_len, HV]

    q_exp = q.repeat_interleave(num_v_heads // num_q_heads, dim=1)
    k_exp = k.repeat_interleave(num_v_heads // num_k_heads, dim=1)

    output = torch.zeros(
        (total_seq_len, num_sab_heads, head_size), dtype=torch.bfloat16, device=device
    )
    new_state = torch.zeros(
        (num_seqs, num_sab_heads, head_size, head_size), dtype=torch.float32, device=device
    )

    for seq_idx in range(num_seqs):
        seq_start = int(cu_seqlens[seq_idx].item())
        seq_end = int(cu_seqlens[seq_idx + 1].item())
        seq_len = seq_end - seq_start

        if seq_len <= 0:
            continue

        if state is not None:
            state_HKV = state[seq_idx].clone().float().transpose(-1, -2)  # [H,V,K] -> [H,K,V]
        else:
            state_HKV = torch.zeros(
                (num_sab_heads, head_size, head_size), dtype=torch.float32, device=device
            )

        for i in range(seq_len):
            t = seq_start + i
            q_H1K = q_exp[t].unsqueeze(1).float()
            k_H1K = k_exp[t].unsqueeze(1).float()
            v_H1V = v[t].unsqueeze(1).float()
            g_H11 = g[t].unsqueeze(1).unsqueeze(2)
            beta_H11 = beta[t].unsqueeze(1).unsqueeze(2)

            old_state_HKV = g_H11 * state_HKV
            old_v_H1V = matmul(k_H1K, old_state_HKV)
            new_v_H1V = beta_H11 * v_H1V + (1 - beta_H11) * old_v_H1V
            state_remove = torch.einsum('hkl,hlv->hkv', k_H1K.transpose(-1, -2), old_v_H1V)
            state_update = torch.einsum('hkl,hlv->hkv', k_H1K.transpose(-1, -2), new_v_H1V)
            state_HKV = old_state_HKV - state_remove + state_update

            o_H1V = scale * matmul(q_H1K, state_HKV)
            output[t] = o_H1V.squeeze(1).to(torch.bfloat16)

        new_state[seq_idx] = state_HKV.transpose(-1, -2)  # [H,K,V] -> [H,V,K]

    return output, new_state
```
