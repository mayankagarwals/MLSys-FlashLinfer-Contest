# dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64

## Summary

- Source JSON: `dsa_paged/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.json`
- Operation Type: `dsa_paged`
- Description: Batched Native Sparse Attention (DSA) with sparse TopK KV cache selection. Captured from DeepSeek-V3.2 with tensor parallel size 8. Uses sparse indexing to select only top-K KV cache entries for attention computation. Page size 64 variant. Works for both prefill and decode stages.
- Tags: `status:verified`, `model:deepseek-v3.2`, `sparse:topk`

## Axes

| Axis | Type | Value | Description |
| --- | --- | --- | --- |
| `num_tokens` | `var` | `-` | Number of tokens (batch_size for decode, total_num_tokens for prefill). |
| `num_qo_heads` | `const` | `16` | Number of query heads after tensor parallel split (128/8=16). |
| `head_dim_ckv` | `const` | `512` | Compressed KV head dimension. |
| `head_dim_kpe` | `const` | `64` | Key positional encoding dimension. |
| `page_size` | `const` | `64` | Page size for KV cache (64 tokens per page). |
| `topk` | `const` | `2048` | Number of top-K KV cache entries selected for sparse attention. |
| `num_pages` | `var` | `-` | Total number of allocated pages in the KV cache. |

## Constraints

- `sparse_indices.shape[0] == num_tokens`
- `sparse_indices.shape[-1] == topk`
- `ckv_cache.shape[1] == page_size`

## Inputs

| Name | Shape | Dtype | Optional | Description |
| --- | --- | --- | --- | --- |
| `q_nope` | `["num_tokens", "num_qo_heads", "head_dim_ckv"]` | `bfloat16` | `no` | Query tensor without positional encoding component. |
| `q_pe` | `["num_tokens", "num_qo_heads", "head_dim_kpe"]` | `bfloat16` | `no` | Query positional encoding component. |
| `ckv_cache` | `["num_pages", "page_size", "head_dim_ckv"]` | `bfloat16` | `no` | Compressed key-value cache with page_size=64. |
| `kpe_cache` | `["num_pages", "page_size", "head_dim_kpe"]` | `bfloat16` | `no` | Key positional encoding cache. |
| `sparse_indices` | `["num_tokens", "topk"]` | `int32` | `no` | Sparse indices selecting top-K KV cache entries for each token. Values of -1 indicate padding (invalid indices). For page_size=64, indices encode (page_idx * 64 + offset). |
| `sm_scale` | `scalar` | `float32` | `no` | Softmax scale. For MLA, uses pre-absorption head dimension: 1/sqrt(head_dim_qk + head_dim_kpe) = 1/sqrt(128 + 64) = 1/sqrt(192). |

## Outputs

| Name | Shape | Dtype | Optional | Description |
| --- | --- | --- | --- | --- |
| `output` | `["num_tokens", "num_qo_heads", "head_dim_ckv"]` | `bfloat16` | `no` |  |
| `lse` | `["num_tokens", "num_qo_heads"]` | `float32` | `no` | The 2-based log-sum-exp of attention logits. |

## Reference

```python
import math
import torch


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    num_pages, page_size, _ = ckv_cache.shape
    topk = sparse_indices.shape[-1]

    # Check constants
    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64
    assert page_size == 64
    assert topk == 2048

    # Check constraints
    assert sparse_indices.shape[0] == num_tokens
    assert sparse_indices.shape[-1] == topk
    assert ckv_cache.shape[1] == page_size

    device = q_nope.device

    # Flatten paged KV cache to token-level: [num_pages, page_size, dim] -> [num_pages * page_size, dim]
    Kc_all = ckv_cache.reshape(-1, head_dim_ckv).to(torch.float32)  # [total_kv_tokens, head_dim_ckv]
    Kp_all = kpe_cache.reshape(-1, head_dim_kpe).to(torch.float32)  # [total_kv_tokens, head_dim_kpe]

    output = torch.zeros(
        (num_tokens, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device
    )
    lse = torch.full((num_tokens, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

    for t in range(num_tokens):
        indices = sparse_indices[t]  # [topk]

        # Handle padding: -1 indicates invalid indices
        valid_mask = indices != -1
        valid_indices = indices[valid_mask]

        if valid_indices.numel() == 0:
            output[t].zero_()
            continue

        # For page_size=64, indices encode (page_idx * 64 + offset)
        tok_idx = valid_indices.to(torch.long)

        Kc = Kc_all[tok_idx]  # [num_valid, head_dim_ckv]
        Kp = Kp_all[tok_idx]  # [num_valid, head_dim_kpe]
        qn = q_nope[t].to(torch.float32)  # [num_qo_heads, head_dim_ckv]
        qp = q_pe[t].to(torch.float32)  # [num_qo_heads, head_dim_kpe]

        # Compute attention logits
        logits = (qn @ Kc.T) + (qp @ Kp.T)  # [num_qo_heads, num_valid]
        logits_scaled = logits * sm_scale

        # Compute 2-base LSE
        lse[t] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)

        # Compute attention output
        attn = torch.softmax(logits_scaled, dim=-1)  # [num_qo_heads, num_valid]
        out = attn @ Kc  # [num_qo_heads, head_dim_ckv]
        output[t] = out.to(torch.bfloat16)

    return output, lse
```
