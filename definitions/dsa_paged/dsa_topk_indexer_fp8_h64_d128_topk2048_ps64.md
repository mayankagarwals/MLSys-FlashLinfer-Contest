# dsa_topk_indexer_fp8_h64_d128_topk2048_ps64

## Summary

- Source JSON: `dsa_paged/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.json`
- Operation Type: `dsa_paged`
- Description: Native Sparse Attention (DSA) TopK indexer with FP8 quantization for DeepSeek-V3.2. Computes sparse attention scores using ReLU activation and learned weights, then selects top-K KV cache indices. Formula: sum(relu(q @ K.T) * weights). Matches SGLang/deep_gemm implementation. Page size 64 variant.
- Tags: `stage:indexer`, `status:verified`, `model:deepseek-v3.2`, `sparse:topk`, `quant:fp8`

## Axes

| Axis | Type | Value | Description |
| --- | --- | --- | --- |
| `batch_size` | `var` | `-` |  |
| `num_index_heads` | `const` | `64` | Number of indexer heads (64 required by deep_gemm). |
| `index_head_dim` | `const` | `128` | Indexer head dimension (matches deep_gemm requirement). |
| `page_size` | `const` | `64` | Page size for KV cache (64 tokens per page, required by deep_gemm). |
| `topk` | `const` | `2048` | Number of top-K indices to select. |
| `max_num_pages` | `var` | `-` | Maximum number of pages per sequence. |
| `num_pages` | `var` | `-` | Total number of allocated pages in the KV cache. |
| `kv_cache_num_heads` | `const` | `1` | Number of heads in KV cache (always 1 for deep_gemm MQA format). |
| `head_dim_with_scale` | `const` | `132` | Head dimension (128) + scale bytes (4) = 132 for deep_gemm FP8 format. |

## Constraints

- `topk <= max_num_pages * page_size`

## Inputs

| Name | Shape | Dtype | Optional | Description |
| --- | --- | --- | --- | --- |
| `q_index_fp8` | `["batch_size", "num_index_heads", "index_head_dim"]` | `float8_e4m3fn` | `no` | FP8 quantized query tensor for indexing. |
| `k_index_cache_fp8` | `["num_pages", "page_size", "kv_cache_num_heads", "head_dim_with_scale"]` | `int8` | `no` | FP8 quantized key index cache with embedded scale factors (deep_gemm format). Memory layout: all FP8 values first (page_size * 128 bytes), then all scale factors (page_size * 4 bytes). Reshaped to [num_pages, page_size, 1, 132]. Uses int8 dtype but should be interpreted as uint8. |
| `weights` | `["batch_size", "num_index_heads"]` | `float32` | `no` | Learned weights for combining heads. In SGLang: weights = weights_proj(x) * n_heads^-0.5 * q_scale * softmax_scale. |
| `seq_lens` | `["batch_size"]` | `int32` | `no` | Sequence lengths for each batch element. |
| `block_table` | `["batch_size", "max_num_pages"]` | `int32` | `no` | Page-level block table mapping batch to page indices. |

## Outputs

| Name | Shape | Dtype | Optional | Description |
| --- | --- | --- | --- | --- |
| `topk_indices` | `["batch_size", "topk"]` | `int32` | `no` | Top-K token indices for each batch element. Values of -1 indicate padding. |

## Reference

```python
import torch


def dequant_fp8_kv_cache(k_index_cache_fp8):
    """Dequantize FP8 KV cache from deep_gemm format.
    
    Input: [num_pages, page_size, 1, 132] int8 (interpreted as uint8)
           Memory layout (per page): [fp8_data (page_size * 128 bytes), scales (page_size * 4 bytes)]
           After view to [num_pages, page_size, 1, 132]: NOT directly indexable as [fp8, scale] per token!
    Output: [num_pages, page_size, 128] float32
    """
    # View as uint8 for correct byte interpretation
    k_index_cache_fp8 = k_index_cache_fp8.view(torch.uint8)
    num_pages, page_size, num_heads, head_dim_sf = k_index_cache_fp8.shape
    head_dim = head_dim_sf - 4  # 128
    
    # Go back to flat format to reverse the packing
    kv_flat = k_index_cache_fp8.view(num_pages, page_size * head_dim_sf)
    
    # FP8 part: first page_size * head_dim bytes
    fp8_bytes = kv_flat[:, :page_size * head_dim].contiguous()
    fp8_tensor = fp8_bytes.view(num_pages, page_size, head_dim).view(torch.float8_e4m3fn)
    fp8_float = fp8_tensor.to(torch.float32)
    
    # Scale part: last page_size * 4 bytes -> page_size float32 values
    scale_bytes = kv_flat[:, page_size * head_dim:].contiguous()
    scale = scale_bytes.view(num_pages, page_size, 4).view(torch.float32)  # [num_pages, page_size, 1]
    
    return fp8_float * scale


@torch.no_grad()
def run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table):
    batch_size, num_index_heads, index_head_dim = q_index_fp8.shape
    num_pages, page_size, _, _ = k_index_cache_fp8.shape
    topk = 2048

    # Check constants
    assert num_index_heads == 64
    assert index_head_dim == 128
    assert page_size == 64

    device = q_index_fp8.device

    # Dequantize inputs
    q = q_index_fp8.to(torch.float32)  # [batch, heads, head_dim]
    K_all = dequant_fp8_kv_cache(k_index_cache_fp8)  # [num_pages, page_size, head_dim]

    topk_indices = torch.full((batch_size, topk), -1, dtype=torch.int32, device=device)
    max_num_pages = block_table.shape[1]

    for b in range(batch_size):
        seq_len = int(seq_lens[b].item())
        
        if seq_len == 0:
            continue

        # Get pages for this sequence
        num_pages_for_seq = (seq_len + page_size - 1) // page_size
        page_indices = block_table[b, :num_pages_for_seq].to(torch.long)
        
        # Gather K from pages
        K_paged = K_all[page_indices]  # [num_pages_for_seq, page_size, head_dim]
        K = K_paged.reshape(-1, index_head_dim)[:seq_len]  # [seq_len, head_dim]
        
        # Query for this batch element
        q_b = q[b]  # [num_heads, head_dim]
        
        # Compute attention scores
        scores = q_b @ K.T  # [num_heads, seq_len]
        
        # Apply ReLU (deep_gemm uses ReLU activation)
        scores_relu = torch.relu(scores)  # [num_heads, seq_len]
        
        # Apply learned weights and sum across heads
        w = weights[b]  # [num_heads]
        weighted_scores = scores_relu * w[:, None]  # [num_heads, seq_len]
        final_scores = weighted_scores.sum(dim=0)  # [seq_len]
        
        # Select top-K
        actual_topk = min(topk, seq_len)
        _, topk_idx = torch.topk(final_scores, actual_topk)
        
        # Convert to global token indices
        # Token index = page_idx * page_size + offset_in_page
        page_idx_per_token = topk_idx // page_size
        offset_per_token = topk_idx % page_size
        global_page_idx = page_indices[page_idx_per_token]
        topk_tokens = global_page_idx * page_size + offset_per_token
        
        topk_indices[b, :actual_topk] = topk_tokens.to(torch.int32)

    return (topk_indices,)
```
