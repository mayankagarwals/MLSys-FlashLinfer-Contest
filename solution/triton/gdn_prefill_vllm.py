# https://github.com/vllm-project/vllm/blob/v0.17.0/vllm/model_executor/layers/fla/ops/chunk.py

import torch
import torch.nn.functional as F
import triton
from torch import Tensor
from vllm.model_executor.layers.fla.ops.chunk import chunk_gated_delta_rule_fwd


def alloc_fn(size: int, alignment: int, stream: int | None):
    return torch.empty(size, device="cuda", dtype=torch.int8)


triton.set_allocator(alloc_fn)


def run(
    q: Tensor,  # (total_seqlen, num_q_heads, head_dim)
    k: Tensor,  # (total_seqlen, num_k_heads, head_dim)
    v: Tensor,  # (total_seqlen, num_v_heads, head_dim)
    state: Tensor,  # (num_seqs, num_v_heads, head_dim, head_dim)
    A_log: Tensor,  # (num_v_heads)
    a: Tensor,  # (total_seqlen, num_v_heads)
    dt_bias: Tensor,  # (num_v_heads)
    b: Tensor,  # (total_seqlen, num_v_heads)
    cu_seqlens: Tensor,  # (num_seqlens + 1)
    scale: float,
):
    # compute g and beta
    # TODO: fuse this with one of the kernels inside chunk_gated_delta_rule_fwd()
    x = a.float() + dt_bias.float()
    g = -torch.exp(A_log.float()) * F.softplus(x)  # this is actually log(g)
    beta = b.float().sigmoid()

    _, out, _, final_state, _, _, _ = chunk_gated_delta_rule_fwd(
        q=q.unsqueeze(0),
        k=k.unsqueeze(0),
        v=v.unsqueeze(0),
        g=g.unsqueeze(0),
        beta=beta.unsqueeze(0),
        scale=scale,
        initial_state=state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    return out.squeeze(0), final_state
