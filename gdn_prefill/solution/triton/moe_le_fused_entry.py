import torch
import triton
import triton.language as tl


# Fused per-expert kernel:
# - On-the-fly FP8 block dequantization for hidden_states, W13 (gate/up), and W2 (down)
# - GEMM1 (split into two heads) -> SwiGLU -> GEMM2
# - Accumulate per-token with routing weights into output
@triton.jit
def _moe_le_fused_kernel(
    # Hidden states and scales
    hs_ptr: tl.pointer_type(tl.float8e4nv),         # [T, H], fp8 e4m3fn (NV)
    hs_scale_ptr: tl.pointer_type(tl.float32),      # [H/128, T], fp32
    T,
    H,
    I,
    # Token index list for this local expert
    tok_idx_ptr: tl.pointer_type(tl.int32),         # [Tk]
    Tk,                                              # int32
    # Expert weights and scales (for one local expert)
    w13_ptr: tl.pointer_type(tl.float8e4nv),        # [2I, H], fp8
    s13_ptr: tl.pointer_type(tl.float32),           # [num_gemm1_out_blocks(=32), num_hidden_blocks(=56)], fp32
    w2_ptr: tl.pointer_type(tl.float8e4nv),         # [H, I], fp8
    s2_ptr: tl.pointer_type(tl.float32),            # [num_hidden_blocks(=56), num_intermediate_blocks(=16)], fp32
    # Routing weights for tokens of this expert
    w_tok_ptr: tl.pointer_type(tl.float32),         # [Tk]
    # Output (accumulating)
    out_ptr: tl.pointer_type(tl.float32),           # [T, H]
    # Strides (in elements)
    stride_hs_t,
    stride_hs_h,
    stride_hs_scale_hb,
    stride_hs_scale_t,
    stride_w13_o,
    stride_w13_h,
    stride_s13_o,
    stride_s13_hb,
    stride_w2_h,
    stride_w2_i,
    stride_s2_hb,
    stride_s2_ib,
    stride_out_t,
    stride_out_h,
    # Compile-time constants
    NUM_H_BLOCKS: tl.constexpr,          # 56
    NUM_G1_BLOCKS: tl.constexpr,         # 32
    NUM_I_BLOCKS: tl.constexpr,          # 16
    BLOCK_M: tl.constexpr,               # tokens per program
    BLOCK_N: tl.constexpr,               # H tile (128)
    BLOCK_K: tl.constexpr,               # K=H block (128)
    BLOCK_I: tl.constexpr                # I block (128)
):
    pid_m = tl.program_id(0)  # token tile id
    pid_n = tl.program_id(1)  # hidden output H tile id (also H block index when BLOCK_N=128)

    # Offsets and masks
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < Tk

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < H

    # Gather token indices for this tile [BLOCK_M]
    tok_idx = tl.load(tok_idx_ptr + offs_m, mask=mask_m, other=0).to(tl.int32)
    # Per-token routing weights [BLOCK_M]
    w_tok = tl.load(w_tok_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)

    # Accumulator for output tile [BLOCK_M, BLOCK_N]
    out_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Hidden block index for this H tile; with BLOCK_N == 128, this equals pid_n
    hb = pid_n

    # Pre-create "other" tensors for masked loads of fp8 tiles (avoid dtype cast errors)
    other_a_fp8 = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float8e4nv)
    other_w13_fp8 = tl.zeros((BLOCK_I, BLOCK_K), dtype=tl.float8e4nv)
    other_w2_fp8 = tl.zeros((BLOCK_N, BLOCK_I), dtype=tl.float8e4nv)

    # Iterate over intermediate blocks (I in blocks of 128)
    for ib in range(0, NUM_I_BLOCKS):
        # Accumulators for GEMM1 partials for this ib: U1 and U2 tiles [BLOCK_M, BLOCK_I]
        u1 = tl.zeros((BLOCK_M, BLOCK_I), dtype=tl.float32)
        u2 = tl.zeros((BLOCK_M, BLOCK_I), dtype=tl.float32)

        # Row indices within W13 for current ib
        i1_offs = ib * BLOCK_I + tl.arange(0, BLOCK_I)
        i2_offs = I + ib * BLOCK_I + tl.arange(0, BLOCK_I)

        # Loop over K dimension (H) in blocks of 128
        for kb in range(0, NUM_H_BLOCKS):
            k_offs = kb * BLOCK_K + tl.arange(0, BLOCK_K)
            mask_k = k_offs < H

            # Load A tile: [BLOCK_M, BLOCK_K] from hs_ptr using gathered token rows
            a_ptrs = hs_ptr + (tok_idx[:, None] * stride_hs_t) + (k_offs[None, :] * stride_hs_h)
            a_fp8 = tl.load(a_ptrs, mask=(mask_m[:, None] & mask_k[None, :]), other=other_a_fp8)
            a = a_fp8.to(tl.float32)

            # Load and apply per-block scaling for A: hs_scale_ptr[kb, tok_idx]
            sA = tl.load(
                hs_scale_ptr + kb * stride_hs_scale_hb + tok_idx * stride_hs_scale_t,
                mask=mask_m,
                other=0.0,
            )
            a = a * sA[:, None]

            # Load W13_1 tile: [BLOCK_I, BLOCK_K]
            w13_1_ptrs = w13_ptr + (i1_offs[:, None] * stride_w13_o) + (k_offs[None, :] * stride_w13_h)
            w13_1_fp8 = tl.load(w13_1_ptrs, mask=(mask_k[None, :]), other=other_w13_fp8)
            w13_1 = w13_1_fp8.to(tl.float32)
            # Scale for W13_1: s13[ib, kb]
            s13_1 = tl.load(s13_ptr + ib * stride_s13_o + kb * stride_s13_hb)
            w13_1 = w13_1 * s13_1

            # Load W13_2 tile: [BLOCK_I, BLOCK_K]
            w13_2_ptrs = w13_ptr + (i2_offs[:, None] * stride_w13_o) + (k_offs[None, :] * stride_w13_h)
            w13_2_fp8 = tl.load(w13_2_ptrs, mask=(mask_k[None, :]), other=other_w13_fp8)
            w13_2 = w13_2_fp8.to(tl.float32)
            # Scale for W13_2: s13[NUM_I_BLOCKS + ib, kb]
            s13_2 = tl.load(s13_ptr + (NUM_I_BLOCKS + ib) * stride_s13_o + kb * stride_s13_hb)
            w13_2 = w13_2 * s13_2

            # GEMM1 partials: [BLOCK_M, BLOCK_I]
            u1 += tl.dot(a, tl.trans(w13_1))
            u2 += tl.dot(a, tl.trans(w13_2))

        # SwiGLU on the block
        silu_u2 = u2 / (1.0 + tl.exp(-u2))
        c_blk = silu_u2 * u1  # [BLOCK_M, BLOCK_I]

        # Load W2 tile corresponding to current H tile and ib block: [BLOCK_N, BLOCK_I]
        w2_ptrs = w2_ptr + (offs_n[:, None] * stride_w2_h) + (i1_offs[None, :] * stride_w2_i)
        w2_fp8 = tl.load(w2_ptrs, mask=(mask_n[:, None]), other=other_w2_fp8)
        w2 = w2_fp8.to(tl.float32)
        # Scale for W2: s2[hb, ib] (one scalar per [128,128] tile)
        s2 = tl.load(s2_ptr + hb * stride_s2_hb + ib * stride_s2_ib)
        w2 = w2 * s2

        # Accumulate into output tile: [BLOCK_M, BLOCK_N] += [BLOCK_M, BLOCK_I] @ [BLOCK_I, BLOCK_N]
        out_acc += tl.dot(c_blk, tl.trans(w2))

    # Apply per-token routing weights
    out_acc = out_acc * w_tok[:, None]

    # Accumulate into global output
    out_ptrs = out_ptr + (tok_idx[:, None] * stride_out_t) + (offs_n[None, :] * stride_out_h)
    out_prev = tl.load(out_ptrs, mask=(mask_m[:, None] & mask_n[None, :]), other=0.0)
    out_new = out_prev + out_acc
    tl.store(out_ptrs, out_new, mask=(mask_m[:, None] & mask_n[None, :]))


def _check_cuda_and_move(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    if t.device.type == 'cuda':
        return t
    if device.type != 'cuda':
        raise RuntimeError("CUDA is required to run this kernel; no CUDA device available.")
    return t.to(device, non_blocking=True)


def _ensure_cuda(*tensors):
    # Ensure CUDA is available. If not, raise clear error.
    if not torch.cuda.is_available():
        for t in tensors:
            if isinstance(t, torch.Tensor) and t.is_cuda:
                raise RuntimeError("CUDA inputs provided but CUDA is reported unavailable.")
        raise RuntimeError("CUDA is required to run this kernel; no CUDA device available.")
    return torch.device('cuda')


@torch.no_grad()
def run(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float,
):
    # Constants per spec
    H = 7168
    I = 2048
    E_global = 256
    E_local = 32
    TOP_K = 8
    N_GROUP = 8
    TOPK_GROUP = 4
    BLOCK = 128
    NUM_H_BLOCKS = H // BLOCK            # 56
    NUM_I_BLOCKS = I // BLOCK            # 16
    NUM_G1_BLOCKS = (2 * I) // BLOCK     # 32

    # Validate shapes and dtypes
    assert hidden_states.dtype == torch.float8_e4m3fn, "hidden_states must be FLOAT8_E4M3FN"
    assert gemm1_weights.dtype == torch.float8_e4m3fn, "gemm1_weights must be FLOAT8_E4M3FN"
    assert gemm2_weights.dtype == torch.float8_e4m3fn, "gemm2_weights must be FLOAT8_E4M3FN"
    assert routing_logits.dtype == torch.float32, "routing_logits must be float32"
    assert routing_bias.dtype in (torch.float32, torch.bfloat16, torch.float16), "routing_bias must be float or bf16/fp16"
    assert hidden_states_scale.dtype == torch.float32, "hidden_states_scale must be float32"
    assert gemm1_weights_scale.dtype == torch.float32, "gemm1_weights_scale must be float32"
    assert gemm2_weights_scale.dtype == torch.float32, "gemm2_weights_scale must be float32"

    T = int(routing_logits.shape[0])
    assert routing_logits.shape[-1] == E_global, "routing_logits last dim must be 256"
    assert hidden_states.shape == (T, H), "hidden_states must be [T, 7168]"
    assert hidden_states_scale.shape == (NUM_H_BLOCKS, T), "hidden_states_scale must be [56, T]"
    assert gemm1_weights.shape == (E_local, 2 * I, H), "gemm1_weights must be [32, 4096, 7168]"
    assert gemm1_weights_scale.shape == (E_local, NUM_G1_BLOCKS, NUM_H_BLOCKS), "gemm1_weights_scale must be [32, 32, 56]"
    assert gemm2_weights.shape == (E_local, H, I), "gemm2_weights must be [32, 7168, 2048]"
    assert gemm2_weights_scale.shape == (E_local, NUM_H_BLOCKS, NUM_I_BLOCKS), "gemm2_weights_scale must be [32, 56, 16]"

    # Device management
    device = _ensure_cuda(
        routing_logits,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
    )
    orig_device = routing_logits.device

    # Move tensors to CUDA if needed
    routing_logits_cu = _check_cuda_and_move(routing_logits, device).contiguous()
    routing_bias_cu = _check_cuda_and_move(routing_bias.to(torch.float32), device).contiguous()
    hidden_states_cu = _check_cuda_and_move(hidden_states, device).contiguous()
    hidden_states_scale_cu = _check_cuda_and_move(hidden_states_scale, device).contiguous()
    gemm1_weights_cu = _check_cuda_and_move(gemm1_weights, device).contiguous()
    gemm1_weights_scale_cu = _check_cuda_and_move(gemm1_weights_scale, device).contiguous()
    gemm2_weights_cu = _check_cuda_and_move(gemm2_weights, device).contiguous()
    gemm2_weights_scale_cu = _check_cuda_and_move(gemm2_weights_scale, device).contiguous()

    # 1) Routing (DeepSeek-V3 no-aux) on CUDA (PyTorch)
    logits = routing_logits_cu.to(torch.float32)                      # [T, E]
    bias = routing_bias_cu.to(torch.float32).reshape(-1)              # [E]
    s = torch.sigmoid(logits)                                         # [T, E]
    s_with_bias = s + bias                                            # [T, E]

    group_size = E_global // N_GROUP  # 32
    s_wb_grouped = s_with_bias.view(T, N_GROUP, group_size)           # [T, 8, 32]
    top2_vals, _ = torch.topk(s_wb_grouped, k=2, dim=2, largest=True, sorted=False)  # [T, 8, 2]
    group_scores = top2_vals.sum(dim=2)                               # [T, 8]
    _, group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False)  # [T, 4]

    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = group_mask.unsqueeze(2).expand(T, N_GROUP, group_size).reshape(T, E_global)

    neg_inf = torch.finfo(torch.float32).min
    scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)
    _, topk_idx = torch.topk(scores_pruned, k=TOP_K, dim=1, largest=True, sorted=False)  # [T, 8]

    M = torch.zeros_like(s)
    M.scatter_(1, topk_idx, 1.0)
    weights = s * M
    weights_sum = weights.sum(dim=1, keepdim=True) + 1e-20
    weights = (weights / weights_sum) * float(routed_scaling_factor)

    # 2) Allocate output accumulator in float32
    out_accum = torch.zeros((T, H), dtype=torch.float32, device=device)

    # 3) Launch fused per-local-expert kernels
    # Tuned for B200: 64x128x128 tiles, 8 warps
    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 128
    BLOCK_I = 128

    # Strides (in elements)
    stride_hs_t = hidden_states_cu.stride(0)
    stride_hs_h = hidden_states_cu.stride(1)
    stride_hs_scale_hb = hidden_states_scale_cu.stride(0)
    stride_hs_scale_t = hidden_states_scale_cu.stride(1)

    local_start = int(local_expert_offset)
    for le in range(E_local):
        ge = local_start + le
        if ge < 0 or ge >= E_global:
            continue

        # Tokens routed to this expert
        sel_mask = (topk_idx == ge).any(dim=1)  # [T]
        if not torch.any(sel_mask):
            continue

        tok_idx = torch.nonzero(sel_mask, as_tuple=False).squeeze(1).to(torch.int32).contiguous()
        Tk_local = int(tok_idx.numel())

        # Per-token routing weights for this expert
        w_tok = weights.index_select(0, tok_idx.to(torch.int64))[:, ge].to(torch.float32).contiguous()

        # Expert slices
        w13_e = gemm1_weights_cu[le]                     # [2I, H], fp8
        s13_e = gemm1_weights_scale_cu[le]               # [32, 56], fp32
        w2_e = gemm2_weights_cu[le]                      # [H, I], fp8
        s2_e = gemm2_weights_scale_cu[le]                # [56, 16], fp32

        # Strides for expert tensors (in elements)
        stride_w13_o = w13_e.stride(0)
        stride_w13_h = w13_e.stride(1)
        stride_s13_o = s13_e.stride(0)
        stride_s13_hb = s13_e.stride(1)
        stride_w2_h = w2_e.stride(0)
        stride_w2_i = w2_e.stride(1)
        stride_s2_hb = s2_e.stride(0)
        stride_s2_ib = s2_e.stride(1)
        stride_out_t = out_accum.stride(0)
        stride_out_h = out_accum.stride(1)

        # Grid: tokens and H tiles
        grid_m = (Tk_local + BLOCK_M - 1) // BLOCK_M
        grid_n = (H + BLOCK_N - 1) // BLOCK_N
        if grid_m == 0 or grid_n == 0:
            continue

        _moe_le_fused_kernel[(grid_m, grid_n)](
            # Pointers
            hidden_states_cu,
            hidden_states_scale_cu,
            T,
            H,
            I,
            tok_idx,
            Tk_local,
            w13_e,
            s13_e,
            w2_e,
            s2_e,
            w_tok,
            out_accum,
            # Strides
            stride_hs_t,
            stride_hs_h,
            stride_hs_scale_hb,
            stride_hs_scale_t,
            stride_w13_o,
            stride_w13_h,
            stride_s13_o,
            stride_s13_hb,
            stride_w2_h,
            stride_w2_i,
            stride_s2_hb,
            stride_s2_ib,
            stride_out_t,
            stride_out_h,
            # Consts
            NUM_H_BLOCKS,
            NUM_G1_BLOCKS,
            NUM_I_BLOCKS,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            BLOCK_I,
            num_warps=8,
            num_stages=3,
        )

    # 4) Convert to BF16 for output
    out_bf16 = out_accum.to(torch.bfloat16)

    # Move back to original device if needed
    if orig_device.type != 'cuda':
        out_bf16 = out_bf16.cpu()

    return out_bf16


@torch.no_grad()
def kernel(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float,
    output: torch.Tensor,
) -> None:
    """Destination-passing wrapper expected by flashinfer_bench."""
    out = run(
        routing_logits,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        local_expert_offset,
        routed_scaling_factor,
    )
    output.copy_(out.to(device=output.device, dtype=output.dtype))
