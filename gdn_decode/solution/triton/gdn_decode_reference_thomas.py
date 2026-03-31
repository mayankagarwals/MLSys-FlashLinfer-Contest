import math
import os
import torch
import torch.nn.functional as F

torch.set_float32_matmul_precision("high")
import triton
import triton.language as tl
import triton.profiler.language as pl
from flashinfer.gdn_decode import gated_delta_rule_decode_pretranspose

# Allocator for Triton kernels that need scratch memory (TMA on Blackwell)
triton.set_allocator(
    lambda size, align, stream: torch.empty(
        size,
        device="cuda",
        dtype=torch.int8,
    )
)


@torch.no_grad()
def kernel_pt_reference(q, k, v, state, A_log, a, dt_bias, b, scale, **_kwargs):
    """
    Gated Delta Net decode reference implementation (k-last layout).
    Returns (output, new_state) to match the competition reference exactly (non-DPS).

    Single-token recurrent step: reads from a fixed-size state matrix using q,
    then updates the state by erasing old information at key k and writing new
    value v. GVA config: q/k heads are repeat-interleaved to match v heads.

    Args:
        q:         [B, 1, num_q_heads=4, K=128]     bf16 — query, used to read from state.
        k:         [B, 1, num_k_heads=4, K=128]     bf16 — key, the "address" to erase/write in state.
        v:         [B, 1, num_v_heads=8, V=128]     bf16 — value, new information to write into state.
        state:     [B, num_v_heads=8, V=128, K=128]  f32 — recurrent state (k-last layout [H,V,K]).
                   Optional; zeros if None.
        A_log:     [num_v_heads=8]                    f32 — learnable log-space base decay rate per head.
                   Controls how fast each head forgets: g = exp(-exp(A_log) * softplus(a + dt_bias)).
        a:         [B, 1, num_v_heads=8]             bf16 — input-dependent decay (per token, per head).
        dt_bias:   [num_v_heads=8]                    f32 — learnable decay bias, added to a before softplus.
        b:         [B, 1, num_v_heads=8]             bf16 — update gate input. beta = sigmoid(b) controls
                   how much of the new value to write vs retaining old value at key k.
        scale:     scalar f32 — output scale factor, default 1/sqrt(K) = 1/sqrt(128).

    Returns:
        output:    [B, 1, num_v_heads=8, V=128]     bf16 — attention output.
        new_state: [B, num_v_heads=8, V=128, K=128]  f32 — updated recurrent state.

    Recurrence (per head, working in [K,V] layout):
        g     = exp(-exp(A_log) * softplus(a + dt_bias))   # global decay ∈ (0,1)
        beta  = sigmoid(b)                                  # update gate ∈ (0,1)
        S_old = g * S                                       # decay entire state
        S_new = S_old - k^T @ (k @ S_old) + k^T @ (beta * v + (1-beta) * (k @ S_old))
        out   = scale * q @ S_new
    """
    B, T, num_q_heads, K = q.shape
    _, _, num_k_heads, _ = k.shape
    _, _, num_v_heads, V = v.shape
    num_heads = num_v_heads
    device = q.device

    assert num_q_heads == 4
    assert num_k_heads == 4
    assert num_v_heads == 8
    assert K == 128 and V == 128
    assert T == 1

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(K)

    # Compute g and beta from raw parameters
    x = a.float() + dt_bias.float()  # [B, 1, HV]
    g = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))  # [B, 1, HV]
    beta = torch.sigmoid(b.float())  # [B, 1, HV]

    q_f32 = q.squeeze(1).float()
    k_f32 = k.squeeze(1).float()
    v_f32 = v.squeeze(1).float()
    g_f32 = g.squeeze(1).float()
    beta_f32 = beta.squeeze(1).float()

    if state is not None:
        state_f32 = state.float()
    else:
        state_f32 = torch.zeros(B, num_heads, V, K, dtype=torch.float32, device=device)

    q_exp = q_f32.repeat_interleave(num_v_heads // num_q_heads, dim=1)
    k_exp = k_f32.repeat_interleave(num_v_heads // num_k_heads, dim=1)

    new_state = torch.zeros_like(state_f32)
    output = torch.zeros(B, num_heads, V, dtype=torch.float32, device=device)

    # fmt: off
    for b_idx in range(B):
        for h_idx in range(num_heads):
            q_h = q_exp[b_idx, h_idx]       # [K=128]        — query vector
            k_h = k_exp[b_idx, h_idx]       # [K=128]        — key vector (L2-normalized)
            v_h = v_f32[b_idx, h_idx]       # [V=128]        — value vector
            h_state = (
                state_f32[b_idx, h_idx].clone().transpose(-1, -2)
            )                                # [K=128, V=128] — state transposed from [V,K] storage
            g_val = g_f32[b_idx, h_idx]     # scalar          — global decay gate ∈ (0,1)
            beta_val = beta_f32[b_idx, h_idx]  # scalar       — update gate ∈ (0,1)

            old_state = g_val * h_state     # [K, V]          — decayed state
            old_v = k_h @ old_state         # [K] @ [K, V] -> [V] — matvec: value currently stored at key k
            new_v = beta_val * v_h + (1 - beta_val) * old_v  # [V] — blended new/old value
            state_remove = k_h.unsqueeze(1) @ old_v.unsqueeze(0)  # [K,1] @ [1,V] -> [K, V] — outer product: erase old
            state_update = k_h.unsqueeze(1) @ new_v.unsqueeze(0)  # [K,1] @ [1,V] -> [K, V] — outer product: write new
            h_state = old_state - state_remove + state_update     # [K, V] — updated state

            output[b_idx, h_idx] = scale * (q_h @ h_state)
            new_state[b_idx, h_idx] = h_state.transpose(-1, -2)  # [K,V] -> [V,K] back to storage layout
    # fmt: on

    output = output.unsqueeze(1).to(torch.bfloat16)
    return output, new_state


kernel_pt_compiled = torch.compile(kernel_pt_reference, fullgraph=True)


@torch.no_grad()
def kernel_fi_baseline(q, k, v, state, A_log, a, dt_bias, b, scale, output, new_state):
    """
    FlashInfer's official CuTe-DSL GDN decode kernel (pretranspose / K-last layout).
    FI updates state in-place. The benchmark clones all args each iteration, so state
    is already a fresh copy. We alias new_state → state's storage (zero-copy) so the
    framework reads back the mutated state via new_state.
    """
    new_state.set_(
        state.untyped_storage(), state.storage_offset(), state.shape, state.stride()
    )
    gated_delta_rule_decode_pretranspose(
        q,
        k,
        v,
        state,
        A_log,
        a,
        dt_bias,
        b,
        scale=scale,
        output=output,
        use_qk_l2norm=False,
    )


@torch.no_grad()
def kernel_fla_recurrent(
    q, k, v, state, A_log, a, dt_bias, b, scale, output, new_state
):
    """
    Wrapper adapting our competition interface to the fused Triton kernel.
    Gate computation (decay, sigmoid) is fused into the kernel.
    Kernel works directly with k-last [V, K] state layout — no transposes needed.
    """
    K = q.shape[-1]
    B, T, H, _ = k.shape
    assert T == 1
    V = v.shape[-1]
    HV = v.shape[2]

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(K)

    N = B
    BK = 128
    BV = 8

    grid = lambda meta: (triton.cdiv(V, meta["BV"]), N * HV)
    fused_recurrent_gated_delta_rule_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        A_log=A_log,
        a_gate=a,
        dt_bias=dt_bias,
        b_gate=b,
        o=output,
        h0=state,
        ht=new_state,
        scale=scale,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        PROFILE=bool(os.environ.get("PROTON_PROFILE")),
        num_warps=8,
        num_stages=2,
    )


# ---------------------------------------------------------------------------
# Triton kernel: fused recurrent GDN forward
# Inlined from fla.ops.gated_delta_rule.fused_recurrent (flash-linear-attention)
# and then heavily simplified and optimized for the FlashInfer competition.
# ---------------------------------------------------------------------------


# To sweep configs, uncomment @triton.autotune below and remove num_warps/num_stages/BV
# from the launch call. Grid is already adaptive via lambda meta.
# Run with TRITON_PRINT_AUTOTUNING=1, logs go to ./logs/fib-bench/.
# Best config on RTX 3090: BV=8 or 16 (tied at ~4.25 µs), num_warps=8
# Best config on B200:    BV=8,  num_warps=8, num_stages=2
# BV matters: 4.25 µs (BV=8) vs 7.27 µs (BV=128). See findings/research.md.
# @triton.autotune(
#     configs=[
#         triton.Config({"BV": bv}, num_warps=nw, num_stages=ns)
#         for bv in [8, 16, 32, 64, 128]
#         for nw in [1, 2, 4, 8]
#         for ns in [1, 2, 3]
#     ],
#     key=["B", "H", "HV", "K", "V", "BK"],
# )
@triton.jit
def fused_recurrent_gated_delta_rule_fwd_kernel(
    q,
    k,
    v,
    A_log,
    a_gate,
    dt_bias,
    b_gate,
    o,
    h0,
    ht,
    scale,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    PROFILE: tl.constexpr = False,
):
    # Grid: (NV=V/BV, B*HV) — each program handles a [BK, BV] tile of one head's state
    # State is k-last layout [V, K]: element (k, v) at offset v*K + k
    if PROFILE:
        pl.enter_scope("gdn_recurrent")
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV  # batch index, v-head index
    i_h = i_hv // (HV // H)  # q/k-head index (GVA: 2 v-heads per qk-head)

    o_k = tl.arange(0, BK)  # [BK] — offsets into K dimension
    o_v = i_v * BV + tl.arange(0, BV)  # [BV] — offsets into V dimension (tile)

    # Pointers into this (batch=i_n, head) for the single token (T=1)
    p_q = q + (i_n * H + i_h) * K + o_k  # q: [B, H, K] flattened
    p_k = k + (i_n * H + i_h) * K + o_k  # k: [B, H, K] flattened
    p_v = v + (i_n * HV + i_hv) * V + o_v  # v: [B, HV, V] flattened
    p_o = o + (i_n * HV + i_hv) * V + o_v  # o: [B, HV, V] flattened

    mask_k = o_k < K  # [BK]
    mask_v = o_v < V  # [BV]
    mask_h = mask_v[:, None] & mask_k[None, :]  # [BV, BK]

    # Load state tile
    if PROFILE:
        pl.enter_scope("load_initial_state")
    p_h0 = h0 + i_nh * V * K + o_v[:, None] * K + o_k[None, :]  # k-last [V, K]
    b_h = tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)  # [BV, BK] from GMEM
    if PROFILE:
        pl.exit_scope("load_initial_state")

    # Load inputs (single token)
    if PROFILE:
        pl.enter_scope("load_qkv")
    # Issue all loads before converting, so they can overlap in flight
    b_q = tl.load(p_q, mask=mask_k, other=0)  # [BK] bf16
    b_k = tl.load(p_k, mask=mask_k, other=0)  # [BK] bf16
    b_v = tl.load(p_v, mask=mask_v, other=0)  # [BV] bf16
    b_q = b_q.to(tl.float32)
    b_k = b_k.to(tl.float32)
    b_v = b_v.to(tl.float32)
    if PROFILE:
        pl.exit_scope("load_qkv")

    b_q = b_q * scale  # [BK] — scaled query

    if PROFILE:
        pl.enter_scope("load_decay")
    # Compute gates from raw inputs (fused — avoids separate PyTorch kernels)
    b_A = tl.load(A_log + i_hv).to(tl.float32)  # scalar f32 — log base decay
    b_a = tl.load(a_gate + i_n * HV + i_hv).to(tl.float32)  # scalar bf16→f32
    b_dt = tl.load(dt_bias + i_hv).to(tl.float32)  # scalar f32 — decay bias
    b_b = tl.load(b_gate + i_n * HV + i_hv).to(tl.float32)  # scalar bf16→f32
    if PROFILE:
        pl.exit_scope("load_decay")

    if PROFILE:
        pl.enter_scope("compute_decay")
    x = b_a + b_dt  # scalar f32
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))  # softplus: scalar f32
    # decay state: g = exp(-exp(A_log) * softplus(a + dt_bias))
    b_h *= tl.exp(-tl.exp(b_A) * sp)  # scalar f32
    if PROFILE:
        pl.exit_scope("compute_decay")

    b_beta = 1.0 / (1.0 + tl.exp(-b_b))  # sigmoid(b) — update gate, scalar f32

    if PROFILE:
        pl.enter_scope("state_update")
    # Delta rule: retrieve old value, blend with new, update state
    # k@S:   sum([BV, BK] * [1, BK], dim=1) -> [BV]  (matvec: value stored at key k)
    # blend: scalar * ([BV] - [BV]) -> [BV]           (beta-weighted delta)
    b_v = b_beta * (b_v - tl.sum(b_h * b_k[None, :], 1))
    # outer product: [BV, 1] * [1, BK] -> [BV, BK]   (state update)
    b_h += b_v[:, None] * b_k[None, :]
    if PROFILE:
        pl.exit_scope("state_update")

    if PROFILE:
        pl.enter_scope("compute_output")
    # q@S: sum([BV, BK] * [1, BK], dim=1) -> [BV]    (matvec: read output from state)
    b_o = tl.sum(b_h * b_q[None, :], 1)
    if PROFILE:
        pl.exit_scope("compute_output")

    if PROFILE:
        pl.enter_scope("store_output")
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)  # [BV] -> GMEM
    if PROFILE:
        pl.exit_scope("store_output")

    # Store final state
    p_ht = ht + i_nh * V * K + o_v[:, None] * K + o_k[None, :]  # k-last [V, K]
    if PROFILE:
        pl.enter_scope("store_final_state")
    tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)  # [BV, BK] -> GMEM
    if PROFILE:
        pl.exit_scope("store_final_state")
        pl.exit_scope("gdn_recurrent")


# ---------------------------------------------------------------------------
# TMA + warp-specialized kernel (Hopper/Blackwell only, cc >= 9)
# One CTA per head, loops over V-tiles internally.
# Producer warps issue TMA loads; consumer warps compute.
# ---------------------------------------------------------------------------


@torch.no_grad()
def kernel_fla_tma(q, k, v, state, A_log, a, dt_bias, b, scale, output, new_state):
    """
    TMA + warp-specialized variant. One CTA per (batch, v-head).
    Requires Hopper/Blackwell (compute capability >= 9).
    """
    K = q.shape[-1]
    B, T, H, _ = k.shape
    assert T == 1
    V = v.shape[-1]
    HV = v.shape[2]

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(K)

    BK = 128
    BV = 64  # Best B200 config: BV=64, num_warps=4, num_stages=2

    grid = (B * HV,)  # one CTA per head
    fused_recurrent_gated_delta_rule_tma_kernel[grid](
        q=q,
        k=k,
        v=v,
        A_log=A_log,
        a_gate=a,
        dt_bias=dt_bias,
        b_gate=b,
        o=output,
        h0=state,
        ht=new_state,
        scale=scale,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        num_warps=4,
        num_stages=2,
    )


# To re-run autotune, replace @triton.jit with:
# @triton.autotune(
#     configs=[
#         triton.Config({"BV": bv}, num_warps=nw, num_stages=ns)
#         for bv in [8, 16, 32, 64, 128]
#         for nw in [4, 8]  # warp_specialize on Blackwell requires num_warps % 4 == 0
#         for ns in [1, 2, 3]
#     ],
#     key=["B", "H", "HV", "K", "V", "BK"],
# )
@triton.jit
def fused_recurrent_gated_delta_rule_tma_kernel(
    q,
    k,
    v,
    A_log,
    a_gate,
    dt_bias,
    b_gate,
    o,
    h0,
    ht,
    scale,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    # Grid: (B*HV,) — one CTA per (batch, v-head)
    # Loops over V-tiles internally with warp specialization:
    #   producer warps: TMA state loads/stores
    #   consumer warps: decay + delta rule + output compute
    i_nh = tl.program_id(0)
    i_n = i_nh // HV
    i_hv = i_nh % HV
    i_h = i_hv // (HV // H)  # q/k-head index (GVA: 2 v-heads per qk-head)

    # Load q, k once (shared across all V-tiles) — tiny, regular loads
    o_k = tl.arange(0, BK)
    b_q = tl.load(q + (i_n * H + i_h) * K + o_k).to(tl.float32) * scale
    b_k = tl.load(k + (i_n * H + i_h) * K + o_k).to(tl.float32)

    # Load and compute gate scalars once
    b_A = tl.load(A_log + i_hv).to(tl.float32)
    b_a = tl.load(a_gate + i_n * HV + i_hv).to(tl.float32)
    b_dt = tl.load(dt_bias + i_hv).to(tl.float32)
    b_b = tl.load(b_gate + i_n * HV + i_hv).to(tl.float32)

    x = b_a + b_dt
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))  # softplus
    g = tl.exp(-tl.exp(b_A) * sp)  # decay gate
    b_beta = 1.0 / (1.0 + tl.exp(-b_b))  # sigmoid — update gate

    # TMA descriptors for state [V, K] — the dominant memory traffic (64 KB per head)
    h0_desc = tl.make_tensor_descriptor(
        h0 + i_nh * V * K,
        shape=[V, K],
        strides=[K, 1],
        block_shape=[BV, BK],
    )
    ht_desc = tl.make_tensor_descriptor(
        ht + i_nh * V * K,
        shape=[V, K],
        strides=[K, 1],
        block_shape=[BV, BK],
    )

    NV: tl.constexpr = V // BV
    for i_v in tl.range(0, NV, 1, flatten=True, warp_specialize=True):
        # TMA load state tile [BV, BK]
        b_h = h0_desc.load([i_v * BV, 0]).to(tl.float32)

        # Load v tile [BV] — tiny, regular load
        o_v = i_v * BV + tl.arange(0, BV)
        b_v = tl.load(v + (i_n * HV + i_hv) * V + o_v).to(tl.float32)

        # Decay
        b_h *= g

        # Delta rule: retrieve old value, blend with new, update state
        b_v = b_beta * (b_v - tl.sum(b_h * b_k[None, :], 1))
        b_h += b_v[:, None] * b_k[None, :]

        # Output
        b_o = tl.sum(b_h * b_q[None, :], 1)

        # TMA store updated state tile
        ht_desc.store([i_v * BV, 0], b_h.to(ht_desc.dtype))

        # Store output [BV] — tiny, regular store
        tl.store(o + (i_n * HV + i_hv) * V + o_v, b_o.to(tl.bfloat16))
