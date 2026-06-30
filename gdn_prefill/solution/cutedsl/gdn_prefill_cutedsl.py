"""
GDN prefill -- from-scratch CuTe DSL workspace (parallel to solution/python).

This file is the *entry point* for a from-scratch CuTe DSL implementation of
Gated DeltaNet prefill. It currently ships a numerically-correct **pure-PyTorch
recurrent reference** so the benchmark harness PASSES immediately, giving a green
baseline to iterate from. The plan is to incrementally replace the marked
sections below with CuTe DSL (`cutlass.cute`) kernels, validating correctness
against this same reference at every step.

How to run it (see MODAL_RUNBOOK.md):
    # point config.toml at this dir (uncomment the cutedsl block there), then:
    uv run gdn_prefill/scripts/run_local_fast.py            # local B200
    uv run modal run gdn_prefill/scripts/run_modal.py       # remote B200

--------------------------------------------------------------------------------
Staged replacement roadmap (chunked parallel form -- see notes/parallel_form/)
--------------------------------------------------------------------------------
The recurrent reference below is O(T) sequential per sequence. The CuTe DSL
target is the *chunked parallel form*. Each stage is an independent kernel and a
natural correctness checkpoint:

  Stage 0  (this file, done):  recurrent reference, green baseline / oracle.

  Stage 1  gate precompute:    g = exp(-exp(A_log) * softplus(a + dt_bias)),
                               beta = sigmoid(b), plus per-chunk cumulative
                               decay Gamma (log-space cumsum to avoid over/
                               underflow).                [_compute_gate_and_beta]

  Stage 2  intra-chunk build:  per chunk, form
                               C = strictLower(B * Gamma * (K @ K.T)),
                               T = (I + C)^-1 diag(B)   (unit-lower tri solve),
                               U = T V,  W = T diag(G) K.   [the KKT / INV / UW
                               kernels]

  Stage 3  inter-chunk scan:   carry state S across chunks
                               (V' = U - W S0; S update),   the sequential glue.
                               [the H kernel]

  Stage 4  output:             O = scale * (Q S) with the intra-chunk causal
                               correction term.            [the O kernel]

Keep this reference callable (e.g. behind a `USE_REFERENCE` flag) so each new
kernel can be diffed against it before being trusted.
"""

import math

import torch
import torch.nn.functional as F

# When you start writing kernels, enable these (available on the B200 image):
# import cutlass
# import cutlass.cute as cute


def _matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Float32 matmul for numerical stability (matches the contest reference)."""
    return a.float() @ b.float()


# === [CuTe DSL replacement point -- Stage 1: gate precompute] ================
def _compute_gate_and_beta(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """g = exp(-exp(A_log) * softplus(a + dt_bias)),  beta = sigmoid(b)."""
    x = a.float() + dt_bias.float()  # [T, HV]
    g = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))  # [T, HV]
    beta = torch.sigmoid(b.float())  # [T, HV]
    return g, beta


# === [CuTe DSL replacement point -- Stages 2-4: per-sequence delta rule] =====
def _recurrent_sequence(
    q_HK: torch.Tensor,  # [seq_len, HV, K]   (q broadcast to HV heads)
    k_HK: torch.Tensor,  # [seq_len, HV, K]
    v_HV: torch.Tensor,  # [seq_len, HV, V]
    g_H: torch.Tensor,  # [seq_len, HV]
    beta_H: torch.Tensor,  # [seq_len, HV]
    state_HKV: torch.Tensor,  # [HV, K, V]  (k-first internal layout)
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One sequence of the gated delta rule. Returns (output[seq_len,HV,V], state)."""
    seq_len, num_heads, head_v = v_HV.shape
    out = torch.empty(
        (seq_len, num_heads, head_v), dtype=torch.bfloat16, device=v_HV.device
    )

    for i in range(seq_len):
        q_H1K = q_HK[i].unsqueeze(1).float()
        k_H1K = k_HK[i].unsqueeze(1).float()
        v_H1V = v_HV[i].unsqueeze(1).float()
        g_H11 = g_H[i].unsqueeze(1).unsqueeze(2)
        beta_H11 = beta_H[i].unsqueeze(1).unsqueeze(2)

        old_state_HKV = g_H11 * state_HKV
        old_v_H1V = _matmul(k_H1K, old_state_HKV)
        new_v_H1V = beta_H11 * v_H1V + (1 - beta_H11) * old_v_H1V
        state_remove = torch.einsum("hkl,hlv->hkv", k_H1K.transpose(-1, -2), old_v_H1V)
        state_update = torch.einsum("hkl,hlv->hkv", k_H1K.transpose(-1, -2), new_v_H1V)
        state_HKV = old_state_HKV - state_remove + state_update

        o_H1V = scale * _matmul(q_H1K, state_HKV)
        out[i] = o_H1V.squeeze(1).to(torch.bfloat16)

    return out, state_HKV


@torch.no_grad()
def run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    """Gated DeltaNet prefill (k-last state layout [H, V, K]).

    Args / shapes match the contest definition gdn_prefill_qk4_v8_d128_k_last:
      q: [T, Hq=4, K=128]   k: [T, Hk=4, K=128]   v: [T, Hv=8, V=128]
      state: [num_seqs, Hv=8, V=128, K=128]
      A_log, dt_bias: [Hv]   a, b: [T, Hv]   cu_seqlens: [num_seqs+1]
    Returns (output: [T, Hv, V] bf16, new_state: [num_seqs, Hv, V, K] f32).
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

    g, beta = _compute_gate_and_beta(A_log, a, dt_bias, b)

    # GQA: broadcast q/k heads up to the number of v heads.
    q_exp = q.repeat_interleave(num_v_heads // num_q_heads, dim=1)  # [T, HV, K]
    k_exp = k.repeat_interleave(num_v_heads // num_k_heads, dim=1)  # [T, HV, K]

    output = torch.zeros(
        (total_seq_len, num_sab_heads, head_size), dtype=torch.bfloat16, device=device
    )
    new_state = torch.zeros(
        (num_seqs, num_sab_heads, head_size, head_size),
        dtype=torch.float32,
        device=device,
    )

    for seq_idx in range(num_seqs):
        seq_start = int(cu_seqlens[seq_idx].item())
        seq_end = int(cu_seqlens[seq_idx + 1].item())
        seq_len = seq_end - seq_start
        if seq_len <= 0:
            continue

        if state is not None:
            # [H, V, K] (k-last, as stored) -> [H, K, V] internal (k-first).
            state_HKV = state[seq_idx].clone().float().transpose(-1, -2)
        else:
            state_HKV = torch.zeros(
                (num_sab_heads, head_size, head_size),
                dtype=torch.float32,
                device=device,
            )

        out_seq, state_HKV = _recurrent_sequence(
            q_exp[seq_start:seq_end],
            k_exp[seq_start:seq_end],
            v[seq_start:seq_end],
            g[seq_start:seq_end],
            beta[seq_start:seq_end],
            state_HKV,
            scale,
        )
        output[seq_start:seq_end] = out_seq
        new_state[seq_idx] = state_HKV.transpose(-1, -2)  # [H,K,V] -> [H,V,K]

    return output, new_state
