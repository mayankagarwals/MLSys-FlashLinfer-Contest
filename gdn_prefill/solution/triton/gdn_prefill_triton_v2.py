"""
gdn_prefill_triton_v2 -- parallel form, full sequence, no chunking (commit e665875).

Processes the entire sequence as one N×N block per head (no BT chunk loop).
Log-space gate cumsum: g = -exp(A_log)*softplus(...), then cumsum + exp for Gamma.

GDN prefill -- from-scratch Triton workspace (parallel to solution/python).

This file is the *entry point* for a from-scratch Triton implementation of
Gated DeltaNet prefill. It currently ships a numerically-correct **pure-PyTorch
recurrent reference** so the benchmark harness PASSES immediately, giving a green
baseline to iterate from. The plan is to incrementally replace the marked
sections below with Triton kernels, validating correctness against this same
reference at every step.

How to run it (see MODAL_RUNBOOK.md):
    # point config.toml at this dir (uncomment the triton block there), then:
    uv run gdn_prefill/scripts/run_local_fast.py            # local B200
    uv run modal run gdn_prefill/scripts/run_modal.py       # remote B200

--------------------------------------------------------------------------------
Staged replacement roadmap (chunked parallel form -- see notes/parallel_form/)
--------------------------------------------------------------------------------
The recurrent reference below is O(T) sequential per sequence. The Triton
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

Prior art for the chunked parallel kernels lives in
`solution/python_old/triton_v1.py` (and later revisions).
"""

import math

import torch
import torch.nn.functional as F
import triton


def _alloc_fn(size: int, alignment: int, stream: int | None):
    return torch.empty(size, device="cuda", dtype=torch.int8)


triton.set_allocator(_alloc_fn)


def _matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Float32 matmul for numerical stability (matches the contest reference)."""
    return a.float() @ b.float()


# === [Triton replacement point -- Stage 1: gate precompute] ==================
def _compute_gate_and_beta(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """g = exp(-exp(A_log) * softplus(a + dt_bias)),  beta = sigmoid(b)."""
    x = a.float() + dt_bias.float()  # [T, HV]
    g = -torch.exp(A_log.float()) * F.softplus(x)  # [T, HV]
    beta = torch.sigmoid(b.float())  # [T, HV]
    return g, beta


# === [Triton replacement point -- Stages 2-4: per-sequence delta rule] =======
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
    seq_len, num_heads, _ = v_HV.shape
    out = []
    s_out = []

    for head in range(num_heads):
        
        G: torch.Tensor = g_H[:,head] # [N, 1]
        G = torch.cumsum(G, dim=0)  # [N] in log space

        B: torch.Tensor = beta_H[:, head]# [N, 1]
        V: torch.Tensor = v_HV[:, head, :].float() # [N, V]
        K: torch.Tensor = k_HK[:, head, :].float() # [N, K]
        Q: torch.Tensor = q_HK[:, head, :].float() # [N, K]
        S_in: torch.Tensor = state_HKV[head, :, :] # [K, V]
        

        Gamma: torch.Tensor = torch.exp(G[:, None] - G[None, :]) # [N, N]
        C: torch.Tensor = K @ K.T # [N, N]
        C = Gamma * C # [N, N] (pointwise mul)
        C = B[:, None] * C # [N, N] (broadcast)
        C = torch.tril(C, diagonal=-1)

        I: torch.Tensor = torch.eye(seq_len, seq_len, device=q_HK.device, dtype=torch.float32)
        C = I + C # [N, N]
        T: torch.Tensor = torch.linalg.inv(C) # [N, N]
        T: torch.Tensor = T * B[None, :]

        U: torch.Tensor = T @ V  # [N, V]
        W: torch.Tensor = (T * torch.exp(G[None, :])) @ K # [N, K]

        V_p: torch.Tensor = U - (W @ S_in) #[N , V]

        M: torch.Tensor = torch.tril(torch.ones(seq_len, seq_len, device=q_HK.device), diagonal=0)
        M_p: torch.Tensor = M * Gamma # [N, N]


        O: torch.Tensor = torch.exp(G[:, None]) * (Q @ S_in) + ((Q @ K.T) * M_p) @ V_p # [N, V]
        S_out: torch.Tensor = (
            torch.exp(G[-1]) * S_in + 
            K.T @ (V_p * torch.exp(G[-1] - G[:, None]))
        )  # [K, V]

        out.append((scale * O).to(torch.bfloat16))
        s_out.append(S_out)
    
    return (torch.stack(out , dim=1), torch.stack(s_out, dim = 0))

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
