# Gated DeltaNet: From the Recurrent Update to a Chunked Form

We start from the recurrent Gated DeltaNet update:

$$
S_t
=

g_tS_{t-1}
+
\beta_tk_t^\top
\left(
v_t-g_tk_tS_{t-1}
\right).
$$


## Motivation

The recurrent form updates the state one token at a time:

$$
S_0
\longrightarrow
S_1
\longrightarrow
S_2
\longrightarrow
\cdots
\longrightarrow
S_L.
$$

Each output $o_t$ depends on its corresponding prefix state $S_t$.

This sequential form is convenient during decoding, where tokens arrive one at a time. During training or prefill, however, we would like to process a chunk of tokens in parallel.

The main question is therefore:

> Can we rewrite the recurrence in a chunked form, instead of updating the state one token at a time?

## Isolating the previous-state transformation

Start by expanding the recurrent equation:

$$
S_t
=

g_tS_{t-1}
+
\beta_tk_t^\top v_t
-

g_t\beta_tk_t^\top k_tS_{t-1}.
$$

Group the terms involving the previous state:

$$
S_t
=

g_t
\left(
I-\beta_tk_t^\top k_t
\right)
S_{t-1}
+
\beta_tk_t^\top v_t.
$$

At this point we have a hunch that there is an identity matrix - outer product. If on unrolling, we end up multiplying matrices of these kind, they can be represented in a compressed manner instead of series of matmuls

Define:

$$
H_t
=

I-\beta_tk_t^\top k_t.
$$

The recurrence becomes:

$$
S_t
=

g_tH_tS_{t-1}
+
\beta_tk_t^\top v_t.
$$

## What happens when we unroll?

Unrolling one additional step gives:

$$
S_t
=

g_tH_t
\left(
g_{t-1}H_{t-1}S_{t-2}
+
\beta_{t-1}k_{t-1}^\top v_{t-1}
\right)
+
\beta_tk_t^\top v_t.
$$

Expanding:

$$
S_t
=

g_tg_{t-1}H_tH_{t-1}S_{t-2}
+
g_tH_t\beta_{t-1}k_{t-1}^\top v_{t-1}
+
\beta_tk_t^\top v_t.
$$

Continuing this pattern all the way back to $S_0$, the fully unrolled form is:

$$
S_i
=

G_iH_iH_{i-1}\cdots H_1S_0
+
\sum_{j=1}^{i}
\gamma_{ij}H_iH_{i-1}\cdots H_{j+1}\beta_jk_j^\top v_j,
$$

where

$$
G_i
=

g_ig_{i-1}\cdots g_1
$$

is the cumulative gate product up to position $i$, and

$$
\Gamma_{ij}
=

g_ig_{i-1}\cdots g_{j+1}
$$

is the gate product from position $j+1$ through $i$ (with $\Gamma_{ii}=1$).

### Two terms: remove from memory vs. add to memory

This decomposition has two structurally distinct pieces.

**Term 1 — propagate and correct the carried state:**

$$
G_iH_iH_{i-1}\cdots H_1S_0.
$$



**Term 2 — inject new associations:**

$$
\sum_{j=1}^{i}
\gamma_{ij}H_iH_{i-1}\cdots H_{j+1}\beta_jk_j^\top v_j.
$$


For now we focus on Term 1. Term 2 is deferred to a later step.

If we continue unrolling, we encounter products such as:

$$
H_bH_{b-1}\cdots H_{a+1}.
$$

For arbitrary dense matrices, repeatedly computing these products would be expensive.

However, every $H_t$ has the special form:

$$
H_t
=

I-\beta_tk_t^\top k_t.
$$

Equivalently:

$$
H_t
=

I-k_t^\top(\beta_tk_t).
$$

Each $H_t$ is therefore the identity matrix minus a rank-one matrix.

Products of arbitrary matrices are difficult to simplify, but products of matrices with this rank-one structure may admit a compressed representation.

This motivates the following ansatz:

$$
H_bH_{b-1}\cdots H_a
=\

I-\sum_{t=a}^{b}k_t^\top w_t,
$$

where

$$
w_t\in\mathbb{R}^{1\times d_k}
$$

is a vector that we have not yet determined.

This is initially a motivated guess rather than an established result.

The reason for trying this form is that every individual matrix already satisfies it:

$$
H_t
=

I-k_t^\top(\beta_tk_t).
$$

The hypothesis is whether the form will be preserved even after multiplications


