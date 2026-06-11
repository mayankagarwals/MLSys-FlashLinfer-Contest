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


## Step 1: Exploration to find the state decay and update

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
\Gamma_{ij}H_iH_{i-1}\cdots H_{j+1}\beta_jk_j^\top v_j,
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


## Step 2: Simpler composition for Term 1

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

<details>
<summary><strong>Proof by Induction</strong></summary>


Let

$$
H_i = I - \beta_i k_i^\top k_i,
$$

where $k_i \in \mathbb{R}^{1 \times d}$ and $\beta_i$ is a scalar.

Although the product

$$
H_i H_{i-1}\cdots H_1
$$

contains complicated interactions between all the keys, it can always be written as

$$
\boxed{
H_i H_{i-1}\cdots H_1
=
I-\sum_{j=1}^{i} k_j^\top \hat{w}_j
}
$$

for appropriately defined row vectors $\hat{w}_1,\ldots,\hat{w}_i$.

The vectors satisfy the recurrence

$$
\boxed{
\hat{w}_i
=
\beta_i
\left(
k_i
-
\sum_{j=1}^{i-1}
(k_i k_j^\top)\hat{w}_j
\right)
}
$$

with

$$
\hat{w}_1=\beta_1 k_1.
$$

---

### Base Case: $i=1$

We have

$$
H_1=I-\beta_1 k_1^\top k_1.
$$

Define

$$
\hat{w}_1=\beta_1 k_1.
$$

Then

$$
H_1=I-k_1^\top \hat{w}_1,
$$

so the claimed form holds for $i=1$.

---

### Example: $i=2$

Consider

$$
H_2H_1
=
\left(I-\beta_2 k_2^\top k_2\right)
\left(I-k_1^\top \hat{w}_1\right).
$$

Expanding,

$$
\begin{aligned}
H_2H_1
&=
I-k_1^\top \hat{w}_1
-\beta_2 k_2^\top k_2
+\beta_2 k_2^\top k_2k_1^\top \hat{w}_1
\\
&=
I-k_1^\top \hat{w}_1
-k_2^\top
\beta_2
\left(
k_2-(k_2k_1^\top)\hat{w}_1
\right).
\end{aligned}
$$

Define

$$
\hat{w}_2
=
\beta_2
\left(
k_2-(k_2k_1^\top)\hat{w}_1
\right).
$$

Therefore,

$$
H_2H_1
=
I-k_1^\top \hat{w}_1-k_2^\top \hat{w}_2.
$$

---

### Inductive Step

Assume that the claim holds for $i-1$:

$$
H_{i-1}\cdots H_1
=
I-\sum_{j=1}^{i-1}k_j^\top \hat{w}_j.
$$

Now left-multiply by $H_i$:

$$
\begin{aligned}
H_iH_{i-1}\cdots H_1
&=
\left(I-\beta_i k_i^\top k_i\right)
\left(
I-\sum_{j=1}^{i-1}k_j^\top \hat{w}_j
\right)
\\
&=
I-\sum_{j=1}^{i-1}k_j^\top \hat{w}_j
-\beta_i k_i^\top k_i
+\beta_i k_i^\top k_i
\sum_{j=1}^{i-1}k_j^\top \hat{w}_j.
\end{aligned}
$$

Since $k_i k_j^\top$ is a scalar,

$$
\beta_i k_i^\top k_i k_j^\top \hat{w}_j
=
k_i^\top\beta_i(k_i k_j^\top)\hat{w}_j.
$$

Therefore,

$$
\begin{aligned}
H_iH_{i-1}\cdots H_1
&=
I-\sum_{j=1}^{i-1}k_j^\top \hat{w}_j
\\
&\quad
-k_i^\top
\beta_i
\left(
k_i-
\sum_{j=1}^{i-1}
(k_i k_j^\top)\hat{w}_j
\right).
\end{aligned}
$$

Define

$$
\hat{w}_i
=
\beta_i
\left(
k_i-
\sum_{j=1}^{i-1}
(k_i k_j^\top)\hat{w}_j
\right).
$$

Hence,

$$
H_iH_{i-1}\cdots H_1
=
I-\sum_{j=1}^{i}k_j^\top \hat{w}_j.
$$

Thus, the claim holds for every $i$.

</details>

We also derived 

$$
\boxed{
\hat{w}_i
=
\beta_i
\left(
k_i
-
\sum_{j=1}^{i-1}
(k_i k_j^\top)\hat{w}_j
\right)
}
$$

## Step 3: Simpler composition for Term 2

Define the accumulated new information:

$$
R_i
=
\sum_{j=1}^{i}
\Gamma_{ij}
H_i H_{i-1}\cdots H_{j+1}
\beta_j k_j^\top v_j.
$$

This is exactly Term 2 from the unrolled state update. There is a recurrence hidden inside this expression.

Instead of directly working with the complicated product

$$
H_iH_{i-1}\cdots H_{j+1},
$$

we first rewrite $R_i$ recursively:

$$
\boxed{
R_i
=
g_iH_iR_{i-1}
+
\beta_i k_i^\top v_i
}
$$

where

$$
H_i=I-\beta_i k_i^\top k_i.
$$

<details>

<summary><strong>Proof by Induction</strong></summary>

## Representation Claim

Assume that $R_{i-1}$ can be represented as

$$
R_{i-1}
=
\sum_{j<i}
\Gamma_{i-1,j}k_j^\top u_j.
$$

We will show that $R_i$ has the same form:

$$
\boxed{
R_i
=
\sum_{j\leq i}
\Gamma_{ij}k_j^\top u_j
}
$$

for appropriately defined vectors $u_j$.

The decay factors satisfy

$$
\Gamma_{ij}
=
g_i\Gamma_{i-1,j},
\qquad j<i,
$$

and

$$
\Gamma_{ii}=1.
$$

---

## Inductive Step

Starting from the recurrence,

$$
R_i
=
g_iH_iR_{i-1}
+
\beta_i k_i^\top v_i,
$$

substitute the assumed representation of $R_{i-1}$:

$$
R_i
=
g_iH_i
\left(
\sum_{j<i}
\Gamma_{i-1,j}k_j^\top u_j
\right)
+
\beta_i k_i^\top v_i.
$$

Since

$$
g_i\Gamma_{i-1,j}
=
\Gamma_{ij},
$$

we obtain

$$
R_i
=
H_i
\left(
\sum_{j<i}
\Gamma_{ij}k_j^\top u_j
\right)
+
\beta_i k_i^\top v_i.
$$

Using

$$
H_i
=
I-\beta_i k_i^\top k_i,
$$

we get

$$
\begin{aligned}
R_i
&=
\left(
I-\beta_i k_i^\top k_i
\right)
\left(
\sum_{j<i}
\Gamma_{ij}k_j^\top u_j
\right)
+
\beta_i k_i^\top v_i
\\
&=
\sum_{j<i}
\Gamma_{ij}k_j^\top u_j
-
\beta_i k_i^\top
\left(
\sum_{j<i}
\Gamma_{ij}
(k_i k_j^\top)u_j
\right)
+
\beta_i k_i^\top v_i.
\end{aligned}
$$

Group the terms involving $k_i^\top$:

$$
R_i
=
\sum_{j<i}
\Gamma_{ij}k_j^\top u_j
+
k_i^\top
\beta_i
\left(
v_i
-
\sum_{j<i}
\Gamma_{ij}
(k_i k_j^\top)u_j
\right).
$$

Define

$$
\boxed{
u_i
=
\beta_i
\left(
v_i
-
\sum_{j<i}
\Gamma_{ij}
(k_i k_j^\top)u_j
\right)
}
$$

Then

$$
R_i
=
\sum_{j<i}
\Gamma_{ij}k_j^\top u_j
+
k_i^\top u_i.
$$

Because

$$
\Gamma_{ii}=1,
$$

this becomes

$$
\boxed{
R_i
=
\sum_{j\leq i}
\Gamma_{ij}k_j^\top u_j
}
$$

which proves that the representation is preserved by induction.

</details>

## Step 4: Substituting the Compressions Back into the State

Recall the Gated DeltaNet recurrence:

$$
S_i
=
g_i S_{i-1}
+
\beta_i k_i^\top
\left(
v_i-g_i k_iS_{i-1}
\right).
$$

Equivalently, defining

$$
H_i=I-\beta_i k_i^\top k_i,
$$

we have

$$
S_i
=
g_iH_iS_{i-1}
+
\beta_i k_i^\top v_i.
$$

After unrolling from the initial state $S_0$,

$$
S_i
=
G_iH_iH_{i-1}\cdots H_1S_0
+
R_i,
$$

where

$$
G_i=\prod_{t=1}^{i}g_t
$$

and

$$
\Gamma_{ij}
=
\prod_{t=j+1}^{i}g_t
=
\frac{G_i}{G_j},
\qquad
\Gamma_{ii}=1.
$$

From the previous two sections, we derived the compressions

$$
H_iH_{i-1}\cdots H_1
=
I-\sum_{j\leq i}k_j^\top \hat{w}_j
$$

and

$$
R_i
=
\sum_{j\leq i}
\Gamma_{ij}k_j^\top u_j.
$$

Here, $\hat{w}_j$ denotes the vectors from the compression of the product of the $H$ matrices. We use the hat temporarily because we will soon introduce a decay-scaled version of these vectors.

Substituting both results into the unrolled state gives

$$
S_i
=
G_i
\left(
I-\sum_{j\leq i}k_j^\top\hat{w}_j
\right)S_0
+
\sum_{j\leq i}
\Gamma_{ij}k_j^\top u_j.
$$

Expanding the first term,

$$
S_i
=
G_iS_0
-
\sum_{j\leq i}
G_i k_j^\top\hat{w}_jS_0
+
\sum_{j\leq i}
\Gamma_{ij}k_j^\top u_j.
$$

We would like the two sums to have the same decay factor $\Gamma_{ij}$.

Since

$$
G_i=\Gamma_{ij}G_j,
$$

we can rewrite each term in the first sum as

$$
\begin{aligned}
G_i k_j^\top\hat{w}_jS_0
&=
\Gamma_{ij}G_jk_j^\top\hat{w}_jS_0
\\
&=
\Gamma_{ij}k_j^\top
\left(
G_j\hat{w}_jS_0
\right).
\end{aligned}
$$

Therefore,

$$
S_i
=
G_iS_0
+
\sum_{j\leq i}
\Gamma_{ij}k_j^\top
\left(
u_j-G_j\hat{w}_jS_0
\right).
$$

Define the decay-scaled removal vector

$$
\boxed{
w_j=G_j\hat{w}_j
}
$$

and the corrected value

$$
\boxed{
v'_j=u_j-w_jS_0.
}
$$

Then the state becomes

$$
\boxed{
S_i
=
G_iS_0
+
\sum_{j\leq i}
\Gamma_{ij}k_j^\top v'_j.
}
$$

This now has exactly the same structural form as unrolled gated linear attention.

### Interpretation of the Corrected Value

The two components of $v'_j$ have different meanings:

$$
v'_j
=
\underbrace{u_j}_{\text{what token }j\text{ writes}}
-
\underbrace{w_jS_0}_{\text{what token }j\text{ removes from the initial state}}.
$$

The vector $u_j$ is the information that token $j$ wants to write, corrected for the writes of earlier tokens.

The term $w_jS_0$ is the information that token $j$ removes from the portion of the initial state that remains inside the chunk.

Thus, $v'_j$ is the token's complete, non-decayed delta to the state.

### Comparison with Gated Linear Attention

Gated linear attention has the recurrence

$$
S_i
=
g_iS_{i-1}
+
k_i^\top v_i.
$$

Unrolling it gives

$$
S_i
=
G_iS_0
+
\sum_{j\leq i}
\Gamma_{ij}k_j^\top v_j.
$$

Gated DeltaNet begins with the more complicated recurrence

$$
S_i
=
g_iS_{i-1}
+
\beta_i k_i^\top
\left(
v_i-g_i k_iS_{i-1}
\right),
$$

where the update itself depends on the previous state.

After compressing both the state-removal transformations and the new information, however, it becomes

$$
\boxed{
S_i
=
G_iS_0
+
\sum_{j\leq i}
\Gamma_{ij}k_j^\top v'_j,
}
$$

which is algebraically identical to the unrolled GLA form, with $v_j$ replaced by the transformed value $v'_j$.

## Step 5:  Converting the Tokenwise Recurrences into Matrix Form

We have already rewritten the state at position $i$ in a GLA-like form:

$$
S_i
=
G_iS_0
+
\sum_{j\leq i}
\Gamma_{ij}k_j^\top v'_j,
$$

where

$$
G_i=\prod_{t=1}^{i}g_t,
\qquad
\Gamma_{ij}
=
\frac{G_i}{G_j}
=
\prod_{t=j+1}^{i}g_t,
\qquad j\leq i.
$$

The corrected value is

$$
v'_i=u_i-w_iS_0.
$$

We now derive parallel matrix expressions for

$$
T,\quad U,\quad W,\quad V',\quad M',\quad S_N,\quad O.
$$

### Shapes and Notation

For a chunk of $N$ tokens, stack the token vectors row-wise:

$$
K=
\begin{bmatrix}
k_1\\
\vdots\\
k_N
\end{bmatrix}
\in\mathbb{R}^{N\times d_k},
\qquad
Q\in\mathbb{R}^{N\times d_k},
\qquad
V=
\begin{bmatrix}
v_1\\
\vdots\\
v_N
\end{bmatrix}
\in\mathbb{R}^{N\times d_v}.
$$

Similarly,

$$
U=
\begin{bmatrix}
u_1\\
\vdots\\
u_N
\end{bmatrix}
\in\mathbb{R}^{N\times d_v},
\qquad
W=
\begin{bmatrix}
w_1\\
\vdots\\
w_N
\end{bmatrix}
\in\mathbb{R}^{N\times d_k}.
$$

The incoming state is

$$
S_0\in\mathbb{R}^{d_k\times d_v}.
$$

Let

$$
B=
\begin{bmatrix}
\beta_1\\
\vdots\\
\beta_N
\end{bmatrix}
\in\mathbb{R}^{N\times 1},
\qquad
G=
\begin{bmatrix}
G_1\\
\vdots\\
G_N
\end{bmatrix}
\in\mathbb{R}^{N\times 1}.
$$

We use $\odot$ for elementwise multiplication. When $B$ or $G$ appears in an elementwise product with an $N\times N$ matrix, it broadcasts across columns (row $i$ is scaled by $\beta_i$ or $G_i$).

### 1. Deriving $T$ and $U$

From the previous derivation, the transformed write vectors satisfy

$$
u_i
=
\beta_i
\left(
v_i-
\sum_{j<i}
\Gamma_{ij}(k_i k_j^\top)u_j
\right).
$$

Expanding,

$$
u_i
=
\beta_iv_i
-
\sum_{j<i}
\beta_i\Gamma_{ij}(k_i k_j^\top)u_j.
$$

Define the scalar coefficient

$$
C_{ij}
=
\begin{cases}
\beta_i\Gamma_{ij}(k_i k_j^\top), & j<i,\\
0, & j\geq i.
\end{cases}
$$

Then

$$
u_i+\sum_{j<i}C_{ij}u_j=\beta_iv_i.
$$

For example,

$$
\begin{aligned}
u_1 &= \beta_1v_1,\\
u_2+C_{21}u_1 &= \beta_2v_2,\\
u_3+C_{31}u_1+C_{32}u_2 &= \beta_3v_3.
\end{aligned}
$$

Stacking all $N$ equations gives

$$
\begin{bmatrix}
1      & 0      & 0      & \cdots \\
C_{21} & 1      & 0      & \cdots \\
C_{31} & C_{32} & 1      & \cdots \\
\vdots & \vdots & \vdots & \ddots
\end{bmatrix}
\begin{bmatrix}
u_1\\
u_2\\
u_3\\
\vdots
\end{bmatrix}
=
\begin{bmatrix}
\beta_1v_1\\
\beta_2v_2\\
\beta_3v_3\\
\vdots
\end{bmatrix}.
$$

Therefore,

$$
(I+C)U
=
\operatorname{diag}(B)V.
$$

Solving for $U$,

$$
U
=
(I+C)^{-1}
\operatorname{diag}(B)V.
$$

Define

$$
\boxed{
T
=
(I+C)^{-1}
\operatorname{diag}(B)
}
$$

so that

$$
\boxed{
U=TV.
}
$$

#### Constructing $C$ in Parallel

The matrix

$$
KK^\top\in\mathbb{R}^{N\times N}
$$

contains every pairwise key dot product:

$$
(KK^\top)_{ij}=k_i k_j^\top.
$$

The complete coefficient matrix is therefore

$$
\boxed{
C
=
\operatorname{strictLower}
\left(
B
\odot
\Gamma
\odot
(KK^\top)
\right).
}
$$

Here:

- $B\in\mathbb{R}^{N\times 1}$ broadcasts $\beta_i$ across row $i$;
- $\Gamma_{ij}$ supplies the decay from token $j$ to token $i$;
- $KK^\top$ supplies $k_i k_j^\top$;
- `strictLower` retains only $j<i$.

Thus,

$$
\boxed{
T
=
\left[
I+
\operatorname{strictLower}
\left(
B\odot\Gamma\odot(KK^\top)
\right)
\right]^{-1}
\operatorname{diag}(B).
}
$$

In code,

```python
C = strictLower(B * Gamma * (K @ K.T))
T = inverse(I + C) * B.T
```

For $C$, `B` alone suffices: with shape `[N, 1]` it broadcasts across columns.

For $T$, `B.T` is needed (not `B`): it broadcasts $\beta_j$ across rows of $(I+C)^{-1}$, which is equivalent to right-multiplication by $\operatorname{diag}(B)$.

Because $I+C$ is unit lower triangular, an implementation can use a triangular solve instead of explicitly computing the inverse.

### 2. Deriving $W$

Recall that the decay-scaled removal vectors satisfy

$$
w_i
=
\beta_i
\left(
G_i k_i
-
\sum_{j<i}
\Gamma_{ij}(k_i k_j^\top)w_j
\right).
$$

Expanding,

$$
w_i
=
\beta_iG_i k_i
-
\sum_{j<i}
\beta_i\Gamma_{ij}(k_i k_j^\top)w_j.
$$

The coefficient multiplying $w_j$ is exactly the same $C_{ij}$ used in the recurrence for $u_i$. Therefore,

$$
w_i+\sum_{j<i}C_{ij}w_j
=
\beta_iG_i k_i.
$$

Stacking these equations gives

$$
(I+C)W
=
\operatorname{diag}(B)
\operatorname{diag}(G)K.
$$

Hence,

$$
\begin{aligned}
W
&=
(I+C)^{-1}
\operatorname{diag}(B)
\operatorname{diag}(G)K\\
&=
T\operatorname{diag}(G)K.
\end{aligned}
$$

Therefore,

$$
\boxed{
W=T\operatorname{diag}(G)K.
}
$$

With broadcasting,

```python
W = (T * G.T) @ K
```

because multiplying $T$ elementwise by $G^\top$ scales column $j$ of $T$ by $G_j$, which is equivalent to

$$
T\operatorname{diag}(G).
$$

The important observation is that both $U$ and $W$ use the same lower-triangular transformation $T$:

$$
\boxed{
U=TV,
\qquad
W=T\operatorname{diag}(G)K.
}
$$

### 3. Deriving $V'$

For each token,

$$
v'_i=u_i-w_iS_0.
$$

Stacking the vectors row-wise gives

$$
\boxed{
V'=U-WS_0.
}
$$

Using the expressions for $U$ and $W$,

$$
\begin{aligned}
V'
&=
TV-
T\operatorname{diag}(G)KS_0\\
&=
T
\left(
V-\operatorname{diag}(G)KS_0
\right).
\end{aligned}
$$

Thus,

$$
\boxed{
V'
=
U-WS_0
=
T
\left(
V-\operatorname{diag}(G)KS_0
\right).
}
$$

In implementation form,

$$
V' = U - WS_0
$$

```python
U = T @ V
W = (T * G.T) @ K
V_p = U - W @ S_in
```

Here $S_{\text{in}}$ is the state entering the chunk.

The two terms have a direct interpretation:

$$
v'_i
=
\underbrace{u_i}_{\text{corrected information written}}
-
\underbrace{w_iS_0}_{\text{information removed from the old state}}.
$$

### 4. Deriving the Decayed Causal Mask $M'$

For an output at position $i$, only tokens $j\leq i$ may contribute.

Define the ordinary causal mask

$$
M_{ij}
=
\begin{cases}
1, & j\leq i,\\
0, & j>i.
\end{cases}
$$

The contribution from token $j$ to token $i$ must additionally be scaled by

$$
\Gamma_{ij}
=
\frac{G_i}{G_j}.
$$

The matrix whose $(i,j)$-th element is $G_i/G_j$ can be constructed using broadcasting:

$$
\left(\frac{G}{G^\top}\right)_{ij}
=
\frac{G_i}{G_j}.
$$

Therefore, define

$$
\boxed{
M'
=
M\odot\frac{G}{G^\top}.
}
$$

Elementwise,

$$
M'_{ij}
=
\begin{cases}
\Gamma_{ij}, & j\leq i,\\
0, & j>i.
\end{cases}
$$

In implementation form,

$$
M' = M \odot \frac{G}{G^\top}
$$

```python
M_p = M * (G / G.T)
```

where `/` and `*` are elementwise operations with broadcasting.

### 5. Computing the Final State of the Chunk

The state at any position $i$ is

$$
S_i
=
G_iS_0
+
\sum_{j\leq i}
\Gamma_{ij}k_j^\top v'_j.
$$

For the final token $i=N$,

$$
S_N
=
G_NS_0
+
\sum_{j=1}^{N}
\Gamma_{Nj}k_j^\top v'_j.
$$

Since

$$
\Gamma_{Nj}
=
\frac{G_N}{G_j},
$$

we have

$$
S_N
=
G_NS_0
+
\sum_{j=1}^{N}
k_j^\top
\left(
\frac{G_N}{G_j}v'_j
\right).
$$

In matrix form,

$$
\boxed{
S_N
=
G_NS_0
+
K^\top
\left[
V'
\odot
\left(
\frac{G_N}{G}
\right)
\right].
}
$$

Here $G_N/G\in\mathbb{R}^{N\times1}$ scales row $j$ of $V'$ by

$$
\frac{G_N}{G_j}.
$$

In implementation form,

```python
S_out = G[-1] * S_in + K.T @ (V_p * (G[-1] / G))
```

This computes only the final state of the chunk, which is the state that must be passed to the next chunk.

### 6. Computing All Outputs in the Chunk

For token $i$,

$$
o_i=q_iS_i.
$$

Substitute the compressed state:

$$
o_i
=
q_i
\left(
G_iS_0+
\sum_{j\leq i}
\Gamma_{ij}k_j^\top v'_j
\right).
$$

Distributing $q_i$,

$$
o_i
=
G_iq_iS_0
+
\sum_{j\leq i}
\Gamma_{ij}
(q_i k_j^\top)v'_j.
$$

The first term for every query can be computed as

$$
\operatorname{diag}(G)QS_0.
$$

Using broadcasting,

$$
\operatorname{diag}(G)QS_0
=
G\odot(QS_0).
$$

For the second term,

$$
QK^\top\in\mathbb{R}^{N\times N}
$$

contains all query-key dot products:

$$
(QK^\top)_{ij}=q_i k_j^\top.
$$

Multiplying elementwise by $M'$ applies both causality and the correct decay:

$$
\left[(QK^\top)\odot M'\right]_{ij}
=
\begin{cases}
\Gamma_{ij}(q_i k_j^\top), & j\leq i,\\
0, & j>i.
\end{cases}
$$

Multiplying by $V'$ then performs the weighted sum over previous values:

$$
\boxed{
O
=
G\odot(QS_0)
+
\left[
(QK^\top)\odot M'
\right]V'.
}
$$

In implementation form,

```python
O = G * (Q @ S_in) + ((Q @ K.T) * M_p) @ V_p
```

This has the same computational structure as causal attention:

1. compute every query-key score;
2. apply a causal, decay-weighted mask;
3. multiply the scores by the transformed values.

### Final Parallel Equations

Let

$$
\Gamma_{ij}=\frac{G_i}{G_j}.
$$

First construct the strictly lower-triangular interaction matrix:

$$
C
=
\operatorname{strictLower}
\left(
B\odot\Gamma\odot(KK^\top)
\right).
$$

Then

$$
\boxed{
T=(I+C)^{-1}\operatorname{diag}(B)
}
$$

$$
\boxed{
U=TV
}
$$

$$
\boxed{
W=T\operatorname{diag}(G)K
}
$$

$$
\boxed{
V'=U-WS_0
}
$$

$$
\boxed{
M'=M\odot\frac{G}{G^\top}
}
$$

$$
\boxed{
S_N
=
G_NS_0+
K^\top
\left[
V'\odot\frac{G_N}{G}
\right]
}
$$

and

$$
\boxed{
O
=
G\odot(QS_0)
+
\left[
(QK^\top)\odot M'
\right]V'.
}
$$

### Compact Implementation Form


```python
# B, G: [N, 1]
# Q, K: [N, d_k]
# V:    [N, d_v]
# S_in: [d_k, d_v]

C = strictLower(B * Gamma * (K @ K.T))

# B.T broadcasts beta_j across rows (= right-multiply by diag(B)).
T = inverse(I + C) * B.T

U = T @ V
W = (T * G.T) @ K
V_p = U - W @ S_in
M_p = M * (G / G.T)

O = G * (Q @ S_in) + ((Q @ K.T) * M_p) @ V_p

S_out = (
    G[-1] * S_in
    + K.T @ (V_p * (G[-1] / G))
)
```


