# Step 1: From Softmax Attention to Linear Attention (Recurrent Form)

This is **Step 1** in learning the full GDN (Gated Delta Net) derivation.

**Goal:** understand why linear attention can be written as a fixed-size recurrent state update, and why querying that state is *exactly* the same computation as a weighted sum over values — not an approximation.


---

## 1. Softmax attention (baseline)

For a query $q$ and a sequence of keys $k_i$ and values $v_i$:

$$
\text{Attn}(q) = \sum_i \underbrace{\frac{\exp(q \cdot k_i)}{\sum_j \exp(q \cdot k_j)}}_{\text{softmax weight}} \, v_i
$$



---

## 2. Linear attention (drop the softmax)

**Linear attention** keeps the dot-product scoring but removes the softmax normalization:

$$
\text{LinAttn}(q) = \sum_i \underbrace{(q \cdot k_i)}_{\text{unnormalized weight}} \, v_i
$$

Equivalently, with an optional scale $s$ (as in scaled dot-product attention):

$$
\text{LinAttn}(q) = s \sum_i (q \cdot k_i)\, v_i
$$

What changed:

| | Softmax attention | Linear attention |
|---|---|---|
| Weights | $\text{softmax}_i(q \cdot k)$ | $q \cdot k_i$ |
| Normalization | Yes — weights sum to 1 | No |
| Associativity trick | Blocked by softmax | Available |

The associativity is the key. Because there is no softmax denominator, we can regroup the sum:

$$
\sum_i (q \cdot k_i)\, v_i = q \sum_i k_i^T v_i
$$

Define the **state matrix**:

$$
S = \sum_i k_i^T v_i
$$

Then:

$$
\text{LinAttn}(q) = q S
$$

So linear attention is not just "similar" to a matrix–vector product — at this step it **is** one.

---

## 3. Why $qS$ is exactly the weighted sum

Claim:

> In plain linear attention, $qS$ is not "close to" the weighted sum. It is **exactly the same computation**, just regrouped.

The reason is that each outer product $k_i^T v_i$ behaves like a tiny memory unit.

### One key–value pair

Suppose the state contains just one pair:

$$
S_i = k_i^T v_i
$$

Query it with $q$:

$$
q S_i = q(k_i^T v_i)
$$

By associativity:

$$
q(k_i^T v_i) = (q k_i^T)\, v_i
$$

Here $q k_i^T$ is exactly the dot-product similarity between $q$ and $k_i$. So:

$$
q S_i = (q \cdot k_i)\, v_i
$$

That single outer product means:

> "When queried by $q$, measure how aligned $q$ is with $k_i$, and output that much of $v_i$."

That is precisely one attention term.

### Many key–value pairs

Store them by adding their outer products:

$$
S = \sum_i k_i^T v_i
$$

Then:

$$
q S = q \sum_i k_i^T v_i = \sum_i q(k_i^T v_i) = \sum_i (q k_i^T)\, v_i
$$

That is exactly:

$$
\sum_i \underbrace{(q \cdot k_i)}_{\text{attention weight}} \underbrace{v_i}_{\text{value}}
$$

So the state matrix is all the individual "compare with this key, then emit this value" operations added together.

---

## 4. Recurrent form (token-by-token update)

The sum over all positions can be written as an online recurrence. After processing tokens $1, \ldots, t$:

$$
S_t = \sum_{i=1}^{t} k_i^T v_i = S_{t-1} + k_t^T v_t
$$

**Decode step** (given current state $S_{t-1}$, new key $k_t$, new value $v_t$, query $q_t$):

1. **Output:** $\text{out}_t = q_t S_{t-1}$ (or include scale: $s\, q_t S_{t-1}$)
2. **Update:** $S_t = S_{t-1} + k_t^T v_t$

This is the recurrent form of linear attention:

- **$O(1)$ per token** in sequence length (for fixed head dimension).
- **Fixed state size** $S \in \mathbb{R}^{d \times d}$ — the same shape GDN uses (e.g. `[V, K]` per head in k-last layout).

Plain linear attention grows the state without bound in principle (every token adds a rank-1 term). GDN fixes that in later steps by gating and the delta rule.

---

## 5. Concrete example

Let:

$$
k_1 = [1, 0], \qquad v_1 = [10, 0]
$$

$$
k_2 = [1, 1], \qquad v_2 = [0, 20]
$$

and query:

$$
q = [2, 3]
$$

### Dot-product (linear) attention

Weights:

$$
q k_1^T = 2, \qquad q k_2^T = 5
$$

Weighted sum:

$$
2 v_1 + 5 v_2 = 2[10, 0] + 5[0, 20] = [20, 100]
$$

### State-matrix version

$$
k_1^T v_1 = \begin{bmatrix} 10 & 0 \\ 0 & 0 \end{bmatrix}, \qquad
k_2^T v_2 = \begin{bmatrix} 0 & 20 \\ 0 & 20 \end{bmatrix}
$$

$$
S = \begin{bmatrix} 10 & 20 \\ 0 & 20 \end{bmatrix}
$$

$$
q S = [2, 3] \begin{bmatrix} 10 & 20 \\ 0 & 20 \end{bmatrix} = [20, 100]
$$

Exactly the same result.

### Recurrent check

After token 1: $S_1 = k_1^T v_1$. After token 2: $S_2 = S_1 + k_2^T v_2 = S$. Querying $S_2$ with $q$ still gives $[20, 100]$.

---

