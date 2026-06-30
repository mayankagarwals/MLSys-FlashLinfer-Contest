# Step 2: From Linear Attention to DeltaNet and Gated DeltaNet


**Goal:** understand why DeltaNet changes the linear-attention write rule, why the change can be interpreted as online learning, and how adding a forget gate produces Gated DeltaNet.

---

## 1. Starting point: recurrent linear attention

From Step 1, plain linear attention maintains:

$$
S_t = S_{t-1} + k_t^T v_t
$$

and retrieves from the state using:

$$
\text{out}_t = q_t S_t
$$

Because:

$$
S_t = \sum_{i=1}^{t} k_i^T v_i
$$

we have:

$$
q_t S_t
=

\sum_{i=1}^{t}(q_t k_i^T)v_i
$$

So plain linear attention is exactly the dot-product-weighted sum over all previous values.

Each update:

$$
k_t^T v_t
$$

stores one key–value association as an outer product.

---

## 2. The problem with the additive write rule

The linear-attention write rule always adds the entire value:

$$
S_t = S_{t-1} + k_t^T v_t
$$

It does not ask whether the state already contains that association.

### Repeated key–value pair

Assume $k$ is normalized:

$$
k k^T = |k|^2 = 1
$$

After storing $(k,v)$ once:

$$
S_1 = k^T v
$$

Querying with $k$ gives:

$$
kS_1 = k(k^Tv) = (kk^T)v = v
$$

The state already returns the correct value.

But if the same pair appears again, plain linear attention writes it again:

$$
S_2 = S_1 + k^Tv = 2k^Tv
$$

Now:

$$
kS_2 = 2v
$$

This is correct for unnormalized linear attention: two occurrences produce two contributions.

However, it is not ideal if we want the fixed-size state to behave like a stable associative memory. Without kv cache, we need some structure that can reliably tell us if a query aligns with a previous key, we get the associated value. Not give double of the value.

$$
k \longmapsto v
$$

Ideally, once the state already maps $k$ to $v$, seeing the same association again should not double the result.

---

## 3. Reinterpret the state as an associative memory

The retrieval operation is:

$$
\text{out}(q) = qS
$$

If a future query is equal or close to some stored key $k_t$, we would like the output to be equal or close to its associated value $v_t$.

In particular, when:

$$
q = k_t
$$

we want:

$$
k_t S \approx v_t
$$

This is where the idea of **predicting the value from the key** comes from.

It is not a separate task introduced arbitrarily. It is simply the desired behavior of the memory:

> When the memory is addressed using key $k_t$, it should retrieve value $v_t$.

The state matrix is therefore treated as a learned linear map:

$$
\text{key} \rightarrow \text{value}
$$

$$
k_t S \rightarrow v_t
$$

---

## 4. Check what the old state already remembers

Before writing the new association, query the old state with the incoming key:

$$
\hat v_t = k_t S_{t-1}
$$

Here:

* $v_t$ is the value we want the state to return.
* $\hat v_t$ is the value the old state currently returns.

Define the retrieval error, or **delta**:

$$
\delta_t = v_t - \hat v_t
$$

Substituting the prediction:

$$
\delta_t = v_t - k_t S_{t-1}
$$

This residual tells us what information is missing from the current memory.

---

## 5. Write the error instead of the full value

Plain linear attention associates a value with a key using:

$$
k_t^T v_t
$$

DeltaNet uses the same outer-product write mechanism, but replaces the full value with the missing correction:

$$
k_t^T \delta_t
$$

Therefore:

$$
S_t = S_{t-1} + \beta_t k_t^T \delta_t
$$

Substituting the delta:

$$
\boxed{
S_t = S_{t-1} + \beta_t k_t^T \left(v_t-k_tS_{t-1}\right)
}
$$

This is the **DeltaNet recurrence**.

The conceptual change from linear attention is:

$$
\boxed{
v_t
\quad\longrightarrow\quad
v_t-k_tS_{t-1}
}
$$

Plain linear attention says:

> Store the full value $v_t$.

DeltaNet says:

> Check what the memory already retrieves for $k_t$, then store only the missing part.

The parameter $\beta_t$ controls how strongly the state is updated:

$$
0 \leq \beta_t \leq 1
$$

It acts like an online learning rate or write gate.

---


## 6. Why repeated associations no longer accumulate

Suppose the state already retrieves the correct value:

$$
k_tS_{t-1}=v_t
$$

Then the delta is:

$$
\delta_t = v_t-k_tS_{t-1} = 0
$$

Therefore the update is zero:

$$
S_t = S_{t-1} + \beta_tk_t^T(0) = S_{t-1}
$$

The same association is not written repeatedly.

This gives DeltaNet a stable mapping-like behavior:

$$
k_t \mapsto v_t
$$

rather than the count-accumulating behavior of raw linear attention.

This resembles one useful property of softmax attention: if multiple highly matching keys all contain the same value, normalization prevents the output magnitude from simply growing in proportion to the number of copies.

DeltaNet does not exactly reproduce softmax attention, but it learns a more normalized and stable key-to-value memory than plain additive linear attention.

---

## 7. DeltaNet as online gradient descent

The same recurrence can be derived as a learning rule.

We want the state to map the current key to the current value:

$$
k_tS \approx v_t
$$

Define the squared-error loss:

$$
L_t(S)
=

\frac{1}{2}
\left|
k_tS-v_t
\right|_2^2
$$

The gradient with respect to $S$ is:

$$
\nabla_S L_t
=

k_t^T(k_tS-v_t)
$$

Perform one gradient-descent step from $S_{t-1}$:

$$
S_t
=

S_{t-1} -

\beta_t
\nabla_S L_t
$$

Substitute the gradient:

$$
S_t
=

S_{t-1} -

\beta_t
k_t^T
(k_tS_{t-1}-v_t)
$$

Move the negative sign inside:

$$
S_t
=

S_{t-1}
+
\beta_t
k_t^T
(v_t-k_tS_{t-1})
$$

This is exactly the DeltaNet update.

Therefore DeltaNet can be interpreted as:

> A fixed-size associative memory trained online, one key–value pair at a time.

The model does not run a separate optimizer during inference. The recurrence itself performs the learning step.

---

## 8. The remaining problem: old memories never disappear

DeltaNet improves how information is written, but without gating it still begins from the full old state:

$$
S_t
=

S_{t-1}
+
\beta_tk_t^T(v_t-k_tS_{t-1})
$$

Information can remain in the state indefinitely.

For long or changing sequences, this can cause problems:

* Very old information may no longer be relevant.
* The correct mapping for a key may change over time.
* Similar key directions may interfere.
* The state may need to free capacity for new associations.

To solve this, introduce a **forget gate**.

---

## 9. Add a decay gate

Let:

$$
0 \leq g_t \leq 1
$$

First decay the previous state:

$$
\bar S_{t-1}=g_tS_{t-1}
$$

Interpretation:

* $g_t \approx 1$: preserve most of the previous memory.
* $g_t \approx 0$: forget most of the previous memory.

After forgetting, the decayed memory predicts:

$$
\hat v_t = k_t\bar S_{t-1}
$$

Substitute the decayed state:

$$
\hat v_t = k_t(g_tS_{t-1})
$$

For a scalar gate:

$$
\hat v_t = g_tk_tS_{t-1}
$$

The new residual is therefore:

$$
\delta_t = v_t-k_t\bar S_{t-1}
$$

$$
\delta_t = v_t-g_tk_tS_{t-1}
$$

Now write this correction into the decayed state:

$$
S_t = \bar S_{t-1} + \beta_tk_t^T\delta_t
$$

Substitute both terms:

$$
\boxed{
S_t = g_tS_{t-1} + \beta_tk_t^T \left(v_t-g_tk_tS_{t-1}\right)
}
$$

This is the **Gated DeltaNet recurrence**.

---

## 10. Gated DeltaNet as three operations

The recurrence is easiest to understand as three explicit steps.

### Step A: Forget some old memory

$$
\bar S_{t-1}=g_tS_{t-1}
$$

### Step B: Read what the remaining memory predicts

$$
\hat v_t=k_t\bar S_{t-1}
$$

### Step C: Correct the remaining memory

$$
S_t = \bar S_{t-1} + \beta_tk_t^T(v_t-\hat v_t)
$$

In code-like form:

```python
S_old = g_t * S
v_hat = k_t @ S_old
delta = v_t - v_hat
S = S_old + beta_t * outer(k_t, delta)
out_t = q_t @ S
```

Expanded:

```python
S = (
    g_t * S
    + beta_t
    * k_t.T
    @ (v_t - k_t @ (g_t * S))
)
```

Conceptually:

> Forget old information, check what the remaining memory retrieves, then write only the missing correction.

---
