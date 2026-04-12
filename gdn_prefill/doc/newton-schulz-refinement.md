# chunk_v6 tf32 Inverse Precision Analysis

The Newton-Schulz refinement mentioned in this doc is source from:
https://www.scirp.org/reference/referencespapers?referenceid=1405587

"New Approach for the Inversion of Structured Matrices via Newton’s Iteration"

## What the kernel does (math)

The GDN prefill needs to compute the **inverse of a 64x64 lower-triangular matrix** `M = (I + A)` per chunk, where A is strictly lower triangular (zeros on and above diagonal).

### Matrix inverse ≠ element-wise 1/x

A matrix inverse `M^{-1}` satisfies `M @ M^{-1} = I`. This is NOT just `-1 * A[i,j]` per element. For example:

```
M = [[1,    0,    0],       M^{-1} = [[1,       0,      0],
     [0.5,  1,    0],                 [-0.5,    1,      0],
     [0.3,  0.4,  1]]                 [-0.10,  -0.4,    1]]

M^{-1}[2,0] = -0.10, NOT -0.3
Because: -A[2,0] + A[2,1]*A[1,0] = -0.3 + 0.4*0.5 = -0.10
```

Each element of the inverse is a sum over **all paths** through the matrix — direct and indirect. The indirect path `row2 → row1 → row0` contributes `+0.2`, significantly changing the result. Matrix multiply is exactly the operation that computes "sum over all paths."

### The Neumann series

For `M = I + A` where A is strictly lower triangular, `A^n = 0` for an n×n matrix (nilpotent — the longest path has n-1 hops). This gives the convergent series:

```
(I + A)^{-1} = I - A + A² - A³ + ... + (-A)^{n-1}
```

This is the matrix version of the scalar geometric series `1/(1+x) = 1 - x + x² - ...`.

### Fast factorization (repeated squaring)

Computing each power separately would need n-1 matmuls. The factorization groups terms into powers of 2:

```
(I-A)                           captures terms through A¹
(I-A)(I+A²)                    captures through A³
(I-A)(I+A²)(I+A⁴)             captures through A⁷
(I-A)(I+A²)(I+A⁴)(I+A⁸)      captures through A¹⁵   ← enough for 16×16
```

You can verify: `(I-A)(I+A²) = I - A + A² - A³ = I - A⁴`. Since `A^16 = 0` for 16×16, the product equals `(I+A)^{-1}`.

The number of factors = `ceil(log2(n))`:

| Block size | Nilpotent at | Factors | Matmuls (squarings + multiplies) |
|---|---|---|---|
| 4×4 | A⁴=0 | 2 | 2 |
| 8×8 | A⁸=0 | 3 | 4 |
| **16×16** | A¹⁶=0 | **4** | **6** |
| 64×64 | A⁶⁴=0 | 6 | 10 |

### Why 16×16 blocks (not 64×64 or 8×8)

**Hierarchical approach**: Instead of inverting the full 64×64 directly (10 matmuls of [64,64]), split into four 16×16 diagonal blocks, invert each, then compute off-diagonal blocks.

**Compute comparison**:
- A [64,64] matmul costs 64³ = 262,144 multiply-adds
- A [16,16] matmul costs 16³ = 4,096 multiply-adds (64× cheaper)
- Direct 64×64: 10 matmuls × 64 = **640 equivalent** [16,16] matmuls
- Hierarchical 16×16: 40 actual [16,16] matmuls → **16× less compute**

**Why not 8×8?** Tensor cores have minimum tile m16n8k8 — need at least 16 rows. An 8×8 matmul would either waste half the tile or fall back to scalar CUDA cores. **16×16 is the smallest size that fully utilizes tensor cores.**

## Hierarchical inverse: how it works

### Step 1: Diagonal block inverse (24 matmuls)

Split the 64×64 into four 16×16 diagonal blocks:

```
┌─────────┬─────────┬─────────┬─────────┐
│  I+A_11 │    0    │    0    │    0    │
├─────────┼─────────┼─────────┼─────────┤
│  A_21   │  I+A_22 │    0    │    0    │
├─────────┼─────────┼─────────┼─────────┤
│  A_31   │  A_32   │  I+A_33 │    0    │
├─────────┼─────────┼─────────┼─────────┤
│  A_41   │  A_42   │  A_43   │  I+A_44 │
└─────────┴─────────┴─────────┴─────────┘
```

Invert each diagonal block independently using the Neumann series:
- `Ai_11 = (I + A_11)^{-1}` — 6 matmuls of [16,16]
- Same for Ai_22, Ai_33, Ai_44
- Total: **4 × 6 = 24 matmuls**

### Step 2: Off-diagonal blocks (16 matmuls)

The off-diagonal blocks are determined by enforcing `M @ M^{-1} = I`. For each off-diagonal block, look at the corresponding block equation:

```
M[row_i, :] @ M^{-1}[:, col_j] = 0   (for i ≠ j)
```

Solving for the unknown block:

```
Ai_21 = -Ai_22 @ A_21 @ Ai_11                                          → 2 matmuls
Ai_32 = -Ai_33 @ A_32 @ Ai_22                                          → 2 matmuls
Ai_43 = -Ai_44 @ A_43 @ Ai_33                                          → 2 matmuls
Ai_31 = -Ai_33 @ (A_31 @ Ai_11 + A_32 @ Ai_21)                        → 3 matmuls
Ai_42 = -Ai_44 @ (A_42 @ Ai_22 + A_43 @ Ai_32)                        → 3 matmuls
Ai_41 = -Ai_44 @ (A_41 @ Ai_11 + A_42 @ Ai_21 + A_43 @ Ai_31)        → 4 matmuls
                                                                   Total: 16 matmuls
```

The pattern: `Ai[i,j]` needs one matmul per intermediate column between j and i (the "paths"), plus one outer multiply by `Ai[i,i]`. Ai_41 has the most (4) because it has 3 intermediate columns.

**Derivation example for Ai_21**: enforce `M[row1,:] @ M^{-1}[:,col0] = 0`:
```
A_21 @ Ai_11 + (I+A_22) @ Ai_21 = 0
(I+A_22) @ Ai_21 = -A_21 @ Ai_11
Ai_21 = (I+A_22)^{-1} @ (-A_21 @ Ai_11) = -Ai_22 @ A_21 @ Ai_11
```

### Total: 24 + 16 = 40 matmuls of [16,16]

## The precision problem with tf32

`tl.dot(..., input_precision="tf32")` truncates fp32 inputs to TF32 (10-bit mantissa, vs 23-bit for fp32) before multiplying. This is fine for **single-shot** matmuls. But the inverse has **chained multiplies**:

```
A^2 = dot(A, A)        ← error ε₁
A^4 = dot(A^2, A^2)    ← error compounds: ε₁ × ε₁
A^8 = dot(A^4, A^4)    ← ε₁⁴
Ai = dot(Ai, I + A^8)  ← all errors compound
```

After 6 chained dots: error ~6 × 2^{-11} ≈ 0.003. BF16 precision is ~2^{-8} ≈ 0.004. The tf32 error (0.003) is borderline — sometimes it crosses the bf16 rounding boundary, causing wrong results. This is why people saw large errors with `dot_precision="tf32"`.

## Why the old code (triton_v4/chunk_v5) avoided tf32

triton_v4 uses `DOT_PRECISION="tf32x3"` — three-pass TF32, which approximates full fp32 precision (~23 bits). After 6 chained dots: error ~6 × 2^{-23} ≈ 7×10^{-7}, far below bf16's 0.004. Safe but slower (3× more MMA passes).

The old pipeline was:
```
Compute Ai (fp32, tf32x3 precision)
  → Store Ai to global memory as BF16 (Ai tensor dtype=k.dtype=bf16)
  → debug_barrier()
  → Reload full 64x64 Ai from global as BF16
  → Cast to fp32 implicitly for the dot
  → Compute W, U
```

The reload converts Ai to **BF16** (7-bit mantissa, 8-bit significand), which **truncates** the fp32 values. So even though the inverse was computed precisely, the W/U computation uses BF16-precision Ai.

## Why chunk_v6 can use tf32 and NOT have errors

The key insight (chunk_v6 lines 161-170):

```python
Ai_11 = Ai_11.to(tl.bfloat16).to(tl.float32)
Ai_21 = Ai_21.to(tl.bfloat16).to(tl.float32)
# ... all 10 blocks
```

This does a **bf16 roundtrip** — truncate to BF16 precision, then back to fp32 — **before** using Ai for W/U computation. This is equivalent to what the old code did (store BF16 to global, reload as BF16), but without the global memory trip.

The crucial effect: **BF16 has only 7-bit mantissa (8-bit significand), while TF32 has 10-bit.** When you roundtrip through BF16, you've already destroyed all precision below 7 bits. So using tf32 (10-bit) instead of tf32x3 for the inverse computation **mostly doesn't matter** — the extra precision from tf32x3 gets largely thrown away by the BF16 roundtrip. However, this alone isn't sufficient — edge cases near the rounding boundary still fail (see Newton-Schulz below).

## Newton-Schulz refinement (current approach)

The bf16 roundtrip approach alone was insufficient — tf32-only still caused edge-case failures near tolerance boundaries (e.g., T=973 N=3 with max_abs_error=0.0123 > atol=0.01). The **Newton-Schulz refinement** is the proper fix: use fast tf32 for the bulk computation, then cheaply correct the accumulated error.

### Background: tf32 vs tf32x3

`tl.dot(A, B, input_precision="tf32")` truncates fp32 inputs (23-bit mantissa) to TF32 (10-bit mantissa) before multiplying. This is 1 MMA pass — fast but imprecise.

`tf32x3` splits the 23-bit mantissa into three ~10-bit chunks, does 3 separate MMA passes, and sums the results in fp32 accumulators. Same accuracy as fp32, 3× slower than tf32.

Concrete example — multiplying a value `1.23456789`:
```
fp32:  1.23456789  = 1.00111100000011001010010  (23-bit mantissa)
tf32:  1.234375    = 1.0011110000               (10-bit, rest chopped)
error: 0.000193 ≈ 2^{-12}

tf32x3: splits into 3 chunks, each MMA uses 10-bit inputs:
  Pass 1: mma(A_hi, B_hi)         — top 10 bits × top 10 bits
  Pass 2: mma(A_hi, B_mid_lo)     — top 10 × bottom 13
  Pass 3: mma(A_mid_lo, B_hi)     — bottom 13 × top 10
  result = sum of 3 passes        — recovers full 23-bit precision
```

### Why tf32 is dangerous for chained inverse

The diagonal inverse has 6 chained dots where output feeds into the next:
```
A² = dot(A, A)           ← error ε per element
A⁴ = dot(A², A²)         ← input already has error ε, PLUS new truncation
A⁸ = dot(A⁴, A⁴)         ← errors compound further
Ai = dot(Ai, I + A⁸)     ← all errors accumulated
```

With tf32: each dot adds ~2^{-11} error, compounding over 6 dots → ~6 × 2^{-11} ≈ 0.003.
With tf32x3: each dot adds ~2^{-23} error → ~6 × 2^{-23} ≈ 7×10^{-7}. Safe but 3× slower.

### Newton-Schulz: the key idea

Instead of making every dot precise, compute a **rough** inverse fast (tf32), then **correct** it cheaply.

**Scalar analogy**: say you want `1/3` but your calculator only gives 3 digits:
```
x = 0.333                          (rough estimate, error ≈ 0.001)
x_new = x × (2 - 3 × x)
      = 0.333 × (2 - 0.999)
      = 0.333 × 1.001
      = 0.333333                    (error² ≈ 0.000001)
```

One correction step **squared** the error. This is Newton's method for `f(x) = 1/M - x`.

**Matrix version**: same formula, just with matrices:
```
Ai_new = Ai @ (2I - M @ Ai)        where M = I + A
```

### Why the formula works (step-by-step derivation)

Say our rough inverse `Ai` has error E: `Ai = M^{-1}(I + E)`.

This means `Ai` is the true inverse `M^{-1}` plus some error. Let's see what happens:

```
Step 1: Compute  M @ Ai
        = M @ M^{-1}(I + E)
        = I + E                     (if Ai were exact, this would be I)

Step 2: Compute  2I - M @ Ai
        = 2I - (I + E)
        = I - E                     (this is the "correction factor")

Step 3: Compute  Ai_new = Ai @ (I - E)
        = M^{-1}(I + E)(I - E)
        = M^{-1}(I - E²)           (because (1+x)(1-x) = 1-x²)
```

The error went from **E to E²** (quadratic convergence):
- tf32 error: E ≈ 10^{-3}
- After 1 refinement: E² ≈ 10^{-6}
- bf16 precision needs: ~10^{-2.4}
- 10^{-6} << 10^{-2.4} → safely within tolerance

`Ai_new` replaces `Ai` — it's just a better approximation of the same inverse. You don't need the old `Ai` anymore.

### Step-by-step MMA comparison

**Method 1: tf32x3 (old)** — every dot uses 3 MMA passes:
```
# One 16×16 diagonal block inverse: 6 dots × 3 passes = 18 MMAs
Ai = I - A
A² = dot3(A, A)           # 3 MMAs
Ai = dot3(Ai, I + A²)     # 3 MMAs
A⁴ = dot3(A², A²)         # 3 MMAs
Ai = dot3(Ai, I + A⁴)     # 3 MMAs
A⁸ = dot3(A⁴, A⁴)         # 3 MMAs
Ai = dot3(Ai, I + A⁸)     # 3 MMAs
                            ──────── 18 MMAs per block, × 4 = 72
```

**Method 2: tf32 + Newton-Schulz (new)** — 6 fast dots + 2 precise correction dots:
```
# One 16×16 diagonal block: 6 dots × 1 pass = 6 MMAs (rough inverse)
Ai = I - A
A² = dot1(A, A)           # 1 MMA (imprecise!)
Ai = dot1(Ai, I + A²)     # 1 MMA
A⁴ = dot1(A², A²)         # 1 MMA
Ai = dot1(Ai, I + A⁴)     # 1 MMA
A⁸ = dot1(A⁴, A⁴)         # 1 MMA
Ai = dot1(Ai, I + A⁸)     # 1 MMA
                            ──────── 6 MMAs (Ai has ~10^{-3} error)

# Newton-Schulz correction: 2 dots × 3 passes = 6 MMAs
R  = dot3(M, Ai)           # 3 MMAs  (R ≈ I + E, checks "how wrong is Ai?")
Ai = dot3(Ai, 2I - R)      # 3 MMAs  (corrects Ai, error² ≈ 10^{-6})
                            ──────── 6 MMAs

Total: 12 MMAs per block × 4 blocks = 48
```

The 2 refinement dots **must** use tf32x3 — if they used tf32, you'd be correcting a 10^{-3} error with a tool that itself has 10^{-3} error, which gets you nowhere.

### MMA pass count

| Component | tf32x3 (old) | tf32 + Newton-Schulz (new) |
|-----------|-------------|---------------------------|
| Diagonal inverse (4 blocks × 6 dots) | 24 × 3 = 72 | 24 × 1 = 24 |
| Newton-Schulz refinement (4 blocks × 2 dots) | — | 8 × 3 = 24 |
| Off-diagonal merge (16 dots) | 16 × 3 = 48 | 16 × 3 = 48 |
| **Total** | **120** | **96** |

20% fewer MMA passes → **-313 us on T>=1024 workloads (-4.2%)**.

### Precision strategy summary

| Dot category | Count | Precision | Why |
|-------------|-------|-----------|-----|
| Diagonal inverse | 24 (6 per block × 4) | tf32 (1 pass) | Fast; errors corrected by Newton-Schulz |
| Newton-Schulz refinement | 8 (2 per block × 4) | tf32x3 (3 passes) | Must be precise to actually correct the error |
| Off-diagonal merge | 16 | tf32x3 (3 passes) | Up to 4 chained dots, borderline for bf16 with tf32 |
| W/U block-wise dots | 20 | bf16 MMA | Standard bf16 precision, unchanged from chunk_v5 |

**Result**: 100/100 correct on all workloads (including previously-failing edge cases like T=973 N=3).

## Why CUDA wmma tf32 works without Newton-Schulz

cuda_parallel_v3 uses `nvcuda::wmma` for the inverse. The PTX shows `wmma.mma.sync.m16n16k8.f32.tf32.tf32.f32` — ostensibly the same single-pass tf32. But the SASS tells a different story:

| Code path | PTX instruction | SASS instruction | K per instruction |
|-----------|----------------|------------------|-------------------|
| Triton `tl.dot(input_precision="tf32")` | `mma.sync.m16n8k8.f32.tf32` | `HMMA.1688.F32.TF32` | **K=8** |
| CUDA `wmma::mma_sync` tf32 | `wmma.mma.sync.m16n16k8.f32.tf32` | `HMMA.1684.F32.TF32` | **K=4** |

The ptxas assembler **splits each wmma K=8 into 2× K=4 SASS instructions**, giving finer-grained FP32 accumulation (more, smaller partial sums → less rounding error per step). Triton's `mma.sync` maps directly to K=8 SASS without splitting.

Verified by instruction count: 8 HMMA.1684 per 16×16 matmul = 4 K-tiles × 2 N-halves → K=4 per instruction.

This is why wmma "happens to work" — K=4 accumulation has roughly half the error per dot compared to K=8. But it's **not mathematically guaranteed**; it just has enough margin for these workloads. Newton-Schulz provides a provable guarantee (E² convergence).

## Comparison of all approaches

| Pipeline | Inverse precision | Ai precision at W/U step | MMA passes | Result |
|----------|------------------|--------------------------|------------|--------|
| **Old (triton_v4/v5)** | tf32x3 (~fp32) | BF16 (global store/reload) | 120 | Correct — but slow |
| **chunk_v6 bf16 roundtrip only** | tf32 (10-bit) | BF16 roundtrip (8-bit) | ~48 | Edge-case failures (T=973 N=3) |
| **chunk_v6 Newton-Schulz** | tf32 + NS refinement | BF16 roundtrip (8-bit) | 96 | **Correct — best tradeoff** |
| **Naive tf32 (no roundtrip)** | tf32 (10-bit) | fp32 (23-bit) | ~48 | Errors — chained tf32 errors visible |
| **CUDA wmma (cuda_parallel_v3)** | tf32 K=4 (hardware) | BF16 (smem store/reload) | ~80 | Works empirically, not guaranteed |

## Summary

The Newton-Schulz approach is the sweet spot: fast tf32 for the bulk diagonal inverse (where 6 chained dots accumulate the most error), then a cheap 2-dot refinement to recover precision (squaring the error from ~10^{-3} to ~10^{-6}), with tf32x3 kept for the 16 off-diagonal merges. Combined with the bf16 roundtrip before W/U computation, this achieves full correctness with 20% fewer MMA passes than all-tf32x3.

### Additional benefit of chunk_v6

Besides the precision optimization, chunk_v6 also **fuses** the W/U computation into the same kernel as the inverse (block-wise, using the 16x16 Ai sub-blocks directly from registers). This eliminates:
- Global store of full 64x64 Ai
- debug_barrier()
- Global reload of 64x64 Ai

The `acc=` chaining (e.g., `u1 = tl.dot(Ab_10, v0); u1 = tl.dot(Ab_11, v1, acc=u1)`) preserves the same MMA accumulation order as a single [64,64] dot, ensuring numerical equivalence.
