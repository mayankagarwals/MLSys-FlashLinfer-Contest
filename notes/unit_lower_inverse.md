# Inverting the unit-lower matrix in the GDN Triton kernel

V2 forms a strictly lower-triangular matrix `L` for each `BT x BT` token
chunk, then needs

```text
T = (I + L)^-1 diag(beta)
```

The helper `_unit_lower_inverse` computes the inverse factor. This note
explains why it does not use a general-purpose matrix inverse.

## Strictly lower triangular implies nilpotent

In a **lower-triangular** matrix, every entry above the main diagonal is zero.
For example:

```text
[ a  0  0  0 ]
[ b  c  0  0 ]
[ d  e  f  0 ]
[ g  h  i  j ]
```

The main diagonal runs from the upper-left to the lower-right: `a, c, f, j`.
A **strictly lower-triangular** matrix also has zeroes on that diagonal:

```text
L = [ 0  0  0  0 ]
    [ a  0  0  0 ]
    [ b  c  0  0 ]
    [ d  e  f  0 ]
```

Adding the identity gives a **unit-lower-triangular** matrix, whose diagonal
is all ones:

```text
I + L = [ 1  0  0  0 ]
        [ a  1  0  0 ]
        [ b  c  1  0 ]
        [ d  e  f  1 ]
```

For a `BT x BT` strictly lower-triangular matrix, every multiplication pushes
non-zero entries at least one diagonal farther below the main diagonal. After
`BT` multiplications there are no diagonals left:

```text
L^BT = 0
```

With `BT = 16`, this gives `L^16 = 0`.

For an arbitrary matrix, the Neumann series is normally infinite:

```text
(I + L)^-1 = I - L + L^2 - L^3 + ...
```

Here it terminates exactly rather than approximately:

```text
(I + L)^-1 = I - L + L^2 - L^3 + ... - L^15
```

To verify it, let `R = I - L + L^2 - ... - L^15`. The intermediate powers
cancel:

```text
(I + L) R = I - L^16 = I
```

This does not require `L` to be small; nilpotence is enough.

## Factored evaluation

The kernel evaluates the same polynomial as

```text
(I - L)(I + L^2)(I + L^4)(I + L^8)
```

The first two factors give

```text
(I - L)(I + L^2) = I - L + L^2 - L^3
```

Multiplying by `(I + L^4)` extends the alternating series through `L^7`, and
multiplying by `(I + L^8)` extends it through `L^15`. The required powers are
obtained with repeated squaring: `L^2`, `L^4`, and `L^8`.

The corresponding Triton code is:

```python
A = A_orig
Ai = m_I - A
A = tl.dot(A, A)             # L^2
Ai = tl.dot(Ai, m_I + A)
A = tl.dot(A, A)             # L^4
Ai = tl.dot(Ai, m_I + A)
A = tl.dot(A, A)             # L^8
Ai = tl.dot(Ai, m_I + A)
```

In exact arithmetic, `Ai` is now `(I + L)^-1`.

## Newton--Schulz refinement

Tensor-core matrix multiplication has finite-precision rounding error, so the
computed `Ai` is only an approximate inverse. Let

```text
M = I + L
R ~= M^-1
E = I - MR
```

The code applies one Newton--Schulz correction:

```text
R_new = R(2I - MR)
```

Its residual is squared:

```text
M R_new = MR(2I - MR)
        = (I - E)(I + E)
        = I - E^2
```

Thus an already-small residual becomes much smaller in one step. In the
kernel, `Ai` is `R` and `A_orig` is `L`:

```python
MAi = Ai + tl.dot(A_orig, Ai, input_precision="tf32x3")  # (I + L) R
Ai = tl.dot(Ai, 2.0 * m_I - MAi, input_precision="tf32x3")
```

The refinement corrects numerical error only; it is not required to make the
finite Neumann identity mathematically valid.

## Relation to the launch

`_unit_lower_inverse` is a `@triton.jit` device helper. It is compiled into
and called from `_recurrent_sequence_kernel`; it is not a separate kernel
launch and therefore does not have its own grid. The caller launches one
program per value head and invokes this helper once per chunk while retaining
the head's recurrent state.
