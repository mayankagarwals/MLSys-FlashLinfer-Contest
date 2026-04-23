# Cutlass CuTe DSL: inline asm and global memory hints

Notes from implementing `gdn_decode_cutedsl_small_batch_gmem.py` (matching patterns in `gdn_decode_kernel_7.cu`). Intended for future agent sessions.

## 1. `llvm.inline_asm` (MLIR `llvm` dialect)

**Import:** `from cutlass._mlir.dialects import llvm`

**Signature (Python bindings):**

```text
llvm.inline_asm(res, operands_, asm_string, constraints, *,
                  has_side_effects=None, is_align_stack=None, asm_dialect=None,
                  operand_attrs=None, loc=None, ip=None)
```

**Side-effect only (no MLIR result):** pass **`res=None`**. Example in-tree: `cute/arch/nvvm_wrappers.py` → `barrier_arrive` uses `llvm.inline_asm(None, [...], "...", "...", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)`.

**PTX / NVPTX:** use **`asm_dialect=llvm.AsmDialect.AD_ATT`**. Operand placeholders in the template are **`$0`, `$1`, …** (ATT-style).

**Global store (kernel_7 `StoreF32x4RelaxedNoAllocate`):**

- PTX string: `st.relaxed.cta.global.L1::no_allocate.v4.f32 [$0], {$1, $2, $3, $4};`
- Operands: `[ptr_llvm, f0, f1, f2, f3]` with `f*` = `cutlass.Float32(...).ir_value()` (or equivalent `f32` MLIR values).
- Constraints: **`"l,f,f,f,f"`** (`l` = global address, `f` = float).
- **`has_side_effects=True`**, **`is_align_stack=False`**.

**Global load + in-asm convert (kernel_7 `LoadBf16x4GlobalNc`):**

- Return type: **`llvm.StructType.get_literal([f32_ty, f32_ty, f32_ty, f32_ty])`** with `f32_ty = Float32.mlir_type` from **`from cutlass.cute.typing import Float32`**.
- PTX (conceptually): `ld.global.nc.L1::evict_first.v4.b16 {h0,h1,h2,h3}, [$4];` then `cvt.rn.f32.bf16 $0, h0;` … for `$1`–`$3`.
- Operands: **`[ptr_bf16_llvm]`** only.
- Constraints: **`"=f,=f,=f,=f,l"`** (four float outputs + one address input; address is **`$4`** in the string when outputs occupy `$0`–`$3`).
- Unpack: **`llvm.extractvalue(f32_ty, struct_val, [i])`** for `i in {0,1,2,3}`, then wrap with **`cutlass.Float32(...)`** if assigning into DSL scalars / `rmem` tensors.

## 2. CuTe pointers for asm operands

- Tile a global tensor: **`cute.local_tile(...)`** → subtensor with a pointer **iterator**.
- Cast element type: **`cute.recast_ptr(subtensor.iterator, dtype=cutlass.BFloat16)`** (or **`cutlass.Float32`** for `f32` rows).
- LLVM pointer for NVVM/asm: **`.to_llvm_ptr()`** on that `Pointer` (see **`cute/core.py`** → `Pointer.to_llvm_ptr` / `llvm_ptr`).

Internal normalizer for **`cute.arch.load` / `store`** accepts the same pointer objects via **`to_llvm_ptr`** (see **`cute/arch/nvvm_wrappers.py`** → `_normalize_ptr`).

## 3. `cute.arch.load` / `cute.arch.store` (extended NVVM loads)

**Import:** `cute.arch` (e.g. `import cutlass.cute as cute` → `cute.arch.load`).

**Docs / implementation:** `nvidia_cutlass_dsl/.../cutlass/cute/arch/nvvm_wrappers.py` (`load` / `store`).

**Gotchas observed:**

- For **`sem="relaxed"`**, MLIR verification required an explicit **`scope="gpu"`** (error: *Scope is required for acquire/relaxed ordering*).
- **`cop`** (cache modifier, e.g. `"cg"`) **cannot** be combined with **`level1_eviction_priority`** (e.g. `"evict_first"`) on **`nvvm.load.ext`**: *load_cache_modifier and eviction priority are not allowed together*.
- **`cute.arch.load`** with **`vector<4xf32>`** on global `f32` led to a **SIGILL / illegal instruction** at runtime in one environment; **`llvm.inline_asm`** for the same logical load was stable for **`q`/`k`**. **State load** uses **`llvm.inline_asm`** with **`ld.global.v4.f32`** (same idea as kernel_7’s per-lane **`float4`** load; requires **`vec_size == 4`**).

When extended loads work, they are the “structured” alternative to raw PTX; for PTX that must match CUDA reference kernels exactly, **`llvm.inline_asm`** is more predictable.

## 4. In-repo references

| Item | Path |
|------|------|
| CuTe `load` / `store` + enums | `.venv/.../cutlass/cute/arch/nvvm_wrappers.py` (search `def load`, `def store`) |
| Void `inline_asm` | same file, `barrier_arrive` |
| `Pointer.to_llvm_ptr` | `.venv/.../cutlass/cute/core.py` |
| CUDA reference asm | `gdn_decode/solution/python/gdn_decode_kernel_7.cu` (`LoadBf16x4GlobalNc`, `StoreF32x4RelaxedNoAllocate`) |
| Working DSL usage | `gdn_decode/solution/python/gdn_decode_cutedsl_small_batch_gmem.py` |

## 5. Quick checklist for new asm in a `@cute.kernel`

1. Get **`to_llvm_ptr()`** after **`recast_ptr`** to the correct scalar element type.
2. Choose **`llvm.inline_asm`** **`res`**: `None` for pure store, or **`llvm.StructType.get_literal([...])`** for multi-output loads.
3. Use **`AsmDialect.AD_ATT`**, **`has_side_effects=True`** for loads/stores.
4. Match **PTX operand indices** to the **constraint string** order (outputs first, then inputs).
5. Verify numerics against **`gdn_decode_baseline.run`** or the CUDA kernel on small random tensors.
