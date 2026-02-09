# AGENTS Guide

Concise operating guide for AI agents working in this repo.

## Repo Intent
- Fast local iteration on FlashInfer-Bench workloads.
- Keep docs and commands practical and reproducible.
- Current maintained examples:
  - Triton MoE: `solution/triton/moe_le_fused_entry.py`
  - Python GDN decode: `solution/python/gdn_decode_reference.py`

## Environment
- Use `uv` (no conda).
- Setup:
  ```bash
  uv venv
  uv pip install flashinfer-bench
  ```
- If `flashinfer_bench.agents` import is missing:
  ```bash
  uv pip install git+https://github.com/flashinfer-ai/flashinfer-bench.git
  ```

## Dataset
- Full dataset path used in this repo:
  - `/home/simon/flashinfer-competition/mlsys26-contest`
- Always set:
  ```bash
  export FIB_DATASET_PATH=/home/simon/flashinfer-competition/mlsys26-contest
  ```

## Core Workflow
1. Edit `config.toml`.
2. Run:
   ```bash
   uv run python scripts/run_local.py
   ```
   (`run_local.py` calls `pack_solution.py` automatically.)

## Verified Example Configs

### 1) Triton MoE
```toml
[solution]
name = "moe-triton-example"
definition = "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"
author = "team-name"

[build]
language = "triton"
entry_point = "moe_le_fused_entry.py::kernel"
```
Mini run:
```bash
FIB_DATASET_PATH=$PWD/mini_datasets/moe_single uv run python scripts/run_local.py
```
Observed status: `PASSED`.

### 2) Python GDN Decode
```toml
[solution]
name = "gdn-decode-python-reference"
definition = "gdn_decode_qk4_v8_d128_k_last"
author = "team-name"

[build]
language = "python"
entry_point = "gdn_decode_reference.py::run"
destination_passing_style = false
```
Mini run:
```bash
FIB_DATASET_PATH=$PWD/mini_datasets/gdn_decode_single uv run python scripts/run_local.py
```
Observed status: `PASSED`.

## Mini-Benchmark Pattern
- Mini datasets in `mini_datasets/*_single` contain one definition + one workload line.
- Keep `blob` symlink to full dataset blob directory or safetensor inputs will fail.
- Creation template:
  ```bash
  NAME=<definition_name>
  OP=<op_dir>
  ROOT=mini_datasets/${NAME}_single
  SRC=/home/simon/flashinfer-competition/mlsys26-contest

  mkdir -p "$ROOT/definitions/$OP" "$ROOT/workloads/$OP"
  cp "$SRC/definitions/$OP/$NAME.json" "$ROOT/definitions/$OP/"
  head -n 1 "$SRC/workloads/$OP/$NAME.jsonl" > "$ROOT/workloads/$OP/$NAME.jsonl"
  ln -sfn "$SRC/blob" "$ROOT/blob"
  ```

## Important Constraints / Pitfalls
- `definition` must exactly match a dataset definition name.
- `entry_point` must be `"<file_path>::<function_name>"`.
- `scripts/pack_solution.py` packs top-level files only in language folders (non-recursive).
- For multi-output Python value-returning solutions:
  - set `destination_passing_style = false`
  - return a tuple
  - safest: avoid strict return annotations if signature validation is flaky.
- Triton entrypoint must be a normal Python callable wrapper; do not expose raw `@triton.jit` directly as the benchmark entrypoint.

## Result Status Quick Meaning
- `COMPILE_ERROR`: build/signature/entrypoint issue.
- `RUNTIME_ERROR`: callable raised at execution.
- `INCORRECT_NUMERICAL`: executed but values wrong.
- `PASSED`: correctness check passed.

## Current Config State
- `config.toml` may be switched during experiments. Before running, always verify it matches your intended example.
