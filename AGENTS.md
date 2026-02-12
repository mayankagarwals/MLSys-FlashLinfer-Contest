# AGENTS Guide

Concise operating guide for AI agents in this repo.

## Goal
- Fast local iteration on FlashInfer-Bench workloads.
- Prefer reproducible commands and minimal config changes.

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
- Full dataset path:
  - `/home/simon/flashinfer-competition/mlsys26-contest`
- Default env:
  ```bash
  export FIB_DATASET_PATH=/home/simon/flashinfer-competition/mlsys26-contest
  ```

## Core Workflow
1. Update `config.toml` for target kernel entrypoint.
2. Run:
   ```bash
   uv run python scripts/run_local.py
   ```
3. Validate status (`PASSED`/`INCORRECT_NUMERICAL`/etc.).

## Mini Dataset Workflow
- Use mini datasets for fast iteration: one definition JSON + one workload JSONL row.
- Template:
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
- Run with mini dataset:
  ```bash
  FIB_DATASET_PATH=$PWD/mini_datasets/${NAME}_single uv run python scripts/run_local.py
  ```

## Generic Config Pattern
Use this as the default template when switching kernels:

```toml
[solution]
name = "my-kernel-run"
definition = "<exact_definition_name>"
author = "team-name"

[build]
language = "python"  # or "triton"
entry_point = "<file_name>.py::<function_name>"
destination_passing_style = false  # value-returning Python callables
```

## Constraints / Pitfalls
- `definition` must exactly match a dataset definition name.
- `entry_point` format must be `"<file>::<function>"`.
- `scripts/pack_solution.py` packs top-level files only in language folders (non-recursive).
- For multi-output Python value-returning solutions:
  - set `destination_passing_style = false`
  - return a tuple
- For `gdn_decode_qk4_v8_d128_k_last` entrypoints:
  - keep baseline-compatible callable signature: `run(q, k, v, state, A_log, a, dt_bias, b, scale)`
  - public state layout must be k-last `[B, HV, V, K]` (internal transforms allowed)

## Result Status Quick Meaning
- `COMPILE_ERROR`: build/signature/entrypoint issue.
- `RUNTIME_ERROR`: callable raised at execution.
- `INCORRECT_NUMERICAL`: executed but values wrong.
- `PASSED`: correctness check passed.
