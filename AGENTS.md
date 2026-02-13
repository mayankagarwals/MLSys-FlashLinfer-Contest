# AGENTS Guide

Concise operating guide for AI agents in this repo.

## Quick Commands
- Create mini dataset:
  ```bash
  bash_scripts/create_mini.sh <definition_name> <op_dir>
  ```
- `<op_dir>` values: `dsa_paged`, `gdn`, `moe`
- `<definition_name>` values:
  - `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64`
  - `dsa_topk_indexer_fp8_h64_d128_topk2048_ps64`
  - `gdn_decode_qk4_v8_d128_k_last`
  - `gdn_prefill_qk4_v8_d128_k_last`
  - `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048`
- Run benchmark:
  ```bash
  bash_scripts/run_local.sh mini
  bash_scripts/run_local.sh full
  ```
- Run benchmark + dump current run trace to markdown:
  ```bash
  bash_scripts/run_local.sh mini-dump-trace-md
  bash_scripts/run_local.sh full-dump-trace-md
  ```
  - Writes markdown log file to `logs/<definition>_<run-timestamp>.md`
  - Override output folder with `FIB_RUN_LOG_DIR`

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
- Default env for wrapper scripts:
  ```bash
  export FIB_FULL_DATASET_PATH=/home/simon/flashinfer-competition/mlsys26-contest
  ```

## Core Workflow
1. Update `config.toml` for target kernel entrypoint.
2. Run on full dataset:
   ```bash
   bash_scripts/run_local.sh full
   ```
3. Dump current full run trace to markdown (optional):
   ```bash
   bash_scripts/run_local.sh full-dump-trace-md
   ```
4. Validate status (`PASSED`/`INCORRECT_NUMERICAL`/etc.).

## Mini Dataset Workflow
- Use mini datasets for fast iteration: one definition JSON + one workload JSONL row.
- Create:
  ```bash
  bash_scripts/create_mini.sh <definition_name> <op_dir> [output_dir]
  ```
- `<op_dir>` values: `dsa_paged`, `gdn`, `moe`
- `<definition_name>` values:
  - `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64`
  - `dsa_topk_indexer_fp8_h64_d128_topk2048_ps64`
  - `gdn_decode_qk4_v8_d128_k_last`
  - `gdn_prefill_qk4_v8_d128_k_last`
  - `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048`
- Run with mini dataset (`mini_dataset_name` defaults to `<definition>_single` from `config.toml`):
  ```bash
  bash_scripts/run_local.sh mini [mini_dataset_name]
  bash_scripts/run_local.sh mini-dump-trace-md [mini_dataset_name]
  ```
- Run on full dataset and dump full trace log to markdown:
  ```bash
  bash_scripts/run_local.sh full-dump-trace-md
  ```
  - Writes markdown log file to `logs/<definition>_<run-timestamp>.md`

## Manual Command (Equivalent)
- Direct runner invocation remains available:
  ```bash
  FIB_DATASET_PATH=<dataset_path> uv run python scripts/run_local.py
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
