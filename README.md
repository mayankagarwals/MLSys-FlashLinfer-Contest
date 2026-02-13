# FlashInfer Competition Starter

Minimal local workflow for iterating on kernels with `flashinfer-bench`.

## Quick Commands

```bash
# Create mini dataset
bash_scripts/create_mini.sh <definition_name> <op_dir>

# Run
bash_scripts/run_local.sh mini
bash_scripts/run_local.sh full

# Run + dump current run trace to markdown
bash_scripts/run_local.sh mini-dump-trace-md
bash_scripts/run_local.sh full-dump-trace-md
```

`<op_dir>` values:
- `dsa_paged`
- `gdn`
- `moe`

`<definition_name>` values:
- `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64`
- `dsa_topk_indexer_fp8_h64_d128_topk2048_ps64`
- `gdn_decode_qk4_v8_d128_k_last`
- `gdn_prefill_qk4_v8_d128_k_last`
- `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048`

## 1) Environment Setup

```bash
uv venv
uv pip install flashinfer-bench
```

If `flashinfer_bench.agents` is missing:

```bash
uv pip install git+https://github.com/flashinfer-ai/flashinfer-bench.git
```

## 2) Dataset Path

Default full dataset path used by scripts:

```bash
/home/simon/flashinfer-competition/mlsys26-contest
```

Override with:

```bash
export FIB_FULL_DATASET_PATH=/your/full/dataset/path
```

## 3) Create a Mini Dataset (Single Workload Row)

```bash
bash_scripts/create_mini.sh <definition_name> <op_dir> [output_dir]
```

`<op_dir>` values:
- `dsa_paged`
- `gdn`
- `moe`

`<definition_name>` values:
- `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64`
- `dsa_topk_indexer_fp8_h64_d128_topk2048_ps64`
- `gdn_decode_qk4_v8_d128_k_last`
- `gdn_prefill_qk4_v8_d128_k_last`
- `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048`

Example:

```bash
bash_scripts/create_mini.sh gdn_decode_qk4_v8_d128_k_last gdn
```

This creates `mini_datasets/<definition_name>_single` by default.

## 4) Run Benchmarks

All run modes are in one script:

```bash
bash_scripts/run_local.sh full
bash_scripts/run_local.sh mini [mini_dataset_name]
bash_scripts/run_local.sh full-dump-trace-md
bash_scripts/run_local.sh mini-dump-trace-md [mini_dataset_name]
```

Notes:
- `mini_dataset_name` defaults to `<definition>_single` from `config.toml`.
- `*-dump-trace-md` runs the benchmark, then writes trace entries from the current run to a timestamped markdown file in `logs/`.
  - File pattern: `logs/<definition>_<run-timestamp>.md`
  - Override output folder with: `FIB_RUN_LOG_DIR=/your/path`

## 5) Manual Command (Equivalent)

```bash
FIB_DATASET_PATH=<dataset_path> uv run python scripts/run_local.py
```

`run_local.py` calls `scripts/pack_solution.py` automatically.

## 6) Common Status Meanings

- `COMPILE_ERROR`: build/signature/entrypoint mismatch.
- `RUNTIME_ERROR`: callable raised at runtime.
- `INCORRECT_NUMERICAL`: output mismatch.
- `PASSED`: correctness check passed.
