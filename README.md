# FlashInfer Competition Starter

Minimal local workflow for iterating on kernels with `flashinfer-bench`.
Agent-specific workflow details are indexed in `AGENTS.md` and stored as skills in `.agents/skills/`.

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

# Run benchmark repeatedly and report per-workload mean/std latency
bash_scripts/run_local_stats.sh full [runs]
bash_scripts/run_local_stats.sh mini [mini_dataset_name] [runs]
bash_scripts/run_local_stats.sh [runs]

# NCU profile (all selected workloads / one profiled forward pass each)
bash_scripts/run_ncu.sh mini
bash_scripts/run_ncu.sh full

# GDN experiment loop (mini preflight + full benchmark + LOGS.md append + revert)
bash_scripts/run_gdn_experiment.sh "<hypothesis>" "<change_summary>" [decision]
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

Repeated-run stats command:

```bash
bash_scripts/run_local_stats.sh full [runs]
bash_scripts/run_local_stats.sh mini [mini_dataset_name] [runs]
bash_scripts/run_local_stats.sh [runs]
```

- Default `runs` is `10`.
- Shorthand: passing only `runs` is treated as `full` mode (e.g. `bash_scripts/run_local_stats.sh 10`).
- The script runs `scripts/run_local.py` repeatedly and slices new trace rows after each run.
- Aggregates per-workload `latency_ms` and prints: `n_pass`, `n_seen`, `mean_ms`, `std_ms`, `min_ms`, `max_ms`.

## 5) Manual Command (Equivalent)

```bash
FIB_DATASET_PATH=<dataset_path> uv run python scripts/run_local.py
```

`run_local.py` calls `scripts/pack_solution.py` automatically.

## 6) NCU Single-Pass Profiling

Use this for a simple Nsight Compute run without benchmark loops:

```bash
bash_scripts/run_ncu.sh mini [mini_dataset_name]
bash_scripts/run_ncu.sh full
```

Default behavior:
- Profiles all workloads for the current `config.toml` definition.
- Runs one profiled forward pass per workload.
- Uses native API: `flashinfer_bench.agents.ncu.flashinfer_bench_run_ncu`.
- Applies NVTX include expression `flashinfer_bench_ncu_profile]` for push/pop range matching.
- Uses `--target-processes all`.
- Uses kernel filter `regex:kernel_cutlass` by default.
- Writes artifacts to `ncu_logs/<definition>_<timestamp>/workloads/<index>_<uuid>/`.
  - Includes `ncu_report.ncu-rep` per workload.

Note:
- `ncu` here is one-pass profiling. `run_local.sh` latency is an average over many iterations with cache-clearing in the benchmark loop, so values will not match exactly.

Useful overrides:

```bash
FIB_NCU_OUTPUT_DIR=/your/path
FIB_NCU_KERNEL_NAME='regex:kernel_cutlass'
FIB_NCU_SET=basic
FIB_NCU_PAGE=details
FIB_NCU_DEVICE=cuda:0
FIB_NCU_TARGET_PROCESSES=all
FIB_NCU_WORKLOAD_SCOPE=first
FIB_NCU_MAX_WORKLOADS=5
FIB_NCU_TIMEOUT=300
FIB_NCU_SECTIONS='LaunchStats,Occupancy'
FIB_NCU_NVTX_INCLUDE='flashinfer_bench_ncu_profile]'
```

## 7) Common Status Meanings

- `COMPILE_ERROR`: build/signature/entrypoint mismatch.
- `RUNTIME_ERROR`: callable raised at runtime.
- `INCORRECT_NUMERICAL`: output mismatch.
- `PASSED`: correctness check passed.

## 8) Ephemeral GDN Experiment Loop

For iterative tuning of `solution/python/gdn_decode_cutedsl.py`, use:

```bash
bash_scripts/run_gdn_experiment.sh "<hypothesis>" "<change_summary>" [decision]
```

It will:
- Require a local edit in `solution/python/gdn_decode_cutedsl.py`.
- Run mini preflight (`run_local.sh mini`).
- Run full benchmark (`run_local.sh full-dump-trace-md`) if mini passes.
- Append metrics to root `LOGS.md`.
- Revert `solution/python/gdn_decode_cutedsl.py` to `HEAD`.
