# FlashInfer Competition Starter

Minimal local workflow for iterating on kernels with `flashinfer-bench`.

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

Full dataset path used in this repo:

```bash
export FIB_DATASET_PATH=/home/simon/flashinfer-competition/mlsys26-contest
```

## 3) Run Benchmark

```bash
uv run python scripts/run_local.py
```

`run_local.py` calls `scripts/pack_solution.py` automatically.

## 4) Create a Mini Dataset (Single Workload Row)

Use this for fast debug loops.

```bash
NAME=<definition_name>
OP=<op_dir>  # e.g. gdn, moe, dsa_paged
ROOT=mini_datasets/${NAME}_single
SRC=/home/simon/flashinfer-competition/mlsys26-contest

mkdir -p "$ROOT/definitions/$OP" "$ROOT/workloads/$OP"
cp "$SRC/definitions/$OP/$NAME.json" "$ROOT/definitions/$OP/"
head -n 1 "$SRC/workloads/$OP/$NAME.jsonl" > "$ROOT/workloads/$OP/$NAME.jsonl"
ln -sfn "$SRC/blob" "$ROOT/blob"
```

Run with mini dataset:

```bash
FIB_DATASET_PATH=$PWD/mini_datasets/${NAME}_single uv run python scripts/run_local.py
```

## 5) Generic Config Update for a New Kernel Run

Edit `config.toml`:

```toml
[solution]
name = "my-kernel-run"
definition = "<exact_definition_name>"
author = "team-name"

[build]
language = "python"  # or "triton"
entry_point = "<file_name>.py::<function_name>"
destination_passing_style = false  # for value-returning Python callables
```

Rules:
- `definition` must exactly match a dataset definition name.
- `entry_point` must be `"<file>::<function>"`.
- Put entry files directly in `solution/python/` or `solution/triton/`.
- For value-returning Python kernels, keep `destination_passing_style = false` and return a tuple for multi-output workloads.

## 6) Common Status Meanings

- `COMPILE_ERROR`: build/signature/entrypoint mismatch.
- `RUNTIME_ERROR`: callable raised at runtime.
- `INCORRECT_NUMERICAL`: output mismatch.
- `PASSED`: correctness check passed.
