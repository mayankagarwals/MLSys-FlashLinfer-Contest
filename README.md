# FlashInfer Competition Starter (Fast Local Iteration)

This repo is set up for fast local benchmarking with `uv` and `flashinfer-bench`.

Working examples in this repo:
- Triton MoE: `solution/triton/moe_le_fused_entry.py`
- Pure Python GDN decode: `solution/python/gdn_decode_reference.py`

---

## 1) Quick Setup (uv)

```bash
uv venv
uv pip install flashinfer-bench
```

If you hit `No module named 'flashinfer_bench.agents'`:

```bash
uv pip install git+https://github.com/flashinfer-ai/flashinfer-bench.git
```

Set dataset path:

```bash
export FIB_DATASET_PATH=/home/simon/flashinfer-competition/mlsys26-contest
```

Run benchmark:

```bash
uv run python scripts/run_local.py
```

---

## 2) Working Workload Examples

## A) MoE (Triton)

File:
- `solution/triton/moe_le_fused_entry.py`

Config:

```toml
[solution]
name = "moe-triton-example"
definition = "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"
author = "team-name"

[build]
language = "triton"
entry_point = "moe_le_fused_entry.py::kernel"
```

Mini benchmark:

```bash
FIB_DATASET_PATH=$PWD/mini_datasets/moe_single uv run python scripts/run_local.py
```

Full benchmark:

```bash
FIB_DATASET_PATH=/home/simon/flashinfer-competition/mlsys26-contest uv run python scripts/run_local.py
```

How to add a new Triton workload:
1. Put the entry file directly in `solution/triton/`.
2. Implement callable matching destination-passing style (`inputs..., outputs...`) unless you intentionally switch style.
3. Set exact definition name and `entry_point = "<file>.py::<fn>"` in `config.toml`.

## B) GDN Decode (Pure Python)

File:
- `solution/python/gdn_decode_reference.py`

Config:

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

Mini benchmark:

```bash
FIB_DATASET_PATH=$PWD/mini_datasets/gdn_decode_single uv run python scripts/run_local.py
```

Full benchmark:

```bash
FIB_DATASET_PATH=/home/simon/flashinfer-competition/mlsys26-contest uv run python scripts/run_local.py
```

How to add a new Python workload:
1. Put file directly in `solution/python/`.
2. For value-returning mode, set `destination_passing_style = false` and return outputs as `tuple` for multi-output definitions.
3. Set exact definition name and `entry_point = "<file>.py::<fn>"`.

---

## 3) How Mini-Benchmarks Were Created

Each mini dataset keeps one definition + one workload line for fast debugging.

Template:

```bash
NAME=<definition_name>
OP=<op_dir>  # e.g. moe, gdn, dsa_paged
ROOT=mini_datasets/${NAME}_single
SRC=/home/simon/flashinfer-competition/mlsys26-contest

mkdir -p "$ROOT/definitions/$OP" "$ROOT/workloads/$OP"
cp "$SRC/definitions/$OP/$NAME.json" "$ROOT/definitions/$OP/"
head -n 1 "$SRC/workloads/$OP/$NAME.jsonl" > "$ROOT/workloads/$OP/$NAME.jsonl"
ln -sfn "$SRC/blob" "$ROOT/blob"
```

Why the `blob` symlink matters:
- workload JSONL references `./blob/...`; without this symlink, safetensor inputs fail to load.

---

## 4) Keep in Mind

- `definition` must be an exact dataset definition name (no aliases like `fused_moe`).
- `entry_point` format is strict: `"<file_path>::<function_name>"`.
- `scripts/pack_solution.py` packs top-level files only (non-recursive) in language folder.
- Status meanings:
  - `RUNTIME_ERROR`: callable or kernel crashed.
  - `COMPILE_ERROR`: build/signature mismatch.
  - `INCORRECT_NUMERICAL`: runs, but incorrect outputs.
  - `PASSED`: correctness check passed.
- Trace details are written under `FIB_DATASET_PATH/traces/...`.
