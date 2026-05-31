# Running the GDN benchmarks on Modal from macOS (uv, from scratch)

This is a step-by-step runbook for benchmarking the **final** GDN decode/prefill
solutions on a remote **B200** via [Modal](https://modal.com), driving everything
from an Apple-Silicon Mac using a fresh `uv` environment.

It exists because the obvious path (`uv pip install -r requirements.txt` then
`modal run ...`) **does not work on macOS** — several dependencies are Linux/GPU
only. The steps below install a *minimal local driver* and a tiny shim so the
Mac can pack the solution and dispatch the GPU work to Modal.

---

## TL;DR (copy-paste)

```bash
# 0. from the repo root
cd /path/to/MLSys-FlashLinfer-Contest

# 1. fresh uv env + minimal LOCAL driver (no GPU/Linux-only packages)
uv venv --python 3.12
uv pip install modal pandas huggingface-hub torch pydantic safetensors docstring-parser pyyaml
uv pip install --no-deps "git+https://github.com/flashinfer-ai/flashinfer-bench.git"

# 2. macOS shim for the CUDA `flashinfer` package (import-only; never runs locally)
mkdir -p ~/.cache/flashinfer-macos-stub/flashinfer
cat > ~/.cache/flashinfer-macos-stub/flashinfer/__init__.py <<'PY'
"""macOS import stub for the CUDA `flashinfer` package. See MODAL_MACOS_RUNBOOK.md."""
PY
cat > ~/.cache/flashinfer-macos-stub/flashinfer/testing.py <<'PY'
def bench_gpu_time_with_cupti(*args, **kwargs):
    raise RuntimeError("flashinfer.testing is a macOS stub; benchmarking runs remotely on Modal.")
PY
export PYTHONPATH=~/.cache/flashinfer-macos-stub

# 3. Modal auth (one-time)
uv run modal setup

# 4. dataset -> Modal volume (see "Dataset & volume" for the root-layout gotcha)
DS=$(uv run python -c "from huggingface_hub import snapshot_download; print(snapshot_download('flashinfer-ai/mlsys26-contest', repo_type='dataset'))")
uv run modal volume create flashinfer-trace
uv run modal volume put flashinfer-trace "$DS/definitions" /definitions
uv run modal volume put flashinfer-trace "$DS/workloads"   /workloads
uv run modal volume put flashinfer-trace "$DS/solutions"   /solutions
uv run modal volume put flashinfer-trace "$DS/blob"        /blob

# 5. run on B200 (PYTHONPATH from step 2 must be exported in this shell)
uv run modal run gdn_decode/scripts/run_modal.py
uv run modal run gdn_prefill/scripts/run_modal.py
```

> Every `modal` invocation must see the shim, so keep `PYTHONPATH=~/.cache/flashinfer-macos-stub`
> exported in the shell (or prefix each command with it).

---

## Why this is fiddly on macOS

`modal run script.py` **imports the script in your local process** before it
ships anything to the cloud — it has to, in order to discover the `App` and the
`@app.local_entrypoint()`. `gdn_decode/scripts/run_modal.py` does
`from flashinfer_bench import ...` at the top, so `flashinfer_bench` must import
on your Mac.

But the normal dependency chain is Linux/GPU-only:

| Package | Why it fails on macOS |
|---|---|
| `cupti-python` (in `requirements.txt`) | no macOS wheels at all (Linux only) |
| `tilelang` (in `requirements.txt`) | no macOS wheels |
| `flashinfer-python` (dep of `flashinfer-bench`) | pulls `nvidia-cudnn-frontend`, which has **no macOS wheels** |

So a full `uv pip install -r requirements.txt` is unsatisfiable on a Mac.

The trick: the **local** side only needs to *pack* the solution and *serialize*
the request — it never touches a GPU. All CUDA work happens **remotely** in the
Modal image. So locally we install only the pure-Python/CPU pieces, skip
`flashinfer-python`, and stub the one symbol `flashinfer_bench` imports from it.

---

## Step 1 — uv env + minimal local driver

```bash
uv venv --python 3.12
uv pip install modal pandas huggingface-hub torch pydantic safetensors docstring-parser pyyaml
uv pip install --no-deps "git+https://github.com/flashinfer-ai/flashinfer-bench.git"
```

- `--no-deps` on `flashinfer-bench` is the key — it skips the unsatisfiable
  `flashinfer-python` (and friends). The packages on the line above it are
  `flashinfer-bench`'s *actual* runtime deps **minus** the CUDA one, all of which
  have macOS wheels.
- Python 3.12 is a safe choice for wheel availability; 3.13 also works for this
  minimal set. (The full `requirements.txt` additionally trips on `cupti-python`
  having no `cp313`/macOS wheel — another reason we don't install it locally.)

## Step 2 — the `flashinfer` macOS shim

`flashinfer_bench/bench/timing.py` does, at module load:

```python
from flashinfer.testing import bench_gpu_time_with_cupti
```

That's the **only** top-level reference to the CUDA `flashinfer` package in the
whole library (verified by grep). Since timing only runs during an *actual*
benchmark (which happens remotely), we satisfy the import with a stub:

```bash
mkdir -p ~/.cache/flashinfer-macos-stub/flashinfer
cat > ~/.cache/flashinfer-macos-stub/flashinfer/__init__.py <<'PY'
"""macOS import stub for the CUDA `flashinfer` package."""
PY
cat > ~/.cache/flashinfer-macos-stub/flashinfer/testing.py <<'PY'
def bench_gpu_time_with_cupti(*args, **kwargs):
    raise RuntimeError("flashinfer.testing is a macOS stub; benchmarking runs remotely on Modal.")
PY
export PYTHONPATH=~/.cache/flashinfer-macos-stub
```

Notes:
- The stub lives **outside** the repo and **outside** site-packages — it changes
  no project code and no managed environment. It's reversible (`rm -rf` the dir).
- `PYTHONPATH` is searched before site-packages, so this shadows nothing real and
  only affects the shell where you export it.
- The stub function raises if ever called locally — a guard, since it should
  never execute on the host.

Verify:

```bash
uv run python -c "from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet; print('OK')"
```

## Step 3 — Modal auth

```bash
uv run modal setup
```

Confirm it worked (cheap, no GPU):

```bash
uv run modal profile current   # prints your active profile
uv run modal app list          # authenticates against the server
```

## Step 4 — Dataset & volume (the root-layout gotcha)

The benchmark reads its trace set from a Modal **Volume** mounted at `/data`, and
the remote code does `TraceSet.from_path("/data")` — so `definitions/`,
`workloads/`, `solutions/`, `blob/` must sit **at the volume root**.

The dataset is the HuggingFace dataset `flashinfer-ai/mlsys26-contest`:

```bash
DS=$(uv run python -c "from huggingface_hub import snapshot_download; print(snapshot_download('flashinfer-ai/mlsys26-contest', repo_type='dataset'))")
echo "$DS"
uv run modal volume create flashinfer-trace
```

**Gotcha:** `modal volume put <vol> <dir>` appends the source folder's basename
whenever the *remote path ends in `/`* (and the default remote path is `/`). So
`modal volume put flashinfer-trace "$DS"` lands everything under
`/<snapshot-hash>/...` — wrong. You also **can't** fix it with
`modal volume cp -r` because *recursive copy isn't supported on V1 volumes*.

The fix: upload each top-level entry with an **explicit, no-trailing-slash**
remote path (which means "use exactly this path"):

```bash
uv run modal volume put flashinfer-trace "$DS/definitions" /definitions
uv run modal volume put flashinfer-trace "$DS/workloads"   /workloads
uv run modal volume put flashinfer-trace "$DS/solutions"   /solutions
uv run modal volume put flashinfer-trace "$DS/blob"        /blob
```

Verify the root layout:

```bash
uv run modal volume ls flashinfer-trace
# expect: definitions, workloads, solutions, blob   (NOT a single hash-named dir)
```

If you previously did the naive upload, remove the stray nested folder:

```bash
uv run modal volume rm -r flashinfer-trace <snapshot-hash>
```

## Step 5 — Run

Make sure `PYTHONPATH` (step 2) is exported in this shell, then:

```bash
uv run modal run gdn_decode/scripts/run_modal.py
uv run modal run gdn_prefill/scripts/run_modal.py
```

What happens:
1. Local: packs the final solution from `config.toml`
   (`gdn_decode_final.py::run` / `gdn_prefill_mix.py::run`) into a `Solution`.
2. First run builds the remote image (downloads torch/triton/cuDNN/CUDA — a few
   minutes; cached afterwards).
3. Spins up a **B200**, runs all workloads, prints status / latency / speedup /
   error, then **tears the container down automatically** (you pay only for the
   GPU-seconds actually used).

Each run prints a live link like
`https://modal.com/apps/<profile>/main/ap-xxxx` for streaming logs.

---

## Billing & cleanup

- The runner is serverless: `@app.function(gpu="B200:1", timeout=3600)` with no
  `keep_warm`/`min_containers`. It scales to zero when the entrypoint returns.
- `timeout=3600` is a 1-hour safety cap if a run hangs; normal runs are minutes.
- Sanity check nothing is lingering: `uv run modal app list` (should read
  `stopped` after a run).
- To undo the macOS shim entirely: `rm -rf ~/.cache/flashinfer-macos-stub` and
  `unset PYTHONPATH`.

---

## Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| `ModuleNotFoundError: No module named 'flashinfer_bench'` | Local driver not installed — run Step 1. |
| `No solution found ... cupti-python` / `nvidia-cudnn-frontend ... no macOS wheels` | You tried the full `requirements.txt` locally. Don't — use the minimal Step 1 set + Step 2 stub. |
| `ModuleNotFoundError: No module named 'flashinfer'` (from `timing.py`) | The shim isn't on `PYTHONPATH`. `export PYTHONPATH=~/.cache/flashinfer-macos-stub` (Step 2). |
| `Volume 'flashinfer-trace' not found` | Run `modal volume create flashinfer-trace` first. |
| Volume contents nested under a hash dir | You uploaded the whole snapshot. Re-upload per-dir with explicit no-slash remote paths (Step 4). |
| `recursive is not supported for V1 volumes` | Can't `modal volume cp -r`. Re-upload per-dir instead (Step 4). |
| Workloads return `COMPILE_ERROR` on the B200 | **Known issue (open).** The remote image installs `flashinfer-bench==0.1.2` from PyPI, while we pack locally with the newer git `dev98`. A schema/evaluator mismatch (or a missing CUDA compile flag) can cause the remote solution build to fail. See below. |

### Known issue: remote `COMPILE_ERROR`

In our first end-to-end run the pipeline worked — image built, B200 spun up,
workloads executed — but the decode workloads came back `COMPILE_ERROR`. Likely
culprits, in order of suspicion:

1. **Version skew.** The image installs `flashinfer-bench` from PyPI (`0.1.2`),
   but the solution is packed locally with the git build (`0.1.3.dev98`). Pin the
   image to the same git revision so packer and runner agree. In
   `gdn_decode/scripts/run_modal.py` the image is:
   ```python
   modal.Image.debian_slim(python_version="3.12")
       .pip_install("flashinfer-bench", "torch", "triton", "numpy")
   ```
   Changing the first arg to
   `"flashinfer-bench @ git+https://github.com/flashinfer-ai/flashinfer-bench.git"`
   makes the remote match the local packer. *(This is a script change — out of
   scope for a commands-only setup, but it's the most likely fix.)*
2. **CUDA toolchain.** The final decode solution JIT-compiles inline CUDA
   (`gdn_decode_small_batch.cu` via `tvm_ffi`). The image must contain `nvcc`
   and matching CUDA headers. `flashinfer-python` pulls `nvidia-cuda-nvcc`, but a
   full toolchain / arch flags (`sm_100` for B200) may still be needed.

Next step is to capture the remote build log for one workload (the Modal app
page → failed function → logs) to see the exact compiler error, then apply (1)
and/or (2).
