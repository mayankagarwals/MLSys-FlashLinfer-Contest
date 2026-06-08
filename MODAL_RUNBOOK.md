# Running the GDN benchmarks on Modal (from any machine, incl. macOS)

Benchmark the **final** GDN decode/prefill solutions on a remote **B200** via
[Modal](https://modal.com). The host only needs the `modal` client; all CUDA/GPU
work (packing, compiling, benchmarking) happens remotely.

## Steps

```bash
# 0. from the repo root
cd /path/to/MLSys-FlashLinfer-Contest

# 1. local client only (no GPU/Linux-only deps needed on the host)
pip install modal huggingface-hub          # or: uv pip install / conda

# 2. Modal auth (one-time)
modal setup

# 3. dataset -> Modal volume (one-time; see the volume gotcha below)
DS=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('flashinfer-ai/mlsys26-contest', repo_type='dataset'))")
modal volume create flashinfer-trace
modal volume put flashinfer-trace "$DS/definitions" /definitions
modal volume put flashinfer-trace "$DS/workloads"   /workloads
modal volume put flashinfer-trace "$DS/solutions"   /solutions
modal volume put flashinfer-trace "$DS/blob"        /blob

# 4. run on B200 (first run builds the image — a few min; cached after)
modal run gdn_decode/scripts/run_modal.py                        # all decode workloads
modal run gdn_prefill/scripts/run_modal.py                       # all prefill workloads
modal run gdn_prefill/scripts/run_modal.py --workload-uuid 77daf91d-0660-4c4b-8c32-336a69281cd9   # one workload
```

`--workload-uuid <id>` requires the **full** UUID (exact match); omit it to run
all. On any non-`PASSED` status the runner prints the captured compiler/runtime
log inline under a `--- log ---` block.

## Dataset & volume: the root-layout gotcha

The remote code does `TraceSet.from_path("/data")`, so `definitions/`,
`workloads/`, `solutions/`, `blob/` must sit **at the volume root**.

`modal volume put <vol> <dir>` appends the source folder's basename when the
remote path ends in `/` (the default) — so `modal volume put ... "$DS"` lands
everything under `/<snapshot-hash>/...` (wrong), and `modal volume cp -r` can't
fix it (no recursive copy on V1 volumes). Upload each entry with an explicit,
no-trailing-slash remote path, as in step 3. Verify:

```bash
modal volume ls flashinfer-trace
# expect: definitions, workloads, solutions, blob   (NOT a single hash-named dir)
# if you see a hash dir: modal volume rm -r flashinfer-trace <snapshot-hash>, then re-upload per-dir
```

## Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| `command not found: modal` | `pip install modal` (and ensure its bin dir is on `PATH`). |
| `Volume 'flashinfer-trace' not found` | Run `modal volume create flashinfer-trace` first. |
| Volume contents nested under a hash dir | You uploaded the whole snapshot. Re-upload per-dir with explicit no-slash remote paths (above). |
| `recursive is not supported for V1 volumes` | Can't `modal volume cp -r`. Re-upload per-dir instead. |
