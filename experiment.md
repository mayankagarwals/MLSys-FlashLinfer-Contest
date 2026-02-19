# Experiment: Small-Kernel Single-Tile Specialization

Date: 2026-02-19

## Goal
Evaluate experiment 1 from NCU analysis: specialize the small-batch kernel for the single-tile case to remove cp.async pipeline overhead.

## Code change
File:
- `solution/python/gdn_decode_cutedsl.py`

What changed:
- In `gdn_decode_kernel_small_batch_pretranspose`, replaced cp.async prefetch/pipeline loop with direct single-tile shared-memory load and compute path for the small-kernel shape.
- Kept numerical path and writeback semantics intact.

## Where tested
Remote host:
- `mayank@95.133.252.11` (`verda-b200`)

Repo path:
- `/home/mayank/codebases/MLSys-FlashLinfer-Contest`

## Commands run
Mini sanity:
- `./bash_scripts/run_local.sh mini-dump-trace-md`

Full benchmark:
- `FIB_FULL_DATASET_PATH=/home/mayank/datasets/mlsys26-contest ./bash_scripts/run_local.sh full-dump-trace-md`

## Result summary
Status:
- Mini: `PASSED`
- Full: `20/20 PASSED`

Performance comparison vs provided baseline:
- Baseline avg speedup: `55.38x`
- Experiment avg speedup: `55.00x`
- Delta: `-0.37x` (~`-0.67%`)

Conclusion:
- Experiment is functionally correct but not a net performance win on this run.

## Logs
Remote logs:
- `/home/mayank/codebases/MLSys-FlashLinfer-Contest/logs/gdn_decode_qk4_v8_d128_k_last_2026-02-19T16-53-27-589901.md`
- `/home/mayank/codebases/MLSys-FlashLinfer-Contest/logs/gdn_decode_qk4_v8_d128_k_last_2026-02-19T16-53-52-928362.md`
