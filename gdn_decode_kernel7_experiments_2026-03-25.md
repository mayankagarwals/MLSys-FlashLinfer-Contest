# GdnDecodeKernel7 Experiment Notes (2026-03-25)

This note summarizes the optimization experiments run on `GdnDecodeKernel7` on `verda-b200` on March 25, 2026.

## Baseline

Kernel:
- `GdnDecodeKernel7`

Important baseline observations from the March 25, 2026 NCU report:
- Grid: `256` CTAs, block: `128` threads
- Duration: about `5.47 us`
- Achieved occupancy: about `11.86%`
- Active warps per scheduler: `1.74`
- Eligible warps per scheduler: `0.15`
- Dominant stall reason: `long scoreboard`
- Top long-scoreboard consumers:
  - `67.57%` at the first consumer of the `a` load
  - `29.73%` at the first consumer of the packed `q` load

Interpretation:
- The kernel is underfilled at the GPU level.
- The visible hotspots are not just "expensive instructions"; they are the first consumers of late-arriving data.

## Experiment Summary

| Experiment | Hypothesis | Mini result | Outcome |
| --- | --- | --- | --- |
| Tile-2 CTA reuse | Process 2 `tile_idx` values per CTA and reuse `q`, `k`, `g`, `beta` | `3.699 us -> 3.789 us` | Failed |
| Warp-broadcast scalar path | Have one lane compute/load scalar gating terms and broadcast within warp | `3.709 us -> 4.150 us` | Failed |
| Precompute `g/beta` helper kernel | Remove scalar gating from decode entirely | `3.706 us -> 9.369 us` | Failed badly |
| CTA-shared `q/k` cache | Stage `q` and `k` once per CTA in shared memory | `3.702 us -> 3.779 us` | Failed |

## 1. Tile-2 CTA Reuse

Change:
- One CTA processed `2` `tile_idx` values instead of `1`.
- `q_vec`, `k_vec`, `g`, and `beta` were hoisted so they could be reused across both tiles.

Hypothesis:
- Reduce redundant per-tile setup and some exposed long-scoreboard waits without changing the math.

Measured result:
- Baseline mini: `3.699 us`
- Candidate mini: `3.789 us`
- Delta: about `+2.4%`

Why it failed:
- Grid size dropped from `256` CTAs to `128`.
- That cut active and eligible warps per scheduler roughly in half.
- NCU showed the local dependency picture improved slightly, but the scheduler had fewer warps available to hide the remaining latency.

Useful learning:
- A small per-warp improvement is not enough if the change collapses launch-level latency hiding.

Reference logs:
- Baseline: `/home/mayank/codebases/MLSys-FlashLinfer-Contest-gdn-v7-baseline-20260325/logs/gdn_decode_qk4_v8_d128_k_last_2026-03-25T11-24-23-718702.md`
- Candidate: `/home/mayank/codebases/MLSys-FlashLinfer-Contest-gdn-v7-tile2-20260325/logs/gdn_decode_qk4_v8_d128_k_last_2026-03-25T11-24-58-187341.md`

## 2. Warp-Broadcast Scalar Path

Change:
- One lane per warp loaded/computed the scalar gating path (`a`, `A_log`, `dt_bias`, `b` -> `g`, `beta`).
- The values were broadcast to the rest of the warp with `__shfl_sync`.

Hypothesis:
- Remove redundant scalar loads and reduce the hot `a`-side long-scoreboard path without changing the CTA count.

Measured result:
- Baseline mini: `3.709 us`
- Candidate mini: `4.150 us`
- Delta: about `+11.9%`

Why it failed:
- Stall attribution moved away from the old `a` hotspot, so the change was not pointless.
- But overall long-scoreboard got worse.
- The bottleneck redistributed into other paths, especially `k` unpack and `state` use.
- The serialized scalar handling did not create enough benefit to outweigh the new dependency mix.

Useful learning:
- Moving stall attribution is not the same as reducing total stall cost.

Reference logs:
- Baseline: `/home/mayank/codebases/MLSys-FlashLinfer-Contest-gdn-v7-baseline-20260325/logs/gdn_decode_qk4_v8_d128_k_last_2026-03-25T12-28-24-880207.md`
- Candidate: `/home/mayank/codebases/MLSys-FlashLinfer-Contest-gdn-v7-warp-bcast-20260325/logs/gdn_decode_qk4_v8_d128_k_last_2026-03-25T12-27-46-355092.md`

## 3. Precompute `g` and `beta` in a Helper Kernel

Change:
- Added a helper kernel to precompute per-head `float2(g, beta)`.
- The decode kernel then loaded `scalar_pairs[hv_base]` instead of computing `g` and `beta` inline.

Hypothesis:
- Remove the hottest scalar gating path from the decode kernel and simplify the decode critical path.

Measured result:
- Baseline mini: `3.706 us`
- Candidate mini: `9.369 us`
- Delta: about `2.53x` slower

NCU result:
- `PrecomputeGdnScalarPairsKernel`: about `5.18 us`
- `GdnDecodeKernel7`: about `5.98 us`

Why it failed:
- The helper kernel launch was extremely expensive relative to a `~5 us` decode kernel.
- More importantly, the decode kernel itself also got slower.
- The old scalar gating chain was removed, but decode now had a new global-memory dependency on `scalar_pairs`.
- The recurrence path was exposed earlier:
  - `g * state`
  - `dot(k, old_state)`
  - update
  - `dot(q, updated)`
- NCU showed the old `a` hotspot disappeared, but the new decode hotspots became the first `k` use and the first `g * state` use.

Useful learning:
- In this kernel, the scalar gating path was not purely overhead. It also provided instruction distance before the recurrence consumed `state` and `k`.

Reference logs and report:
- Baseline: `/home/mayank/codebases/MLSys-FlashLinfer-Contest-gdn-v7-baseline-20260325/logs/gdn_decode_qk4_v8_d128_k_last_2026-03-25T12-54-37-981714.md`
- Candidate: `/home/mayank/codebases/MLSys-FlashLinfer-Contest-gdn-v7-precompute-20260325-182043/logs/gdn_decode_qk4_v8_d128_k_last_2026-03-25T12-54-02-176633.md`
- Direct NCU report: `/Users/mayaagar/Documents/Linkedin/codebases/MLSys-FlashLinfer-Contest/ncu_logs/from_verda_b200/gdn_decode_qk4_v8_d128_k_last_2026-03-25T12-57-44_precompute_direct/ncu_report.ncu-rep`

## 4. CTA-Shared `q/k` Cache

Change:
- Restored baseline `v7` behavior.
- Added `__shared__` staging for one `q` head and one `k` head per CTA.
- All four warps in the block reused the staged values after one `__syncthreads()`.

Hypothesis:
- Since all four warps in a CTA read the same `q` and `k` head, shared-memory staging should remove 4x redundant global `q/k` loads while preserving the original launch shape.

Measured result:
- Baseline mini: `3.702 us`
- Candidate mini: `3.779 us`
- Delta: about `+2.1%`

Why it failed:
- The saved global loads were not enough to pay for:
  - shared-memory traffic
  - one CTA-wide barrier
- This kernel is small enough that the barrier cost appears to matter.

Useful learning:
- Even obviously redundant traffic is not automatically worth removing if the fix adds synchronization.

Reference logs:
- Baseline: `/home/mayank/codebases/MLSys-FlashLinfer-Contest-gdn-v7-baseline-20260325/logs/gdn_decode_qk4_v8_d128_k_last_2026-03-25T13-39-18-776069.md`
- Candidate: `/home/mayank/codebases/MLSys-FlashLinfer-Contest-gdn-v7-shared-qk-20260325-185829/logs/gdn_decode_qk4_v8_d128_k_last_2026-03-25T13-38-16-106527.md`

## Main Takeaways

1. This kernel is extremely sensitive to any change that reduces latency hiding at the scheduler level.

2. The hottest visible long-scoreboard site is not automatically the best optimization target. The fix can easily be worse than the hotspot if it:
- reduces CTA count
- adds a barrier
- serializes work
- replaces ALU work with another memory dependency

3. The real algorithmic core is the recurrence:
- `old_state = g * state`
- `old_v = dot(k, old_state)`
- `updated = old_state + k * beta * (v - old_v)`
- `out = dot(q, updated)`

4. The scalar gating path can look hot in the source view, but in some versions it also helped hide latency before the recurrence started consuming `state` and `k`.

5. So far, every successful-looking local idea reduced one visible cost while introducing a more expensive global cost.
