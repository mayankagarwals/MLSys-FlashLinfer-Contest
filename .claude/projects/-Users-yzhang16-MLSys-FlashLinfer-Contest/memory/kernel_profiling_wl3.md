---
name: Kernel profiling results for WL3
description: Per-kernel timing breakdown for CUDA v2 vs Triton v4 on WL3 (T=8192, 32 seqs)
type: project
---

# Kernel Profiling WL3 (T=8192, 32 seqs) — 2026-04-03

## CUDA v2 Breakdown (total ~2000us)
- HRecurrenceKernel: 817us (36%)
- FusedPrepKernel: 622us (28%)
- OOutputKernel: 528us (23%)
- PrepMeta+Preprocess+Memset: ~33us (1.5%)

## Triton v4 Breakdown (total ~251us)
- h_kernel: 107us (43%)
- merge_16x16_inverse_kernel: 64us (25%)
- o_kernel: 51us (20%)
- kkt_kernel: 23us (9%)
- compute_chunks: 6us (2%)

## Per-kernel Gaps
- H kernel: 817/107 = 7.6x
- Prep: 622/87 = 7.1x
- O kernel: 528/51 = 10.4x

**Key Insight**: ALL major kernels (H, Prep, O) are 7-10x slower. O-kernel has the worst ratio (10.4x).

## Triton h-kernel PTX Analysis
- Uses hybrid MMA: tcgen05 for MMA1 (w@h^T, K=128), mma.sync m16n8k16 for MMA2 (vnew^T@k)
- BV=16 (not 32): more parallelism (8 v-tiles), h fits in 16 regs/thread
- TMEM allocated with 32 columns (BN=32)
- h state kept in fp32 registers (accumulated via mma.sync, scaled via mul.f32)
- cp.async for ALL global→smem loads (software pipelined)
- stmatrix/ldmatrix for efficient smem↔register transfers
- Only 1 bar.sync per phase inside loop body
