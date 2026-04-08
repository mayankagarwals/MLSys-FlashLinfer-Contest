Here are some optimization ideas on optimize the cuda_parallel_v1.md.
We learn those ideas in different implementations and do in cuda.

Learn and try all of these ideas until you make the cuda parallel version faster than the triton v4 version performance. Don't stop iteration until you achieve it. If run out of idea, reread all of the ideas below. Do your best.

1. cuLA re-implement the FLA kernels (that we also base on) in CuteDSL, learn the idea from https://github.com/inclusionAI/cuLA/tree/main/cula/ops implementation and adapt to cuda.

2. it's quite tricky because 2nd MMA depends on the result of 1st MMA. and we need to use CUDA cores to do all of the scaling and stuff between the 2 MMAs. it means that if it's not pipelined well, basically Tensor cores execute once, wait for CUDA cores, then execute again -> can't reach good utilization
basically we would need something like FA4 
back-to-back MMA, with CUDA cores processing in between

3. learn inverse formula in triton v4, i think we can fuse chunk_scaled_dot_kkt_fwd_kernel and merge_16x16_to_64x64_inverse_kernel. 

4. learn the optimization did for cudadsl - insane speedup by fusing the chunk metadata preparation logic into a single triton kernel https://github.com/mayankagarwals/MLSys-FlashLinfer-Contest/pull/54
around 100us faster across all workloads

5. Run ncu and iterate on it.

6. Learn optimization for b200 cuda from https://gau-nernst.github.io/tcgen05/, https://github.com/gau-nernst/learn-cuda/tree/3b90ac9b/02e_matmul_sm100/. Try out each details.

7. Some Concrete ideas:
1. Eliminate the sequential dot product in Step 4
The intra-chunk ((Q @ K.T) * M') @ V' is computed with a manual loop:
cudafor (int j = 0; j <= t; j++)
    dot += s_wh[t][j] * s_vnew[j][bv];
This is scalar sequential code on a GPU — very underutilized. Options:

Reformulate as a tcgen05 MMA by reshaping the masked attention matrix and V' into tile layout, then mask out upper triangle by zeroing tiles before MMA
Use warp-level reductions or shared memory parallel reduction instead of a serial loop
Split into two passes: a full (non-causal) matmul via tcgen05 minus a correction for the upper triangle

2. Increase BV from 32 to 64
Currently each block handles only 32 value columns (BV=32), requiring 4 blocks per (seq, head). Doubling to BV=64 would halve the number of blocks and halve the redundant work — steps 1, 3, 4's Q @ K.T and the attention masking are identical across all v_tiles but recomputed 4 times. The cost is more shared memory for s_h (doubles from ~16KB to ~33KB), which may still fit given the ~85KB budget.
3. Pipeline chunk iterations
Currently each chunk iteration is fully serial: step 1 finishes before step 2 starts. With double-buffering of tiles, you could overlap:

TMA load of W for chunk ct+1 while step 5 of chunk ct runs
Prefetch U for chunk ct+1 during step 5's state update MMA

This would hide more of the TMA and global memory latency across chunk boundaries.
4. Fuse Kernels 2a + 2b + 3
Currently three separate kernel launches with global memory round-trips for A_mat between them. Since all three operate on the same (C, C) data per chunk:

Kernel 2a writes A_mat to global → Kernel 2b reads it back → writes inv → Kernel 3 reads it back
Fusing would keep A in shared memory or registers throughout, eliminating two global memory round-trips of the C×C matrix (64×64×4 bytes = 16KB per chunk per head)

5. Overlap Kernel 4 across sequences
Currently the grid is (kNVT, num_seqs * kHv), so different sequences already run in parallel. But if some sequences are much longer than others, SMs handling short sequences finish early and sit idle. A persistent kernel approach could let idle SMs pick up chunks from longer sequences via a work-stealing queue.
6. SolveTrilKernel parallelism
The forward substitution is inherently sequential across rows (row i depends on rows 0..i-1). But the 64 column solves are independent. Currently this uses 64 threads — one per column. On Blackwell with 128-thread blocks, half the threads are wasted. Possible improvements:

Use 2 threads per column for parallel reduction within each row's dot product
Block-recursive inversion: split the 64×64 into four 32×32 blocks, invert the top-left, use it to compute the bottom-left, etc. The 32×32 inversions can use more parallelism

7. Reduce global memory traffic for Kernel 3
W and U are written to global memory by Kernel 3, then read back by Kernel 4. If Kernel 3 and Kernel 4 could be fused (at least for the first chunk of each sequence), the W and U data could stay in shared memory. This is hard in general because Kernel 3 runs across all chunks in parallel while Kernel 4 is sequential, but for the common case of short sequences (1-2 chunks), a specialized path could help.
8. Use tcgen05 for the state update accumulation
In Step 5, after the MMA computes K.T @ gated_V', the TMEM results are added to s_h element by element. If s_h were kept in tile layout (bf16) rather than fp32, the accumulation could potentially be done as another MMA with an identity-like operand, avoiding the TMEM read + scalar add + shared memory write cycle. The tradeoff is precision loss from bf16 state.
9. Precompute Q @ K.T in a separate kernel
The Q @ K.T computation in Step 4 has no dependency on state S. It could be precomputed for all chunks in parallel (like W and U) and stored as a (C, C) matrix. Kernel 4 would then just load and apply the mask. This trades global memory bandwidth for removing one MMA from the critical sequential path.
10. Async copy for U in Kernel 4
Currently u_in is read from global memory with regular loads in Step 2. Since TMA is only used for W, Q, K (which have head-grouped strides matching TMA descriptor layouts), U uses scalar loads. Creating a TMA descriptor for U and async-loading it during Step 1's MMA would hide that latency.

