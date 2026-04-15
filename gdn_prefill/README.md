# GDN Prefill

Changelog
- `triton_v1`: Initial Triton kernels, adapted from vLLM (modified from FLA). Fused beta and gate computationg to kkt kernel. Fuse W and U computation to inverse kernel.
- `triton_v2`: Replace all TMAs (`tl.make_tensor_descriptor()`) with pointers.
- `triton_v2b`: Add metadata kernel to avoid CUDA sync and CPU ops. There is still 1 CUDA sync (copy `total_num_chunks` to CPU).
- `triton_v3`: Fuse O kernel with H kernel. Generally slower. Not used.
- `triton_v4`: Use Neumann series for 16x16 inverse (instead of forward substitution): `inv(I + A) = I - A + A^2 - A^3 + ... + A^14 - A^15 = (I - A)(I + A^2)(I + A^4)(I + A^8)` -> use MMA.
- `chunk_v5`: Replace kkt kernel with CUDA C++. Persistent kernel (over all chunks) with pipelining.
- `chunk_v6`: Improve inverse kernel: TF32 MMA for everything, add Newton-Schulz refinement (`new Ai = Ai @ (2I - (I + A) @ Ai)`, MMA done in TF32x3), no round-trip to L2 for WU computation.
- `chunk_v6b`: Eliminate CUDA sync, and kernels read `total_num_chunks` from CUDA memory directly. For inverse and O kernels, launch excessive threadblocks (there is an upper bound), where the extra threadblocks can early exit. kkt kernel is persistent, and H kernel launches `num_seqs x num_heads` threadblocks, so they are not affected.
- `chunk_v6c`: Improve inverse kernel: BF16 MMA for Neuman, TF32 MMA for Newton-Schulz, off-diagonal blocks use TF32x3.
- `chunk_v7`: 
