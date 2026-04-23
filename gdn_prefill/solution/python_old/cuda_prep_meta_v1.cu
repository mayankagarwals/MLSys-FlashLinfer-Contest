// cuda_prep_meta.cu — Compute chunk metadata from cu_seqlens
//
// Replaces Triton compute_chunks_kernel with a simple CUDA kernel.
// Independent of KKT — can be reused by any chunk-based pipeline.
//
// Outputs:
//   chunk_offsets[N+1]: cumulative chunk count per sequence (chunk_offsets[N] = total)
//   chunk_indices[total_chunks, 2]: (seq_id, chunk_id) pairs

#include "tvm_ffi_utils.h"
#include <cstdint>

constexpr int BT = 64;

__global__ void __launch_bounds__(128)
prep_meta_v1_kernel(
    const int64_t *cu_seqlens,
    int32_t *chunk_indices,   // [max_chunks, 2]
    int32_t *chunk_offsets,   // [N+1]
    int32_t *total_chunks_out,// [1]
    int64_t num_seqs
) {
  const int tid = threadIdx.x;

  // Phase 1: thread 0 computes chunk_offsets (sequential cumsum, N <= 57)
  if (tid == 0) {
    int32_t total = 0;
    chunk_offsets[0] = 0;
    for (int64_t i = 0; i < num_seqs; i++) {
      int64_t slen = cu_seqlens[i + 1] - cu_seqlens[i];
      int32_t nc = (int32_t)((slen + BT - 1) / BT);
      total += nc;
      chunk_offsets[i + 1] = total;
    }
    total_chunks_out[0] = total;
  }
  __syncthreads();

  // Phase 2: all 128 threads fill chunk_indices in parallel
  if (tid < num_seqs) {
    int32_t offset = chunk_offsets[tid];
    int64_t slen = cu_seqlens[tid + 1] - cu_seqlens[tid];
    int32_t nc = (int32_t)((slen + BT - 1) / BT);
    for (int32_t c = 0; c < nc; c++) {
      chunk_indices[(offset + c) * 2] = (int32_t)tid;
      chunk_indices[(offset + c) * 2 + 1] = c;
    }
  }
}

void prep_meta_v1(
  TensorView cu_seqlens,
  TensorView chunk_indices,
  TensorView chunk_offsets  // [N+1] — last element is total_chunks
) {
  const int64_t num_seqs = cu_seqlens.size(0) - 1;
  auto *cu_seqlens_ptr    = reinterpret_cast<const int64_t *>(cu_seqlens.data_ptr());
  auto *chunk_indices_ptr = reinterpret_cast<int32_t *>(chunk_indices.data_ptr());
  auto *chunk_offsets_ptr = reinterpret_cast<int32_t *>(chunk_offsets.data_ptr());
  auto *total_chunks_ptr  = chunk_offsets_ptr + num_seqs;

  prep_meta_v1_kernel<<<1, 128>>>(
    cu_seqlens_ptr, chunk_indices_ptr, chunk_offsets_ptr, total_chunks_ptr, num_seqs);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(prep_meta_v1, prep_meta_v1);
