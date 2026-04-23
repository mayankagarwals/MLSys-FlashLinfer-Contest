// Outputs:
//   chunk_offsets[N+1]: cumulative chunk count per sequence (chunk_offsets[N] = total)
//   chunk_indices[total_chunks, 2]: (seq_id, chunk_id) pairs

#include "tvm_ffi_utils.h"
#include <cstdint>

constexpr int WARP_SIZE = 32;
constexpr int BT = 64;

template<int NUM_WARPS>
__global__
__launch_bounds__(NUM_WARPS * WARP_SIZE)
void prep_meta_v2_kernel(
  const int64_t *cu_seqlens,
  int32_t *chunk_indices,   // [max_chunks, 2]
  int32_t *chunk_offsets,   // [N+1]
  int64_t N
) {
  const int tid = threadIdx.x;
  const int lane_id = tid % WARP_SIZE;
  const int warp_id = tid / WARP_SIZE;

  if (tid == 0)
    chunk_offsets[0] = 0;

  __shared__ int cu_num_chunks_smem[NUM_WARPS];
  int num_chunks = 0;

  // compute number of chunks
  if (tid < N)
    num_chunks = (cu_seqlens[tid + 1] - cu_seqlens[tid] + BT - 1) / BT;

  // parallel scan within a warp
  // illustrating for 4 lanes
  // lane  | lane0 | lane1 | lane2    | lane3
  // iter0 | a0    |    a1 |       a2 |          a3
  // iter1 | a0    | a0+a1 |    a1+a2 |       a2+a3
  // iter2 | a0    | a0+a1 | a0+a1+a2 | a0+a1+a2+a3
  int cu_num_chunks = num_chunks;
  for (int i = 1; i < WARP_SIZE; i *= 2) {
    int lower = __shfl_up_sync(0xFFFF'FFFF, cu_num_chunks, i);  // from lower lane
    if (lane_id >= i)
      cu_num_chunks += lower;
  }

  if constexpr (NUM_WARPS > 1) {
    // store warp sum
    if (lane_id == WARP_SIZE - 1)
      cu_num_chunks_smem[warp_id] = cu_num_chunks;
    __syncthreads();

    // add warp sum from lower warps
    for (int i = 1; i < NUM_WARPS; i++) {
      if (warp_id >= i)
        cu_num_chunks += cu_num_chunks_smem[i - 1];
    }
  }

  // write outputs
  if (tid < N) {
    const int seq_id = tid;
    chunk_offsets[1 + seq_id] = cu_num_chunks;

    // fill chunk_indices
    int bos = cu_num_chunks - num_chunks;
    for (int i = 0; i < num_chunks; i++)
      reinterpret_cast<int2 *>(chunk_indices)[bos + i] = int2{seq_id, i};
  }
}

void prep_meta_v2(
  TensorView cu_seqlens,    // [N+1]
  TensorView chunk_indices, // [total_num_chunks, 2]
  TensorView chunk_offsets  // [N+1]
) {
  const int64_t N = cu_seqlens.size(0) - 1;

  auto *cu_seqlens_ptr    = reinterpret_cast<const int64_t *>(cu_seqlens.data_ptr());
  auto *chunk_indices_ptr = reinterpret_cast<int32_t *>(chunk_indices.data_ptr());
  auto *chunk_offsets_ptr = reinterpret_cast<int32_t *>(chunk_offsets.data_ptr());

#define DISPATCH(NUM_WARPS) else if (N <= NUM_WARPS * WARP_SIZE) \
  prep_meta_v2_kernel<NUM_WARPS><<<1, NUM_WARPS * WARP_SIZE>>>(cu_seqlens_ptr, chunk_indices_ptr, chunk_offsets_ptr, N);

  // support up to N=256
  if (false) {}
  DISPATCH(1)
  DISPATCH(2)
  DISPATCH(4)
  DISPATCH(8)

#undef DISPATCH
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(prep_meta_v2, prep_meta_v2);
