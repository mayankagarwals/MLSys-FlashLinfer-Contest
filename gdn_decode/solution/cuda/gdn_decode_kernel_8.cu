#include "gdn_decode_utils.h"
#include "tvm_ffi_utils.h"

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <math.h>

namespace {

constexpr int kHeadSize = gdn_decode::kHeadSize;
constexpr int64_t kNumQHeads = gdn_decode::kNumQHeads;
constexpr int64_t kNumKHeads = gdn_decode::kNumKHeads;
constexpr int64_t kNumVHeads = gdn_decode::kNumVHeads;

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 4;
constexpr int kNumThreads = kWarpSize * kWarpsPerBlock;
constexpr int kRowsPerBlock = kWarpsPerBlock;
constexpr int kElemsPerLane = kHeadSize / kWarpSize;
constexpr int kSmallNumVTiles = kHeadSize / kRowsPerBlock;

constexpr int kTileV = 8;
constexpr int kNumVTiles = kHeadSize / kTileV;
constexpr int kRowsPerIter = kWarpsPerBlock;
constexpr int kItersPerTile = kTileV / kRowsPerIter;

constexpr int kNumCTAs = 8;
constexpr int kVTilesPerCTA = kNumVTiles / kNumCTAs;
constexpr int kNumStages = 2;

constexpr int64_t kQGroupSize = kNumVHeads / kNumQHeads;
constexpr int64_t kKGroupSize = kNumVHeads / kNumKHeads;
constexpr unsigned kFullWarpMask = 0xffffffffu;

constexpr int64_t kBatchThreshold = 4;

constexpr int kLocalVCount = kTileV * kVTilesPerCTA;

static_assert(kHeadSize == 128);
static_assert(kNumThreads == 128);
static_assert(kElemsPerLane == 4);
static_assert(kVTilesPerCTA == 2);
static_assert(kItersPerTile == 2);

// ============================================================================
// Device helpers
// ============================================================================

__device__ __forceinline__ float SoftplusStable(float x) {
  const float abs_x = fabsf(x);
  return log1pf(expf(-abs_x)) + fmaxf(x, 0.0f);
}

__device__ __forceinline__ float Sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float WarpAllReduceSum(float value) {
#pragma unroll
  for (int mask = kWarpSize / 2; mask > 0; mask >>= 1) {
    value += __shfl_xor_sync(kFullWarpMask, value, mask);
  }
  return value;
}

__device__ __forceinline__ float4
LoadBf16x4GlobalNc(const __nv_bfloat16 *__restrict__ ptr) {
  float4 out;
  asm volatile("{\n\t"
               ".reg .b16 h<4>;\n\t"
               "ld.global.nc.L1::evict_first.v4.b16 {h0, h1, h2, h3}, [%4];\n\t"
               "cvt.rn.f32.bf16 %0, h0;\n\t"
               "cvt.rn.f32.bf16 %1, h1;\n\t"
               "cvt.rn.f32.bf16 %2, h2;\n\t"
               "cvt.rn.f32.bf16 %3, h3;\n\t"
               "}\n"
               : "=f"(out.x), "=f"(out.y), "=f"(out.z), "=f"(out.w)
               : "l"(ptr));
  return out;
}

__device__ __forceinline__ void
StoreF32x4Global(float *addr, const float4 &value) {
  asm volatile(
      "st.global.v4.f32 [%0], {%1, %2, %3, %4};"
      :
      : "l"(addr), "f"(value.x), "f"(value.y), "f"(value.z), "f"(value.w));
}

__device__ __forceinline__ void CpAsyncCg16(float *smem_ptr,
                                            const float *gmem_ptr) {
  const uint32_t smem_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "cp.async.cg.shared.global [%0], [%1], 16;"
      :
      : "r"(smem_addr), "l"(gmem_ptr)
      : "memory");
}

__device__ __forceinline__ void CpAsyncCommit() {
  asm volatile("cp.async.commit_group;" ::: "memory");
}

__device__ __forceinline__ void CpAsyncWaitAll() {
  asm volatile("cp.async.wait_all;" ::: "memory");
}

// ============================================================================
// Small-batch kernel: grid = B * HV * 32, no shared memory.
// ============================================================================

__global__ void GdnDecodeSmallBatch(
    const __nv_bfloat16 *__restrict__ q, const __nv_bfloat16 *__restrict__ k,
    const __nv_bfloat16 *__restrict__ v, const float *__restrict__ state,
    const float *__restrict__ A_log, const __nv_bfloat16 *__restrict__ a,
    const float *__restrict__ dt_bias, const __nv_bfloat16 *__restrict__ b,
    float scale, __nv_bfloat16 *__restrict__ output,
    float *__restrict__ new_state) {

  const int64_t block_linear = static_cast<int64_t>(blockIdx.x);
  const int64_t tile_idx = block_linear % kSmallNumVTiles;
  const int64_t bh = block_linear / kSmallNumVTiles;

  const int64_t batch_idx = bh / kNumVHeads;
  const int64_t hv_idx = bh % kNumVHeads;
  const float negated_exp_A_log = -expf(A_log[hv_idx]);

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & (kWarpSize - 1);

  const int64_t v_idx = tile_idx * kRowsPerBlock + warp_id;

  const int64_t q_head = hv_idx / kQGroupSize;
  const int64_t k_head = hv_idx / kKGroupSize;

  const int64_t q_base = (batch_idx * kNumQHeads + q_head) * kHeadSize;
  const int64_t k_base = (batch_idx * kNumKHeads + k_head) * kHeadSize;
  const int64_t hv_base = (batch_idx * kNumVHeads + hv_idx);
  const float beta = Sigmoid(__bfloat162float(b[hv_base]));

  const int64_t v_offset = hv_base * kHeadSize + v_idx;
  const int64_t state_row_base = v_offset * kHeadSize;
  const float v_scalar = v[v_offset];

  const float4 state_vec =
      reinterpret_cast<const float4 *>(state + state_row_base)[lane];

  const int kk_base = lane * kElemsPerLane;
  const float4 q_vec = LoadBf16x4GlobalNc(q + q_base + kk_base);
  const float4 k_vec = LoadBf16x4GlobalNc(k + k_base + kk_base);

  const float g = [&]() {
    const float x = __bfloat162float(a[hv_base]) + dt_bias[hv_idx];
    return expf(negated_exp_A_log * SoftplusStable(x));
  }();

  float4 h;
  h.x = g * state_vec.x;
  h.y = g * state_vec.y;
  h.z = g * state_vec.z;
  h.w = g * state_vec.w;

  float sum_hk = k_vec.x * h.x;
  sum_hk = fmaf(k_vec.y, h.y, sum_hk);
  sum_hk = fmaf(k_vec.z, h.z, sum_hk);
  sum_hk = fmaf(k_vec.w, h.w, sum_hk);

  const float old_v = WarpAllReduceSum(sum_hk);
  const float delta = beta * (v_scalar - old_v);

  h.x = fmaf(k_vec.x, delta, h.x);
  h.y = fmaf(k_vec.y, delta, h.y);
  h.z = fmaf(k_vec.z, delta, h.z);
  h.w = fmaf(k_vec.w, delta, h.w);

  StoreF32x4Global(new_state + state_row_base + lane * kElemsPerLane, h);

  float sum_hq = q_vec.x * h.x;
  sum_hq = fmaf(q_vec.y, h.y, sum_hq);
  sum_hq = fmaf(q_vec.z, h.z, sum_hq);
  sum_hq = fmaf(q_vec.w, h.w, sum_hq);

  const float out_acc = WarpAllReduceSum(sum_hq);
  if (lane == 0) {
    output[v_offset] = __float2bfloat16_rn(scale * out_acc);
  }
}

// ============================================================================
// Multi-CTA pipelined kernel: grid = B * HV * kNumCTAs.
// Each CTA handles kVTilesPerCTA v-tiles with cp.async double-buffering.
// ============================================================================

struct SmemPipelined {
  float sData[kNumStages][kTileV][kHeadSize];
  float sV[kHeadSize];
  __nv_bfloat16 sOutput[kLocalVCount];
};

__device__ __forceinline__ void IssueTileAsyncCopy(
    float sData_stage[][kHeadSize], const float *__restrict__ state,
    int64_t hv_base, int v_start_global, int warp_id, int lane) {
#pragma unroll
  for (int pass = 0; pass < kItersPerTile; ++pass) {
    const int row_in_tile = pass * kRowsPerIter + warp_id;
    const int global_v = v_start_global + row_in_tile;
    const int64_t row_base =
        (hv_base * kHeadSize + global_v) * static_cast<int64_t>(kHeadSize);
    CpAsyncCg16(&sData_stage[row_in_tile][lane * kElemsPerLane],
                state + row_base + lane * kElemsPerLane);
  }
}

__global__ void GdnDecodePipelined(
    const __nv_bfloat16 *__restrict__ q, const __nv_bfloat16 *__restrict__ k,
    const __nv_bfloat16 *__restrict__ v, const float *__restrict__ state,
    const float *__restrict__ A_log, const __nv_bfloat16 *__restrict__ a,
    const float *__restrict__ dt_bias, const __nv_bfloat16 *__restrict__ b,
    float scale, __nv_bfloat16 *__restrict__ output,
    float *__restrict__ new_state) {

  extern __shared__ char smem_raw[];
  SmemPipelined &smem = *reinterpret_cast<SmemPipelined *>(smem_raw);

  const int64_t block_linear = static_cast<int64_t>(blockIdx.x);
  const int64_t cta_idx = block_linear % kNumCTAs;
  const int64_t bh = block_linear / kNumCTAs;

  const int64_t batch_idx = bh / kNumVHeads;
  const int64_t hv_idx = bh % kNumVHeads;

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & (kWarpSize - 1);

  const int64_t q_head = hv_idx / kQGroupSize;
  const int64_t k_head = hv_idx / kKGroupSize;
  const int64_t q_base = (batch_idx * kNumQHeads + q_head) * kHeadSize;
  const int64_t k_base = (batch_idx * kNumKHeads + k_head) * kHeadSize;
  const int64_t hv_base = (batch_idx * kNumVHeads + hv_idx);

  const int v_start = cta_idx * kVTilesPerCTA * kTileV;

  // Read gate scalars early (hides latency behind cp.async)
  const float r_A_log = A_log[hv_idx];
  const float r_a = __bfloat162float(a[hv_base]);
  const float r_dt_bias = dt_bias[hv_idx];
  const float r_b = __bfloat162float(b[hv_base]);

  // Load ALL 128 v values into smem (128 threads, 1 element each)
  smem.sV[tid] = __bfloat162float(v[hv_base * kHeadSize + tid]);

  // Prefetch first tile (stage 0)
  IssueTileAsyncCopy(smem.sData[0], state, hv_base, v_start, warp_id, lane);
  CpAsyncCommit();

  // Load q, k while cp.async is in flight
  const int kk_base = lane * kElemsPerLane;
  float4 q_vec = LoadBf16x4GlobalNc(q + q_base + kk_base);
  const float4 k_vec = LoadBf16x4GlobalNc(k + k_base + kk_base);

  // Compute g and beta (lane 0 only, then broadcast)
  float g, beta;
  {
    const float neg_exp_A = -expf(r_A_log);
    if (lane == 0) {
      const float x = r_a + r_dt_bias;
      g = expf(neg_exp_A * SoftplusStable(x));
      beta = Sigmoid(r_b);
    }
    g = __shfl_sync(kFullWarpMask, g, 0);
    beta = __shfl_sync(kFullWarpMask, beta, 0);
  }

  q_vec.x *= scale;
  q_vec.y *= scale;
  q_vec.z *= scale;
  q_vec.w *= scale;

  __syncthreads();

#pragma unroll
  for (int tile = 0; tile < kVTilesPerCTA; ++tile) {
    const int stage = tile % kNumStages;
    const int tile_v_start = v_start + tile * kTileV;

    CpAsyncWaitAll();
    __syncthreads();

    if (tile + 1 < kVTilesPerCTA) {
      const int next_stage = (tile + 1) % kNumStages;
      IssueTileAsyncCopy(smem.sData[next_stage], state, hv_base,
                         v_start + (tile + 1) * kTileV, warp_id, lane);
      CpAsyncCommit();
    }

#pragma unroll
    for (int iter = 0; iter < kItersPerTile; ++iter) {
      const int row_in_tile = iter * kRowsPerIter + warp_id;
      const int local_v = tile * kTileV + row_in_tile;
      const int global_v = tile_v_start + row_in_tile;
      const int64_t v_offset = hv_base * kHeadSize + global_v;
      const int64_t state_row_base = v_offset * static_cast<int64_t>(kHeadSize);

      const float4 sv = *reinterpret_cast<const float4 *>(
          &smem.sData[stage][row_in_tile][lane * kElemsPerLane]);

      float4 h;
      h.x = g * sv.x;
      h.y = g * sv.y;
      h.z = g * sv.z;
      h.w = g * sv.w;

      float sum_hk = k_vec.x * h.x;
      sum_hk = fmaf(k_vec.y, h.y, sum_hk);
      sum_hk = fmaf(k_vec.z, h.z, sum_hk);
      sum_hk = fmaf(k_vec.w, h.w, sum_hk);
      const float old_v = WarpAllReduceSum(sum_hk);

      const float delta = beta * (smem.sV[global_v] - old_v);

      h.x = fmaf(k_vec.x, delta, h.x);
      h.y = fmaf(k_vec.y, delta, h.y);
      h.z = fmaf(k_vec.z, delta, h.z);
      h.w = fmaf(k_vec.w, delta, h.w);

      StoreF32x4Global(new_state + state_row_base + lane * kElemsPerLane, h);

      float sum_hq = q_vec.x * h.x;
      sum_hq = fmaf(q_vec.y, h.y, sum_hq);
      sum_hq = fmaf(q_vec.z, h.z, sum_hq);
      sum_hq = fmaf(q_vec.w, h.w, sum_hq);
      const float out_acc = WarpAllReduceSum(sum_hq);

      if (lane == 0) {
        smem.sOutput[local_v] = __float2bfloat16_rn(out_acc);
      }
    }
  }

  __syncthreads();

  if (tid < kLocalVCount) {
    output[hv_base * kHeadSize + v_start + tid] = smem.sOutput[tid];
  }
}

// ============================================================================
// Host dispatch
// ============================================================================

__host__ __forceinline__ float ResolveScale(double scale) {
  float scale_f = static_cast<float>(scale);
  if (scale_f == 0.0f) {
    scale_f = 1.0f / sqrtf(static_cast<float>(kHeadSize));
  }
  return scale_f;
}

void RunGdnDecodeKernel8(TensorView q, TensorView k, TensorView v,
                         TensorView state, TensorView A_log, TensorView a,
                         TensorView dt_bias, TensorView b, double scale,
                         TensorView output, TensorView new_state) {
  gdn_decode::ValidateShapesAndTypes(q, k, v, state, A_log, a, dt_bias, b,
                                     output, new_state);

  const int64_t B = q.size(0);
  const float scale_f = ResolveScale(scale);

  ffi::CUDADeviceGuard guard(q.device().device_id);
  const cudaStream_t stream = get_cuda_stream(q.device());

  const float *state_ptr = static_cast<const float *>(state.data_ptr());
  const __nv_bfloat16 *q_ptr =
      static_cast<const __nv_bfloat16 *>(q.data_ptr());
  const __nv_bfloat16 *k_ptr =
      static_cast<const __nv_bfloat16 *>(k.data_ptr());
  const __nv_bfloat16 *v_ptr =
      static_cast<const __nv_bfloat16 *>(v.data_ptr());
  const float *A_log_ptr = static_cast<const float *>(A_log.data_ptr());
  const __nv_bfloat16 *a_ptr =
      static_cast<const __nv_bfloat16 *>(a.data_ptr());
  const float *dt_bias_ptr = static_cast<const float *>(dt_bias.data_ptr());
  const __nv_bfloat16 *b_ptr =
      static_cast<const __nv_bfloat16 *>(b.data_ptr());
  __nv_bfloat16 *output_ptr = static_cast<__nv_bfloat16 *>(output.data_ptr());
  float *new_state_ptr = static_cast<float *>(new_state.data_ptr());

  if (B <= kBatchThreshold) {
    const dim3 grid(B * kNumVHeads * kSmallNumVTiles, 1, 1);
    GdnDecodeSmallBatch<<<grid, kNumThreads, 0, stream>>>(
        q_ptr, k_ptr, v_ptr, state_ptr, A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
        scale_f, output_ptr, new_state_ptr);
  } else {
    const dim3 grid(B * kNumVHeads * kNumCTAs, 1, 1);
    const size_t smem_bytes = sizeof(SmemPipelined);
    GdnDecodePipelined<<<grid, kNumThreads, smem_bytes, stream>>>(
        q_ptr, k_ptr, v_ptr, state_ptr, A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
        scale_f, output_ptr, new_state_ptr);
  }

  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    TVM_FFI_THROW(RuntimeError)
        << "GdnDecodeKernel8 launch failed: " << cudaGetErrorString(err);
  }
}

} // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(gdn_decode_v8, RunGdnDecodeKernel8);
