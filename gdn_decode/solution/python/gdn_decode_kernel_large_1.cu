#include "gdn_decode_utils.h"
#include "tvm_ffi_utils.h"

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <math.h>

namespace {

constexpr int kHeadSize = gdn_decode::kHeadSize;
constexpr int kNumQHeads = gdn_decode::kNumQHeads;
constexpr int kNumKHeads = gdn_decode::kNumKHeads;
constexpr int kNumVHeads = gdn_decode::kNumVHeads;

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 2;
constexpr int kNumThreads = kWarpSize * kWarpsPerBlock;
constexpr int kElemsPerLane = kHeadSize / kWarpSize;

constexpr int kTileV = 8;
constexpr int kNumVTiles = kHeadSize / kTileV;
constexpr int kRowsPerIter = kWarpsPerBlock;
constexpr int kItersPerTile = kTileV / kRowsPerIter;
constexpr int kNumStages = 1;

constexpr int kQGroupSize = kNumVHeads / kNumQHeads;
constexpr int kKGroupSize = kNumVHeads / kNumKHeads;
constexpr unsigned kFullWarpMask = 0xffffffffu;

static_assert(kHeadSize == 128);
static_assert(kNumThreads == 64);
static_assert(kElemsPerLane == 4);
static_assert(kItersPerTile == 4);

// ============================================================================
// Device helpers
// ============================================================================

__device__ __forceinline__ float SoftplusStable(float x) {
  const float abs_x = fabsf(x);
  return log1pf(expf(-abs_x)) + fmaxf(x, 0.0f);
}

__device__ __forceinline__ float Sigmoid(float x) {
  return fmaf(0.5f, tanhf(x * 0.5f), 0.5f);
}

__device__ __forceinline__ float WarpAllReduceSum(float value) {
  float tmp;
  asm volatile("shfl.sync.bfly.b32 %0,%1,16,31,0xffffffff;" : "=f"(tmp) : "f"(value)); value += tmp;
  asm volatile("shfl.sync.bfly.b32 %0,%1, 8,31,0xffffffff;" : "=f"(tmp) : "f"(value)); value += tmp;
  asm volatile("shfl.sync.bfly.b32 %0,%1, 4,31,0xffffffff;" : "=f"(tmp) : "f"(value)); value += tmp;
  asm volatile("shfl.sync.bfly.b32 %0,%1, 2,31,0xffffffff;" : "=f"(tmp) : "f"(value)); value += tmp;
  asm volatile("shfl.sync.bfly.b32 %0,%1, 1,31,0xffffffff;" : "=f"(tmp) : "f"(value)); value += tmp;
  return value;
}

__device__ __forceinline__ float4
LoadBf16x4GlobalNc(const __nv_bfloat16 *__restrict__ ptr) {
  float4 out;
  asm volatile("{\n\t"
               ".reg .b16 h<4>;\n\t"
               "ld.global.nc.v4.b16 {h0, h1, h2, h3}, [%4];\n\t"
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
      "st.global.wt.v4.f32 [%0], {%1, %2, %3, %4};"
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

struct SmemTile {
  float sData[kTileV][kHeadSize];
};

__device__ __forceinline__ void IssueTileAsyncCopy(
    float sData_stage[][kHeadSize], const float *__restrict__ state,
    int hv_base, int v_start_global, int warp_id, int lane) {
#pragma unroll
  for (int pass = 0; pass < kItersPerTile; ++pass) {
    const int row_in_tile = pass * kRowsPerIter + warp_id;
    const int global_v = v_start_global + row_in_tile;
    const int row_base = (hv_base * kHeadSize + global_v) * kHeadSize;
    CpAsyncCg16(&sData_stage[row_in_tile][lane * kElemsPerLane],
                state + row_base + lane * kElemsPerLane);
  }
}

// kNumCTAs is templated so the tile loop is fully unrolled at compile time.
// kNumCTAs=8  → kVTilesPerCTA=2 (B>=48)
// kNumCTAs=16 → kVTilesPerCTA=1 (B=8..32, more blocks/SM)
template<int kNumCTAs>
__global__ void GdnDecodePipelinedV11(
    const __nv_bfloat16 *__restrict__ q, const __nv_bfloat16 *__restrict__ k,
    const __nv_bfloat16 *__restrict__ v, const float *__restrict__ state,
    const float *__restrict__ A_log, const __nv_bfloat16 *__restrict__ a,
    const float *__restrict__ dt_bias, const __nv_bfloat16 *__restrict__ b,
    float scale, __nv_bfloat16 *__restrict__ output,
    float *__restrict__ new_state) {

  constexpr int kVTilesPerCTA = kNumVTiles / kNumCTAs;

  extern __shared__ char smem_raw[];
  SmemTile &smem = *reinterpret_cast<SmemTile *>(smem_raw);

  const int block_linear = static_cast<int>(blockIdx.x);
  const int cta_idx = block_linear % kNumCTAs;
  const int bh = block_linear / kNumCTAs;

  const int batch_idx = bh / kNumVHeads;
  const int hv_idx = bh % kNumVHeads;

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & (kWarpSize - 1);

  const int q_head = hv_idx / kQGroupSize;
  const int k_head = hv_idx / kKGroupSize;
  const int q_base = (batch_idx * kNumQHeads + q_head) * kHeadSize;
  const int k_base = (batch_idx * kNumKHeads + k_head) * kHeadSize;
  const int hv_base = batch_idx * kNumVHeads + hv_idx;

  const int v_start = cta_idx * kVTilesPerCTA * kTileV;

  // Prefetch tile 0 — overlaps with q/k/g/beta loads below
  IssueTileAsyncCopy(smem.sData, state, hv_base, v_start, warp_id, lane);
  CpAsyncCommit();

  const float r_A_log = A_log[hv_idx];
  const float r_a = __bfloat162float(a[hv_base]);
  const float r_dt_bias = dt_bias[hv_idx];
  const float r_b = __bfloat162float(b[hv_base]);

  const int kk_base = lane * kElemsPerLane;
  float4 q_vec = LoadBf16x4GlobalNc(q + q_base + kk_base);
  const float4 k_vec = LoadBf16x4GlobalNc(k + k_base + kk_base);

  const float neg_exp_A = -expf(r_A_log);
  const float g    = expf(neg_exp_A * SoftplusStable(r_a + r_dt_bias));
  const float beta = Sigmoid(r_b);

  q_vec.x *= scale; q_vec.y *= scale;
  q_vec.z *= scale; q_vec.w *= scale;

#pragma unroll
  for (int tile = 0; tile < kVTilesPerCTA; ++tile) {
    const int tile_v_start = v_start + tile * kTileV;

    CpAsyncWaitAll();
    __syncwarp(kFullWarpMask);

    if (tile + 1 < kVTilesPerCTA) {
      IssueTileAsyncCopy(smem.sData, state, hv_base,
                         v_start + (tile + 1) * kTileV, warp_id, lane);
      CpAsyncCommit();
    }

#pragma unroll
    for (int iter = 0; iter < kItersPerTile; ++iter) {
      const int row_in_tile = iter * kRowsPerIter + warp_id;
      const int global_v    = tile_v_start + row_in_tile;
      const int v_offset_i  = hv_base * kHeadSize + global_v;
      const int state_row_i = v_offset_i * kHeadSize;

      const float v_scalar = __bfloat162float(v[v_offset_i]);
      const float4 sv = *reinterpret_cast<const float4 *>(
          &smem.sData[row_in_tile][lane * kElemsPerLane]);

      float sum_hk; float4 h;
      h.x = g * sv.x; sum_hk  = k_vec.x * h.x;
      h.y = g * sv.y; sum_hk  = fmaf(k_vec.y, h.y, sum_hk);
      h.z = g * sv.z; sum_hk  = fmaf(k_vec.z, h.z, sum_hk);
      h.w = g * sv.w; sum_hk  = fmaf(k_vec.w, h.w, sum_hk);
      const float old_v = WarpAllReduceSum(sum_hk);

      const float delta = beta * (v_scalar - old_v);

      float sum_hq;
      h.x = fmaf(k_vec.x, delta, h.x); sum_hq  = q_vec.x * h.x;
      h.y = fmaf(k_vec.y, delta, h.y); sum_hq  = fmaf(q_vec.y, h.y, sum_hq);
      h.z = fmaf(k_vec.z, delta, h.z); sum_hq  = fmaf(q_vec.z, h.z, sum_hq);
      h.w = fmaf(k_vec.w, delta, h.w); sum_hq  = fmaf(q_vec.w, h.w, sum_hq);

      StoreF32x4Global(new_state + state_row_i + lane * kElemsPerLane, h);

      const float out_acc = WarpAllReduceSum(sum_hq);
      if (lane == 0) output[v_offset_i] = __float2bfloat16_rn(out_acc);
    }
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

void RunGdnDecodeKernel11(TensorView q, TensorView k, TensorView v,
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
  const __nv_bfloat16 *q_ptr = static_cast<const __nv_bfloat16 *>(q.data_ptr());
  const __nv_bfloat16 *k_ptr = static_cast<const __nv_bfloat16 *>(k.data_ptr());
  const __nv_bfloat16 *v_ptr = static_cast<const __nv_bfloat16 *>(v.data_ptr());
  const float *A_log_ptr     = static_cast<const float *>(A_log.data_ptr());
  const __nv_bfloat16 *a_ptr = static_cast<const __nv_bfloat16 *>(a.data_ptr());
  const float *dt_bias_ptr   = static_cast<const float *>(dt_bias.data_ptr());
  const __nv_bfloat16 *b_ptr = static_cast<const __nv_bfloat16 *>(b.data_ptr());
  __nv_bfloat16 *output_ptr  = static_cast<__nv_bfloat16 *>(output.data_ptr());
  float *new_state_ptr       = static_cast<float *>(new_state.data_ptr());

  const size_t smem_bytes = sizeof(SmemTile);

  const dim3 grid(B * kNumVHeads * 8, 1, 1);
  GdnDecodePipelinedV11<8><<<grid, kNumThreads, smem_bytes, stream>>>(
      q_ptr, k_ptr, v_ptr, state_ptr, A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
      scale_f, output_ptr, new_state_ptr);

  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    TVM_FFI_THROW(RuntimeError)
        << "GdnDecodeKernel11 launch failed: " << cudaGetErrorString(err);
  }
}

} // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(gdn_decode_large_1, RunGdnDecodeKernel11);
