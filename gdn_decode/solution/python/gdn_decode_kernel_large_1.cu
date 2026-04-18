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

constexpr int kTileV = 4;
constexpr int kNumVTiles = kHeadSize / kTileV;
constexpr int kRowsPerIter = kWarpsPerBlock;
constexpr int kItersPerTile = kTileV / kRowsPerIter;
constexpr int kNumStages = 2;

constexpr int kNumCTAs4 = 5;
constexpr int kNumCTAs3 = 4;
constexpr int kNumCTAs  = kNumCTAs4 + kNumCTAs3;

constexpr int kQGroupSize = kNumVHeads / kNumQHeads;
constexpr int kKGroupSize = kNumVHeads / kNumKHeads;
constexpr unsigned kFullWarpMask = 0xffffffffu;

static_assert(kHeadSize == 128);
static_assert(kNumThreads == 64);
static_assert(kElemsPerLane == 4);
static_assert(kItersPerTile == 2);
static_assert(kNumCTAs4 * 16 + kNumCTAs3 * 12 == kHeadSize);

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

__device__ __forceinline__ void StoreF32x4Global(float *addr, const float4 &v) {
  asm volatile("st.global.wt.v4.f32 [%0], {%1, %2, %3, %4};"
               : : "l"(addr), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w));
}

__device__ __forceinline__ void CpAsyncCg16(float *smem_ptr, const float *gmem_ptr) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" : : "r"(s), "l"(gmem_ptr) : "memory");
}

__device__ __forceinline__ void CpAsyncCommit()  { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void CpAsyncWaitAll() { asm volatile("cp.async.wait_all;" ::: "memory"); }

struct SmemPipelined { float sData[kNumStages][kTileV][kHeadSize]; };

__device__ __forceinline__ void IssueTileAsyncCopy(
    float sData_stage[][kHeadSize], const float *__restrict__ state,
    int hv_base, int v_start_global, int warp_id, int lane) {
#pragma unroll
  for (int pass = 0; pass < kItersPerTile; ++pass) {
    const int row = pass * kRowsPerIter + warp_id;
    const int gbase = (hv_base * kHeadSize + v_start_global + row) * kHeadSize;
    CpAsyncCg16(&sData_stage[row][lane * kElemsPerLane], state + gbase + lane * kElemsPerLane);
  }
}

template <int kVTilesPerCTA>
__device__ __noinline__ void RunBlock(
    SmemPipelined &smem,
    const __nv_bfloat16 *q, const __nv_bfloat16 *k, const __nv_bfloat16 *v,
    const float *state, const float *A_log, const __nv_bfloat16 *a,
    const float *dt_bias, const __nv_bfloat16 *b, float scale,
    __nv_bfloat16 *output, float *new_state,
    int hv_base, int v_start, int q_base, int k_base, int hv_idx,
    int warp_id, int lane) {

  IssueTileAsyncCopy(smem.sData[0], state, hv_base, v_start, warp_id, lane);
  CpAsyncCommit();

  const float r_A_log   = A_log[hv_idx];
  const float r_a       = __bfloat162float(a[hv_base]);
  const float r_dt_bias = dt_bias[hv_idx];
  const float r_b       = __bfloat162float(b[hv_base]);

  const int kk = lane * kElemsPerLane;
  float4 q_vec = LoadBf16x4GlobalNc(q + q_base + kk);
  const float4 k_vec = LoadBf16x4GlobalNc(k + k_base + kk);

  const float g    = expf(-expf(r_A_log) * SoftplusStable(r_a + r_dt_bias));
  const float beta = Sigmoid(r_b);
  q_vec.x *= scale; q_vec.y *= scale; q_vec.z *= scale; q_vec.w *= scale;

#pragma unroll
  for (int tile = 0; tile < kVTilesPerCTA; ++tile) {
    const int stage = tile % kNumStages;
    const int tv_start = v_start + tile * kTileV;

    CpAsyncWaitAll();
    __syncwarp(kFullWarpMask);

    if (tile + 1 < kVTilesPerCTA) {
      IssueTileAsyncCopy(smem.sData[(tile+1)%kNumStages], state, hv_base,
                         v_start + (tile+1)*kTileV, warp_id, lane);
      CpAsyncCommit();
    }

#pragma unroll
    for (int iter = 0; iter < kItersPerTile; ++iter) {
      const int row = iter * kRowsPerIter + warp_id;
      const int gv  = tv_start + row;
      const int voi = hv_base * kHeadSize + gv;

      const float vs  = __bfloat162float(v[voi]);
      const float4 sv = *reinterpret_cast<const float4*>(&smem.sData[stage][row][kk]);

      float shk; float4 h;
      h.x = g*sv.x; shk  = k_vec.x*h.x;
      h.y = g*sv.y; shk  = fmaf(k_vec.y, h.y, shk);
      h.z = g*sv.z; shk  = fmaf(k_vec.z, h.z, shk);
      h.w = g*sv.w; shk  = fmaf(k_vec.w, h.w, shk);
      const float delta = beta * (vs - WarpAllReduceSum(shk));

      float shq;
      h.x = fmaf(k_vec.x, delta, h.x); shq  = q_vec.x*h.x;
      h.y = fmaf(k_vec.y, delta, h.y); shq  = fmaf(q_vec.y, h.y, shq);
      h.z = fmaf(k_vec.z, delta, h.z); shq  = fmaf(q_vec.z, h.z, shq);
      h.w = fmaf(k_vec.w, delta, h.w); shq  = fmaf(q_vec.w, h.w, shq);

      StoreF32x4Global(new_state + voi*kHeadSize + kk, h);
      if (lane == 0) output[voi] = __float2bfloat16_rn(WarpAllReduceSum(shq));
    }
  }
}

__global__ void GdnDecodeLarge1(
    const __nv_bfloat16 *q, const __nv_bfloat16 *k, const __nv_bfloat16 *v,
    const float *state, const float *A_log, const __nv_bfloat16 *a,
    const float *dt_bias, const __nv_bfloat16 *b, float scale,
    __nv_bfloat16 *output, float *new_state, int bh_count) {

  extern __shared__ char smem_raw[];
  SmemPipelined &smem = *reinterpret_cast<SmemPipelined*>(smem_raw);

  const int bl = static_cast<int>(blockIdx.x);
  const int warp_id = threadIdx.x >> 5;
  const int lane    = threadIdx.x & (kWarpSize - 1);

  int bh, v_start;
  if (bl < bh_count * kNumCTAs4) {
    bh = bl / kNumCTAs4; v_start = (bl % kNumCTAs4) * 16;
  } else {
    const int adj = bl - bh_count * kNumCTAs4;
    bh = adj / kNumCTAs3; v_start = 80 + (adj % kNumCTAs3) * 12;
  }

  const int bi = bh / kNumVHeads, hv = bh % kNumVHeads;
  const int hv_base = bi * kNumVHeads + hv;
  const int q_base  = (bi * kNumQHeads + hv / kQGroupSize) * kHeadSize;
  const int k_base  = (bi * kNumKHeads + hv / kKGroupSize) * kHeadSize;

  if (bl < bh_count * kNumCTAs4)
    RunBlock<4>(smem, q,k,v,state,A_log,a,dt_bias,b,scale,output,new_state,
                hv_base, v_start, q_base, k_base, hv, warp_id, lane);
  else
    RunBlock<3>(smem, q,k,v,state,A_log,a,dt_bias,b,scale,output,new_state,
                hv_base, v_start, q_base, k_base, hv, warp_id, lane);
}

__host__ __forceinline__ float ResolveScale(double s) {
  float f = static_cast<float>(s);
  return f == 0.0f ? 1.0f/sqrtf(static_cast<float>(kHeadSize)) : f;
}

void RunGdnDecodeKernel11(TensorView q, TensorView k, TensorView v,
                          TensorView state, TensorView A_log, TensorView a,
                          TensorView dt_bias, TensorView b, double scale,
                          TensorView output, TensorView new_state) {
  gdn_decode::ValidateShapesAndTypes(q,k,v,state,A_log,a,dt_bias,b,output,new_state);
  const int B = static_cast<int>(q.size(0));
  ffi::CUDADeviceGuard guard(q.device().device_id);
  const cudaStream_t stream = get_cuda_stream(q.device());
  const int BH = B * kNumVHeads;
  GdnDecodeLarge1<<<BH*kNumCTAs, kNumThreads, sizeof(SmemPipelined), stream>>>(
      static_cast<const __nv_bfloat16*>(q.data_ptr()),
      static_cast<const __nv_bfloat16*>(k.data_ptr()),
      static_cast<const __nv_bfloat16*>(v.data_ptr()),
      static_cast<const float*>(state.data_ptr()),
      static_cast<const float*>(A_log.data_ptr()),
      static_cast<const __nv_bfloat16*>(a.data_ptr()),
      static_cast<const float*>(dt_bias.data_ptr()),
      static_cast<const __nv_bfloat16*>(b.data_ptr()),
      ResolveScale(scale),
      static_cast<__nv_bfloat16*>(output.data_ptr()),
      static_cast<float*>(new_state.data_ptr()), BH);
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    TVM_FFI_THROW(RuntimeError) << "GdnDecodeLarge1 failed: " << cudaGetErrorString(err);
}

} // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(gdn_decode_large_1, RunGdnDecodeKernel11);
