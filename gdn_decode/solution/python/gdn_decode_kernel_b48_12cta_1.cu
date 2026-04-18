#include "gdn_decode_utils.h"
#include "tvm_ffi_utils.h"

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <math.h>

namespace {

constexpr int kHeadSize    = gdn_decode::kHeadSize;
constexpr int kNumQHeads   = gdn_decode::kNumQHeads;
constexpr int kNumKHeads   = gdn_decode::kNumKHeads;
constexpr int kNumVHeads   = gdn_decode::kNumVHeads;

constexpr int kWarpSize      = 32;
constexpr int kWarpsPerBlock = 2;
constexpr int kNumThreads    = kWarpSize * kWarpsPerBlock;
constexpr int kElemsPerLane  = kHeadSize / kWarpSize;  // 4

// Keep kTileV=8 (same 4KB smem, same pipeline structure as pipe2).
// Non-uniform 12 CTAs: 4 CTAs × 2 tiles (pipelined) + 8 CTAs × 1 tile.
// 4*2 + 8*1 = 16 = kNumVTiles. Grid B=48: 48*8*12=4608 → ~0.97 waves.
constexpr int kTileV          = 8;
constexpr int kNumVTiles      = kHeadSize / kTileV;     // 16
constexpr int kRowsPerStage   = 4;                      // 2 stages × 4 rows × 128f × 4B = 4KB
constexpr int kSubRowsPerWarp = kRowsPerStage / kWarpsPerBlock;  // 2
constexpr int kNumStages      = 2;

constexpr int kNumCTAs   = 12;
constexpr int kNumCTAs_A = 4;   // 2-tile group (pipelined)
constexpr int kTilesA    = 2;
constexpr int kNumCTAs_B = 8;   // 1-tile group
constexpr int kTilesB    = 1;

constexpr int kQGroupSize    = kNumVHeads / kNumQHeads;
constexpr int kKGroupSize    = kNumVHeads / kNumKHeads;
constexpr unsigned kFullWarpMask = 0xffffffffu;

static_assert(kHeadSize == 128);
static_assert(kNumThreads == 64);
static_assert(kElemsPerLane == 4);
static_assert(kNumVTiles == 16);
static_assert(kNumCTAs_A * kTilesA + kNumCTAs_B * kTilesB == kNumVTiles);

// SmemPipelined: 2 stages × 4 rows × 128 floats × 4B = 4KB — same as pipe2
struct SmemPipelined {
  float sData[kNumStages][kRowsPerStage][kHeadSize];
};

__device__ __forceinline__ float SoftplusStable(float x) {
  const float abs_x = fabsf(x);
  return __logf(1.0f + __expf(-abs_x)) + fmaxf(x, 0.0f);
}

__device__ __forceinline__ float Sigmoid(float x) {
  return 1.0f / (1.0f + __expf(-x));
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

__device__ __forceinline__ void WarpAllReduceSum2(float &a, float &b) {
  float ta, tb;
  asm volatile("shfl.sync.bfly.b32 %0,%2,16,31,0xffffffff;\n\t"
               "shfl.sync.bfly.b32 %1,%3,16,31,0xffffffff;"
               : "=f"(ta),"=f"(tb) : "f"(a),"f"(b)); a += ta; b += tb;
  asm volatile("shfl.sync.bfly.b32 %0,%2, 8,31,0xffffffff;\n\t"
               "shfl.sync.bfly.b32 %1,%3, 8,31,0xffffffff;"
               : "=f"(ta),"=f"(tb) : "f"(a),"f"(b)); a += ta; b += tb;
  asm volatile("shfl.sync.bfly.b32 %0,%2, 4,31,0xffffffff;\n\t"
               "shfl.sync.bfly.b32 %1,%3, 4,31,0xffffffff;"
               : "=f"(ta),"=f"(tb) : "f"(a),"f"(b)); a += ta; b += tb;
  asm volatile("shfl.sync.bfly.b32 %0,%2, 2,31,0xffffffff;\n\t"
               "shfl.sync.bfly.b32 %1,%3, 2,31,0xffffffff;"
               : "=f"(ta),"=f"(tb) : "f"(a),"f"(b)); a += ta; b += tb;
  asm volatile("shfl.sync.bfly.b32 %0,%2, 1,31,0xffffffff;\n\t"
               "shfl.sync.bfly.b32 %1,%3, 1,31,0xffffffff;"
               : "=f"(ta),"=f"(tb) : "f"(a),"f"(b)); a += ta; b += tb;
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

// Load kRowsPerStage=4 rows (kTileV rows), 2 per warp.
__device__ __forceinline__ void IssueStageAsyncCopy(
    float sData_stage[][kHeadSize], const float *__restrict__ state,
    int hv_base, int v_row_start, int warp_id, int lane) {
#pragma unroll
  for (int sub = 0; sub < kSubRowsPerWarp; ++sub) {
    const int row   = sub * kWarpsPerBlock + warp_id;
    const int gv    = v_row_start + row;
    const int rbase = (hv_base * kHeadSize + gv) * kHeadSize;
    CpAsyncCg16(&sData_stage[row][lane * kElemsPerLane],
                state + rbase + lane * kElemsPerLane);
  }
}

// Non-uniform 12-CTA kernel: group A (CTAs 0-7) handle 3 tiles, group B (CTAs 8-11) handle 2 tiles.
// kTileV=4 → 4KB smem, same as pipe2, preserving block-limit occupancy.
template<int kNumTiles>
__device__ __forceinline__ void RunCTA(
    SmemPipelined &smem,
    const __nv_bfloat16 *__restrict__ q, const __nv_bfloat16 *__restrict__ k,
    const __nv_bfloat16 *__restrict__ v, const float *__restrict__ state,
    const float *__restrict__ A_log, const __nv_bfloat16 *__restrict__ a,
    const float *__restrict__ dt_bias, const __nv_bfloat16 *__restrict__ b,
    float scale, __nv_bfloat16 *__restrict__ output, float *__restrict__ new_state,
    int q_base, int k_base, int hv_base, int hv_idx, int v_start,
    int warp_id, int lane, int kk_base) {

  IssueStageAsyncCopy(smem.sData[0], state, hv_base, v_start, warp_id, lane);
  CpAsyncCommit();

  const float r_A_log   = A_log[hv_idx];
  const float r_a       = __bfloat162float(a[hv_base]);
  const float r_dt_bias = dt_bias[hv_idx];
  const float r_b       = __bfloat162float(b[hv_base]);

  float4 q_vec = LoadBf16x4GlobalNc(q + q_base + kk_base);
  const float4 k_vec = LoadBf16x4GlobalNc(k + k_base + kk_base);

  const float g    = __expf(-__expf(r_A_log) * SoftplusStable(r_a + r_dt_bias));
  const float beta = Sigmoid(r_b);

  q_vec.x *= scale; q_vec.y *= scale;
  q_vec.z *= scale; q_vec.w *= scale;

  const float qk_partial = fmaf(q_vec.x, k_vec.x,
                            fmaf(q_vec.y, k_vec.y,
                            fmaf(q_vec.z, k_vec.z, q_vec.w * k_vec.w)));
  const float qk = WarpAllReduceSum(qk_partial);

  const int hv_v_base = hv_base * kHeadSize;

  constexpr int kItersPerTile = kTileV / kRowsPerStage;  // 2

#pragma unroll
  for (int tile = 0; tile < kNumTiles; ++tile) {
    const int cur_stage  = tile & 1;
    const int nxt_stage  = cur_stage ^ 1;
    const int tile_start = v_start + tile * kTileV;

    CpAsyncWaitAll();

    if (tile + 1 < kNumTiles) {
      IssueStageAsyncCopy(smem.sData[nxt_stage], state, hv_base,
                          v_start + (tile + 1) * kTileV, warp_id, lane);
      CpAsyncCommit();
    }

#pragma unroll
    for (int iter = 0; iter < kItersPerTile; ++iter) {
#pragma unroll
      for (int sub = 0; sub < kSubRowsPerWarp; ++sub) {
        const int row      = iter * kRowsPerStage + sub * kWarpsPerBlock + warp_id;
        const int v_offset = hv_v_base + tile_start + row;

        const float v_scalar = __bfloat162float(v[v_offset]);
        const float4 sv = *reinterpret_cast<const float4 *>(&smem.sData[cur_stage][row][kk_base]);

        float4 h;
        h.x = g*sv.x; h.y = g*sv.y; h.z = g*sv.z; h.w = g*sv.w;

        float kh  = fmaf(k_vec.x, h.x, fmaf(k_vec.y, h.y, fmaf(k_vec.z, h.z, k_vec.w * h.w)));
        float qsv = fmaf(q_vec.x, sv.x, fmaf(q_vec.y, sv.y, fmaf(q_vec.z, sv.z, q_vec.w * sv.w)));
        WarpAllReduceSum2(kh, qsv);

        const float delta = beta * (v_scalar - kh);

        h.x = fmaf(k_vec.x, delta, h.x); h.y = fmaf(k_vec.y, delta, h.y);
        h.z = fmaf(k_vec.z, delta, h.z); h.w = fmaf(k_vec.w, delta, h.w);

        StoreF32x4Global(new_state + v_offset * kHeadSize + kk_base, h);

        const float out = fmaf(g, qsv, delta * qk);
        if (lane == 0) output[v_offset] = __float2bfloat16_rn(out);
      }
    }
  }
}

__global__ void GdnDecodeB48_12CTA(
    const __nv_bfloat16 *__restrict__ q, const __nv_bfloat16 *__restrict__ k,
    const __nv_bfloat16 *__restrict__ v, const float *__restrict__ state,
    const float *__restrict__ A_log, const __nv_bfloat16 *__restrict__ a,
    const float *__restrict__ dt_bias, const __nv_bfloat16 *__restrict__ b,
    float scale, __nv_bfloat16 *__restrict__ output,
    float *__restrict__ new_state) {

  extern __shared__ char smem_raw[];
  SmemPipelined &smem = *reinterpret_cast<SmemPipelined *>(smem_raw);

  const int bl        = static_cast<int>(blockIdx.x);
  const int cta_idx   = bl % kNumCTAs;
  const int bh        = bl / kNumCTAs;
  const int batch_idx = bh / kNumVHeads;
  const int hv_idx    = bh % kNumVHeads;

  const int tid     = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane    = tid & (kWarpSize - 1);
  const int kk_base = lane * kElemsPerLane;

  const int q_base  = (batch_idx * kNumQHeads + hv_idx / kQGroupSize) * kHeadSize;
  const int k_base  = (batch_idx * kNumKHeads + hv_idx / kKGroupSize) * kHeadSize;
  const int hv_base = batch_idx * kNumVHeads + hv_idx;

  // Assign tile range: group A (0..kNumCTAs_A-1) covers 3 tiles each,
  // group B covers 2 tiles each. Tiles are packed sequentially.
  const bool is_a   = (cta_idx < kNumCTAs_A);
  // Group A: CTAs 0-3, each covers 2 tiles → rows [cta*16, cta*16+16)
  // Group B: CTAs 4-11, each covers 1 tile → rows starting at 64
  const int v_start = is_a
      ? cta_idx * kTilesA * kTileV
      : kNumCTAs_A * kTilesA * kTileV + (cta_idx - kNumCTAs_A) * kTilesB * kTileV;

  if (is_a) {
    RunCTA<kTilesA>(smem, q, k, v, state, A_log, a, dt_bias, b, scale, output, new_state,
                    q_base, k_base, hv_base, hv_idx, v_start, warp_id, lane, kk_base);
  } else {
    RunCTA<kTilesB>(smem, q, k, v, state, A_log, a, dt_bias, b, scale, output, new_state,
                    q_base, k_base, hv_base, hv_idx, v_start, warp_id, lane, kk_base);
  }
}

__host__ __forceinline__ float ResolveScale(double s) {
  float f = static_cast<float>(s);
  return f == 0.0f ? 1.0f / sqrtf(static_cast<float>(kHeadSize)) : f;
}

void RunGdnDecodeB48_12CTA(TensorView q, TensorView k, TensorView v,
                           TensorView state, TensorView A_log, TensorView a,
                           TensorView dt_bias, TensorView b, double scale,
                           TensorView output, TensorView new_state) {
  gdn_decode::ValidateShapesAndTypes(q, k, v, state, A_log, a, dt_bias, b, output, new_state);
  const int64_t B = q.size(0);
  ffi::CUDADeviceGuard guard(q.device().device_id);
  const cudaStream_t stream = get_cuda_stream(q.device());

  const dim3 grid(B * kNumVHeads * kNumCTAs, 1, 1);
  GdnDecodeB48_12CTA<<<grid, kNumThreads, sizeof(SmemPipelined), stream>>>(
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
      static_cast<float*>(new_state.data_ptr()));

  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    TVM_FFI_THROW(RuntimeError) << "GdnDecodeB48_12CTA failed: " << cudaGetErrorString(err);
}

} // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(gdn_decode_b48_12cta, RunGdnDecodeB48_12CTA);
