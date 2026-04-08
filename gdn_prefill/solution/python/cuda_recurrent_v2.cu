/*
 * GDN Prefill — Recurrent CUDA Kernel (v2)
 *
 * Multi-CTA recurrent kernel ported from gdn_decode_kernel_8.cu.
 * Each CTA handles 16 V-rows (kVTilesPerCTA * kTileV), with each warp
 * carrying 4 state vectors in registers across the token loop. q/k loads
 * are reused across all V-rows within a CTA, and gate scalars (g, beta)
 * are computed once per token on lane 0 and broadcast.
 *
 * Grid: num_seqs * kNumVHeads * kNumCTAs  (8 CTAs per seq/head)
 *
 * ---------------------------------------------------------------------------
 * WHY THIS IS SLOWER THAN v1 FOR PREFILL (~1.5-2.5x)
 * ---------------------------------------------------------------------------
 *
 * The multi-CTA approach trades block count for q/k load reuse:
 *   v1: 32 blocks per (seq, head), each warp handles 1 V-row   → N*8*32 total
 *   v2:  8 blocks per (seq, head), each warp handles 4 V-rows  → N*8*8  total
 *
 * Total compute is identical (128 V-rows × seqlen per seq/head). v2 packs
 * 4x more work into each block but launches 4x fewer blocks. With dynamic
 * scheduling on 192 SMs, wall time ≈ total_work / 192 — same for both.
 * So in theory v2 should break even, not be slower.
 *
 * Multiple factors compound to cause the 1.5-2.5x regression:
 *
 * 1. Register pressure → reduced occupancy
 *    cuobjdump shows:
 *      v1: 32 regs/thread → 16 blocks/SM max → 64 warps/SM
 *      v2: 48 regs/thread → 10 blocks/SM max → 40 warps/SM
 *    Carrying 4 float4 state vectors (vs 1 in v1) uses 50% more registers,
 *    cutting max concurrent warps by 37.5%. Alone this accounts for ~1.6x
 *    slowdown (64/40) from reduced latency hiding.
 *
 * 2. SM under-utilization (low N)
 *    With N=1, H=8: v1 launches 256 blocks (all 192 SMs busy), v2 launches
 *    64 blocks (128 SMs idle). Fewer blocks can't fill the GPU.
 *
 * 3. Reduced scheduling flexibility (variable-length sequences)
 *    With variable seqlens (14 to 2300 in one batch), short-seq blocks
 *    finish early and their SMs pick up new blocks. v1's 8192 fine-grained
 *    blocks give the scheduler 4x more work units to fill these gaps vs
 *    v2's 2048 coarser blocks. Coarser blocks = more idle SM time between
 *    short and long sequences.
 *
 * 4. Likely ILP loss from the unrolled 4-V-row inner loop
 *    Higher register pressure may prevent the compiler from interleaving
 *    independent V-row computations. Each WarpAllReduceSum (5 dependent
 *    shuffles) is a serial chain; with 8 such chains per token (vs 2 in v1),
 *    the compiler may not effectively overlap them despite independence.
 *    (Would need ncu profiling to confirm the exact contribution.)
 *
 * Why it works for DECODE:
 *   Decode processes 1 token — block duration is tiny so occupancy matters
 *   less. The benefit of fewer blocks (less L2 contention for state data
 *   across many batches, cp.async pipelining) outweighs the cost.
 * ---------------------------------------------------------------------------
 */

#include "cuda_utils.h"
#include "tvm_ffi_utils.h"

#include <cstdint>
#include <cuda_bf16.h>
#include <math.h>

namespace {

constexpr int kHeadSize = gdn_prefill::kHeadSize;
constexpr int64_t kNumQHeads = gdn_prefill::kNumQHeads;
constexpr int64_t kNumKHeads = gdn_prefill::kNumKHeads;
constexpr int64_t kNumVHeads = gdn_prefill::kNumVHeads;

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 4;
constexpr int kNumThreads = kWarpSize * kWarpsPerBlock;
constexpr int kElemsPerLane = kHeadSize / kWarpSize;

//// Multi-CTA tiling
constexpr int kTileV = 8;
constexpr int kNumVTiles = kHeadSize / kTileV;
constexpr int kNumCTAs = 8;
constexpr int kVTilesPerCTA = kNumVTiles / kNumCTAs;
constexpr int kRowsPerIter = kWarpsPerBlock;
constexpr int kItersPerTile = kTileV / kRowsPerIter;

constexpr int64_t kQGroupSize = kNumVHeads / kNumQHeads;
constexpr int64_t kKGroupSize = kNumVHeads / kNumKHeads;
constexpr unsigned kFullWarpMask = 0xffffffffu;

static_assert(kHeadSize == 128);
static_assert(kNumThreads == 128);
static_assert(kElemsPerLane == 4);
static_assert(kVTilesPerCTA == 2);
static_assert(kItersPerTile == 2);

//// Device helpers

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

//// Multi-CTA recurrent kernel
//// Each CTA handles kVTilesPerCTA * kTileV = 16 V-rows.
//// Each warp carries 4 state vectors in registers across the token loop,
//// reusing the same q/k loads for all V-rows within the CTA.

__global__ void __launch_bounds__(kNumThreads)
GdnPrefillRecurrentV2Kernel(
    const __nv_bfloat16 *__restrict__ q,
    const __nv_bfloat16 *__restrict__ k,
    const __nv_bfloat16 *__restrict__ v,
    const float *__restrict__ state,
    const float *__restrict__ A_log,
    const __nv_bfloat16 *__restrict__ a,
    const float *__restrict__ dt_bias,
    const __nv_bfloat16 *__restrict__ b,
    const int64_t *__restrict__ cu_seqlens,
    float scale,
    __nv_bfloat16 *__restrict__ output,
    float *__restrict__ new_state,
    int64_t num_seqs) {

  const int64_t block_linear = static_cast<int64_t>(blockIdx.x);
  const int64_t cta_idx = block_linear % kNumCTAs;
  const int64_t bh = block_linear / kNumCTAs;
  const int64_t seq_idx = bh / kNumVHeads;
  const int64_t hv_idx = bh % kNumVHeads;

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & (kWarpSize - 1);

  const int64_t q_head = hv_idx / kQGroupSize;
  const int64_t k_head = hv_idx / kKGroupSize;

  const int v_start = cta_idx * kVTilesPerCTA * kTileV;
  const int64_t state_base = (seq_idx * kNumVHeads + hv_idx) * kHeadSize;

  const float negated_exp_A_log = -expf(A_log[hv_idx]);
  const float r_dt_bias = dt_bias[hv_idx];

  // Each warp carries kItersPerTile * kVTilesPerCTA = 4 state vectors
  float4 state_regs[kVTilesPerCTA * kItersPerTile];

  if (state != nullptr) {
#pragma unroll
    for (int tile = 0; tile < kVTilesPerCTA; ++tile) {
#pragma unroll
      for (int iter = 0; iter < kItersPerTile; ++iter) {
        const int row_in_tile = iter * kRowsPerIter + warp_id;
        const int global_v = v_start + tile * kTileV + row_in_tile;
        const int64_t row_base =
            (state_base + global_v) * static_cast<int64_t>(kHeadSize);
        state_regs[tile * kItersPerTile + iter] =
            reinterpret_cast<const float4 *>(state + row_base)[lane];
      }
    }
  } else {
#pragma unroll
    for (int i = 0; i < kVTilesPerCTA * kItersPerTile; ++i) {
      state_regs[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
  }

  const int64_t seq_start = cu_seqlens[seq_idx];
  const int64_t seq_end = cu_seqlens[seq_idx + 1];
  const int kk_base = lane * kElemsPerLane;

  for (int64_t t = seq_start; t < seq_end; t++) {
    const int64_t hv_base = t * kNumVHeads + hv_idx;
    const int64_t q_base = (t * kNumQHeads + q_head) * kHeadSize;
    const int64_t k_base = (t * kNumKHeads + k_head) * kHeadSize;

    // Load q, k once per token (reused across all V rows in this CTA)
    const float4 q_vec = LoadBf16x4GlobalNc(q + q_base + kk_base);
    const float4 k_vec = LoadBf16x4GlobalNc(k + k_base + kk_base);

    // Compute g, beta on lane 0 and broadcast (shared across all V rows)
    float g, beta;
    if (lane == 0) {
      const float x = __bfloat162float(a[hv_base]) + r_dt_bias;
      g = expf(negated_exp_A_log * SoftplusStable(x));
      beta = Sigmoid(__bfloat162float(b[hv_base]));
    }
    g = __shfl_sync(kFullWarpMask, g, 0);
    beta = __shfl_sync(kFullWarpMask, beta, 0);

    // Process all V rows this warp handles
#pragma unroll
    for (int tile = 0; tile < kVTilesPerCTA; ++tile) {
#pragma unroll
      for (int iter = 0; iter < kItersPerTile; ++iter) {
        const int reg_idx = tile * kItersPerTile + iter;
        const int row_in_tile = iter * kRowsPerIter + warp_id;
        const int global_v = v_start + tile * kTileV + row_in_tile;
        const int64_t v_offset = hv_base * kHeadSize + global_v;

        const float v_scalar = __bfloat162float(v[v_offset]);

        float4 sv = state_regs[reg_idx];
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
        const float delta = beta * (v_scalar - old_v);

        h.x = fmaf(k_vec.x, delta, h.x);
        h.y = fmaf(k_vec.y, delta, h.y);
        h.z = fmaf(k_vec.z, delta, h.z);
        h.w = fmaf(k_vec.w, delta, h.w);

        state_regs[reg_idx] = h;

        float sum_hq = q_vec.x * h.x;
        sum_hq = fmaf(q_vec.y, h.y, sum_hq);
        sum_hq = fmaf(q_vec.z, h.z, sum_hq);
        sum_hq = fmaf(q_vec.w, h.w, sum_hq);
        const float out_acc = WarpAllReduceSum(sum_hq);

        if (lane == 0) {
          output[v_offset] = __float2bfloat16_rn(scale * out_acc);
        }
      }
    }
  }

  // Write final state to global memory
#pragma unroll
  for (int tile = 0; tile < kVTilesPerCTA; ++tile) {
#pragma unroll
    for (int iter = 0; iter < kItersPerTile; ++iter) {
      const int reg_idx = tile * kItersPerTile + iter;
      const int row_in_tile = iter * kRowsPerIter + warp_id;
      const int global_v = v_start + tile * kTileV + row_in_tile;
      const int64_t row_base =
          (state_base + global_v) * static_cast<int64_t>(kHeadSize);
      StoreF32x4Global(new_state + row_base + lane * kElemsPerLane,
                        state_regs[reg_idx]);
    }
  }
}

//// Host launch

__host__ __forceinline__ float ResolveScale(double scale) {
  float scale_f = static_cast<float>(scale);
  if (scale_f == 0.0f) {
    scale_f = 1.0f / sqrtf(static_cast<float>(kHeadSize));
  }
  return scale_f;
}

void RunGdnPrefillRecurrentV2(
    TensorView q, TensorView k, TensorView v,
    ffi::Optional<TensorView> state_opt, TensorView A_log, TensorView a,
    TensorView dt_bias, TensorView b, TensorView cu_seqlens, double scale,
    TensorView output, TensorView new_state) {

  gdn_prefill::ValidateShapesAndTypes(q, k, v, A_log, a, dt_bias, b,
                                      cu_seqlens, output, new_state);

  const int64_t num_seqs = cu_seqlens.size(0) - 1;

  const float *state_ptr = nullptr;
  if (state_opt.has_value()) {
    TensorView state = state_opt.value();
    gdn_prefill::ValidateState(state, num_seqs);
    CHECK_DEVICE(q, state);
    state_ptr = static_cast<const float *>(state.data_ptr());
  }

  const float scale_f = ResolveScale(scale);
  ffi::CUDADeviceGuard guard(q.device().device_id);
  const cudaStream_t stream = get_cuda_stream(q.device());

  const dim3 grid(num_seqs * kNumVHeads * kNumCTAs, 1, 1);
  GdnPrefillRecurrentV2Kernel<<<grid, kNumThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(q.data_ptr()),
      static_cast<const __nv_bfloat16 *>(k.data_ptr()),
      static_cast<const __nv_bfloat16 *>(v.data_ptr()),
      state_ptr,
      static_cast<const float *>(A_log.data_ptr()),
      static_cast<const __nv_bfloat16 *>(a.data_ptr()),
      static_cast<const float *>(dt_bias.data_ptr()),
      static_cast<const __nv_bfloat16 *>(b.data_ptr()),
      static_cast<const int64_t *>(cu_seqlens.data_ptr()),
      scale_f,
      static_cast<__nv_bfloat16 *>(output.data_ptr()),
      static_cast<float *>(new_state.data_ptr()),
      num_seqs);

  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    TVM_FFI_THROW(RuntimeError)
        << "GdnPrefillRecurrentV2 launch failed: "
        << cudaGetErrorString(err);
  }
}

} // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(gdn_prefill_recurrent_v2,
                               RunGdnPrefillRecurrentV2);
