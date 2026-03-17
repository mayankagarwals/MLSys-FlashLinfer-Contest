/*
SV: Adopted from
https://github.com/tomasruizt/flashinfer-competition-codebase/blob/main/solution/cuda/kernel.cu
for our reference
*/

/*
 * CUDA kernels for GDN (Gated Delta Net) decode.
 *
 * Two architectures:
 *   v1: 1 warp (32 threads) per block, BV=8 V-rows per block, 128 blocks.
 *       Best on B200 (maximizes block count for SM coverage).
 *   v4: 8 warps (256 threads) per block, 1 V-row per warp, 128 blocks.
 *       Matches Triton FLA parallelism (3x more active warps per scheduler).
 *
 * Uses TVM FFI for framework integration (compiled by TVMFFIBuilder).
 *
 * Fast math: __expf/__logf intrinsics and --use_fast_math are already applied.
 * Benchmarked on both RTX 3090 and B200: no measurable effect (gate math is
 * fully hidden under state memory latency). See findings/research.md.
 * We keep it to prevent future readers from wasting time trying to "optimize"
 * the math.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

static constexpr int BK = 128;
static constexpr int WARP_SIZE = 32;
static constexpr int KVEC = BK / WARP_SIZE; // 4

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, offset);
  }
  return val;
}

// ===========================================================================
// V1 kernel: 1 warp per block, BV V-rows per block, state in registers
// ===========================================================================
//
// Grid: (V/BV, B*HV) = (16, 8) = 128 blocks with BV=8.
// Each block handles a [BV, BK=128] tile of one head's state matrix.
// Each thread owns KVEC=4 consecutive K-elements across BV rows.

template <int BV>
__global__ void
gdn_decode_kernel(const __nv_bfloat16 *__restrict__ q, // [B, 1, H_qk, K]
                  const __nv_bfloat16 *__restrict__ k, // [B, 1, H_qk, K]
                  const __nv_bfloat16 *__restrict__ v, // [B, 1, HV, V]
                  const float *__restrict__ h0, // [B, HV, V, K] state (k-last)
                  const float *__restrict__ A_log,          // [HV]
                  const __nv_bfloat16 *__restrict__ a_gate, // [B, 1, HV]
                  const float *__restrict__ dt_bias,        // [HV]
                  const __nv_bfloat16 *__restrict__ b_gate, // [B, 1, HV]
                  float scale,
                  __nv_bfloat16 *__restrict__ output, // [B, 1, HV, V]
                  float *__restrict__ ht, // [B, HV, V, K] new state (k-last)
                  int H_qk, int HV, int K_dim, int V_dim) {
  // Grid mapping (same as Triton kernel)
  const int i_v = blockIdx.x;  // V-tile index [0..V/BV-1]
  const int i_nh = blockIdx.y; // batch*HV flattened
  const int i_hv = i_nh % HV;  // v-head index
  const int i_h =
      i_hv / (HV / H_qk);    // q/k-head index (GVA: 2 v-heads per qk-head)
  const int i_n = i_nh / HV; // batch index

  const int tid = threadIdx.x;   // [0..31]
  const int k_base = tid * KVEC; // K-offset for this thread

  // --- Load state tile [BV, BK] f32 via float4 (coalesced 128-bit) ---
  const int state_head_offset = i_nh * V_dim * K_dim;
  float h[BV][KVEC];
#pragma unroll
  for (int bv = 0; bv < BV; bv++) {
    const int v_idx = i_v * BV + bv;
    float4 val = *reinterpret_cast<const float4 *>(h0 + state_head_offset +
                                                   v_idx * K_dim + k_base);
    h[bv][0] = val.x;
    h[bv][1] = val.y;
    h[bv][2] = val.z;
    h[bv][3] = val.w;
  }

  // --- Load q, k [KVEC per thread] bf16 -> f32 via vectorized 8-byte loads ---
  float q_reg[KVEC], k_reg[KVEC];
  {
    const int qk_offset = (i_n * H_qk + i_h) * K_dim + k_base;
    uint2 q_raw = *reinterpret_cast<const uint2 *>(q + qk_offset);
    uint2 k_raw = *reinterpret_cast<const uint2 *>(k + qk_offset);
    __nv_bfloat162 q01 = *reinterpret_cast<__nv_bfloat162 *>(&q_raw.x);
    __nv_bfloat162 q23 = *reinterpret_cast<__nv_bfloat162 *>(&q_raw.y);
    __nv_bfloat162 k01 = *reinterpret_cast<__nv_bfloat162 *>(&k_raw.x);
    __nv_bfloat162 k23 = *reinterpret_cast<__nv_bfloat162 *>(&k_raw.y);
    q_reg[0] = __bfloat162float(q01.x) * scale;
    q_reg[1] = __bfloat162float(q01.y) * scale;
    q_reg[2] = __bfloat162float(q23.x) * scale;
    q_reg[3] = __bfloat162float(q23.y) * scale;
    k_reg[0] = __bfloat162float(k01.x);
    k_reg[1] = __bfloat162float(k01.y);
    k_reg[2] = __bfloat162float(k23.x);
    k_reg[3] = __bfloat162float(k23.y);
  }

  // --- Load v [BV] bf16 -> f32 via vectorized 16-byte load (broadcast) ---
  float v_reg[BV];
  {
    const __nv_bfloat16 *v_ptr = v + (i_n * HV + i_hv) * V_dim + i_v * BV;
    uint4 v_raw = *reinterpret_cast<const uint4 *>(v_ptr);
    __nv_bfloat162 v01 = *reinterpret_cast<__nv_bfloat162 *>(&v_raw.x);
    __nv_bfloat162 v23 = *reinterpret_cast<__nv_bfloat162 *>(&v_raw.y);
    __nv_bfloat162 v45 = *reinterpret_cast<__nv_bfloat162 *>(&v_raw.z);
    __nv_bfloat162 v67 = *reinterpret_cast<__nv_bfloat162 *>(&v_raw.w);
    v_reg[0] = __bfloat162float(v01.x);
    v_reg[1] = __bfloat162float(v01.y);
    v_reg[2] = __bfloat162float(v23.x);
    v_reg[3] = __bfloat162float(v23.y);
    v_reg[4] = __bfloat162float(v45.x);
    v_reg[5] = __bfloat162float(v45.y);
    v_reg[6] = __bfloat162float(v67.x);
    v_reg[7] = __bfloat162float(v67.y);
  }

  // --- Load scalars and compute gates ---
  const float b_A = A_log[i_hv];
  const float b_a = __bfloat162float(a_gate[i_n * HV + i_hv]);
  const float b_dt = dt_bias[i_hv];
  const float b_b = __bfloat162float(b_gate[i_n * HV + i_hv]);

  const float x = b_a + b_dt;
  const float sp = (x > 20.0f) ? x : __logf(1.0f + __expf(x)); // softplus
  const float g = __expf(-__expf(b_A) * sp);                   // decay gate
  const float beta = __frcp_rn(1.0f + __expf(-b_b));           // sigmoid

// --- Decay state: h *= g ---
#pragma unroll
  for (int bv = 0; bv < BV; bv++) {
#pragma unroll
    for (int i = 0; i < KVEC; i++) {
      h[bv][i] *= g;
    }
  }

// --- Delta rule: matvec k@S, blend, outer product (fused per v-row) ---
#pragma unroll
  for (int bv = 0; bv < BV; bv++) {
    // Partial dot product k . h[bv] (4 elements per thread)
    float partial = 0.0f;
#pragma unroll
    for (int i = 0; i < KVEC; i++) {
      partial = fmaf(k_reg[i], h[bv][i], partial);
    }
    // Warp all-reduce: every thread gets old_v[bv]
    float old_v_bv = warp_reduce_sum(partial);

    // Blend: delta_v = beta * (v - old_v)
    float dv = beta * (v_reg[bv] - old_v_bv);

// Outer product: h[bv][k] += dv * k[k]
#pragma unroll
    for (int i = 0; i < KVEC; i++) {
      h[bv][i] = fmaf(dv, k_reg[i], h[bv][i]);
    }
  }

  // --- Output matvec q@S -> out[BV] (warp reduce, 8 threads each store 1 bf16)
  // ---
  {
    __nv_bfloat16 *o_ptr = output + (i_n * HV + i_hv) * V_dim + i_v * BV;
    float out_vals[BV];
#pragma unroll
    for (int bv = 0; bv < BV; bv++) {
      float partial = 0.0f;
#pragma unroll
      for (int i = 0; i < KVEC; i++) {
        partial = fmaf(q_reg[i], h[bv][i], partial);
      }
      out_vals[bv] = warp_reduce_sum(partial);
    }
    if (tid < BV) {
      o_ptr[tid] = __float2bfloat16(out_vals[tid]);
    }
  }

// --- Store state tile [BV, BK] f32 via float4, streaming (bypass L2) ---
#pragma unroll
  for (int bv = 0; bv < BV; bv++) {
    const int v_idx = i_v * BV + bv;
    float4 val;
    val.x = h[bv][0];
    val.y = h[bv][1];
    val.z = h[bv][2];
    val.w = h[bv][3];
    __stcs(reinterpret_cast<float4 *>(ht + state_head_offset + v_idx * K_dim +
                                      k_base),
           val);
  }
}

// ---------------------------------------------------------------------------
// TVM FFI host wrapper for v1 (BV=8, 128 blocks)
// ---------------------------------------------------------------------------

void KernelCuda(tvm::ffi::TensorView q, tvm::ffi::TensorView k,
                tvm::ffi::TensorView v, tvm::ffi::TensorView state,
                tvm::ffi::TensorView A_log, tvm::ffi::TensorView a,
                tvm::ffi::TensorView dt_bias, tvm::ffi::TensorView b_gate,
                double scale, tvm::ffi::TensorView output,
                tvm::ffi::TensorView new_state) {
  const int B = q.size(0);
  const int H_qk = q.size(2);
  const int HV = v.size(2);
  const int V_dim = v.size(3);
  const int K_dim = q.size(3);
  dim3 grid(V_dim / 8, B * HV);
  dim3 block(WARP_SIZE);
  DLDevice dev = q.device();
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(dev.device_type, dev.device_id));
  gdn_decode_kernel<8><<<grid, block, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(q.data_ptr()),
      static_cast<const __nv_bfloat16 *>(k.data_ptr()),
      static_cast<const __nv_bfloat16 *>(v.data_ptr()),
      static_cast<const float *>(state.data_ptr()),
      static_cast<const float *>(A_log.data_ptr()),
      static_cast<const __nv_bfloat16 *>(a.data_ptr()),
      static_cast<const float *>(dt_bias.data_ptr()),
      static_cast<const __nv_bfloat16 *>(b_gate.data_ptr()),
      static_cast<float>(scale),
      static_cast<__nv_bfloat16 *>(output.data_ptr()),
      static_cast<float *>(new_state.data_ptr()), H_qk, HV, K_dim, V_dim);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(kernel_cuda, KernelCuda);

// ===========================================================================
// V4 kernel: multi-warp, 1 V-row per warp, no shared memory
// ===========================================================================
//
// Grid: (V / NUM_WARPS, B * HV).  With NUM_WARPS=8: (16, 8) = 128 blocks.
// Block: NUM_WARPS * 32 threads.  Each warp independently handles 1 V-row.
// No cross-warp communication, no shared memory.
//
// Matches the Triton FLA kernel's parallelism model: same 128-block grid but
// 8 warps per block (256 threads) instead of v1's 1 warp (32 threads).
// This gives 3x more active warps per scheduler for latency hiding.

template <int NUM_WARPS>
__global__ __launch_bounds__(NUM_WARPS *WARP_SIZE) void gdn_decode_v4_kernel(
    const __nv_bfloat16 *__restrict__ q,      // [B, 1, H_qk, K]
    const __nv_bfloat16 *__restrict__ k,      // [B, 1, H_qk, K]
    const __nv_bfloat16 *__restrict__ v,      // [B, 1, HV, V]
    const float *__restrict__ h0,             // [B, HV, V, K] state (k-last)
    const float *__restrict__ A_log,          // [HV]
    const __nv_bfloat16 *__restrict__ a_gate, // [B, 1, HV]
    const float *__restrict__ dt_bias,        // [HV]
    const __nv_bfloat16 *__restrict__ b_gate, // [B, 1, HV]
    float scale,
    __nv_bfloat16 *__restrict__ output, // [B, 1, HV, V]
    float *__restrict__ ht,             // [B, HV, V, K] new state (k-last)
    int H_qk, int HV, int K_dim, int V_dim) {
  const int i_v_block = blockIdx.x; // V-tile index
  const int i_nh = blockIdx.y;      // batch*HV flattened
  const int i_hv = i_nh % HV;
  const int i_h = i_hv / (HV / H_qk); // q/k-head index (GVA)
  const int i_n = i_nh / HV;

  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE; // [0..NUM_WARPS-1]
  const int lane = tid % WARP_SIZE;    // [0..31]
  const int k_base = lane * KVEC;

  // Each warp handles 1 V-row
  const int v_idx = i_v_block * NUM_WARPS + warp_id;

  // --- Load state [1, 128] for this V-row via float4 ---
  const int state_head_offset = i_nh * V_dim * K_dim;
  float h[KVEC];
  {
    float4 val = *reinterpret_cast<const float4 *>(h0 + state_head_offset +
                                                   v_idx * K_dim + k_base);
    h[0] = val.x;
    h[1] = val.y;
    h[2] = val.z;
    h[3] = val.w;
  }

  // --- Load q, k [KVEC per thread] bf16 -> f32 via vectorized 8-byte loads ---
  // Each warp loads independently; hits L1/L2 cache (256 bytes each)
  float q_reg[KVEC], k_reg[KVEC];
  {
    const int qk_offset = (i_n * H_qk + i_h) * K_dim + k_base;
    uint2 q_raw = *reinterpret_cast<const uint2 *>(q + qk_offset);
    uint2 k_raw = *reinterpret_cast<const uint2 *>(k + qk_offset);
    __nv_bfloat162 q01 = *reinterpret_cast<__nv_bfloat162 *>(&q_raw.x);
    __nv_bfloat162 q23 = *reinterpret_cast<__nv_bfloat162 *>(&q_raw.y);
    __nv_bfloat162 k01 = *reinterpret_cast<__nv_bfloat162 *>(&k_raw.x);
    __nv_bfloat162 k23 = *reinterpret_cast<__nv_bfloat162 *>(&k_raw.y);
    q_reg[0] = __bfloat162float(q01.x) * scale;
    q_reg[1] = __bfloat162float(q01.y) * scale;
    q_reg[2] = __bfloat162float(q23.x) * scale;
    q_reg[3] = __bfloat162float(q23.y) * scale;
    k_reg[0] = __bfloat162float(k01.x);
    k_reg[1] = __bfloat162float(k01.y);
    k_reg[2] = __bfloat162float(k23.x);
    k_reg[3] = __bfloat162float(k23.y);
  }

  // --- Load v (1 scalar per warp, broadcast) ---
  const float v_val = __bfloat162float(v[(i_n * HV + i_hv) * V_dim + v_idx]);

  // --- Gates (same across all threads in block) ---
  const float b_A = A_log[i_hv];
  const float b_a = __bfloat162float(a_gate[i_n * HV + i_hv]);
  const float b_dt = dt_bias[i_hv];
  const float b_b = __bfloat162float(b_gate[i_n * HV + i_hv]);

  const float x = b_a + b_dt;
  const float sp = (x > 20.0f) ? x : __logf(1.0f + __expf(x)); // softplus
  const float g = __expf(-__expf(b_A) * sp);                   // decay gate
  const float beta = __frcp_rn(1.0f + __expf(-b_b));           // sigmoid

// --- Decay state: h *= g ---
#pragma unroll
  for (int i = 0; i < KVEC; i++) {
    h[i] *= g;
  }

  // --- Delta rule: k@h dot product, blend, outer product ---
  {
    float partial = 0.0f;
#pragma unroll
    for (int i = 0; i < KVEC; i++) {
      partial = fmaf(k_reg[i], h[i], partial);
    }
    float old_v = warp_reduce_sum(partial);
    float dv = beta * (v_val - old_v);
#pragma unroll
    for (int i = 0; i < KVEC; i++) {
      h[i] = fmaf(dv, k_reg[i], h[i]);
    }
  }

  // --- Output: q@h -> 1 scalar per warp, lane 0 stores ---
  {
    float partial = 0.0f;
#pragma unroll
    for (int i = 0; i < KVEC; i++) {
      partial = fmaf(q_reg[i], h[i], partial);
    }
    float out_val = warp_reduce_sum(partial);
    if (lane == 0) {
      output[(i_n * HV + i_hv) * V_dim + v_idx] = __float2bfloat16(out_val);
    }
  }

  // --- Store state via float4, streaming (bypass L2) ---
  {
    float4 val;
    val.x = h[0];
    val.y = h[1];
    val.z = h[2];
    val.w = h[3];
    __stcs(reinterpret_cast<float4 *>(ht + state_head_offset + v_idx * K_dim +
                                      k_base),
           val);
  }
}

// ---------------------------------------------------------------------------
// TVM FFI host wrapper for v4 (8 warps, 128 blocks)
// ---------------------------------------------------------------------------

void KernelCudaV4(tvm::ffi::TensorView q, tvm::ffi::TensorView k,
                  tvm::ffi::TensorView v, tvm::ffi::TensorView state,
                  tvm::ffi::TensorView A_log, tvm::ffi::TensorView a,
                  tvm::ffi::TensorView dt_bias, tvm::ffi::TensorView b_gate,
                  double scale, tvm::ffi::TensorView output,
                  tvm::ffi::TensorView new_state) {
  const int B = q.size(0);
  const int H_qk = q.size(2);
  const int HV = v.size(2);
  const int V_dim = v.size(3);
  const int K_dim = q.size(3);
  dim3 grid(V_dim / 8, B * HV);
  dim3 block(8 * WARP_SIZE);
  DLDevice dev = q.device();
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(dev.device_type, dev.device_id));
  gdn_decode_v4_kernel<8><<<grid, block, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(q.data_ptr()),
      static_cast<const __nv_bfloat16 *>(k.data_ptr()),
      static_cast<const __nv_bfloat16 *>(v.data_ptr()),
      static_cast<const float *>(state.data_ptr()),
      static_cast<const float *>(A_log.data_ptr()),
      static_cast<const __nv_bfloat16 *>(a.data_ptr()),
      static_cast<const float *>(dt_bias.data_ptr()),
      static_cast<const __nv_bfloat16 *>(b_gate.data_ptr()),
      static_cast<float>(scale),
      static_cast<__nv_bfloat16 *>(output.data_ptr()),
      static_cast<float *>(new_state.data_ptr()), H_qk, HV, K_dim, V_dim);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(kernel_cuda_v4, KernelCudaV4);