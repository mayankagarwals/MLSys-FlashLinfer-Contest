/*
 * Fused H+O Kernel for GDN Prefill — Blackwell SM_100a (B200)
 *
 * Fuses the H recurrence and O output computation into a single kernel,
 * eliminating ~60MB of global memory traffic per call (h: 40MB, v_new: 20MB).
 *
 * Grid: (1, num_seqs * kHv)  — one TB per (seq, head), processes ALL chunks
 * Block: 128 threads (4 warps)
 * Shared memory: ~136KB (fits B200's 228KB)
 *
 * Per-chunk computation phases:
 *   Phase 1: Load w,k,gc. MMA w@H^T → wh, load u, v_new = u - wh
 *   Phase 2: Load q. MMA q@H^T → o_inter (scaled by exp(g))
 *   Phase 3: MMA q@K^T → A_causal (gated + causal mask)
 *   Phase 4: Convert A_causal→bf16, vnew→bf16. MMA A@vnew → o_intra
 *            o = (o_inter + o_intra) * scale → global output
 *   Phase 5: Scale h by alpha, prepare vnew_scaled^T, MMA vnew_scaled^T@k → h update
 *
 * Shared memory layout (~136KB):
 *   s_h:      [128][128] fp32 = 64KB  (persistent state, row-major)
 *   s_buf1:   [64][128]  bf16 = 16KB  (w → q → vnew_bf16 → vnew_scaled_T)
 *   s_buf2:   [64][128]  bf16 = 16KB  (k, persistent for chunk)
 *   s_fp:     [64][128]  fp32 = 32KB  (vnew fp32 intermediate)
 *   s_acaus:  [64][64]   bf16 = 8KB   (A_causal bf16 for MMA)
 *   s_gc:     [64]       fp32 = 256B
 *
 * All bf16 smem buffers use XOR swizzle for bank-conflict-free ldmatrix.
 * Uses mma.sync m16n8k16 bf16 for ALL matrix multiplications.
 */

#include "cuda_utils.h"
#include <cstdint>
#include <math.h>

namespace {

constexpr int kK = 128, kV = 128;
constexpr int64_t kHq = 4, kHk = 4, kHv = 8;
constexpr int kBT = 64;
constexpr int TB_SIZE = 128;

// XOR swizzle for bank-conflict-free ldmatrix
__device__ __forceinline__
int swz(int row, int col, int cg_mask) {
  int cg = col >> 3;
  return ((cg ^ (row & cg_mask)) << 3) | (col & 7);
}

constexpr int CGM_128 = 15;  // (128/8)-1
constexpr int CGM_64  = 7;   // (64/8)-1

// Smem layout (with bf16 H conversion buffer)
constexpr int OFF_H     = 0;
constexpr int SZ_H      = 128 * 128 * 4;            // 65536 (fp32 h state)
constexpr int OFF_HBF16 = SZ_H;
constexpr int SZ_HBF16  = 128 * 128 * 2;            // 32768 (bf16 h for ldmatrix_trans)
constexpr int OFF_BUF1  = OFF_HBF16 + SZ_HBF16;
constexpr int SZ_BUF    = 64 * 128 * 2;             // 16384
constexpr int OFF_BUF2  = OFF_BUF1 + SZ_BUF;
constexpr int OFF_FP    = OFF_BUF2 + SZ_BUF;
constexpr int SZ_FP     = 64 * 128 * 4;             // 32768
constexpr int OFF_ACAUS = OFF_FP + SZ_FP;
constexpr int SZ_ACAUS  = 64 * 64 * 4;              // 16384 (fp32 for precision)
constexpr int OFF_GC    = OFF_ACAUS + SZ_ACAUS;
constexpr int SZ_GC     = 64 * 4;                   // 256
constexpr int SMEM_TOTAL = OFF_GC + SZ_GC;          // 172288 ≈ 168.25KB (fits B200's 228KB)

// ═══════════════════════════════════════════════════════════════════
// Helper: load [rows, cols] bf16 from global → XOR-swizzled smem
// ═══════════════════════════════════════════════════════════════════
__device__ __forceinline__
void load_bf16_swizzled(
    __nv_bfloat16 *dst, int stride, int cg_mask,
    const __nv_bfloat16 *src, int src_stride,
    int rows, int cols, int valid_rows, int tid)
{
  for (int i = tid; i < rows * cols / 8; i += TB_SIZE) {
    int row = i / (cols / 8), col8 = (i % (cols / 8)) * 8;
    int sc = swz(row, col8, cg_mask);
    if (row < valid_rows) {
      *reinterpret_cast<int4 *>(&dst[row * stride + sc]) =
          *reinterpret_cast<const int4 *>(&src[row * src_stride + col8]);
    } else {
      *reinterpret_cast<int4 *>(&dst[row * stride + sc]) = make_int4(0, 0, 0, 0);
    }
  }
}

// ═══════════════════════════════════════════════════════════════════
// Helper: construct B fragment for mma.sync m16n8k16 from transposed data.
// Used when B = X^T and X is in row-major smem with XOR swizzle.
// B[kr, nj] = X[nj, kr] where kr is K-row, nj is N-col.
//
// For mma.sync m16n8k16 B fragment (col-major [K=16, N=8]):
//   lane_id%4 → column pair index within N-tile (cp: 0..3)
//   lane_id/4 → K-row offset within 8-row group (kr_off: 0..7)
//   b[0] packs {X[n0, k0], X[n1, k0]} for K-rows 0..7
//   b[1] packs {X[n0, k1], X[n1, k1]} for K-rows 8..15
//   where n0 = nt*8 + cp*2, n1 = n0+1, k0 = kt*16 + kr_off, k1 = k0+8
// ═══════════════════════════════════════════════════════════════════
__device__ __forceinline__
void load_b_transposed(
    uint32_t b[2],
    const __nv_bfloat16 *X,  // row-major [N_total, K_total] with XOR swizzle
    int X_stride,             // stride (cols) of X
    int cg_mask,              // XOR swizzle mask for X
    int kt, int nt,           // K-tile and N-tile indices
    int lane_id)
{
  int cp  = lane_id % 4;
  int kr0 = lane_id / 4;
  int kr1 = kr0 + 8;
  int n0  = nt * 8 + cp * 2;
  int n1  = n0 + 1;
  int k0  = kt * 16 + kr0;
  int k1  = kt * 16 + kr1;

  __nv_bfloat16 v00 = X[n0 * X_stride + swz(n0, k0, cg_mask)];
  __nv_bfloat16 v01 = X[n1 * X_stride + swz(n1, k0, cg_mask)];
  __nv_bfloat16 v10 = X[n0 * X_stride + swz(n0, k1, cg_mask)];
  __nv_bfloat16 v11 = X[n1 * X_stride + swz(n1, k1, cg_mask)];

  b[0] = (uint32_t)__bfloat16_as_ushort(v00) | ((uint32_t)__bfloat16_as_ushort(v01) << 16);
  b[1] = (uint32_t)__bfloat16_as_ushort(v10) | ((uint32_t)__bfloat16_as_ushort(v11) << 16);
}

// Same but reading from fp32 smem (s_h) and converting to bf16
__device__ __forceinline__
void load_b_transposed_fp32(
    uint32_t b[2],
    const float *X,  // row-major [N_total, K_total]
    int X_stride,     // stride
    int kt, int nt,
    int lane_id)
{
  int cp  = lane_id % 4;
  int kr0 = lane_id / 4;
  int kr1 = kr0 + 8;
  int n0  = nt * 8 + cp * 2;
  int n1  = n0 + 1;
  int k0  = kt * 16 + kr0;
  int k1  = kt * 16 + kr1;

  __nv_bfloat16 v00 = __float2bfloat16(X[n0 * X_stride + k0]);
  __nv_bfloat16 v01 = __float2bfloat16(X[n1 * X_stride + k0]);
  __nv_bfloat16 v10 = __float2bfloat16(X[n0 * X_stride + k1]);
  __nv_bfloat16 v11 = __float2bfloat16(X[n1 * X_stride + k1]);

  b[0] = (uint32_t)__bfloat16_as_ushort(v00) | ((uint32_t)__bfloat16_as_ushort(v01) << 16);
  b[1] = (uint32_t)__bfloat16_as_ushort(v10) | ((uint32_t)__bfloat16_as_ushort(v11) << 16);
}

// ═══════════════════════════════════════════════════════════════════
// FusedHOKernel
// ═══════════════════════════════════════════════════════════════════
__global__ void __launch_bounds__(128, 1)
FusedHOKernel(
    const __nv_bfloat16 *__restrict__ q_in,      // [T, Hq, K] bf16
    const __nv_bfloat16 *__restrict__ k_in,      // [T, Hk, K] bf16
    const __nv_bfloat16 *__restrict__ w_in,      // [T, Hv, K] bf16
    const __nv_bfloat16 *__restrict__ u_in,      // [T, Hv, V] bf16
    const float *__restrict__ g_cu_in,            // [T, Hv] fp32
    const float *__restrict__ state0,             // [N, Hv, V, K] fp32 or nullptr
    const int64_t *__restrict__ cu_seqlens,       // [N+1] int64
    float scale,
    __nv_bfloat16 *__restrict__ output,           // [T, Hv, V] bf16
    float *__restrict__ new_state,                // [N, Hv, V, K] fp32
    int64_t num_seqs)
{
  const int hv = blockIdx.y % kHv;
  const int seq_id = blockIdx.y / kHv;
  if (seq_id >= num_seqs) return;

  const int64_t bos = cu_seqlens[seq_id];
  const int64_t eos = cu_seqlens[seq_id + 1];
  const int seqlen = (int)(eos - bos);
  const int num_chunks = (seqlen + kBT - 1) / kBT;
  const int k_head = hv / (kHv / kHk);
  const int q_head = hv / (kHv / kHq);

  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;

  extern __shared__ __align__(256) char smem[];
  float         *s_h     = reinterpret_cast<float *>(smem + OFF_H);
  __nv_bfloat16 *s_buf1  = reinterpret_cast<__nv_bfloat16 *>(smem + OFF_BUF1);
  __nv_bfloat16 *s_hbf16 = reinterpret_cast<__nv_bfloat16 *>(smem + OFF_HBF16);
  __nv_bfloat16 *s_buf2  = reinterpret_cast<__nv_bfloat16 *>(smem + OFF_BUF2);
  float         *s_fp    = reinterpret_cast<float *>(smem + OFF_FP);
  float         *s_acaus = reinterpret_cast<float *>(smem + OFF_ACAUS);  // fp32 for precision
  float         *s_gc    = reinterpret_cast<float *>(smem + OFF_GC);

  // ═══ Initialize h state ═══
  {
    const float *h0 = state0 ? (state0 + ((int64_t)seq_id * kHv + hv) * kV * kK) : nullptr;
    for (int i = tid; i < kV * kK; i += TB_SIZE)
      s_h[i] = h0 ? h0[i] : 0.0f;
  }
  __syncthreads();

  // ═══ Chunk loop ═══
  for (int ct = 0; ct < num_chunks; ct++) {
    const int64_t cstart = bos + (int64_t)ct * kBT;
    const int clen = min(kBT, seqlen - ct * kBT);

    // Convert h fp32 → h bf16 with XOR swizzle (for ldmatrix_trans in MMA)
    // s_hbf16 stores H^T layout: s_hbf16[k_row * kV + v_col] = bf16(s_h[v_col * kK + k_row])
    // This way ldmatrix_trans on s_hbf16 gives the correct B fragment for C = A @ H^T
    for (int i = tid; i < kV * kK / 8; i += TB_SIZE) {
      int idx = i * 8;
      int v_row = idx / kK, k_col8 = idx % kK;
      // Read 8 fp32 values from H[v_row, k_col8..k_col8+7]
      // Store transposed: s_hbf16[k_col+j][v_row] for j=0..7
      // In row-major with XOR swizzle: s_hbf16[(k_col8+j) * kV + swz(k_col8+j, v_row, CGM_128)]
      // But this is a scatter — not vectorized. Let me store ROW-MAJOR H^T instead:
      // s_hbf16[k_row * kV + v_col] = bf16(H[v_col, k_row])
      // Then: s_hbf16[(k_col8+j) * kV + v_row] — still scatter.
      //
      // Alternative: store H (not H^T) as bf16 with XOR swizzle, then use
      // load_b_transposed pattern on bf16 smem. This is simpler:
      // s_hbf16[v_row * kK + swz(v_row, k_col8, CGM_128)] = bf16(s_h[v_row * kK + k_col8..+7])
      int sc = swz(v_row, k_col8, CGM_128);
      int4 dst;
      __nv_bfloat162 *dp = reinterpret_cast<__nv_bfloat162 *>(&dst);
      float *src = &s_h[v_row * kK + k_col8];
      dp[0] = {__float2bfloat16(src[0]), __float2bfloat16(src[1])};
      dp[1] = {__float2bfloat16(src[2]), __float2bfloat16(src[3])};
      dp[2] = {__float2bfloat16(src[4]), __float2bfloat16(src[5])};
      dp[3] = {__float2bfloat16(src[6]), __float2bfloat16(src[7])};
      *reinterpret_cast<int4 *>(&s_hbf16[v_row * kK + sc]) = dst;
    }

    // Load g_cumsum
    if (tid < kBT)
      s_gc[tid] = (tid < clen) ? g_cu_in[(cstart + tid) * kHv + hv] : 0.0f;

    // Load w → s_buf1 [64, 128] bf16 XOR swizzled
    load_bf16_swizzled(s_buf1, kK, CGM_128,
                       w_in + cstart * kHv * kK + hv * kK, kHv * kK,
                       kBT, kK, clen, tid);

    // Load k → s_buf2 [64, 128] bf16 XOR swizzled
    load_bf16_swizzled(s_buf2, kK, CGM_128,
                       k_in + cstart * kHk * kK + k_head * kK, kHk * kK,
                       kBT, kK, clen, tid);
    __syncthreads();

    // ═════════════════════════════════════════════════════════════
    // Phase 1: v_new = U - W @ H^T
    //
    // MMA: W[64,128] @ H^T[128,128] → WH[64,128]
    // A = s_buf1 (w, [64,128] XOR swizzled)
    // B = H^T: H is [V=128, K=128] in s_h fp32. H^T[k,v] = s_h[v*128+k].
    // Manual B-fragment from s_h (fp32→bf16 on the fly).
    //
    // Each warp: 16 M-rows, 16 N-tiles(×8), 8 K-tiles(×16)
    // ═════════════════════════════════════════════════════════════
    {
      const int m_base = warp_id * 16;
      float wh[16][4];
      #pragma unroll 1
      for (int nt = 0; nt < 16; nt++)
        wh[nt][0] = wh[nt][1] = wh[nt][2] = wh[nt][3] = 0.0f;

      for (int kt = 0; kt < 8; kt++) {
        uint32_t a[4];
        {
          int r = (lane_id % 8) + ((lane_id & 8) ? 8 : 0) + m_base;
          int c = (lane_id >= 16) ? 8 : 0;
          ldmatrix<4>(a, cvt_smem_ptr(&s_buf1[r * kK + swz(r, kt * 16 + c, CGM_128)]));
        }

        for (int nt = 0; nt < 16; nt++) {
          // B = H^T[K, V]: B[kr, nj] = H[nj, kr] = s_h[nj * 128 + kr]
          uint32_t b[2];
          load_b_transposed(b, s_hbf16, kK, CGM_128, kt, nt, lane_id);

          mma_m16n8k16_bf16(
              wh[nt][0], wh[nt][1], wh[nt][2], wh[nt][3],
              a[0], a[1], a[2], a[3], b[0], b[1],
              wh[nt][0], wh[nt][1], wh[nt][2], wh[nt][3]);
        }
      }

      // v_new = U - WH → s_fp [64, 128] fp32
      {
        int gID = lane_id / 4, tIG = lane_id % 4;
        int r0 = m_base + gID, r1 = r0 + 8;

        for (int nt = 0; nt < 16; nt++) {
          int c0 = nt * 8 + tIG * 2;
          if (r0 < clen) {
            float u0 = __bfloat162float(u_in[(cstart + r0) * kHv * kV + hv * kV + c0]);
            float u1 = __bfloat162float(u_in[(cstart + r0) * kHv * kV + hv * kV + c0 + 1]);
            s_fp[r0 * kV + c0]     = u0 - wh[nt][0];
            s_fp[r0 * kV + c0 + 1] = u1 - wh[nt][1];
          } else if (r0 < kBT) {
            s_fp[r0 * kV + c0]     = 0.0f;
            s_fp[r0 * kV + c0 + 1] = 0.0f;
          }
          if (r1 < clen) {
            float u0 = __bfloat162float(u_in[(cstart + r1) * kHv * kV + hv * kV + c0]);
            float u1 = __bfloat162float(u_in[(cstart + r1) * kHv * kV + hv * kV + c0 + 1]);
            s_fp[r1 * kV + c0]     = u0 - wh[nt][2];
            s_fp[r1 * kV + c0 + 1] = u1 - wh[nt][3];
          } else if (r1 < kBT) {
            s_fp[r1 * kV + c0]     = 0.0f;
            s_fp[r1 * kV + c0 + 1] = 0.0f;
          }
        }
      }
    }
    __syncthreads();

    // Load q → s_buf1 (reuse, w no longer needed)
    load_bf16_swizzled(s_buf1, kK, CGM_128,
                       q_in + cstart * kHq * kK + q_head * kK, kHq * kK,
                       kBT, kK, clen, tid);
    __syncthreads();

    // ═════════════════════════════════════════════════════════════
    // Phase 2: o_inter = Q @ H_prev^T * exp(g)
    // Same MMA pattern as Phase 1 (Q in s_buf1, H in s_h fp32).
    // Result in o_acc registers, pre-scaled by exp(g[row]).
    // ═════════════════════════════════════════════════════════════
    float o_acc[16][4];
    {
      const int m_base = warp_id * 16;
      #pragma unroll 1
      for (int nt = 0; nt < 16; nt++)
        o_acc[nt][0] = o_acc[nt][1] = o_acc[nt][2] = o_acc[nt][3] = 0.0f;

      for (int kt = 0; kt < 8; kt++) {
        uint32_t a[4];
        {
          int r = (lane_id % 8) + ((lane_id & 8) ? 8 : 0) + m_base;
          int c = (lane_id >= 16) ? 8 : 0;
          ldmatrix<4>(a, cvt_smem_ptr(&s_buf1[r * kK + swz(r, kt * 16 + c, CGM_128)]));
        }

        for (int nt = 0; nt < 16; nt++) {
          uint32_t b[2];
          load_b_transposed(b, s_hbf16, kK, CGM_128, kt, nt, lane_id);

          mma_m16n8k16_bf16(
              o_acc[nt][0], o_acc[nt][1], o_acc[nt][2], o_acc[nt][3],
              a[0], a[1], a[2], a[3], b[0], b[1],
              o_acc[nt][0], o_acc[nt][1], o_acc[nt][2], o_acc[nt][3]);
        }
      }

      // Scale by exp(g[row])
      {
        int gID = lane_id / 4;
        int r0 = m_base + gID, r1 = r0 + 8;
        float eg0 = (r0 < clen) ? __expf(s_gc[r0]) : 0.0f;
        float eg1 = (r1 < clen) ? __expf(s_gc[r1]) : 0.0f;
        for (int nt = 0; nt < 16; nt++) {
          o_acc[nt][0] *= eg0;
          o_acc[nt][1] *= eg0;
          o_acc[nt][2] *= eg1;
          o_acc[nt][3] *= eg1;
        }
      }
    }
    // No __syncthreads needed — o_acc is in registers, and we still read s_buf1 (Q)
    // and s_buf2 (K) which haven't changed.

    // ═════════════════════════════════════════════════════════════
    // Phase 3: A_causal = Q @ K^T * exp(g_i - g_j) * causal_mask
    //
    // Q[64,128] in s_buf1, K[64,128] in s_buf2 (both XOR swizzled).
    // For B = K^T: B[k,n] = K[n,k]. Manual B-fragment from swizzled K.
    // Result: [64,64] → s_acaus as bf16 (XOR swizzled CGM_64).
    // ═════════════════════════════════════════════════════════════
    {
      const int m_base = warp_id * 16;
      float qk[8][4];
      for (int nt = 0; nt < 8; nt++)
        qk[nt][0] = qk[nt][1] = qk[nt][2] = qk[nt][3] = 0.0f;

      for (int kt = 0; kt < 8; kt++) {
        uint32_t a[4];
        {
          int r = (lane_id % 8) + ((lane_id & 8) ? 8 : 0) + m_base;
          int c = (lane_id >= 16) ? 8 : 0;
          ldmatrix<4>(a, cvt_smem_ptr(&s_buf1[r * kK + swz(r, kt * 16 + c, CGM_128)]));
        }

        for (int nt = 0; nt < 8; nt++) {
          // B = K^T: B[kr, nj] = K[nj, kr], K in s_buf2 XOR swizzled
          uint32_t b[2];
          load_b_transposed(b, s_buf2, kK, CGM_128, kt, nt, lane_id);

          mma_m16n8k16_bf16(
              qk[nt][0], qk[nt][1], qk[nt][2], qk[nt][3],
              a[0], a[1], a[2], a[3], b[0], b[1],
              qk[nt][0], qk[nt][1], qk[nt][2], qk[nt][3]);
        }
      }

      // Apply gating + causal mask → s_acaus fp32
      // A[i,j] = qk[i,j] * exp(g_i - g_j) for j <= i (causal), else 0
      // Direct exp (matches Triton reference exactly)
      {
        int gID = lane_id / 4, tIG = lane_id % 4;
        int r0 = m_base + gID, r1 = r0 + 8;
        float g_r0 = (r0 < clen) ? s_gc[r0] : 0.0f;
        float g_r1 = (r1 < clen) ? s_gc[r1] : 0.0f;

        for (int nt = 0; nt < 8; nt++) {
          int c0 = nt * 8 + tIG * 2;
          int c1 = c0 + 1;
          float g_c0 = (c0 < clen) ? s_gc[c0] : 0.0f;
          float g_c1 = (c1 < clen) ? s_gc[c1] : 0.0f;

          float a00 = (c0 <= r0 && r0 < clen && c0 < clen) ? qk[nt][0] * __expf(g_r0 - g_c0) : 0.0f;
          float a01 = (c1 <= r0 && r0 < clen && c1 < clen) ? qk[nt][1] * __expf(g_r0 - g_c1) : 0.0f;
          float a10 = (c0 <= r1 && r1 < clen && c0 < clen) ? qk[nt][2] * __expf(g_r1 - g_c0) : 0.0f;
          float a11 = (c1 <= r1 && r1 < clen && c1 < clen) ? qk[nt][3] * __expf(g_r1 - g_c1) : 0.0f;

          s_acaus[r0 * kBT + c0] = a00;
          s_acaus[r0 * kBT + c1] = a01;
          s_acaus[r1 * kBT + c0] = a10;
          s_acaus[r1 * kBT + c1] = a11;
        }
      }
    }

    // ═════════════════════════════════════════════════════════════
    // Phase 4: o_intra = A_causal[64,64] @ v_new[64,128]
    //
    // Convert v_new fp32 → bf16 XOR swizzled → s_buf1 (q no longer needed).
    // Convert A_causal fp32 → bf16 XOR swizzled → s_buf2 (k no longer needed for this phase).
    // v_new_bf16 in s_buf1 [64,128] bf16 XOR swizzled (CGM_128).
    // MMA: [64,64] @ [64,128] → [64,128], accumulated onto o_acc.
    // ═════════════════════════════════════════════════════════════
    // Convert A_causal fp32 → bf16 XOR swizzled into s_buf2 (first 8KB)
    {
      __nv_bfloat16 *s_acaus_bf16 = s_buf2;  // reuse K buffer (K not needed until Phase 5)
      for (int i = tid; i < kBT * kBT / 8; i += TB_SIZE) {
        int idx = i * 8;
        int row = idx / kBT, col8 = idx % kBT;
        int sc = swz(row, col8, CGM_64);
        const float *src = &s_acaus[row * kBT + col8];
        int4 dst;
        __nv_bfloat162 *dp = reinterpret_cast<__nv_bfloat162 *>(&dst);
        dp[0] = {__float2bfloat16(src[0]), __float2bfloat16(src[1])};
        dp[1] = {__float2bfloat16(src[2]), __float2bfloat16(src[3])};
        dp[2] = {__float2bfloat16(src[4]), __float2bfloat16(src[5])};
        dp[3] = {__float2bfloat16(src[6]), __float2bfloat16(src[7])};
        *reinterpret_cast<int4 *>(&s_acaus_bf16[row * kBT + sc]) = dst;
      }
    }
    // Convert v_new fp32 → bf16 XOR swizzled
    for (int i = tid; i < kBT * kV / 8; i += TB_SIZE) {
      int idx = i * 8;
      int row = idx / kV, col8 = idx % kV;
      int sc = swz(row, col8, CGM_128);
      const float *src = &s_fp[row * kV + col8];
      int4 dst;
      __nv_bfloat162 *dp = reinterpret_cast<__nv_bfloat162 *>(&dst);
      dp[0] = {__float2bfloat16(src[0]), __float2bfloat16(src[1])};
      dp[1] = {__float2bfloat16(src[2]), __float2bfloat16(src[3])};
      dp[2] = {__float2bfloat16(src[4]), __float2bfloat16(src[5])};
      dp[3] = {__float2bfloat16(src[6]), __float2bfloat16(src[7])};
      *reinterpret_cast<int4 *>(&s_buf1[row * kV + sc]) = dst;
    }
    __syncthreads();

    // MMA: A_causal[64,64] @ v_new[64,128] → accumulate onto o_acc
    {
      const int m_base = warp_id * 16;

      for (int kt = 0; kt < 4; kt++) {
        // A: A_causal[m_base..+16, kt*16..+16] from s_buf2 (bf16 converted) [64,64] CGM_64
        uint32_t a[4];
        {
          __nv_bfloat16 *s_acaus_bf16 = s_buf2;
          int r = (lane_id % 8) + ((lane_id & 8) ? 8 : 0) + m_base;
          int c = (lane_id >= 16) ? 8 : 0;
          ldmatrix<4>(a, cvt_smem_ptr(&s_acaus_bf16[r * kBT + swz(r, kt * 16 + c, CGM_64)]));
        }

        for (int nt = 0; nt < 16; nt++) {
          // B: v_new[kt*16..+16, nt*8..+8] from s_buf1 [64,128] CGM_128
          uint32_t b[2];
          {
            int kr = lane_id % 16;
            ldmatrix_trans<2>(b, cvt_smem_ptr(&s_buf1[(kt * 16 + kr) * kV + swz(kt * 16 + kr, nt * 8, CGM_128)]));
          }
          mma_m16n8k16_bf16(
              o_acc[nt][0], o_acc[nt][1], o_acc[nt][2], o_acc[nt][3],
              a[0], a[1], a[2], a[3], b[0], b[1],
              o_acc[nt][0], o_acc[nt][1], o_acc[nt][2], o_acc[nt][3]);
        }
      }

      // Write output = (o_inter + o_intra) * scale
      {
        int gID = lane_id / 4, tIG = lane_id % 4;
        int r0 = m_base + gID, r1 = r0 + 8;

        for (int nt = 0; nt < 16; nt++) {
          int c0 = nt * 8 + tIG * 2;
          if (r0 < clen) {
            __nv_bfloat162 pair = {__float2bfloat16_rn(scale * o_acc[nt][0]),
                                   __float2bfloat16_rn(scale * o_acc[nt][1])};
            *reinterpret_cast<__nv_bfloat162 *>(
                &output[(cstart + r0) * kHv * kV + hv * kV + c0]) = pair;
          }
          if (r1 < clen) {
            __nv_bfloat162 pair = {__float2bfloat16_rn(scale * o_acc[nt][2]),
                                   __float2bfloat16_rn(scale * o_acc[nt][3])};
            *reinterpret_cast<__nv_bfloat162 *>(
                &output[(cstart + r1) * kHv * kV + hv * kV + c0]) = pair;
          }
        }
      }
    }

    // Reload K into s_buf2 (was overwritten by A_causal bf16 in Phase 4)
    load_bf16_swizzled(s_buf2, kK, CGM_128,
                       k_in + cstart * kHk * kK + k_head * kK, kHk * kK,
                       kBT, kK, clen, tid);

    // ═════════════════════════════════════════════════════════════
    // Phase 5: Update h state
    //   h[v,k] *= exp(g_last)
    //   h[v,k] += sum_t vnew_scaled[t,v] * k[t,k]
    //
    // MMA: vnew_scaled_T[V=128, T=64] @ K[T=64, K=128] → [128, 128]
    // Each warp handles 32 V-rows (2 M-tiles of 16).
    //
    // vnew_scaled_T[v, t] = vnew[t, v] * exp(g_last - g[t])
    // Stored in s_buf1 reinterpreted as [128, 64] bf16 XOR swizzled CGM_64.
    // K stays in s_buf2 [64, 128] bf16 XOR swizzled CGM_128.
    // ═════════════════════════════════════════════════════════════
    __syncthreads();
    {
      float g_last = s_gc[clen - 1];
      float alpha = __expf(g_last);

      // Scale h by alpha
      for (int i = tid; i < kV * kK; i += TB_SIZE)
        s_h[i] *= alpha;

      // Prepare vnew_scaled_T: [V=128, T=64] bf16 XOR swizzled CGM_64
      // s_buf1 has 16KB = 128*64*2 = 16384B — exactly fits.
      // vnew is in s_fp [t * kV + v] fp32.
      // vnew_scaled_T[v, t] = vnew[t, v] * exp(g_last - g[t])
      __nv_bfloat16 *s_vst = s_buf1;  // [128, 64] bf16, stride kBT=64
      for (int i = tid; i < 128 * 64 / 8; i += TB_SIZE) {
        int idx = i * 8;
        int v_row = idx / kBT;
        int t_start = idx % kBT;
        __nv_bfloat16 vals[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
          int t = t_start + j;
          if (t < clen) {
            float vn = s_fp[t * kV + v_row];
            float gate = __expf(g_last - s_gc[t]);
            vals[j] = __float2bfloat16(vn * gate);
          } else {
            vals[j] = __float2bfloat16(0.0f);
          }
        }
        int sc = swz(v_row, t_start, CGM_64);
        *reinterpret_cast<int4 *>(&s_vst[v_row * kBT + sc]) =
            *reinterpret_cast<int4 *>(vals);
      }
      __syncthreads();

      // MMA: vnew_scaled_T[128,64] @ K[64,128] → h_update[128,128]
      // Each warp: 32 V-rows = 2 M-tiles of 16.
      // Per M-tile: 4 K-tiles × 16 N-tiles = 64 MMAs.
      for (int mt = 0; mt < 2; mt++) {
        int m_base = warp_id * 32 + mt * 16;

        float hu[16][4];
        #pragma unroll 1
        for (int nt = 0; nt < 16; nt++)
          hu[nt][0] = hu[nt][1] = hu[nt][2] = hu[nt][3] = 0.0f;

        for (int kt = 0; kt < 4; kt++) {
          // A: vnew_scaled_T[m_base..+16, kt*16..+16] in s_vst [128, 64] CGM_64
          uint32_t a[4];
          {
            int r = (lane_id % 8) + ((lane_id & 8) ? 8 : 0) + m_base;
            int c = (lane_id >= 16) ? 8 : 0;
            ldmatrix<4>(a, cvt_smem_ptr(&s_vst[r * kBT + swz(r, kt * 16 + c, CGM_64)]));
          }

          for (int nt = 0; nt < 16; nt++) {
            // B: K[kt*16..+16, nt*8..+8] from s_buf2 [64, 128] CGM_128
            uint32_t b[2];
            {
              int kr = lane_id % 16;
              ldmatrix_trans<2>(b, cvt_smem_ptr(&s_buf2[(kt * 16 + kr) * kK + swz(kt * 16 + kr, nt * 8, CGM_128)]));
            }
            mma_m16n8k16_bf16(
                hu[nt][0], hu[nt][1], hu[nt][2], hu[nt][3],
                a[0], a[1], a[2], a[3], b[0], b[1],
                hu[nt][0], hu[nt][1], hu[nt][2], hu[nt][3]);
          }
        }

        // Accumulate h_update into s_h[v, k]
        {
          int gID = lane_id / 4, tIG = lane_id % 4;
          int r0 = m_base + gID, r1 = r0 + 8;

          for (int nt = 0; nt < 16; nt++) {
            int c0 = nt * 8 + tIG * 2;
            s_h[r0 * 128 + c0]     += hu[nt][0];
            s_h[r0 * 128 + c0 + 1] += hu[nt][1];
            s_h[r1 * 128 + c0]     += hu[nt][2];
            s_h[r1 * 128 + c0 + 1] += hu[nt][3];
          }
        }
      }
    }
    __syncthreads();
  } // end chunk loop

  // ═══ Store final h state ═══
  {
    float *ns = new_state + ((int64_t)seq_id * kHv + hv) * kV * kK;
    for (int i = tid; i < kV * kK; i += TB_SIZE)
      ns[i] = s_h[i];
  }
}

} // namespace

// ═══════════════════════════════════════════════════════════════════
// Host wrapper — TVM FFI export
// ═══════════════════════════════════════════════════════════════════
void RunFusedHO(
    TensorView q,          // [T, Hq, K] bf16
    TensorView k,          // [T, Hk, K] bf16
    TensorView v,          // [T, Hv, V] bf16 — not used
    TensorView w,          // [T, Hv, K] bf16 (from inverse)
    TensorView u,          // [T, Hv, V] bf16 (from inverse)
    TensorView g_cu,       // [T, Hv] fp32
    ffi::Optional<TensorView> state,
    TensorView cu_seqlens, // [N+1] int64
    double scale,
    TensorView output,     // [T, Hv, V] bf16
    TensorView new_state   // [N, Hv, V, K] fp32
) {
  CHECK_INPUT(q);
  CHECK_INPUT(k);
  CHECK_INPUT(w);
  CHECK_INPUT(u);
  CHECK_INPUT(g_cu);
  CHECK_INPUT(cu_seqlens);
  CHECK_INPUT(output);
  CHECK_INPUT(new_state);

  const int64_t num_seqs = cu_seqlens.size(0) - 1;
  const float scale_f = (scale == 0.0) ? (1.0f / sqrtf(128.0f)) : static_cast<float>(scale);

  const float *state_ptr = nullptr;
  if (state.has_value()) {
    TensorView st = state.value();
    CHECK_INPUT(st);
    state_ptr = static_cast<const float *>(st.data_ptr());
  }

  ffi::CUDADeviceGuard guard(q.device().device_id);
  const cudaStream_t stream = get_cuda_stream(q.device());

  auto *q_p    = static_cast<const __nv_bfloat16 *>(q.data_ptr());
  auto *k_p    = static_cast<const __nv_bfloat16 *>(k.data_ptr());
  auto *w_p    = static_cast<const __nv_bfloat16 *>(w.data_ptr());
  auto *u_p    = static_cast<const __nv_bfloat16 *>(u.data_ptr());
  auto *g_p    = static_cast<const float *>(g_cu.data_ptr());
  auto *cusl_p = static_cast<const int64_t *>(cu_seqlens.data_ptr());
  auto *out_p  = static_cast<__nv_bfloat16 *>(output.data_ptr());
  auto *ns_p   = static_cast<float *>(new_state.data_ptr());

  constexpr int smem_size = (SMEM_TOTAL + 1023) & ~1023;
  static bool s_attrs_set = false;
  if (!s_attrs_set) {
    s_attrs_set = true;
    cudaFuncSetAttribute(FusedHOKernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  }

  dim3 grid(1, num_seqs * kHv);

  FusedHOKernel<<<grid, TB_SIZE, smem_size, stream>>>(
      q_p, k_p, w_p, u_p, g_p, state_ptr, cusl_p,
      scale_f, out_p, ns_p, num_seqs);

  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    TVM_FFI_THROW(RuntimeError)
        << "FusedHO kernel launch failed: " << cudaGetErrorString(err);
  }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fused_ho, RunFusedHO);
