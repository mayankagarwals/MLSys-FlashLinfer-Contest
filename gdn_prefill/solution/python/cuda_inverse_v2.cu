/*
 * cuda_inverse_v2.cu -- SIMPLE scalar FP32 inverse kernel
 *
 * Replaces Triton merge_16x16_to_64x64_inverse_kernel_v2 from chunk_v6c.py.
 * Uses SCALAR FP32 math (no MMA/wmma) for 16x16 block inverse.
 * Correctness first, performance later.
 *
 * Grid: (upper_bound_chunks, H=8), Block: 128 threads (4 warps)
 */

#include "cuda_utils.h"
#include <cuda_bf16.h>
#include <mma.h>
#include <math.h>

namespace {

constexpr int INV2_BT = 64;
constexpr int INV2_BS = 16;
constexpr int INV2_THREADS = 128;

// =====================================================================
// Helper: bf16 roundtrip
// =====================================================================
__device__ __forceinline__
float bf16_rt(float x) {
    return __bfloat162float(__float2bfloat16(x));
}

// Shared memory: ~30 KB total
struct InverseV2Smem {
    float Ai[10][16][16];    // 10 * 1024 = 10240 bytes
    float A_orig[16][16];    // 1024 bytes - save original A for Newton correction
    float tmp1[16][16];      // 1024 bytes - matmul scratch
    float tmp2[16][16];      // 1024 bytes - matmul scratch
    float beta_s[64];        // 256 bytes
    float eg_s[64];          // 256 bytes - precomputed beta*exp(g)
    // For Phase 3 W/U tiled matmul:
    float Ab_tile[16][16];   // 1024 bytes - scaled Ai block
    __nv_bfloat16 kv_tile[16][128]; // 4096 bytes - k or v tile from global
};

// Map (bi, bj) -> linear index in lower triangular storage
// (0,0)->0, (1,0)->1, (1,1)->2, (2,0)->3, (2,1)->4, (2,2)->5,
// (3,0)->6, (3,1)->7, (3,2)->8, (3,3)->9
__device__ __forceinline__
int ai_idx(int bi, int bj) {
    return bi * (bi + 1) / 2 + bj;
}

// =====================================================================
// matmul_16x16: C = A @ B, all in shared memory
// 256 elements, 128 threads -> 2 per thread
// =====================================================================
__device__ void matmul_16x16(
    float dst[16][16],
    const float lhs[16][16],
    const float rhs[16][16],
    int tid
) {
    for (int idx = tid; idx < 256; idx += INV2_THREADS) {
        int r = idx >> 4;
        int c = idx & 15;
        float acc = 0.0f;
        #pragma unroll
        for (int kk = 0; kk < 16; kk++)
            acc = fmaf(lhs[r][kk], rhs[kk][c], acc);
        dst[r][c] = acc;
    }
    __syncthreads();
}

// C += A @ B
__device__ void matmul_16x16_acc(
    float dst[16][16],
    const float lhs[16][16],
    const float rhs[16][16],
    int tid
) {
    for (int idx = tid; idx < 256; idx += INV2_THREADS) {
        int r = idx >> 4;
        int c = idx & 15;
        float acc = dst[r][c];
        #pragma unroll
        for (int kk = 0; kk < 16; kk++)
            acc = fmaf(lhs[r][kk], rhs[kk][c], acc);
        dst[r][c] = acc;
    }
    __syncthreads();
}

// C = -(A @ B)
__device__ void matmul_16x16_neg(
    float dst[16][16],
    const float lhs[16][16],
    const float rhs[16][16],
    int tid
) {
    for (int idx = tid; idx < 256; idx += INV2_THREADS) {
        int r = idx >> 4;
        int c = idx & 15;
        float acc = 0.0f;
        #pragma unroll
        for (int kk = 0; kk < 16; kk++)
            acc = fmaf(lhs[r][kk], rhs[kk][c], acc);
        dst[r][c] = -acc;
    }
    __syncthreads();
}

// Neumann inversion: (I-D)(I+D^2)(I+D^4) + Newton correction (scalar fp32)
// A_orig_save: separate buffer to preserve original A when Ai_out aliases A_in
__device__ void invert_16x16_neumann(
    float Ai_out[16][16],       // output inverse (can alias A_in for diagonal)
    const float A_in[16][16],   // input A block
    float A_orig_save[16][16],  // separate buffer to save original A
    float T1[16][16],           // scratch 1
    float T2[16][16],           // scratch 2
    int tid
) {
    // Save original A_in before we overwrite Ai_out (which may alias A_in)
    for (int idx = tid; idx < 256; idx += INV2_THREADS) {
        int r = idx >> 4;
        int c = idx & 15;
        A_orig_save[r][c] = A_in[r][c];
    }
    __syncthreads();

    // Step 1: D = strictly lower triangular of A (bf16 rounded)
    //         Ai = (I - D) in bf16
    for (int idx = tid; idx < 256; idx += INV2_THREADS) {
        int r = idx >> 4;
        int c = idx & 15;
        float d_val = (r > c) ? bf16_rt(A_orig_save[r][c]) : 0.0f;
        float i_val = (r == c) ? 1.0f : 0.0f;
        Ai_out[r][c] = bf16_rt(i_val - d_val);  // (I - D) bf16
        T1[r][c] = d_val;  // D bf16
    }
    __syncthreads();

    // Step 2: D^2 = D @ D -> T2 (fp32 accum, then bf16 roundtrip)
    for (int idx = tid; idx < 256; idx += INV2_THREADS) {
        int r = idx >> 4;
        int c = idx & 15;
        float acc = 0.0f;
        #pragma unroll
        for (int kk = 0; kk < 16; kk++)
            acc = fmaf(T1[r][kk], T1[kk][c], acc);
        T2[r][c] = acc;  // D^2 fp32
    }
    __syncthreads();

    // Step 3: Ai = Ai @ (I + bf16(D^2))
    // Store bf16(D^2) in T1 for reuse, form (I + bf16(D^2)) in T1
    for (int idx = tid; idx < 256; idx += INV2_THREADS) {
        int r = idx >> 4;
        int c = idx & 15;
        float d2_bf16 = bf16_rt(T2[r][c]);
        T2[r][c] = d2_bf16;  // save D^2_bf16 for step 4
        float i_plus_d2 = ((r == c) ? 1.0f : 0.0f) + d2_bf16;
        T1[r][c] = bf16_rt(i_plus_d2);  // (I + D^2)_bf16
    }
    __syncthreads();

    // Ai_new = Ai @ (I + D^2_bf16) -> use A_orig_save as scratch for result
    for (int idx = tid; idx < 256; idx += INV2_THREADS) {
        int r = idx >> 4;
        int c = idx & 15;
        float acc = 0.0f;
        #pragma unroll
        for (int kk = 0; kk < 16; kk++)
            acc = fmaf(Ai_out[r][kk], T1[kk][c], acc);
        // T1 is free (done with I+D^2), use as output to avoid read-write alias
        T1[r][c] = acc;
    }
    __syncthreads();
    // Copy T1 -> Ai_out
    for (int idx = tid; idx < 256; idx += INV2_THREADS) {
        int r = idx >> 4;
        int c = idx & 15;
        Ai_out[r][c] = T1[r][c];
    }
    __syncthreads();

    // Step 4: D^4 = D^2_bf16 @ D^2_bf16 (T2 has D^2_bf16) -> T1
    for (int idx = tid; idx < 256; idx += INV2_THREADS) {
        int r = idx >> 4;
        int c = idx & 15;
        float acc = 0.0f;
        #pragma unroll
        for (int kk = 0; kk < 16; kk++)
            acc = fmaf(T2[r][kk], T2[kk][c], acc);
        T1[r][c] = bf16_rt(acc);  // D^4 bf16
    }
    __syncthreads();

    // Step 5: Ai = bf16(Ai) @ (I + D^4_bf16)
    // Form (I + D^4_bf16) in T1 in-place, bf16 roundtrip Ai_out
    for (int idx = tid; idx < 256; idx += INV2_THREADS) {
        int r = idx >> 4;
        int c = idx & 15;
        T1[r][c] = bf16_rt(((r == c) ? 1.0f : 0.0f) + T1[r][c]);
        Ai_out[r][c] = bf16_rt(Ai_out[r][c]);
    }
    __syncthreads();

    // T2 = Ai_bf16 @ (I + D^4_bf16)
    for (int idx = tid; idx < 256; idx += INV2_THREADS) {
        int r = idx >> 4;
        int c = idx & 15;
        float acc = 0.0f;
        #pragma unroll
        for (int kk = 0; kk < 16; kk++)
            acc = fmaf(Ai_out[r][kk], T1[kk][c], acc);
        T2[r][c] = acc;  // Ai after full Neumann
    }
    __syncthreads();

    // Step 6: Newton correction: MAi = Ai + A_orig @ Ai, then Ai = Ai @ (2I - MAi)
    // T1 = MAi = T2 + A_orig_save @ T2
    for (int idx = tid; idx < 256; idx += INV2_THREADS) {
        int r = idx >> 4;
        int c = idx & 15;
        float acc = T2[r][c];  // identity part
        #pragma unroll
        for (int kk = 0; kk < 16; kk++)
            acc = fmaf(A_orig_save[r][kk], T2[kk][c], acc);
        T1[r][c] = acc;  // MAi
    }
    __syncthreads();

    // Ai_out = T2 @ (2I - T1)
    for (int idx = tid; idx < 256; idx += INV2_THREADS) {
        int r = idx >> 4;
        int c = idx & 15;
        float acc = 0.0f;
        #pragma unroll
        for (int kk = 0; kk < 16; kk++) {
            float correction = ((kk == c) ? 2.0f : 0.0f) - T1[kk][c];
            acc = fmaf(T2[r][kk], correction, acc);
        }
        Ai_out[r][c] = acc;
    }
    __syncthreads();
}

// =====================================================================
// Main kernel
// =====================================================================
__global__ void __launch_bounds__(INV2_THREADS)
inverse_kernel_v2(
    const float * __restrict__ A_ptr,
    const __nv_bfloat16 * __restrict__ k_ptr,
    const __nv_bfloat16 * __restrict__ v_ptr,
    __nv_bfloat16 * __restrict__ w_ptr,
    __nv_bfloat16 * __restrict__ u_ptr,
    const float * __restrict__ beta_ptr,
    const float * __restrict__ g_cu_ptr,
    const int64_t * __restrict__ cu_seqlens_ptr,
    const int32_t * __restrict__ chunk_indices_ptr,
    const int32_t * __restrict__ total_chunks_ptr,
    int H, int Hg, int K_dim, int V_dim, int BT
) {
    extern __shared__ char smem_raw[];
    auto &smem = *reinterpret_cast<InverseV2Smem *>(smem_raw);

    int global_chunk_id = blockIdx.x;
    int head_id = blockIdx.y;
    int tid = threadIdx.x;

    if (global_chunk_id >= *total_chunks_ptr)
        return;

    int seq_id = chunk_indices_ptr[global_chunk_id * 2];
    int chunk_id = chunk_indices_ptr[global_chunk_id * 2 + 1];
    int bos = static_cast<int>(cu_seqlens_ptr[seq_id]);
    int eos = static_cast<int>(cu_seqlens_ptr[seq_id + 1]);
    int seqlen = eos - bos;
    int k_head = head_id / (H / Hg);

    // =============================================================
    // Phase 1: Load 4 diagonal A blocks and invert each
    // =============================================================
    // A is [total_T, H, BT] fp32
    // A[(bos + chunk_id*BT + row), head_id, col] at linear offset:
    //   (bos + chunk_id*BT + row) * H * BT + head_id * BT + col

    for (int blk = 0; blk < 4; blk++) {
        int diag_idx = ai_idx(blk, blk);
        int row_offset = blk * 16;
        for (int idx = tid; idx < 256; idx += INV2_THREADS) {
            int r = idx >> 4;
            int c = idx & 15;
            int global_row = chunk_id * BT + row_offset + r;
            float val = 0.0f;
            if (global_row < seqlen) {
                val = A_ptr[(bos + global_row) * H * BT + head_id * BT + (row_offset + c)];
            }
            smem.Ai[diag_idx][r][c] = val;
        }
    }
    __syncthreads();

    // Invert each diagonal block
    for (int blk = 0; blk < 4; blk++) {
        int diag_idx = ai_idx(blk, blk);
        invert_16x16_neumann(
            smem.Ai[diag_idx], smem.Ai[diag_idx],
            smem.A_orig, smem.tmp1, smem.tmp2, tid
        );
    }

    // bf16 roundtrip diagonal blocks
    for (int blk = 0; blk < 4; blk++) {
        int diag_idx = ai_idx(blk, blk);
        for (int idx = tid; idx < 256; idx += INV2_THREADS) {
            int r = idx >> 4;
            int c = idx & 15;
            smem.Ai[diag_idx][r][c] = bf16_rt(smem.Ai[diag_idx][r][c]);
        }
    }
    __syncthreads();

    // =============================================================
    // Phase 2: Off-diagonal Ai blocks via block back-substitution
    // =============================================================
    // Helper: load A block (bi,bj) from global into tmp1
    auto load_A_offdiag = [&](int bi, int bj) {
        int row_off = bi * 16;
        int col_off = bj * 16;
        for (int idx = tid; idx < 256; idx += INV2_THREADS) {
            int r = idx >> 4;
            int c = idx & 15;
            int global_row = chunk_id * BT + row_off + r;
            float val = 0.0f;
            if (global_row < seqlen) {
                val = A_ptr[(bos + global_row) * H * BT + head_id * BT + (col_off + c)];
            }
            smem.tmp1[r][c] = val;
        }
        __syncthreads();
    };

    // Level 0: Ai_21 = -(Ai_22 @ A_21) @ Ai_11
    load_A_offdiag(1, 0);
    matmul_16x16(smem.tmp2, smem.Ai[ai_idx(1,1)], smem.tmp1, tid);
    matmul_16x16_neg(smem.Ai[ai_idx(1,0)], smem.tmp2, smem.Ai[ai_idx(0,0)], tid);

    // Ai_32 = -(Ai_33 @ A_32) @ Ai_22
    load_A_offdiag(2, 1);
    matmul_16x16(smem.tmp2, smem.Ai[ai_idx(2,2)], smem.tmp1, tid);
    matmul_16x16_neg(smem.Ai[ai_idx(2,1)], smem.tmp2, smem.Ai[ai_idx(1,1)], tid);

    // Ai_43 = -(Ai_44 @ A_43) @ Ai_33
    load_A_offdiag(3, 2);
    matmul_16x16(smem.tmp2, smem.Ai[ai_idx(3,3)], smem.tmp1, tid);
    matmul_16x16_neg(smem.Ai[ai_idx(3,2)], smem.tmp2, smem.Ai[ai_idx(2,2)], tid);

    // Level 1: Ai_31 = -Ai_33 @ (A_31 @ Ai_11 + A_32 @ Ai_21)
    load_A_offdiag(2, 0);
    matmul_16x16(smem.tmp2, smem.tmp1, smem.Ai[ai_idx(0,0)], tid);
    load_A_offdiag(2, 1);
    matmul_16x16_acc(smem.tmp2, smem.tmp1, smem.Ai[ai_idx(1,0)], tid);
    matmul_16x16_neg(smem.Ai[ai_idx(2,0)], smem.Ai[ai_idx(2,2)], smem.tmp2, tid);

    // Ai_42 = -Ai_44 @ (A_42 @ Ai_22 + A_43 @ Ai_32)
    load_A_offdiag(3, 1);
    matmul_16x16(smem.tmp2, smem.tmp1, smem.Ai[ai_idx(1,1)], tid);
    load_A_offdiag(3, 2);
    matmul_16x16_acc(smem.tmp2, smem.tmp1, smem.Ai[ai_idx(2,1)], tid);
    matmul_16x16_neg(smem.Ai[ai_idx(3,1)], smem.Ai[ai_idx(3,3)], smem.tmp2, tid);

    // Level 2: Ai_41 = -Ai_44 @ (A_41@Ai_11 + A_42@Ai_21 + A_43@Ai_31)
    load_A_offdiag(3, 0);
    matmul_16x16(smem.tmp2, smem.tmp1, smem.Ai[ai_idx(0,0)], tid);
    load_A_offdiag(3, 1);
    matmul_16x16_acc(smem.tmp2, smem.tmp1, smem.Ai[ai_idx(1,0)], tid);
    load_A_offdiag(3, 2);
    matmul_16x16_acc(smem.tmp2, smem.tmp1, smem.Ai[ai_idx(2,0)], tid);
    matmul_16x16_neg(smem.Ai[ai_idx(3,0)], smem.Ai[ai_idx(3,3)], smem.tmp2, tid);

    // bf16 roundtrip all off-diagonal blocks
    for (int bi = 1; bi < 4; bi++) {
        for (int bj = 0; bj < bi; bj++) {
            int ab_idx = ai_idx(bi, bj);
            for (int idx = tid; idx < 256; idx += INV2_THREADS) {
                int r = idx >> 4;
                int c = idx & 15;
                smem.Ai[ab_idx][r][c] = bf16_rt(smem.Ai[ab_idx][r][c]);
            }
        }
    }
    __syncthreads();

    // =============================================================
    // Phase 3: Compute W and U using tiled shared memory matmul
    //
    // For each output row-group i (0..3), contributing blocks j=0..i:
    //   Ab_ij = Ai[i][j] * diag(scale_j)  where scale=beta*exp(g) for W, scale=beta for U
    //   W_i += Ab_ij @ k_j (tiled over K_dim in 128-wide chunks)
    //   U_i += Ab_ij @ v_j (tiled over V_dim in 128-wide chunks)
    //
    // We load k_j/v_j tiles cooperatively into smem, then each thread
    // processes its output rows using the smem data.
    // =============================================================
    for (int idx = tid; idx < 64; idx += INV2_THREADS) {
        int t = chunk_id * BT + idx;
        if (t < seqlen) {
            float b = beta_ptr[(bos + t) * H + head_id];
            float g = g_cu_ptr[(bos + t) * H + head_id];
            smem.beta_s[idx] = b;
            smem.eg_s[idx] = b * expf(g);
        } else {
            smem.beta_s[idx] = 0.0f;
            smem.eg_s[idx] = 0.0f;
        }
    }
    __syncthreads();

    // Process W and U using wmma bf16 m16n16k16
    // Each warp handles 2 column tiles of 16 cols (= 32 cols) out of 128
    using namespace nvcuda::wmma;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    for (int bi = 0; bi < 4; bi++) {
        int out_row_base = bi * 16;

        fragment<accumulator, 16, 16, 16, float> w_acc[2], u_acc[2];
        fill_fragment(w_acc[0], 0.0f); fill_fragment(w_acc[1], 0.0f);
        fill_fragment(u_acc[0], 0.0f); fill_fragment(u_acc[1], 0.0f);

        for (int bj = 0; bj <= bi; bj++) {
            int ab = ai_idx(bi, bj);

            // Compute Ab_w = Ai[ab] * diag(eg_s[bj]) -> bf16 smem
            __nv_bfloat16 *Ab_bf16 = reinterpret_cast<__nv_bfloat16 *>(smem.Ab_tile);
            for (int idx = tid; idx < 256; idx += INV2_THREADS) {
                int r = idx >> 4;
                int c = idx & 15;
                Ab_bf16[r * 16 + c] = __float2bfloat16(smem.Ai[ab][r][c] * smem.eg_s[bj * 16 + c]);
            }
            // Load k_j tile [16, 128] bf16
            for (int idx = tid; idx < 16 * 128; idx += INV2_THREADS) {
                int r = idx / 128, c = idx % 128;
                int t_local = bj * 16 + r;
                int t_global = bos + chunk_id * BT + t_local;
                smem.kv_tile[r][c] = (chunk_id * BT + t_local < seqlen) ?
                    k_ptr[t_global * Hg * K_dim + k_head * K_dim + c] : __float2bfloat16(0.0f);
            }
            __syncthreads();

            // wmma W: Ab[16,16] @ k_tile[16,16] for 2 column tiles per warp
            {
                fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> a_frag;
                load_matrix_sync(a_frag, Ab_bf16, 16);
                for (int ct = 0; ct < 2; ct++) {
                    int col_off = warp_id * 32 + ct * 16;
                    fragment<matrix_b, 16, 16, 16, __nv_bfloat16, row_major> b_frag;
                    load_matrix_sync(b_frag, &smem.kv_tile[0][col_off], 128);
                    mma_sync(w_acc[ct], a_frag, b_frag, w_acc[ct]);
                }
            }

            // Compute Ab_u = Ai[ab] * diag(beta_s[bj]) -> bf16 smem
            for (int idx = tid; idx < 256; idx += INV2_THREADS) {
                int r = idx >> 4;
                int c = idx & 15;
                Ab_bf16[r * 16 + c] = __float2bfloat16(smem.Ai[ab][r][c] * smem.beta_s[bj * 16 + c]);
            }
            // Load v_j tile
            for (int idx = tid; idx < 16 * 128; idx += INV2_THREADS) {
                int r = idx / 128, c = idx % 128;
                int t_local = bj * 16 + r;
                int t_global = bos + chunk_id * BT + t_local;
                smem.kv_tile[r][c] = (chunk_id * BT + t_local < seqlen) ?
                    v_ptr[t_global * H * V_dim + head_id * V_dim + c] : __float2bfloat16(0.0f);
            }
            __syncthreads();

            // wmma U: Ab[16,16] @ v_tile[16,16] for 2 column tiles per warp
            {
                fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> a_frag;
                load_matrix_sync(a_frag, Ab_bf16, 16);
                for (int ct = 0; ct < 2; ct++) {
                    int col_off = warp_id * 32 + ct * 16;
                    fragment<matrix_b, 16, 16, 16, __nv_bfloat16, row_major> b_frag;
                    load_matrix_sync(b_frag, &smem.kv_tile[0][col_off], 128);
                    mma_sync(u_acc[ct], a_frag, b_frag, u_acc[ct]);
                }
            }
            __syncthreads();
        }

        // Store W and U wmma results to global memory via smem
        for (int ct = 0; ct < 2; ct++) {
            int col_off = warp_id * 32 + ct * 16;
            // Use tmp1/tmp2 as per-warp temp — serialize across warps
            for (int w_id = 0; w_id < 4; w_id++) {
                if (warp_id == w_id) {
                    store_matrix_sync(&smem.tmp1[0][0], w_acc[ct], 16, mem_row_major);
                    store_matrix_sync(&smem.tmp2[0][0], u_acc[ct], 16, mem_row_major);
                }
                __syncthreads();
                if (warp_id == w_id) {
                    for (int idx = lane_id; idx < 256; idx += 32) {
                        int r = idx >> 4, c = idx & 15;
                        int t_out = chunk_id * BT + out_row_base + r;
                        if (t_out < seqlen) {
                            int tg = bos + t_out;
                            w_ptr[tg * H * K_dim + head_id * K_dim + col_off + c] =
                                __float2bfloat16(smem.tmp1[r][c]);
                            u_ptr[tg * H * V_dim + head_id * V_dim + col_off + c] =
                                __float2bfloat16(smem.tmp2[r][c]);
                        }
                    }
                }
                __syncthreads();
            }
        }
    }
}

} // anonymous namespace

// =====================================================================
// Host-side launch function
// =====================================================================
void inverse_v2(
    TensorView A,              // [total_T, H, BT] fp32
    TensorView k,              // [total_T, Hg, K_dim] bf16
    TensorView v,              // [total_T, H, V_dim] bf16
    TensorView w,              // [total_T, H, K_dim] bf16 output
    TensorView u,              // [total_T, H, V_dim] bf16 output
    TensorView beta,           // [total_T, H] fp32
    TensorView g_cu,           // [total_T, H] fp32
    TensorView cu_seqlens,     // [N+1] int64
    TensorView chunk_indices,  // [total_chunks, 2] int32
    TensorView total_chunks,   // [1] int32
    int upper_bound_chunks
) {
    const int H_val = static_cast<int>(A.size(1));
    const int BT_val = static_cast<int>(A.size(2));
    const int Hg_val = static_cast<int>(k.size(1));
    const int K_dim_val = static_cast<int>(k.size(2));
    const int V_dim_val = static_cast<int>(v.size(2));

    auto *A_p = reinterpret_cast<const float *>(A.data_ptr());
    auto *k_p = reinterpret_cast<const __nv_bfloat16 *>(k.data_ptr());
    auto *v_p = reinterpret_cast<const __nv_bfloat16 *>(v.data_ptr());
    auto *w_p = reinterpret_cast<__nv_bfloat16 *>(w.data_ptr());
    auto *u_p = reinterpret_cast<__nv_bfloat16 *>(u.data_ptr());
    auto *beta_p = reinterpret_cast<const float *>(beta.data_ptr());
    auto *g_cu_p = reinterpret_cast<const float *>(g_cu.data_ptr());
    auto *cu_seqlens_p = reinterpret_cast<const int64_t *>(cu_seqlens.data_ptr());
    auto *chunk_indices_p = reinterpret_cast<const int32_t *>(chunk_indices.data_ptr());
    auto *total_chunks_p = reinterpret_cast<const int32_t *>(total_chunks.data_ptr());

    int smem_size = sizeof(InverseV2Smem);

    dim3 grid(upper_bound_chunks, H_val);
    dim3 block(INV2_THREADS);

    cudaStream_t stream = get_cuda_stream(A.device());

    cudaFuncSetAttribute(inverse_kernel_v2,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    inverse_kernel_v2<<<grid, block, smem_size, stream>>>(
        A_p, k_p, v_p, w_p, u_p,
        beta_p, g_cu_p,
        cu_seqlens_p, chunk_indices_p, total_chunks_p,
        H_val, Hg_val, K_dim_val, V_dim_val, BT_val
    );
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(inverse_v2, inverse_v2);
