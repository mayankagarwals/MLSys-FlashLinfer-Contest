// test_ldmatrix.cu — Verify ldmatrix_trans behavior on SM100a
// Tests: load A[16,16] via ldmatrix<4>, B^T[8,16] via ldmatrix_trans<2>,
// do one MMA, and write D registers to global with standard mapping.
// Compare against torch.mm() reference.

#include <cuda_bf16.h>
#include <cstdint>
#include "cuda_utils.h"

// Single 16x8 MMA test: A[16,16] @ B[16,8] = C[16,8]
// A stored row-major [16, stride_A] bf16 in smem
// B^T stored row-major [8, 16] bf16 in smem (= B col-major [16, 8])
__global__ void test_mma_1tile(
    const nv_bfloat16 *__restrict__ A_ptr,   // [16, 16] row-major
    const nv_bfloat16 *__restrict__ BT_ptr,  // [8, 16] row-major = B^T
    float *__restrict__ C_ptr                 // [16, 8] row-major output
) {
    __shared__ nv_bfloat16 s_A[16 * 16];
    __shared__ nv_bfloat16 s_BT[8 * 16];
    uint32_t sa = __cvta_generic_to_shared(s_A);
    uint32_t sb = __cvta_generic_to_shared(s_BT);

    int tid = threadIdx.x;  // 0-31

    // Load to smem
    for (int i = tid; i < 16 * 16; i += 32) s_A[i] = A_ptr[i];
    for (int i = tid; i < 8 * 16; i += 32) s_BT[i] = BT_ptr[i];
    __syncwarp();

    // A via ldmatrix<4> from s_A [16, 16] row-major (stride = 16 bf16 = 32 bytes)
    uint32_t a[4];
    {
        int lr = (tid % 8) + ((tid & 8) ? 8 : 0);
        int lc = (tid >= 16) ? 8 : 0;
        ldmatrix<4>(a, sa + lr * 32 + lc * 2);
    }

    // B via ldmatrix_trans<2> from s_BT [8, 16] row-major
    // Using the "same rows, different cols" pattern (our O-kernel approach)
    uint32_t b[2];
    {
        int kr = tid % 16;
        int row = kr % 8;
        int col = (kr >= 8) ? 8 : 0;
        ldmatrix_trans<2>(b, sb + row * 32 + col * 2);
    }

    // MMA
    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    mma_m16n8k16_bf16(d0, d1, d2, d3, a[0], a[1], a[2], a[3], b[0], b[1], 0.f, 0.f, 0.f, 0.f);

    // Store with standard D mapping
    int gid = tid / 4, thr = tid % 4;
    C_ptr[gid * 8 + thr * 2]       = d0;
    C_ptr[gid * 8 + thr * 2 + 1]   = d1;
    C_ptr[(gid + 8) * 8 + thr * 2]     = d2;
    C_ptr[(gid + 8) * 8 + thr * 2 + 1] = d3;
}

// Bigger test: A[64, 128] @ B^T_transposed → one N-tile result
// A stored row-major [64, 128], B^T stored row-major [64, 128]
// Computes one (M=16, N=8, K_total=128) output tile
__global__ void test_mma_full_k(
    const nv_bfloat16 *__restrict__ A_ptr,   // [64, 128] row-major
    const nv_bfloat16 *__restrict__ BT_ptr,  // [64, 128] row-major = B^T [64, 128]
    float *__restrict__ C_ptr,               // [16, 8] output
    int M_start, int N_start                 // which M-tile (0-3), N-tile (0-7)
) {
    __shared__ nv_bfloat16 s_A[64 * 128];
    __shared__ nv_bfloat16 s_BT[64 * 128];
    uint32_t sa = __cvta_generic_to_shared(s_A);
    uint32_t sb = __cvta_generic_to_shared(s_BT);
    int tid = threadIdx.x;

    // Load to smem
    for (int i = tid; i < 64 * 128; i += 32) s_A[i] = A_ptr[i];
    for (int i = tid; i < 64 * 128; i += 32) s_BT[i] = BT_ptr[i];
    __syncwarp();

    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;

    for (int kt = 0; kt < 128 / 16; kt++) {
        int kc = kt * 16;

        uint32_t a[4];
        {
            int lr = (tid % 8) + ((tid & 8) ? 8 : 0);
            int lc = (tid >= 16) ? 8 : 0;
            ldmatrix<4>(a, sa + (M_start * 16 + lr) * 128 * 2 + (kc + lc) * 2);
        }

        uint32_t b[2];
        {
            int kr = tid % 16;
            int row = N_start * 8 + (kr % 8);
            int col = kc + (kr >= 8 ? 8 : 0);
            ldmatrix_trans<2>(b, sb + row * 128 * 2 + col * 2);
        }

        mma_m16n8k16_bf16(d0, d1, d2, d3, a[0], a[1], a[2], a[3], b[0], b[1], d0, d1, d2, d3);
    }

    int gid = tid / 4, thr = tid % 4;
    C_ptr[gid * 8 + thr * 2]           = d0;
    C_ptr[gid * 8 + thr * 2 + 1]       = d1;
    C_ptr[(gid + 8) * 8 + thr * 2]     = d2;
    C_ptr[(gid + 8) * 8 + thr * 2 + 1] = d3;
}

// Test with MANUAL B packing (no ldmatrix_trans)
__global__ void test_mma_manual_b(
    const nv_bfloat16 *__restrict__ A_ptr,   // [16, 16]
    const nv_bfloat16 *__restrict__ BT_ptr,  // [8, 16] = B^T
    float *__restrict__ C_ptr                 // [16, 8]
) {
    __shared__ nv_bfloat16 s_A[16 * 16];
    __shared__ nv_bfloat16 s_BT[8 * 16];

    int tid = threadIdx.x;
    int gid = tid / 4, thr = tid % 4;

    for (int i = tid; i < 256; i += 32) s_A[i] = A_ptr[i];
    for (int i = tid; i < 128; i += 32) s_BT[i] = BT_ptr[i];
    __syncwarp();

    // A via ldmatrix
    uint32_t sa = __cvta_generic_to_shared(s_A);
    uint32_t a[4];
    {
        int lr = (tid % 8) + ((tid & 8) ? 8 : 0);
        int lc = (tid >= 16) ? 8 : 0;
        ldmatrix<4>(a, sa + lr * 32 + lc * 2);
    }

    // B via MANUAL packing from BT [8, 16] row-major
    // B[k, n] = BT[n, k]. b0 = {B[thr*2, gid], B[thr*2+1, gid]} = {BT[gid, thr*2], BT[gid, thr*2+1]}
    uint32_t b0 = *(const uint32_t*)(s_BT + gid * 16 + thr * 2);
    uint32_t b1 = *(const uint32_t*)(s_BT + gid * 16 + thr * 2 + 8);

    float d0=0, d1=0, d2=0, d3=0;
    mma_m16n8k16_bf16(d0, d1, d2, d3, a[0], a[1], a[2], a[3], b0, b1, 0.f, 0.f, 0.f, 0.f);

    // Standard D mapping
    C_ptr[gid * 8 + thr * 2]           = d0;
    C_ptr[gid * 8 + thr * 2 + 1]       = d1;
    C_ptr[(gid + 8) * 8 + thr * 2]     = d2;
    C_ptr[(gid + 8) * 8 + thr * 2 + 1] = d3;
}

// Test with SM100a D mapping (from tf32 findings)
__global__ void test_mma_sm100a_d(
    const nv_bfloat16 *__restrict__ A_ptr,
    const nv_bfloat16 *__restrict__ BT_ptr,
    float *__restrict__ C_ptr
) {
    __shared__ nv_bfloat16 s_A[16 * 16];
    __shared__ nv_bfloat16 s_BT[8 * 16];
    int tid = threadIdx.x;
    int gid = tid / 4, thr = tid % 4;

    for (int i = tid; i < 256; i += 32) s_A[i] = A_ptr[i];
    for (int i = tid; i < 128; i += 32) s_BT[i] = BT_ptr[i];
    __syncwarp();

    uint32_t sa = __cvta_generic_to_shared(s_A);
    uint32_t a[4];
    {
        int lr = (tid % 8) + ((tid & 8) ? 8 : 0);
        int lc = (tid >= 16) ? 8 : 0;
        ldmatrix<4>(a, sa + lr * 32 + lc * 2);
    }

    uint32_t b0 = *(const uint32_t*)(s_BT + gid * 16 + thr * 2);
    uint32_t b1 = *(const uint32_t*)(s_BT + gid * 16 + thr * 2 + 8);

    float d0=0, d1=0, d2=0, d3=0;
    mma_m16n8k16_bf16(d0, d1, d2, d3, a[0], a[1], a[2], a[3], b0, b1, 0.f, 0.f, 0.f, 0.f);

    // SM100a D mapping: d0=D[gid+(thr/2)*8, (thr%2)*4], d1=[.., (thr%2)*4+2], d2=[.., +1], d3=[.., +3]
    int d_row = gid + (thr / 2) * 8;
    int d_col = (thr % 2) * 4;
    C_ptr[d_row * 8 + d_col]     = d0;
    C_ptr[d_row * 8 + d_col + 2] = d1;
    C_ptr[d_row * 8 + d_col + 1] = d2;
    C_ptr[d_row * 8 + d_col + 3] = d3;
}

void test_mma_manual_b_ffi(TensorView A, TensorView BT, TensorView C) {
    test_mma_manual_b<<<1, 32>>>(
        (const nv_bfloat16*)A.data_ptr(),
        (const nv_bfloat16*)BT.data_ptr(),
        (float*)C.data_ptr());
}

void test_mma_sm100a_d_ffi(TensorView A, TensorView BT, TensorView C) {
    test_mma_sm100a_d<<<1, 32>>>(
        (const nv_bfloat16*)A.data_ptr(),
        (const nv_bfloat16*)BT.data_ptr(),
        (float*)C.data_ptr());
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(test_mma_manual_b, test_mma_manual_b_ffi);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(test_mma_sm100a_d, test_mma_sm100a_d_ffi);

void test_mma_1tile_ffi(TensorView A, TensorView BT, TensorView C) {
    test_mma_1tile<<<1, 32>>>(
        (const nv_bfloat16*)A.data_ptr(),
        (const nv_bfloat16*)BT.data_ptr(),
        (float*)C.data_ptr());
}

void test_mma_full_k_ffi(TensorView A, TensorView BT, TensorView C, int m_tile, int n_tile) {
    int smem = (64 * 128 * 2) * 2;
    auto kern = test_mma_full_k;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    kern<<<1, 32, smem>>>(
        (const nv_bfloat16*)A.data_ptr(),
        (const nv_bfloat16*)BT.data_ptr(),
        (float*)C.data_ptr(),
        m_tile, n_tile);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(test_mma_1tile, test_mma_1tile_ffi);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(test_mma_full_k, test_mma_full_k_ffi);
