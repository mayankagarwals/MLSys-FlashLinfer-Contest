/*
 * CUDA implementation of merge_16x16_to_64x64_inverse_kernel_v2 from chunk_v6c.py
 *
 * Replaces the Triton kernel with mma.sync instructions to produce BIT-IDENTICAL
 * results on sm_100a (B200).
 *
 * Algorithm:
 * 1. Load 4 diagonal 16x16 blocks from A matrix
 * 2. Invert each using Neumann series + Newton correction
 * 3. Compute off-diagonal blocks via Schur complement
 * 4. bf16 roundtrip all blocks
 * 5. Compute W = Ai * beta * exp(g) @ k and U = Ai * beta @ v
 * 6. Store W, U to global memory
 *
 * Grid: (upper_bound_chunks, H), blockDim: 128 (4 warps)
 */

#include "cuda_utils.h"
#include <cuda_bf16.h>
#include <math.h>

// ═══════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════

static constexpr int WARP_SIZE = 32;
static constexpr int NUM_WARPS_INV = 4;
static constexpr int BLOCK_SIZE_INV = NUM_WARPS_INV * WARP_SIZE; // 128 threads

// ═══════════════════════════════════════════════════════════════════
// Shared memory layout for 16x16 matrices
//
// For mma.sync m16n8k16 bf16:
//   A operand (row-major in smem): 16 rows x 16 cols of bf16 = 512 bytes
//     ldmatrix x4 loads 4 m8n8 tiles
//   B operand (col-major in smem): 16 rows x 16 cols of bf16 = 512 bytes
//     ldmatrix_trans x2 loads 2 m8n8 tiles (for 8 columns at a time)
//
// For tf32 mma.sync m16n8k8:
//   A operand: 16x8 fp32 = 512 bytes
//   B operand: 8x8 fp32 = 256 bytes (but we need 16 cols, so 8x16 = 512 bytes)
// ═══════════════════════════════════════════════════════════════════

// bf16 roundtrip: convert fp32 -> bf16 -> fp32
__device__ __forceinline__
float bf16_roundtrip(float x) {
    return __bfloat162float(__float2bfloat16(x));
}

// ═══════════════════════════════════════════════════════════════════
// 16x16 register mapping for mma.sync m16n8k16 output:
//   Thread t: groupId = t >> 2, within = t & 3
//   d0 = C[groupId, within*2]       (cols 0..7 via first MMA call)
//   d1 = C[groupId, within*2+1]
//   d2 = C[groupId+8, within*2]
//   d3 = C[groupId+8, within*2+1]
//   For 16x16: need TWO m16n8k16 calls (cols 0..7, cols 8..15)
//   Total: 8 floats per thread = {c0_0..c0_3, c1_0..c1_3}
// ═══════════════════════════════════════════════════════════════════

// Store a 16x16 fp32 matrix (distributed across warp) into smem as bf16 (row-major)
// Each thread knows its owned elements from the register mapping
__device__ __forceinline__
void store_fp32_to_bf16_smem(
    __nv_bfloat16 *smem_bf16, // pointer to 16x16 bf16 smem region
    // 8 floats: c0_0..c0_3 for cols 0..7, c1_0..c1_3 for cols 8..15
    float c0_0, float c0_1, float c0_2, float c0_3,
    float c1_0, float c1_1, float c1_2, float c1_3,
    int lane_id
) {
    // Thread t owns: groupId = lane_id >> 2, within = lane_id & 3
    int groupId = lane_id >> 2;
    int within = lane_id & 3;

    // From mma result (m16n8):
    // c0_0 -> C[groupId, within*2]
    // c0_1 -> C[groupId, within*2+1]
    // c0_2 -> C[groupId+8, within*2]
    // c0_3 -> C[groupId+8, within*2+1]
    // c1_0 -> C[groupId, within*2+8]
    // c1_1 -> C[groupId, within*2+9]
    // c1_2 -> C[groupId+8, within*2+8]
    // c1_3 -> C[groupId+8, within*2+9]

    smem_bf16[groupId * 16 + within * 2]       = __float2bfloat16(c0_0);
    smem_bf16[groupId * 16 + within * 2 + 1]   = __float2bfloat16(c0_1);
    smem_bf16[(groupId+8) * 16 + within * 2]     = __float2bfloat16(c0_2);
    smem_bf16[(groupId+8) * 16 + within * 2 + 1] = __float2bfloat16(c0_3);

    smem_bf16[groupId * 16 + within * 2 + 8]       = __float2bfloat16(c1_0);
    smem_bf16[groupId * 16 + within * 2 + 9]       = __float2bfloat16(c1_1);
    smem_bf16[(groupId+8) * 16 + within * 2 + 8]     = __float2bfloat16(c1_2);
    smem_bf16[(groupId+8) * 16 + within * 2 + 9]     = __float2bfloat16(c1_3);
}

// Store fp32 registers into smem as fp32 (for tf32 MMA operand loading)
__device__ __forceinline__
void store_fp32_to_fp32_smem(
    float *smem_fp32, // pointer to 16x16 fp32 smem region
    float c0_0, float c0_1, float c0_2, float c0_3,
    float c1_0, float c1_1, float c1_2, float c1_3,
    int lane_id
) {
    int groupId = lane_id >> 2;
    int within = lane_id & 3;

    smem_fp32[groupId * 16 + within * 2]       = c0_0;
    smem_fp32[groupId * 16 + within * 2 + 1]   = c0_1;
    smem_fp32[(groupId+8) * 16 + within * 2]     = c0_2;
    smem_fp32[(groupId+8) * 16 + within * 2 + 1] = c0_3;

    smem_fp32[groupId * 16 + within * 2 + 8]       = c1_0;
    smem_fp32[groupId * 16 + within * 2 + 9]       = c1_1;
    smem_fp32[(groupId+8) * 16 + within * 2 + 8]     = c1_2;
    smem_fp32[(groupId+8) * 16 + within * 2 + 9]     = c1_3;
}

// Load from smem fp32 back into registers
__device__ __forceinline__
void load_fp32_from_fp32_smem(
    const float *smem_fp32,
    float &c0_0, float &c0_1, float &c0_2, float &c0_3,
    float &c1_0, float &c1_1, float &c1_2, float &c1_3,
    int lane_id
) {
    int groupId = lane_id >> 2;
    int within = lane_id & 3;

    c0_0 = smem_fp32[groupId * 16 + within * 2];
    c0_1 = smem_fp32[groupId * 16 + within * 2 + 1];
    c0_2 = smem_fp32[(groupId+8) * 16 + within * 2];
    c0_3 = smem_fp32[(groupId+8) * 16 + within * 2 + 1];

    c1_0 = smem_fp32[groupId * 16 + within * 2 + 8];
    c1_1 = smem_fp32[groupId * 16 + within * 2 + 9];
    c1_2 = smem_fp32[(groupId+8) * 16 + within * 2 + 8];
    c1_3 = smem_fp32[(groupId+8) * 16 + within * 2 + 9];
}

// Compute C[16x16] += A[16x16] @ B[16x16], all in bf16 smem (row-major), result accumulated in fp32 regs
// smem_A: bf16 row-major 16x16 (512 bytes)
// smem_B: bf16 row-major 16x16 (512 bytes)
// Accumulates into c0_{0..3} (cols 0..7) and c1_{0..3} (cols 8..15)
__device__ __forceinline__
void mma_16x16_bf16(
    uint32_t smem_A_addr, // shared mem address of A[16x16] bf16
    uint32_t smem_B_addr, // shared mem address of B[16x16] bf16
    float &c0_0, float &c0_1, float &c0_2, float &c0_3,  // C[:,0:8]
    float &c1_0, float &c1_1, float &c1_2, float &c1_3,  // C[:,8:16]
    int lane_id
) {
    // Load A operand using ldmatrix x4
    // A is 16x16 bf16 row-major. ldmatrix loads 4 m8n8 tiles.
    // For m16n8k16 A operand, we need 4 tiles:
    //   a_regs[0] = tile(rows 0-7, kcols 0-7)   <- lanes 0-7
    //   a_regs[1] = tile(rows 8-15, kcols 0-7)  <- lanes 8-15
    //   a_regs[2] = tile(rows 0-7, kcols 8-15)  <- lanes 16-23
    //   a_regs[3] = tile(rows 8-15, kcols 8-15) <- lanes 24-31
    // Each tile is 8 rows x 8 bf16 cols = 128 bytes.
    // Row-major 16x16 bf16: stride = 16 * 2 = 32 bytes per row.
    uint32_t a_regs[4];
    {
        int a_row = (lane_id % 8) + ((lane_id >> 3) & 1) * 8;
        int a_col_byte = ((lane_id >> 4) & 1) * 16; // 0 for kcols 0-7, 16 for kcols 8-15
        uint32_t addr = smem_A_addr + a_row * 32 + a_col_byte;
        ldmatrix<4>(a_regs, addr);
    }

    // Load B operand for cols 0..7 using ldmatrix_trans x2
    // B is 16x16 bf16 row-major. For cols 0..7:
    //   Lane j addresses: row (j%8) for tile (j/8 % 2)
    //   Tile 0: rows 0..7, Tile 1: rows 8..15
    //   addr = smem_B + (j%8 + (j/8 %2)*8) * 16 * 2 + 0 (col offset 0)
    //        = smem_B + ((j%8) + ((j>>3)&1)*8) * 32
    uint32_t b0_regs[2];
    {
        uint32_t r = (lane_id % 8) + ((lane_id >> 3) & 1) * 8;
        uint32_t addr = smem_B_addr + r * 16 * sizeof(__nv_bfloat16);
        ldmatrix_trans<2>(b0_regs, addr);
    }

    // MMA for C[:,0:8]
    mma_m16n8k16_bf16(
        c0_0, c0_1, c0_2, c0_3,
        a_regs[0], a_regs[1], a_regs[2], a_regs[3],
        b0_regs[0], b0_regs[1],
        c0_0, c0_1, c0_2, c0_3
    );

    // Load B operand for cols 8..15 using ldmatrix_trans x2
    uint32_t b1_regs[2];
    {
        uint32_t r = (lane_id % 8) + ((lane_id >> 3) & 1) * 8;
        // Offset by 8 columns = 8 * sizeof(bf16) = 16 bytes
        uint32_t addr = smem_B_addr + r * 16 * sizeof(__nv_bfloat16) + 8 * sizeof(__nv_bfloat16);
        ldmatrix_trans<2>(b1_regs, addr);
    }

    // MMA for C[:,8:16]
    mma_m16n8k16_bf16(
        c1_0, c1_1, c1_2, c1_3,
        a_regs[0], a_regs[1], a_regs[2], a_regs[3],
        b1_regs[0], b1_regs[1],
        c1_0, c1_1, c1_2, c1_3
    );
}

// Same as above but does NOT accumulate (zero-initializes accumulators)
__device__ __forceinline__
void mma_16x16_bf16_zero(
    uint32_t smem_A_addr,
    uint32_t smem_B_addr,
    float &c0_0, float &c0_1, float &c0_2, float &c0_3,
    float &c1_0, float &c1_1, float &c1_2, float &c1_3,
    int lane_id
) {
    c0_0 = c0_1 = c0_2 = c0_3 = 0.0f;
    c1_0 = c1_1 = c1_2 = c1_3 = 0.0f;
    mma_16x16_bf16(smem_A_addr, smem_B_addr,
                   c0_0, c0_1, c0_2, c0_3,
                   c1_0, c1_1, c1_2, c1_3,
                   lane_id);
}

// ═══════════════════════════════════════════════════════════════════
// tf32 MMA: C[16x16] += A[16x16] @ B[16x16], both A and B in fp32 smem
// Uses mma.sync m16n8k8 tf32. K=16 means 2 k-tiles (k=0..7, k=8..15).
// ═══════════════════════════════════════════════════════════════════

// tf32 MMA using direct shared memory float pointers
__device__ __forceinline__
void mma_16x16_tf32_ptr(
    const float *smem_A, // fp32 16x16 row-major in shared memory
    const float *smem_B, // fp32 16x16 row-major in shared memory
    float &c0_0, float &c0_1, float &c0_2, float &c0_3,
    float &c1_0, float &c1_1, float &c1_2, float &c1_3,
    int lane_id
) {
    int groupId = lane_id >> 2;
    int within = lane_id & 3;

    // k-step 0: k=0..7
    {
        // A operand: A[16x8], rows via groupId, cols via within
        // For m16n8k8 A fragment (row operand):
        //   a0 = A[groupId, within*2]
        //   a1 = A[groupId, within*2+1]
        //   a2 = A[groupId+8, within*2]
        //   a3 = A[groupId+8, within*2+1]
        float a0 = smem_A[groupId * 16 + within * 2];
        float a1 = smem_A[groupId * 16 + within * 2 + 1];
        float a2 = smem_A[(groupId + 8) * 16 + within * 2];
        float a3 = smem_A[(groupId + 8) * 16 + within * 2 + 1];

        // B operand for cols 0..7
        // For m16n8k8 tf32, B fragment: b0 = B[within*2, groupId], b1 = B[within*2+1, groupId]
        // B stored row-major: B[k][n] = smem_B[k * 16 + n]
        float b0 = smem_B[(within * 2) * 16 + groupId];
        float b1 = smem_B[(within * 2 + 1) * 16 + groupId];

        mma_m16n8k8_tf32(
            c0_0, c0_1, c0_2, c0_3,
            a0, a1, a2, a3,
            b0, b1,
            c0_0, c0_1, c0_2, c0_3
        );

        // B operand for cols 8..15
        float b0_hi = smem_B[(within * 2) * 16 + groupId + 8];
        float b1_hi = smem_B[(within * 2 + 1) * 16 + groupId + 8];

        mma_m16n8k8_tf32(
            c1_0, c1_1, c1_2, c1_3,
            a0, a1, a2, a3,
            b0_hi, b1_hi,
            c1_0, c1_1, c1_2, c1_3
        );
    }

    // k-step 1: k=8..15
    {
        // A operand from columns 8..15 of smem_A
        float a0 = smem_A[groupId * 16 + within * 2 + 8];
        float a1 = smem_A[groupId * 16 + within * 2 + 9];
        float a2 = smem_A[(groupId + 8) * 16 + within * 2 + 8];
        float a3 = smem_A[(groupId + 8) * 16 + within * 2 + 9];

        // B operand: rows 8..15, cols 0..7
        float b0 = smem_B[(within * 2 + 8) * 16 + groupId];
        float b1 = smem_B[(within * 2 + 9) * 16 + groupId];

        mma_m16n8k8_tf32(
            c0_0, c0_1, c0_2, c0_3,
            a0, a1, a2, a3,
            b0, b1,
            c0_0, c0_1, c0_2, c0_3
        );

        float b0_hi = smem_B[(within * 2 + 8) * 16 + groupId + 8];
        float b1_hi = smem_B[(within * 2 + 9) * 16 + groupId + 8];

        mma_m16n8k8_tf32(
            c1_0, c1_1, c1_2, c1_3,
            a0, a1, a2, a3,
            b0_hi, b1_hi,
            c1_0, c1_1, c1_2, c1_3
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
// W/U output helpers
// Strategy: Tile over output columns in 16-col chunks.
// For each chunk, accumulate contributions from all Ai blocks.
// ═══════════════════════════════════════════════════════════════════

// Helper: multiply each element of a register-distributed 16x16 matrix by a
// per-row scalar vector (16 scalars, one per row), producing the result in the
// same register layout.
// scale[16] is in shared memory (float).
__device__ __forceinline__
void scale_rows(
    float &c0_0, float &c0_1, float &c0_2, float &c0_3,
    float &c1_0, float &c1_1, float &c1_2, float &c1_3,
    const float *smem_scale, // 16 floats in shared memory
    int lane_id
) {
    int groupId = lane_id >> 2;
    float s0 = smem_scale[groupId];
    float s1 = smem_scale[groupId + 8];
    c0_0 *= s0; c0_1 *= s0;
    c0_2 *= s1; c0_3 *= s1;
    c1_0 *= s0; c1_1 *= s0;
    c1_2 *= s1; c1_3 *= s1;
}

// Helper: negate all elements
__device__ __forceinline__
void negate_regs(
    float &c0_0, float &c0_1, float &c0_2, float &c0_3,
    float &c1_0, float &c1_1, float &c1_2, float &c1_3
) {
    c0_0 = -c0_0; c0_1 = -c0_1; c0_2 = -c0_2; c0_3 = -c0_3;
    c1_0 = -c1_0; c1_1 = -c1_1; c1_2 = -c1_2; c1_3 = -c1_3;
}

// Helper: bf16 roundtrip all elements
__device__ __forceinline__
void bf16_roundtrip_regs(
    float &c0_0, float &c0_1, float &c0_2, float &c0_3,
    float &c1_0, float &c1_1, float &c1_2, float &c1_3
) {
    c0_0 = bf16_roundtrip(c0_0); c0_1 = bf16_roundtrip(c0_1);
    c0_2 = bf16_roundtrip(c0_2); c0_3 = bf16_roundtrip(c0_3);
    c1_0 = bf16_roundtrip(c1_0); c1_1 = bf16_roundtrip(c1_1);
    c1_2 = bf16_roundtrip(c1_2); c1_3 = bf16_roundtrip(c1_3);
}

// Helper: set to identity matrix
__device__ __forceinline__
void set_identity(
    float &c0_0, float &c0_1, float &c0_2, float &c0_3,
    float &c1_0, float &c1_1, float &c1_2, float &c1_3,
    int lane_id
) {
    int groupId = lane_id >> 2;
    int within = lane_id & 3;

    // I[r][c] = (r == c) ? 1.0 : 0.0
    // c0_0 = I[groupId, within*2]
    c0_0 = (groupId == within * 2) ? 1.0f : 0.0f;
    c0_1 = (groupId == within * 2 + 1) ? 1.0f : 0.0f;
    c0_2 = ((groupId + 8) == within * 2) ? 1.0f : 0.0f;
    c0_3 = ((groupId + 8) == (within * 2 + 1)) ? 1.0f : 0.0f;
    c1_0 = (groupId == (within * 2 + 8)) ? 1.0f : 0.0f;
    c1_1 = (groupId == (within * 2 + 9)) ? 1.0f : 0.0f;
    c1_2 = ((groupId + 8) == (within * 2 + 8)) ? 1.0f : 0.0f;
    c1_3 = ((groupId + 8) == (within * 2 + 9)) ? 1.0f : 0.0f;
}

// Helper: add identity to registers
__device__ __forceinline__
void add_identity(
    float &c0_0, float &c0_1, float &c0_2, float &c0_3,
    float &c1_0, float &c1_1, float &c1_2, float &c1_3,
    int lane_id
) {
    int groupId = lane_id >> 2;
    int within = lane_id & 3;

    if (groupId == within * 2) c0_0 += 1.0f;
    if (groupId == within * 2 + 1) c0_1 += 1.0f;
    if ((groupId + 8) == within * 2) c0_2 += 1.0f;
    if ((groupId + 8) == (within * 2 + 1)) c0_3 += 1.0f;
    if (groupId == (within * 2 + 8)) c1_0 += 1.0f;
    if (groupId == (within * 2 + 9)) c1_1 += 1.0f;
    if ((groupId + 8) == (within * 2 + 8)) c1_2 += 1.0f;
    if ((groupId + 8) == (within * 2 + 9)) c1_3 += 1.0f;
}

// Helper: subtract identity (C = C - I)
__device__ __forceinline__
void sub_identity(
    float &c0_0, float &c0_1, float &c0_2, float &c0_3,
    float &c1_0, float &c1_1, float &c1_2, float &c1_3,
    int lane_id
) {
    int groupId = lane_id >> 2;
    int within = lane_id & 3;

    if (groupId == within * 2) c0_0 -= 1.0f;
    if (groupId == within * 2 + 1) c0_1 -= 1.0f;
    if ((groupId + 8) == within * 2) c0_2 -= 1.0f;
    if ((groupId + 8) == (within * 2 + 1)) c0_3 -= 1.0f;
    if (groupId == (within * 2 + 8)) c1_0 -= 1.0f;
    if (groupId == (within * 2 + 9)) c1_1 -= 1.0f;
    if ((groupId + 8) == (within * 2 + 8)) c1_2 -= 1.0f;
    if ((groupId + 8) == (within * 2 + 9)) c1_3 -= 1.0f;
}

// Helper: C = I - C
__device__ __forceinline__
void identity_minus(
    float &c0_0, float &c0_1, float &c0_2, float &c0_3,
    float &c1_0, float &c1_1, float &c1_2, float &c1_3,
    int lane_id
) {
    negate_regs(c0_0, c0_1, c0_2, c0_3, c1_0, c1_1, c1_2, c1_3);
    add_identity(c0_0, c0_1, c0_2, c0_3, c1_0, c1_1, c1_2, c1_3, lane_id);
}

// Helper: add two register sets: dst += src
__device__ __forceinline__
void add_regs(
    float &d0_0, float &d0_1, float &d0_2, float &d0_3,
    float &d1_0, float &d1_1, float &d1_2, float &d1_3,
    float s0_0, float s0_1, float s0_2, float s0_3,
    float s1_0, float s1_1, float s1_2, float s1_3
) {
    d0_0 += s0_0; d0_1 += s0_1; d0_2 += s0_2; d0_3 += s0_3;
    d1_0 += s1_0; d1_1 += s1_1; d1_2 += s1_2; d1_3 += s1_3;
}

// ═══════════════════════════════════════════════════════════════════
// Shared memory layout:
//
// We need several temporary buffers. Let's define offsets:
// - smem_A_bf16: 2 x 16x16 bf16 buffers for MMA operands (A, B)  = 2*512 = 1024 bytes
// - smem_fp32:   2 x 16x16 fp32 buffers for tf32 MMA operands    = 2*1024 = 2048 bytes
// - smem_scale:  16 floats for per-row scaling                    = 64 bytes
// - smem_B_kv:   16x16 bf16 for k/v tile loading                 = 512 bytes
//
// Total: ~3648 bytes. Well within limits.
//
// But we also need to store the 10 Ai blocks. With 4 warps all sharing,
// the Ai blocks are in registers. Good.
// ═══════════════════════════════════════════════════════════════════

// Shared memory offsets
struct InverseSmem {
    // Two 16x16 bf16 buffers for MMA operands (A and B for bf16 MMA)
    __nv_bfloat16 buf_A[16 * 16]; // 512 bytes
    __nv_bfloat16 buf_B[16 * 16]; // 512 bytes
    // Two 16x16 fp32 buffers for tf32 MMA
    float buf_A_fp32[16 * 16];    // 1024 bytes
    float buf_B_fp32[16 * 16];    // 1024 bytes
    // Per-row scalars (beta, exp(g))
    float scale[16];              // 64 bytes
    float scale2[16];             // 64 bytes
};
// Total: ~3200 bytes

// ═══════════════════════════════════════════════════════════════════
// Main kernel
// ═══════════════════════════════════════════════════════════════════

__global__ void __launch_bounds__(BLOCK_SIZE_INV)
inverse_kernel_v1(
    const float       *A_ptr,        // [total_T, H, BT] fp32
    const __nv_bfloat16 *k_ptr,      // [total_T, Hg, K_dim] bf16
    const __nv_bfloat16 *v_ptr,      // [total_T, H, V_dim] bf16
    __nv_bfloat16     *w_ptr,        // [total_T, H, K_dim] bf16 output
    __nv_bfloat16     *u_ptr,        // [total_T, H, V_dim] bf16 output
    const float       *beta_ptr,     // [total_T, H] fp32
    const float       *g_cu_ptr,     // [total_T, H] fp32
    const int64_t     *cu_seqlens_ptr,   // [N+1]
    const int32_t     *chunk_indices_ptr, // [total_chunks, 2]
    const int32_t     *total_chunks_ptr,  // [1]
    int H,
    int Hg,
    int K_dim,   // 128
    int V_dim,   // 128
    int BT       // 64
) {
    const int global_chunk_id = blockIdx.x;
    const int head_id = blockIdx.y;

    // Early exit if beyond total chunks
    if (global_chunk_id >= *total_chunks_ptr) return;

    const int seq_id = chunk_indices_ptr[global_chunk_id * 2];
    const int chunk_id = chunk_indices_ptr[global_chunk_id * 2 + 1];
    const int bos = static_cast<int>(cu_seqlens_ptr[seq_id]);
    const int eos = static_cast<int>(cu_seqlens_ptr[seq_id + 1]);
    const int seqlen = eos - bos;

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    // All 4 warps cooperate on the same work.

    // Set up shared memory
    extern __shared__ char smem_raw[];
    InverseSmem *smem = reinterpret_cast<InverseSmem *>(smem_raw);

    // ═══════════════════════════════════════════════════════════════
    // Step 1: Load diagonal blocks A_11, A_22, A_33, A_44 from global memory
    //
    // A layout: [total_T, H, BT], base = A_ptr + bos * H * BT + head_id * BT
    // For block (i,j): rows = chunk_id*BT + i*16 + row, col = j*16 + col
    // A[row][col] = A_ptr[base + (chunk_id*BT + i*16 + row) * H * BT + (j*16 + col)]
    //
    // Each thread loads multiple elements cooperatively.
    // ═══════════════════════════════════════════════════════════════

    const float *A_base = A_ptr + bos * H * BT + head_id * BT;
    const int t_base = chunk_id * BT;

    // Each 16x16 block: 256 elements. With 128 threads, each thread loads 2 elements.
    // Diagonal blocks: A_11 at (0,0), A_22 at (16,16), A_33 at (32,32), A_44 at (48,48)
    // Off-diagonal blocks needed later: A_21(16,0), A_31(32,0), A_32(32,16),
    //                                   A_41(48,0), A_42(48,16), A_43(48,32)

    // We'll process blocks sequentially, loading each to smem, then computing in registers.
    // But actually, each thread needs to own 8 fp32 regs per 16x16 block.
    // 10 blocks * 8 regs = 80 regs for Ai blocks. Plus temporaries.
    // With 255 max regs per thread, this should be fine.

    // Register storage for all Ai blocks (inverted A blocks)
    // Each stored as 8 floats: {c0_0, c0_1, c0_2, c0_3, c1_0, c1_1, c1_2, c1_3}
    // Using the m16n8k16 output mapping.
    float Ai[10][8]; // [block_idx][reg_idx]
    // Block ordering: 0=11, 1=22, 2=33, 3=44, 4=21, 5=31, 6=32, 7=41, 8=42, 9=43

    #define Ai_11 Ai[0]
    #define Ai_22 Ai[1]
    #define Ai_33 Ai[2]
    #define Ai_44 Ai[3]
    #define Ai_21 Ai[4]
    #define Ai_31 Ai[5]
    #define Ai_32 Ai[6]
    #define Ai_41 Ai[7]
    #define Ai_42 Ai[8]
    #define Ai_43 Ai[9]

    // Helper: load a 16x16 fp32 block from global memory to shared memory (fp32 buf)
    // row_offset, col_offset are the block's position in the 64x64 matrix
    auto load_A_block_to_fp32_smem = [&](int row_off, int col_off) {
        // 256 elements, 128 threads -> 2 elements per thread
        for (int i = tid; i < 256; i += BLOCK_SIZE_INV) {
            int r = i / 16;
            int c = i % 16;
            int global_row = t_base + row_off + r;
            float val = 0.0f;
            if (global_row < seqlen) {
                val = A_base[global_row * H * BT + col_off + c];
            }
            smem->buf_A_fp32[r * 16 + c] = val;
        }
        __syncthreads();
    };

    // Helper: load a 16x16 block from smem_fp32 into registers
    auto load_regs_from_fp32_smem = [&](float *regs) {
        load_fp32_from_fp32_smem(smem->buf_A_fp32,
            regs[0], regs[1], regs[2], regs[3],
            regs[4], regs[5], regs[6], regs[7],
            lane_id);
    };

    // Helper: store registers to bf16 smem buffer A
    auto store_regs_to_bf16_A = [&](float *regs) {
        store_fp32_to_bf16_smem(smem->buf_A,
            regs[0], regs[1], regs[2], regs[3],
            regs[4], regs[5], regs[6], regs[7],
            lane_id);
        __syncthreads();
    };

    // Helper: store registers to bf16 smem buffer B
    auto store_regs_to_bf16_B = [&](float *regs) {
        store_fp32_to_bf16_smem(smem->buf_B,
            regs[0], regs[1], regs[2], regs[3],
            regs[4], regs[5], regs[6], regs[7],
            lane_id);
        __syncthreads();
    };

    // Helper: store registers to fp32 smem buffer A (for tf32 MMA)
    auto store_regs_to_fp32_A = [&](float *regs) {
        store_fp32_to_fp32_smem(smem->buf_A_fp32,
            regs[0], regs[1], regs[2], regs[3],
            regs[4], regs[5], regs[6], regs[7],
            lane_id);
        __syncthreads();
    };

    auto store_regs_to_fp32_B = [&](float *regs) {
        store_fp32_to_fp32_smem(smem->buf_B_fp32,
            regs[0], regs[1], regs[2], regs[3],
            regs[4], regs[5], regs[6], regs[7],
            lane_id);
        __syncthreads();
    };

    uint32_t smem_A_bf16_addr = cvt_smem_ptr(smem->buf_A);
    uint32_t smem_B_bf16_addr = cvt_smem_ptr(smem->buf_B);

    // ═══════════════════════════════════════════════════════════════
    // Step 2: Invert diagonal blocks using Neumann + Newton correction
    // ═══════════════════════════════════════════════════════════════

    // Process each diagonal block
    auto invert_diagonal = [&](int block_idx, int diag_off) {
        // Load A_diag from global memory to fp32 smem
        load_A_block_to_fp32_smem(diag_off, diag_off);

        // Load A_orig into registers (for Newton correction later)
        float A_orig[8];
        load_regs_from_fp32_smem(A_orig);

        // A_bf16 = A_orig.to(bf16)
        // Store A_bf16 to bf16 smem
        float A_bf16_regs[8];
        for (int i = 0; i < 8; i++) A_bf16_regs[i] = bf16_roundtrip(A_orig[i]);

        // Ai = I - A_bf16
        float Ai_regs[8];
        for (int i = 0; i < 8; i++) Ai_regs[i] = -A_bf16_regs[i];
        // Add identity
        {
            int groupId = lane_id >> 2;
            int within = lane_id & 3;
            if (groupId == within * 2) Ai_regs[0] += 1.0f;
            if (groupId == within * 2 + 1) Ai_regs[1] += 1.0f;
            if ((groupId + 8) == within * 2) Ai_regs[2] += 1.0f;
            if ((groupId + 8) == (within * 2 + 1)) Ai_regs[3] += 1.0f;
            if (groupId == (within * 2 + 8)) Ai_regs[4] += 1.0f;
            if (groupId == (within * 2 + 9)) Ai_regs[5] += 1.0f;
            if ((groupId + 8) == (within * 2 + 8)) Ai_regs[6] += 1.0f;
            if ((groupId + 8) == (within * 2 + 9)) Ai_regs[7] += 1.0f;
        }

        // Iteration 1: A_pow = A_bf16 @ A_bf16 (bf16 MMA)
        // Store A_bf16 to both smem A and B
        store_regs_to_bf16_A(A_bf16_regs);
        store_regs_to_bf16_B(A_bf16_regs);

        float A_pow[8];
        mma_16x16_bf16_zero(smem_A_bf16_addr, smem_B_bf16_addr,
            A_pow[0], A_pow[1], A_pow[2], A_pow[3],
            A_pow[4], A_pow[5], A_pow[6], A_pow[7],
            lane_id);

        // A_pow_bf16 = A_pow.to(bf16)
        float A_pow_bf16[8];
        for (int i = 0; i < 8; i++) A_pow_bf16[i] = bf16_roundtrip(A_pow[i]);

        // I_plus_Apow = I + A_pow_bf16 (store to smem B)
        float I_plus_Apow[8];
        for (int i = 0; i < 8; i++) I_plus_Apow[i] = A_pow_bf16[i];
        add_identity(I_plus_Apow[0], I_plus_Apow[1], I_plus_Apow[2], I_plus_Apow[3],
                     I_plus_Apow[4], I_plus_Apow[5], I_plus_Apow[6], I_plus_Apow[7],
                     lane_id);

        // Ai = Ai @ (I + A_pow_bf16) (bf16 MMA)
        // Ai is the A operand, I_plus_Apow is B operand
        store_regs_to_bf16_A(Ai_regs);
        store_regs_to_bf16_B(I_plus_Apow);

        mma_16x16_bf16_zero(smem_A_bf16_addr, smem_B_bf16_addr,
            Ai_regs[0], Ai_regs[1], Ai_regs[2], Ai_regs[3],
            Ai_regs[4], Ai_regs[5], Ai_regs[6], Ai_regs[7],
            lane_id);

        // Iteration 2: A_pow = A_pow_bf16 @ A_pow_bf16
        store_regs_to_bf16_A(A_pow_bf16);
        store_regs_to_bf16_B(A_pow_bf16);

        mma_16x16_bf16_zero(smem_A_bf16_addr, smem_B_bf16_addr,
            A_pow[0], A_pow[1], A_pow[2], A_pow[3],
            A_pow[4], A_pow[5], A_pow[6], A_pow[7],
            lane_id);

        for (int i = 0; i < 8; i++) A_pow_bf16[i] = bf16_roundtrip(A_pow[i]);

        // I_plus_Apow2 = I + A_pow_bf16
        for (int i = 0; i < 8; i++) I_plus_Apow[i] = A_pow_bf16[i];
        add_identity(I_plus_Apow[0], I_plus_Apow[1], I_plus_Apow[2], I_plus_Apow[3],
                     I_plus_Apow[4], I_plus_Apow[5], I_plus_Apow[6], I_plus_Apow[7],
                     lane_id);

        // Ai = Ai.to(bf16) @ (I + A_pow_bf16)
        // Note: Triton does Ai.to(tl.bfloat16) before the dot
        for (int i = 0; i < 8; i++) Ai_regs[i] = bf16_roundtrip(Ai_regs[i]);

        store_regs_to_bf16_A(Ai_regs);
        store_regs_to_bf16_B(I_plus_Apow);

        mma_16x16_bf16_zero(smem_A_bf16_addr, smem_B_bf16_addr,
            Ai_regs[0], Ai_regs[1], Ai_regs[2], Ai_regs[3],
            Ai_regs[4], Ai_regs[5], Ai_regs[6], Ai_regs[7],
            lane_id);

        // Newton correction: MAi = Ai + A_orig @ Ai (tf32)
        // MAi = Ai + dot(A_orig, Ai, tf32)
        // Store A_orig and Ai to fp32 smem for tf32 MMA
        store_regs_to_fp32_A(A_orig);
        store_regs_to_fp32_B(Ai_regs);

        float MAi[8];
        for (int i = 0; i < 8; i++) MAi[i] = Ai_regs[i]; // start with Ai
        mma_16x16_tf32_ptr(smem->buf_A_fp32, smem->buf_B_fp32,
            MAi[0], MAi[1], MAi[2], MAi[3],
            MAi[4], MAi[5], MAi[6], MAi[7],
            lane_id);

        // R = MAi - I
        float R[8];
        for (int i = 0; i < 8; i++) R[i] = MAi[i];
        sub_identity(R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7], lane_id);

        // I_minus_R = I - R
        float I_minus_R[8];
        for (int i = 0; i < 8; i++) I_minus_R[i] = R[i];
        identity_minus(I_minus_R[0], I_minus_R[1], I_minus_R[2], I_minus_R[3],
                       I_minus_R[4], I_minus_R[5], I_minus_R[6], I_minus_R[7],
                       lane_id);

        // Ai = Ai @ (I - R) (tf32)
        store_regs_to_fp32_A(Ai_regs);
        store_regs_to_fp32_B(I_minus_R);

        for (int i = 0; i < 8; i++) Ai_regs[i] = 0.0f;
        mma_16x16_tf32_ptr(smem->buf_A_fp32, smem->buf_B_fp32,
            Ai_regs[0], Ai_regs[1], Ai_regs[2], Ai_regs[3],
            Ai_regs[4], Ai_regs[5], Ai_regs[6], Ai_regs[7],
            lane_id);

        // Store final Ai to the register array
        for (int i = 0; i < 8; i++) Ai[block_idx][i] = Ai_regs[i];
    };

    invert_diagonal(0, 0);   // Ai_11
    invert_diagonal(1, 16);  // Ai_22
    invert_diagonal(2, 32);  // Ai_33
    invert_diagonal(3, 48);  // Ai_44

    // Early bf16 roundtrip on diagonal blocks
    for (int b = 0; b < 4; b++) {
        bf16_roundtrip_regs(Ai[b][0], Ai[b][1], Ai[b][2], Ai[b][3],
                            Ai[b][4], Ai[b][5], Ai[b][6], Ai[b][7]);
    }

    // ═══════════════════════════════════════════════════════════════
    // Step 3: Load off-diagonal A blocks
    // ═══════════════════════════════════════════════════════════════

    // We need A_21, A_31, A_32, A_41, A_42, A_43
    // Store them as register arrays temporarily. But we have limited regs.
    // Let's load them as needed.

    // Off-diagonal block register storage
    float A_off[6][8]; // 0=A_21, 1=A_31, 2=A_32, 3=A_41, 4=A_42, 5=A_43

    // Load all 6 off-diagonal blocks
    auto load_off_diag = [&](int idx, int row_off, int col_off) {
        load_A_block_to_fp32_smem(row_off, col_off);
        load_regs_from_fp32_smem(A_off[idx]);
    };

    load_off_diag(0, 16, 0);   // A_21
    load_off_diag(1, 32, 0);   // A_31
    load_off_diag(2, 32, 16);  // A_32
    load_off_diag(3, 48, 0);   // A_41
    load_off_diag(4, 48, 16);  // A_42
    load_off_diag(5, 48, 32);  // A_43

    // ═══════════════════════════════════════════════════════════════
    // Step 4: Off-diagonal via Schur complement
    // ═══════════════════════════════════════════════════════════════

    // Helper: bf16 matmul C = A @ B (zero-init result)
    auto matmul_bf16 = [&](float *C, float *A, float *B) {
        store_regs_to_bf16_A(A);
        store_regs_to_bf16_B(B);
        C[0] = C[1] = C[2] = C[3] = C[4] = C[5] = C[6] = C[7] = 0.0f;
        mma_16x16_bf16(smem_A_bf16_addr, smem_B_bf16_addr,
            C[0], C[1], C[2], C[3], C[4], C[5], C[6], C[7],
            lane_id);
    };

    // Helper: bf16 matmul C += A @ B (accumulate)
    auto matmul_bf16_acc = [&](float *C, float *A, float *B) {
        store_regs_to_bf16_A(A);
        store_regs_to_bf16_B(B);
        mma_16x16_bf16(smem_A_bf16_addr, smem_B_bf16_addr,
            C[0], C[1], C[2], C[3], C[4], C[5], C[6], C[7],
            lane_id);
    };

    // Helper: tf32 matmul C = A @ B (zero-init result)
    auto matmul_tf32 = [&](float *C, float *A, float *B) {
        store_regs_to_fp32_A(A);
        store_regs_to_fp32_B(B);
        C[0] = C[1] = C[2] = C[3] = C[4] = C[5] = C[6] = C[7] = 0.0f;
        mma_16x16_tf32_ptr(smem->buf_A_fp32, smem->buf_B_fp32,
            C[0], C[1], C[2], C[3], C[4], C[5], C[6], C[7],
            lane_id);
    };

    // Helper: tf32 matmul C += A @ B (accumulate)
    auto matmul_tf32_acc = [&](float *C, float *A, float *B) {
        store_regs_to_fp32_A(A);
        store_regs_to_fp32_B(B);
        mma_16x16_tf32_ptr(smem->buf_A_fp32, smem->buf_B_fp32,
            C[0], C[1], C[2], C[3], C[4], C[5], C[6], C[7],
            lane_id);
    };

    // Level 0 (bf16):
    // tmp = Ai_22 @ A_21; Ai_21 = -(tmp @ Ai_11)
    {
        float tmp[8];
        // Convert inputs to bf16 for the dot product (matching Triton's .to(bf16))
        float Ai22_bf16[8], A21_bf16[8], Ai11_bf16[8];
        for (int i = 0; i < 8; i++) {
            Ai22_bf16[i] = bf16_roundtrip(Ai_22[i]);
            A21_bf16[i] = bf16_roundtrip(A_off[0][i]);
            Ai11_bf16[i] = bf16_roundtrip(Ai_11[i]);
        }
        matmul_bf16(tmp, Ai22_bf16, A21_bf16);
        // tmp.to(bf16) before next dot
        for (int i = 0; i < 8; i++) tmp[i] = bf16_roundtrip(tmp[i]);
        matmul_bf16(Ai_21, tmp, Ai11_bf16);
        negate_regs(Ai_21[0], Ai_21[1], Ai_21[2], Ai_21[3],
                    Ai_21[4], Ai_21[5], Ai_21[6], Ai_21[7]);
    }

    // tmp = Ai_33 @ A_32; Ai_32 = -(tmp @ Ai_22)
    {
        float tmp[8];
        float Ai33_bf16[8], A32_bf16[8], Ai22_bf16[8];
        for (int i = 0; i < 8; i++) {
            Ai33_bf16[i] = bf16_roundtrip(Ai_33[i]);
            A32_bf16[i] = bf16_roundtrip(A_off[2][i]);
            Ai22_bf16[i] = bf16_roundtrip(Ai_22[i]);
        }
        matmul_bf16(tmp, Ai33_bf16, A32_bf16);
        for (int i = 0; i < 8; i++) tmp[i] = bf16_roundtrip(tmp[i]);
        matmul_bf16(Ai_32, tmp, Ai22_bf16);
        negate_regs(Ai_32[0], Ai_32[1], Ai_32[2], Ai_32[3],
                    Ai_32[4], Ai_32[5], Ai_32[6], Ai_32[7]);
    }

    // tmp = Ai_44 @ A_43; Ai_43 = -(tmp @ Ai_33)
    {
        float tmp[8];
        float Ai44_bf16[8], A43_bf16[8], Ai33_bf16[8];
        for (int i = 0; i < 8; i++) {
            Ai44_bf16[i] = bf16_roundtrip(Ai_44[i]);
            A43_bf16[i] = bf16_roundtrip(A_off[5][i]);
            Ai33_bf16[i] = bf16_roundtrip(Ai_33[i]);
        }
        matmul_bf16(tmp, Ai44_bf16, A43_bf16);
        for (int i = 0; i < 8; i++) tmp[i] = bf16_roundtrip(tmp[i]);
        matmul_bf16(Ai_43, tmp, Ai33_bf16);
        negate_regs(Ai_43[0], Ai_43[1], Ai_43[2], Ai_43[3],
                    Ai_43[4], Ai_43[5], Ai_43[6], Ai_43[7]);
    }

    // Level 1 (bf16):
    // tmp = A_31.bf16 @ Ai_11.bf16; tmp += A_32.bf16 @ Ai_21.bf16
    // Ai_31 = -(Ai_33.bf16 @ tmp.bf16)
    {
        float tmp[8];
        float A31_bf16[8], Ai11_bf16[8], A32_bf16[8], Ai21_bf16[8], Ai33_bf16[8];
        for (int i = 0; i < 8; i++) {
            A31_bf16[i] = bf16_roundtrip(A_off[1][i]);
            Ai11_bf16[i] = bf16_roundtrip(Ai_11[i]);
            A32_bf16[i] = bf16_roundtrip(A_off[2][i]);
            Ai21_bf16[i] = bf16_roundtrip(Ai_21[i]);
            Ai33_bf16[i] = bf16_roundtrip(Ai_33[i]);
        }
        matmul_bf16(tmp, A31_bf16, Ai11_bf16);
        matmul_bf16_acc(tmp, A32_bf16, Ai21_bf16);
        for (int i = 0; i < 8; i++) tmp[i] = bf16_roundtrip(tmp[i]);
        matmul_bf16(Ai_31, Ai33_bf16, tmp);
        negate_regs(Ai_31[0], Ai_31[1], Ai_31[2], Ai_31[3],
                    Ai_31[4], Ai_31[5], Ai_31[6], Ai_31[7]);
    }

    // tmp = A_42.bf16 @ Ai_22.bf16; tmp += A_43.bf16 @ Ai_32.bf16
    // Ai_42 = -(Ai_44.bf16 @ tmp.bf16)
    {
        float tmp[8];
        float A42_bf16[8], Ai22_bf16[8], A43_bf16[8], Ai32_bf16[8], Ai44_bf16[8];
        for (int i = 0; i < 8; i++) {
            A42_bf16[i] = bf16_roundtrip(A_off[4][i]);
            Ai22_bf16[i] = bf16_roundtrip(Ai_22[i]);
            A43_bf16[i] = bf16_roundtrip(A_off[5][i]);
            Ai32_bf16[i] = bf16_roundtrip(Ai_32[i]);
            Ai44_bf16[i] = bf16_roundtrip(Ai_44[i]);
        }
        matmul_bf16(tmp, A42_bf16, Ai22_bf16);
        matmul_bf16_acc(tmp, A43_bf16, Ai32_bf16);
        for (int i = 0; i < 8; i++) tmp[i] = bf16_roundtrip(tmp[i]);
        matmul_bf16(Ai_42, Ai44_bf16, tmp);
        negate_regs(Ai_42[0], Ai_42[1], Ai_42[2], Ai_42[3],
                    Ai_42[4], Ai_42[5], Ai_42[6], Ai_42[7]);
    }

    // Level 2 (tf32x3 -> we use regular tf32 for now):
    // tmp = A_41 @ Ai_11 (tf32)
    // tmp += A_42 @ Ai_21 (tf32)
    // tmp += A_43 @ Ai_31 (tf32)
    // Ai_41 = -(Ai_44 @ tmp) (tf32)
    {
        float tmp[8];
        matmul_tf32(tmp, A_off[3], Ai_11); // A_41 @ Ai_11
        matmul_tf32_acc(tmp, A_off[4], Ai_21); // += A_42 @ Ai_21
        matmul_tf32_acc(tmp, A_off[5], Ai_31); // += A_43 @ Ai_31

        float neg_tmp[8];
        matmul_tf32(neg_tmp, Ai_44, tmp); // Ai_44 @ tmp
        negate_regs(neg_tmp[0], neg_tmp[1], neg_tmp[2], neg_tmp[3],
                    neg_tmp[4], neg_tmp[5], neg_tmp[6], neg_tmp[7]);
        for (int i = 0; i < 8; i++) Ai_41[i] = neg_tmp[i];
    }

    // bf16 roundtrip on off-diagonal blocks
    for (int b = 4; b < 10; b++) {
        bf16_roundtrip_regs(Ai[b][0], Ai[b][1], Ai[b][2], Ai[b][3],
                            Ai[b][4], Ai[b][5], Ai[b][6], Ai[b][7]);
    }

    // ═══════════════════════════════════════════════════════════════
    // Step 5: Load k, v, beta, g_cu and compute W, U
    //
    // For each 16-row group i (0..3):
    //   Load beta[i], g_cu[i] (16 values each)
    //   For row-group i, the contributing Ai blocks are:
    //     Row 0: Ai_11 (with b0)
    //     Row 1: Ai_21 (with b0), Ai_22 (with b1)
    //     Row 2: Ai_31 (with b0), Ai_32 (with b1), Ai_33 (with b2)
    //     Row 3: Ai_41 (with b0), Ai_42 (with b1), Ai_43 (with b2), Ai_44 (with b3)
    //
    // For W[i] and U[i]: output is 16xK_dim and 16xV_dim respectively.
    // W[i] = sum_j (Ai_ij * beta_j * exp(g_j)) @ k_j
    // U[i] = sum_j (Ai_ij * beta_j) @ v_j
    //
    // Process by tiling over K_dim/V_dim in 16-column chunks.
    // ═══════════════════════════════════════════════════════════════

    // Pointers adjusted to this chunk/head
    const __nv_bfloat16 *k_base = k_ptr + bos * Hg * K_dim + (head_id / (H / Hg)) * K_dim;
    const __nv_bfloat16 *v_base = v_ptr + bos * H * V_dim + head_id * V_dim;
    __nv_bfloat16 *w_base = w_ptr + bos * H * K_dim + head_id * K_dim;
    __nv_bfloat16 *u_base = u_ptr + bos * H * V_dim + head_id * V_dim;
    const float *beta_base = beta_ptr + bos * H + head_id;
    const float *g_cu_base = g_cu_ptr + bos * H + head_id;

    // Load beta and g_cu for all 4 groups into shared memory
    // beta[i] is at beta_base + (t_base + i*16 + row) * H
    // g_cu[i] is at g_cu_base + (t_base + i*16 + row) * H
    // But we only have 16 scalars per group.

    // Process each output row-group
    // Row-group i output row range: [i*16, (i+1)*16)
    // Contributing blocks: Ai_{i+1, j+1} for j = 0..i, paired with beta_j, k_j, v_j

    // Ai block index mapping:
    // Ai_{1,1} = Ai[0], Ai_{2,1} = Ai[4], Ai_{2,2} = Ai[1],
    // Ai_{3,1} = Ai[5], Ai_{3,2} = Ai[6], Ai_{3,3} = Ai[2],
    // Ai_{4,1} = Ai[7], Ai_{4,2} = Ai[8], Ai_{4,3} = Ai[9], Ai_{4,4} = Ai[3]

    // For row-group i (0-indexed), the contributing (Ai_block_idx, source_group j):
    static constexpr int contrib_block[4][4] = {
        {0, -1, -1, -1},  // row 0: Ai_11
        {4,  1, -1, -1},  // row 1: Ai_21, Ai_22
        {5,  6,  2, -1},  // row 2: Ai_31, Ai_32, Ai_33
        {7,  8,  9,  3},  // row 3: Ai_41, Ai_42, Ai_43, Ai_44
    };

    for (int i = 0; i < 4; i++) {
        int num_contrib = i + 1;

        // Process W (16 x K_dim) and U (16 x V_dim) outputs for this row group
        // Tile over output columns in chunks of 16

        // First, do W output (K_dim = 128, 8 tiles of 16)
        for (int col_tile = 0; col_tile < K_dim; col_tile += 16) {
            float out[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

            for (int j = 0; j < num_contrib; j++) {
                int src_group = j; // source row group providing k/beta/g
                int ai_idx = contrib_block[i][j];

                // Load beta[src_group] and g_cu[src_group] to shared memory
                // beta and g_cu are per-row scalars
                if (tid < 16) {
                    int t = t_base + src_group * 16 + tid;
                    float b_val = 0.0f;
                    float g_val = 0.0f;
                    if (t < seqlen) {
                        b_val = beta_base[t * H];
                        g_val = g_cu_base[t * H];
                    }
                    smem->scale[tid] = b_val;
                    smem->scale2[tid] = g_val;
                }
                __syncthreads();

                // Compute Ab = (Ai[ai_idx] * beta) * exp(g_cu) and store to bf16 smem
                // Matching Triton: Ab_xx = Ai_xx * b_j, then (Ab_xx * eg_j).to(bf16)
                float Ab[8];
                {
                    int groupId = lane_id >> 2;

                    float beta_0 = smem->scale[groupId];
                    float beta_1 = smem->scale[groupId + 8];
                    float eg_0 = expf(smem->scale2[groupId]);
                    float eg_1 = expf(smem->scale2[groupId + 8]);

                    // First multiply by beta (Ai * beta), then by exp(g)
                    Ab[0] = (Ai[ai_idx][0] * beta_0) * eg_0;
                    Ab[1] = (Ai[ai_idx][1] * beta_0) * eg_0;
                    Ab[2] = (Ai[ai_idx][2] * beta_1) * eg_1;
                    Ab[3] = (Ai[ai_idx][3] * beta_1) * eg_1;
                    Ab[4] = (Ai[ai_idx][4] * beta_0) * eg_0;
                    Ab[5] = (Ai[ai_idx][5] * beta_0) * eg_0;
                    Ab[6] = (Ai[ai_idx][6] * beta_1) * eg_1;
                    Ab[7] = (Ai[ai_idx][7] * beta_1) * eg_1;
                }

                // Store Ab to bf16 smem A
                store_regs_to_bf16_A(Ab);

                // Load k[src_group] tile [16x16] from global memory to bf16 smem B
                {
                    for (int idx = tid; idx < 256; idx += BLOCK_SIZE_INV) {
                        int r = idx / 16;
                        int c = idx % 16;
                        int t = t_base + src_group * 16 + r;
                        __nv_bfloat16 val = __float2bfloat16(0.0f);
                        if (t < seqlen) {
                            val = k_base[t * Hg * K_dim + col_tile + c];
                        }
                        smem->buf_B[r * 16 + c] = val;
                    }
                    __syncthreads();
                }

                // Accumulate: out += Ab @ k_tile
                mma_16x16_bf16(smem_A_bf16_addr, smem_B_bf16_addr,
                    out[0], out[1], out[2], out[3],
                    out[4], out[5], out[6], out[7],
                    lane_id);
            }

            // Store out[16x16] to w_base (bf16)
            {
                int groupId = lane_id >> 2;
                int within = lane_id & 3;

                auto store_elem = [&](int row, int col, float val) {
                    int t = t_base + i * 16 + row;
                    if (t < seqlen) {
                        w_base[t * H * K_dim + col_tile + col] = __float2bfloat16(val);
                    }
                };

                store_elem(groupId, within * 2, out[0]);
                store_elem(groupId, within * 2 + 1, out[1]);
                store_elem(groupId + 8, within * 2, out[2]);
                store_elem(groupId + 8, within * 2 + 1, out[3]);
                store_elem(groupId, within * 2 + 8, out[4]);
                store_elem(groupId, within * 2 + 9, out[5]);
                store_elem(groupId + 8, within * 2 + 8, out[6]);
                store_elem(groupId + 8, within * 2 + 9, out[7]);
            }
        }

        // Now do U output (V_dim = 128, 8 tiles of 16)
        for (int col_tile = 0; col_tile < V_dim; col_tile += 16) {
            float out[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

            for (int j = 0; j < num_contrib; j++) {
                int src_group = j;
                int ai_idx = contrib_block[i][j];

                // Load beta[src_group] to shared memory (no exp(g) for U)
                if (tid < 16) {
                    int t = t_base + src_group * 16 + tid;
                    float b_val = 0.0f;
                    if (t < seqlen) {
                        b_val = beta_base[t * H];
                    }
                    smem->scale[tid] = b_val;
                }
                __syncthreads();

                // Compute Ab = Ai[ai_idx] * beta and store to bf16 smem A
                float Ab[8];
                {
                    int groupId = lane_id >> 2;

                    float scale_0 = smem->scale[groupId];
                    float scale_1 = smem->scale[groupId + 8];

                    Ab[0] = Ai[ai_idx][0] * scale_0;
                    Ab[1] = Ai[ai_idx][1] * scale_0;
                    Ab[2] = Ai[ai_idx][2] * scale_1;
                    Ab[3] = Ai[ai_idx][3] * scale_1;
                    Ab[4] = Ai[ai_idx][4] * scale_0;
                    Ab[5] = Ai[ai_idx][5] * scale_0;
                    Ab[6] = Ai[ai_idx][6] * scale_1;
                    Ab[7] = Ai[ai_idx][7] * scale_1;
                }

                store_regs_to_bf16_A(Ab);

                // Load v[src_group] tile [16x16] from global memory to bf16 smem B
                {
                    for (int idx = tid; idx < 256; idx += BLOCK_SIZE_INV) {
                        int r = idx / 16;
                        int c = idx % 16;
                        int t = t_base + src_group * 16 + r;
                        __nv_bfloat16 val = __float2bfloat16(0.0f);
                        if (t < seqlen) {
                            val = v_base[t * H * V_dim + col_tile + c];
                        }
                        smem->buf_B[r * 16 + c] = val;
                    }
                    __syncthreads();
                }

                // Accumulate: out += Ab @ v_tile
                mma_16x16_bf16(smem_A_bf16_addr, smem_B_bf16_addr,
                    out[0], out[1], out[2], out[3],
                    out[4], out[5], out[6], out[7],
                    lane_id);
            }

            // Store out[16x16] to u_base (bf16)
            {
                int groupId = lane_id >> 2;
                int within = lane_id & 3;

                auto store_elem = [&](int row, int col, float val) {
                    int t = t_base + i * 16 + row;
                    if (t < seqlen) {
                        u_base[t * H * V_dim + col_tile + col] = __float2bfloat16(val);
                    }
                };

                store_elem(groupId, within * 2, out[0]);
                store_elem(groupId, within * 2 + 1, out[1]);
                store_elem(groupId + 8, within * 2, out[2]);
                store_elem(groupId + 8, within * 2 + 1, out[3]);
                store_elem(groupId, within * 2 + 8, out[4]);
                store_elem(groupId, within * 2 + 9, out[5]);
                store_elem(groupId + 8, within * 2 + 8, out[6]);
                store_elem(groupId + 8, within * 2 + 9, out[7]);
            }
        }
    }

    #undef Ai_11
    #undef Ai_22
    #undef Ai_33
    #undef Ai_44
    #undef Ai_21
    #undef Ai_31
    #undef Ai_32
    #undef Ai_41
    #undef Ai_42
    #undef Ai_43
}

// ═══════════════════════════════════════════════════════════════════
// Host launcher
// ═══════════════════════════════════════════════════════════════════

void inverse_v1(
    TensorView A,            // [total_T, H, BT] fp32
    TensorView k,            // [total_T, Hg, K_dim] bf16
    TensorView v,            // [total_T, H, V_dim] bf16
    TensorView w,            // [total_T, H, K_dim] bf16 output
    TensorView u,            // [total_T, H, V_dim] bf16 output
    TensorView beta,         // [total_T, H] fp32
    TensorView g_cu,         // [total_T, H] fp32
    TensorView cu_seqlens,   // [N+1] int64
    TensorView chunk_indices,// [total_chunks, 2] int32
    TensorView total_chunks, // [1] int32 (value = total number of chunks)
    int upper_bound_chunks   // grid dim X
) {
    const int H_val = static_cast<int>(A.size(1));
    const int BT_val = static_cast<int>(A.size(2));
    const int Hg_val = static_cast<int>(k.size(1));
    const int K_dim_val = static_cast<int>(k.size(2));
    const int V_dim_val = static_cast<int>(v.size(2));

    auto *A_ptr = reinterpret_cast<const float *>(A.data_ptr());
    auto *k_ptr = reinterpret_cast<const __nv_bfloat16 *>(k.data_ptr());
    auto *v_ptr = reinterpret_cast<const __nv_bfloat16 *>(v.data_ptr());
    auto *w_ptr_out = reinterpret_cast<__nv_bfloat16 *>(w.data_ptr());
    auto *u_ptr_out = reinterpret_cast<__nv_bfloat16 *>(u.data_ptr());
    auto *beta_ptr = reinterpret_cast<const float *>(beta.data_ptr());
    auto *g_cu_ptr = reinterpret_cast<const float *>(g_cu.data_ptr());
    auto *cu_seqlens_ptr = reinterpret_cast<const int64_t *>(cu_seqlens.data_ptr());
    auto *chunk_indices_ptr = reinterpret_cast<const int32_t *>(chunk_indices.data_ptr());
    auto *total_chunks_ptr = reinterpret_cast<const int32_t *>(total_chunks.data_ptr());

    int smem_size = sizeof(InverseSmem);

    dim3 grid(upper_bound_chunks, H_val);
    dim3 block(BLOCK_SIZE_INV);

    cudaFuncSetAttribute(inverse_kernel_v1,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    inverse_kernel_v1<<<grid, block, smem_size>>>(
        A_ptr, k_ptr, v_ptr, w_ptr_out, u_ptr_out,
        beta_ptr, g_cu_ptr,
        cu_seqlens_ptr, chunk_indices_ptr, total_chunks_ptr,
        H_val, Hg_val, K_dim_val, V_dim_val, BT_val
    );
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(inverse_v1, inverse_v1);
