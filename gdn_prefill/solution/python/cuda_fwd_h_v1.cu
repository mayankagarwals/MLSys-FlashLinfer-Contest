// # CUDA H warp
// load h0 as h - [V_dim, K_dim]
//
// for chunk_id in range(num_chunks):
//     # TMA warp
//     load w, v, k from gmem->smem
//     w - [BT, K_dim]
//     v - [BT, V_dim]
//     k - [BT, K_dim]
//
//     # CUDA H warp
//     convert h to h_bf16
//     store h_bf16 to gmem (for O kernel)
//     store h_bf16 to tmem (for wh MMA)
//
//     # MMA warp
//     wait w(smem) from TMA
//     wait h(smem) from CUDA H warp
//     issue hw(tmem) = h @ w.T - [V_dim, BT]
//
//     # CUDA V warp
//     load g_cu from gmem->rmem
//     wait and load v(smem) from TMA
//     wait and load hw(tmem) from MMA
//     v_new.T = v.T - hw - [V_dim, BT]
//     store bf16(v_new) to gmem (for O kernel)
//     v_new *= tl.exp(g_cu_last - g_cu)
//     store v_new to tmem (for MMA)
//
//     # MMA warp
//     wait and load k(smem) from TMA
//     wait v_new(tmem) from CUDA V warp
//     issue vk(tmem) = v_new.T @ k - [V_dim, K_dim]
//
//     # CUDA H warp
//     load g_cu_last from gmem->rmem
//     wait and load vk(tmem) from MMA
//     h = tl.exp(g_cu_last) * h + vk

#include <cuda_bf16.h>
#include <cudaTypedefs.h>
#include <cstdint>
#include "cuda_utils.h"

constexpr int WARP_SIZE = 32;

template <typename T>
__device__ inline
T warp_uniform(T x) { return __shfl_sync(0xFFFF'FFFF, x, 0); }

__device__ inline
int cdiv(int a, int b) { return (a + b - 1) / b; }

__device__ inline
uint32_t fp32x2_to_bf16x2(float a, float b) {
  uint32_t tmp;
  asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(tmp) : "f"(b), "f"(a));
  return tmp;
}

__device__ inline
void bf16x2_to_fp32x2(float *out, uint32_t data) {
  asm volatile(
    "shl.b32 %0, %2, 16;\n"
    "and.b32 %1, %2, 0xFFFF;"
    : "=f"(out[0]), "=f"(out[1]) : "r"(data));
}

__device__ inline
float lds_f32(uint32_t addr) {
  float tmp;
  asm volatile("ld.shared.f32 %0, [%1];" : "=f"(tmp) : "r"(addr));
  return tmp;
}

__device__ inline
void sts_f32(uint32_t addr, float x) {
  asm volatile("st.shared.f32 [%0], %1;" :: "r"(addr), "f"(x));
}

constexpr int H = 8;
constexpr int Hg = 4;
constexpr int BT = 16;
constexpr int K_dim = 128;
constexpr int V_dim = 128;

constexpr uint32_t W_size = BT * K_dim * sizeof(nv_bfloat16);
constexpr uint32_t V_size = BT * V_dim * sizeof(nv_bfloat16);
constexpr uint32_t K_size = BT * K_dim * sizeof(nv_bfloat16);
constexpr uint32_t STAGE_SIZE = W_size + V_size + K_size;

constexpr int NUM_WARPS = 4 + 4 + 2;
constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;

template <int NUM_STAGES>
__global__
__block_size__((TB_SIZE, 1, 1))
void fwd_h_kernel_cutlass(
  const __grid_constant__ CUtensorMap K_tmap,      // [total_T, Hg, K_dim]
  const __grid_constant__ CUtensorMap V_tmap,      // [total_T, H, V_dim]
  const __grid_constant__ CUtensorMap W_tmap,      // [total_T, H, K_dim]
  const __grid_constant__ CUtensorMap V_new_tmap,  // [total_T, H, V_dim]
  const float       *g_cu_ptr,                     // [total_T, H]
        nv_bfloat16 *h_ptr,                        // [total_num_chunks, H, V_dim, K_dim]
  const float       *h0_ptr,                       // [N, H, V_dim, K_dim]
        float       *ht_ptr,                       // [N, H, V_dim, K_dim]
  const int64_t     *cu_seqlens_ptr,               // [N+1]
  const int32_t     *chunk_offsets_ptr             // [total_num_chunks]
) {
  const int tid = threadIdx.x;
  const int warp_id = warp_uniform(tid / WARP_SIZE);
  const int lane_id = tid % WARP_SIZE;

  const int head_id = blockIdx.x;
  const int seq_id  = blockIdx.y;

  const int bos = cu_seqlens_ptr[seq_id];
  const int eos = cu_seqlens_ptr[seq_id + 1];
  const int seqlen = eos - bos;
  const int num_chunks = cdiv(seqlen, BT);

  // set up smem
  extern __shared__ __align__(1024) char smem_ptr[];
  const uint32_t smem = __cvta_generic_to_shared(smem_ptr);

  const uint32_t V_new_smem = smem + NUM_STAGES * STAGE_SIZE;
  const uint32_t v_scale_smem = V_new_smem + V_size;

  const uint32_t tma_mbar_addr = v_scale_smem + BT * sizeof(float);
  const uint32_t mma_mbar_addr = tma_mbar_addr + NUM_STAGES * 8;
  const uint32_t h_mbar_addr = mma_mbar_addr + NUM_STAGES * 8;
  const uint32_t v_mbar_addr = h_mbar_addr + 8;
  const uint32_t wh_mbar_addr = v_mbar_addr + 8;
  const uint32_t vk_mbar_addr = wh_mbar_addr + 8;

  const uint32_t taddr = vk_mbar_addr + 8;

  // set up tmem
  // since we issue wh and vk MMA sequentially, we can overlap the acc and input buffer.
  const uint32_t acc_tmem = 0;
  const uint32_t a_tmem = acc_tmem + max(BT, K_dim);

  if (warp_id == 0) {
    // init mbar
    if (elect_sync()) {
      for (int i = 0; i < NUM_STAGES; i++) {
        mbarrier_init(tma_mbar_addr + i * 8, 1);
        mbarrier_init(mma_mbar_addr + i * 8, 1);
      }
      mbarrier_init(h_mbar_addr, WARP_SIZE * 4);
      mbarrier_init(wh_mbar_addr, 1);
      mbarrier_init(vk_mbar_addr, 1);
      fence_mbarrier_init();
    }
  }
  else if (warp_id == 1) {
    // prefetch TMA descriptor
    if (elect_sync()) {
      prefetch_tensormap(&K_tmap);
      prefetch_tensormap(&V_tmap);
      prefetch_tensormap(&W_tmap);
    }
  }
  __syncthreads();

  if (warp_id == NUM_WARPS - 1) {
    // TMA warp
    if (elect_sync()) {
      int parity = 1;
      int stage_id = 0;

      const int k_head_id = head_id / (H / Hg);

      for (int chunk_id = 0; chunk_id < num_chunks; chunk_id++) {
        // compute addresses
        const int off_t = bos + chunk_id * BT;
        const uint32_t W_smem = smem + stage_id * STAGE_SIZE;
        const uint32_t V_smem = W_smem + W_size;
        const uint32_t K_smem = V_smem + V_size;
        const uint32_t mbar_addr = tma_mbar_addr + stage_id * 8;
  
        // wait MMA warp to release the buffer
        mbarrier_wait(mma_mbar_addr + stage_id * 8, parity);

        // issue TMA and arrive
        // natural shape: [T, H, dim]
        // permuted shape: [H, dim/64, T, 64]
        tma_load_4d(W_smem, &W_tmap, 0, off_t, 0, head_id, mbar_addr);
        tma_load_4d(V_smem, &V_tmap, 0, off_t, 0, head_id, mbar_addr);
        tma_load_4d(K_smem, &K_tmap, 0, off_t, 0, k_head_id, mbar_addr);
        mbarrier_arrive_expect_tx(mbar_addr, STAGE_SIZE);

        // increment stage
        stage_id = (stage_id + 1) % NUM_STAGES;
        if (stage_id == 0)
          parity ^= 1;
      }
    }
  }
  else if (warp_id == NUM_WARPS - 2) {
    // MMA warp
    tcgen05_alloc(taddr, 512);

    if (elect_sync()) {
      int tma_parity = 0;
      int tma_stage = 0;

      int h_parity = 0;

      constexpr uint32_t wh_idesc = make_tcgen05_idesc(V_dim, BT);
      constexpr uint32_t vk_idesc = make_tcgen05_idesc(V_dim, K_dim);

      for (int chunk_id = 0; chunk_id < num_chunks; chunk_id++) {
        const uint32_t W_smem = smem + tma_stage * STAGE_SIZE;
        const uint32_t V_smem = W_smem + W_size;
        const uint32_t K_smem = V_smem + V_size;

        // 128B swizzling
        constexpr uint64_t desc_base = (desc_encode(8 * 128) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
        uint64_t w_desc = desc_base | (W_smem >> 4);
        uint64_t k_desc = desc_base | (K_smem >> 4);

        // wh MMA
        mbarrier_wait(tma_mbar_addr + tma_stage * 8, tma_parity);  // wait for TMA
        mbarrier_wait(h_mbar_addr, h_parity);            // wait for h store to smem
        tcgen05_fence_after_thread_sync();

        int h_tmem = a_tmem;
        tcgen05_mma_tmem(acc_tmem, h_tmem, w_desc, wh_idesc, 0);
        for (int k = 1; k < K_dim / 16; k++) {
          h_tmem += 8;  // next 8 tmem columns (16 BF16)
          w_desc += 32 >> 4;  // next 32 byte
          tcgen05_mma_tmem(acc_tmem, h_tmem, w_desc, wh_idesc, 1);
        }
        tcgen05_commit(wh_mbar_addr);

        // vk MMA
        mbarrier_wait(v_mbar_addr, h_parity);
        tcgen05_fence_after_thread_sync();

        int v_tmem = a_tmem;
        tcgen05_mma_tmem(acc_tmem, v_tmem, k_desc, vk_idesc, 0);
        for (int k = 1; k < 64 / 16; k++) {
          v_tmem += 8;  // next 8 tmem columns (16 BF16)
          k_desc += 32 >> 4;  // next 32 byte
          tcgen05_mma_tmem(acc_tmem, v_tmem, k_desc, vk_idesc, 1);
        }
        tcgen05_commit(vk_mbar_addr);

        tma_stage = (tma_stage + 1) % NUM_STAGES;
        if (tma_stage == 0)
          tma_parity ^= 1;

        h_parity ^= 1;
      }
    }
  }
  else if (warp_id >= 4) {
    // CUDA H warps
    const int tid_ = tid % 128;
    const int warp_id_ = warp_id % 4;

    // collectively, 4 warps represent H[V_dim, K_dim]
    // since V_dim=128, each thread holds H[K_dim] elements
    float h_f32[K_dim];

    int vk_parity = 0;

    // load H0
    for (int i = 0; i < K_dim / 8; i++) {
      const int offset = (seq_id * H + head_id) * V_dim * K_dim + (tid_ * K_dim) + i * 8;
      ldg_u32x8(h_f32 + i * 8, h0_ptr + offset);
    }

    const int chunk_offset = chunk_offsets_ptr[seq_id];
    h_ptr += (chunk_offset * H + head_id) * V_dim * K_dim + (tid_ * K_dim);

    for (int chunk_id = 0; chunk_id < num_chunks; chunk_id++) {
      // pack H to BF16 and store to gmem (for O kernel) and tmem (for WH MMA)
      for (int i = 0; i < K_dim / 16; i++) {
        uint32_t tmp[8];
        for (int j = 0; j < 8; j++)
          tmp[j] = fp32x2_to_bf16x2(h_f32[i * 16 + j * 2], h_f32[i * 16 + j * 2 + 1]);

        stg_u32x8(h_ptr + (chunk_id * H * V_dim * K_dim + i * 16), tmp);  // for O kernel
        tcgen05_st<SHAPE::_32x32b, 16>(warp_id_ * 32, a_tmem + i * 8, tmp);  // for WH MMA
      }
      tcgen05_wait_st();
      tcgen05_fence_before_thread_sync();
      mbarrier_arrive(h_mbar_addr);
  
      // load g_cu_last and compute scaling for H
      float h_scale;
      if (lane_id == 0) {
        const int last_idx = min(bos + chunk_id * BT + chunk_id - 1, eos);
        h_scale = __expf(g_cu_ptr[last_idx * H + head_id]);
      }
      h_scale = warp_uniform(h_scale);

      // wait for vk MMA and update H
      if (warp_id_ == 0)
        mbarrier_wait(vk_mbar_addr, vk_parity);
      bar_sync<1>(128);

      constexpr int WIDTH = 16;  // adjustable
      for (int i = 0; i < K_dim / WIDTH; i++) {
        float vk[WIDTH];
        tcgen05_ld<SHAPE::_32x32b, WIDTH>(vk, warp_id_ * 32, i * WIDTH);

        // TODO: fma.f32x2?
        for (int j = 0; j < WIDTH; j++)
          h_f32[i * WIDTH + j] = h_f32[i * WIDTH + j] * h_scale + vk[j];
      }

      vk_parity ^= 1;
    }

    // store final H
    for (int i = 0; i < K_dim / 8; i++) {
      const int offset = (seq_id * H + head_id) * V_dim * K_dim + (tid_ * K_dim) + i * 8;
      stg_u32x8(h_f32 + i * 8, ht_ptr + offset);
    }
  }
  else {
    // V CUDA warps
    int tma_parity = 0;
    int tma_stage = 0;

    int wh_parity = 0;

    // set up invariance
    if (warp_id == 0 && elect_sync())
      cp_async_bulk_commit_group();

    for (int chunk_id = 0; chunk_id < num_chunks; chunk_id++) {
      // load g_cu and compute scaling for v_new
      if (tid < BT) {
        const int last_idx = min(bos + chunk_id * BT + chunk_id - 1, eos);
        float g_cu = g_cu_ptr[(bos + chunk_id * BT + tid) * H + head_id];
        float g_cu_last = g_cu_ptr[last_idx * H + head_id];

        float v_new_scale = 0.0f;
        if (chunk_id * BT + tid < seqlen)
          v_new_scale = __expf(g_cu_last - g_cu);
        sts_f32(v_scale_smem + tid * sizeof(float), v_new_scale);
      }

      const uint32_t W_smem = smem + tma_stage * STAGE_SIZE;
      const uint32_t V_smem = W_smem + W_size;

      if (warp_id == 2) {
        // wait for V TMA and wh MMA
        mbarrier_wait(tma_mbar_addr + tma_stage * 8, tma_parity);
        mbarrier_wait(wh_mbar_addr, wh_parity);
      }
      else if (warp_id == 3) {
        // wait for previous V_new TMA store to finish
        if (elect_sync())
          cp_async_bulk_wait_group_read<0>();
      }
      bar_sync<2>(128);

      // compute v_new
      // total tile is [V_dim, BT]
      // each warp processes [32, BT]
      // each i iteration processes [16, BT] tile per warp
      for (int i = 0; i < 2; i++) {
        // each 16x256b tile corresponds to 16x8 tile of wh
        float tmp[BT / 2];
        const int t_row = warp_id * 32 + i * 16;
        tcgen05_ld<SHAPE::_16x256b, BT / 8>(tmp, t_row, 0);

        // each j iteration processes a [16, 16] tile per warp
        // V smem layout is [2, BT, V_dim/2] with swizzling
        for (int j = 0; j < BT / 16; j++) {
          uint32_t v_tmp[4];
          const uint32_t offset = (warp_id / 2) * BT * 128;
          const uint32_t s_row = j * 16 + (lane_id / 16) * 8 + (lane_id % 8);
          const uint32_t s_col = ((warp_id % 2) * 32 + i * 16 + (lane_id % 16) / 8) ^ (lane_id % 8);
          ldmatrix_trans<4>(v_tmp, V_smem + offset + s_row * 128 + s_col * 16);

          for (int k = 0; k < 4; k++) {
            // unpack V to FP32
            float v_fp32[2];
            bf16x2_to_fp32x2(v_fp32, v_tmp[k]);

            // compute v_new = v - w @ h.T
            v_fp32[0] -= tmp[j * 8 + k * 2 + 0];
            v_fp32[1] -= tmp[j * 8 + k * 2 + 1];

            // pack v_new for O kernel
            v_tmp[k] = fp32x2_to_bf16x2(v_fp32[0], v_fp32[1]);

            // scale v_new for vk MMA
            v_fp32[0] *= lds_f32(v_scale_smem + 0);
            v_fp32[1] *= lds_f32(v_scale_smem + 0);
            reinterpret_cast<uint32_t *>(tmp)[j * 4 + k] = fp32x2_to_bf16x2(v_fp32[0], v_fp32[0]);
          }

          // store v_new for O kernel to smem
          stmatrix_trans<4>(V_new_smem + offset + s_row * 128 + s_col * 16, v_tmp);
        }

        // store scaled v_new for vk MMA
        tcgen05_st<SHAPE::_16x128b, BT / 8>(a_tmem + t_row, 0, tmp);
      }
      mbarrier_arrive(v_mbar_addr);

      // issue V_new store TMA
      if (warp_id == 3 && elect_sync()) {
        const int off_t = bos + chunk_id * BT;
        tma_store_4d(&V_new_tmap, V_new_smem, 0, off_t, 0, head_id);
        cp_async_bulk_commit_group();
      }

      tma_stage = (tma_stage + 1) % NUM_STAGES;
      if (tma_stage == 0)
        tma_parity ^= 1;

      wh_parity ^= 1;
    }
  }
}

static
CUtensorMap encode_tma(void *ptr, uint64_t T, uint64_t H, uint64_t dim) {
  CUtensorMap tmap;

  // natural shape: [T, H, dim]
  // permuted shape: [H, dim/64, T, 64]
  constexpr uint32_t rank = 4;
  uint64_t globalDim[rank] = {64, T, dim / 64, H};
  uint64_t globalStrides[rank - 1] = {H * dim * sizeof(nv_bfloat16), 128, dim};
  uint32_t boxDim[rank] = {64, BT, dim / 64, 1};
  uint32_t elementStrides[rank] = {1, 1, 1, 1};

  cuTensorMapEncodeTiled(
    &tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, ptr,
    globalDim, globalStrides, boxDim, elementStrides,
    CU_TENSOR_MAP_INTERLEAVE_NONE,
    CU_TENSOR_MAP_SWIZZLE_128B,
    CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return tmap;
}

void fwd_h_v1(
  TensorView K,
  TensorView V,
  TensorView W,
  TensorView V_new,
  TensorView g_cu,
  TensorView h,
  TensorView h0,
  TensorView ht,
  TensorView cu_seqlens,
  TensorView chunk_offsets
) {
  const int T = K.size(0);
  const int N = h.size(0);

  auto K_tmap     = encode_tma(K.data_ptr(), T, Hg, K_dim);
  auto V_tmap     = encode_tma(V.data_ptr(), T, H, V_dim);
  auto W_tmap     = encode_tma(W.data_ptr(), T, H, K_dim);
  auto V_new_tmap = encode_tma(V_new.data_ptr(), T, H, V_dim);

  auto *g_cu_ptr          = reinterpret_cast<const float *>(g_cu.data_ptr());
  auto *h_ptr             = reinterpret_cast<nv_bfloat16 *>(h.data_ptr());
  auto *h0_ptr            = reinterpret_cast<const float *>(h0.data_ptr());
  auto *ht_ptr            = reinterpret_cast<      float *>(ht.data_ptr());
  auto *cu_seqlens_ptr    = reinterpret_cast<int64_t *>(cu_seqlens.data_ptr());
  auto *chunk_offsets_ptr = reinterpret_cast<int32_t *>(chunk_offsets.data_ptr());

  constexpr int NUM_STAGES = 3;
  constexpr int smem_size = STAGE_SIZE * NUM_STAGES
                          + V_size              // V_new
                          + BT * sizeof(float)  // v_scale
                          + 8 * NUM_STAGES      // TMA mbar
                          + 8 * NUM_STAGES      // MMA mbar
                          + 8 * 4               // h, v, wh, vk mbar
                          + 4;                  // tmem addr
  
  auto kernel = fwd_h_kernel_cutlass<NUM_STAGES>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  dim3 grid(H, N);
  kernel<<<TB_SIZE, grid, smem_size>>>(
    K_tmap, V_tmap, W_tmap, V_new_tmap,
    g_cu_ptr, h_ptr, h0_ptr, ht_ptr, cu_seqlens_ptr, chunk_offsets_ptr);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fwd_h_v1, fwd_h_v1);
