// cuda_h_kernel.h — tcgen05 H kernel for v5 integration
// Adapted from cuda_h_v1.cu with V_new TMA store fixed for token-indexed access
// (bos + chunk_id * BT instead of (chunk_offset + chunk_id) * BT)
#pragma once

#include "cuda_utils.h"
#include <cuda_bf16.h>
#include <cudaTypedefs.h>
#include <cstdint>

namespace hv1 {

constexpr int H = 8;
constexpr int Hg = 4;
constexpr int BT = 64;
constexpr int K_dim = 128;
constexpr int V_dim = 128;

constexpr uint32_t W_size = BT * K_dim * sizeof(nv_bfloat16);
constexpr uint32_t V_size = BT * V_dim * sizeof(nv_bfloat16);
constexpr uint32_t K_size = BT * K_dim * sizeof(nv_bfloat16);
constexpr uint32_t STAGE_SIZE = W_size + V_size + K_size;

constexpr uint32_t H_fp32_size = V_dim * K_dim * sizeof(float);
constexpr uint32_t v_scale_size = BT * sizeof(float);

constexpr int NUM_WARPS = 4 + 4 + 2;
constexpr int WARP_SIZE_HV1 = 32;
constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE_HV1;

__device__ inline int cdiv(int a, int b) { return (a + b - 1) / b; }

__device__ inline uint32_t fp32x2_to_bf16x2(float a, float b) {
  uint32_t tmp;
  asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(tmp) : "f"(b), "f"(a));
  return tmp;
}

__device__ inline void bf16x2_to_fp32x2(float *out, uint32_t data) {
  asm volatile("shl.b32 %0, %2, 16;\n"
               "and.b32 %1, %2, 0xFFFF0000;"
              : "=f"(out[0]), "=f"(out[1]) : "r"(data));
}

__device__ inline void lds_f32x4(float *data, uint32_t addr) {
  asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];"
              : "=f"(data[0]), "=f"(data[1]), "=f"(data[2]), "=f"(data[3])
              : "r"(addr));
}

__device__ inline void sts_b32x4(uint32_t addr, const uint32_t *data) {
  asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
              :: "r"(addr),
                 "r"(data[0]), "r"(data[1]), "r"(data[2]), "r"(data[3]));
}

// Compute smem size for given number of pipeline stages
inline int smem_size_for(int num_stages) {
  return STAGE_SIZE * num_stages
       + H_fp32_size         // H0
       + H_fp32_size / 2     // H (bf16)
       + V_size              // V_new
       + v_scale_size        // v_scale
       + 5 * num_stages * 8  // mbarriers
       + 8                   // h0 mbar
       + 4;                  // tmem addr
}

template <int NUM_STAGES>
__global__
__launch_bounds__(TB_SIZE, 1)
void h_kernel_v5(
  const __grid_constant__ CUtensorMap K_tmap,
  const __grid_constant__ CUtensorMap V_tmap,
  const __grid_constant__ CUtensorMap W_tmap,
  const __grid_constant__ CUtensorMap H0_tmap,
  const __grid_constant__ CUtensorMap HT_tmap,
  const __grid_constant__ CUtensorMap H_tmap,
  const __grid_constant__ CUtensorMap V_new_tmap,
  const float       *g_cu_ptr,
  const int64_t     *cu_seqlens_ptr,
  const int32_t     *chunk_offsets_ptr
) {
  const int tid = threadIdx.x;
  const int warp_id = warp_uniform(tid / WARP_SIZE_HV1);
  const int lane_id = tid % WARP_SIZE_HV1;

  const int head_id = blockIdx.x;
  const int seq_id  = blockIdx.y;

  const int bos = cu_seqlens_ptr[seq_id];
  const int eos = cu_seqlens_ptr[seq_id + 1];
  const int seqlen = eos - bos;
  const int num_chunks = cdiv(seqlen, BT);

  // set up smem
  extern __shared__ __align__(1024) char smem_ptr[];
  const uint32_t smem = __cvta_generic_to_shared(smem_ptr);
  const uint32_t H0_f32_smem = smem + NUM_STAGES * STAGE_SIZE;
  const uint32_t H_smem = H0_f32_smem + H_fp32_size;
  const uint32_t V_new_smem = H_smem + H_fp32_size / 2;

  const uint32_t v_scale_smem = V_new_smem + V_size;
  float *v_scale_smem_ptr = reinterpret_cast<float *>(smem_ptr + (v_scale_smem - smem));

  const uint32_t tma_mbar_addr      = v_scale_smem + v_scale_size;
  const uint32_t wh_in_mbar_addr    = tma_mbar_addr + NUM_STAGES * 8;
  const uint32_t wh_done_mbar_addr  = wh_in_mbar_addr + NUM_STAGES * 8;
  const uint32_t vk_in_mbar_addr    = wh_done_mbar_addr + NUM_STAGES * 8;
  const uint32_t vk_done_mbar_addr  = vk_in_mbar_addr + NUM_STAGES * 8;

  const uint32_t h0_mbar_addr = vk_done_mbar_addr + NUM_STAGES * 8;
  const uint32_t taddr        = h0_mbar_addr + 8;

  // set up tmem
  const uint32_t wh_tmem = 0;
  const uint32_t vk_tmem = wh_tmem + BT;
  const uint32_t h_tmem_base = vk_tmem + K_dim;
  const uint32_t v_tmem_base = h_tmem_base + K_dim / 2;

  if (warp_id == 0) {
    if (elect_sync()) {
      for (int i = 0; i < NUM_STAGES; i++) {
        mbarrier_init(tma_mbar_addr + i * 8, 1);
        mbarrier_init(wh_in_mbar_addr + i * 8, WARP_SIZE_HV1 * 8);
        mbarrier_init(wh_done_mbar_addr + i * 8, 1);
        mbarrier_init(vk_in_mbar_addr + i * 8, WARP_SIZE_HV1 * 8);
        mbarrier_init(vk_done_mbar_addr + i * 8, 1);
      }
      mbarrier_init(h0_mbar_addr, 1);
      fence_mbarrier_init();
    }
  }
  else if (warp_id == 1) {
    if (elect_sync()) {
      prefetch_tensormap(&H0_tmap);
      prefetch_tensormap(&W_tmap);
      prefetch_tensormap(&V_tmap);
      prefetch_tensormap(&K_tmap);
    }
  }
  __syncthreads();

  if (warp_id == NUM_WARPS - 1) {
    // TMA warp
    if (elect_sync()) {
      int stage_id = 0;
      int parity = 1;

      const int k_head_id = head_id / (H / Hg);

      tma_load_4d(H0_f32_smem, &H0_tmap, 0, 0, 0, seq_id * H + head_id, h0_mbar_addr, EVICT_FIRST);
      mbarrier_arrive_expect_tx(h0_mbar_addr, H_fp32_size);

      for (int chunk_id = 0; chunk_id < num_chunks; chunk_id++) {
        const int off_t = bos + chunk_id * BT;
        const uint32_t W_smem = smem + stage_id * STAGE_SIZE;
        const uint32_t V_smem_local = W_smem + W_size;
        const uint32_t K_smem = V_smem_local + V_size;
        const uint32_t mbar_addr = tma_mbar_addr + stage_id * 8;

        mbarrier_wait(vk_done_mbar_addr + stage_id * 8, parity);

        tma_load_4d(W_smem, &W_tmap, 0, off_t, 0, head_id, mbar_addr, EVICT_FIRST);
        tma_load_4d(V_smem_local, &V_tmap, 0, off_t, 0, head_id, mbar_addr, EVICT_FIRST);
        tma_load_4d(K_smem, &K_tmap, 0, off_t, 0, k_head_id, mbar_addr);
        mbarrier_arrive_expect_tx(mbar_addr, STAGE_SIZE);

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
      int stage_id = 0;
      int parity = 0;

      for (int chunk_id = 0; chunk_id < num_chunks; chunk_id++) {
        const uint32_t W_smem = smem + stage_id * STAGE_SIZE;
        const uint32_t V_smem_local = W_smem + W_size;
        const uint32_t K_smem = V_smem_local + V_size;

        // wh MMA: [V_dim, K_dim] x [BT, K_dim] -> [V_dim, BT]
        constexpr uint32_t wh_idesc = make_tcgen05_idesc(V_dim, BT) | (1U << 13U);
        constexpr uint64_t w_desc_base = (desc_encode(8 * 128) << 32ULL)
                                       | (1ULL << 46ULL) | (2ULL << 61ULL);

        mbarrier_wait(tma_mbar_addr + stage_id * 8, parity);
        mbarrier_wait(wh_in_mbar_addr + stage_id * 8, parity);
        tcgen05_fence_after_thread_sync();

        for (int i = 0; i < K_dim / 64; i++)
          for (int j = 0; j < 64 / 16; j++) {
            const int h_tmem = h_tmem_base + i * 32 + j * 8;
            const uint64_t w_desc = w_desc_base | ((W_smem + i * BT * 128 + j * 32) >> 4);
            tcgen05_mma_tmem(wh_tmem, h_tmem, w_desc, wh_idesc, 1);
          }
        tcgen05_commit(wh_done_mbar_addr + stage_id * 8);

        // vk MMA: [V_dim, BT] x [BT, K_dim] -> [V_dim, K_dim]
        constexpr uint32_t vk_idesc = make_tcgen05_idesc(V_dim, K_dim) | (1U << 16U);
        constexpr uint64_t k_desc_base = (desc_encode(BT * 128) << 16ULL)
                                       | (desc_encode(8 * 128) << 32ULL)
                                       | (1ULL << 46ULL) | (2ULL << 61ULL);

        mbarrier_wait(vk_in_mbar_addr + stage_id * 8, parity);
        tcgen05_fence_after_thread_sync();

        for (int k = 0; k < BT / 16; k++) {
          const int v_tmem = v_tmem_base + k * 8;
          const uint64_t k_desc = k_desc_base | ((K_smem + k * 16 * 128) >> 4);
          tcgen05_mma_tmem(vk_tmem, v_tmem, k_desc, vk_idesc, 1);
        }
        tcgen05_commit(vk_done_mbar_addr + stage_id * 8);

        stage_id = (stage_id + 1) % NUM_STAGES;
        if (stage_id == 0)
          parity ^= 1;
      }
    }
  }
  else if (warp_id >= 4) {
    // CUDA H warps (4 warps: warp_id 4..7)
    const int tid_ = tid % 128;
    const int warp_id_ = warp_id % 4;

    const int chunk_offset = chunk_offsets_ptr[seq_id];

    int stage_id = 0;
    int vk_stage_id = 0;
    int vk_parity = 0;

    auto process = [&](int chunk_id) {
      float h_scale;
      if (lane_id == 0) {
        const int last_idx = min(bos + (chunk_id + 1) * BT, eos) - 1;
        h_scale = __expf(g_cu_ptr[last_idx * H + head_id]);
      }
      h_scale = warp_uniform(h_scale);

      if (chunk_id == 0) {
        if (warp_id_ == 0)
          mbarrier_wait(h0_mbar_addr, 0);
        bar_sync<1>(128);
      }
      else {
        if (warp_id_ == 0) {
          mbarrier_wait(vk_done_mbar_addr + vk_stage_id * 8, vk_parity);
          vk_stage_id = (vk_stage_id + 1) % NUM_STAGES;
          if (vk_stage_id == 0)
            vk_parity ^= 1;
        }
        else if (warp_id_ == 3) {
          if (elect_sync())
            cp_async_bulk_wait_group_read<0>();
        }
        bar_sync<1>(128);
        tcgen05_fence_after_thread_sync();
      }

      // convert H to BF16 for wh MMA input and store to gmem smem for O kernel
      for (int i = 0; i < K_dim / 32; i++) {
        float h_f32[32];
        uint32_t tmp[16];

        if (chunk_id == 0) {
          for (int j = 0; j < 32 / 4; j++) {
            const int col = j ^ (tid_ % 8);
            const int addr = H0_f32_smem + i * V_dim * 128 + tid_ * 128 + col * 16;
            lds_f32x4(h_f32 + j * 4, addr);
          }
        }
        else {
          tcgen05_ld<SHAPE::_32x32b, 32>(h_f32, warp_id_ * 32, vk_tmem + i * 32);
        }

        for (int j = 0; j < 16; j++)
          tmp[j] = fp32x2_to_bf16x2(h_f32[j * 2], h_f32[j * 2 + 1]);

        tcgen05_st<SHAPE::_32x32b, 16>(warp_id_ * 32, h_tmem_base + i * 16, tmp);

        // H smem for O kernel: [K_dim/64, V_dim, 64]
        for (int j = 0; j < 32 / 8; j++) {
          const int col = ((i % 2) * 4 + j) ^ (tid_ % 8);
          const int addr = H_smem + (i / 2) * V_dim * 128 + tid_ * 128 + col * 16;
          sts_b32x4(addr, tmp + j * 4);
        }
      }
      tcgen05_wait_st();
      tcgen05_fence_before_thread_sync();
      mbarrier_arrive(wh_in_mbar_addr + stage_id * 8);

      // scale H for vk MMA acc
      for (int i = 0; i < K_dim / 32; i++) {
        float h_f32[32];

        if (chunk_id == 0) {
          for (int j = 0; j < 32 / 4; j++) {
            const int col = j ^ (tid_ % 8);
            const int addr = H0_f32_smem + i * V_dim * 128 + tid_ * 128 + col * 16;
            lds_f32x4(h_f32 + j * 4, addr);
          }
        }
        else {
          tcgen05_ld<SHAPE::_32x32b, 32>(h_f32, warp_id_ * 32, vk_tmem + i * 32);
        }

        for (int j = 0; j < 32; j++)
          h_f32[j] *= h_scale;
        tcgen05_st<SHAPE::_32x32b, 32>(warp_id_ * 32, vk_tmem + i * 32, h_f32);
      }
      tcgen05_wait_st();
      tcgen05_fence_before_thread_sync();
      mbarrier_arrive(vk_in_mbar_addr + stage_id * 8);

      bar_sync<1>(128);
      asm volatile("fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster;");
      if (warp_id_ == 3 && elect_sync()) {
        tma_store_4d(&H_tmap, H_smem, 0, 0, 0, (chunk_offset + chunk_id) * H + head_id);
        cp_async_bulk_commit_group();
      }

      stage_id = (stage_id + 1) % NUM_STAGES;
    };

    process(0);
    for (int chunk_id = 1; chunk_id < num_chunks; chunk_id++)
      process(chunk_id);

    // store final H
    if (warp_id_ == 0)
      mbarrier_wait(vk_done_mbar_addr + vk_stage_id * 8, vk_parity);
    bar_sync<1>(128);
    tcgen05_fence_after_thread_sync();

    for (int i = 0; i < K_dim / 32; i++) {
      float h_f32[32];
      tcgen05_ld<SHAPE::_32x32b, 32>(h_f32, warp_id_ * 32, vk_tmem + i * 32);

      for (int j = 0; j < 32 / 4; j++) {
        const int col = j ^ (tid_ % 8);
        const int addr = H0_f32_smem + i * V_dim * 128 + tid_ * 128 + col * 16;
        sts_b32x4(addr, reinterpret_cast<const uint32_t *>(h_f32 + j * 4));
      }
    }
    bar_sync<1>(128);

    if (warp_id_ == 0) {
      if (elect_sync()) {
        tma_store_4d(&HT_tmap, H0_f32_smem, 0, 0, 0, seq_id * H + head_id);
        cp_async_bulk_commit_group();
      }
    }
    else if (warp_id_ == 1) {
      tcgen05_dealloc(0, 512);
    }
  }
  else {
    // V CUDA warps (4 warps: warp_id 0..3)
    int stage_id = 0;
    int parity = 0;

    for (int chunk_id = 0; chunk_id < num_chunks; chunk_id++) {
      const uint32_t W_smem = smem + stage_id * STAGE_SIZE;
      const uint32_t V_smem_local = W_smem + W_size;
      const uint32_t K_smem = V_smem_local + V_size;

      if (warp_id == 0)
        mbarrier_wait(tma_mbar_addr + stage_id * 8, parity);
      bar_sync<2>(128);

      // unpack V from BF16->FP32, then store as acc for wh MMA
      for (int i = 0; i < BT / 8; i++) {
        uint32_t v_bf16[4];
        const uint32_t offset = (warp_id / 2) * BT * 128;
        const uint32_t s_row = i * 8 + (lane_id % 8);
        const uint32_t s_col = ((warp_id % 2) * 4 + (lane_id / 8)) ^ (lane_id % 8);
        ldmatrix_trans<4>(v_bf16, V_smem_local + offset + s_row * 128 + s_col * 16);

        float v_fp32[8];
        for (int k = 0; k < 4; k++)
          bf16x2_to_fp32x2(v_fp32 + k * 2, v_bf16[k]);

        tcgen05_st<SHAPE::_16x256b, 1>(warp_id * 32 +  0, wh_tmem + i * 8, v_fp32 + 0);
        tcgen05_st<SHAPE::_16x256b, 1>(warp_id * 32 + 16, wh_tmem + i * 8, v_fp32 + 4);
      }
      tcgen05_wait_st();
      tcgen05_fence_before_thread_sync();
      mbarrier_arrive(wh_in_mbar_addr + stage_id * 8);

      // load g_cu and compute scaling for v_new
      if (tid < BT) {
        const int last_idx = min(bos + (chunk_id + 1) * BT, eos) - 1;
        const int t = bos + chunk_id * BT + tid;
        float g_cu = g_cu_ptr[t * H + head_id];
        float g_cu_last = g_cu_ptr[last_idx * H + head_id];

        float v_new_scale = 0.0f;
        if (t < eos)
          v_new_scale = __expf(g_cu_last - g_cu);
        v_scale_smem_ptr[tid] = v_new_scale;
      }

      if (warp_id == 2) {
        mbarrier_wait(wh_done_mbar_addr + stage_id * 8, parity);
      }
      else if (warp_id == 3) {
        if (elect_sync())
          cp_async_bulk_wait_group_read<0>();
      }
      bar_sync<2>(128);
      tcgen05_fence_after_thread_sync();

      // compute v_new
      for (int i = 0; i < BT / 8; i++) {
        float tmp[8];
        tcgen05_ld<SHAPE::_16x256b, 1>(tmp + 0, warp_id * 32 +  0, wh_tmem + i * 8);
        tcgen05_ld<SHAPE::_16x256b, 1>(tmp + 4, warp_id * 32 + 16, wh_tmem + i * 8);

        uint32_t v_tmp[4];
        float2 v_scale = reinterpret_cast<float2 *>(v_scale_smem_ptr + (i * 8 + (lane_id % 4) * 2))[0];

        for (int k = 0; k < 4; k++) {
          v_tmp[k] = fp32x2_to_bf16x2(tmp[k * 2 + 0], tmp[k * 2 + 1]);
          reinterpret_cast<uint32_t *>(tmp)[k] = fp32x2_to_bf16x2(tmp[k * 2 + 0] * v_scale.x,
                                                                    tmp[k * 2 + 1] * v_scale.y);
        }

        const uint32_t offset = (warp_id / 2) * BT * 128;
        const uint32_t s_row = i * 8 + (lane_id % 8);
        const uint32_t s_col = ((warp_id % 2) * 4 + (lane_id / 8)) ^ (lane_id % 8);
        stmatrix_trans<4>(V_new_smem + offset + s_row * 128 + s_col * 16, v_tmp);

        tcgen05_st<SHAPE::_16x128b, 1>(warp_id * 32 +  0, v_tmem_base + i * 4, tmp + 0);
        tcgen05_st<SHAPE::_16x128b, 1>(warp_id * 32 + 16, v_tmem_base + i * 4, tmp + 2);
      }

      tcgen05_wait_st();
      tcgen05_fence_before_thread_sync();
      mbarrier_arrive(vk_in_mbar_addr + stage_id * 8);

      bar_sync<2>(128);
      asm volatile("fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster;");
      if (warp_id == 3 && elect_sync()) {
        // V_new TMA store: TOKEN-INDEXED (bos + chunk_id * BT)
        // NOT chunk-indexed ((chunk_offset + chunk_id) * BT)
        // This is needed because v5's OOutputKernel uses token-indexed v_new access
        tma_store_4d(&V_new_tmap, V_new_smem, 0, bos + chunk_id * BT, 0, head_id);
        cp_async_bulk_commit_group();
      }

      stage_id = (stage_id + 1) % NUM_STAGES;
      if (stage_id == 0)
        parity ^= 1;
    }
  }
}

// TMA descriptor for [T, H, dim] layout with 128B swizzle
static inline
CUtensorMap encode_tma(void *ptr, uint64_t T, uint64_t H_, uint64_t dim) {
  CUtensorMap tmap;
  constexpr uint32_t rank = 4;
  uint64_t globalDim[rank] = {64, T, dim / 64, H_};
  uint64_t globalStrides[rank - 1] = {H_ * dim * sizeof(nv_bfloat16),
                                                                128,
                                           dim * sizeof(nv_bfloat16)};
  uint32_t boxDim[rank] = {64, BT, (uint32_t)(dim / 64), 1};
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

// TMA descriptor for H state: [N, H, V_dim, K_dim] with 128B swizzle
static inline
CUtensorMap encode_h_tma(void *ptr, uint64_t N, CUtensorMapDataType dtype) {
  CUtensorMap tmap;
  int elem_width = (dtype == CU_TENSOR_MAP_DATA_TYPE_FLOAT32) ? 4 : 2;
  const int num_elems = 128 / elem_width;

  constexpr uint32_t rank = 4;
  uint64_t globalDim[rank] = {(uint64_t)num_elems, V_dim, K_dim / (uint64_t)num_elems, N * H};
  uint64_t globalStrides[rank - 1] = {        K_dim * (uint64_t)elem_width,
                                                             128,
                                      V_dim * K_dim * (uint64_t)elem_width};
  uint32_t boxDim[rank] = {(uint32_t)num_elems, V_dim, K_dim / (uint32_t)num_elems, 1};
  uint32_t elementStrides[rank] = {1, 1, 1, 1};

  cuTensorMapEncodeTiled(
    &tmap, dtype, rank, ptr,
    globalDim, globalStrides, boxDim, elementStrides,
    CU_TENSOR_MAP_INTERLEAVE_NONE,
    CU_TENSOR_MAP_SWIZZLE_128B,
    CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return tmap;
}

} // namespace hv1
