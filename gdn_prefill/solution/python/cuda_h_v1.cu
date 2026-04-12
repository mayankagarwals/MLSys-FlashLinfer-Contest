// for chunk_id in range(num_chunks):
//     # TMA warp
//     load w, v, k from gmem->smem
//     w - [BT, K_dim]
//     v - [BT, V_dim]
//     k - [BT, K_dim]
//
//     # CUDA H warp
//     if 1st chunk:
//       load h from h0(gmem)
//     else:
//       wait and load h(tmem) from vk MMA
//     convert h to BF16, store to tmem (input for wh MMA)
//
//     # CUDA V warp
//     wait and load v(smem) from TMA
//     convert v to FP32, store to tmem (acc for wh MMA)
//
//     # MMA warp (wh MMA)
//     wait w(smem) from TMA
//     wait h(tmem) from CUDA H warp, and v(tmem) from CUDA V warp
//     issue MMA: v_new.T = v.T - h @ w.T - [V_dim, BT]
//
//     # CUDA H warp
//     load g_cu_last from gmem->rmem
//     compute h_scale = exp(g_cu_last)
//     if 1st chunk:
//       load h from h0(gmem)
//     else:
//       load h(tmem) from vk MMA
//     convert h to BF16, store to gmem (for O kernel)
//     compute scaled_h = h * h_scale, store to tmem (acc for vk MMA)
//
//     # CUDA V warp
//     load g_cu from gmem->rmem
//     compute v_scale = exp(g_cu_last - g_cu)
//     wait and load v_new.T(tmem) from wh MMA
//     convert v_new to BF16, store to gmem (for O kernel)
//     compute scaled_v_new = v_new * v_scale, store to tmem (input for vk MMA)
//
//     # MMA warp (vk MMA)
//     wait k(smem) from TMA
//     wait scaled_h(tmem) from CUDA H warp, and scaled_v_new(tmem) from CUDA V warp
//     issue MMA: h_new = scaled_h + scaled_v_new.T @ k - [V_dim, K_dim]

#include <cuda_bf16.h>
#include <cudaTypedefs.h>
#include <cstdint>
#include "cuda_utils.h"

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
  asm volatile("shl.b32 %0, %2, 16;\n"        // low 16-bit
               "and.b32 %1, %2, 0xFFFF0000;"  // high 16-bit
              : "=f"(out[0]), "=f"(out[1]) : "r"(data));
}

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
constexpr int WARP_SIZE = 32;
constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;

__device__ inline
void lds_f32x4(float *data, uint32_t addr) {
  asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];"
              : "=f"(data[0]), "=f"(data[1]), "=f"(data[2]), "=f"(data[3])
              : "r"(addr));
};

__device__ inline
void sts_b32x4(uint32_t addr, const uint32_t *data) {
  asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
              :: "r"(addr),
                 "r"(data[0]), "r"(data[1]), "r"(data[2]), "r"(data[3]));
};

enum ProfilerTag {
  START = 0,
  SETUP,
  WAIT_MMA,
  WAIT_TMA,
  WAIT_H0,
  WAIT_WH_IN,
  WAIT_VK_IN,
  WAIT_WH_MMA,
  WAIT_VK_MMA,
  ISSUE_TMA,
  ISSUE_WH_MMA,
  ISSUE_VK_MMA,
  // H warps
  COMPUTE_H_SCALE,
  PROCESS_H,
  PROCESS_SCALED_H,
  STORE_HT,
  // V warps
  COMPUTE_V_SCALE,
  PROCESS_V,
  PROCESS_SCALED_V,
  END,
};

__device__ inline
int64_t globaltimer() {
  int64_t t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t) :: "memory");
  return t;
}

// layout: [NUM_SMS][NUM_WARPS][1 + num_entries * 2]
template <bool ENABLE, int NUM_WARPS>
struct Profiler {
  int64_t *data_ptr_;
  int cnt_;
  int64_t now_;

  __device__
  void init(int64_t *data_ptr, int num_entries) {
    if (!ENABLE) return;
    int sm_id;
    asm volatile("mov.u32 %0, %smid;\n" : "=r"(sm_id));
    int warp_id = threadIdx.x / WARP_SIZE;

    data_ptr_ = data_ptr + (sm_id * NUM_WARPS + warp_id) * (1 + num_entries * 2);
    cnt_ = data_ptr_[0];
    stamp(START);
  }

  __device__
  void stamp(ProfilerTag tag) {
    if (!ENABLE) return;
    data_ptr_[1 + cnt_ * 2 + 0] = tag;
    data_ptr_[1 + cnt_ * 2 + 1] = globaltimer();
    cnt_++;
  }

  __device__
  void flush() {
    if (!ENABLE) return;
    stamp(END);
    data_ptr_[0] = cnt_;
  }
};

template <int NUM_STAGES, bool DO_PROFILE>
__global__
__block_size__((TB_SIZE, 1, 1))
void h_kernel_cutlass(
  const __grid_constant__ CUtensorMap K_tmap,      // [total_T, Hg, K_dim]
  const __grid_constant__ CUtensorMap V_tmap,      // [total_T, H, V_dim]
  const __grid_constant__ CUtensorMap W_tmap,      // [total_T, H, K_dim]
  const __grid_constant__ CUtensorMap H0_tmap,     // [N, H, V_dim, K_dim]
  const __grid_constant__ CUtensorMap H_tmap,      // [total_num_chunks, H, V_dim, K_dim]
  const __grid_constant__ CUtensorMap V_new_tmap,  // [total_num_chunks, BT, H, V_dim]
  const float       *g_cu_ptr,                     // [total_T, H]
        float       *ht_ptr,                       // [N, H, V_dim, K_dim]
  const int64_t     *cu_seqlens_ptr,               // [N+1]
  const int32_t     *chunk_offsets_ptr,            // [N]
        int64_t     *profiler_ptr,                 // [NUM_SMS][NUM_WARPS][1 + num_entries * 2]
        int          num_entries
) {
  const int tid = threadIdx.x;
  const int warp_id = warp_uniform(tid / WARP_SIZE);
  const int lane_id = tid % WARP_SIZE;

  const int head_id = blockIdx.x;
  const int seq_id  = blockIdx.y;

  Profiler<DO_PROFILE, NUM_WARPS> profiler;
  profiler.init(profiler_ptr, num_entries);

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

  constexpr int nregs_lo = 24;
  constexpr int nregs_hi = 240;

  if (warp_id == 0) {
    // init mbar
    if (elect_sync()) {
      for (int i = 0; i < NUM_STAGES; i++) {
        mbarrier_init(tma_mbar_addr + i * 8, 1);                // 1 TMA
        mbarrier_init(wh_in_mbar_addr + i * 8, WARP_SIZE * 8);  // h from H warps and v from V warps
        mbarrier_init(wh_done_mbar_addr + i * 8, 1);            // 1 MMA
        mbarrier_init(vk_in_mbar_addr + i * 8, WARP_SIZE * 8);  // scaled_h from H warps and scaled_v from V warps
        mbarrier_init(vk_done_mbar_addr + i * 8, 1);            // 1 MMA
      }
      mbarrier_init(h0_mbar_addr, 1);
      fence_mbarrier_init();
    }
  }
  else if (warp_id == 1) {
    // prefetch TMA descriptor
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
      profiler.stamp(SETUP);
      int stage_id = 0;
      int parity = 1;

      const int k_head_id = head_id / (H / Hg);

      // natural shape:  [N, H, V_dim, K_dim]
      // permuted shape: [N*H, K_dim/32, V_dim, 32]
      tma_load_4d(H0_f32_smem, &H0_tmap, 0, 0, 0, seq_id * H + head_id, h0_mbar_addr, EVICT_FIRST);
      mbarrier_arrive_expect_tx(h0_mbar_addr, H_fp32_size);

      for (int chunk_id = 0; chunk_id < num_chunks; chunk_id++) {
        // compute addresses
        const int off_t = bos + chunk_id * BT;
        const uint32_t W_smem = smem + stage_id * STAGE_SIZE;
        const uint32_t V_smem = W_smem + W_size;
        const uint32_t K_smem = V_smem + V_size;
        const uint32_t mbar_addr = tma_mbar_addr + stage_id * 8;
  
        // wait MMA warp to release the buffer
        mbarrier_wait(vk_done_mbar_addr + stage_id * 8, parity);
        profiler.stamp(WAIT_MMA);

        // issue TMA and arrive
        // natural shape: [T, H, dim]
        // permute shape: [H, dim/64, T, 64]
        tma_load_4d(W_smem, &W_tmap, 0, off_t, 0, head_id, mbar_addr, EVICT_FIRST);
        tma_load_4d(V_smem, &V_tmap, 0, off_t, 0, head_id, mbar_addr, EVICT_FIRST);
        tma_load_4d(K_smem, &K_tmap, 0, off_t, 0, k_head_id, mbar_addr);
        mbarrier_arrive_expect_tx(mbar_addr, STAGE_SIZE);
        profiler.stamp(ISSUE_TMA);

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
      profiler.stamp(SETUP);
      int stage_id = 0;
      int parity = 0;

      for (int chunk_id = 0; chunk_id < num_chunks; chunk_id++) {
        const uint32_t W_smem = smem + stage_id * STAGE_SIZE;
        const uint32_t V_smem = W_smem + W_size;
        const uint32_t K_smem = V_smem + V_size;

        // wh MMA
        // H in tmem, W in smem. both are K-major
        // [V_dim, K_dim] x [BT, K_dim] -> [V_dim, BT]
        constexpr uint32_t wh_idesc = make_tcgen05_idesc(V_dim, BT) | (1U << 13U);  // negate A

        // 128B swizzling
        constexpr uint64_t w_desc_base = (desc_encode(8 * 128) << 32ULL)  // SBO
                                       | (1ULL << 46ULL) | (2ULL << 61ULL);

        // wait for TMA
        mbarrier_wait(tma_mbar_addr + stage_id * 8, parity);
        profiler.stamp(WAIT_TMA);

        // wait for h(tmem, input) and v(tmem, acc)
        mbarrier_wait(wh_in_mbar_addr + stage_id * 8, parity);
        profiler.stamp(WAIT_WH_IN);
        tcgen05_fence_after_thread_sync();

        // i selects the [V_dim/BT, 64] tile (increment 32 tmem columns for H, increment by BT x 128B for W)
        // j selects the [V_dim/BT, 16] tile (increment 8  tmem columns for H, increment by 32B due to swizzling for W)
        for (int i = 0; i < K_dim / 64; i++)
          for (int j = 0; j < 64 / 16; j++) {
            const int h_tmem = h_tmem_base + i * 32 + j * 8;
            const uint64_t w_desc = w_desc_base | ((W_smem + i * BT * 128 + j * 32) >> 4);
            tcgen05_mma_tmem(wh_tmem, h_tmem, w_desc, wh_idesc, 1);  // always enable input d
          }
        tcgen05_commit(wh_done_mbar_addr + stage_id * 8);
        profiler.stamp(ISSUE_WH_MMA);

        // vk MMA
        // scaled v_new in tmem, K in smem. K is MN-major
        // [V_dim, BT] x [BT, K_dim] -> [V_dim, K_dim]
        constexpr uint32_t vk_idesc = make_tcgen05_idesc(V_dim, K_dim) | (1U << 16U);  // transpose B

        // MN-major, 128B swizzling
        constexpr uint64_t k_desc_base = (desc_encode(BT * 128) << 16ULL)  // LBO
                                       | (desc_encode(8 * 128) << 32ULL)   // SBO
                                       | (1ULL << 46ULL) | (2ULL << 61ULL);

        // wait for scaled_h(tmem, acc) and scaled_v_new(tmem, input)
        mbarrier_wait(vk_in_mbar_addr + stage_id * 8, parity);
        profiler.stamp(WAIT_VK_IN);
        tcgen05_fence_after_thread_sync();

        // k selects [V_dim/K_dim, 16] tile
        for (int k = 0; k < BT / 16; k++) {
          const int v_tmem = v_tmem_base + k * 8;
          const uint64_t k_desc = k_desc_base | ((K_smem + k * 16 * 128) >> 4);
          tcgen05_mma_tmem(vk_tmem, v_tmem, k_desc, vk_idesc, 1);  // always enable input d
        }
        tcgen05_commit(vk_done_mbar_addr + stage_id * 8);
        profiler.stamp(ISSUE_VK_MMA);

        stage_id = (stage_id + 1) % NUM_STAGES;
        if (stage_id == 0)
          parity ^= 1;
      }
    }
  }
  else if (warp_id >= 4) {
    // CUDA H warps
    if (elect_sync()) profiler.stamp(SETUP);
    const int tid_ = tid % 128;
    const int warp_id_ = warp_id % 4;

    const int chunk_offset = chunk_offsets_ptr[seq_id];

    // we need separate "normal" stage_id and vk_stage_id
    // since vk_stage_id is from the previous iteration
    int stage_id = 0;
    int vk_stage_id = 0;
    int vk_parity = 0;

    auto process = [&](int chunk_id) {
      // load g_cu_last and compute scaling for H
      // putting this after PROCESS_H is slower
      float h_scale;
      if (lane_id == 0) {
        const int last_idx = min(bos + (chunk_id + 1) * BT, eos) - 1;
        h_scale = __expf(g_cu_ptr[last_idx * H + head_id]);
      }
      if (elect_sync()) profiler.stamp(COMPUTE_H_SCALE);

      // for chunk_id > 0, wait for vk MMA to update H
      if (chunk_id == 0) {
        if (warp_id_ == 0)
          mbarrier_wait(h0_mbar_addr, 0);
        bar_sync<1>(128);
        if (elect_sync()) profiler.stamp(WAIT_H0);
      }
      else {
        if (warp_id_ == 0) {
          mbarrier_wait(vk_done_mbar_addr + vk_stage_id * 8, vk_parity);
          vk_stage_id = (vk_stage_id + 1) % NUM_STAGES;
          if (vk_stage_id == 0)
            vk_parity ^= 1;
        }
        bar_sync<1>(128);
        tcgen05_fence_after_thread_sync();
        if (elect_sync()) profiler.stamp(WAIT_VK_MMA);
      }

      // prepare H input for wh MMA to unblock it ASAP
      // collectively, 4 warps represent H[V_dim, K_dim]
      // each thread processes 1 row
      for (int i = 0; i < K_dim / 32; i++) {
        float h_f32[32];
        uint32_t tmp[16];

        if (chunk_id == 0) {
          // load H0 from smem for 1st chunk
          // natural shape:  [N, H, V_dim, K_dim]
          // permuted shape: [N, H, K_dim/32, V_dim, 32]
          // box shape:      [K_dim/32, V_dim, 32] with swizzling
          for (int j = 0; j < 32 / 4; j++) {
            const int col = j ^ (tid_ % 8);
            const int addr = H0_f32_smem + i * V_dim * 128 + tid_ * 128 + col * 16;
            lds_f32x4(h_f32 + j * 4, addr);
          }
        }
        else {
          // for subsequent chunks, load from tmem
          tcgen05_ld<SHAPE::_32x32b, 32>(h_f32, warp_id_ * 32, vk_tmem + i * 32);
        }

        // pack to BF16 for wh MMA
        for (int j = 0; j < 16; j++)
          tmp[j] = fp32x2_to_bf16x2(h_f32[j * 2], h_f32[j * 2 + 1]);
        tcgen05_st<SHAPE::_32x32b, 16>(warp_id_ * 32, h_tmem_base + i * 16, tmp);
      }
      tcgen05_wait_st();
      tcgen05_fence_before_thread_sync();
      mbarrier_arrive(wh_in_mbar_addr + stage_id * 8);
      if (elect_sync()) profiler.stamp(PROCESS_H);

      // drain H TMA store of the previous iteration before overwriting its smem
      if (warp_id_ == 0 && elect_sync())
        cp_async_bulk_wait_group_read<0>();
      bar_sync<1>(128);

      // slowly prepare for O kernel and store vk MMA acc
      // TODO: for O kernel, may want to store to smem then TMA store
      for (int i = 0; i < K_dim / 32; i++) {
        float h_f32[32];
        uint32_t tmp[16];

        if (chunk_id == 0) {
          // load H0 from smem for 1st chunk
          // natural shape:  [N, H, V_dim, K_dim]
          // permuted shape: [N, H, K_dim/32, V_dim, 32]
          // box shape:      [K_dim/32, V_dim, 32] with swizzling
          for (int j = 0; j < 32 / 4; j++) {
            const int col = j ^ (tid_ % 8);
            const int addr = H0_f32_smem + i * V_dim * 128 + tid_ * 128 + col * 16;
            lds_f32x4(h_f32 + j * 4, addr);
          }
        }
        else {
          // for subsequent chunks, load from tmem
          tcgen05_ld<SHAPE::_32x32b, 32>(h_f32, warp_id_ * 32, vk_tmem + i * 32);
        }

        // pack to BF16 for O kernel
        for (int j = 0; j < 16; j++)
          tmp[j] = fp32x2_to_bf16x2(h_f32[j * 2], h_f32[j * 2 + 1]);

        // H smem layout: [K_dim/64, V_dim, 64]
        for (int j = 0; j < 32 / 8; j++) {
          const int col = ((i % 2) * 4 + j) ^ (tid_ % 8);
          const int addr = H_smem + (i / 2) * V_dim * 128 + tid_ * 128 + col * 16;
          sts_b32x4(addr, tmp + j * 4);
        }

        // scaled H for vk MMA
        for (int j = 0; j < 32; j++)
          h_f32[j] *= h_scale;
        tcgen05_st<SHAPE::_32x32b, 32>(warp_id_ * 32, vk_tmem + i * 32, h_f32);  // for vk MMA
      }
      tcgen05_wait_st();
      tcgen05_fence_before_thread_sync();
      mbarrier_arrive(vk_in_mbar_addr + stage_id * 8);
      if (elect_sync()) profiler.stamp(PROCESS_SCALED_H);

      bar_sync<1>(128);
      asm volatile("fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster;");
      if (warp_id_ == 0 && elect_sync()) {
        // natural shape:  [total_num_chunks, H, V_dim, K_dim]
        // permuted shape: [total_num_chunks*H, K_dim/64, V_dim, 64]
        tma_store_4d(&H_tmap, H_smem, 0, 0, 0, (chunk_offset + chunk_id) * H + head_id);
        cp_async_bulk_commit_group();
      }

      stage_id = (stage_id + 1) % NUM_STAGES;
    };

    process(0);
    for (int chunk_id = 1; chunk_id < num_chunks; chunk_id++)
      process(chunk_id);

    // store final H
    // wait for vk MMA
    if (warp_id_ == 0)
      mbarrier_wait(vk_done_mbar_addr + vk_stage_id * 8, vk_parity);
    bar_sync<1>(128);
    tcgen05_fence_after_thread_sync();
    if (elect_sync()) profiler.stamp(WAIT_VK_MMA);

    for (int i = 0; i < K_dim / 8; i++) {
      float h_f32[8];
      tcgen05_ld<SHAPE::_32x32b, 8>(h_f32, warp_id_ * 32, vk_tmem + i * 8);

      const int offset = (seq_id * H + head_id) * V_dim * K_dim + (tid_ * K_dim) + i * 8;
      stg_u32x8_fast(ht_ptr + offset, h_f32);
    }
    if (elect_sync()) profiler.stamp(STORE_HT);

    if (warp_id_ == 0)
      tcgen05_dealloc(0, 512);
  }
  else {
    // V CUDA warps
    if (elect_sync()) profiler.stamp(SETUP);
    int stage_id = 0;
    int parity = 0;

    const int chunk_offset = chunk_offsets_ptr[seq_id];

    for (int chunk_id = 0; chunk_id < num_chunks; chunk_id++) {
      const uint32_t W_smem = smem + stage_id * STAGE_SIZE;
      const uint32_t V_smem = W_smem + W_size;
      const uint32_t K_smem = V_smem + V_size;

      // wait for V
      if (warp_id == 0)
        mbarrier_wait(tma_mbar_addr + stage_id * 8, parity);
      bar_sync<2>(128);
      if (elect_sync()) profiler.stamp(WAIT_TMA);

      // unpack V from BF16->FP32, then store as acc for wh MMA
      // total tile:  [V_dim, BT]
      // each warp:   [32, BT]
      // each thread: [1, BT]
      for (int i = 0; i < BT / 8; i++) {
        // V smem layout: [V_dim/64, BT, 64] = [2, BT, 64]
        // each ldmatrix loads [8, 32] along the last 2 dims
        uint32_t v_bf16[4];
        const uint32_t offset = (warp_id / 2) * BT * 128;
        const uint32_t s_row = i * 8 + (lane_id % 8);
        const uint32_t s_col = ((warp_id % 2) * 4 + (lane_id / 8)) ^ (lane_id % 8);
        ldmatrix_trans<4>(v_bf16, V_smem + offset + s_row * 128 + s_col * 16);

        float v_fp32[8];
        for (int k = 0; k < 4; k++)
          bf16x2_to_fp32x2(v_fp32 + k * 2, v_bf16[k]);

        tcgen05_st<SHAPE::_16x256b, 1>(warp_id * 32 +  0, wh_tmem + i * 8, v_fp32 + 0);
        tcgen05_st<SHAPE::_16x256b, 1>(warp_id * 32 + 16, wh_tmem + i * 8, v_fp32 + 4);
      }
      tcgen05_wait_st();
      tcgen05_fence_before_thread_sync();
      mbarrier_arrive(wh_in_mbar_addr + stage_id * 8);
      if (elect_sync()) profiler.stamp(PROCESS_V);

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
        if (elect_sync()) profiler.stamp(COMPUTE_V_SCALE);
      }

      if (warp_id == 2) {
        // wait for wh MMA
        mbarrier_wait(wh_done_mbar_addr + stage_id * 8, parity);
      }
      else if (warp_id == 3) {
        // drain V_new TMA store in the previous iteration
        if (elect_sync())
          cp_async_bulk_wait_group_read<0>();
      }
      bar_sync<2>(128);
      tcgen05_fence_after_thread_sync();
      if (elect_sync()) profiler.stamp(WAIT_WH_MMA);

      // compute v_new
      // total tile:  [V_dim, BT]
      // each warp:   [32, BT]
      // each thread: [1, BT]
      for (int i = 0; i < BT / 8; i++) {
        float tmp[8];
        tcgen05_ld<SHAPE::_16x256b, 1>(tmp + 0, warp_id * 32 +  0, wh_tmem + i * 8);
        tcgen05_ld<SHAPE::_16x256b, 1>(tmp + 4, warp_id * 32 + 16, wh_tmem + i * 8);

        uint32_t v_tmp[4];
        float2 v_scale = reinterpret_cast<float2 *>(v_scale_smem_ptr + (i * 8 + (lane_id % 4) * 2))[0];

        for (int k = 0; k < 4; k++) {
          // pack v_new for O kernel
          v_tmp[k] = fp32x2_to_bf16x2(tmp[k * 2 + 0], tmp[k * 2 + 1]);

          // scale v_new for vk MMA
          reinterpret_cast<uint32_t *>(tmp)[k] = fp32x2_to_bf16x2(tmp[k * 2 + 0] * v_scale.x,
                                                                  tmp[k * 2 + 1] * v_scale.y);
        }

        // store v_new for O kernel to smem
        // this mirrors ldmatrix for V
        // V smem layout: [V_dim/64, BT, 64] = [2, BT, 64]
        // each stmatrix stores [8, 32] along the last 2 dims
        const uint32_t offset = (warp_id / 2) * BT * 128;
        const uint32_t s_row = i * 8 + (lane_id % 8);
        const uint32_t s_col = ((warp_id % 2) * 4 + (lane_id / 8)) ^ (lane_id % 8);
        stmatrix_trans<4>(V_new_smem + offset + s_row * 128 + s_col * 16, v_tmp);

        // store scaled v_new for vk MMA
        tcgen05_st<SHAPE::_16x128b, 1>(warp_id * 32 +  0, v_tmem_base + i * 4, tmp + 0);
        tcgen05_st<SHAPE::_16x128b, 1>(warp_id * 32 + 16, v_tmem_base + i * 4, tmp + 2);
      }

      tcgen05_wait_st();
      tcgen05_fence_before_thread_sync();
      mbarrier_arrive(vk_in_mbar_addr + stage_id * 8);
      if (elect_sync()) profiler.stamp(PROCESS_SCALED_V);

      // all warps finish storing scaled v_new to smem
      bar_sync<2>(128);
      asm volatile("fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster;");
      if (warp_id == 3 && elect_sync()) {
        // natural shape:  [total_num_chunks*BT, H, V_dim]
        // permuted shape: [H, V_dim/64, total_num_chunks*BT, 64]
        tma_store_4d(&V_new_tmap, V_new_smem, 0, (chunk_offset + chunk_id) * BT, 0, head_id);
        cp_async_bulk_commit_group();
      }

      stage_id = (stage_id + 1) % NUM_STAGES;
      if (stage_id == 0)
        parity ^= 1;
    }
  }
  if (elect_sync()) profiler.flush();
}

static
CUtensorMap encode_tma(void *ptr, uint64_t T, uint64_t H, uint64_t dim) {
  CUtensorMap tmap;

  // natural shape:  [T, H, dim]
  // permuted shape: [H, dim/64, T, 64]
  constexpr uint32_t rank = 4;
  uint64_t globalDim[rank] = {64, T, dim / 64, H};
  uint64_t globalStrides[rank - 1] = {H * dim * sizeof(nv_bfloat16),
                                                                128,
                                           dim * sizeof(nv_bfloat16)};  // in bytes
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

static
CUtensorMap encode_h_tma(void *ptr, uint64_t N, CUtensorMapDataType dtype) {
  CUtensorMap tmap;

  int elem_width = 0;
  if (dtype == CU_TENSOR_MAP_DATA_TYPE_FLOAT32) elem_width = 4;
  else if (dtype == CU_TENSOR_MAP_DATA_TYPE_BFLOAT16) elem_width = 2;

  const int num_elems = 128 / elem_width;

  // for FP32
  // natural shape:  [N, H, V_dim, K_dim]
  // permuted shape: [N*H, K_dim/32, V_dim, 32]
  constexpr uint32_t rank = 4;
  uint64_t globalDim[rank] = {num_elems, V_dim, K_dim / num_elems, N * H};
  uint64_t globalStrides[rank - 1] = {        K_dim * elem_width,
                                                             128,
                                      V_dim * K_dim * elem_width};  // in bytes
  uint32_t boxDim[rank] = {num_elems, V_dim, K_dim / num_elems, 1};
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

void h_v1(
  TensorView K,
  TensorView V,
  TensorView W,
  TensorView V_new,
  TensorView g_cu,
  TensorView h,
  TensorView h0,
  TensorView ht,
  TensorView cu_seqlens,
  TensorView chunk_offsets,
  ffi::Optional<TensorView> profiler
) {
  const int T = K.size(0);
  const int N = h0.size(0);

  auto K_tmap     = encode_tma(K.data_ptr(), T, Hg, K_dim);
  auto V_tmap     = encode_tma(V.data_ptr(), T, H, V_dim);
  auto W_tmap     = encode_tma(W.data_ptr(), T, H, K_dim);
  auto V_new_tmap = encode_tma(V_new.data_ptr(), 100000, H, V_dim);  // padded layout
  auto H0_tmap    = encode_h_tma(h0.data_ptr(), N, CU_TENSOR_MAP_DATA_TYPE_FLOAT32);
  auto H_tmap     = encode_h_tma(h.data_ptr(), 100000, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16);

  auto *V_new_ptr         = reinterpret_cast<nv_bfloat16 *>(V_new.data_ptr());
  auto *g_cu_ptr          = reinterpret_cast<const float *>(g_cu.data_ptr());
  auto *ht_ptr            = reinterpret_cast<      float *>(ht.data_ptr());
  auto *cu_seqlens_ptr    = reinterpret_cast<int64_t *>(cu_seqlens.data_ptr());
  auto *chunk_offsets_ptr = reinterpret_cast<int32_t *>(chunk_offsets.data_ptr());

  // deeper pipeline is slower?
  constexpr int NUM_STAGES = 2;
  constexpr int smem_size = STAGE_SIZE * NUM_STAGES
                          + H_fp32_size         // H0
                          + H_fp32_size / 2     // H
                          + V_size              // V_new
                          + v_scale_size        // v_scale
                          + 5 * NUM_STAGES * 8  // TMA, wh_in, wh_done, vk_in, vk_done mbar
                          + 8                   // h0
                          + 4;                  // tmem addr

  if (profiler.has_value()) {
    const int num_entries = (profiler.value().size(2) - 1) / 2;
    auto *profiler_ptr = reinterpret_cast<int64_t *>(profiler.value().data_ptr());

    auto kernel = h_kernel_cutlass<NUM_STAGES, true>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    dim3 grid(H, N);
    kernel<<<grid, TB_SIZE, smem_size>>>(
      K_tmap, V_tmap, W_tmap, H0_tmap, H_tmap, V_new_tmap,
      g_cu_ptr, ht_ptr, cu_seqlens_ptr, chunk_offsets_ptr, profiler_ptr, num_entries);
  }
  else {
    auto kernel = h_kernel_cutlass<NUM_STAGES, false>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    dim3 grid(H, N);
    kernel<<<grid, TB_SIZE, smem_size>>>(
      K_tmap, V_tmap, W_tmap, H0_tmap, H_tmap, V_new_tmap,
      g_cu_ptr, ht_ptr, cu_seqlens_ptr, chunk_offsets_ptr, nullptr, 0);
  }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(h_v1, h_v1);
