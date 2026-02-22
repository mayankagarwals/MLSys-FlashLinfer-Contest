"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
# Adopted by Simon Veitner

import functools
from typing import Optional, Tuple
import torch
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda

# ============================================================================
# Global configuration for PRETRANSPOSE version ([B*HV, V, K])
# ============================================================================
TILE_V = 8
SMALL_TILE_V = 4
TILE_K = 128
WARP_SIZE = 32
NUM_WARPS = 4
NUM_THREADS = WARP_SIZE * NUM_WARPS
NUM_BLOCKS_PER_STATE = 32
WARP_REDUCE_STEPS = 5


_TORCH_TO_CUTLASS_DTYPE = {
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
    torch.int32: cutlass.Int32,
}


def _make_fake_row_major_tensor(tensor: torch.Tensor, *, assumed_align: int = 16):
    # Workload definition constrains tensors to bf16/f32/i32 dtypes.
    cutlass_dtype = _TORCH_TO_CUTLASS_DTYPE[tensor.dtype]
    stride_order = tuple(reversed(range(tensor.dim())))
    return cute.runtime.make_fake_compact_tensor(
        cutlass_dtype,
        tuple(tensor.shape),
        stride_order=stride_order,
        assumed_align=assumed_align,
    )


@cute.kernel
def gdn_decode_kernel(
    tiled_copy_load: cute.TiledCopy,
    h0_source: cute.Tensor,
    vec_size: cutlass.Constexpr[int],
    num_v_tiles: cutlass.Constexpr[int],
    A_log: cute.Tensor,  # [HV]
    a: cute.Tensor,  # [B, T, HV]
    dt_bias: cute.Tensor,  # [HV]
    q: cute.Tensor,  # [B, T, H, K]
    k: cute.Tensor,  # [B, T, H, K]
    v: cute.Tensor,  # [B, T, HV, V]
    b: cute.Tensor,  # [B, T, HV]
    o: cute.Tensor,  # [B, T, HV, V] - output
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
):
    """Each block synchronously tiles state through SMEM and vectorized writeback."""

    tidx, _, _ = cute.arch.thread_idx()
    lane_id = cute.arch.lane_idx()
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    block_idx, _, _ = cute.arch.block_idx()
    batch_idx = block_idx // NUM_BLOCKS_PER_STATE
    batch_inner = block_idx % NUM_BLOCKS_PER_STATE
    num_v_tiles_per_block = num_v_tiles // NUM_BLOCKS_PER_STATE
    i_n = batch_idx // HV
    i_hv = batch_idx % HV
    i_h = i_hv // (HV // H)
    i_t = 0

    r_A_log = cutlass.Float32(A_log[i_hv])
    r_a = cutlass.Float32(a[i_n, i_t, i_hv])
    r_dt_bias = cutlass.Float32(dt_bias[i_hv])
    r_b = cutlass.Float32(b[i_n, i_t, i_hv])

    smem = cutlass.utils.SmemAllocator()

    # ===================================================================
    # Allocate shared memory for state tile and output.
    # ===================================================================
    sData = smem.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((SMALL_TILE_V, TILE_K), stride=(TILE_K, 1)),
        128,
    )
    sOutput = smem.allocate_tensor(cutlass.BFloat16, cute.make_layout((V,)), 16)

    r_k = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    r_q = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    r_h = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    # BF16 register tensors for vectorized q and k loading
    r_q_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )
    r_k_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )

    # cute.arch.barrier() # No write at this point, redundant.

    # Get current batch
    gSrc_batch = h0_source[(batch_idx, None, None)]  # (V, K)
    gDst = cute.local_tile(h0_source, (1, SMALL_TILE_V, TILE_K), (batch_idx, None, 0))

    # V 方向分 tiles
    gSrc = cute.local_tile(
        gSrc_batch, (SMALL_TILE_V, TILE_K), (None, 0)
    )  # (SMALL_TILE_V, TILE_K, num_v_tiles)
    thr_copy_load = tiled_copy_load.get_slice(tidx)

    start_v_tiles = batch_inner * num_v_tiles_per_block

    # Load q, k into BF16 registers using autovec_copy (contiguous pattern)
    q_tile = cute.local_tile(q, (1, 1, 1, vec_size), (i_n, i_t, i_h, lane_id))
    k_tile = cute.local_tile(k, (1, 1, 1, vec_size), (i_n, i_t, i_h, lane_id))
    cute.autovec_copy(q_tile, r_q_bf16)
    cute.autovec_copy(k_tile, r_k_bf16)

    # Convert BF16 to FP32
    for i in cutlass.range_constexpr(vec_size):
        r_q[i] = cutlass.Float32(r_q_bf16[i])
        r_k[i] = cutlass.Float32(r_k_bf16[i])

    # ===================================================================
    # Compute g and beta (scalar values)
    # ===================================================================
    r_g = 0.0
    r_beta = 0.0
    if lane_id == 0:
        x = r_a + r_dt_bias
        beta_x = softplus_beta * x
        softplus_x = 0.0

        if beta_x <= softplus_threshold:
            # softplus(x) = (1/beta) * log(1 + exp(beta*x))
            exp_beta_x = cute.exp(beta_x, fastmath=True)
            log_input = cutlass.Float32(1.0 + exp_beta_x)
            log_result = cutlass.Float32(cute.log(log_input, fastmath=True))
            softplus_x = cutlass.Float32(
                (cutlass.Float32(1.0) / softplus_beta) * log_result
            )
        else:
            softplus_x = x

        # Compute g = exp(A_log) * softplus_x
        r_g_value = -cute.exp(r_A_log, fastmath=True) * softplus_x

        # Compute beta = 1 / (1 + exp(-b))
        r_beta = 1.0 / (1.0 + cute.exp(-r_b, fastmath=True))

        # Store to scalar (Float32)
        r_g = cute.exp(r_g_value, fastmath=True)

    r_g = cute.arch.shuffle_sync(r_g, 0)
    r_beta = cute.arch.shuffle_sync(r_beta, 0)

    # Apply scaling in Float32
    for i in cutlass.range_constexpr(vec_size):
        r_q[i] = r_q[i] * scale

    # ===================================================================
    # Mainloop: All threads participate
    # ===================================================================
    end_v_tiles = start_v_tiles + num_v_tiles_per_block
    for v_tiles in range(start_v_tiles, end_v_tiles):
        # Step 1: Synchronously stage one state tile from global to shared memory.
        gSrc_copy_tile = gSrc[(None, None, v_tiles)]
        thr_gSrc = thr_copy_load.partition_S(gSrc_copy_tile)
        thr_sData = thr_copy_load.partition_D(sData)
        cute.copy(tiled_copy_load, thr_gSrc, thr_sData)
        cute.arch.barrier()

        # Step 2: Compute from shared memory tile.
        for row in cutlass.range_constexpr(0, SMALL_TILE_V, NUM_THREADS // WARP_SIZE):
            row_offset = warp_idx
            sum_hk = 0.0

            # Load h from sData using 2D local_tile + autovec_copy (contiguous in K).
            sData_tile = cute.local_tile(
                sData, (1, vec_size), (row + row_offset, lane_id)
            )
            cute.autovec_copy(sData_tile, r_h)

            for i in cutlass.range_constexpr(vec_size):
                r_h[i] = r_h[i] * r_g
                sum_hk += r_h[i] * r_k[i]

            for reduce_step in cutlass.range_constexpr(WARP_REDUCE_STEPS):
                sum_hk += cute.arch.shuffle_sync_bfly(
                    sum_hk,
                    offset=(WARP_SIZE >> (reduce_step + 1)),
                    mask=-1,
                    mask_and_clamp=WARP_SIZE - 1,
                )

            o_idx = v_tiles * SMALL_TILE_V + row + row_offset

            v_scalar = cutlass.Float32(0.0)
            if lane_id == 0 and o_idx < V:
                v_scalar = cutlass.Float32(v[(i_n, i_t, i_hv, o_idx)])
            v_scalar = cute.arch.shuffle_sync(v_scalar, 0)  # Broadcast to all lanes

            v_new = v_scalar - sum_hk
            v_new = v_new * r_beta

            sum_hq = 0.0
            for i in cutlass.range_constexpr(vec_size):
                r_h[i] += r_k[i] * v_new
                sum_hq += r_h[i] * r_q[i]

            # Write h to gDst using 4D local_tile + autovec_copy (contiguous in K)
            gDst_tile = cute.local_tile(
                gDst, (1, 1, vec_size, 1), (0, row + row_offset, lane_id, v_tiles)
            )
            cute.autovec_copy(r_h, gDst_tile)

            for reduce_step in cutlass.range_constexpr(WARP_REDUCE_STEPS):
                sum_hq += cute.arch.shuffle_sync_bfly(
                    sum_hq,
                    offset=(WARP_SIZE >> (reduce_step + 1)),
                    mask=-1,
                    mask_and_clamp=WARP_SIZE - 1,
                )

            if lane_id == 0 and o_idx < V:
                sOutput[o_idx] = cutlass.BFloat16(sum_hq)
        cute.arch.barrier()

    # ===================================================================
    # Final writeback: Copy output from shared memory to global memory
    # All threads write (V=128, NUM_THREADS)
    # ===================================================================
    cute.arch.barrier()  # Ensure all writes to sOutput are complete
    if tidx >= start_v_tiles * SMALL_TILE_V and tidx < end_v_tiles * SMALL_TILE_V:
        o[(i_n, i_t, i_hv, tidx)] = sOutput[tidx]


@cute.jit
def run_gdn_decode_kernel(
    h0_source: cute.Tensor,  # [B*HV, K, V]
    A_log: cute.Tensor,
    a: cute.Tensor,
    dt_bias: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    b: cute.Tensor,
    o: cute.Tensor,
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    """Launch decode kernel with synchronous SMEM staging for state loads."""
    # h0_source: (B*HV, V, K)
    batch_size, v_dim, _ = (
        h0_source.layout.shape[0],
        h0_source.layout.shape[1],
        h0_source.layout.shape[2],
    )

    copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        cutlass.Float32,
        num_bits_per_copy=128,
    )
    thread_layout = cute.make_layout(
        (NUM_THREADS // WARP_SIZE, WARP_SIZE),
        stride=(WARP_SIZE, 1),
    )
    val_layout = cute.make_layout((1, 4))
    tiled_copy_load = cute.make_tiled_copy_tv(copy_atom, thread_layout, val_layout)

    num_v_tiles = cute.ceil_div(v_dim, SMALL_TILE_V)

    vec_size = (
        TILE_K // WARP_SIZE
    )  # Each thread in a warp processes this many elements (always 4 for TILE_K=128)

    # sData: SMALL_TILE_V * TILE_K * 4 bytes (Float32)
    # sOutput: V * 2 bytes (BFloat16)
    smem_bytes = 4 * SMALL_TILE_V * TILE_K + 2 * v_dim + 32

    gdn_decode_kernel(
        tiled_copy_load,
        h0_source,
        vec_size,
        num_v_tiles,
        A_log,
        a,
        dt_bias,
        q,
        k,
        v,
        b,
        o,
        softplus_beta,
        softplus_threshold,
        scale,
        HV,
        H,
        V,
    ).launch(
        grid=(batch_size * NUM_BLOCKS_PER_STATE, 1, 1),
        block=[NUM_THREADS, 1, 1],
        smem=smem_bytes,
        stream=stream,
    )


# ============================================================================
# FlashInfer API Layer
# ============================================================================


@functools.cache
def _get_compiled_decode_kernel(cache_key: tuple):
    """Cache compiled kernel for given configuration (pretranspose version)."""
    # The key participates in functools.cache memoization via function arguments.
    _ = cache_key
    # This will be populated on first call
    return {}


def gated_delta_rule_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: Optional[torch.Tensor],
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Gated Delta Rule Decode kernel for single-token generation.

    This implements the decode phase of gated delta rule linear attention,
    processing one token at a time and updating the recurrent state.

    Args:
        q (torch.Tensor):
            Current query of shape ``[B, 1, H, K]``. Must be float16/bfloat16.
        k (torch.Tensor):
            Current key of shape ``[B, 1, H, K]``. Must be float16/bfloat16.
        v (torch.Tensor):
            Current value of shape ``[B, 1, HV, V]``. Must be float16/bfloat16.
        state (Optional[torch.Tensor]):
            Current state of shape ``[B, HV, V, K]`` (v-major layout).
            Must be float32. If ``None``, zero state is used.
        A_log (torch.Tensor):
            Log decay parameter of shape ``[HV]``. Must be float32.
        a (torch.Tensor):
            Input-dependent decay of shape ``[B, 1, HV]``. Must be float16/bfloat16.
        dt_bias (torch.Tensor):
            Decay bias of shape ``[HV]``. Must be bfloat16 or float32.
        b (torch.Tensor):
            Update gate (beta) input of shape ``[B, 1, HV]``. Must be float16/bfloat16.
        scale (Optional[float]):
            Scale factor for queries. If None, defaults to ``1 / sqrt(K)``.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - output: Output tensor of shape ``[B, 1, HV, V]``
            - state: Updated state tensor of shape ``[B, HV, V, K]``

    Note:
        - Requires SM90 (Hopper) architecture
        - K and V must be multiples of 4 for vectorized loads
        - State layout is v-major: [B, HV, V, K]
    """
    # Validate input shapes
    B, T, H, K = q.shape
    assert T == 1, f"Decode only supports T=1, got T={T}"
    _, _, HV, V = v.shape

    # Validate K and V constraints
    assert K >= 128, f"K must be at least 128, got K={K}"
    assert V >= 128, f"V must be at least 128, got V={V}"
    assert V % TILE_V == 0, (
        f"V must be divisible by {TILE_V} to prevent out-of-bounds access, got V={V}"
    )

    # Validate dtypes
    assert q.dtype in (torch.float16, torch.bfloat16), (
        f"q must be float16/bfloat16, got {q.dtype}"
    )
    assert A_log.dtype == torch.float32, f"A_log must be float32, got {A_log.dtype}"

    # Set default scale
    if scale is None or scale == 0.0:
        scale = K**-0.5

    # Match benchmark API assumptions for this workload:
    # - output is not destination-passed
    output = torch.empty((B, T, HV, V), dtype=torch.bfloat16, device=q.device)

    # Public API uses k-last state layout [B, HV, V, K].
    if state is None:
        state_work = torch.zeros((B, HV, V, K), dtype=torch.float32, device=q.device)
    else:
        assert state.shape == (B, HV, V, K), (
            f"Expected state shape [B={B}, HV={HV}, V={V}, K={K}], got {state.shape}"
        )
        assert state.dtype == torch.float32, f"state must be float32, got {state.dtype}"
        state_work = state

    # Flatten [B, HV, V, K] -> [B*HV, V, K] for kernel
    h0_source = state_work.reshape(B * HV, V, K)

    # Compile kernel with TVM FFI (cached)
    cache_key = (B, T, H, HV, K, V, q.dtype, scale)
    cache = _get_compiled_decode_kernel(cache_key)

    if "compiled" not in cache:
        # Compile against fake tensors to reduce TVM FFI call overhead.
        h0_source_tensor = _make_fake_row_major_tensor(h0_source)
        A_log_tensor = _make_fake_row_major_tensor(A_log)
        a_tensor = _make_fake_row_major_tensor(a)
        dt_bias_tensor = _make_fake_row_major_tensor(dt_bias)
        q_tensor = _make_fake_row_major_tensor(q)
        k_tensor = _make_fake_row_major_tensor(k)
        v_tensor = _make_fake_row_major_tensor(v)
        b_tensor = _make_fake_row_major_tensor(b)
        o_tensor = _make_fake_row_major_tensor(output)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

        run_func = run_gdn_decode_kernel

        # Use TVM FFI to reduce runtime overhead
        cache["compiled"] = cute.compile(
            run_func,
            h0_source_tensor,
            A_log_tensor,
            a_tensor,
            dt_bias_tensor,
            q_tensor,
            k_tensor,
            v_tensor,
            b_tensor,
            o_tensor,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            scale=scale,
            HV=HV,
            H=H,
            V=V,
            stream=stream,
            options="--enable-tvm-ffi",
        )

    # Run kernel directly with PyTorch tensors on the current torch stream.
    cache["compiled"](h0_source, A_log, a, dt_bias, q, k, v, b, output)

    updated_state = h0_source.reshape(B, HV, V, K)  # .contigous()
    return output, updated_state
