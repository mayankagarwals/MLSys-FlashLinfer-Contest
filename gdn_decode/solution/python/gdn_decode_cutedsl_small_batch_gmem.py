# CuTe DSL decode kernel for small batch: register-only state (no SMEM staging).
# Block mapping matches gdn_decode_kernel_7.cu: grid = B*HV*(V/4), 128 threads,
# one warp owns one V row within a 4-row tile. Numerics follow gdn_decode_baseline.py.
#
# Perf-oriented tweaks: triple-issue q/k/state global tiles; uniform g/beta (no shfl);
# fused bf16→f32, q*scale, h*=g, and partial dot in one K-loop.

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

TILE_K = 128
NUM_WARPS = 4
NUM_THREADS = 128
ROWS_PER_BLOCK = NUM_WARPS
WARP_SIZE = 32
WARP_REDUCE_STEPS = 5


@cute.kernel
def gdn_decode_kernel_small_batch_gmem(
    vec_size: cutlass.Constexpr[int],
    num_v_tiles: cutlass.Constexpr[int],
    h0_source: cute.Tensor,
    A_log: cute.Tensor,
    a: cute.Tensor,
    dt_bias: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    b: cute.Tensor,
    o: cute.Tensor,
    scale: cutlass.Float32,
    HV: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
):
    """One block = one V-tile of ROWS_PER_BLOCK rows; state read/write in registers / GMEM."""
    tidx, _, _ = cute.arch.thread_idx()
    lane_id = cute.arch.lane_idx()
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    block_idx, _, _ = cute.arch.block_idx()

    tile_idx = block_idx % num_v_tiles
    bh = block_idx // num_v_tiles
    i_n = bh // HV
    i_hv = bh % HV
    i_h = i_hv // (HV // H)
    i_t = 0

    v_row = tile_idx * ROWS_PER_BLOCK + warp_idx

    r_k = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    r_q = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    r_h = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    r_q_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )
    r_k_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )

    r_A_log = cutlass.Float32(A_log[i_hv])
    r_a = cutlass.Float32(a[i_n, i_t, i_hv])
    r_dt_bias = cutlass.Float32(dt_bias[i_hv])
    r_b = cutlass.Float32(b[i_n, i_t, i_hv])

    q_tile = cute.local_tile(q, (1, 1, 1, vec_size), (i_n, i_t, i_h, lane_id))
    k_tile = cute.local_tile(k, (1, 1, 1, vec_size), (i_n, i_t, i_h, lane_id))
    g_h = cute.local_tile(h0_source, (1, 1, vec_size), (bh, v_row, lane_id))
    # Issue three independent global tiles back-to-back for better memory-level ILP.
    cute.autovec_copy(q_tile, r_q_bf16)
    cute.autovec_copy(k_tile, r_k_bf16)
    cute.autovec_copy(g_h, r_h)

    # All lanes compute identical g / beta (uniform inputs → no warp shuffle).
    x = r_a + r_dt_bias
    beta_x = x
    softplus_x = 0.0
    if beta_x <= 20.0:
        exp_beta_x = cute.exp(beta_x, fastmath=True)
        log_input = cutlass.Float32(1.0 + exp_beta_x)
        log_result = cutlass.Float32(cute.log(log_input, fastmath=True))
        softplus_x = log_result
    else:
        softplus_x = x
    r_g_value = -cute.exp(r_A_log, fastmath=True) * softplus_x
    r_beta = 1.0 / (1.0 + cute.exp(-r_b, fastmath=True))
    r_g = cute.exp(r_g_value, fastmath=True)

    sum_hk = 0.0
    for i in cutlass.range_constexpr(vec_size):
        r_q[i] = cutlass.Float32(r_q_bf16[i]) * scale
        r_k[i] = cutlass.Float32(r_k_bf16[i])
        r_h[i] = r_h[i] * r_g
        sum_hk += r_h[i] * r_k[i]
    for offset in [16, 8, 4, 2, 1]:
        sum_hk += cute.arch.shuffle_sync_bfly(
            sum_hk, offset=offset, mask=-1, mask_and_clamp=31
        )

    v_scalar_bf16 = cutlass.BFloat16(0.0)
    if lane_id == 0 and v_row < V:
        v_scalar_bf16 = v[i_n, i_t, i_hv, v_row]
    v_scalar = cutlass.Float32(v_scalar_bf16)
    v_scalar = cute.arch.shuffle_sync(v_scalar, 0)

    v_new = (v_scalar - sum_hk) * r_beta

    sum_hq = 0.0
    for i in cutlass.range_constexpr(vec_size):
        r_h[i] += r_k[i] * v_new
        sum_hq += r_h[i] * r_q[i]
    for offset in [16, 8, 4, 2, 1]:
        sum_hq += cute.arch.shuffle_sync_bfly(
            sum_hq, offset=offset, mask=-1, mask_and_clamp=31
        )

    g_out = cute.local_tile(h0_source, (1, 1, vec_size), (bh, v_row, lane_id))
    cute.autovec_copy(r_h, g_out)

    if lane_id == 0 and v_row < V:
        o[i_n, i_t, i_hv, v_row] = cutlass.BFloat16(sum_hq)


@cute.jit
def run_gdn_decode_kernel_small_batch_gmem(
    h0_source: cute.Tensor,
    A_log: cute.Tensor,
    a: cute.Tensor,
    dt_bias: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    b: cute.Tensor,
    o: cute.Tensor,
    scale: cutlass.Float32,
    HV: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    _, v_dim, k_dim = (
        h0_source.layout.shape[0],
        h0_source.layout.shape[1],
        h0_source.layout.shape[2],
    )
    num_v_tiles = v_dim // ROWS_PER_BLOCK
    vec_size = k_dim // WARP_SIZE

    gdn_decode_kernel_small_batch_gmem(
        vec_size,
        num_v_tiles,
        h0_source,
        A_log,
        a,
        dt_bias,
        q,
        k,
        v,
        b,
        o,
        scale,
        HV,
        H,
        K,
        V,
    ).launch(
        grid=(h0_source.layout.shape[0] * num_v_tiles, 1, 1),
        block=[NUM_THREADS, 1, 1],
        smem=0,
        stream=stream,
    )


kernel_cache: dict = {}


def run(q, k, v, state, A_log, a, dt_bias, b, scale):
    """Same interface as gdn_decode_baseline.run; state layout [B, HV, V, K] (v-major)."""
    B, HV, V, K = state.shape
    H = q.shape[2]
    output = torch.empty_like(v)

    if K != TILE_K or V % ROWS_PER_BLOCK != 0:
        raise ValueError(
            f"gdn_decode_cutedsl_small_batch_gmem expects K={TILE_K} and V divisible by "
            f"{ROWS_PER_BLOCK}; got K={K}, V={V}"
        )

    h0_source = state.view(B * HV, V, K)

    cache_key = (B, H, HV, K, V, q.dtype)
    if cache_key not in kernel_cache:
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        h0_source_tensor = from_dlpack(h0_source, assumed_align=16)
        A_log_tensor = from_dlpack(A_log, assumed_align=16)
        a_tensor = from_dlpack(a, assumed_align=16)
        dt_bias_tensor = from_dlpack(dt_bias, assumed_align=16)
        q_tensor = from_dlpack(q, assumed_align=16)
        k_tensor = from_dlpack(k, assumed_align=16)
        v_tensor = from_dlpack(v, assumed_align=16)
        b_tensor = from_dlpack(b, assumed_align=16)
        o_tensor = from_dlpack(output, assumed_align=16)

        kernel_cache[cache_key] = cute.compile(
            run_gdn_decode_kernel_small_batch_gmem,
            h0_source_tensor,
            A_log_tensor,
            a_tensor,
            dt_bias_tensor,
            q_tensor,
            k_tensor,
            v_tensor,
            b_tensor,
            o_tensor,
            scale,
            HV=HV,
            H=H,
            K=K,
            V=V,
            stream=stream,
            options="--enable-tvm-ffi",
        )

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    kernel_cache[cache_key](
        h0_source, A_log, a, dt_bias, q, k, v, b, output, scale, stream
    )

    return output, state
