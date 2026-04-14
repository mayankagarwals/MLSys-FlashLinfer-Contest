# chunk_v7b with CUDA graph to eliminate inter-kernel launch overhead
# CUDA graph captures the 4-kernel pipeline and replays with zero launch overhead
# Requires pre-allocated tensors (fixed addresses for graph replay)

from pathlib import Path
import torch
import triton
from torch import Tensor

from . import chunk_v7 as _chunk_v7
from .chunk_v6c import merge_16x16_to_64x64_inverse_kernel_v2

chunk_fwd_kernel_o = _chunk_v7.chunk_fwd_kernel_o
mod = _chunk_v7.mod

# Cache for CUDA graphs (key: (T, N, device))
_graph_cache = {}
_buf_cache = {}


def _get_or_create_graph(T, N, H, Hg, K_dim, V_dim, BT, device,
                         q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    """Create and cache a CUDA graph for this workload shape."""
    key = (T, N, id(device) if isinstance(device, torch.device) else device)

    if key in _graph_cache:
        bufs = _buf_cache[key]
        # Copy inputs to pre-allocated buffers
        bufs['q'].copy_(q)
        bufs['k'].copy_(k)
        bufs['v'].copy_(v)
        bufs['state'].copy_(state)
        bufs['A_log'].copy_(A_log)
        bufs['a'].copy_(a)
        bufs['dt_bias'].copy_(dt_bias)
        bufs['b'].copy_(b)
        bufs['cu_seqlens'].copy_(cu_seqlens)
        return _graph_cache[key], bufs

    # Pre-allocate all buffers
    upper_bound_chunks = (N - 1) + triton.cdiv(T - (N - 1), BT)

    bufs = {
        'q': q.clone(),
        'k': k.clone(),
        'v': v.clone(),
        'state': state.clone(),
        'A_log': A_log.clone(),
        'a': a.clone(),
        'dt_bias': dt_bias.clone(),
        'b': b.clone(),
        'cu_seqlens': cu_seqlens.clone(),
        'co': q.new_empty(N + 1, dtype=torch.int32),
        'ci': q.new_empty((upper_bound_chunks, 2), dtype=torch.int32),
        'g_cu': torch.empty(T, H, device=device, dtype=torch.float32),
        'beta': torch.empty(T, H, device=device, dtype=torch.float32),
        'A': torch.empty(T, H, BT, device=device, dtype=torch.float32),
        'u': torch.empty(T, H, V_dim, device=device, dtype=k.dtype),
        'w': torch.empty(T, H, K_dim, device=device, dtype=k.dtype),
        'h': torch.empty(upper_bound_chunks, H, V_dim, K_dim, device=device, dtype=k.dtype),
        'fs': torch.empty(N, H, V_dim, K_dim, device=device, dtype=torch.float32),
        'vn': torch.empty(upper_bound_chunks, BT, H, V_dim, device=device, dtype=q.dtype),
        'o': torch.empty(T, H, V_dim, device=device, dtype=v.dtype),
    }
    bufs['tc'] = bufs['co'][N:]

    # Warmup (required before graph capture)
    for _ in range(3):
        _run_pipeline(bufs, H, Hg, K_dim, V_dim, BT, upper_bound_chunks, scale)
    torch.cuda.synchronize()

    # Capture graph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run_pipeline(bufs, H, Hg, K_dim, V_dim, BT, upper_bound_chunks, scale)

    _graph_cache[key] = graph
    _buf_cache[key] = bufs
    return graph, bufs


def _run_pipeline(bufs, H, Hg, K_dim, V_dim, BT, upper_bound_chunks, scale):
    """Execute the 4-kernel pipeline using pre-allocated buffers."""
    mod.kkt_v1b_with_meta(
        bufs['k'], bufs['A_log'], bufs['a'], bufs['dt_bias'], bufs['b'],
        bufs['g_cu'], bufs['beta'], bufs['A'],
        bufs['cu_seqlens'], bufs['ci'], bufs['co'])

    merge_16x16_to_64x64_inverse_kernel_v2[(upper_bound_chunks, H)](
        bufs['k'], bufs['v'], bufs['w'], bufs['u'],
        bufs['A'], bufs['beta'], bufs['g_cu'],
        bufs['cu_seqlens'], bufs['ci'], bufs['tc'],
        H=H, Hg=Hg, K_dim=K_dim, V_dim=V_dim, BT=BT, num_warps=2)

    mod.h_v1(
        bufs['k'], bufs['u'], bufs['w'], bufs['vn'], bufs['g_cu'],
        bufs['h'], bufs['state'], bufs['fs'],
        bufs['cu_seqlens'], bufs['co'], None)

    chunk_fwd_kernel_o[(upper_bound_chunks, H)](
        bufs['q'], bufs['k'], bufs['vn'], bufs['h'], bufs['g_cu'], bufs['o'],
        bufs['cu_seqlens'], bufs['ci'], bufs['tc'],
        scale=scale, H=H, Hg=Hg, K_dim=K_dim, V_dim=V_dim, BT=BT, BV=64, num_warps=4)


def run(
    q: Tensor, k: Tensor, v: Tensor, state: Tensor,
    A_log: Tensor, a: Tensor, dt_bias: Tensor, b: Tensor,
    cu_seqlens: Tensor, scale: float,
):
    T, Hg, K_dim = k.shape
    N, H, V_dim, _ = state.shape
    BT = 64

    graph, bufs = _get_or_create_graph(
        T, N, H, Hg, K_dim, V_dim, BT, k.device,
        q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)

    # Copy inputs to graph buffers (if not already done in _get_or_create_graph)
    bufs['q'].copy_(q)
    bufs['k'].copy_(k)
    bufs['v'].copy_(v)
    bufs['state'].copy_(state)
    bufs['A_log'].copy_(A_log)
    bufs['a'].copy_(a)
    bufs['dt_bias'].copy_(dt_bias)
    bufs['b'].copy_(b)
    bufs['cu_seqlens'].copy_(cu_seqlens)

    # Replay graph (zero inter-kernel overhead!)
    graph.replay()

    # Return output views (same memory as graph buffers)
    return bufs['o'][:T], bufs['fs'][:N]
