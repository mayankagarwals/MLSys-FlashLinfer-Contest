"""Test CUDA O-kernel (cuda_o_v1) against Triton O-kernel (triton_v4).

Usage: cd gdn_prefill/solution/python && python test_o_kernel.py
"""
import os, sys, time
os.environ["TVM_FFI_CUDA_ARCH_LIST"] = "10.0a"
from pathlib import Path

import torch
import triton
import tvm_ffi

# Build CUDA O-kernel
CURRENT_DIR = Path(__file__).parent
lib_path = tvm_ffi.cpp.build(
    name="gdn_prefill_cuda_o_v1",
    cuda_files=[str(CURRENT_DIR / "cuda_o_v1.cu")],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
    extra_ldflags=["-lcuda"],
)
mod_o = tvm_ffi.load_module(lib_path)

# Import Triton O-kernel for reference
sys.path.insert(0, str(CURRENT_DIR))
from triton_v4 import chunk_fwd_kernel_o as triton_o_kernel

def test_o_kernel(T=256, N=2, BT=64, H=8, Hg=4, K=128, V=128, BV=64):
    """Compare CUDA vs Triton O-kernel output."""
    device = "cuda"

    # Create fake inputs
    cu_seqlens = torch.zeros(N + 1, dtype=torch.int64, device=device)
    seqlens = torch.full((N,), T // N, dtype=torch.int64, device=device)
    # Last seq gets remainder
    seqlens[-1] += T - seqlens.sum()
    cu_seqlens[1:] = seqlens.cumsum(0)

    # Compute chunks
    num_chunks_per_seq = torch.div(seqlens + BT - 1, BT, rounding_mode='trunc')
    total_chunks = int(num_chunks_per_seq.sum().item())

    chunk_indices = []
    for s in range(N):
        nc = int(num_chunks_per_seq[s].item())
        for c in range(nc):
            chunk_indices.append([s, c])
    chunk_indices = torch.tensor(chunk_indices, dtype=torch.int32, device=device)

    # Random inputs
    q = torch.randn(T, Hg, K, dtype=torch.bfloat16, device=device) * 0.1
    k = torch.randn(T, Hg, K, dtype=torch.bfloat16, device=device) * 0.1
    v_new = torch.randn(total_chunks * BT, H, V, dtype=torch.bfloat16, device=device) * 0.1
    h = torch.randn(total_chunks, H, V, K, dtype=torch.bfloat16, device=device) * 0.1
    g_cu = torch.randn(T, H, dtype=torch.float32, device=device) * 0.5
    scale = 1.0 / (K ** 0.5)

    # ── Triton reference ──
    o_ref = torch.zeros(T, H, V, dtype=torch.bfloat16, device=device)
    grid = (triton.cdiv(V, BV), total_chunks, H)
    triton_o_kernel[grid](
        q, k, v_new, h, g_cu, o_ref,
        cu_seqlens, chunk_indices,
        scale=scale,
        H=H, Hg=Hg, K_dim=K, V_dim=V, BT=BT, BV=BV,
        num_warps=8,
    )
    torch.cuda.synchronize()

    # ── CUDA kernel ──
    o_cuda = torch.zeros(T, H, V, dtype=torch.bfloat16, device=device)
    mod_o.o_v1(q, k, v_new, h, g_cu, o_cuda, cu_seqlens, chunk_indices, total_chunks, scale)
    torch.cuda.synchronize()

    # ── Compare ──
    diff = (o_ref.float() - o_cuda.float()).abs()
    max_abs = diff.max().item()
    rel_diff = diff / (o_ref.float().abs() + 1e-6)
    max_rel = rel_diff.max().item()

    # Count elements failing BOTH abs and rel tolerance
    atol, rtol = 1e-2, 1e-2
    abs_fail = diff > atol
    rel_fail = rel_diff > rtol
    both_fail = abs_fail & rel_fail
    n_fail = both_fail.sum().item()
    n_total = diff.numel()

    status = "PASS" if n_fail == 0 else "FAIL"
    print(f"  T={T:>5} N={N}: {status}  max_abs={max_abs:.6f}  max_rel={max_rel:.6f}  "
          f"fail={n_fail}/{n_total}")

    if n_fail > 0:
        # Find first failing element
        idx = torch.where(both_fail)
        t, h_idx, v_idx = idx[0][0].item(), idx[1][0].item(), idx[2][0].item()
        print(f"    First fail at [{t}, {h_idx}, {v_idx}]: "
              f"ref={o_ref[t, h_idx, v_idx].item():.6f} "
              f"cuda={o_cuda[t, h_idx, v_idx].item():.6f} "
              f"diff={diff[t, h_idx, v_idx].item():.6f}")

    return n_fail == 0


def bench_o_kernel(T=8192, N=2, BT=64, H=8, Hg=4, K=128, V=128, BV=64, n_iter=200):
    """Benchmark CUDA vs Triton O-kernel."""
    device = "cuda"

    cu_seqlens = torch.zeros(N + 1, dtype=torch.int64, device=device)
    seqlens = torch.full((N,), T // N, dtype=torch.int64, device=device)
    seqlens[-1] += T - seqlens.sum()
    cu_seqlens[1:] = seqlens.cumsum(0)

    num_chunks_per_seq = torch.div(seqlens + BT - 1, BT, rounding_mode='trunc')
    total_chunks = int(num_chunks_per_seq.sum().item())

    chunk_indices = []
    for s in range(N):
        nc = int(num_chunks_per_seq[s].item())
        for c in range(nc):
            chunk_indices.append([s, c])
    chunk_indices = torch.tensor(chunk_indices, dtype=torch.int32, device=device)

    q = torch.randn(T, Hg, K, dtype=torch.bfloat16, device=device) * 0.1
    k = torch.randn(T, Hg, K, dtype=torch.bfloat16, device=device) * 0.1
    v_new = torch.randn(total_chunks * BT, H, V, dtype=torch.bfloat16, device=device) * 0.1
    h = torch.randn(total_chunks, H, V, K, dtype=torch.bfloat16, device=device) * 0.1
    g_cu = torch.randn(T, H, dtype=torch.float32, device=device) * 0.5
    scale = 1.0 / (K ** 0.5)
    o = torch.zeros(T, H, V, dtype=torch.bfloat16, device=device)

    # Warmup
    for _ in range(10):
        mod_o.o_v1(q, k, v_new, h, g_cu, o, cu_seqlens, chunk_indices, total_chunks, scale)
    torch.cuda.synchronize()

    # CUDA timing
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n_iter):
        mod_o.o_v1(q, k, v_new, h, g_cu, o, cu_seqlens, chunk_indices, total_chunks, scale)
    e.record()
    torch.cuda.synchronize()
    cuda_us = s.elapsed_time(e) / n_iter * 1000

    # Triton timing
    grid = (triton.cdiv(V, BV), total_chunks, H)
    for _ in range(10):
        triton_o_kernel[grid](
            q, k, v_new, h, g_cu, o, cu_seqlens, chunk_indices,
            scale=scale, H=H, Hg=Hg, K_dim=K, V_dim=V, BT=BT, BV=BV, num_warps=8)
    torch.cuda.synchronize()

    s.record()
    for _ in range(n_iter):
        triton_o_kernel[grid](
            q, k, v_new, h, g_cu, o, cu_seqlens, chunk_indices,
            scale=scale, H=H, Hg=Hg, K_dim=K, V_dim=V, BT=BT, BV=BV, num_warps=8)
    e.record()
    torch.cuda.synchronize()
    triton_us = s.elapsed_time(e) / n_iter * 1000

    speedup = triton_us / cuda_us
    print(f"  T={T:>5} N={N}: CUDA={cuda_us:.1f}us  Triton={triton_us:.1f}us  "
          f"speedup={speedup:.2f}x")


def test_vs_pytorch(T=128, N=1, BT=64, H=8, Hg=4, K=128, V=128, BV=64):
    """Compare CUDA O-kernel against PURE PyTorch (not Triton) for k=0 case."""
    device = "cuda"
    cu_seqlens = torch.zeros(N + 1, dtype=torch.int64, device=device)
    seqlens = torch.full((N,), T // N, dtype=torch.int64, device=device)
    seqlens[-1] += T - seqlens.sum()
    cu_seqlens[1:] = seqlens.cumsum(0)

    num_chunks_per_seq = torch.div(seqlens + BT - 1, BT, rounding_mode='trunc')
    total_chunks = int(num_chunks_per_seq.sum().item())
    chunk_indices = []
    for s in range(N):
        nc = int(num_chunks_per_seq[s].item())
        for c in range(nc):
            chunk_indices.append([s, c])
    chunk_indices_t = torch.tensor(chunk_indices, dtype=torch.int32, device=device)

    q = torch.randn(T, Hg, K, dtype=torch.bfloat16, device=device) * 0.1
    k_zero = torch.zeros(T, Hg, K, dtype=torch.bfloat16, device=device)
    v_new = torch.zeros(total_chunks * BT, H, V, dtype=torch.bfloat16, device=device)
    h = torch.randn(total_chunks, H, V, K, dtype=torch.bfloat16, device=device) * 0.1
    g_cu = torch.randn(T, H, dtype=torch.float32, device=device) * 0.5
    scale = 1.0 / (K ** 0.5)

    # ── PyTorch reference for k=0: o = q @ h^T * exp(g) * scale ──
    o_ref = torch.zeros(T, H, V, dtype=torch.float32, device=device)
    for ci, (sid, cloc) in enumerate(chunk_indices):
        bos = int(cu_seqlens[sid].item())
        eos = int(cu_seqlens[sid + 1].item())
        sl = eos - bos
        tb = cloc * BT
        clen = min(BT, sl - tb)
        for hid in range(H):
            kh = hid // (H // Hg)
            q_chunk = q[bos+tb:bos+tb+clen, kh, :].float()  # [clen, K]
            g = g_cu[bos+tb:bos+tb+clen, hid].float()  # [clen]
            exp_g = torch.exp(g)
            for bv in range(V // BV):
                bv0 = bv * BV
                h_tile = h[ci, hid, bv0:bv0+BV, :].float()  # [BV, K]
                o_tile = q_chunk @ h_tile.T * exp_g.unsqueeze(1) * scale  # [clen, BV]
                o_ref[bos+tb:bos+tb+clen, hid, bv0:bv0+BV] = o_tile

    # ── CUDA kernel ──
    o_cuda = torch.zeros(T, H, V, dtype=torch.bfloat16, device=device)
    mod_o.o_v1(q, k_zero, v_new, h, g_cu, o_cuda, cu_seqlens, chunk_indices_t, total_chunks, scale)
    torch.cuda.synchronize()

    diff = (o_ref - o_cuda.float()).abs()
    max_diff = diff.max().item()
    print(f"  CUDA vs PyTorch (k=0): max_diff={max_diff:.6f}  {'PASS' if max_diff < 0.02 else 'FAIL'}")

    if max_diff >= 0.02:
        idx = torch.where(diff == diff.max())
        t, hh, v = idx[0][0].item(), idx[1][0].item(), idx[2][0].item()
        print(f"    Max diff at [{t}, {hh}, {v}]: ref={o_ref[t,hh,v]:.6f} cuda={o_cuda[t,hh,v].float().item():.6f}")
        # Also check Triton
        o_triton = torch.zeros(T, H, V, dtype=torch.bfloat16, device=device)
        grid = (triton.cdiv(V, BV), total_chunks, H)
        triton_o_kernel[grid](
            q, k_zero, v_new, h, g_cu, o_triton, cu_seqlens, chunk_indices_t,
            scale=scale, H=H, Hg=Hg, K_dim=K, V_dim=V, BT=BT, BV=BV, num_warps=8)
        torch.cuda.synchronize()
        triton_diff = (o_ref - o_triton.float()).abs().max().item()
        cuda_vs_triton = (o_cuda.float() - o_triton.float()).abs().max().item()
        print(f"    Triton vs PyTorch: max_diff={triton_diff:.6f}")
        print(f"    CUDA vs Triton:    max_diff={cuda_vs_triton:.6f}")
        print(f"    ref[{t},{hh},{v}]={o_ref[t,hh,v]:.6f}  triton={o_triton[t,hh,v].float().item():.6f}  cuda={o_cuda[t,hh,v].float().item():.6f}")


def test_isolated(T=128, N=1, BT=64, H=8, Hg=4, K=128, V=128, BV=64):
    """Test with h=0, v_new=0 to isolate q@k^T computation."""
    device = "cuda"
    cu_seqlens = torch.zeros(N + 1, dtype=torch.int64, device=device)
    seqlens = torch.full((N,), T // N, dtype=torch.int64, device=device)
    seqlens[-1] += T - seqlens.sum()
    cu_seqlens[1:] = seqlens.cumsum(0)

    num_chunks_per_seq = torch.div(seqlens + BT - 1, BT, rounding_mode='trunc')
    total_chunks = int(num_chunks_per_seq.sum().item())
    chunk_indices = []
    for s in range(N):
        nc = int(num_chunks_per_seq[s].item())
        for c in range(nc):
            chunk_indices.append([s, c])
    chunk_indices = torch.tensor(chunk_indices, dtype=torch.int32, device=device)

    q = torch.randn(T, Hg, K, dtype=torch.bfloat16, device=device) * 0.1
    k = torch.randn(T, Hg, K, dtype=torch.bfloat16, device=device) * 0.1
    g_cu = torch.randn(T, H, dtype=torch.float32, device=device) * 0.5
    scale = 1.0 / (K ** 0.5)

    # Test 1: h=0, v_new=0 → only A@v contributes (which is zero), so o = q@h^T * exp_g = 0
    # Actually, with h=0: q@h^T = 0, and with v_new=0: A@v_new = 0. So o should be 0.
    h = torch.zeros(total_chunks, H, V, K, dtype=torch.bfloat16, device=device)
    v_new = torch.zeros(total_chunks * BT, H, V, dtype=torch.bfloat16, device=device)

    o_cuda = torch.zeros(T, H, V, dtype=torch.bfloat16, device=device)
    mod_o.o_v1(q, k, v_new, h, g_cu, o_cuda, cu_seqlens, chunk_indices, total_chunks, scale)
    torch.cuda.synchronize()

    max_val = o_cuda.float().abs().max().item()
    print(f"  Test h=0, v=0 → o should be 0: max_abs={max_val:.6f}  {'PASS' if max_val < 1e-3 else 'FAIL'}")

    # Test 2: q=0 → both terms zero
    q_zero = torch.zeros_like(q)
    h = torch.randn(total_chunks, H, V, K, dtype=torch.bfloat16, device=device) * 0.1
    v_new = torch.randn(total_chunks * BT, H, V, dtype=torch.bfloat16, device=device) * 0.1

    o_cuda.zero_()
    mod_o.o_v1(q_zero, k, v_new, h, g_cu, o_cuda, cu_seqlens, chunk_indices, total_chunks, scale)
    torch.cuda.synchronize()

    max_val = o_cuda.float().abs().max().item()
    print(f"  Test q=0 → o should be 0: max_abs={max_val:.6f}  {'PASS' if max_val < 1e-3 else 'FAIL'}")

    # Test 3: k=0 → A = q@k^T = 0, so A@v_new = 0. Only q@h^T * exp_g * scale remains.
    k_zero = torch.zeros_like(k)
    o_ref = torch.zeros(T, H, V, dtype=torch.bfloat16, device=device)
    o_cuda.zero_()

    grid = (triton.cdiv(V, BV), total_chunks, H)
    triton_o_kernel[grid](
        q, k_zero, v_new, h, g_cu, o_ref, cu_seqlens, chunk_indices,
        scale=scale, H=H, Hg=Hg, K_dim=K, V_dim=V, BT=BT, BV=BV, num_warps=8)
    mod_o.o_v1(q, k_zero, v_new, h, g_cu, o_cuda, cu_seqlens, chunk_indices, total_chunks, scale)
    torch.cuda.synchronize()

    diff = (o_ref.float() - o_cuda.float()).abs()
    max_diff = diff.max().item()
    print(f"  Test k=0 (only q@h^T term): max_abs_diff={max_diff:.6f}  "
          f"{'PASS' if max_diff < 0.01 else 'FAIL'}")

    # Test 4: h=0 → only A@v_new term
    h_zero = torch.zeros(total_chunks, H, V, K, dtype=torch.bfloat16, device=device)
    o_ref.zero_()
    o_cuda.zero_()

    triton_o_kernel[grid](
        q, k, v_new, h_zero, g_cu, o_ref, cu_seqlens, chunk_indices,
        scale=scale, H=H, Hg=Hg, K_dim=K, V_dim=V, BT=BT, BV=BV, num_warps=8)
    mod_o.o_v1(q, k, v_new, h_zero, g_cu, o_cuda, cu_seqlens, chunk_indices, total_chunks, scale)
    torch.cuda.synchronize()

    diff = (o_ref.float() - o_cuda.float()).abs()
    max_diff = diff.max().item()
    print(f"  Test h=0 (only A@v_new term): max_abs_diff={max_diff:.6f}  "
          f"{'PASS' if max_diff < 0.01 else 'FAIL'}")

    if max_diff >= 0.01:
        # Drill down: check A matrix itself
        # A = causal(q@k^T * exp_gate) — let's check with v_new = identity-like
        pass


def test_minimal_qht(T=64, N=1, BT=64, H=8, Hg=4, K=128, V=128, BV=64):
    """Test MINIMAL kernel: just q @ h^T * scale."""
    device = "cuda"
    cu_seqlens = torch.tensor([0, T], dtype=torch.int64, device=device)
    total_chunks = (T + BT - 1) // BT
    chunk_indices = torch.tensor([[0, c] for c in range(total_chunks)], dtype=torch.int32, device=device)

    q = torch.randn(T, Hg, K, dtype=torch.bfloat16, device=device) * 0.1
    k = torch.zeros(T, Hg, K, dtype=torch.bfloat16, device=device)
    v_new = torch.zeros(total_chunks * BT, H, V, dtype=torch.bfloat16, device=device)
    h = torch.randn(total_chunks, H, V, K, dtype=torch.bfloat16, device=device) * 0.1
    g_cu = torch.zeros(T, H, dtype=torch.float32, device=device)  # zeros → exp(g)=1
    scale = 1.0 / (K ** 0.5)

    # PyTorch ref: o = q @ h^T * scale (no exp_g since g=0)
    o_ref = torch.zeros(T, H, V, dtype=torch.float32, device=device)
    for ci in range(total_chunks):
        tb = ci * BT
        clen = min(BT, T - tb)
        for hid in range(H):
            kh = hid // (H // Hg)
            q_c = q[tb:tb+clen, kh, :].float()
            for bv in range(V // BV):
                bv0 = bv * BV
                h_t = h[ci, hid, bv0:bv0+BV, :].float()
                o_ref[tb:tb+clen, hid, bv0:bv0+BV] = q_c @ h_t.T * scale

    # CUDA
    o_cuda = torch.zeros(T, H, V, dtype=torch.bfloat16, device=device)
    mod_o.o_v1(q, k, v_new, h, g_cu, o_cuda, cu_seqlens, chunk_indices, total_chunks, scale)
    torch.cuda.synchronize()

    diff = (o_ref - o_cuda.float()).abs()
    max_diff = diff.max().item()
    print(f"  T={T}: max_diff={max_diff:.6f}  {'PASS' if max_diff < 0.02 else 'FAIL'}")
    if max_diff >= 0.02:
        idx = torch.where(diff == diff.max())
        t, hh, v = idx[0][0].item(), idx[1][0].item(), idx[2][0].item()
        print(f"    [{t},{hh},{v}]: ref={o_ref[t,hh,v]:.6f}  cuda={o_cuda[t,hh,v].float().item():.6f}")
        # Check first row of first head
        print(f"    ref[0,0,:8] = {o_ref[0,0,:8].tolist()}")
        print(f"    cuda[0,0,:8] = {o_cuda[0,0,:8].float().tolist()}")


def test_smem_dump(T=64, N=1, BT=64, H=8, Hg=4, K=128, V=128, BV=64):
    """Verify smem loading by reading back dumped values."""
    device = "cuda"
    cu_seqlens = torch.tensor([0, T], dtype=torch.int64, device=device)
    total_chunks = 1
    chunk_indices = torch.tensor([[0, 0]], dtype=torch.int32, device=device)

    q = torch.randn(T, Hg, K, dtype=torch.bfloat16, device=device) * 0.1
    k = torch.zeros(T, Hg, K, dtype=torch.bfloat16, device=device)
    v_new = torch.zeros(total_chunks * BT, H, V, dtype=torch.bfloat16, device=device)
    h = torch.randn(total_chunks, H, V, K, dtype=torch.bfloat16, device=device) * 0.1
    g_cu = torch.zeros(T, H, dtype=torch.float32, device=device)
    scale = 1.0

    # The kernel dumps s_q[0, 0:8] to o[0, 0, 0:8] for block (cid=0, hid=0)
    # kh = hid // 2 = 0. s_q loads q[0:BT, kh=0, :] → s_q[0, 0:8] should be q[0, 0, 0:8]
    o_cuda = torch.zeros(T, H, V, dtype=torch.bfloat16, device=device)
    mod_o.o_v1(q, k, v_new, h, g_cu, o_cuda, cu_seqlens, chunk_indices, total_chunks, scale)
    torch.cuda.synchronize()

    expected = q[0, 0, 0:8]  # q[t=0, kh=0, k=0:8]
    actual = o_cuda[0, 0, 0:8]

    print(f"  q[0,0,0:8]   = {expected.float().tolist()}")
    print(f"  smem dump     = {actual.float().tolist()}")
    match = torch.allclose(expected.float(), actual.float(), atol=1e-4)
    print(f"  smem q load:  {'PASS' if match else 'FAIL'}")


if __name__ == "__main__":
    print("=== Full correctness tests ===")
    all_pass = True
    for T, N in [(128, 1), (256, 2), (512, 4), (1024, 2), (973, 3), (4096, 8)]:
        ok = test_o_kernel(T=T, N=N)
        all_pass = all_pass and ok

    print(f"\n{'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")

    if all_pass:
        print("\n=== Performance benchmarks ===")
        for T, N in [(1024, 2), (4096, 4), (8192, 2)]:
            bench_o_kernel(T=T, N=N)
