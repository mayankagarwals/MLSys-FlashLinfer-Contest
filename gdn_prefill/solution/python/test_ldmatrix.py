"""Test ldmatrix + mma.sync on SM100a to verify register mapping."""
import os
os.environ["TVM_FFI_CUDA_ARCH_LIST"] = "10.0a"
from pathlib import Path
import torch
import tvm_ffi

CURRENT_DIR = Path(__file__).parent
lib = tvm_ffi.cpp.build(
    name="test_ldmatrix",
    cuda_files=[str(CURRENT_DIR / "test_ldmatrix.cu")],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "-lineinfo"],
    extra_ldflags=["-lcuda"],
)
mod = tvm_ffi.load_module(lib)

def test_1tile():
    """Test single 16x8 MMA: A[16,16] x B[16,8] = C[16,8]"""
    A = torch.randn(16, 16, dtype=torch.bfloat16, device="cuda")
    BT = torch.randn(8, 16, dtype=torch.bfloat16, device="cuda")  # B^T
    C = torch.zeros(16, 8, dtype=torch.float32, device="cuda")

    mod.test_mma_1tile(A, BT, C)
    torch.cuda.synchronize()

    # Reference: C = A @ B where B = BT.T
    B = BT.T.contiguous()
    C_ref = A.float() @ B.float()

    diff = (C - C_ref).abs()
    max_diff = diff.max().item()
    print(f"test_1tile: max_diff={max_diff:.6f}  {'PASS' if max_diff < 0.1 else 'FAIL'}")
    if max_diff >= 0.1:
        # Print first failing element
        idx = torch.where(diff > 0.1)
        if len(idx[0]) > 0:
            r, c = idx[0][0].item(), idx[1][0].item()
            print(f"  [{r},{c}]: ref={C_ref[r,c]:.4f} cuda={C[r,c]:.4f} diff={diff[r,c]:.4f}")
        # Print first few elements for comparison
        print(f"  C_ref[0,:4] = {C_ref[0,:4].tolist()}")
        print(f"  C_cuda[0,:4] = {C[0,:4].tolist()}")
        print(f"  C_ref[8,:4] = {C_ref[8,:4].tolist()}")
        print(f"  C_cuda[8,:4] = {C[8,:4].tolist()}")

def test_full_k():
    """Test full K accumulation: A[64,128] x BT^T[128,64] → check one 16x8 tile."""
    A = torch.randn(64, 128, dtype=torch.bfloat16, device="cuda") * 0.1
    BT = torch.randn(64, 128, dtype=torch.bfloat16, device="cuda") * 0.1
    # BT is B^T [N=64, K=128]. B = BT^T [K=128, N=64].
    # C = A @ B = A @ BT^T = [64, 64]

    C_ref = A.float() @ BT.T.float().contiguous()

    for m_tile in range(4):
        for n_tile in [0, 3, 7]:
            C = torch.zeros(16, 8, dtype=torch.float32, device="cuda")
            mod.test_mma_full_k(A, BT, C, m_tile, n_tile)
            torch.cuda.synchronize()

            C_ref_tile = C_ref[m_tile*16:(m_tile+1)*16, n_tile*8:(n_tile+1)*8]
            diff = (C - C_ref_tile).abs()
            max_diff = diff.max().item()
            status = "PASS" if max_diff < 0.5 else "FAIL"
            print(f"  m={m_tile} n={n_tile}: max_diff={max_diff:.4f} {status}")
            if max_diff >= 0.5:
                idx = torch.where(diff == diff.max())
                r, c = idx[0][0].item(), idx[1][0].item()
                print(f"    [{r},{c}]: ref={C_ref_tile[r,c]:.4f} cuda={C[r,c]:.4f}")

def test_manual_b():
    """Test with manual B packing (no ldmatrix_trans)."""
    A = torch.randn(16, 16, dtype=torch.bfloat16, device="cuda")
    BT = torch.randn(8, 16, dtype=torch.bfloat16, device="cuda")
    C = torch.zeros(16, 8, dtype=torch.float32, device="cuda")
    C_ref = A.float() @ BT.T.float().contiguous()

    mod.test_mma_manual_b(A, BT, C)
    torch.cuda.synchronize()
    diff = (C - C_ref).abs()
    max_diff = diff.max().item()
    print(f"manual_b (standard D): max_diff={max_diff:.6f}  {'PASS' if max_diff < 0.5 else 'FAIL'}")
    if max_diff >= 0.5:
        print(f"  C_ref[0,:4] = {C_ref[0,:4].tolist()}")
        print(f"  C_cuda[0,:4] = {C[0,:4].tolist()}")
        print(f"  C_ref[8,:4] = {C_ref[8,:4].tolist()}")
        print(f"  C_cuda[8,:4] = {C[8,:4].tolist()}")

    # Now try SM100a D mapping
    C2 = torch.zeros(16, 8, dtype=torch.float32, device="cuda")
    mod.test_mma_sm100a_d(A, BT, C2)
    torch.cuda.synchronize()
    diff2 = (C2 - C_ref).abs()
    max_diff2 = diff2.max().item()
    print(f"manual_b (SM100a D):   max_diff={max_diff2:.6f}  {'PASS' if max_diff2 < 0.5 else 'FAIL'}")
    if max_diff2 < 0.5:
        print(f"  *** SM100a D mapping works for bf16 m16n8k16! ***")
    if max_diff2 >= 0.5:
        print(f"  C_ref[0,:4] = {C_ref[0,:4].tolist()}")
        print(f"  C_sm100a[0,:4] = {C2[0,:4].tolist()}")


def test_multi_tile_matmul():
    """Test multi-tile matmul with 4 warps using direct uint32 loads, matching O-kernel pattern."""
    # This is the exact q@h^T computation from the O-kernel
    # A = q [64, 128] bf16, B^T = h [64, 128] bf16
    # C = A @ B = q @ h^T = [64, 64]
    q = torch.randn(64, 128, dtype=torch.bfloat16, device="cuda") * 0.1
    h = torch.randn(64, 128, dtype=torch.bfloat16, device="cuda") * 0.1

    C_ref = q.float() @ h.T.float().contiguous()  # [64, 64]

    # Use the test_mma_full_k kernel (1 warp) for each M-tile, N-tile
    for m_tile in range(4):
        for n_tile in range(8):
            C_tile = torch.zeros(16, 8, dtype=torch.float32, device="cuda")
            mod.test_mma_full_k(q, h, C_tile, m_tile, n_tile)
            torch.cuda.synchronize()

            C_ref_tile = C_ref[m_tile*16:(m_tile+1)*16, n_tile*8:(n_tile+1)*8]
            diff = (C_tile - C_ref_tile).abs().max().item()
            if diff > 0.5:
                print(f"  FAIL m={m_tile} n={n_tile}: max_diff={diff:.4f}")
                return

    print(f"  All 32 tiles pass (max_diff < 0.5)")


if __name__ == "__main__":
    print("=== Manual B packing test ===")
    test_manual_b()
    print("\n=== Multi-tile matmul test (using ldmatrix_trans) ===")
    test_multi_tile_matmul()
    print("\n=== ldmatrix_trans single tile test ===")
    test_1tile()
