from pathlib import Path

import tvm_ffi

CURRENT_DIR = Path(__file__).parent

lib_path = tvm_ffi.cpp.build(
    name="gdn_decode_kernel_7_mix",
    cuda_files=[str(CURRENT_DIR / "gdn_decode_kernel_7.cu")],
    extra_cflags=["-O3"],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "-lineinfo",
    ],
)
mod = tvm_ffi.load_module(lib_path)
run = mod.gdn_decode_v7
