# from pathlib import Path

# import torch
# from torch.utils.cpp_extension import load

# CURRENT_DIR = Path(__file__).parent

# load(
#     name="my_module",
#     sources=[str(CURRENT_DIR / "gdn_decode_thien_v1.cu")],
#     extra_cflags=["-O3"],
#     extra_cuda_cflags=[
#         "-O3",
#         "-gencode=arch=compute_100a,code=sm_100a",
#         "--use_fast_math",
#         "-lineinfo",
#     ],
#     is_python_module=False,
# )
# run = torch.ops.my_module.gdn_decode

from pathlib import Path

import tvm_ffi

CURRENT_DIR = Path(__file__).parent

lib_path = tvm_ffi.cpp.build(
    name="my_package",
    cuda_files=[str(CURRENT_DIR / "gdn_decode_thien_v1.cu")],
    extra_cflags=["-O3"],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "-lineinfo",
    ],
)
mod = tvm_ffi.load_module(lib_path)
run = mod.gdn_decode_thien_v1
