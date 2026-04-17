from pathlib import Path
import tvm_ffi

CURRENT_DIR = Path(__file__).parent

lib_path = tvm_ffi.cpp.build(
    name="gdn_decode_smem_pipeline_1",
    cuda_files=[str(CURRENT_DIR / "gdn_decode_smem_pipeline_1.cu")],
    extra_cflags=["-O3"],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "-lineinfo",
    ],
)
mod = tvm_ffi.load_module(lib_path)
run = mod.gdn_decode_smem_pipeline_1
