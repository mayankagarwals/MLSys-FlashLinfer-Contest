# Minimal TVM FFI Example

Adapted from [TVM FFI repo](https://github.com/apache/tvm-ffi/tree/main/examples/kernel_library).

## Run From Repo Root

These commands assume your current working directory is the repository root.

### 1. Create a virtual environment and install dependencies

```bash
uv venv
uv pip install --python .venv/bin/python numpy torch apache-tvm-ffi
```

### 2. Build the CUDA shared library

```bash
cmake -S other/tvm_ffi_example -B build
cmake --build build
```

### 3. Run the example

```bash
uv run other/tvm_ffi_example/load_scale.py
```

The script loads `build/libscale_kernel.so` from the repository root build directory.

## Troubleshooting

If CMake reports `No module named 'tvm_ffi'`, it is using a Python interpreter
that does not have `apache-tvm-ffi` installed. Re-run step 1, or specify an
interpreter explicitly:

```bash
cmake -S other/tvm_ffi_example -B build -DPython_EXECUTABLE=$PWD/.venv/bin/python
```
