cmake_minimum_required(VERSION 3.18)
project(tvm_ffi_scale LANGUAGES CXX CUDA)

# Prefer the repository venv Python when available.
get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(REPO_VENV_PYTHON "${REPO_ROOT}/.venv/bin/python")
if(NOT DEFINED Python_EXECUTABLE AND EXISTS "${REPO_VENV_PYTHON}")
    set(Python_EXECUTABLE "${REPO_VENV_PYTHON}" CACHE FILEPATH
        "Python interpreter used to locate tvm_ffi")
endif()

# Find Python and get tvm_ffi CMake directory.
find_package(Python COMPONENTS Interpreter REQUIRED)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m tvm_ffi.config --cmakedir
    RESULT_VARIABLE TVM_FFI_CMAKEDIR_STATUS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE tvm_ffi_DIR
    ERROR_VARIABLE TVM_FFI_CMAKEDIR_ERROR
)

# Find tvm_ffi package.
find_package(tvm_ffi CONFIG REQUIRED PATHS "${tvm_ffi_DIR}")

# Build your CUDA library
add_library(scale_kernel SHARED scale_kernel.cu)
tvm_ffi_configure_target(scale_kernel)
