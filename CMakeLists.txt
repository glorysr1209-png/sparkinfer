cmake_minimum_required(VERSION 3.20)
project(sparkinfer_kernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Primary target: sm_100 = Blackwell (RTX Spark, RTX 5090, GB200)
# Also support: sm_90 = Hopper (H100), sm_89 = Ada (RTX 4090)
set(CMAKE_CUDA_ARCHITECTURES "89;90;100" CACHE STRING "CUDA architectures")

option(BUILD_TESTS      "Build unit tests"  ON)
option(BUILD_BENCHMARKS "Build benchmarks"  ON)

# CUTLASS — provides CuTe DSL, tensor layouts, TMA, WGMMA abstractions
# CuTe DSL is the preferred way to write high-performance kernels on Blackwell.
# All fused kernels (GroupGEMM+SwiGLU, etc.) use CuTe epilogue visitors.
include(FetchContent)
FetchContent_Declare(
    cutlass
    GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
    GIT_TAG        main
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(cutlass)

include_directories(include)

# --- Native CUDA kernel domains ---
add_subdirectory(csrc/cuda/attention)   # flash decode variants (GQA, hd256, hd512)
add_subdirectory(csrc/cuda/gemm)        # batched GEMM, GEMV
add_subdirectory(csrc/cuda/moe)         # MoE router, token dispatch
add_subdirectory(csrc/cuda/quant)       # quantization helpers
add_subdirectory(csrc/cuda/fused)       # CODA-style GEMM+epilogue (non-CuTe path)

# --- CuTe DSL kernel domains ---
# CuTe provides: tensor layout algebra, TMA (async bulk copy),
# WGMMA (warpgroup MMA), persistent kernel support, epilogue visitors.
# These are the highest-performance kernels targeting sm_100.
add_subdirectory(csrc/cute/moe_swiglu)  # GroupGEMM + SwiGLU epilogue (sync-free)
add_subdirectory(csrc/cute/moe_gemm)    # sync-free GroupGEMM (token counts on GPU)

# Unified interface library
add_library(sparkinfer_kernels INTERFACE)
target_link_libraries(sparkinfer_kernels INTERFACE
    si_attn
    si_gemm
    si_moe
    si_quant
    si_fused
    si_cute_moe_swiglu
    si_cute_moe_gemm
)
target_include_directories(sparkinfer_kernels INTERFACE
    include
    ${cutlass_SOURCE_DIR}/include          # CUTLASS headers
    ${cutlass_SOURCE_DIR}/tools/util/include
)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
