file(GLOB_RECURSE SRC "*.cu" "*.cpp")
add_library(si_moe STATIC ${SRC})
target_include_directories(si_moe PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${cutlass_SOURCE_DIR}/include
)
target_compile_options(si_moe PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3 --use_fast_math -lineinfo --ptxas-options=-v
        --generate-line-info
    >
)
