cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

if(NOT DEFINED CUDAToolkit_ROOT AND DEFINED ENV{CUDA_PATH})
    set(CUDAToolkit_ROOT "$ENV{CUDA_PATH}")
endif()

project(AdaptiveGEMM LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(CUDAToolkit REQUIRED)

find_program(NVCC_EXECUTABLE
    NAMES nvcc
    HINTS
        "$ENV{CUDA_PATH}/bin"
        "$ENV{CUDA_PATH_V13_0}/bin"
        "$ENV{CUDA_PATH_V12_6}/bin"
)

if(NOT NVCC_EXECUTABLE)
    message(FATAL_ERROR "nvcc not found. Set CUDA_PATH to a valid CUDA Toolkit installation.")
endif()

list(GET CUDAToolkit_INCLUDE_DIRS 0 CUDA_INCLUDE_DIR)

set(ADAPTIVEGEMM_ROOT "${CMAKE_CURRENT_LIST_DIR}" CACHE PATH "Path to the standalone public release source tree")

set(ADAPTIVEGEMM_CU "${ADAPTIVEGEMM_ROOT}/src/AdaptiveOzaki.cu")
set(ADAPTIVEGEMM_OBJ "${CMAKE_CURRENT_BINARY_DIR}/AdaptiveOzaki.cu.obj")

if(MSVC)
    set(NVCC_EXTRA_FLAGS "-Xcompiler=/MD")
else()
    set(NVCC_EXTRA_FLAGS "-Xcompiler=-fPIC")
endif()

function(adaptive_nvcc_cu_obj CU_FILE OBJ_FILE)
    add_custom_command(
        OUTPUT "${OBJ_FILE}"
        COMMAND ${NVCC_EXECUTABLE}
            --std=c++17
            -O3
            -arch=sm_89
            ${NVCC_EXTRA_FLAGS}
            -I
            ${ADAPTIVEGEMM_ROOT}/include
            -I
            ${CUDA_INCLUDE_DIR}
            -c
            ${CU_FILE}
            -o
            ${OBJ_FILE}
        DEPENDS
            "${CU_FILE}"
            "${ADAPTIVEGEMM_ROOT}/include/AdaptiveOzaki.h"
            "${ADAPTIVEGEMM_ROOT}/include/scheme_i.h"
            "${ADAPTIVEGEMM_ROOT}/include/scheme_ii_fused.h"
            "${ADAPTIVEGEMM_ROOT}/include/scheme_ii_gpu.h"
            "${ADAPTIVEGEMM_ROOT}/include/scheme_ii_moduli.h"
            "${ADAPTIVEGEMM_ROOT}/include/scheme_ii_int192.h"
            "${ADAPTIVEGEMM_ROOT}/include/int8_gemm.h"
            "${ADAPTIVEGEMM_ROOT}/include/mma_sm89.cuh"
            "${ADAPTIVEGEMM_ROOT}/include/scheme_ii_mma_group.cuh"
        COMMENT "Compiling ${CU_FILE} with nvcc"
        VERBATIM
    )
endfunction()

set(ADAPTIVEGEMM_OBJ "${CMAKE_CURRENT_BINARY_DIR}/AdaptiveOzaki.cu.obj")
adaptive_nvcc_cu_obj("${ADAPTIVEGEMM_CU}" "${ADAPTIVEGEMM_OBJ}")

set(INT8GEMM_CU "${ADAPTIVEGEMM_ROOT}/src/int8_gemm.cu")
set(INT8GEMM_OBJ "${CMAKE_CURRENT_BINARY_DIR}/int8_gemm.cu.obj")
adaptive_nvcc_cu_obj("${INT8GEMM_CU}" "${INT8GEMM_OBJ}")

set(SCHEME_I_CU "${ADAPTIVEGEMM_ROOT}/src/scheme_i.cu")
set(SCHEME_I_OBJ "${CMAKE_CURRENT_BINARY_DIR}/scheme_i.cu.obj")
adaptive_nvcc_cu_obj("${SCHEME_I_CU}" "${SCHEME_I_OBJ}")

set(SCHEME_I_FUSED_CU "${ADAPTIVEGEMM_ROOT}/src/scheme_i_fused.cu")
set(SCHEME_I_FUSED_OBJ "${CMAKE_CURRENT_BINARY_DIR}/scheme_i_fused.cu.obj")
adaptive_nvcc_cu_obj("${SCHEME_I_FUSED_CU}" "${SCHEME_I_FUSED_OBJ}")

set(SCHEME_II_GPU_CU "${ADAPTIVEGEMM_ROOT}/src/scheme_ii_gpu.cu")
set(SCHEME_II_GPU_OBJ "${CMAKE_CURRENT_BINARY_DIR}/scheme_ii_gpu.cu.obj")
adaptive_nvcc_cu_obj("${SCHEME_II_GPU_CU}" "${SCHEME_II_GPU_OBJ}")

set(SCHEME_II_FUSED_CU "${ADAPTIVEGEMM_ROOT}/src/scheme_ii_fused.cu")
set(SCHEME_II_FUSED_OBJ "${CMAKE_CURRENT_BINARY_DIR}/scheme_ii_fused.cu.obj")
adaptive_nvcc_cu_obj("${SCHEME_II_FUSED_CU}" "${SCHEME_II_FUSED_OBJ}")

add_library(adaptive_gemm STATIC
    "${ADAPTIVEGEMM_OBJ}"
    "${INT8GEMM_OBJ}"
    "${SCHEME_I_OBJ}"
    "${SCHEME_I_FUSED_OBJ}"
    "${SCHEME_II_GPU_OBJ}"
    "${SCHEME_II_FUSED_OBJ}"
    "${ADAPTIVEGEMM_ROOT}/src/bench_int8.cpp"
    "${ADAPTIVEGEMM_ROOT}/src/scheme_ii_plan.cpp"
)

set_target_properties(adaptive_gemm PROPERTIES
    LINKER_LANGUAGE CXX
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(adaptive_gemm
    PUBLIC
    "${ADAPTIVEGEMM_ROOT}/include"
)

target_link_libraries(adaptive_gemm
    PUBLIC
        CUDA::cublas
        CUDA::cudart
)

# Optional Python binding via pybind11
option(ADAPTIVEGEMM_BUILD_PYTHON "Build Python bindings" OFF)
if(ADAPTIVEGEMM_BUILD_PYTHON)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)

    find_package(Python COMPONENTS Interpreter Development REQUIRED)

    pybind11_add_module(adaptive_gemm_py
            src/pybind_module.cpp
    )

    target_include_directories(adaptive_gemm_py PUBLIC ${ADAPTIVEGEMM_ROOT}/include)

    target_link_libraries(adaptive_gemm_py PRIVATE
        pybind11::module
        adaptive_gemm
        CUDA::cublas
        CUDA::cudart
    )
endif()

add_executable(adaptive_gemm_demo
    examples/demo_main.cpp
)

target_link_libraries(adaptive_gemm_demo
    PRIVATE
        adaptive_gemm
        CUDA::cublas
        CUDA::cudart
)

enable_testing()

add_executable(adaptive_gemm_precision_test
    tests/precision_test.cpp
)

target_link_libraries(adaptive_gemm_precision_test
    PRIVATE
        adaptive_gemm
        CUDA::cublas
        CUDA::cudart
)

add_test(NAME precision COMMAND adaptive_gemm_precision_test)

add_executable(adaptive_gemm_precision_contract_test
    tests/precision_contract_test.cpp
)

target_link_libraries(adaptive_gemm_precision_contract_test
    PRIVATE
        adaptive_gemm
        CUDA::cublas
        CUDA::cudart
)

add_test(NAME precision_contract COMMAND adaptive_gemm_precision_contract_test)

add_executable(adaptive_gemm_int8_test
    tests/int8_gemm_test.cpp
)

target_link_libraries(adaptive_gemm_int8_test
    PRIVATE
        adaptive_gemm
        CUDA::cublas
        CUDA::cudart
)

add_test(NAME int8_gemm COMMAND adaptive_gemm_int8_test)

add_executable(adaptive_gemm_scheme_i_test
    tests/scheme_i_precision_test.cpp
)

target_link_libraries(adaptive_gemm_scheme_i_test
    PRIVATE
        adaptive_gemm
        CUDA::cublas
        CUDA::cudart
)

add_test(NAME scheme_i COMMAND adaptive_gemm_scheme_i_test)

add_executable(adaptive_gemm_int8_roofline_test
    tests/int8_roofline_test.cpp
)

target_link_libraries(adaptive_gemm_int8_roofline_test
    PRIVATE
        adaptive_gemm
        CUDA::cublas
        CUDA::cudart
)

add_test(NAME int8_roofline COMMAND adaptive_gemm_int8_roofline_test)

add_executable(adaptive_gemm_scheme_ii_fused_compare_test tests/scheme_ii_fused_compare_test.cpp)
target_link_libraries(adaptive_gemm_scheme_ii_fused_compare_test PRIVATE adaptive_gemm CUDA::cublas CUDA::cudart)
add_test(NAME scheme_ii_fused COMMAND adaptive_gemm_scheme_ii_fused_compare_test)
add_test(NAME scheme_ii_fused_custom_g4 COMMAND adaptive_gemm_scheme_ii_fused_compare_test --scheme_ii_backend custom_mma --scheme_ii_group_size 4)
add_test(NAME scheme_ii_fused_partial_g4 COMMAND adaptive_gemm_scheme_ii_fused_compare_test --scheme_ii_backend custom_mma --scheme_ii_group_size 4 --use_true_fused_kernel)
add_test(NAME scheme_ii_fused_true COMMAND adaptive_gemm_scheme_ii_fused_compare_test --scheme_ii_backend custom_mma --scheme_ii_group_size 16 --use_true_fused_kernel)
add_test(NAME scheme_ii_fused_partial_g4_rect COMMAND adaptive_gemm_scheme_ii_fused_compare_test -M 768 -N 640 -K 1000 --precision_policy Auto --num_moduli 0 --scheme_ii_backend custom_mma --scheme_ii_group_size 4 --use_true_fused_kernel)

add_executable(adaptive_gemm_dd_test tests/dd_gemm_test.cpp)
target_link_libraries(adaptive_gemm_dd_test PRIVATE adaptive_gemm CUDA::cublas CUDA::cudart)
add_test(NAME dd_gemm COMMAND adaptive_gemm_dd_test)

add_executable(adaptive_gemm_dd_error_128_test tests/dd_error_128_test.cpp)
target_link_libraries(adaptive_gemm_dd_error_128_test PRIVATE adaptive_gemm CUDA::cublas CUDA::cudart)
add_test(NAME dd_error_128 COMMAND adaptive_gemm_dd_error_128_test)
if(ADAPTIVEGEMM_BUILD_PYTHON)
    add_test(NAME pytest_binding
        COMMAND ${Python_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_pybind.py
    )
    set_tests_properties(pytest_binding PROPERTIES
        ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:adaptive_gemm_py>"
    )
endif()
