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

add_custom_command(
    OUTPUT "${ADAPTIVEGEMM_OBJ}"
    COMMAND ${NVCC_EXECUTABLE}
        --std=c++17
        -arch=sm_89
        -Xcompiler=/MD
        -I
        ${ADAPTIVEGEMM_ROOT}/include
        -I
        ${CUDA_INCLUDE_DIR}
        -c
        ${ADAPTIVEGEMM_CU}
        -o
        ${ADAPTIVEGEMM_OBJ}
    DEPENDS "${ADAPTIVEGEMM_CU}"
    COMMENT "Compiling AdaptiveOzaki.cu with nvcc"
    VERBATIM
)

add_library(adaptive_gemm STATIC
    "${ADAPTIVEGEMM_OBJ}"
)

set_target_properties(adaptive_gemm PROPERTIES
    LINKER_LANGUAGE CXX
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

# Optional Python binding (CPU reference GEMM) via pybind11
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

target_link_libraries(adaptive_gemm_py PRIVATE pybind11::module)

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