cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

if(NOT DEFINED CUDAToolkit_ROOT AND DEFINED ENV{CUDA_PATH})
    set(CUDAToolkit_ROOT "$ENV{CUDA_PATH}")
endif()

project(AdaptiveOzakiPublic LANGUAGES CXX)

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

set(ADAPTIVEOZAKI_ROOT "${CMAKE_CURRENT_LIST_DIR}" CACHE PATH "Path to the standalone public release source tree")

set(ADAPTIVEOZAKI_CU "${ADAPTIVEOZAKI_ROOT}/src/AdaptiveOzaki.cu")
set(ADAPTIVEOZAKI_OBJ "${CMAKE_CURRENT_BINARY_DIR}/AdaptiveOzaki.cu.obj")

add_custom_command(
    OUTPUT "${ADAPTIVEOZAKI_OBJ}"
    COMMAND ${NVCC_EXECUTABLE}
        --std=c++17
        -arch=sm_89
        -Xcompiler=/MD
        -I
        ${ADAPTIVEOZAKI_ROOT}/include
        -I
        ${CUDA_INCLUDE_DIR}
        -c
        ${ADAPTIVEOZAKI_CU}
        -o
        ${ADAPTIVEOZAKI_OBJ}
    DEPENDS "${ADAPTIVEOZAKI_CU}"
    COMMENT "Compiling AdaptiveOzaki.cu with nvcc"
    VERBATIM
)

add_library(adaptive_ozaki STATIC
    "${ADAPTIVEOZAKI_OBJ}"
)

set_target_properties(adaptive_ozaki PROPERTIES
    LINKER_LANGUAGE CXX
)

target_include_directories(adaptive_ozaki
    PUBLIC
    "${ADAPTIVEOZAKI_ROOT}/include"
)

target_link_libraries(adaptive_ozaki
    PUBLIC
        CUDA::cublas
        CUDA::cudart
)

add_executable(adaptive_ozaki_demo
    examples/demo_main.cpp
)

target_link_libraries(adaptive_ozaki_demo
    PRIVATE
        adaptive_ozaki
        CUDA::cublas
        CUDA::cudart
)

enable_testing()

add_executable(adaptive_ozaki_precision_test
    tests/precision_test.cpp
)

target_link_libraries(adaptive_ozaki_precision_test
    PRIVATE
        adaptive_ozaki
        CUDA::cublas
        CUDA::cudart
)

add_test(NAME precision COMMAND adaptive_ozaki_precision_test)