#include "int8_gemm.h"
#include <vector>
#include <random>
#include <iostream>
#include <cublas_v2.h>

#define CHECK_CUDA(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " "   \
                      << cudaGetErrorString(err) << "\n";                         \
            std::exit(1);                                                         \
        }                                                                         \
    } while (0)

namespace ozaki {

static void fill_random_s8(std::vector<int8_t>& v, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(-127, 127);
    for (auto& x : v) x = static_cast<int8_t>(dist(rng));
}

Int8GemmBench bench_int8_gemm_vs_cublas(int m, int n, int k) {
    Int8GemmBench res = {0};
    
    size_t sizeA = (size_t)m * k;
    size_t sizeB = (size_t)k * n;
    size_t sizeC = (size_t)m * n;
    std::vector<int8_t> hA(sizeA), hB(sizeB);
    fill_random_s8(hA, 42);
    fill_random_s8(hB, 99);

    int8_t  *dA, *dB;
    int32_t *dC;
    CHECK_CUDA(cudaMalloc(&dA, sizeA * sizeof(int8_t)));
    CHECK_CUDA(cudaMalloc(&dB, sizeB * sizeof(int8_t)));
    CHECK_CUDA(cudaMalloc(&dC, sizeC * sizeof(int32_t)));
    CHECK_CUDA(cudaMemcpy(dA, hA.data(), sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), sizeB, cudaMemcpyHostToDevice));
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    cublasHandle_t cblas_h;
    if (cublasCreate(&cblas_h) != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cublasCreate failed in bench\n";
        std::exit(1);
    }
    
    int runs = 100;
    
    // Warmup for our kernel
    for(int i=0; i<10; i++) {
        int8_gemm_s8s8s32(dA, dB, dC, m, n, k, 0);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaEventRecord(start));
    for(int i=0; i<runs; i++) {
        int8_gemm_s8s8s32(dA, dB, dC, m, n, k, 0);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float our_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&our_ms, start, stop));
    res.our_ms = our_ms / runs;
    
    int32_t *dC_cublas;
    CHECK_CUDA(cudaMalloc(&dC_cublas, sizeC * sizeof(int32_t)));

    // Warmup for cuBLAS
    int32_t alpha32 = 1, beta32 = 0;
    for(int i=0; i<10; i++) {
        cublasGemmEx(cblas_h, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha32, dB, CUDA_R_8I, n, dA, CUDA_R_8I, k, &beta32, dC_cublas, CUDA_R_32I, n, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaEventRecord(start));
    for(int i=0; i<runs; i++) {
        cublasGemmEx(cblas_h, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha32, dB, CUDA_R_8I, n, dA, CUDA_R_8I, k, &beta32, dC_cublas, CUDA_R_32I, n, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float cublas_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&cublas_ms, start, stop));
    res.cublas_ms = cublas_ms / runs;

    std::vector<int32_t> hC(sizeC), hC_cublas(sizeC);
    CHECK_CUDA(cudaMemcpy(hC.data(), dC, sizeC * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hC_cublas.data(), dC_cublas, sizeC * sizeof(int32_t), cudaMemcpyDeviceToHost));

    int32_t max_err = 0;
    for (size_t i = 0; i < sizeC; ++i) {
        int32_t err = std::abs(hC[i] - hC_cublas[i]);
        if (err > max_err) max_err = err;
    }
    res.max_abs_int32 = max_err;
    
    double ops = 2.0 * m * n * k;
    res.our_tops = ops / (res.our_ms * 1e-3) / 1e12;
    res.cublas_tops = ops / (res.cublas_ms * 1e-3) / 1e12;
    res.peak_frac = res.our_tops / 121.0;
    res.vs_cublas = res.our_tops / res.cublas_tops;
    
    cublasDestroy(cblas_h);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFree(dC_cublas);
    
    return res;
}
} // namespace ozaki
