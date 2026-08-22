// tests/int8_gemm_test.cpp
// PR1 correctness tests for int8_gemm_s8s8s32.
//
// Test matrix:
//  1. CPU gold n=128 and n=512 vs our kernel (random s8, int32 ref)
//  2. Negative test: 32^3 must return cudaErrorInvalidValue
//  3. n=128 and n=512 bit-exact vs cublasGemmEx(CUDA_R_8I, COMPUTE_32I)
//     (if cuBLAS returns NOT_SUPPORTED for that size, skip gracefully)
//
// Forbidden:
//  - n=32 as gold (violates alignment contract)
//  - 2048^3 CPU gold (~8e9 MAC)
//
// Every new launch followed by CHECK_CUDA(cudaGetLastError()).

#include "int8_gemm.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

using namespace ozaki;

// ---------------------------------------------------------------------------
// Error-checking macros
// ---------------------------------------------------------------------------
#define CHECK_CUDA(call)                                                          \
    do {                                                                          \
        cudaError_t _e = (call);                                                  \
        if (_e != cudaSuccess) {                                                  \
            std::cerr << "CUDA error " << cudaGetErrorString(_e)                 \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";          \
            std::exit(1);                                                         \
        }                                                                         \
    } while (0)

// ---------------------------------------------------------------------------
// CPU reference GEMM: C[m,n] = A[m,k] * B[k,n]  (int32 accumulator)
// ---------------------------------------------------------------------------
static void cpu_gemm_s8s8s32(
        const int8_t* A, const int8_t* B, int32_t* C,
        int m, int n_cols, int k)
{
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n_cols; ++j) {
            int32_t s = 0;
            for (int l = 0; l < k; ++l)
                s += (int32_t)A[(size_t)i*k+l] * (int32_t)B[(size_t)l*n_cols+j];
            C[(size_t)i*n_cols+j] = s;
        }
}

// ---------------------------------------------------------------------------
// Random s8 fill
// ---------------------------------------------------------------------------
static void fill_random_s8(std::vector<int8_t>& v, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(-127, 127);
    for (auto& x : v) x = static_cast<int8_t>(dist(rng));
}

// ---------------------------------------------------------------------------
// run_correctness_test: CPU gold + (optional) bit-exact vs cuBLAS
// ---------------------------------------------------------------------------
static bool run_correctness_test(int n, cublasHandle_t cblas_h)
{
    std::cout << "\n=== Correctness test n=" << n << " ===\n";

    size_t cnt = (size_t)n * n;
    std::vector<int8_t>  hA(cnt), hB(cnt);
    std::vector<int32_t> hC_ref(cnt, 0);
    fill_random_s8(hA, 42 + (unsigned)n);
    fill_random_s8(hB, 99 + (unsigned)n);

    // CPU reference (safe: n <= 512 so at most 512^3 = 134M ops)
    cpu_gemm_s8s8s32(hA.data(), hB.data(), hC_ref.data(), n, n, n);
    std::cout << "  CPU gold computed\n";

    // GPU allocations
    int8_t  *dA = nullptr, *dB = nullptr;
    int32_t *dC_ours = nullptr, *dC_cublas = nullptr;
    CHECK_CUDA(cudaMalloc(&dA,        cnt * sizeof(int8_t)));
    CHECK_CUDA(cudaMalloc(&dB,        cnt * sizeof(int8_t)));
    CHECK_CUDA(cudaMalloc(&dC_ours,   cnt * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&dC_cublas, cnt * sizeof(int32_t)));
    CHECK_CUDA(cudaMemcpy(dA, hA.data(), cnt, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), cnt, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC_ours,   0, cnt * sizeof(int32_t)));
    CHECK_CUDA(cudaMemset(dC_cublas, 0, cnt * sizeof(int32_t)));

    // ---- Our kernel ----
    cudaError_t launch_err = int8_gemm_s8s8s32(dA, dB, dC_ours, n, n, n, 0);
    CHECK_CUDA(launch_err);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy our result to host
    std::vector<int32_t> hC_ours(cnt);
    CHECK_CUDA(cudaMemcpy(hC_ours.data(), dC_ours, cnt * sizeof(int32_t), cudaMemcpyDeviceToHost));

    // Compare our vs CPU gold
    bool vs_cpu_ok = true;
    int mismatch_cpu = 0;
    for (size_t i = 0; i < cnt; ++i) {
        if (hC_ours[i] != hC_ref[i]) {
            if (mismatch_cpu < 5)
                std::cerr << "  CPU gold mismatch [" << i/n << "," << i%n << "]: "
                          << "ours=" << hC_ours[i] << " ref=" << hC_ref[i] << "\n";
            mismatch_cpu++;
            vs_cpu_ok = false;
        }
    }
    if (vs_cpu_ok)
        std::cout << "  [PASS] our kernel vs CPU gold: bit-exact\n";
    else
        std::cout << "  [FAIL] our kernel vs CPU gold: " << mismatch_cpu << " mismatches\n";

    // ---- cuBLAS INT8 ----
    // C=A*B row-major == GemmEx(N,N, n,m,k, dB,n, dA,n, dC,n, COMPUTE_32I)
    // cuBLAS INT8 may return NOT_SUPPORTED for certain sizes/drivers — skip gracefully.
    int32_t alpha32 = 1, beta32 = 0;
    cublasStatus_t cb_err = cublasGemmEx(cblas_h,
                                         CUBLAS_OP_N, CUBLAS_OP_N,
                                         n, n, n,
                                         &alpha32, dB, CUDA_R_8I, n,
                                                   dA, CUDA_R_8I, n,
                                         &beta32,  dC_cublas, CUDA_R_32I, n,
                                         CUBLAS_COMPUTE_32I,
                                         CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    CHECK_CUDA(cudaGetLastError());

    bool vs_cublas_ok = true;
    if (cb_err == CUBLAS_STATUS_NOT_SUPPORTED) {
        std::cout << "  [SKIP] cublasGemmEx INT8 NOT_SUPPORTED for n=" << n
                  << " — bit-exact comparison skipped\n";
    } else if (cb_err != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "  [FAIL] cublasGemmEx INT8 error=" << cb_err << " for n=" << n << "\n";
        vs_cublas_ok = false;
    } else {
        CHECK_CUDA(cudaDeviceSynchronize());
        std::vector<int32_t> hC_cublas(cnt);
        CHECK_CUDA(cudaMemcpy(hC_cublas.data(), dC_cublas, cnt * sizeof(int32_t),
                              cudaMemcpyDeviceToHost));

        int mismatch_cb = 0;
        for (size_t i = 0; i < cnt; ++i) {
            if (hC_ours[i] != hC_cublas[i]) {
                if (mismatch_cb < 5)
                    std::cerr << "  cuBLAS mismatch [" << i/n << "," << i%n << "]: "
                              << "ours=" << hC_ours[i] << " cublas=" << hC_cublas[i] << "\n";
                mismatch_cb++;
                vs_cublas_ok = false;
            }
        }
        if (vs_cublas_ok)
            std::cout << "  [PASS] our kernel vs cublasGemmEx INT8: bit-exact\n";
        else
            std::cout << "  [FAIL] our kernel vs cublasGemmEx INT8: " << mismatch_cb << " mismatches\n";
    }

    cudaFree(dA); cudaFree(dB); cudaFree(dC_ours); cudaFree(dC_cublas);
    return vs_cpu_ok && vs_cublas_ok;
}

// ---------------------------------------------------------------------------
// Negative tests
// ---------------------------------------------------------------------------
static bool run_negative_test_32()
{
    std::cout << "\n=== Negative test (32^3 -> cudaErrorInvalidValue) ===\n";
    int8_t  *dA, *dB; int32_t *dC;
    CHECK_CUDA(cudaMalloc(&dA, 32*32));
    CHECK_CUDA(cudaMalloc(&dB, 32*32));
    CHECK_CUDA(cudaMalloc(&dC, 32*32*sizeof(int32_t)));
    cudaError_t err = int8_gemm_s8s8s32(dA, dB, dC, 32, 32, 32, 0);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    if (err == cudaErrorInvalidValue) {
        std::cout << "  [PASS] 32^3 returned cudaErrorInvalidValue\n";
        return true;
    }
    std::cerr << "  [FAIL] 32^3 returned " << cudaGetErrorString(err) << "\n";
    return false;
}

static bool run_negative_test_k()
{
    std::cout << "\n=== Negative test (128x128x32: k%64!=0) ===\n";
    int8_t  *dA, *dB; int32_t *dC;
    CHECK_CUDA(cudaMalloc(&dA, 128*32));
    CHECK_CUDA(cudaMalloc(&dB, 32*128));
    CHECK_CUDA(cudaMalloc(&dC, 128*128*sizeof(int32_t)));
    cudaError_t err = int8_gemm_s8s8s32(dA, dB, dC, 128, 128, 32, 0);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    if (err == cudaErrorInvalidValue) {
        std::cout << "  [PASS] 128x128x32 returned cudaErrorInvalidValue\n";
        return true;
    }
    std::cerr << "  [FAIL] 128x128x32 returned " << cudaGetErrorString(err) << "\n";
    return false;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main()
{
    std::cout << "=== adaptive_gemm_int8_test (PR1) ===\n";

    cublasHandle_t cblas_h;
    if (cublasCreate(&cblas_h) != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cublasCreate failed\n"; return 1;
    }

    bool ok = true;
    ok &= run_negative_test_32();
    ok &= run_negative_test_k();
    ok &= run_correctness_test(128, cblas_h);
    ok &= run_correctness_test(512, cblas_h);

    cublasDestroy(cblas_h);
    cudaDeviceSynchronize();

    if (ok) { std::cout << "\n[ALL PASS] int8_gemm_test\n"; return 0; }
    std::cerr << "\n[SOME TESTS FAILED]\n"; return 1;
}
