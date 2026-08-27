#include "scheme_ii_gpu.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <cmath>
#include <algorithm>

namespace ozaki {

struct Int192 {
    uint32_t limbs[6];
    
    __host__ __device__ Int192() {
        #pragma unroll
        for(int i=0; i<6; ++i) limbs[i] = 0;
    }
    
    __host__ __device__ void add(uint32_t val) {
        uint64_t carry = val;
        #pragma unroll
        for(int i=0; i<6; ++i) {
            uint64_t sum = limbs[i] + carry;
            limbs[i] = (uint32_t)sum;
            carry = sum >> 32;
        }
    }
    
    __host__ __device__ void mul(uint32_t val) {
        uint64_t carry = 0;
        #pragma unroll
        for(int i=0; i<6; ++i) {
            uint64_t prod = (uint64_t)limbs[i] * val + carry;
            limbs[i] = (uint32_t)prod;
            carry = prod >> 32;
        }
    }
    
    __device__ void add_mul_scalar(const Int192& P, uint32_t v) {
        uint64_t carry = 0;
        #pragma unroll
        for(int i=0; i<6; ++i) {
            uint64_t prod = (uint64_t)P.limbs[i] * v + limbs[i] + carry;
            limbs[i] = (uint32_t)prod;
            carry = prod >> 32;
        }
    }
    
    __host__ __device__ void shr1() {
        uint32_t carry = 0;
        for(int i=5; i>=0; --i) {
            uint32_t next_carry = (limbs[i] & 1) << 31;
            limbs[i] = (limbs[i] >> 1) | carry;
            carry = next_carry;
        }
    }

    __device__ bool is_greater_than(const Int192& b) const {
        #pragma unroll
        for(int i=5; i>=0; --i) {
            if (limbs[i] > b.limbs[i]) return true;
            if (limbs[i] < b.limbs[i]) return false;
        }
        return false;
    }

    __host__ __device__ double to_double() const {
        double r = 0;
        double base = 1.0;
        #pragma unroll
        for(int i=0; i<6; ++i) {
            r += limbs[i] * base;
            base *= 4294967296.0;
        }
        return r;
    }
};

__global__ void compute_exp_A_kernel(const double* A, int* eA, int m, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        double max_val = 0.0;
        for(int kk=0; kk<k; ++kk) {
            double v = fabs(A[(size_t)i*k + kk]);
            if (v > max_val) max_val = v;
        }
        int exp = 0;
        if (max_val > 0.0) frexp(max_val, &exp);
        eA[i] = exp;
    }
}

__global__ void compute_exp_B_kernel(const double* B, int* eB, int k, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) {
        double max_val = 0.0;
        for(int kk=0; kk<k; ++kk) {
            double v = fabs(B[(size_t)kk*n + j]);
            if (v > max_val) max_val = v;
        }
        int exp = 0;
        if (max_val > 0.0) frexp(max_val, &exp);
        eB[j] = exp;
    }
}

__global__ void quantize_A_kernel(const double* __restrict__ A, int8_t* __restrict__ A_mod, const int* __restrict__ eA, 
                                  int m, int k, int m_pad, int k_pad, int num_moduli, const int32_t* __restrict__ primes, int S) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int kk = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m && kk < k) {
        double val = A[(size_t)i * k + kk];
        int exp = eA[i];
        double scaled = val * ldexp(1.0, S - exp);
        int64_t v_int = llround(scaled);
        for (int p_idx = 0; p_idx < num_moduli; ++p_idx) {
            int p = primes[p_idx];
            int rem = v_int % p;
            if (rem < 0) rem += p;
            if (rem > p / 2) rem -= p;
            A_mod[(size_t)p_idx * m_pad * k_pad + i * k_pad + kk] = rem;
        }
    } else if (i < m_pad && kk < k_pad) {
        for (int p_idx = 0; p_idx < num_moduli; ++p_idx) {
            A_mod[(size_t)p_idx * m_pad * k_pad + i * k_pad + kk] = 0;
        }
    }
}

__global__ void quantize_B_kernel(const double* __restrict__ B, int8_t* __restrict__ B_mod, const int* __restrict__ eB, 
                                  int k, int n, int k_pad, int n_pad, int num_moduli, const int32_t* __restrict__ primes, int S) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int kk = blockIdx.y * blockDim.y + threadIdx.y;
    if (kk < k && j < n) {
        double val = B[(size_t)kk * n + j];
        int exp = eB[j];
        double scaled = val * ldexp(1.0, S - exp);
        int64_t v_int = llround(scaled);
        for (int p_idx = 0; p_idx < num_moduli; ++p_idx) {
            int p = primes[p_idx];
            int rem = v_int % p;
            if (rem < 0) rem += p;
            if (rem > p / 2) rem -= p;
            B_mod[(size_t)p_idx * k_pad * n_pad + kk * n_pad + j] = rem;
        }
    } else if (kk < k_pad && j < n_pad) {
        for (int p_idx = 0; p_idx < num_moduli; ++p_idx) {
            B_mod[(size_t)p_idx * k_pad * n_pad + kk * n_pad + j] = 0;
        }
    }
}

__global__ void crt_reconstruct_kernel(const int32_t* __restrict__ C_mod, double* __restrict__ C, 
                                       const int* __restrict__ eA, const int* __restrict__ eB, 
                                       int m, int n, int m_pad, int n_pad, int num_moduli, int S, 
                                       const int32_t* __restrict__ primes, const int32_t* __restrict__ inv_table, 
                                       Int192 M, Int192 M_half, const Int192* __restrict__ P) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < m && j < n) {
        int v[20];
        for(int p_idx = 0; p_idx < num_moduli; ++p_idx) {
            int r = C_mod[(size_t)p_idx * m_pad * n_pad + i * n_pad + j];
            int p = primes[p_idx];
            r = r % p;
            if (r < 0) r += p;
            v[p_idx] = r;
        }
        
        for(int p_idx = 0; p_idx < num_moduli - 1; ++p_idx) {
            int v_ii = v[p_idx];
            for(int q_idx = p_idx + 1; q_idx < num_moduli; ++q_idx) {
                int p = primes[q_idx];
                long long diff = v[q_idx] - v_ii;
                diff = diff % p;
                if (diff < 0) diff += p;
                v[q_idx] = (diff * inv_table[p_idx * 20 + q_idx]) % p;
            }
        }
        
        Int192 X;
        for(int p_idx = 0; p_idx < num_moduli; ++p_idx) {
            X.add_mul_scalar(P[p_idx], v[p_idx]);
        }
        
        bool is_neg = X.is_greater_than(M_half);
        if (is_neg) {
            Int192 diff;
            uint64_t borrow = 0;
            #pragma unroll
            for(int k_idx = 0; k_idx < 6; ++k_idx) {
                long long d = (long long)M.limbs[k_idx] - borrow - X.limbs[k_idx];
                if (d < 0) {
                    d += 4294967296LL;
                    borrow = 1;
                } else {
                    borrow = 0;
                }
                diff.limbs[k_idx] = (uint32_t)d;
            }
            X = diff;
        }
        
        double C_int_dbl = X.to_double();
        if (is_neg) C_int_dbl = -C_int_dbl;
        
        int exp_sum = 2 * S - eA[i] - eB[j];
        double final_C = C_int_dbl * ldexp(1.0, -exp_sum);
        C[(size_t)i * n + j] = final_C;
    }
}

static int modInverse(int a, int m) {
    int m0 = m, y = 0, x = 1;
    if (m == 1) return 0;
    while (a > 1) {
        int q = a / m;
        int t = m;
        m = a % m, a = t;
        t = y;
        y = x - q * y;
        x = t;
    }
    if (x < 0) x += m0;
    return x;
}

void execute_scheme_ii_gpu(cublasHandle_t handle, const double* d_A, const double* d_B, double* d_C, 
                           int m, int n, int k, int num_moduli) {
    std::cout << "[Scheme II GPU] executing with num_moduli = " << num_moduli << std::endl;

    int m_pad = ((m + 127) / 128) * 128;
    int n_pad = ((n + 127) / 128) * 128;
    int k_pad = ((k + 31) / 32) * 32;
    
    static const int32_t MODULI_INT8[20] = {
        256, 255, 253, 251, 247, 241, 239, 233,
        229, 227, 223, 217, 211, 199, 197, 193,
        191, 181, 179, 173
    };
    if (num_moduli > 20) num_moduli = 20;

    int32_t h_primes[20];
    int32_t h_inv_table[400];
    for (int i=0; i<num_moduli; ++i) {
        h_primes[i] = MODULI_INT8[i];
        for (int j=i+1; j<num_moduli; ++j) {
            h_inv_table[i * 20 + j] = modInverse(MODULI_INT8[i], MODULI_INT8[j]);
        }
    }
    
    Int192 M;
    M.add(1);
    for(int i=0; i<num_moduli; ++i) M.mul(MODULI_INT8[i]);
    
    Int192 M_half = M;
    M_half.shr1();
    
    Int192 h_P[20];
    h_P[0].add(1);
    for(int i=1; i<num_moduli; ++i) {
        h_P[i] = h_P[i-1];
        h_P[i].mul(MODULI_INT8[i-1]);
    }
    
    double max_val = M.to_double() / (2.0 * k);
    int S = std::floor(std::log2(max_val) / 2.0);
    if (S > 53) S = 53;

    int32_t* d_primes;
    int32_t* d_inv_table;
    Int192* d_P;
    cudaMalloc(&d_primes, 20 * sizeof(int32_t));
    cudaMalloc(&d_inv_table, 400 * sizeof(int32_t));
    cudaMalloc(&d_P, 20 * sizeof(Int192));
    
    cudaMemcpy(d_primes, h_primes, num_moduli * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inv_table, h_inv_table, 400 * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_P, h_P, num_moduli * sizeof(Int192), cudaMemcpyHostToDevice);

    int* d_eA;
    int* d_eB;
    cudaMalloc(&d_eA, m * sizeof(int));
    cudaMalloc(&d_eB, n * sizeof(int));
    
    int8_t *d_A_mod, *d_B_mod;
    size_t A_bytes = (size_t)num_moduli * m_pad * k_pad * sizeof(int8_t);
    size_t B_bytes = (size_t)num_moduli * k_pad * n_pad * sizeof(int8_t);
    cudaMalloc(&d_A_mod, A_bytes);
    cudaMalloc(&d_B_mod, B_bytes);
    
    int32_t* d_C_mod;
    size_t C_bytes = (size_t)num_moduli * m_pad * n_pad * sizeof(int32_t);
    cudaMalloc(&d_C_mod, C_bytes);

    int block_size = 256;
    compute_exp_A_kernel<<<(m + block_size - 1) / block_size, block_size>>>(d_A, d_eA, m, k);
    compute_exp_B_kernel<<<(n + block_size - 1) / block_size, block_size>>>(d_B, d_eB, k, n);

    dim3 block2d(16, 16);
    dim3 gridA((k_pad + 15) / 16, (m_pad + 15) / 16);
    quantize_A_kernel<<<gridA, block2d>>>(d_A, d_A_mod, d_eA, m, k, m_pad, k_pad, num_moduli, d_primes, S);
    
    dim3 gridB((n_pad + 15) / 16, (k_pad + 15) / 16);
    quantize_B_kernel<<<gridB, block2d>>>(d_B, d_B_mod, d_eB, k, n, k_pad, n_pad, num_moduli, d_primes, S);

    int alpha = 1;
    int beta = 0;
    long long stride_B = (long long)k_pad * n_pad; 
    long long stride_A = (long long)m_pad * k_pad;
    long long stride_C = (long long)m_pad * n_pad;

    for (int p_idx = 0; p_idx < num_moduli; ++p_idx) {
        cublasStatus_t st = cublasGemmEx(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n_pad, m_pad, k_pad,
            &alpha,
            d_B_mod + (size_t)p_idx * stride_B, CUDA_R_8I, n_pad,
            d_A_mod + (size_t)p_idx * stride_A, CUDA_R_8I, k_pad,
            &beta,
            d_C_mod + (size_t)p_idx * stride_C, CUDA_R_32I, n_pad,
            CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        if (st != CUBLAS_STATUS_SUCCESS) {
            printf("cublasGemmEx failed with status %d at p_idx %d\n", (int)st, p_idx);
            return;
        }
    }
        
    dim3 gridC((n + 15) / 16, (m + 15) / 16);
    crt_reconstruct_kernel<<<gridC, block2d>>>(d_C_mod, d_C, d_eA, d_eB, m, n, m_pad, n_pad, num_moduli, S, d_primes, d_inv_table, M, M_half, d_P);

    cudaFree(d_primes);
    cudaFree(d_inv_table);
    cudaFree(d_P);
    cudaFree(d_eA);
    cudaFree(d_eB);
    cudaFree(d_A_mod);
    cudaFree(d_B_mod);
    cudaFree(d_C_mod);
}

} // namespace ozaki
