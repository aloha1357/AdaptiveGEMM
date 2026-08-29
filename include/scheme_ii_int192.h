#pragma once
#include <cstdint>
#include <cuda_runtime.h>

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
} // namespace ozaki
