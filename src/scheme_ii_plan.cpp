#include "scheme_ii.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

namespace ozaki {

struct BigInt {
    std::vector<uint32_t> limbs;
    BigInt(uint64_t v = 0) {
        if (v == 0) return;
        limbs.push_back(v & 0xFFFFFFFF);
        if (v >> 32) limbs.push_back(v >> 32);
    }
    void add(const BigInt& o) {
        uint64_t carry = 0;
        size_t len = std::max(limbs.size(), o.limbs.size());
        limbs.resize(len, 0);
        for(size_t i=0; i<len; ++i) {
            uint64_t sum = limbs[i] + carry + (i < o.limbs.size() ? o.limbs[i] : 0);
            limbs[i] = sum & 0xFFFFFFFF;
            carry = sum >> 32;
        }
        if (carry) limbs.push_back(carry);
    }
    void mul(uint32_t v) {
        if (v == 0) { limbs.clear(); return; }
        uint64_t carry = 0;
        for(size_t i=0; i<limbs.size(); ++i) {
            uint64_t prod = (uint64_t)limbs[i] * v + carry;
            limbs[i] = prod & 0xFFFFFFFF;
            carry = prod >> 32;
        }
        if (carry) limbs.push_back(carry);
    }
    double to_double() const {
        double r = 0;
        double base = 1.0;
        for(size_t i=0; i<limbs.size(); ++i) {
            r += limbs[i] * base;
            base *= 4294967296.0;
        }
        return r;
    }
};

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

static double reconstruct_crt(const std::vector<int>& r, const std::vector<int>& m) {
    int k = m.size();
    std::vector<std::vector<int>> v(k, std::vector<int>(k, 0));
    for (int i = 0; i < k; ++i) v[0][i] = r[i] >= 0 ? r[i] : r[i] + m[i];
    
    for (int i = 0; i < k - 1; ++i) {
        for (int j = i + 1; j < k; ++j) {
            int inv = modInverse(m[i], m[j]);
            long long diff = v[i][j] - v[i][i];
            diff = diff % m[j];
            if (diff < 0) diff += m[j];
            v[i+1][j] = (diff * inv) % m[j];
        }
    }
    
    BigInt X(0);
    for (int i = k - 1; i >= 0; --i) {
        if (i < k - 1) X.mul(m[i]);
        BigInt term(v[i][i]);
        X.add(term);
    }
    
    BigInt M(1);
    for(int i = 0; i < k; ++i) M.mul(m[i]);
    
    double x_dbl = X.to_double();
    double M_dbl = M.to_double();
    if (x_dbl > M_dbl / 2.0) {
        x_dbl -= M_dbl;
    }
    return x_dbl;
}

void execute_scheme_ii(const double* d_A, const double* d_B, double* d_C, 
                       int m, int n, int k, int num_moduli) {
    std::cout << "[Scheme II] CPU Reference executing with num_moduli = " << num_moduli << std::endl;
    
    std::vector<double> h_A(m * k), h_B(k * n);
    cudaMemcpy(h_A.data(), d_A, m * k * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B.data(), d_B, k * n * sizeof(double), cudaMemcpyDeviceToHost);
    
    std::vector<int> primes = {256, 255, 253, 251, 247, 241, 239, 233, 229, 227, 223, 217, 211, 199, 197};
    if (num_moduli > primes.size()) num_moduli = primes.size();
    
    double M_dbl = 1.0;
    for (int i = 0; i < num_moduli; ++i) M_dbl *= primes[i];
    
    double maxA = 0.0, maxB = 0.0;
    for (int i = 0; i < m * k; ++i) maxA = std::max(maxA, std::abs(h_A[i]));
    for (int i = 0; i < k * n; ++i) maxB = std::max(maxB, std::abs(h_B[i]));
    if (maxA == 0.0) maxA = 1.0;
    if (maxB == 0.0) maxB = 1.0;
    
    double max_val = maxA * maxB * k;
    double scale = std::sqrt((M_dbl / 2.0) / max_val) * 0.25; 
    scale = std::pow(2.0, std::floor(std::log2(scale)));
    
    std::vector<long long> A_int(m * k);
    std::vector<long long> B_int(k * n);
    for (int i = 0; i < m * k; ++i) A_int[i] = std::round(h_A[i] * scale);
    for (int i = 0; i < k * n; ++i) B_int[i] = std::round(h_B[i] * scale);
    
    std::vector<double> h_C(m * n, 0.0);
    
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            std::vector<int> rems(num_moduli);
            for (int p_idx = 0; p_idx < num_moduli; ++p_idx) {
                int p = primes[p_idx];
                long long sum = 0;
                for (int kk = 0; kk < k; ++kk) {
                    long long a_val = A_int[row * k + kk] % p;
                    if (a_val < 0) a_val += p;
                    long long b_val = B_int[kk * n + col] % p;
                    if (b_val < 0) b_val += p;
                    sum = (sum + a_val * b_val) % p;
                }
                if (sum < 0) sum += p;
                rems[p_idx] = sum;
            }
            
            std::vector<int> used_primes(primes.begin(), primes.begin() + num_moduli);
            double exact_sum = reconstruct_crt(rems, used_primes);
            h_C[row * n + col] = exact_sum / (scale * scale);
        }
    }
    
    cudaMemcpy(d_C, h_C.data(), m * n * sizeof(double), cudaMemcpyHostToDevice);
}

} // namespace ozaki
