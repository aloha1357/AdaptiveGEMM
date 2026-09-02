#pragma once

namespace ozaki {

// Semantics: value = hi + lo, where |lo| < ulp(hi)
// Precision: ~106-bit mantissa (two non-overlapping FP64s)
struct DoubleDouble {
    double hi = 0.0;
    double lo = 0.0;

    // Construct from two doubles (TwoSum normalization)
    static __host__ __device__ DoubleDouble from(double a, double b) {
        double s = a + b;
        double a_prime = s - b;
        double b_prime = s - a_prime;
        double delta_a = a - a_prime;
        double delta_b = b - b_prime;
        double e = delta_a + delta_b;
        return {s, e};
    }
    
    __host__ __device__ DoubleDouble() : hi(0.0), lo(0.0) {}
    __host__ __device__ DoubleDouble(double h, double l) : hi(h), lo(l) {}
    
    // Add: Dekker / Knuth TwoSum for addition (basic version)
    // Actually, TwoSum takes two doubles. For DoubleDouble addition we need something more complex.
    // The prompt says: "DoubleDouble operator+(const DoubleDouble& o) const;"
    // I will implement a standard DoubleDouble addition.
    __host__ __device__ DoubleDouble operator+(const DoubleDouble& o) const {
        double t1 = hi + o.hi;
        double e = t1 - hi;
        double t2 = ((hi - (t1 - e)) + (o.hi - e)) + lo + o.lo;
        double s = t1 + t2;
        return from(s, t2 - (s - t1)); // re-normalize
    }
    
    // Convert to FP64 (discard lo)
    __host__ __device__ double toDouble() const { return hi + lo; }
};

} // namespace ozaki
