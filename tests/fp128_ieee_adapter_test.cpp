#include "Float128.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <stdexcept>

using ozaki::Float128;
using ozaki::IeeeBinary128Bits;
using ozaki::UInt384;

static void expect_bits_eq(IeeeBinary128Bits actual, uint64_t hi, uint64_t lo) {
    assert(actual.hi == hi);
    assert(actual.lo == lo);
}

static void expect_roundtrip(uint64_t hi, uint64_t lo) {
    Float128 x = Float128::from_ieee_binary128(hi, lo);
    IeeeBinary128Bits y = x.to_ieee_binary128_bits();
    expect_bits_eq(y, hi, lo);
}

int main() {
    // Normal finite values.
    expect_roundtrip(0x3fff000000000000ull, 0x0000000000000000ull); // +1
    expect_roundtrip(0xbfff000000000000ull, 0x0000000000000000ull); // -1
    expect_roundtrip(0x4000000000000000ull, 0x0000000000000000ull); // +2
    expect_roundtrip(0x3ffe000000000000ull, 0x0000000000000000ull); // +0.5
    expect_roundtrip(0x7ffeffffffffffffull, 0xffffffffffffffffull); // max finite
    expect_roundtrip(0x0001000000000000ull, 0x0000000000000000ull); // min normal

    // Fraction bits that FP64 cannot carry, but binary128 can.
    expect_roundtrip(0x3fff000000000000ull, 0x0000000000000001ull); // 1 + 2^-112
    expect_roundtrip(0x3fff000000000000ull, 0x0000000000000004ull); // 1 + 2^-110

    Float128 one_plus_2m110 = Float128::from_double_plus_pow2(1.0, -110);
    expect_bits_eq(one_plus_2m110.to_ieee_binary128_bits(),
                   0x3fff000000000000ull,
                   0x0000000000000004ull);

    // Subnormal finite values.
    expect_roundtrip(0x0000ffffffffffffull, 0xffffffffffffffffull); // max subnormal
    expect_roundtrip(0x0000000000000000ull, 0x0000000000000001ull); // min subnormal

    // Signed zero round-trips at the adapter boundary.
    expect_roundtrip(0x0000000000000000ull, 0x0000000000000000ull);
    expect_roundtrip(0x8000000000000000ull, 0x0000000000000000ull);

    // Special IEEE values round-trip as bit patterns. CRT GEMM still rejects
    // these later because NaN/Inf have no residue representation.
    expect_roundtrip(0x7fff000000000000ull, 0x0000000000000000ull); // +inf
    expect_roundtrip(0xffff000000000000ull, 0x0000000000000000ull); // -inf
    expect_roundtrip(0x7fff800000000000ull, 0x0000000000000000ull); // qNaN
    expect_roundtrip(0x7fff000000000001ull, 0x123456789abcdef0ull); // sNaN payload
    expect_roundtrip(0xffff800000000001ull, 0x0000000000000001ull); // negative qNaN payload

    bool nonfinite_rejected_by_crt = false;
    try {
        UInt384 mag;
        bool neg = false;
        Float128::infinity().to_scaled(&mag, &neg, 0);
    } catch (const std::domain_error&) {
        nonfinite_rejected_by_crt = true;
    }
    assert(nonfinite_rejected_by_crt);

    std::cout << "fp128_ieee_adapter_test passed\n";
    return 0;
}
