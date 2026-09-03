#include "Float128.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <stdexcept>

using ozaki::Float128;
using ozaki::IeeeBinary128Bits;

static void expect_bits_eq(IeeeBinary128Bits actual, uint64_t hi, uint64_t lo) {
    assert(actual.hi == hi);
    assert(actual.lo == lo);
}

static void expect_roundtrip(uint64_t hi, uint64_t lo) {
    Float128 x = Float128::from_ieee_binary128(hi, lo);
    IeeeBinary128Bits y = x.to_ieee_binary128_bits();
    expect_bits_eq(y, hi, lo);
}

static bool decode_throws(uint64_t hi, uint64_t lo) {
    try {
        (void)Float128::from_ieee_binary128(hi, lo);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
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

    // Positive zero round-trips. Current Float128 value semantics do not preserve -0.
    expect_roundtrip(0x0000000000000000ull, 0x0000000000000000ull);

    // GEMM Float128 is finite-only; NaN/Inf are rejected at the adapter boundary.
    assert(decode_throws(0x7fff000000000000ull, 0x0000000000000000ull)); // +inf
    assert(decode_throws(0xffff000000000000ull, 0x0000000000000000ull)); // -inf
    assert(decode_throws(0x7fff800000000000ull, 0x0000000000000000ull)); // qNaN

    std::cout << "fp128_ieee_adapter_test passed\n";
    return 0;
}
