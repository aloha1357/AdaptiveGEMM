#pragma once

#include "UInt384.h"
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace ozaki {

struct IeeeBinary128Bits {
    uint64_t hi = 0;
    uint64_t lo = 0;
};

// 113-bit significand M in [2^112, 2^113) when normalized.
// |value| = M * 2^expo.
struct Float128 {
    uint64_t mant_lo = 0; // M bits 0-63
    uint64_t mant_hi = 0; // M bits 64-112
    int32_t expo = 0;
    int32_t sign = 0; // 0 +, 1 -

    static constexpr int kIeeeExponentBias = 16383;
    static constexpr int kIeeeMinNormalExponent = -16382;
    static constexpr int kIeeeFractionBits = 112;
    static constexpr uint64_t kIeeeFractionHiMask = (1ull << 48) - 1ull;
    static constexpr uint16_t kIeeeInfNanExponent = 0x7fff;

    static Float128 zero() { return {}; }

    bool is_zero() const { return mant_lo == 0 && (mant_hi == 0); }

    static Float128 from_ieee_binary128(uint64_t hi_bits, uint64_t lo_bits) {
        Float128 r;
        r.sign = (hi_bits >> 63) ? 1 : 0;

        uint16_t exp_field = (uint16_t)((hi_bits >> 48) & 0x7fffull);
        uint64_t frac_hi = hi_bits & kIeeeFractionHiMask;
        uint64_t frac_lo = lo_bits;

        if (exp_field == kIeeeInfNanExponent) {
            throw std::invalid_argument("Float128 finite adapter does not support IEEE binary128 NaN/Inf");
        }

        if (exp_field == 0) {
            UInt384 frac(frac_lo, frac_hi);
            int B = frac.bit_length();
            if (B == 0) return zero();

            int shift = kIeeeFractionBits + 1 - B;
            UInt384 m = frac << shift;
            r.mant_lo = m.words[0];
            r.mant_hi = m.words[1];
            r.expo = (1 - kIeeeExponentBias - kIeeeFractionBits) - shift;
            r.normalize();
            return r;
        }

        r.mant_lo = frac_lo;
        r.mant_hi = (1ull << 48) | frac_hi;
        r.expo = (int)exp_field - kIeeeExponentBias - kIeeeFractionBits;
        r.normalize();
        return r;
    }

    static Float128 from_ieee_binary128_bits(const IeeeBinary128Bits& bits) {
        return from_ieee_binary128(bits.hi, bits.lo);
    }

    IeeeBinary128Bits to_ieee_binary128_bits() const {
        Float128 x = *this;
        x.normalize();

        IeeeBinary128Bits out;
        uint64_t sign_bit = x.sign ? (1ull << 63) : 0ull;
        if (x.is_zero()) {
            out.hi = sign_bit;
            out.lo = 0;
            return out;
        }

        int E = x.expo + kIeeeFractionBits;
        if (E >= kIeeeMinNormalExponent) {
            int exp_field = E + kIeeeExponentBias;
            if (exp_field >= kIeeeInfNanExponent) {
                throw std::overflow_error("Float128 value is too large for finite IEEE binary128");
            }
            out.hi = sign_bit | ((uint64_t)exp_field << 48) | (x.mant_hi & kIeeeFractionHiMask);
            out.lo = x.mant_lo;
            return out;
        }

        UInt384 m(x.mant_lo, x.mant_hi);
        int shift = (1 - kIeeeExponentBias - kIeeeFractionBits) - x.expo;
        UInt384 frac = shift_right_round_to_even(m, shift);

        if (frac.bit_length() > kIeeeFractionBits) {
            out.hi = sign_bit | (1ull << 48);
            out.lo = 0;
            return out;
        }

        out.hi = sign_bit | (frac.words[1] & kIeeeFractionHiMask);
        out.lo = frac.words[0];
        return out;
    }

    static Float128 from_double(double x) {
        Float128 r;
        if (x == 0.0 || !std::isfinite(x)) return r;
        r.sign = x < 0.0 ? 1 : 0;
        int e = 0;
        double f = std::frexp(std::fabs(x), &e); // [0.5, 1)
        // 53-bit mantissa of f, then pad 60 zero bits to 113.
        uint64_t m53 = (uint64_t)std::llround(f * 9007199254740992.0); // 2^53
        // m53 in [2^52, 2^53)
        r.mant_lo = m53 << 60;
        r.mant_hi = m53 >> 4; // (m53 << 60) overflow into hi: 53-4=49 bits in hi? 
        // m53 << 60: low 4 bits of m53 go to top of lo? 
        // 64-60=4, so lo = m53 << 60, hi = m53 >> 4.
        r.expo = e - 113;
        r.normalize();
        return r;
    }

    static Float128 from_pow2(int e) {
        Float128 r;
        r.sign = 0;
        r.mant_lo = 0;
        r.mant_hi = (1ull << 48); // bit 112 = bit 48 of hi
        r.expo = e - 112;
        return r;
    }

    static Float128 from_double_plus_pow2(double hi, int lo_exp) {
        return from_double(hi) + from_pow2(lo_exp);
    }

    int log2_floor() const {
        if (is_zero()) return -1024;
        return expo + 112;
    }

    double to_double() const {
        if (is_zero()) return 0.0;
        // M / 2^112 * 2^(expo+112) = (M/2^112) * 2^log2_floor
        double hi = (double)mant_hi * 18446744073709551616.0 + (double)mant_lo;
        double v = std::ldexp(hi, expo);
        return sign ? -v : v;
    }

    void to_scaled(UInt384* mag, bool* neg, int scale) const {
        *neg = sign != 0;
        if (is_zero()) {
            *mag = UInt384();
            return;
        }
        UInt384 m(mant_lo, mant_hi);
        int shift = expo + scale;
        if (shift >= 0) {
            *mag = m << shift;
        } else {
            int rsh = -shift;
            int round_bit = m.get_bit(rsh - 1);
            *mag = m >> rsh;
            if (round_bit) *mag = *mag + 1u;
        }
    }

    static Float128 from_scaled(const UInt384& mag, bool neg, int scale) {
        Float128 r;
        int B = mag.bit_length();
        if (B == 0) return r;
        r.sign = neg ? 1 : 0;
        int need = B - 113;
        UInt384 m113;
        if (need >= 0) {
            int round_bit = mag.get_bit(need - 1);
            m113 = mag >> need;
            if (round_bit) m113 = m113 + 1u;
            r.expo = -scale + need;
        } else {
            m113 = mag << (-need);
            r.expo = -scale + need;
        }
        r.mant_lo = m113.words[0];
        r.mant_hi = m113.words[1];
        r.normalize();
        return r;
    }

    Float128 operator+(const Float128& o) const {
        if (is_zero()) return o;
        if (o.is_zero()) return *this;
        const Float128* a = this;
        const Float128* b = &o;
        if (b->expo > a->expo || (b->expo == a->expo && mag_ge(*b, *a))) {
            a = &o;
            b = this;
        }
        int d = a->expo - b->expo;
        uint32_t A[5] = {0, 0, 0, 0, 0};
        uint32_t B[5] = {0, 0, 0, 0, 0};
        split_limbs(*a, A);
        split_limbs(*b, B);
        if (d >= 160) return *a;
        shr_limbs(B, d);
        Float128 r;
        r.expo = a->expo;
        if (a->sign == b->sign) {
            add_limbs(A, B);
            r.sign = a->sign;
        } else {
            if (!sub_limbs(A, B)) {
                // should not happen if |a|>=|b|
            }
            r.sign = a->sign;
        }
        unsplit_limbs(r, A);
        r.normalize();
        return r;
    }

    Float128 operator*(const Float128& o) const {
        if (is_zero() || o.is_zero()) return zero();
        uint32_t A[4], B[4], P[8];
        split_limbs4(*this, A);
        split_limbs4(o, B);
        for (int i = 0; i < 8; ++i) P[i] = 0;
        for (int i = 0; i < 4; ++i) {
            uint64_t carry = 0;
            for (int j = 0; j < 4; ++j) {
                uint64_t t = (uint64_t)A[i] * B[j] + P[i + j] + carry;
                P[i + j] = (uint32_t)t;
                carry = t >> 32;
            }
            P[i + 4] += (uint32_t)carry;
        }
        Float128 r;
        r.sign = sign ^ o.sign;
        r.expo = expo + o.expo;
        // P is 256-bit little-endian 32-bit limbs. Pack to UInt384 then from_scaled with scale=0
        // actually |prod| = Ma*Mb * 2^(ea+eb), Ma*Mb is the integer P.
        UInt384 mag;
        for (int i = 0; i < 8; ++i) {
            int w = i / 2;
            if (i % 2 == 0) mag.words[w] = P[i];
            else mag.words[w] |= ((uint64_t)P[i] << 32);
        }
        // value = mag * 2^(expo) with current r.expo = ea+eb
        // reuse from_scaled: mag * 2^{-scale} with scale = -r.expo
        return from_scaled(mag, r.sign != 0, -r.expo);
    }

private:
    static bool mag_ge(const Float128& a, const Float128& b) {
        if (a.expo != b.expo) return a.expo > b.expo;
        if (a.mant_hi != b.mant_hi) return a.mant_hi > b.mant_hi;
        return a.mant_lo >= b.mant_lo;
    }

    static bool any_bits_below(const UInt384& v, int bit_count) {
        if (bit_count <= 0) return false;
        if (bit_count >= 384) {
            for (int i = 0; i < 6; ++i) {
                if (v.words[i] != 0) return true;
            }
            return false;
        }

        int full_words = bit_count / 64;
        int rem_bits = bit_count % 64;
        for (int i = 0; i < full_words; ++i) {
            if (v.words[i] != 0) return true;
        }
        if (rem_bits == 0) return false;
        uint64_t mask = (1ull << rem_bits) - 1ull;
        return (v.words[full_words] & mask) != 0;
    }

    static UInt384 shift_right_round_to_even(const UInt384& v, int shift) {
        if (shift <= 0) return v << (-shift);

        UInt384 truncated = v >> shift;
        bool guard = (shift - 1 < 384) && v.get_bit(shift - 1);
        bool sticky = any_bits_below(v, shift - 1);
        bool lsb_odd = truncated.get_bit(0) != 0;
        if (guard && (sticky || lsb_odd)) {
            truncated = truncated + 1u;
        }
        return truncated;
    }

    void normalize() {
        if (is_zero()) {
            expo = 0;
            sign = 0;
            return;
        }
        UInt384 m(mant_lo, mant_hi);
        int B = m.bit_length();
        if (B == 0) {
            *this = zero();
            return;
        }
        int need = B - 113;
        if (need > 0) {
            int round_bit = m.get_bit(need - 1);
            m = m >> need;
            if (round_bit) m = m + 1u;
            expo += need;
            if (m.bit_length() > 113) {
                m = m >> 1;
                expo += 1;
            }
        } else if (need < 0) {
            m = m << (-need);
            expo += need;
        }
        mant_lo = m.words[0];
        mant_hi = m.words[1];
    }

    static void split_limbs(const Float128& x, uint32_t L[5]) {
        L[0] = (uint32_t)x.mant_lo;
        L[1] = (uint32_t)(x.mant_lo >> 32);
        L[2] = (uint32_t)x.mant_hi;
        L[3] = (uint32_t)(x.mant_hi >> 32);
        L[4] = 0;
    }
    static void split_limbs4(const Float128& x, uint32_t L[4]) {
        L[0] = (uint32_t)x.mant_lo;
        L[1] = (uint32_t)(x.mant_lo >> 32);
        L[2] = (uint32_t)x.mant_hi;
        L[3] = (uint32_t)(x.mant_hi >> 32);
    }
    static void unsplit_limbs(Float128& x, const uint32_t L[5]) {
        x.mant_lo = (uint64_t)L[0] | ((uint64_t)L[1] << 32);
        x.mant_hi = (uint64_t)L[2] | ((uint64_t)L[3] << 32);
        if (L[4]) {
            // overflow into bit 160; fold by shifting later in normalize
            x.mant_hi |= ((uint64_t)L[4] << 32); // may overflow 49 bits; normalize handles bit_length
        }
    }
    static void shr_limbs(uint32_t L[5], int sh) {
        if (sh <= 0) return;
        if (sh >= 160) {
            for (int i = 0; i < 5; ++i) L[i] = 0;
            return;
        }
        int w = sh / 32;
        int b = sh % 32;
        if (w) {
            for (int i = 0; i < 5; ++i) {
                L[i] = (i + w < 5) ? L[i + w] : 0;
            }
        }
        if (b) {
            uint32_t prev = 0;
            for (int i = 4; i >= 0; --i) {
                uint32_t cur = L[i];
                L[i] = (cur >> b) | (prev << (32 - b));
                prev = cur;
            }
        }
    }
    static void add_limbs(uint32_t A[5], const uint32_t B[5]) {
        uint64_t c = 0;
        for (int i = 0; i < 5; ++i) {
            uint64_t t = (uint64_t)A[i] + B[i] + c;
            A[i] = (uint32_t)t;
            c = t >> 32;
        }
    }
    static bool sub_limbs(uint32_t A[5], const uint32_t B[5]) {
        int64_t br = 0;
        for (int i = 0; i < 5; ++i) {
            int64_t t = (int64_t)A[i] - (int64_t)B[i] - br;
            if (t < 0) {
                t += (1ll << 32);
                br = 1;
            } else br = 0;
            A[i] = (uint32_t)t;
        }
        return br == 0;
    }
};

} // namespace ozaki
