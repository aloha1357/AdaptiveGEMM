#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>
#include <iomanip>
#include <sstream>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace ozaki {

struct UInt384 {
    uint64_t words[6];

    UInt384() {
        for (int i=0; i<6; ++i) words[i] = 0;
    }

    UInt384(uint64_t val) {
        words[0] = val;
        for (int i=1; i<6; ++i) words[i] = 0;
    }

    UInt384(uint64_t lo, uint64_t hi) {
        words[0] = lo;
        words[1] = hi;
        for (int i = 2; i < 6; ++i) words[i] = 0;
    }

    int bit_length() const {
        for (int i = 5; i >= 0; --i) {
            if (words[i] == 0) continue;
            uint64_t w = words[i];
            int b = 63;
            while (b > 0 && (w >> b) == 0) --b;
            return 64 * i + b + 1;
        }
        return 0;
    }

    int get_bit(int idx) const {
        if (idx < 0 || idx >= 384) return 0;
        return (int)((words[idx / 64] >> (idx % 64)) & 1ull);
    }

    UInt384 operator<<(int shift) const {
        if (shift == 0) return *this;
        if (shift >= 384) return UInt384();
        if (shift < 0) return (*this) >> (-shift);
        UInt384 res;
        int word_shift = shift / 64;
        int bit_shift = shift % 64;
        for (int i = 5; i >= word_shift; --i) {
            res.words[i] = words[i - word_shift] << bit_shift;
            if (bit_shift > 0 && i - word_shift - 1 >= 0) {
                res.words[i] |= (words[i - word_shift - 1] >> (64 - bit_shift));
            }
        }
        return res;
    }

    // Add another UInt384
    UInt384 operator+(const UInt384& other) const {
        UInt384 res;
        uint64_t carry = 0;
        for (int i = 0; i < 6; ++i) {
            uint64_t sum = words[i] + other.words[i] + carry;
            carry = (sum < words[i] || (carry == 1 && sum == words[i])) ? 1 : 0;
            res.words[i] = sum;
        }
        return res;
    }

    // Multiply by uint32_t
    UInt384 operator*(uint32_t val) const {
        UInt384 res;
        uint64_t carry = 0;
        for (int i = 0; i < 6; ++i) {
            uint64_t w_low = words[i] & 0xFFFFFFFF;
            uint64_t w_high = words[i] >> 32;
            uint64_t p_low = w_low * val;
            uint64_t p_high = w_high * val + (p_low >> 32);
            uint64_t low = (p_low & 0xFFFFFFFF) | (p_high << 32);
            uint64_t high = p_high >> 32;
            
            uint64_t sum = low + carry;
            uint64_t sum_carry = (sum < low) ? 1 : 0;
            res.words[i] = sum;
            carry = high + sum_carry;
        }
        return res;
    }
    
    // Add uint32_t
    UInt384 operator+(uint32_t val) const {
        UInt384 res;
        uint64_t carry = val;
        for (int i = 0; i < 6; ++i) {
            uint64_t sum = words[i] + carry;
            carry = (sum < words[i]) ? 1 : 0;
            res.words[i] = sum;
        }
        return res;
    }

    // Modulo uint32_t
    uint32_t operator%(uint32_t mod) const {
        uint64_t rem = 0;
        for (int i = 5; i >= 0; --i) {
            // Need (rem * 2^64 + words[i]) % mod
            // We can do it by shifting rem 32 bits at a time
            uint64_t high_part = (rem << 32) | (words[i] >> 32);
            rem = high_part % mod;
            uint64_t low_part = (rem << 32) | (words[i] & 0xFFFFFFFF);
            rem = low_part % mod;
        }
        return (uint32_t)rem;
    }

    // Shift right
    UInt384 operator>>(int shift) const {
        if (shift == 0) return *this;
        if (shift >= 384) return UInt384();
        UInt384 res;
        int word_shift = shift / 64;
        int bit_shift = shift % 64;
        for (int i = 0; i < 6 - word_shift; ++i) {
            res.words[i] = words[i + word_shift] >> bit_shift;
            if (bit_shift > 0 && i + word_shift + 1 < 6) {
                res.words[i] |= (words[i + word_shift + 1] << (64 - bit_shift));
            }
        }
        return res;
    }
    
    // Subtraction
    UInt384 operator-(const UInt384& other) const {
        UInt384 res;
        uint64_t borrow = 0;
        for (int i = 0; i < 6; ++i) {
            uint64_t diff = words[i] - other.words[i] - borrow;
            borrow = (words[i] < other.words[i] + borrow || (borrow == 1 && other.words[i] == ~(uint64_t)0)) ? 1 : 0;
            res.words[i] = diff;
        }
        return res;
    }
    
    bool operator<(const UInt384& other) const {
        for (int i = 5; i >= 0; --i) {
            if (words[i] < other.words[i]) return true;
            if (words[i] > other.words[i]) return false;
        }
        return false;
    }
};

} // namespace ozaki
