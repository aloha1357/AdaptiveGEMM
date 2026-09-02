#pragma once
#include <cstdint>

namespace ozaki {

static const int32_t MODULI_INT8_EXTENDED[49] = {
    // Current 20 moduli
    256, 255, 253, 251, 247, 241, 239, 233,
    229, 227, 223, 217, 211, 199, 197, 193,
    191, 181, 179, 173,
    // Extended 29 moduli (pairwise coprime to all above)
    167, 163, 157, 151, 149, 139, 137, 131,
    127, 113, 109, 107, 103, 101,  97,  89,
     83,  79,  73,  71,  67,  61,  59,  53,
     47,  43,  41,  37,  29
};
static const int MAX_MODULI_INT8 = 49;

} // namespace ozaki
