#include <iostream>
#include <iomanip>
#include <vector>
#include "int8_gemm.h"

int main(int argc, char** argv) {
    struct Case {
        int m, n, k;
    };
    std::vector<Case> cases = {
        {1024, 1024, 1024},
        {2048, 2048, 2048},
        {4096, 4096, 4096},
        {768, 640, 1000}
    };
    
    std::cout << "M,N,K,in_tree_int8_ms,cublas_int8_ms,in_tree_tops,cublas_tops,ratio_vs_cublas,max_abs_int32\n";
    for (auto c : cases) {
        int pad_m = (c.m + 127) / 128 * 128;
        int pad_n = (c.n + 255) / 256 * 256;
        int pad_k = (c.k + 63) / 64 * 64;
        
        // Use padded sizes for the int8 kernel because its contract is m%BM==0, etc.
        auto res = ozaki::bench_int8_gemm_vs_cublas(pad_m, pad_n, pad_k);
        
        std::cout << c.m << "," << c.n << "," << c.k << ","
                  << std::fixed << std::setprecision(3) << res.our_ms << ","
                  << res.cublas_ms << ","
                  << res.our_tops << ","
                  << res.cublas_tops << ","
                  << res.vs_cublas << ","
                  << res.max_abs_int32 << "\n";
    }
    return 0;
}
