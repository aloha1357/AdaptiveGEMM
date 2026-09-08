// include/mma_sm89.cuh
// sm_89 PTX helpers — extracted from AdaptiveOzaki.cu lines 9-21 and the mma helper.
// Do NOT include: fp64_hi, get_A8_offset, get_B8_offset, swizzle_k16 (CRT layout).
// CRT kernel includes this header instead of redeclaring these helpers inline.
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// ---------------------------------------------------------------------------
// cp.async helpers  (AdaptiveOzaki.cu original lines 9-21)
// ---------------------------------------------------------------------------

__device__ __forceinline__ void cp_async_16(void* smem_ptr, const void* global_ptr) {
    uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_addr), "l"(global_ptr));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_0() {
    asm volatile("cp.async.wait_group 0;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_1() {
    asm volatile("cp.async.wait_group 1;\n" ::);
}

// Template version for N-stage pipelines.
// cp_async_wait<0>() drains all; cp_async_wait<1>() leaves 1 in flight.
// 3-stage pipeline uses cp_async_wait<Stages-2>() = wait_group 1.
template<int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

// ---------------------------------------------------------------------------
// mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
// d[0..3] += A[0..3] * B[0..1]  (c = d as in-place accumulate)
// (AdaptiveOzaki.cu mma_m16n8k32_s8)
// ---------------------------------------------------------------------------

__device__ __forceinline__ void mma_m16n8k32_s8(
    int32_t* d, const uint32_t* a, const uint32_t* b, const int32_t* c) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3])
    );
}

// ---------------------------------------------------------------------------
// ldmatrix helpers
// ---------------------------------------------------------------------------

// ldmatrix.sync.aligned.m8n8.x4  (A fragment, no transpose)
// Loads 4 x 8x8 matrices from smem into 4 uint32 registers.
__device__ __forceinline__ void ldmatrix_x4(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
    const void* smem_ptr) {
    uint32_t addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr)
    );
}

// ldmatrix.sync.aligned.m8n8.x2.trans  (B fragment, with transpose)
// Loads 2 x 8x8 matrices transposed from smem into 2 uint32 registers.
// Address must be 16-byte aligned (covers 16 consecutive N-bytes).
// Gives {r0, r1} = {k-lo half, k-hi half} for ONE n8-column group.
__device__ __forceinline__ void ldmatrix_x2_trans(
    uint32_t& r0, uint32_t& r1,
    const void* smem_ptr) {
    uint32_t addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
        : "=r"(r0), "=r"(r1)
        : "r"(addr)
    );
}

// ldmatrix.sync.aligned.m8n8.x4.trans  (two B fragments, with transpose)
// Loads 4 x 8x8 matrices transposed from smem into 4 uint32 registers.
// Address must be 16-byte aligned.
// r0,r1 = b_frag for j_n=(even):  n-cols [col_base, col_base+8)
// r2,r3 = b_frag for j_n=(odd):   n-cols [col_base+8, col_base+16)
__device__ __forceinline__ void ldmatrix_x4_trans(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
    const void* smem_ptr) {
    uint32_t addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr)
    );
}

// Path B: ldmatrix.x4.trans on K-major sB[K][N] (16B along N) produces 2N×2K
// packing. MMA m16n8k32.s8 .col wants 1N×4K. Locked against scatter ldmatrix.x2
// gold (scripts/path_b_probe.cu): shfl inside 4-lane groups + __byte_perm 0x6420/0x7531.
// t0..t3 = x4.trans of 32K × 16N. Emits two n8 B fragments for this lane.
__device__ __forceinline__ void repack_x4_trans_to_mma_b(
    uint32_t t0, uint32_t t1, uint32_t t2, uint32_t t3,
    uint32_t& j_r0, uint32_t& j_r1, uint32_t& j1_r0, uint32_t& j1_r1) {
    int lane = threadIdx.x & 31;
    int q = lane & 3;
    int n8 = lane >> 2;
    int src0 = (n8 & ~1) * 2 + ((q & 1) ? 2 : 0);
    int src1 = src0 + 1;
    int sel = (n8 & 1) ? 0x7531 : 0x6420;

    uint32_t a0 = __shfl_sync(0xffffffff, t0, src0);
    uint32_t b0 = __shfl_sync(0xffffffff, t0, src1);
    uint32_t a1 = __shfl_sync(0xffffffff, t1, src0);
    uint32_t b1 = __shfl_sync(0xffffffff, t1, src1);
    uint32_t a2 = __shfl_sync(0xffffffff, t2, src0);
    uint32_t b2 = __shfl_sync(0xffffffff, t2, src1);
    uint32_t a3 = __shfl_sync(0xffffffff, t3, src0);
    uint32_t b3 = __shfl_sync(0xffffffff, t3, src1);
    if (q < 2) {
        j_r0 = __byte_perm(a0, b0, sel);
        j_r1 = __byte_perm(a2, b2, sel);
    } else {
        j_r0 = __byte_perm(a1, b1, sel);
        j_r1 = __byte_perm(a3, b3, sel);
    }

    a0 = __shfl_sync(0xffffffff, t0, src0 + 16);
    b0 = __shfl_sync(0xffffffff, t0, src1 + 16);
    a1 = __shfl_sync(0xffffffff, t1, src0 + 16);
    b1 = __shfl_sync(0xffffffff, t1, src1 + 16);
    a2 = __shfl_sync(0xffffffff, t2, src0 + 16);
    b2 = __shfl_sync(0xffffffff, t2, src1 + 16);
    a3 = __shfl_sync(0xffffffff, t3, src0 + 16);
    b3 = __shfl_sync(0xffffffff, t3, src1 + 16);
    if (q < 2) {
        j1_r0 = __byte_perm(a0, b0, sel);
        j1_r1 = __byte_perm(a2, b2, sel);
    } else {
        j1_r0 = __byte_perm(a1, b1, sel);
        j1_r1 = __byte_perm(a3, b3, sel);
    }
}
