/* Microbench driver for ds4_metal_matmul_q8_0_tensor (and the wave-per-row
 * Q8_0 kernel it dispatches to on decode).  Skips the 80 GB GGUF load so the
 * kernel-iteration loop is just edit -> rebuild -> run (no 21-second model
 * upload).  Synthesizes a Q8_0 weight matrix on the host, registers it once
 * via ds4_metal_set_model_map, then loops the wrapper.
 *
 * Usage:
 *   ./ds4_rocm_matmul_bench [in_dim] [out_dim] [iters] [n_tok]
 * Defaults: in_dim=4096 out_dim=4096 iters=2000 n_tok=1.
 *
 * Reports: ms/call, ms/token, GiB/s effective. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../ds4_metal.h"

static double now_sec(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1.0e-9 * (double)t.tv_nsec;
}

static uint16_t f32_to_f16_bits(float f) {
    union { float f; uint32_t u; } v = { f };
    uint32_t sign = (v.u >> 16) & 0x8000u;
    int32_t exp = (int32_t)((v.u >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = v.u & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half = mant >> shift;
        if ((mant >> (shift - 1)) & 1u) half++;
        return (uint16_t)(sign | half);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    if (mant & 0x1000u) half++;
    return (uint16_t)half;
}

static void build_q8_0_synth(uint8_t *dst, uint64_t in_dim, uint64_t out_dim) {
    const uint64_t blocks_per_row = (in_dim + 31u) / 32u;
    const uint64_t row_bytes = blocks_per_row * 34u;
    /* Same scale per block keeps things deterministic; per-block qs[] is a
     * cheap pseudo-random walk so the compiler can't fold the dot product
     * to a constant. */
    const float scale = 1.0f / 127.0f;
    const uint16_t hf = f32_to_f16_bits(scale);
    for (uint64_t r = 0; r < out_dim; r++) {
        uint64_t state = (r + 1) * 0x9E3779B97F4A7C15ull;
        for (uint64_t b = 0; b < blocks_per_row; b++) {
            uint8_t *blk = dst + r * row_bytes + b * 34u;
            blk[0] = (uint8_t)(hf & 0xffu);
            blk[1] = (uint8_t)(hf >> 8);
            for (uint32_t i = 0; i < 32; i++) {
                state = state * 6364136223846793005ull + 1442695040888963407ull;
                blk[2 + i] = (uint8_t)((state >> 24) & 0xffu);
            }
        }
    }
}

int main(int argc, char **argv) {
    uint64_t in_dim  = (argc > 1) ? (uint64_t)strtoull(argv[1], NULL, 10) : 4096ull;
    uint64_t out_dim = (argc > 2) ? (uint64_t)strtoull(argv[2], NULL, 10) : 4096ull;
    int iters        = (argc > 3) ? atoi(argv[3]) : 2000;
    uint64_t n_tok   = (argc > 4) ? (uint64_t)strtoull(argv[4], NULL, 10) : 1ull;
    if (iters < 1) iters = 1;
    if (n_tok < 1) n_tok = 1;

    if (!ds4_metal_init()) {
        fprintf(stderr, "matmul_bench: SKIP no ROCm device\n");
        return 77;
    }

    const uint64_t blocks_per_row = (in_dim + 31u) / 32u;
    const uint64_t row_bytes      = blocks_per_row * 34u;
    const uint64_t weight_bytes   = row_bytes * out_dim;

    /* Cache-busting working set: gfx1151 Infinity Cache (MALL) is ~32 MiB.
     * Production decode reads each weight matrix once per token from DRAM,
     * so a hot-cache microbench massively overstates throughput.  Allocate
     * enough copies of the weight so the total working set is at least 256
     * MiB (8x MALL), and rotate through them per iteration. */
    const uint64_t kFloodBytes = 256ull * 1024ull * 1024ull;
    uint64_t n_copies = kFloodBytes / weight_bytes;
    if (n_copies < 2) n_copies = 1;       /* big shapes already exceed MALL */
    const uint64_t total_bytes = weight_bytes * n_copies;

    void *w_aligned = NULL;
    if (posix_memalign(&w_aligned, 64, (size_t)total_bytes) != 0 || !w_aligned) {
        fprintf(stderr, "alloc weight host failed\n"); return 1;
    }
    uint8_t *w_host = (uint8_t *)w_aligned;
    build_q8_0_synth(w_host, in_dim, out_dim);
    /* Replicate the synthetic block across the remaining copies; we only
     * care about cache pressure, not exact values. */
    for (uint64_t i = 1; i < n_copies; i++) {
        memcpy(w_host + i * weight_bytes, w_host, (size_t)weight_bytes);
    }

    if (!ds4_metal_set_model_map(w_host, total_bytes)) {
        fprintf(stderr, "set_model_map failed\n"); return 1;
    }

    ds4_metal_tensor *tx   = ds4_metal_tensor_alloc(n_tok * in_dim  * sizeof(float));
    ds4_metal_tensor *tout = ds4_metal_tensor_alloc(n_tok * out_dim * sizeof(float));
    if (!tx || !tout) { fprintf(stderr, "tensor alloc failed\n"); return 1; }

    float *xh = (float *)malloc(n_tok * in_dim * sizeof(float));
    for (uint64_t i = 0; i < n_tok * in_dim; i++) xh[i] = (float)((i * 1664525u + 1013904223u) & 255u) / 256.0f - 0.5f;
    if (!ds4_metal_tensor_write(tx, 0, xh, n_tok * in_dim * sizeof(float))) {
        fprintf(stderr, "tensor write failed\n"); return 1;
    }
    free(xh);

    /* Warm up: hipMalloc-on-first-use, kernel module load, etc. */
    for (int i = 0; i < 5; i++) {
        const uint64_t off = (uint64_t)(i % (int)n_copies) * weight_bytes;
        if (!ds4_metal_matmul_q8_0_tensor(tout, w_host, total_bytes, off,
                                           in_dim, out_dim, tx, n_tok)) {
            fprintf(stderr, "warmup matmul failed\n"); return 1;
        }
    }
    ds4_metal_synchronize();

    double t0 = now_sec();
    for (int i = 0; i < iters; i++) {
        const uint64_t off = (uint64_t)(i % (int)n_copies) * weight_bytes;
        if (!ds4_metal_matmul_q8_0_tensor(tout, w_host, total_bytes, off,
                                           in_dim, out_dim, tx, n_tok)) {
            fprintf(stderr, "matmul iter %d failed\n", i); return 1;
        }
    }
    ds4_metal_synchronize();
    double dt = now_sec() - t0;

    const double ms_per = (dt * 1000.0) / (double)iters;
    const double ms_tok = ms_per / (double)n_tok;
    const double gib_s  = ((double)weight_bytes / (1024.0 * 1024.0 * 1024.0))
                          / (dt / (double)iters);
    fprintf(stdout,
            "matmul_q8_0 in=%llu out=%llu n_tok=%llu iters=%d  weight=%.2f MiB  copies=%llu  "
            "%.4f ms/call  %.4f ms/token  %.2f GiB/s (weight only)\n",
            (unsigned long long)in_dim, (unsigned long long)out_dim,
            (unsigned long long)n_tok, iters,
            (double)weight_bytes / (1024.0 * 1024.0),
            (unsigned long long)n_copies,
            ms_per, ms_tok, gib_s);

    ds4_metal_tensor_free(tx);
    ds4_metal_tensor_free(tout);
    free(w_host);
    return 0;
}
