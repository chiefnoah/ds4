/* Microbenchmark for MoE matvec kernel.
 * Uses realistic DSv4 expert dimensions: in=4096, mid=1408, out=4096.
 * Builds a synthetic model map, runs N iters, prints t/iter.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include "../ds4_metal.h"

#define IN_DIM 4096u
#define MID_DIM 2048u
#define OUT_DIM 4096u
#define N_USED 6u
#define N_TOTAL 256u

typedef struct { uint16_t d; uint16_t qs[32]; }  __attribute__((packed)) iq2_xxs_block;  /* DS4_QK_K/8 = 32 */
typedef struct {
    uint8_t  scales[16];   /* DS4_QK_K/16 */
    uint8_t  qs[64];       /* DS4_QK_K/4 */
    uint16_t d, dmin;
} __attribute__((packed)) q2_K_block;

static double now() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static uint32_t lcg(uint32_t *s) { *s = *s * 1664525u + 1013904223u; return *s; }
static uint16_t f32_to_f16_bits(float f) {
    /* Quick approximate via __builtin or manual. */
    union { uint32_t u; float f; } v = {.f = f};
    uint32_t s = (v.u >> 31) & 0x1; uint32_t e = (v.u >> 23) & 0xff; uint32_t m = v.u & 0x7fffff;
    int e16 = (int)e - 127 + 15;
    if (e16 <= 0) return (uint16_t)(s << 15);
    if (e16 >= 31) return (uint16_t)((s << 15) | (0x1f << 10));
    return (uint16_t)((s << 15) | (e16 << 10) | (m >> 13));
}

static void fill_iq2(iq2_xxs_block *b, uint32_t seed) {
    b->d = f32_to_f16_bits(0.005f);
    for (int i = 0; i < 32; i++) {
        uint32_t v = lcg(&seed);
        b->qs[i] = (uint16_t)((v >> 16) & 0xff);  /* aux32 split into qs[0..3]: g+s. Rough. */
    }
}
static void fill_q2(q2_K_block *b, uint32_t seed) {
    b->d = f32_to_f16_bits(0.01f);
    b->dmin = f32_to_f16_bits(0.001f);
    for (int i = 0; i < 16; i++) b->scales[i] = (uint8_t)(lcg(&seed) & 0xff);
    for (int i = 0; i < 64; i++) b->qs[i] = (uint8_t)(lcg(&seed) & 0xff);
}

int main(int argc, char **argv) {
    int iters = (argc >= 2) ? atoi(argv[1]) : 50;
    uint32_t n_tokens = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 1u;

    if (!ds4_metal_init()) { fprintf(stderr, "ds4_metal_init failed\n"); return 1; }

    const uint32_t nb_in  = IN_DIM / 256u;
    const uint32_t nb_mid = MID_DIM / 256u;

    const uint64_t gate_row_bytes = (uint64_t)nb_in * sizeof(iq2_xxs_block);
    const uint64_t gate_expert_bytes = (uint64_t)MID_DIM * gate_row_bytes;
    const uint64_t down_row_bytes = (uint64_t)nb_mid * sizeof(q2_K_block);
    const uint64_t down_expert_bytes = (uint64_t)OUT_DIM * down_row_bytes;
    const uint64_t gate_pool = (uint64_t)N_TOTAL * gate_expert_bytes;
    const uint64_t down_pool = (uint64_t)N_TOTAL * down_expert_bytes;
    const uint64_t model_size = gate_pool + gate_pool + down_pool;
    fprintf(stderr, "model size: %.2f MiB (gate %.2f MiB, up %.2f MiB, down %.2f MiB)\n",
        (double)model_size / (1<<20), (double)gate_pool / (1<<20),
        (double)gate_pool / (1<<20), (double)down_pool / (1<<20));
    fprintf(stderr, "per-token weight read: gate %.2f MiB + up %.2f MiB + down %.2f MiB = %.2f MiB\n",
        (double)(N_USED * gate_expert_bytes) / (1<<20),
        (double)(N_USED * gate_expert_bytes) / (1<<20),
        (double)(N_USED * down_expert_bytes) / (1<<20),
        (double)(N_USED * (2*gate_expert_bytes + down_expert_bytes)) / (1<<20));

    uint8_t *model = (uint8_t *)calloc(1, (size_t)model_size);
    if (!model) { fprintf(stderr, "model alloc failed\n"); return 1; }
    iq2_xxs_block *gate_p = (iq2_xxs_block *)(model + 0);
    iq2_xxs_block *up_p   = (iq2_xxs_block *)(model + gate_pool);
    q2_K_block    *down_p = (q2_K_block    *)(model + gate_pool + gate_pool);

    int32_t selected[8];
    float   weights[8];
    for (uint32_t s = 0; s < N_USED; s++) {
        selected[s] = (int32_t)((s * 37) % N_TOTAL);
        weights[s] = 0.15f;
    }
    /* Init only the active experts to keep setup fast. */
    fprintf(stderr, "filling %u active experts...\n", N_USED);
    uint32_t seed = 1u;
    for (uint32_t s = 0; s < N_USED; s++) {
        const int32_t e = selected[s];
        for (uint32_t r = 0; r < MID_DIM; r++) {
            for (uint32_t b = 0; b < nb_in; b++) {
                fill_iq2(&gate_p[(uint64_t)e * MID_DIM * nb_in + r * nb_in + b], lcg(&seed));
                fill_iq2(&up_p[(uint64_t)e * MID_DIM * nb_in + r * nb_in + b], lcg(&seed));
            }
        }
        for (uint32_t r = 0; r < OUT_DIM; r++) {
            for (uint32_t b = 0; b < nb_mid; b++) {
                fill_q2(&down_p[(uint64_t)e * OUT_DIM * nb_mid + r * nb_mid + b], lcg(&seed));
            }
        }
    }

    /* Activation. */
    float *xv = (float *)calloc(n_tokens * IN_DIM, sizeof(float));
    int32_t *sels = (int32_t *)calloc(n_tokens * N_USED, sizeof(int32_t));
    float   *wts  = (float   *)calloc(n_tokens * N_USED, sizeof(float));
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t i = 0; i < IN_DIM; i++) xv[t * IN_DIM + i] = 0.01f * (float)((i + t) % 19 - 9);
        for (uint32_t s = 0; s < N_USED; s++) {
            sels[t * N_USED + s] = selected[s];
            wts[t * N_USED + s] = weights[s];
        }
    }

    ds4_metal_tensor *tx = ds4_metal_tensor_alloc((uint64_t)n_tokens * IN_DIM * sizeof(float));
    ds4_metal_tensor *tsel = ds4_metal_tensor_alloc((uint64_t)n_tokens * N_USED * sizeof(int32_t));
    ds4_metal_tensor *twts = ds4_metal_tensor_alloc((uint64_t)n_tokens * N_USED * sizeof(float));
    ds4_metal_tensor *tmid = ds4_metal_tensor_alloc((uint64_t)n_tokens * N_USED * MID_DIM * sizeof(float));
    ds4_metal_tensor *tout = ds4_metal_tensor_alloc((uint64_t)n_tokens * OUT_DIM * sizeof(float));
    if (!tx || !tsel || !twts || !tmid || !tout) { fprintf(stderr, "tensor alloc fail\n"); return 1; }
    ds4_metal_tensor_write(tx, 0, xv, n_tokens * IN_DIM * sizeof(float));
    ds4_metal_tensor_write(tsel, 0, sels, n_tokens * N_USED * sizeof(int32_t));
    ds4_metal_tensor_write(twts, 0, wts, n_tokens * N_USED * sizeof(float));

    if (!ds4_metal_set_model_map(model, model_size)) { fprintf(stderr,"set_model_map failed\n"); return 1; }
    /* Warmup. */
    int rc;
    if (n_tokens == 1) {
        rc = ds4_metal_routed_moe_one_tensor(tout, NULL, NULL, tmid, NULL,
            model, model_size, 0, gate_pool, gate_pool + gate_pool,
            16u, 10u, gate_expert_bytes, gate_row_bytes,
            down_expert_bytes, down_row_bytes, IN_DIM, MID_DIM, OUT_DIM,
            tsel, twts, N_USED, 5.0f, tx);
    } else {
        rc = ds4_metal_routed_moe_batch_tensor(tout, NULL, NULL, tmid, NULL,
            model, model_size, 0, gate_pool, gate_pool + gate_pool,
            16u, 10u, gate_expert_bytes, gate_row_bytes,
            down_expert_bytes, down_row_bytes, IN_DIM, MID_DIM, OUT_DIM,
            tsel, twts, N_USED, 5.0f, tx, n_tokens);
    }
    if (!rc) { fprintf(stderr, "moe call failed\n"); return 1; }

    double t0 = now();
    for (int i = 0; i < iters; i++) {
        if (n_tokens == 1) {
            ds4_metal_routed_moe_one_tensor(tout, NULL, NULL, tmid, NULL,
                model, model_size, 0, gate_pool, gate_pool + gate_pool,
                16u, 10u, gate_expert_bytes, gate_row_bytes,
                down_expert_bytes, down_row_bytes, IN_DIM, MID_DIM, OUT_DIM,
                tsel, twts, N_USED, 5.0f, tx);
        } else {
            ds4_metal_routed_moe_batch_tensor(tout, NULL, NULL, tmid, NULL,
                model, model_size, 0, gate_pool, gate_pool + gate_pool,
                16u, 10u, gate_expert_bytes, gate_row_bytes,
                down_expert_bytes, down_row_bytes, IN_DIM, MID_DIM, OUT_DIM,
                tsel, twts, N_USED, 5.0f, tx, n_tokens);
        }
    }
    ds4_metal_synchronize();
    double t1 = now();

    const double total_ms = (t1 - t0) * 1000.0;
    const double per_call_ms = total_ms / iters;
    const double per_token_ms = per_call_ms / n_tokens;
    const uint64_t bytes_per_token = N_USED * (2 * gate_expert_bytes + down_expert_bytes);
    const double gibps = ((double)bytes_per_token / (1<<30)) / (per_token_ms / 1000.0);
    fprintf(stderr, "MoE n_tokens=%u: %.2f ms/call, %.2f ms/token, %.2f GiB/s effective weight bw\n",
        n_tokens, per_call_ms, per_token_ms, gibps);
    return 0;
}
