/* Smoke + microbench for ds4_metal_begin_capture / end_capture / replay.
 *
 * Builds against ds4_rocm.o.  Captures K successive add-kernel launches and
 * replays the captured graph N times.  Compares the wall-time + correctness
 * to running K*N direct adds, to measure CPU launch overhead recovered.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../ds4_metal.h"

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void fill(float *p, uint32_t n, float seed) {
    for (uint32_t i = 0; i < n; i++) p[i] = seed + (float)i * 0.001f;
}

static int approx_eq(const float *a, const float *b, uint32_t n, float eps) {
    for (uint32_t i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > eps) {
            fprintf(stderr, "mismatch at %u: %g vs %g (delta %g)\n", i, a[i], b[i], d);
            return 0;
        }
    }
    return 1;
}

int main(void) {
    if (!ds4_metal_init()) {
        fprintf(stderr, "ds4_metal_init failed\n");
        return 1;
    }
    if (!ds4_metal_capture_supported()) {
        fprintf(stderr, "capture not supported on this backend\n");
        return 1;
    }

    const uint32_t N = 4096u;          /* per-tensor element count */
    const uint32_t K = 250u;           /* kernels per captured graph (~1 ds4 token) */
    const uint32_t REPLAYS = 200u;     /* graph replays (= simulated tokens) */
    const uint64_t bytes = (uint64_t)N * sizeof(float);

    ds4_metal_tensor *a = ds4_metal_tensor_alloc(bytes);
    ds4_metal_tensor *b = ds4_metal_tensor_alloc(bytes);
    ds4_metal_tensor *c = ds4_metal_tensor_alloc(bytes);
    if (!a || !b || !c) {
        fprintf(stderr, "tensor alloc failed\n");
        return 1;
    }
    float *host_a = (float *)malloc(bytes);
    float *host_b = (float *)malloc(bytes);
    float *host_c = (float *)malloc(bytes);
    fill(host_a, N, 1.0f);
    fill(host_b, N, 2.0f);
    if (!ds4_metal_tensor_write(a, 0, host_a, bytes)) return 1;
    if (!ds4_metal_tensor_write(b, 0, host_b, bytes)) return 1;
    if (!ds4_metal_tensor_write(c, 0, host_a, bytes)) return 1;

    /* === Reference: K*REPLAYS direct adds in batched mode === */
    double t0 = now_sec();
    if (!ds4_metal_begin_commands()) return 1;
    for (uint32_t r = 0; r < REPLAYS; r++) {
        for (uint32_t k = 0; k < K; k++) {
            if (!ds4_metal_add_tensor(c, a, b, N)) return 1;
        }
    }
    if (!ds4_metal_end_commands()) return 1;
    double direct_sec = now_sec() - t0;
    if (!ds4_metal_tensor_read(c, 0, host_c, bytes)) return 1;
    /* expected: c = a + b after final write — but we accumulated, so c != a+b.
     * Add semantics: c = a + b (overwrite). So after K*REPLAYS additions,
     * c == host_a + host_b regardless of how many. */
    for (uint32_t i = 0; i < N; i++) host_a[i] += 0.0f; /* keep as-is */
    /* Build expected: each call is c = a+b, so end state c[i] = a[i]+b[i]. */
    float *expected = (float *)malloc(bytes);
    for (uint32_t i = 0; i < N; i++) expected[i] = host_a[i] + host_b[i];
    if (!approx_eq(host_c, expected, N, 1e-4f)) {
        fprintf(stderr, "DIRECT mode produced wrong result\n");
        return 1;
    }

    /* === Capture: capture K adds, replay REPLAYS times === */
    if (!ds4_metal_begin_capture()) {
        fprintf(stderr, "begin_capture failed\n");
        return 1;
    }
    for (uint32_t k = 0; k < K; k++) {
        if (!ds4_metal_add_tensor(c, a, b, N)) {
            fprintf(stderr, "add inside capture failed at k=%u\n", k);
            return 1;
        }
    }
    if (!ds4_metal_end_capture()) {
        fprintf(stderr, "end_capture failed\n");
        return 1;
    }
    /* Reset c so replays make a meaningful difference (still ends as a+b). */
    if (!ds4_metal_tensor_write(c, 0, host_a, bytes)) return 1;
    double t1 = now_sec();
    for (uint32_t r = 0; r < REPLAYS; r++) {
        if (!ds4_metal_replay_graph()) {
            fprintf(stderr, "replay failed at r=%u\n", r);
            return 1;
        }
    }
    double replay_sec = now_sec() - t1;
    if (!ds4_metal_tensor_read(c, 0, host_c, bytes)) return 1;
    if (!approx_eq(host_c, expected, N, 1e-4f)) {
        fprintf(stderr, "REPLAY mode produced wrong result\n");
        return 1;
    }

    const uint64_t total_kernels = (uint64_t)K * REPLAYS;
    printf("kernels: %llu (K=%u, replays=%u)\n",
           (unsigned long long)total_kernels, K, REPLAYS);
    printf("direct: %.3f ms  (%.2f us/kernel)\n",
           direct_sec * 1e3, direct_sec * 1e6 / (double)total_kernels);
    printf("replay: %.3f ms  (%.2f us/kernel-equiv)\n",
           replay_sec * 1e3, replay_sec * 1e6 / (double)total_kernels);
    printf("speedup: %.2fx\n", direct_sec / replay_sec);
    printf("OK\n");

    free(expected);
    free(host_a);
    free(host_b);
    free(host_c);
    ds4_metal_tensor_free(c);
    ds4_metal_tensor_free(b);
    ds4_metal_tensor_free(a);
    ds4_metal_destroy_capture();
    ds4_metal_cleanup();
    return 0;
}
