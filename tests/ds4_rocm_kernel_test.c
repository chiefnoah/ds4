#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../ds4_metal.h"

static uint16_t f32_to_f16(float f) {
    union {
        float f;
        uint32_t u;
    } v = { f };
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

static int nearf(float a, float b, float tol) {
    return fabsf(a - b) <= tol;
}

static float e4m3fn_value(int i) {
    int exp = (i >> 3) & 0x0f;
    int mant = i & 0x07;
    return exp == 0
        ? (float)mant * 0.001953125f
        : (1.0f + (float)mant * 0.125f) * ldexpf(1.0f, exp - 7);
}

static float e4m3fn_dequant(float x) {
    float sign = x < 0.0f ? -1.0f : 1.0f;
    float ax = fminf(fabsf(x), 448.0f);
    int lo = 0;
    int hi = 126;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (e4m3fn_value(mid) <= ax) lo = mid;
        else hi = mid - 1;
    }
    int best = lo;
    if (best < 126) {
        float best_diff = fabsf(ax - e4m3fn_value(best));
        float next_diff = fabsf(ax - e4m3fn_value(best + 1));
        if (next_diff < best_diff ||
            (next_diff == best_diff && (((best + 1) & 1) == 0) && ((best & 1) != 0))) {
            best++;
        }
    }
    return sign * e4m3fn_value(best);
}

static void require(int cond, const char *msg) {
    if (cond) return;
    fprintf(stderr, "ds4_rocm_kernel_test: failed: %s\n", msg);
    exit(1);
}

#define STEP(msg) do { fprintf(stderr, "ds4_rocm_kernel_test: %s\n", msg); fflush(stderr); } while (0)

static void write_tensor(ds4_metal_tensor *t, const float *v, uint32_t n) {
    require(ds4_metal_tensor_write(t, 0, v, (uint64_t)n * sizeof(float)) != 0, "tensor write");
}

static void read_tensor(ds4_metal_tensor *t, float *v, uint32_t n) {
    require(ds4_metal_tensor_read(t, 0, v, (uint64_t)n * sizeof(float)) != 0, "tensor read");
}

int main(void) {
    STEP("init");
    if (!ds4_metal_init()) {
        fprintf(stderr, "ds4_rocm_kernel_test: SKIP no ROCm device visible\n");
        return 77;
    }

    STEP("alloc");
    float a[8] = { 1, 2, 3, 4, -1, -2, -3, -4 };
    float b[8] = { 8, 7, 6, 5, 4, 3, 2, 1 };
    float out[16] = {0};
    ds4_metal_tensor *ta = ds4_metal_tensor_alloc(sizeof(a));
    ds4_metal_tensor *tb = ds4_metal_tensor_alloc(sizeof(b));
    ds4_metal_tensor *to = ds4_metal_tensor_alloc(sizeof(out));
    require(ta && tb && to, "alloc");
    STEP("write inputs");
    write_tensor(ta, a, 8);
    write_tensor(tb, b, 8);

    STEP("add");
    require(ds4_metal_add_tensor(to, ta, tb, 8), "add");
    read_tensor(to, out, 8);
    for (uint32_t i = 0; i < 8; i++) require(nearf(out[i], a[i] + b[i], 1e-6f), "add value");

    STEP("swiglu");
    require(ds4_metal_swiglu_tensor(to, ta, tb, 8, 0.0f, 1.0f), "swiglu");
    read_tensor(to, out, 8);
    for (uint32_t i = 0; i < 8; i++) {
        float ref = (a[i] / (1.0f + expf(-a[i]))) * b[i];
        require(nearf(out[i], ref, 1e-5f), "swiglu value");
    }

    STEP("repeat");
    require(ds4_metal_repeat_hc_tensor(to, ta, 8, 2), "repeat");
    read_tensor(to, out, 16);
    for (uint32_t i = 0; i < 16; i++) require(nearf(out[i], a[i % 8], 1e-6f), "repeat value");

    STEP("rms");
    require(ds4_metal_rms_norm_plain_rows_tensor(to, ta, 4, 2, 1e-6f), "rms rows");
    read_tensor(to, out, 8);
    for (uint32_t r = 0; r < 2; r++) {
        float sum = 0.0f;
        for (uint32_t i = 0; i < 4; i++) sum += a[r * 4 + i] * a[r * 4 + i];
        float scale = 1.0f / sqrtf(sum / 4.0f + 1e-6f);
        for (uint32_t i = 0; i < 4; i++) {
            require(nearf(out[r * 4 + i], a[r * 4 + i] * scale, 1e-5f), "rms value");
        }
    }

    STEP("embed token");
    uint16_t weights[3 * 4];
    for (uint32_t i = 0; i < 12; i++) weights[i] = f32_to_f16((float)i + 0.5f);
    require(ds4_metal_embed_token_hc_tensor(to, weights, sizeof(weights), 0, 3, 2, 4, 2), "embed token");
    read_tensor(to, out, 8);
    for (uint32_t i = 0; i < 8; i++) require(nearf(out[i], 8.5f + (float)(i % 4), 1e-3f), "embed token value");

    STEP("embed tokens");
    int32_t toks[2] = { 1, 0 };
    ds4_metal_tensor *tt = ds4_metal_tensor_alloc(sizeof(toks));
    require(tt != NULL, "tokens alloc");
    require(ds4_metal_tensor_write(tt, 0, toks, sizeof(toks)), "tokens write");
    require(ds4_metal_embed_tokens_hc_tensor(to, tt, weights, sizeof(weights), 0, 3, 2, 4, 2), "embed tokens");
    read_tensor(to, out, 16);
    for (uint32_t i = 0; i < 8; i++) require(nearf(out[i], 4.5f + (float)(i % 4), 1e-3f), "embed tokens first");
    for (uint32_t i = 8; i < 16; i++) require(nearf(out[i], 0.5f + (float)(i % 4), 1e-3f), "embed tokens second");

    STEP("matmul f16");
    float x2[8] = { 1, 2, 3, 4, -1, -2, -3, -4 };
    uint16_t wf16[3 * 4];
    for (uint32_t i = 0; i < 12; i++) wf16[i] = f32_to_f16((float)(i + 1));
    write_tensor(ta, x2, 8);
    require(ds4_metal_matmul_f16_tensor(to, wf16, sizeof(wf16), 0, 4, 3, ta, 2), "matmul f16");
    read_tensor(to, out, 6);
    for (uint32_t t = 0; t < 2; t++) {
        for (uint32_t r = 0; r < 3; r++) {
            float ref = 0.0f;
            for (uint32_t i = 0; i < 4; i++) ref += (float)(r * 4 + i + 1) * x2[t * 4 + i];
            require(nearf(out[t * 3 + r], ref, 1e-3f), "matmul f16 value");
        }
    }

    STEP("matmul f32");
    float wf32[2 * 4] = {
        1, 0, 2, 0,
        -1, 1, 0, 3,
    };
    require(ds4_metal_matmul_f32_tensor(to, wf32, sizeof(wf32), 0, 4, 2, ta, 2), "matmul f32");
    read_tensor(to, out, 4);
    for (uint32_t t = 0; t < 2; t++) {
        for (uint32_t r = 0; r < 2; r++) {
            float ref = 0.0f;
            for (uint32_t i = 0; i < 4; i++) ref += wf32[r * 4 + i] * x2[t * 4 + i];
            require(nearf(out[t * 2 + r], ref, 1e-5f), "matmul f32 value");
        }
    }

    STEP("matmul q8_0");
    uint8_t wq8[2 * 34];
    memset(wq8, 0, sizeof(wq8));
    for (uint32_t r = 0; r < 2; r++) {
        uint16_t one = f32_to_f16(1.0f);
        wq8[r * 34 + 0] = (uint8_t)(one & 0xffu);
        wq8[r * 34 + 1] = (uint8_t)(one >> 8);
        for (uint32_t i = 0; i < 4; i++) wq8[r * 34 + 2 + i] = (uint8_t)(int8_t)(r == 0 ? (int)(i + 1) : -(int)(i + 1));
    }
    require(ds4_metal_matmul_q8_0_tensor(to, wq8, sizeof(wq8), 0, 4, 2, ta, 2), "matmul q8");
    read_tensor(to, out, 4);
    for (uint32_t t = 0; t < 2; t++) {
        float ref = 0.0f;
        for (uint32_t i = 0; i < 4; i++) ref += (float)(i + 1) * x2[t * 4 + i];
        require(nearf(out[t * 2 + 0], ref, 1e-5f), "matmul q8 pos");
        require(nearf(out[t * 2 + 1], -ref, 1e-5f), "matmul q8 neg");
    }

    STEP("weighted rms");
    float rw[4] = { 1.0f, 2.0f, -1.0f, 0.5f };
    require(ds4_metal_rms_norm_weight_rows_tensor(to, ta, rw, sizeof(rw), 0, 4, 2, 1e-6f), "weighted rms");
    read_tensor(to, out, 8);
    for (uint32_t r = 0; r < 2; r++) {
        float sum = 0.0f;
        for (uint32_t i = 0; i < 4; i++) sum += x2[r * 4 + i] * x2[r * 4 + i];
        float scale = 1.0f / sqrtf(sum / 4.0f + 1e-6f);
        for (uint32_t i = 0; i < 4; i++) {
            require(nearf(out[r * 4 + i], x2[r * 4 + i] * scale * rw[i], 1e-5f), "weighted rms value");
        }
    }

    STEP("hc split sum norm");
    float mix[24] = {0};
    float residual[4 * 8];
    float hc_params[3 + 24 + 8];
    for (uint32_t i = 0; i < 3; i++) hc_params[i] = 1.0f;
    for (uint32_t i = 0; i < 24; i++) hc_params[3 + i] = 0.0f;
    for (uint32_t i = 0; i < 8; i++) hc_params[3 + 24 + i] = 1.0f + 0.25f * (float)i;
    for (uint32_t h = 0; h < 4; h++) {
        for (uint32_t i = 0; i < 8; i++) residual[h * 8 + i] = (float)(h + 1) + 0.1f * (float)i;
    }
    ds4_metal_tensor *tmix = ds4_metal_tensor_alloc(sizeof(mix));
    ds4_metal_tensor *tres = ds4_metal_tensor_alloc(sizeof(residual));
    ds4_metal_tensor *tsplit = ds4_metal_tensor_alloc(sizeof(mix));
    ds4_metal_tensor *thc_out = ds4_metal_tensor_alloc(8 * sizeof(float));
    ds4_metal_tensor *tnorm = ds4_metal_tensor_alloc(8 * sizeof(float));
    require(tmix && tres && tsplit && thc_out && tnorm, "hc alloc");
    require(ds4_metal_tensor_write(tmix, 0, mix, sizeof(mix)), "hc mix write");
    require(ds4_metal_tensor_write(tres, 0, residual, sizeof(residual)), "hc residual write");
    require(ds4_metal_hc_split_weighted_sum_norm_tensor(thc_out, tnorm, tsplit, tmix, tres,
                                                        hc_params, sizeof(hc_params),
                                                        0,
                                                        3 * sizeof(float),
                                                        (3 + 24) * sizeof(float),
                                                        8, 4, 2, 1e-6f, 1e-6f),
            "hc split sum norm");
    read_tensor(thc_out, out, 8);
    float hc_ref[8];
    float hc_sum = 0.0f;
    for (uint32_t i = 0; i < 8; i++) {
        hc_ref[i] = 0.0f;
        for (uint32_t h = 0; h < 4; h++) hc_ref[i] += residual[h * 8 + i] * (0.5f + 1e-6f);
        hc_sum += hc_ref[i] * hc_ref[i];
        require(nearf(out[i], hc_ref[i], 1e-5f), "hc out value");
    }
    read_tensor(tnorm, out, 8);
    float hc_norm_scale = 1.0f / sqrtf(hc_sum / 8.0f + 1e-6f);
    for (uint32_t i = 0; i < 8; i++) {
        float ref = hc_ref[i] * hc_norm_scale * hc_params[3 + 24 + i];
        require(nearf(out[i], ref, 1e-5f), "hc norm value");
    }

    STEP("rope tail");
    float rope[4] = { 7.0f, 8.0f, 1.0f, 0.0f };
    write_tensor(ta, rope, 4);
    require(ds4_metal_rope_tail_tensor(ta, 1, 1, 4, 2, 1, 4096, false,
                                       10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f),
            "rope tail");
    read_tensor(ta, out, 4);
    require(nearf(out[0], 7.0f, 1e-6f), "rope prefix 0");
    require(nearf(out[1], 8.0f, 1e-6f), "rope prefix 1");
    require(nearf(out[2], cosf(1.0f), 1e-5f), "rope cos");
    require(nearf(out[3], sinf(1.0f), 1e-5f), "rope sin");

    STEP("fp8 kv quantize");
    float kv[8] = { 0.1f, -1.25f, 2.0f, -3.5f, 5.0f, -6.0f, 7.0f, -8.0f };
    write_tensor(ta, kv, 8);
    require(ds4_metal_dsv4_fp8_kv_quantize_tensor(ta, 2, 4, 2), "fp8 kv quantize");
    read_tensor(ta, out, 8);
    for (uint32_t r = 0; r < 2; r++) {
        float amax = fmaxf(fabsf(kv[r * 4]), fabsf(kv[r * 4 + 1]));
        if (amax < 1.0e-4f) amax = 1.0e-4f;
        float scale = exp2f(ceilf(log2f(amax / 448.0f)));
        for (uint32_t i = 0; i < 2; i++) {
            float ref = e4m3fn_dequant(fminf(fmaxf(kv[r * 4 + i] / scale, -448.0f), 448.0f)) * scale;
            require(nearf(out[r * 4 + i], ref, 1e-6f), "fp8 quantized value");
        }
        for (uint32_t i = 2; i < 4; i++) require(nearf(out[r * 4 + i], kv[r * 4 + i], 1e-6f), "fp8 rot value");
    }

    STEP("cleanup");
    ds4_metal_tensor_free(tnorm);
    ds4_metal_tensor_free(thc_out);
    ds4_metal_tensor_free(tsplit);
    ds4_metal_tensor_free(tres);
    ds4_metal_tensor_free(tmix);
    ds4_metal_tensor_free(tt);
    ds4_metal_tensor_free(to);
    ds4_metal_tensor_free(tb);
    ds4_metal_tensor_free(ta);
    ds4_metal_cleanup();
    fprintf(stderr, "ds4_rocm_kernel_test: OK\n");
    return 0;
}
