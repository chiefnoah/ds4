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

/* CPU oracle for the unified batch attention kernel.  Computes
 * out[t,h,d] = softmax(scores)·V.  Mode-static layouts have raw_kv as
 * (n_tokens × head_dim); ring layouts have raw_kv as (raw_cap × head_dim)
 * with valid keys at indices (raw_start + r) mod raw_cap, key positions
 * starting at first_raw_pos = pos0 + n_tokens - n_raw. */
static void attn_batch_oracle(
        float *out, const float *qv, const float *raw_kv, const float *comp_kv,
        const float *comp_mask, const int32_t *topk, const float *sinks,
        uint32_t n_tokens, uint32_t n_head, uint32_t head_dim,
        uint32_t pos0, uint32_t n_raw, uint32_t raw_cap, uint32_t raw_start,
        uint32_t n_comp, uint32_t window, uint32_t ratio,
        uint32_t top_k, int use_topk, int use_comp_mask, int mode_static) {
    const float kq = 1.0f / sqrtf((float)head_dim);
    const uint32_t first_raw_pos = mode_static ? 0u : pos0 + n_tokens - n_raw;
    for (uint32_t t = 0; t < n_tokens; t++) {
        const uint32_t qpos = pos0 + t;
        const uint32_t n_visible_comp = ratio == 0u ? 0u : (qpos + 1u) / ratio;
        for (uint32_t h = 0; h < n_head; h++) {
            float maxs = sinks[h];
            float scores[64];  /* test sizes are small */
            for (uint32_t r = 0; r < n_raw; r++) {
                const uint32_t kpos = first_raw_pos + r;
                if (kpos > qpos || (window != 0u && qpos - kpos >= window)) {
                    scores[r] = -1.0e30f; continue;
                }
                const uint32_t actual = mode_static ? r : (raw_start + r) % raw_cap;
                float s = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++)
                    s += qv[(t * n_head + h) * head_dim + d] * raw_kv[actual * head_dim + d];
                scores[r] = s * kq;
                if (scores[r] > maxs) maxs = scores[r];
            }
            for (uint32_t c = 0; c < n_comp; c++) scores[n_raw + c] = -1.0e30f;
            if (use_topk) {
                for (uint32_t k = 0; k < top_k; k++) {
                    int32_t c = topk[t * top_k + k];
                    if (c < 0 || (uint32_t)c >= n_comp) continue;
                    if ((uint32_t)c >= n_visible_comp) continue;
                    float s = 0.0f;
                    for (uint32_t d = 0; d < head_dim; d++)
                        s += qv[(t * n_head + h) * head_dim + d] * comp_kv[(uint32_t)c * head_dim + d];
                    scores[n_raw + c] = s * kq;
                    if (scores[n_raw + c] > maxs) maxs = scores[n_raw + c];
                }
            } else {
                for (uint32_t c = 0; c < n_comp; c++) {
                    if (c >= n_visible_comp) continue;
                    float s = 0.0f;
                    for (uint32_t d = 0; d < head_dim; d++)
                        s += qv[(t * n_head + h) * head_dim + d] * comp_kv[c * head_dim + d];
                    scores[n_raw + c] = s * kq + (use_comp_mask ? comp_mask[c] : 0.0f);
                    if (scores[n_raw + c] > maxs) maxs = scores[n_raw + c];
                }
            }
            float denom = expf(sinks[h] - maxs);
            float oacc[64] = {0};
            for (uint32_t r = 0; r < n_raw; r++) {
                if (scores[r] <= -5.0e29f) continue;
                float w = expf(scores[r] - maxs);
                denom += w;
                const uint32_t actual = mode_static ? r : (raw_start + r) % raw_cap;
                for (uint32_t d = 0; d < head_dim; d++) oacc[d] += w * raw_kv[actual * head_dim + d];
            }
            for (uint32_t c = 0; c < n_comp; c++) {
                if (scores[n_raw + c] <= -5.0e29f) continue;
                float w = expf(scores[n_raw + c] - maxs);
                denom += w;
                for (uint32_t d = 0; d < head_dim; d++) oacc[d] += w * comp_kv[c * head_dim + d];
            }
            for (uint32_t d = 0; d < head_dim; d++) out[(t * n_head + h) * head_dim + d] = oacc[d] / denom;
        }
    }
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

    STEP("compressor store batch ratio=4");
    {
        const uint32_t head_dim = 8;
        const uint32_t ratio = 4;
        const uint32_t coff = 2;
        const uint32_t width = coff * head_dim;
        const uint32_t state_rows = coff * ratio;
        const uint32_t n_tokens = 6;
        const uint32_t pos0 = 0;
        float kv_in[6 * 16];
        float sc_in[6 * 16];
        float ape_in[4 * 16];
        for (uint32_t t = 0; t < n_tokens; t++) {
            for (uint32_t i = 0; i < width; i++) {
                kv_in[t * width + i] = (float)(t + 1) * 0.125f + (float)i * 0.0625f;
                sc_in[t * width + i] = (float)(t + 1) * 0.5f - (float)i * 0.0625f;
            }
        }
        for (uint32_t r = 0; r < ratio; r++) {
            for (uint32_t i = 0; i < width; i++) ape_in[r * width + i] = (float)r * 0.25f + (float)i * 0.001f;
        }
        ds4_metal_tensor *tkv = ds4_metal_tensor_alloc((uint64_t)n_tokens * width * sizeof(float));
        ds4_metal_tensor *tsc = ds4_metal_tensor_alloc((uint64_t)n_tokens * width * sizeof(float));
        ds4_metal_tensor *tskv = ds4_metal_tensor_alloc((uint64_t)state_rows * width * sizeof(float));
        ds4_metal_tensor *tssc = ds4_metal_tensor_alloc((uint64_t)state_rows * width * sizeof(float));
        require(tkv && tsc && tskv && tssc, "comp store alloc");
        require(ds4_metal_tensor_write(tkv, 0, kv_in, sizeof(kv_in)), "comp kv write");
        require(ds4_metal_tensor_write(tsc, 0, sc_in, sizeof(sc_in)), "comp sc write");
        float zeros[8 * 16] = {0};
        require(ds4_metal_tensor_write(tskv, 0, zeros, (uint64_t)state_rows * width * sizeof(float)), "comp state kv zero");
        require(ds4_metal_tensor_write(tssc, 0, zeros, (uint64_t)state_rows * width * sizeof(float)), "comp state sc zero");
        require(ds4_metal_compressor_store_batch_tensor(tkv, tsc, tskv, tssc,
                                                        ape_in, sizeof(ape_in), 0, 0,
                                                        head_dim, ratio, pos0, n_tokens),
                "comp store batch f32 r4");
        float got_kv[8 * 16], got_sc[8 * 16];
        read_tensor(tskv, got_kv, state_rows * width);
        read_tensor(tssc, got_sc, state_rows * width);
        for (uint32_t r = 0; r < state_rows; r++) {
            int pos_mod;
            int valid = 1;
            uint32_t src_t = 0;
            if (r < ratio) { valid = 0; pos_mod = 0; }
            else {
                pos_mod = (int)(r - ratio);
                int t0 = ((pos_mod - (int)(pos0 % ratio)) + (int)ratio) % (int)ratio;
                if ((uint32_t)t0 >= n_tokens) { valid = 0; }
                else {
                    uint32_t k_max = (n_tokens - 1u - (uint32_t)t0) / ratio;
                    src_t = (uint32_t)t0 + k_max * ratio;
                }
            }
            for (uint32_t i = 0; i < width; i++) {
                if (!valid) {
                    require(nearf(got_kv[r * width + i], 0.0f, 1e-6f), "comp store r4 lower untouched kv");
                    require(nearf(got_sc[r * width + i], 0.0f, 1e-6f), "comp store r4 lower untouched sc");
                } else {
                    require(nearf(got_kv[r * width + i], kv_in[src_t * width + i], 1e-6f), "comp store r4 kv");
                    require(nearf(got_sc[r * width + i], sc_in[src_t * width + i] + ape_in[(uint32_t)pos_mod * width + i], 1e-6f), "comp store r4 sc");
                }
            }
        }
        ds4_metal_tensor_free(tssc);
        ds4_metal_tensor_free(tskv);
        ds4_metal_tensor_free(tsc);
        ds4_metal_tensor_free(tkv);
    }

    STEP("compressor store batch ratio=2 with f16 APE");
    {
        const uint32_t head_dim = 4;
        const uint32_t ratio = 2;
        const uint32_t coff = 1;
        const uint32_t width = coff * head_dim;
        const uint32_t state_rows = coff * ratio;
        const uint32_t n_tokens = 3;
        const uint32_t pos0 = 1;
        float kv_in[3 * 4] = { 1, 2, 3, 4,  5, 6, 7, 8,  9, 10, 11, 12 };
        float sc_in[3 * 4] = { -1, -2, -3, -4,  -5, -6, -7, -8,  -9, -10, -11, -12 };
        uint16_t ape_in_f16[2 * 4];
        float ape_ref[2 * 4] = { 0.5f, 0.25f, -0.25f, 1.0f,  -0.5f, 0.125f, 1.5f, -1.0f };
        for (uint32_t i = 0; i < 8; i++) ape_in_f16[i] = f32_to_f16(ape_ref[i]);
        ds4_metal_tensor *tkv = ds4_metal_tensor_alloc(sizeof(kv_in));
        ds4_metal_tensor *tsc = ds4_metal_tensor_alloc(sizeof(sc_in));
        ds4_metal_tensor *tskv = ds4_metal_tensor_alloc((uint64_t)state_rows * width * sizeof(float));
        ds4_metal_tensor *tssc = ds4_metal_tensor_alloc((uint64_t)state_rows * width * sizeof(float));
        require(tkv && tsc && tskv && tssc, "comp store r2 alloc");
        require(ds4_metal_tensor_write(tkv, 0, kv_in, sizeof(kv_in)), "comp store r2 kv write");
        require(ds4_metal_tensor_write(tsc, 0, sc_in, sizeof(sc_in)), "comp store r2 sc write");
        require(ds4_metal_compressor_store_batch_tensor(tkv, tsc, tskv, tssc,
                                                        ape_in_f16, sizeof(ape_in_f16), 0, 1,
                                                        head_dim, ratio, pos0, n_tokens),
                "comp store batch f16 r2");
        float got_kv[2 * 4], got_sc[2 * 4];
        read_tensor(tskv, got_kv, state_rows * width);
        read_tensor(tssc, got_sc, state_rows * width);
        for (uint32_t r = 0; r < state_rows; r++) {
            int pos_mod = (int)r;
            int t0 = ((pos_mod - (int)(pos0 % ratio)) + (int)ratio) % (int)ratio;
            require((uint32_t)t0 < n_tokens, "comp store r2 row covered");
            uint32_t k_max = (n_tokens - 1u - (uint32_t)t0) / ratio;
            uint32_t src_t = (uint32_t)t0 + k_max * ratio;
            for (uint32_t i = 0; i < width; i++) {
                require(nearf(got_kv[r * width + i], kv_in[src_t * width + i], 1e-6f), "comp store r2 kv");
                /* APE was round-tripped through f16; compare against the f16-decoded reference. */
                float ape = ape_ref[(uint32_t)pos_mod * width + i];
                require(nearf(got_sc[r * width + i], sc_in[src_t * width + i] + ape, 1e-3f), "comp store r2 sc");
            }
        }
        ds4_metal_tensor_free(tssc);
        ds4_metal_tensor_free(tskv);
        ds4_metal_tensor_free(tsc);
        ds4_metal_tensor_free(tkv);
    }

    STEP("compressor update ratio=2 emit");
    {
        const uint32_t head_dim = 4;
        const uint32_t ratio = 2;
        const uint32_t coff = 1;
        const uint32_t width = coff * head_dim;
        const uint32_t state_rows = coff * ratio;
        const uint32_t n_rot = 4;
        const uint32_t pos = 1;
        float kv_cur[4] = { 0.7f, -1.5f, 2.25f, 0.4f };
        float sc_cur[4] = { 1.0f, -0.5f, 0.25f, 0.5f };
        float ape[2 * 4] = { 0, 0, 0, 0,  0.1f, 0.2f, 0.3f, 0.4f };
        float norm_w[4] = { 1.0f, 2.0f, -1.0f, 0.5f };
        ds4_metal_tensor *tkv = ds4_metal_tensor_alloc(sizeof(kv_cur));
        ds4_metal_tensor *tsc = ds4_metal_tensor_alloc(sizeof(sc_cur));
        ds4_metal_tensor *tskv = ds4_metal_tensor_alloc((uint64_t)state_rows * width * sizeof(float));
        ds4_metal_tensor *tssc = ds4_metal_tensor_alloc((uint64_t)state_rows * width * sizeof(float));
        ds4_metal_tensor *tcomp = ds4_metal_tensor_alloc(2u * head_dim * sizeof(float));
        require(tkv && tsc && tskv && tssc && tcomp, "comp update r2 alloc");
        require(ds4_metal_tensor_write(tkv, 0, kv_cur, sizeof(kv_cur)), "comp update r2 kv write");
        require(ds4_metal_tensor_write(tsc, 0, sc_cur, sizeof(sc_cur)), "comp update r2 sc write");
        float mask_min[2 * 4];
        for (uint32_t i = 0; i < 8; i++) mask_min[i] = -1.0e30f;
        require(ds4_metal_tensor_write(tssc, 0, mask_min, sizeof(mask_min)), "comp update r2 score init");
        float zeros[2 * 4] = {0};
        require(ds4_metal_tensor_write(tskv, 0, zeros, sizeof(zeros)), "comp update r2 kv init");
        float comp_init[2 * 4] = {0};
        require(ds4_metal_tensor_write(tcomp, 0, comp_init, sizeof(comp_init)), "comp update r2 cache init");
        uint8_t blob[sizeof(ape) + sizeof(norm_w)];
        memcpy(blob, ape, sizeof(ape));
        memcpy(blob + sizeof(ape), norm_w, sizeof(norm_w));
        require(ds4_metal_compressor_update_tensor(tkv, tsc, tskv, tssc, tcomp,
                                                   blob, sizeof(blob),
                                                   0, 0, sizeof(ape), 0,
                                                   head_dim, ratio, pos, 0, n_rot, 0,
                                                   10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1e-6f),
                "compressor update r2");
        float got[8];
        read_tensor(tcomp, got, 8);
        float ref[4] = { kv_cur[0], kv_cur[1], kv_cur[2], kv_cur[3] };
        float ss = 0.0f;
        for (uint32_t i = 0; i < 4; i++) ss += ref[i] * ref[i];
        float rms = 1.0f / sqrtf(ss / 4.0f + 1e-6f);
        for (uint32_t i = 0; i < 4; i++) ref[i] = ref[i] * rms * norm_w[i];
        /* RoPE at comp_pos=0: theta=0 → identity. */
        for (uint32_t i = 0; i < 4; i++) require(nearf(got[i], ref[i], 1e-5f), "comp update r2 row 0");
        for (uint32_t i = 4; i < 8; i++) require(nearf(got[i], 0.0f, 1e-6f), "comp update r2 row 1 untouched");
        /* state_kv row 1 must equal kv_cur. */
        float skv[8];
        read_tensor(tskv, skv, 8);
        for (uint32_t i = 0; i < 4; i++) require(nearf(skv[width + i], kv_cur[i], 1e-6f), "comp update r2 state kv");
        /* state_score row 1 must equal sc_cur + ape[1, i]. */
        float ssc[8];
        read_tensor(tssc, ssc, 8);
        for (uint32_t i = 0; i < 4; i++) {
            require(nearf(ssc[width + i], sc_cur[i] + ape[width + i], 1e-6f), "comp update r2 state score");
        }
        ds4_metal_tensor_free(tcomp);
        ds4_metal_tensor_free(tssc);
        ds4_metal_tensor_free(tskv);
        ds4_metal_tensor_free(tsc);
        ds4_metal_tensor_free(tkv);
    }

    STEP("compressor update ratio=4 emit + shift");
    {
        const uint32_t head_dim = 8;
        const uint32_t ratio = 4;
        const uint32_t coff = 2;
        const uint32_t width = coff * head_dim;
        const uint32_t state_rows = coff * ratio;
        const uint32_t n_rot = 4;
        const uint32_t pos = 3;
        float kv_cur[16];
        float sc_cur[16];
        for (uint32_t i = 0; i < 16; i++) {
            kv_cur[i] = 0.1f * (float)(i + 1);
            sc_cur[i] = 0.05f * (float)i;
        }
        float ape[4 * 16];
        for (uint32_t r = 0; r < 4; r++) {
            for (uint32_t i = 0; i < 16; i++) ape[r * 16 + i] = 0.01f * (float)(r * 16 + i);
        }
        float norm_w[8] = { 1.0f, -1.0f, 0.5f, 2.0f, 0.25f, 1.5f, -0.5f, 1.0f };
        ds4_metal_tensor *tkv = ds4_metal_tensor_alloc(sizeof(kv_cur));
        ds4_metal_tensor *tsc = ds4_metal_tensor_alloc(sizeof(sc_cur));
        ds4_metal_tensor *tskv = ds4_metal_tensor_alloc((uint64_t)state_rows * width * sizeof(float));
        ds4_metal_tensor *tssc = ds4_metal_tensor_alloc((uint64_t)state_rows * width * sizeof(float));
        ds4_metal_tensor *tcomp = ds4_metal_tensor_alloc(2u * head_dim * sizeof(float));
        require(tkv && tsc && tskv && tssc && tcomp, "comp update r4 alloc");
        require(ds4_metal_tensor_write(tkv, 0, kv_cur, sizeof(kv_cur)), "comp update r4 kv write");
        require(ds4_metal_tensor_write(tsc, 0, sc_cur, sizeof(sc_cur)), "comp update r4 sc write");
        float kv_init[8 * 16];
        float sc_init[8 * 16];
        for (uint32_t i = 0; i < 8 * 16; i++) {
            kv_init[i] = (float)(i + 100);          /* sentinel: must be overwritten by shift */
            sc_init[i] = -1.0e30f;
        }
        require(ds4_metal_tensor_write(tskv, 0, kv_init, sizeof(kv_init)), "comp update r4 state kv init");
        require(ds4_metal_tensor_write(tssc, 0, sc_init, sizeof(sc_init)), "comp update r4 state score init");
        float comp_init[16] = {0};
        require(ds4_metal_tensor_write(tcomp, 0, comp_init, sizeof(comp_init)), "comp update r4 cache init");
        uint8_t blob[sizeof(ape) + sizeof(norm_w)];
        memcpy(blob, ape, sizeof(ape));
        memcpy(blob + sizeof(ape), norm_w, sizeof(norm_w));
        require(ds4_metal_compressor_update_tensor(tkv, tsc, tskv, tssc, tcomp,
                                                   blob, sizeof(blob),
                                                   0, 0, sizeof(ape), 0,
                                                   head_dim, ratio, pos, 0, n_rot, 0,
                                                   10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1e-6f),
                "compressor update r4");
        float got[16];
        read_tensor(tcomp, got, 16);
        /* Pool: only row 7 has non-mask score (row 7, cols 8..15 since coff==2 reads upper half cols).
         * So pooled[j] = state_kv[7, head_dim + j] = kv_cur[head_dim + j]. */
        float pooled[8];
        for (uint32_t j = 0; j < 8; j++) pooled[j] = kv_cur[head_dim + j];
        float ss = 0.0f;
        for (uint32_t j = 0; j < 8; j++) ss += pooled[j] * pooled[j];
        float rms = 1.0f / sqrtf(ss / 8.0f + 1e-6f);
        float ref[8];
        for (uint32_t j = 0; j < 8; j++) ref[j] = pooled[j] * rms * norm_w[j];
        /* RoPE at comp_pos = pos+1-ratio = 0: identity. */
        for (uint32_t j = 0; j < 8; j++) require(nearf(got[j], ref[j], 1e-5f), "comp update r4 pool/norm/rope");
        for (uint32_t j = 8; j < 16; j++) require(nearf(got[j], 0.0f, 1e-6f), "comp update r4 cache row 1 untouched");
        /* After shift: lower half (rows 0..3) of state must equal upper half (rows 4..7) post-store.
         * Upper rows 4,5,6 still hold their initial values (kv_init), row 7 holds kv_cur. */
        float skv[8 * 16];
        read_tensor(tskv, skv, 8 * 16);
        for (uint32_t r = 0; r < 4; r++) {
            for (uint32_t i = 0; i < width; i++) {
                const float expected = (r < 3)
                    ? kv_init[(ratio + r) * width + i]
                    : kv_cur[i];
                require(nearf(skv[r * width + i], expected, 1e-5f), "comp update r4 shift kv");
            }
        }
        /* Upper half rows 4..6 must be unchanged from init; row 7 must be kv_cur. */
        for (uint32_t r = 0; r < 3; r++) {
            for (uint32_t i = 0; i < width; i++) {
                require(nearf(skv[(ratio + r) * width + i], kv_init[(ratio + r) * width + i], 1e-6f),
                        "comp update r4 upper preserved");
            }
        }
        for (uint32_t i = 0; i < width; i++) {
            require(nearf(skv[(ratio + 3) * width + i], kv_cur[i], 1e-6f),
                    "comp update r4 upper row 7 kv_cur");
        }
        ds4_metal_tensor_free(tcomp);
        ds4_metal_tensor_free(tssc);
        ds4_metal_tensor_free(tskv);
        ds4_metal_tensor_free(tsc);
        ds4_metal_tensor_free(tkv);
    }

    STEP("compressor prefill ratio=2 n_tokens=4");
    {
        const uint32_t head_dim = 4;
        const uint32_t ratio = 2;
        const uint32_t coff = 1;
        const uint32_t width = coff * head_dim;
        const uint32_t state_rows = coff * ratio;
        const uint32_t n_tokens = 4;
        const uint32_t pos0 = 0;
        const uint32_t n_rot = 4;
        float kv[4 * 4];
        float sc[4 * 4];
        for (uint32_t t = 0; t < n_tokens; t++) {
            for (uint32_t i = 0; i < width; i++) {
                kv[t * width + i] = 0.5f * (float)(t + 1) + 0.125f * (float)i;
                sc[t * width + i] = 0.25f * (float)t - 0.0625f * (float)i;
            }
        }
        float ape[2 * 4] = { 0, 0, 0, 0,  0.1f, -0.1f, 0.2f, -0.2f };
        float norm_w[4] = { 1.0f, 0.5f, 2.0f, -1.0f };
        ds4_metal_tensor *tkv = ds4_metal_tensor_alloc(sizeof(kv));
        ds4_metal_tensor *tsc = ds4_metal_tensor_alloc(sizeof(sc));
        ds4_metal_tensor *tskv = ds4_metal_tensor_alloc((uint64_t)state_rows * width * sizeof(float));
        ds4_metal_tensor *tssc = ds4_metal_tensor_alloc((uint64_t)state_rows * width * sizeof(float));
        ds4_metal_tensor *tcomp = ds4_metal_tensor_alloc(2u * head_dim * sizeof(float));
        require(tkv && tsc && tskv && tssc && tcomp, "comp prefill r2 alloc");
        require(ds4_metal_tensor_write(tkv, 0, kv, sizeof(kv)), "comp prefill r2 kv write");
        require(ds4_metal_tensor_write(tsc, 0, sc, sizeof(sc)), "comp prefill r2 sc write");
        uint8_t blob[sizeof(ape) + sizeof(norm_w)];
        memcpy(blob, ape, sizeof(ape));
        memcpy(blob + sizeof(ape), norm_w, sizeof(norm_w));
        require(ds4_metal_compressor_prefill_tensor(tcomp, tskv, tssc, tkv, tsc,
                                                    blob, sizeof(blob),
                                                    0, 0, sizeof(ape), 0,
                                                    head_dim, ratio, pos0, n_tokens, n_rot, 0,
                                                    false,
                                                    10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1e-6f),
                "compressor prefill r2");
        float got[2 * 4];
        read_tensor(tcomp, got, 2 * 4);
        /* Reference: at first emit (t=1), state has rows {0,1} with (kv[t], sc[t]+ape[t%2]). */
        for (uint32_t emit = 0; emit < 2; emit++) {
            const uint32_t base_t = emit * ratio;
            float pooled[4];
            for (uint32_t j = 0; j < head_dim; j++) {
                float s0 = sc[(base_t + 0) * width + j] + ape[0 * width + j];
                float s1 = sc[(base_t + 1) * width + j] + ape[1 * width + j];
                float maxs = s0 > s1 ? s0 : s1;
                float e0 = expf(s0 - maxs);
                float e1 = expf(s1 - maxs);
                pooled[j] = (e0 * kv[(base_t + 0) * width + j] +
                             e1 * kv[(base_t + 1) * width + j]) / (e0 + e1);
            }
            float ss = 0.0f;
            for (uint32_t j = 0; j < head_dim; j++) ss += pooled[j] * pooled[j];
            float rms = 1.0f / sqrtf(ss / (float)head_dim + 1e-6f);
            for (uint32_t j = 0; j < head_dim; j++) {
                float ref = pooled[j] * rms * norm_w[j];
                /* RoPE: comp_pos = (base_t + ratio) - ratio = base_t. For emit 0: comp_pos=0 → identity.
                 * For emit 1: comp_pos=2.  We compare against the post-RoPE result. */
                if (emit == 0) {
                    require(nearf(got[emit * head_dim + j], ref, 1e-5f), "comp prefill r2 emit0");
                } else {
                    /* For emit 1, n_rot=head_dim covers all dims. Apply YaRN RoPE manually. */
                    /* Defer: test only emit 0 strictly; for emit 1 just check magnitude is reasonable. */
                    (void)ref;
                }
            }
        }
        /* Spot-check the second emit by recomputing expected with full RoPE. */
        {
            const uint32_t base_t = ratio;
            const uint32_t comp_pos = base_t;  /* ratio + base_t - ratio */
            float pooled[4];
            for (uint32_t j = 0; j < head_dim; j++) {
                float s0 = sc[(base_t + 0) * width + j] + ape[0 * width + j];
                float s1 = sc[(base_t + 1) * width + j] + ape[1 * width + j];
                float maxs = s0 > s1 ? s0 : s1;
                float e0 = expf(s0 - maxs);
                float e1 = expf(s1 - maxs);
                pooled[j] = (e0 * kv[(base_t + 0) * width + j] +
                             e1 * kv[(base_t + 1) * width + j]) / (e0 + e1);
            }
            float ss = 0.0f;
            for (uint32_t j = 0; j < head_dim; j++) ss += pooled[j] * pooled[j];
            float rms = 1.0f / sqrtf(ss / (float)head_dim + 1e-6f);
            float refn[4];
            for (uint32_t j = 0; j < head_dim; j++) refn[j] = pooled[j] * rms * norm_w[j];
            /* RoPE tail with adjacent-pair convention (j0=2*ic, j1=2*ic+1) and YaRN. */
            float ref_rope[4];
            memcpy(ref_rope, refn, sizeof(ref_rope));
            for (uint32_t ic = 0; ic < n_rot / 2; ic++) {
                uint32_t r = ic * 2u;
                float theta = (float)comp_pos * powf(10000.0f, (-1.0f / (float)n_rot) * (float)r);
                float c = cosf(theta);
                float s = sinf(theta);
                uint32_t j0 = r;        /* n_nope = 0 */
                uint32_t j1 = j0 + 1u;
                float x0 = refn[j0];
                float x1 = refn[j1];
                ref_rope[j0] = x0 * c - x1 * s;
                ref_rope[j1] = x0 * s + x1 * c;
            }
            for (uint32_t j = 0; j < head_dim; j++) {
                require(nearf(got[head_dim + j], ref_rope[j], 5e-5f), "comp prefill r2 emit1");
            }
        }
        ds4_metal_tensor_free(tcomp);
        ds4_metal_tensor_free(tssc);
        ds4_metal_tensor_free(tskv);
        ds4_metal_tensor_free(tsc);
        ds4_metal_tensor_free(tkv);
    }

    STEP("compressor prefill_state_ratio4");
    {
        const uint32_t head_dim = 4;
        const uint32_t ratio = 4;
        const uint32_t width = 2 * head_dim;
        const uint32_t state_rows = 8;
        const uint32_t pos0 = 1;
        float kv_tail[4 * 8];
        float sc_tail[4 * 8];
        for (uint32_t r = 0; r < ratio; r++) {
            for (uint32_t i = 0; i < width; i++) {
                kv_tail[r * width + i] = 0.5f * (float)(r + 1) + 0.0625f * (float)i;
                sc_tail[r * width + i] = 0.25f * (float)r - 0.125f * (float)i;
            }
        }
        float ape[4 * 8];
        for (uint32_t r = 0; r < ratio; r++) {
            for (uint32_t i = 0; i < width; i++) ape[r * width + i] = 0.01f * (float)(r * 16 + i);
        }
        ds4_metal_tensor *tk = ds4_metal_tensor_alloc(sizeof(kv_tail));
        ds4_metal_tensor *ts = ds4_metal_tensor_alloc(sizeof(sc_tail));
        ds4_metal_tensor *tskv = ds4_metal_tensor_alloc((uint64_t)state_rows * width * sizeof(float));
        ds4_metal_tensor *tssc = ds4_metal_tensor_alloc((uint64_t)state_rows * width * sizeof(float));
        require(tk && ts && tskv && tssc, "comp r4 state alloc");
        require(ds4_metal_tensor_write(tk, 0, kv_tail, sizeof(kv_tail)), "comp r4 state kv write");
        require(ds4_metal_tensor_write(ts, 0, sc_tail, sizeof(sc_tail)), "comp r4 state sc write");
        /* Pre-fill state with sentinel garbage to confirm the kernel resets it. */
        float garbage[8 * 8];
        for (uint32_t i = 0; i < 8 * 8; i++) garbage[i] = (float)(i + 999);
        require(ds4_metal_tensor_write(tskv, 0, garbage, sizeof(garbage)), "comp r4 state init kv");
        require(ds4_metal_tensor_write(tssc, 0, garbage, sizeof(garbage)), "comp r4 state init sc");
        require(ds4_metal_compressor_prefill_state_ratio4_tensor(tskv, tssc, tk, ts,
                                                                  ape, sizeof(ape), 0, 0,
                                                                  head_dim, pos0),
                "comp r4 state");
        float skv[8 * 8], ssc[8 * 8];
        read_tensor(tskv, skv, 8 * 8);
        read_tensor(tssc, ssc, 8 * 8);
        for (uint32_t r = 0; r < ratio; r++) {
            const uint32_t pm = (pos0 + r) % ratio;
            for (uint32_t i = 0; i < width; i++) {
                require(nearf(skv[r * width + i], kv_tail[r * width + i], 1e-6f), "comp r4 state lower kv");
                require(nearf(ssc[r * width + i], sc_tail[r * width + i] + ape[pm * width + i], 1e-6f),
                        "comp r4 state lower sc");
            }
        }
        for (uint32_t r = ratio; r < state_rows; r++) {
            for (uint32_t i = 0; i < width; i++) {
                require(nearf(skv[r * width + i], 0.0f, 1e-6f), "comp r4 state upper kv reset");
                require(ssc[r * width + i] < -1.0e29f, "comp r4 state upper sc -inf");
            }
        }
        ds4_metal_tensor_free(tssc);
        ds4_metal_tensor_free(tskv);
        ds4_metal_tensor_free(ts);
        ds4_metal_tensor_free(tk);
    }

    STEP("attention decode_heads (raw only, ring wrap)");
    {
        const uint32_t n_head = 2;
        const uint32_t head_dim = 4;
        const uint32_t raw_cap = 4;
        const uint32_t n_raw = 3;
        const uint32_t raw_start = 2;          /* wraps: rows 2,3,0 */
        float qv[2 * 4] = { 0.5f, -0.25f, 1.0f, 0.0f,  -0.1f, 0.3f, 0.0f, 0.5f };
        float raw[4 * 4] = {
            0.1f, 0.2f, -0.3f, 0.4f,           /* row 0 */
            -0.5f, 0.6f, 0.7f, -0.8f,          /* row 1 (unused) */
            0.25f, -0.5f, 0.125f, 0.0625f,     /* row 2 */
            -0.0625f, 0.5f, -0.25f, 0.125f,    /* row 3 */
        };
        float sinks[2] = { 0.0f, 1.0f };
        ds4_metal_tensor *tq = ds4_metal_tensor_alloc(sizeof(qv));
        ds4_metal_tensor *traw = ds4_metal_tensor_alloc(sizeof(raw));
        ds4_metal_tensor *tout = ds4_metal_tensor_alloc(sizeof(qv));
        require(tq && traw && tout, "attn decode raw alloc");
        require(ds4_metal_tensor_write(tq, 0, qv, sizeof(qv)), "attn decode q write");
        require(ds4_metal_tensor_write(traw, 0, raw, sizeof(raw)), "attn decode raw write");
        require(ds4_metal_attention_decode_heads_tensor(tout, sinks, sizeof(sinks), 0,
                                                        tq, traw, n_raw, raw_cap, raw_start,
                                                        NULL, 0, NULL, 0, n_head, head_dim),
                "attn decode raw");
        float got[8];
        read_tensor(tout, got, 8);
        const float kq = 1.0f / sqrtf((float)head_dim);
        for (uint32_t h = 0; h < n_head; h++) {
            float scores[3];
            float maxs = sinks[h];
            for (uint32_t r = 0; r < n_raw; r++) {
                const uint32_t actual = (raw_start + r) % raw_cap;
                float s = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) s += qv[h * head_dim + d] * raw[actual * head_dim + d];
                scores[r] = s * kq;
                if (scores[r] > maxs) maxs = scores[r];
            }
            float denom = expf(sinks[h] - maxs);
            float out[4] = {0};
            for (uint32_t r = 0; r < n_raw; r++) {
                float w = expf(scores[r] - maxs);
                denom += w;
                const uint32_t actual = (raw_start + r) % raw_cap;
                for (uint32_t d = 0; d < head_dim; d++) out[d] += w * raw[actual * head_dim + d];
            }
            for (uint32_t d = 0; d < head_dim; d++) {
                require(nearf(got[h * head_dim + d], out[d] / denom, 1e-5f), "attn decode raw value");
            }
        }
        ds4_metal_tensor_free(tout);
        ds4_metal_tensor_free(traw);
        ds4_metal_tensor_free(tq);
    }

    STEP("attention decode_heads (mixed raw + comp, masked)");
    {
        const uint32_t n_head = 2;
        const uint32_t head_dim = 4;
        const uint32_t raw_cap = 2;
        const uint32_t n_raw = 2;
        const uint32_t raw_start = 0;
        const uint32_t n_comp = 3;
        float qv[2 * 4] = { 0.4f, 0.0f, -0.2f, 0.7f,  0.1f, -0.3f, 0.5f, 0.2f };
        float raw[2 * 4] = { 0.5f, -0.25f, 0.125f, 0.0f,  -0.5f, 0.25f, 0.0f, 0.125f };
        float comp[3 * 4] = {
            1.0f, 0.0f, 0.5f, -0.5f,
            -1.0f, 0.5f, 0.0f, 0.25f,
            0.25f, -0.5f, -0.25f, 1.0f,
        };
        float mask[3] = { 0.0f, -1.0e30f, 0.0f };  /* mask out comp row 1 */
        float sinks[2] = { -0.5f, 0.5f };
        ds4_metal_tensor *tq = ds4_metal_tensor_alloc(sizeof(qv));
        ds4_metal_tensor *traw = ds4_metal_tensor_alloc(sizeof(raw));
        ds4_metal_tensor *tcomp = ds4_metal_tensor_alloc(sizeof(comp));
        ds4_metal_tensor *tmask = ds4_metal_tensor_alloc(sizeof(mask));
        ds4_metal_tensor *tout = ds4_metal_tensor_alloc(sizeof(qv));
        require(tq && traw && tcomp && tmask && tout, "attn mix alloc");
        require(ds4_metal_tensor_write(tq, 0, qv, sizeof(qv)), "attn mix q");
        require(ds4_metal_tensor_write(traw, 0, raw, sizeof(raw)), "attn mix raw");
        require(ds4_metal_tensor_write(tcomp, 0, comp, sizeof(comp)), "attn mix comp");
        require(ds4_metal_tensor_write(tmask, 0, mask, sizeof(mask)), "attn mix mask");
        require(ds4_metal_attention_decode_heads_tensor(tout, sinks, sizeof(sinks), 0,
                                                        tq, traw, n_raw, raw_cap, raw_start,
                                                        tcomp, n_comp, tmask, 1, n_head, head_dim),
                "attn decode mix");
        float got[8];
        read_tensor(tout, got, 8);
        const float kq = 1.0f / sqrtf((float)head_dim);
        for (uint32_t h = 0; h < n_head; h++) {
            float scores_raw[2], scores_comp[3];
            float maxs = sinks[h];
            for (uint32_t r = 0; r < n_raw; r++) {
                float s = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) s += qv[h * head_dim + d] * raw[r * head_dim + d];
                scores_raw[r] = s * kq;
                if (scores_raw[r] > maxs) maxs = scores_raw[r];
            }
            for (uint32_t r = 0; r < n_comp; r++) {
                float s = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) s += qv[h * head_dim + d] * comp[r * head_dim + d];
                scores_comp[r] = s * kq + mask[r];
                if (scores_comp[r] > maxs) maxs = scores_comp[r];
            }
            float denom = expf(sinks[h] - maxs);
            float out[4] = {0};
            for (uint32_t r = 0; r < n_raw; r++) {
                float w = expf(scores_raw[r] - maxs);
                denom += w;
                for (uint32_t d = 0; d < head_dim; d++) out[d] += w * raw[r * head_dim + d];
            }
            for (uint32_t r = 0; r < n_comp; r++) {
                float w = expf(scores_comp[r] - maxs);
                denom += w;
                for (uint32_t d = 0; d < head_dim; d++) out[d] += w * comp[r * head_dim + d];
            }
            for (uint32_t d = 0; d < head_dim; d++) {
                require(nearf(got[h * head_dim + d], out[d] / denom, 1e-5f), "attn decode mix value");
            }
        }
        ds4_metal_tensor_free(tout);
        ds4_metal_tensor_free(tmask);
        ds4_metal_tensor_free(tcomp);
        ds4_metal_tensor_free(traw);
        ds4_metal_tensor_free(tq);
    }

    STEP("attention prefill_raw_heads sliding window");
    {
        const uint32_t n_tokens = 4;
        const uint32_t n_head = 2;
        const uint32_t head_dim = 4;
        const uint32_t window = 2;
        float qv[4 * 2 * 4];
        float raw[4 * 4];
        float sinks[2] = { -0.25f, 0.5f };
        for (uint32_t i = 0; i < 4 * 2 * 4; i++) qv[i] = sinf((float)(i + 1) * 0.5f);
        for (uint32_t i = 0; i < 4 * 4; i++) raw[i] = cosf((float)(i + 1) * 0.4f);
        ds4_metal_tensor *tq = ds4_metal_tensor_alloc(sizeof(qv));
        ds4_metal_tensor *traw = ds4_metal_tensor_alloc(sizeof(raw));
        ds4_metal_tensor *tout = ds4_metal_tensor_alloc(sizeof(qv));
        require(tq && traw && tout, "attn prefill alloc");
        require(ds4_metal_tensor_write(tq, 0, qv, sizeof(qv)), "attn prefill q");
        require(ds4_metal_tensor_write(traw, 0, raw, sizeof(raw)), "attn prefill raw");
        require(ds4_metal_attention_prefill_raw_heads_tensor(tout, sinks, sizeof(sinks), 0,
                                                              tq, traw, n_tokens, window,
                                                              n_head, head_dim),
                "attn prefill raw");
        float got[4 * 2 * 4];
        read_tensor(tout, got, 4 * 2 * 4);
        const float kq = 1.0f / sqrtf((float)head_dim);
        for (uint32_t t = 0; t < n_tokens; t++) {
            const uint32_t r0 = (t + 1u > window) ? t + 1u - window : 0u;
            const uint32_t n_kv = t + 1u - r0;
            for (uint32_t h = 0; h < n_head; h++) {
                float scores[4];
                float maxs = sinks[h];
                for (uint32_t i = 0; i < n_kv; i++) {
                    float s = 0.0f;
                    for (uint32_t d = 0; d < head_dim; d++)
                        s += qv[(t * n_head + h) * head_dim + d] * raw[(r0 + i) * head_dim + d];
                    scores[i] = s * kq;
                    if (scores[i] > maxs) maxs = scores[i];
                }
                float denom = expf(sinks[h] - maxs);
                float out[4] = {0};
                for (uint32_t i = 0; i < n_kv; i++) {
                    float w = expf(scores[i] - maxs);
                    denom += w;
                    for (uint32_t d = 0; d < head_dim; d++) out[d] += w * raw[(r0 + i) * head_dim + d];
                }
                for (uint32_t d = 0; d < head_dim; d++) {
                    float ref = out[d] / denom;
                    require(nearf(got[(t * n_head + h) * head_dim + d], ref, 1e-5f),
                            "attn prefill value");
                }
            }
        }
        ds4_metal_tensor_free(tout);
        ds4_metal_tensor_free(traw);
        ds4_metal_tensor_free(tq);
    }

    STEP("attention decode_raw_batch (ring + window)");
    {
        const uint32_t n_tokens = 3;
        const uint32_t n_head = 2;
        const uint32_t head_dim = 4;
        const uint32_t window = 4;
        const uint32_t pos0 = 5;
        const uint32_t raw_cap = 4;
        const uint32_t n_raw = 4;
        const uint32_t raw_start = 1;
        float qv[3 * 2 * 4];
        float raw[4 * 4];
        float sinks[2] = { 0.1f, -0.2f };
        for (uint32_t i = 0; i < 24; i++) qv[i] = sinf((float)(i + 1) * 0.3f);
        for (uint32_t i = 0; i < 16; i++) raw[i] = cosf((float)(i + 1) * 0.5f);
        ds4_metal_tensor *tq = ds4_metal_tensor_alloc(sizeof(qv));
        ds4_metal_tensor *traw = ds4_metal_tensor_alloc(sizeof(raw));
        ds4_metal_tensor *tout = ds4_metal_tensor_alloc(sizeof(qv));
        require(tq && traw && tout, "decode_raw_batch alloc");
        require(ds4_metal_tensor_write(tq, 0, qv, sizeof(qv)), "write q");
        require(ds4_metal_tensor_write(traw, 0, raw, sizeof(raw)), "write raw");
        require(ds4_metal_attention_decode_raw_batch_heads_tensor(tout, sinks, sizeof(sinks), 0,
                                                                  tq, traw, n_tokens, pos0, n_raw,
                                                                  raw_cap, raw_start, window,
                                                                  n_head, head_dim),
                "decode_raw_batch");
        float got[24], ref[24];
        read_tensor(tout, got, 24);
        attn_batch_oracle(ref, qv, raw, NULL, NULL, NULL, sinks,
                          n_tokens, n_head, head_dim, pos0, n_raw, raw_cap, raw_start,
                          0, window, 1, 0, 0, 0, 0);
        for (uint32_t i = 0; i < 24; i++) require(nearf(got[i], ref[i], 1e-5f), "decode_raw_batch value");
        ds4_metal_tensor_free(tout);
        ds4_metal_tensor_free(traw);
        ds4_metal_tensor_free(tq);
    }

    STEP("attention decode_mixed_batch");
    {
        const uint32_t n_tokens = 2;
        const uint32_t n_head = 2;
        const uint32_t head_dim = 4;
        const uint32_t window = 4;
        const uint32_t pos0 = 6;
        const uint32_t raw_cap = 3;
        const uint32_t n_raw = 3;
        const uint32_t raw_start = 0;
        const uint32_t n_comp = 3;
        const uint32_t ratio = 2;
        float qv[2 * 2 * 4];
        float raw[3 * 4];
        float comp[3 * 4];
        float mask[3] = { 0, -1.0e30f, 0 };
        float sinks[2] = { 0.0f, 0.5f };
        for (uint32_t i = 0; i < 16; i++) qv[i] = 0.1f * (float)((i % 7) - 3);
        for (uint32_t i = 0; i < 12; i++) raw[i] = 0.05f * (float)i + 0.1f;
        for (uint32_t i = 0; i < 12; i++) comp[i] = -0.05f * (float)i + 0.2f;
        ds4_metal_tensor *tq = ds4_metal_tensor_alloc(sizeof(qv));
        ds4_metal_tensor *traw = ds4_metal_tensor_alloc(sizeof(raw));
        ds4_metal_tensor *tcomp = ds4_metal_tensor_alloc(sizeof(comp));
        ds4_metal_tensor *tmask = ds4_metal_tensor_alloc(sizeof(mask));
        ds4_metal_tensor *tout = ds4_metal_tensor_alloc(sizeof(qv));
        require(tq && traw && tcomp && tmask && tout, "decode_mixed_batch alloc");
        require(ds4_metal_tensor_write(tq, 0, qv, sizeof(qv)), "");
        require(ds4_metal_tensor_write(traw, 0, raw, sizeof(raw)), "");
        require(ds4_metal_tensor_write(tcomp, 0, comp, sizeof(comp)), "");
        require(ds4_metal_tensor_write(tmask, 0, mask, sizeof(mask)), "");
        require(ds4_metal_attention_decode_mixed_batch_heads_tensor(tout, sinks, sizeof(sinks), 0,
                                                                    tq, traw, tcomp, tmask, 1,
                                                                    n_tokens, pos0, n_raw, raw_cap, raw_start,
                                                                    n_comp, window, ratio, n_head, head_dim),
                "decode_mixed_batch");
        float got[16], ref[16];
        read_tensor(tout, got, 16);
        attn_batch_oracle(ref, qv, raw, comp, mask, NULL, sinks,
                          n_tokens, n_head, head_dim, pos0, n_raw, raw_cap, raw_start,
                          n_comp, window, ratio, 0, 0, 1, 0);
        for (uint32_t i = 0; i < 16; i++) require(nearf(got[i], ref[i], 1e-5f), "decode_mixed_batch value");
        ds4_metal_tensor_free(tout);
        ds4_metal_tensor_free(tmask);
        ds4_metal_tensor_free(tcomp);
        ds4_metal_tensor_free(traw);
        ds4_metal_tensor_free(tq);
    }

    STEP("attention indexed_mixed_batch");
    {
        const uint32_t n_tokens = 2;
        const uint32_t n_head = 2;
        const uint32_t head_dim = 4;
        const uint32_t window = 8;
        const uint32_t pos0 = 7;
        const uint32_t raw_cap = 3;
        const uint32_t n_raw = 3;
        const uint32_t raw_start = 0;
        const uint32_t n_comp = 4;
        const uint32_t top_k = 2;
        const uint32_t ratio = 2;
        float qv[2 * 2 * 4], raw[3 * 4], comp[4 * 4];
        int32_t topk[2 * 2] = { 0, 2,  1, 3 };
        float sinks[2] = { -0.1f, 0.2f };
        for (uint32_t i = 0; i < 16; i++) qv[i] = 0.05f * (float)((i % 11) - 5);
        for (uint32_t i = 0; i < 12; i++) raw[i] = 0.07f * (float)i;
        for (uint32_t i = 0; i < 16; i++) comp[i] = -0.03f * (float)i;
        ds4_metal_tensor *tq = ds4_metal_tensor_alloc(sizeof(qv));
        ds4_metal_tensor *traw = ds4_metal_tensor_alloc(sizeof(raw));
        ds4_metal_tensor *tcomp = ds4_metal_tensor_alloc(sizeof(comp));
        ds4_metal_tensor *ttk = ds4_metal_tensor_alloc(sizeof(topk));
        ds4_metal_tensor *tout = ds4_metal_tensor_alloc(sizeof(qv));
        require(tq && traw && tcomp && ttk && tout, "indexed alloc");
        require(ds4_metal_tensor_write(tq, 0, qv, sizeof(qv)), "");
        require(ds4_metal_tensor_write(traw, 0, raw, sizeof(raw)), "");
        require(ds4_metal_tensor_write(tcomp, 0, comp, sizeof(comp)), "");
        require(ds4_metal_tensor_write(ttk, 0, topk, sizeof(topk)), "");
        require(ds4_metal_attention_indexed_mixed_batch_heads_tensor(tout, sinks, sizeof(sinks), 0,
                                                                     tq, traw, tcomp, ttk,
                                                                     n_tokens, pos0, n_raw, raw_cap, raw_start,
                                                                     n_comp, top_k, window, ratio,
                                                                     n_head, head_dim),
                "indexed_mixed_batch");
        float got[16], ref[16];
        read_tensor(tout, got, 16);
        attn_batch_oracle(ref, qv, raw, comp, NULL, topk, sinks,
                          n_tokens, n_head, head_dim, pos0, n_raw, raw_cap, raw_start,
                          n_comp, window, ratio, top_k, 1, 0, 0);
        for (uint32_t i = 0; i < 16; i++) require(nearf(got[i], ref[i], 1e-5f), "indexed value");
        ds4_metal_tensor_free(tout);
        ds4_metal_tensor_free(ttk);
        ds4_metal_tensor_free(tcomp);
        ds4_metal_tensor_free(traw);
        ds4_metal_tensor_free(tq);
    }

    STEP("attention prefill_static_mixed");
    {
        const uint32_t n_tokens = 4;
        const uint32_t n_head = 2;
        const uint32_t head_dim = 4;
        const uint32_t window = 0;     /* full causal */
        const uint32_t n_comp = 2;
        const uint32_t ratio = 2;
        float qv[4 * 2 * 4], raw[4 * 4], comp[2 * 4];
        float sinks[2] = { 0.0f, 0.0f };
        for (uint32_t i = 0; i < 32; i++) qv[i] = sinf((float)(i + 1) * 0.2f);
        for (uint32_t i = 0; i < 16; i++) raw[i] = cosf((float)(i + 1) * 0.3f);
        for (uint32_t i = 0; i < 8; i++) comp[i] = sinf((float)(i + 7) * 0.1f);
        ds4_metal_tensor *tq = ds4_metal_tensor_alloc(sizeof(qv));
        ds4_metal_tensor *traw = ds4_metal_tensor_alloc(sizeof(raw));
        ds4_metal_tensor *tcomp = ds4_metal_tensor_alloc(sizeof(comp));
        ds4_metal_tensor *tout = ds4_metal_tensor_alloc(sizeof(qv));
        require(tq && traw && tcomp && tout, "static alloc");
        require(ds4_metal_tensor_write(tq, 0, qv, sizeof(qv)), "");
        require(ds4_metal_tensor_write(traw, 0, raw, sizeof(raw)), "");
        require(ds4_metal_tensor_write(tcomp, 0, comp, sizeof(comp)), "");
        require(ds4_metal_attention_prefill_static_mixed_heads_tensor(tout, sinks, sizeof(sinks), 0,
                                                                      tq, traw, tcomp, n_tokens,
                                                                      n_comp, window, ratio,
                                                                      n_head, head_dim),
                "prefill_static");
        float got[32], ref[32];
        read_tensor(tout, got, 32);
        attn_batch_oracle(ref, qv, raw, comp, NULL, NULL, sinks,
                          n_tokens, n_head, head_dim, 0, n_tokens, n_tokens, 0,
                          n_comp, window, ratio, 0, 0, 0, 1);
        for (uint32_t i = 0; i < 32; i++) require(nearf(got[i], ref[i], 1e-5f), "prefill_static value");
        ds4_metal_tensor_free(tout);
        ds4_metal_tensor_free(tcomp);
        ds4_metal_tensor_free(traw);
        ds4_metal_tensor_free(tq);
    }

    STEP("attention prefill_masked_mixed");
    {
        const uint32_t n_tokens = 3;
        const uint32_t n_head = 2;
        const uint32_t head_dim = 4;
        const uint32_t window = 2;
        const uint32_t n_comp = 2;
        const uint32_t ratio = 2;
        float qv[3 * 2 * 4], raw[3 * 4], comp[2 * 4];
        float mask[2] = { -1.0e30f, 0.0f };
        float sinks[2] = { 0.3f, -0.3f };
        for (uint32_t i = 0; i < 24; i++) qv[i] = 0.1f * (float)((i % 9) - 4);
        for (uint32_t i = 0; i < 12; i++) raw[i] = 0.08f * (float)i;
        for (uint32_t i = 0; i < 8; i++) comp[i] = -0.06f * (float)i;
        ds4_metal_tensor *tq = ds4_metal_tensor_alloc(sizeof(qv));
        ds4_metal_tensor *traw = ds4_metal_tensor_alloc(sizeof(raw));
        ds4_metal_tensor *tcomp = ds4_metal_tensor_alloc(sizeof(comp));
        ds4_metal_tensor *tmask = ds4_metal_tensor_alloc(sizeof(mask));
        ds4_metal_tensor *tout = ds4_metal_tensor_alloc(sizeof(qv));
        require(tq && traw && tcomp && tmask && tout, "masked alloc");
        require(ds4_metal_tensor_write(tq, 0, qv, sizeof(qv)), "");
        require(ds4_metal_tensor_write(traw, 0, raw, sizeof(raw)), "");
        require(ds4_metal_tensor_write(tcomp, 0, comp, sizeof(comp)), "");
        require(ds4_metal_tensor_write(tmask, 0, mask, sizeof(mask)), "");
        require(ds4_metal_attention_prefill_masked_mixed_heads_tensor(tout, sinks, sizeof(sinks), 0,
                                                                      tq, traw, tcomp, tmask,
                                                                      n_tokens, n_comp, window, ratio,
                                                                      n_head, head_dim),
                "prefill_masked");
        float got[24], ref[24];
        read_tensor(tout, got, 24);
        attn_batch_oracle(ref, qv, raw, comp, mask, NULL, sinks,
                          n_tokens, n_head, head_dim, 0, n_tokens, n_tokens, 0,
                          n_comp, window, ratio, 0, 0, 1, 1);
        for (uint32_t i = 0; i < 24; i++) require(nearf(got[i], ref[i], 1e-5f), "prefill_masked value");
        ds4_metal_tensor_free(tout);
        ds4_metal_tensor_free(tmask);
        ds4_metal_tensor_free(tcomp);
        ds4_metal_tensor_free(traw);
        ds4_metal_tensor_free(tq);
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
