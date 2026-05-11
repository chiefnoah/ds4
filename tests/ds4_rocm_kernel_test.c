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

/* CPU mirrors of the GPU routed-MoE block dots, used to oracle the kernels. */

#define MOE_QK_K 256

typedef struct {
    uint8_t  scales[MOE_QK_K / 16];
    uint8_t  qs[MOE_QK_K / 4];
    uint16_t d;
    uint16_t dmin;
} moe_block_q2_K;

typedef struct {
    uint16_t d;
    uint16_t qs[MOE_QK_K / 8];
} moe_block_iq2_xxs;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qs[MOE_QK_K / 2];
} moe_block_q4_K;

static const uint64_t moe_iq2xxs_grid[256] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
};

static const uint8_t moe_ksigns_iq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

static float f16_to_f32(uint16_t bits) {
    uint32_t sign = (uint32_t)(bits >> 15) & 0x1u;
    uint32_t exp  = (uint32_t)(bits >> 10) & 0x1Fu;
    uint32_t mant = (uint32_t)(bits & 0x3FFu);
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) f = sign << 31;
        else {
            while ((mant & 0x400u) == 0) { mant <<= 1; exp = (uint32_t)((int32_t)exp - 1); }
            mant &= 0x3FFu; exp += 1;
            f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | 0x7F800000u | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    }
    union { uint32_t u; float f; } u; u.u = f; return u.f;
}

static float moe_iq2xxs_dot_block_cpu(const moe_block_iq2_xxs *xb, const float *y) {
    const float d = f16_to_f32(xb->d);
    float sum = 0.0f;
    const uint16_t *q = xb->qs;
    for (int ib32 = 0; ib32 < 8; ib32++) {
        const uint32_t aux32_g = (uint32_t)q[0] | ((uint32_t)q[1] << 16);
        const uint32_t aux32_s = (uint32_t)q[2] | ((uint32_t)q[3] << 16);
        q += 4;
        const float dl = d * (0.5f + (float)(aux32_s >> 28)) * 0.25f;
        const uint8_t a[4] = {
            (uint8_t)(aux32_g >>  0), (uint8_t)(aux32_g >>  8),
            (uint8_t)(aux32_g >> 16), (uint8_t)(aux32_g >> 24),
        };
        const uint8_t s[4] = {
            moe_ksigns_iq2xs[(aux32_s >>  0) & 127u],
            moe_ksigns_iq2xs[(aux32_s >>  7) & 127u],
            moe_ksigns_iq2xs[(aux32_s >> 14) & 127u],
            moe_ksigns_iq2xs[(aux32_s >> 21) & 127u],
        };
        for (int sub = 0; sub < 4; sub++) {
            const uint8_t *g = (const uint8_t *)(moe_iq2xxs_grid + a[sub]);
            for (int j = 0; j < 8; j++) {
                const float v = (float)g[j] * (((s[sub] >> j) & 1u) ? -1.0f : 1.0f);
                sum += dl * v * y[sub * 8 + j];
            }
        }
        y += 32;
    }
    return sum;
}

static float moe_q2_K_dot_block_cpu(const moe_block_q2_K *xb, const float *y) {
    const float d = f16_to_f32(xb->d);
    const float dmin = f16_to_f32(xb->dmin);
    const uint8_t *q = xb->qs;
    const uint8_t *sc = xb->scales;
    float sum_d = 0.0f, sum_m = 0.0f;
    int is = 0;
    for (int k = 0; k < 2; k++) {
        int shift = 0;
        for (int j = 0; j < 4; j++) {
            for (int half = 0; half < 2; half++) {
                const float scale = (float)(sc[is] & 0x0F);
                const float minv  = (float)(sc[is] >> 4);
                float dot = 0.0f, sy = 0.0f;
                for (int i = 0; i < 16; i++) {
                    const float yv = y[half * 16 + i];
                    dot += yv * (float)((q[half * 16 + i] >> shift) & 0x3);
                    sy  += yv;
                }
                sum_d += scale * dot;
                sum_m += minv * sy;
                is++;
            }
            shift += 2;
            y += 32;
        }
        q += 32;
    }
    return d * sum_d - dmin * sum_m;
}

static void moe_q4_K_get_scale_min(int j, const uint8_t *q, uint8_t *scale, uint8_t *minv) {
    if (j < 4) {
        *scale = q[j] & 0x3F;
        *minv  = q[j + 4] & 0x3F;
    } else {
        *scale = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        *minv  = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
}

static float moe_q4_K_dot_block_cpu(const moe_block_q4_K *xb, const float *y) {
    const float d = f16_to_f32(xb->d);
    const float dmin = f16_to_f32(xb->dmin);
    const uint8_t *q = xb->qs;
    float sum_d = 0.0f, sum_m = 0.0f;
    for (int j = 0; j < 8; j++) {
        uint8_t scale, mn;
        moe_q4_K_get_scale_min(j, xb->scales, &scale, &mn);
        const int qoff = (j / 2) * 32;
        const int shift = (j & 1) * 4;
        const float *yj = y + j * 32;
        float dot = 0.0f, sy = 0.0f;
        for (int i = 0; i < 32; i++) {
            const float yv = yj[i];
            dot += yv * (float)((q[qoff + i] >> shift) & 0x0F);
            sy += yv;
        }
        sum_d += (float)scale * dot;
        sum_m += (float)mn * sy;
    }
    return d * sum_d - dmin * sum_m;
}

static void moe_build_q4_K_block(moe_block_q4_K *b, uint32_t seed) {
    b->d    = f32_to_f16(0.04f + 0.001f * (float)(seed & 0x7));
    b->dmin = f32_to_f16(0.012f + 0.0006f * (float)(seed & 0xF));
    /* 12-byte scales: 8 sub-groups, each 6-bit scale + 6-bit min. */
    for (int i = 0; i < 12; i++) {
        b->scales[i] = (uint8_t)((seed * 19u + (uint32_t)i * 7u) & 0xFFu);
    }
    for (int i = 0; i < 128; i++) {
        b->qs[i] = (uint8_t)((seed * 23u + (uint32_t)i * 11u) & 0xFFu);
    }
}

/* Build a synthetic IQ2_XXS block with deterministic content from a seed. */
static void moe_build_iq2xxs_block(moe_block_iq2_xxs *b, uint32_t seed) {
    b->d = f32_to_f16(0.05f + 0.001f * (float)(seed & 0x7));
    for (int ib32 = 0; ib32 < 8; ib32++) {
        /* aux32_g: 4 grid indices in [0,256) */
        uint32_t g = 0;
        for (int k = 0; k < 4; k++) {
            uint8_t idx = (uint8_t)((seed * 31u + (uint32_t)ib32 * 17u + (uint32_t)k * 7u) & 0xFFu);
            g |= ((uint32_t)idx) << (k * 8);
        }
        /* aux32_s: 4 sign indices (each 7 bits) + 4-bit scale offset on top. */
        uint32_t s = 0;
        for (int k = 0; k < 4; k++) {
            uint32_t sg = (seed * 13u + (uint32_t)ib32 * 5u + (uint32_t)k * 3u) & 127u;
            s |= sg << (k * 7);
        }
        const uint32_t scale_off = (seed + (uint32_t)ib32) & 0xFu;
        s |= (scale_off << 28);
        b->qs[ib32 * 4 + 0] = (uint16_t)(g & 0xFFFFu);
        b->qs[ib32 * 4 + 1] = (uint16_t)((g >> 16) & 0xFFFFu);
        b->qs[ib32 * 4 + 2] = (uint16_t)(s & 0xFFFFu);
        b->qs[ib32 * 4 + 3] = (uint16_t)((s >> 16) & 0xFFFFu);
    }
}

/* Build a synthetic Q2_K block. */
static void moe_build_q2_K_block(moe_block_q2_K *b, uint32_t seed) {
    b->d    = f32_to_f16(0.03f + 0.001f * (float)(seed & 0x7));
    b->dmin = f32_to_f16(0.01f + 0.0005f * (float)(seed & 0xF));
    for (int i = 0; i < 16; i++) {
        const uint32_t scale = (seed * 11u + (uint32_t)i * 3u) & 0xFu;
        const uint32_t minv  = (seed * 7u  + (uint32_t)i * 5u) & 0xFu;
        b->scales[i] = (uint8_t)((minv << 4) | scale);
    }
    for (int i = 0; i < 64; i++) {
        b->qs[i] = (uint8_t)((seed * 17u + (uint32_t)i * 13u) & 0xFFu);
    }
}

/* Build a Q8_0 weight matrix into `dst` covering `out_dim` rows of `in_dim`
 * inputs.  `gen(r, c)` returns the desired float value for row r, column c.
 * Each block holds 32 quantized int8 values + 2-byte FP16 scale; rows are laid
 * out as ((in_dim+31)/32) blocks contiguously per row.
 *
 * Returns the number of bytes written. */
typedef float (*q8_gen_fn)(uint32_t r, uint32_t c, void *ctx);
__attribute__((unused))
static uint64_t build_q8_0_matrix(uint8_t *dst, uint64_t in_dim, uint64_t out_dim,
                                   q8_gen_fn gen, void *ctx) {
    const uint64_t blocks_per_row = (in_dim + 31u) / 32u;
    const uint64_t row_bytes = blocks_per_row * 34u;
    for (uint64_t r = 0; r < out_dim; r++) {
        for (uint64_t b = 0; b < blocks_per_row; b++) {
            float vals[32];
            float amax = 0.0f;
            for (uint32_t i = 0; i < 32; i++) {
                const uint64_t c = b * 32u + i;
                vals[i] = (c < in_dim) ? gen((uint32_t)r, (uint32_t)c, ctx) : 0.0f;
                if (fabsf(vals[i]) > amax) amax = fabsf(vals[i]);
            }
            float scale = amax / 127.0f;
            if (scale == 0.0f) scale = 1.0f;
            uint16_t hf = f32_to_f16(scale);
            uint8_t *blk = dst + r * row_bytes + b * 34u;
            blk[0] = (uint8_t)(hf & 0xffu);
            blk[1] = (uint8_t)(hf >> 8);
            for (uint32_t i = 0; i < 32; i++) {
                int q = (int)roundf(vals[i] / scale);
                if (q < -127) q = -127;
                if (q > 127) q = 127;
                blk[2 + i] = (uint8_t)(int8_t)q;
            }
        }
    }
    return out_dim * row_bytes;
}

/* CPU oracle for a single Q8_0 row: dot product of row r of W against x[in_dim]. */
static float q8_0_row_dot(const uint8_t *W, uint64_t in_dim, uint64_t r, const float *x) {
    const uint64_t blocks_per_row = (in_dim + 31u) / 32u;
    const uint64_t row_bytes = blocks_per_row * 34u;
    const uint8_t *row = W + r * row_bytes;
    float acc = 0.0f;
    for (uint64_t b = 0; b < blocks_per_row; b++) {
        const uint8_t *blk = row + b * 34u;
        const uint16_t hf = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
        const float scale = f16_to_f32(hf);
        for (uint32_t i = 0; i < 32; i++) {
            const uint64_t c = b * 32u + i;
            if (c >= in_dim) break;
            acc += scale * (float)(int8_t)blk[2 + i] * x[c];
        }
    }
    return acc;
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

    /* Decode-shape F16 matmul: in_dim=256 (multiple of 32*8), out_dim=256
     * (multiple of 8). Hits the new wave_8_n8 dispatch path. The shape was
     * picked small enough to verify on CPU but large enough to exercise the
     * unrolled inner loop and wave32 reduction. */
    STEP("matmul_f16 wave_8_n8 decode shape (in=256, out=256, n_tok=1)");
    {
        const uint32_t IN = 256, OUT = 256;
        uint16_t *wmf = (uint16_t *)malloc((size_t)IN * OUT * sizeof(uint16_t));
        float    *xv  = (float    *)malloc((size_t)IN * sizeof(float));
        float    *ref = (float    *)malloc((size_t)OUT * sizeof(float));
        float    *got = (float    *)malloc((size_t)OUT * sizeof(float));
        require(wmf && xv && ref && got, "wave_8_n8 alloc");
        for (uint32_t i = 0; i < IN; i++) xv[i] = (float)((int)(i % 17) - 8) * 0.125f;
        for (uint32_t r = 0; r < OUT; r++) {
            for (uint32_t i = 0; i < IN; i++) {
                wmf[r * IN + i] = f32_to_f16((float)((int)((r + i) % 23) - 11) * 0.0625f);
            }
        }
        for (uint32_t r = 0; r < OUT; r++) {
            float s = 0.0f;
            for (uint32_t i = 0; i < IN; i++) {
                s += f16_to_f32(wmf[r * IN + i]) * xv[i];
            }
            ref[r] = s;
        }
        ds4_metal_tensor *txw = ds4_metal_tensor_alloc(IN * sizeof(float));
        ds4_metal_tensor *tow = ds4_metal_tensor_alloc(OUT * sizeof(float));
        require(txw && tow, "wave_8_n8 tensor alloc");
        require(ds4_metal_tensor_write(txw, 0, xv, IN * sizeof(float)), "wave_8_n8 x write");
        require(ds4_metal_matmul_f16_tensor(tow, wmf,
                    (uint64_t)IN * OUT * sizeof(uint16_t), 0,
                    IN, OUT, txw, 1), "matmul_f16 wave_8_n8");
        require(ds4_metal_tensor_read(tow, 0, got, OUT * sizeof(float)),
                "wave_8_n8 read");
        for (uint32_t r = 0; r < OUT; r++) {
            require(nearf(got[r], ref[r], 1e-2f), "wave_8_n8 value");
        }
        ds4_metal_tensor_free(tow);
        ds4_metal_tensor_free(txw);
        free(got); free(ref); free(xv); free(wmf);
    }

    /* Same for wave_4_n8 path: out_dim divisible by 4 but not by 8. */
    STEP("matmul_f16 wave_4_n8 decode shape (in=256, out=20, n_tok=1)");
    {
        const uint32_t IN = 256, OUT = 20;
        uint16_t *wmf = (uint16_t *)malloc((size_t)IN * OUT * sizeof(uint16_t));
        float    *xv  = (float    *)malloc((size_t)IN * sizeof(float));
        float    *ref = (float    *)malloc((size_t)OUT * sizeof(float));
        float    *got = (float    *)malloc((size_t)OUT * sizeof(float));
        require(wmf && xv && ref && got, "wave_4_n8 alloc");
        for (uint32_t i = 0; i < IN; i++) xv[i] = (float)((int)(i % 13) - 6) * 0.125f;
        for (uint32_t r = 0; r < OUT; r++) {
            for (uint32_t i = 0; i < IN; i++) {
                wmf[r * IN + i] = f32_to_f16((float)((int)((r * 3 + i) % 19) - 9) * 0.0625f);
            }
        }
        for (uint32_t r = 0; r < OUT; r++) {
            float s = 0.0f;
            for (uint32_t i = 0; i < IN; i++) {
                s += f16_to_f32(wmf[r * IN + i]) * xv[i];
            }
            ref[r] = s;
        }
        ds4_metal_tensor *txw = ds4_metal_tensor_alloc(IN * sizeof(float));
        ds4_metal_tensor *tow = ds4_metal_tensor_alloc(OUT * sizeof(float));
        require(txw && tow, "wave_4_n8 tensor alloc");
        require(ds4_metal_tensor_write(txw, 0, xv, IN * sizeof(float)), "wave_4_n8 x write");
        require(ds4_metal_matmul_f16_tensor(tow, wmf,
                    (uint64_t)IN * OUT * sizeof(uint16_t), 0,
                    IN, OUT, txw, 1), "matmul_f16 wave_4_n8");
        require(ds4_metal_tensor_read(tow, 0, got, OUT * sizeof(float)),
                "wave_4_n8 read");
        for (uint32_t r = 0; r < OUT; r++) {
            require(nearf(got[r], ref[r], 1e-2f), "wave_4_n8 value");
        }
        ds4_metal_tensor_free(tow);
        ds4_metal_tensor_free(txw);
        free(got); free(ref); free(xv); free(wmf);
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

    STEP("attention decode_heads (head_dim=64, mixed + masked)");
    {
        /* Production-shaped head_dim (multiple of wave32) — the existing
         * head_dim=4 STEPs above don't exercise the wave-aligned reduction
         * path the kernel takes for real models. */
        const uint32_t n_head = 3;
        const uint32_t head_dim = 64;
        const uint32_t raw_cap = 5;
        const uint32_t n_raw = 4;
        const uint32_t raw_start = 3;            /* wraps: rows 3,4,0,1 */
        const uint32_t n_comp = 6;
        float qv[3 * 64];
        float raw[5 * 64];
        float comp[6 * 64];
        float mask[6] = { 0.0f, -1.0e30f, 0.0f, -1.0e30f, 0.0f, 0.0f };
        float sinks[3] = { -0.25f, 0.5f, 1.5f };
        for (uint32_t i = 0; i < 3 * 64; i++) qv[i]  = sinf((float)(i + 1) * 0.05f);
        for (uint32_t i = 0; i < 5 * 64; i++) raw[i] = cosf((float)(i + 1) * 0.07f);
        for (uint32_t i = 0; i < 6 * 64; i++) comp[i] = sinf((float)(i + 7) * 0.03f) * 0.5f;
        ds4_metal_tensor *tq    = ds4_metal_tensor_alloc(sizeof(qv));
        ds4_metal_tensor *traw  = ds4_metal_tensor_alloc(sizeof(raw));
        ds4_metal_tensor *tcomp = ds4_metal_tensor_alloc(sizeof(comp));
        ds4_metal_tensor *tmask = ds4_metal_tensor_alloc(sizeof(mask));
        ds4_metal_tensor *tout  = ds4_metal_tensor_alloc(sizeof(qv));
        require(tq && traw && tcomp && tmask && tout, "attn fused alloc");
        require(ds4_metal_tensor_write(tq, 0, qv, sizeof(qv)),       "attn fused q write");
        require(ds4_metal_tensor_write(traw, 0, raw, sizeof(raw)),    "attn fused raw write");
        require(ds4_metal_tensor_write(tcomp, 0, comp, sizeof(comp)), "attn fused comp write");
        require(ds4_metal_tensor_write(tmask, 0, mask, sizeof(mask)), "attn fused mask write");
        require(ds4_metal_attention_decode_heads_tensor(tout, sinks, sizeof(sinks), 0,
                                                        tq, traw, n_raw, raw_cap, raw_start,
                                                        tcomp, n_comp, tmask, 1, n_head, head_dim),
                "attn fused dispatch");
        float got[3 * 64];
        read_tensor(tout, got, 3 * 64);
        const float kq = 1.0f / sqrtf((float)head_dim);
        for (uint32_t h = 0; h < n_head; h++) {
            float scores_raw[4], scores_comp[6];
            float maxs = sinks[h];
            for (uint32_t r = 0; r < n_raw; r++) {
                const uint32_t actual = (raw_start + r) % raw_cap;
                float s = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) s += qv[h * head_dim + d] * raw[actual * head_dim + d];
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
            float out[64] = {0};
            for (uint32_t r = 0; r < n_raw; r++) {
                float w = expf(scores_raw[r] - maxs);
                denom += w;
                const uint32_t actual = (raw_start + r) % raw_cap;
                for (uint32_t d = 0; d < head_dim; d++) out[d] += w * raw[actual * head_dim + d];
            }
            for (uint32_t r = 0; r < n_comp; r++) {
                /* Mask -inf collapses to weight 0; matches what the kernel does. */
                if (scores_comp[r] <= -5.0e29f) continue;
                float w = expf(scores_comp[r] - maxs);
                denom += w;
                for (uint32_t d = 0; d < head_dim; d++) out[d] += w * comp[r * head_dim + d];
            }
            for (uint32_t d = 0; d < head_dim; d++) {
                require(nearf(got[h * head_dim + d], out[d] / denom, 1e-4f),
                        "attn fused value");
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

    STEP("routed_moe_one (iq2xxs gate/up + q2_K down)");
    {
        const uint32_t in_dim = 256;        /* one block per row */
        const uint32_t mid_dim = 256;
        const uint32_t out_dim = 256;
        const uint32_t n_used = 2;
        const uint32_t n_total = 256;       /* host wrapper assumes 256 experts */
        const float clamp = 2.5f;

        const uint64_t gate_row_bytes = sizeof(moe_block_iq2_xxs);
        const uint64_t gate_expert_bytes = (uint64_t)mid_dim * gate_row_bytes;
        const uint64_t down_row_bytes = sizeof(moe_block_q2_K);
        const uint64_t down_expert_bytes = (uint64_t)out_dim * down_row_bytes;

        /* Synthetic mmap: gate_pool || up_pool || down_pool. */
        const uint64_t gate_pool = (uint64_t)n_total * gate_expert_bytes;
        const uint64_t down_pool = (uint64_t)n_total * down_expert_bytes;
        const uint64_t model_size = gate_pool + gate_pool + down_pool;
        uint8_t *model = (uint8_t *)calloc(1, (size_t)model_size);
        require(model != NULL, "moe model alloc");

        moe_block_iq2_xxs *gate_pool_p = (moe_block_iq2_xxs *)(model + 0);
        moe_block_iq2_xxs *up_pool_p   = (moe_block_iq2_xxs *)(model + gate_pool);
        moe_block_q2_K    *down_pool_p = (moe_block_q2_K    *)(model + gate_pool + gate_pool);

        /* Only the experts we'll select need to be initialized; the rest stay zero. */
        const int32_t selected_vals[2] = { 17, 142 };
        const float weights_vals[2] = { 0.6f, 0.4f };
        for (uint32_t s = 0; s < n_used; s++) {
            const int32_t e = selected_vals[s];
            for (uint32_t r = 0; r < mid_dim; r++) {
                moe_build_iq2xxs_block(&gate_pool_p[(uint64_t)e * mid_dim + r],
                                       (uint32_t)e * 1009u + r * 7u);
                moe_build_iq2xxs_block(&up_pool_p[(uint64_t)e * mid_dim + r],
                                       (uint32_t)e * 1013u + r * 11u);
            }
            for (uint32_t r = 0; r < out_dim; r++) {
                moe_build_q2_K_block(&down_pool_p[(uint64_t)e * out_dim + r],
                                     (uint32_t)e * 1019u + r * 13u);
            }
        }

        float xv[256];
        for (uint32_t i = 0; i < 256; i++) xv[i] = 0.02f * (float)((int)(i % 19) - 9);

        ds4_metal_tensor *tx = ds4_metal_tensor_alloc(sizeof(xv));
        ds4_metal_tensor *tsel = ds4_metal_tensor_alloc(sizeof(selected_vals));
        ds4_metal_tensor *twts = ds4_metal_tensor_alloc(sizeof(weights_vals));
        ds4_metal_tensor *tmid = ds4_metal_tensor_alloc((uint64_t)n_used * mid_dim * sizeof(float));
        ds4_metal_tensor *tout_moe = ds4_metal_tensor_alloc((uint64_t)out_dim * sizeof(float));
        require(tx && tsel && twts && tmid && tout_moe, "moe tensor alloc");
        require(ds4_metal_tensor_write(tx, 0, xv, sizeof(xv)), "moe x write");
        require(ds4_metal_tensor_write(tsel, 0, selected_vals, sizeof(selected_vals)), "moe sel write");
        require(ds4_metal_tensor_write(twts, 0, weights_vals, sizeof(weights_vals)), "moe wts write");

        require(ds4_metal_routed_moe_one_tensor(tout_moe, NULL, NULL, tmid, NULL,
                                                 model, model_size,
                                                 0, gate_pool, gate_pool + gate_pool,
                                                 16u /* IQ2_XXS */, 10u /* Q2_K */,
                                                 gate_expert_bytes, gate_row_bytes,
                                                 down_expert_bytes, down_row_bytes,
                                                 in_dim, mid_dim, out_dim,
                                                 tsel, twts, n_used, clamp, tx),
                "routed_moe_one");

        /* CPU oracle. */
        float ref_mid[2 * 256];
        for (uint32_t s = 0; s < n_used; s++) {
            const int32_t e = selected_vals[s];
            for (uint32_t r = 0; r < mid_dim; r++) {
                float gate_v = moe_iq2xxs_dot_block_cpu(&gate_pool_p[(uint64_t)e * mid_dim + r], xv);
                float up_v   = moe_iq2xxs_dot_block_cpu(&up_pool_p[(uint64_t)e * mid_dim + r],   xv);
                if (gate_v > clamp) gate_v = clamp;
                if (up_v > clamp) up_v = clamp;
                if (up_v < -clamp) up_v = -clamp;
                const float silu = gate_v / (1.0f + expf(-gate_v));
                ref_mid[s * mid_dim + r] = silu * up_v * weights_vals[s];
            }
        }
        float got_mid[2 * 256];
        read_tensor(tmid, got_mid, n_used * mid_dim);
        for (uint32_t i = 0; i < n_used * mid_dim; i++) {
            require(nearf(got_mid[i], ref_mid[i], 1e-4f), "routed_moe_one mid value");
        }

        float ref_out[256] = {0};
        for (uint32_t r = 0; r < out_dim; r++) {
            float acc = 0.0f;
            for (uint32_t s = 0; s < n_used; s++) {
                const int32_t e = selected_vals[s];
                acc += moe_q2_K_dot_block_cpu(&down_pool_p[(uint64_t)e * out_dim + r],
                                              ref_mid + (uint64_t)s * mid_dim);
            }
            ref_out[r] = acc;
        }
        float got_out[256];
        read_tensor(tout_moe, got_out, out_dim);
        for (uint32_t i = 0; i < out_dim; i++) {
            require(nearf(got_out[i], ref_out[i], 1e-3f), "routed_moe_one out value");
        }

        ds4_metal_tensor_free(tout_moe);
        ds4_metal_tensor_free(tmid);
        ds4_metal_tensor_free(twts);
        ds4_metal_tensor_free(tsel);
        ds4_metal_tensor_free(tx);
        free(model);
    }

    STEP("routed_moe_batch (n_tokens=2)");
    {
        const uint32_t in_dim = 256;
        const uint32_t mid_dim = 256;
        const uint32_t out_dim = 256;
        const uint32_t n_used = 2;
        const uint32_t n_total = 256;
        const uint32_t n_tokens = 2;
        const float clamp = 2.5f;

        const uint64_t gate_row_bytes = sizeof(moe_block_iq2_xxs);
        const uint64_t gate_expert_bytes = (uint64_t)mid_dim * gate_row_bytes;
        const uint64_t down_row_bytes = sizeof(moe_block_q2_K);
        const uint64_t down_expert_bytes = (uint64_t)out_dim * down_row_bytes;
        const uint64_t gate_pool = (uint64_t)n_total * gate_expert_bytes;
        const uint64_t down_pool = (uint64_t)n_total * down_expert_bytes;
        const uint64_t model_size = gate_pool + gate_pool + down_pool;
        uint8_t *model = (uint8_t *)calloc(1, (size_t)model_size);
        require(model != NULL, "moe batch model alloc");

        moe_block_iq2_xxs *gate_pool_p = (moe_block_iq2_xxs *)(model + 0);
        moe_block_iq2_xxs *up_pool_p   = (moe_block_iq2_xxs *)(model + gate_pool);
        moe_block_q2_K    *down_pool_p = (moe_block_q2_K    *)(model + gate_pool + gate_pool);

        const int32_t selected_vals[4] = { 4, 91, 200, 33 };  /* 2 tokens × 2 used */
        const float weights_vals[4] = { 0.5f, 0.5f, 0.7f, 0.3f };
        for (uint32_t s = 0; s < 4; s++) {
            const int32_t e = selected_vals[s];
            for (uint32_t r = 0; r < mid_dim; r++) {
                moe_build_iq2xxs_block(&gate_pool_p[(uint64_t)e * mid_dim + r],
                                       (uint32_t)e * 1009u + r * 7u);
                moe_build_iq2xxs_block(&up_pool_p[(uint64_t)e * mid_dim + r],
                                       (uint32_t)e * 1013u + r * 11u);
            }
            for (uint32_t r = 0; r < out_dim; r++) {
                moe_build_q2_K_block(&down_pool_p[(uint64_t)e * out_dim + r],
                                     (uint32_t)e * 1019u + r * 13u);
            }
        }

        float xv[2 * 256];
        for (uint32_t t = 0; t < n_tokens; t++) {
            for (uint32_t i = 0; i < 256; i++) {
                xv[t * 256 + i] = 0.015f * (float)((int)((i + t * 5) % 23) - 11);
            }
        }

        ds4_metal_tensor *tx = ds4_metal_tensor_alloc(sizeof(xv));
        ds4_metal_tensor *tsel = ds4_metal_tensor_alloc(sizeof(selected_vals));
        ds4_metal_tensor *twts = ds4_metal_tensor_alloc(sizeof(weights_vals));
        ds4_metal_tensor *tmid = ds4_metal_tensor_alloc((uint64_t)n_tokens * n_used * mid_dim * sizeof(float));
        ds4_metal_tensor *tout_moe = ds4_metal_tensor_alloc((uint64_t)n_tokens * out_dim * sizeof(float));
        require(tx && tsel && twts && tmid && tout_moe, "moe batch tensor alloc");
        require(ds4_metal_tensor_write(tx, 0, xv, sizeof(xv)), "moe batch x write");
        require(ds4_metal_tensor_write(tsel, 0, selected_vals, sizeof(selected_vals)), "moe batch sel write");
        require(ds4_metal_tensor_write(twts, 0, weights_vals, sizeof(weights_vals)), "moe batch wts write");

        require(ds4_metal_routed_moe_batch_tensor(tout_moe, NULL, NULL, tmid, NULL,
                                                   model, model_size,
                                                   0, gate_pool, gate_pool + gate_pool,
                                                   16u, 10u,
                                                   gate_expert_bytes, gate_row_bytes,
                                                   down_expert_bytes, down_row_bytes,
                                                   in_dim, mid_dim, out_dim,
                                                   tsel, twts, n_used, clamp, tx, n_tokens),
                "routed_moe_batch");

        float ref_mid[2 * 2 * 256];
        for (uint32_t t = 0; t < n_tokens; t++) {
            for (uint32_t s = 0; s < n_used; s++) {
                const int32_t e = selected_vals[t * n_used + s];
                const float w  = weights_vals[t * n_used + s];
                for (uint32_t r = 0; r < mid_dim; r++) {
                    float gate_v = moe_iq2xxs_dot_block_cpu(&gate_pool_p[(uint64_t)e * mid_dim + r], xv + t * in_dim);
                    float up_v   = moe_iq2xxs_dot_block_cpu(&up_pool_p[(uint64_t)e * mid_dim + r],   xv + t * in_dim);
                    if (gate_v > clamp) gate_v = clamp;
                    if (up_v > clamp) up_v = clamp;
                    if (up_v < -clamp) up_v = -clamp;
                    const float silu = gate_v / (1.0f + expf(-gate_v));
                    ref_mid[(t * n_used + s) * mid_dim + r] = silu * up_v * w;
                }
            }
        }

        float got_mid[2 * 2 * 256];
        read_tensor(tmid, got_mid, n_tokens * n_used * mid_dim);
        for (uint32_t i = 0; i < n_tokens * n_used * mid_dim; i++) {
            require(nearf(got_mid[i], ref_mid[i], 1e-4f), "routed_moe_batch mid value");
        }

        float ref_out[2 * 256] = {0};
        for (uint32_t t = 0; t < n_tokens; t++) {
            for (uint32_t r = 0; r < out_dim; r++) {
                float acc = 0.0f;
                for (uint32_t s = 0; s < n_used; s++) {
                    const int32_t e = selected_vals[t * n_used + s];
                    acc += moe_q2_K_dot_block_cpu(&down_pool_p[(uint64_t)e * out_dim + r],
                                                  ref_mid + (uint64_t)(t * n_used + s) * mid_dim);
                }
                ref_out[t * out_dim + r] = acc;
            }
        }
        float got_out[2 * 256];
        read_tensor(tout_moe, got_out, n_tokens * out_dim);
        for (uint32_t i = 0; i < n_tokens * out_dim; i++) {
            require(nearf(got_out[i], ref_out[i], 1e-3f), "routed_moe_batch out value");
        }

        ds4_metal_tensor_free(tout_moe);
        ds4_metal_tensor_free(tmid);
        ds4_metal_tensor_free(twts);
        ds4_metal_tensor_free(tsel);
        ds4_metal_tensor_free(tx);
        free(model);
    }

    STEP("routed_moe_one with Q4_K gate/up format");
    {
        /* Mirrors the IQ2_XXS test but uses Q4_K for gate/up to exercise that
         * dequant path (the routed-MoE production checkpoints typically use
         * IQ2_XXS, but the kernel supports Q4_K and we want to be sure the
         * scale/min unpacking math is right). */
        const uint32_t in_dim = 256;
        const uint32_t mid_dim = 256;
        const uint32_t out_dim = 256;
        const uint32_t n_used = 2;
        const uint32_t n_total = 256;
        const float clamp = 4.0f;

        const uint64_t gate_row_bytes = sizeof(moe_block_q4_K);
        const uint64_t gate_expert_bytes = (uint64_t)mid_dim * gate_row_bytes;
        const uint64_t down_row_bytes = sizeof(moe_block_q2_K);
        const uint64_t down_expert_bytes = (uint64_t)out_dim * down_row_bytes;
        const uint64_t gate_pool = (uint64_t)n_total * gate_expert_bytes;
        const uint64_t down_pool = (uint64_t)n_total * down_expert_bytes;
        const uint64_t model_size = gate_pool + gate_pool + down_pool;
        uint8_t *model = (uint8_t *)calloc(1, (size_t)model_size);
        require(model != NULL, "q4 moe model alloc");

        moe_block_q4_K *gate_pool_p = (moe_block_q4_K *)(model + 0);
        moe_block_q4_K *up_pool_p   = (moe_block_q4_K *)(model + gate_pool);
        moe_block_q2_K *down_pool_p = (moe_block_q2_K *)(model + gate_pool + gate_pool);

        const int32_t selected_vals[2] = { 5, 99 };
        const float weights_vals[2] = { 0.7f, 0.3f };
        for (uint32_t s = 0; s < n_used; s++) {
            const int32_t e = selected_vals[s];
            for (uint32_t r = 0; r < mid_dim; r++) {
                moe_build_q4_K_block(&gate_pool_p[(uint64_t)e * mid_dim + r],
                                     (uint32_t)e * 1031u + r * 5u);
                moe_build_q4_K_block(&up_pool_p[(uint64_t)e * mid_dim + r],
                                     (uint32_t)e * 1033u + r * 7u);
            }
            for (uint32_t r = 0; r < out_dim; r++) {
                moe_build_q2_K_block(&down_pool_p[(uint64_t)e * out_dim + r],
                                     (uint32_t)e * 1039u + r * 9u);
            }
        }

        float xv[256];
        for (uint32_t i = 0; i < 256; i++) xv[i] = 0.005f * (float)((int)(i % 11) - 5);

        ds4_metal_tensor *tx = ds4_metal_tensor_alloc(sizeof(xv));
        ds4_metal_tensor *tsel = ds4_metal_tensor_alloc(sizeof(selected_vals));
        ds4_metal_tensor *twts = ds4_metal_tensor_alloc(sizeof(weights_vals));
        ds4_metal_tensor *tmid = ds4_metal_tensor_alloc((uint64_t)n_used * mid_dim * sizeof(float));
        ds4_metal_tensor *tout_moe = ds4_metal_tensor_alloc((uint64_t)out_dim * sizeof(float));
        require(tx && tsel && twts && tmid && tout_moe, "q4 moe tensor alloc");
        require(ds4_metal_tensor_write(tx, 0, xv, sizeof(xv)), "");
        require(ds4_metal_tensor_write(tsel, 0, selected_vals, sizeof(selected_vals)), "");
        require(ds4_metal_tensor_write(twts, 0, weights_vals, sizeof(weights_vals)), "");

        require(ds4_metal_routed_moe_one_tensor(tout_moe, NULL, NULL, tmid, NULL,
                                                 model, model_size,
                                                 0, gate_pool, gate_pool + gate_pool,
                                                 12u /* Q4_K */, 10u /* Q2_K */,
                                                 gate_expert_bytes, gate_row_bytes,
                                                 down_expert_bytes, down_row_bytes,
                                                 in_dim, mid_dim, out_dim,
                                                 tsel, twts, n_used, clamp, tx),
                "routed_moe_one Q4_K");

        float ref_mid[2 * 256];
        for (uint32_t s = 0; s < n_used; s++) {
            const int32_t e = selected_vals[s];
            for (uint32_t r = 0; r < mid_dim; r++) {
                float gate_v = moe_q4_K_dot_block_cpu(&gate_pool_p[(uint64_t)e * mid_dim + r], xv);
                float up_v   = moe_q4_K_dot_block_cpu(&up_pool_p[(uint64_t)e * mid_dim + r], xv);
                if (gate_v > clamp) gate_v = clamp;
                if (up_v > clamp) up_v = clamp;
                if (up_v < -clamp) up_v = -clamp;
                const float silu = gate_v / (1.0f + expf(-gate_v));
                ref_mid[s * mid_dim + r] = silu * up_v * weights_vals[s];
            }
        }
        float got_mid[2 * 256];
        read_tensor(tmid, got_mid, n_used * mid_dim);
        for (uint32_t i = 0; i < n_used * mid_dim; i++) {
            require(nearf(got_mid[i], ref_mid[i], 5e-3f), "Q4_K moe mid value");
        }

        ds4_metal_tensor_free(tout_moe);
        ds4_metal_tensor_free(tmid);
        ds4_metal_tensor_free(twts);
        ds4_metal_tensor_free(tsel);
        ds4_metal_tensor_free(tx);
        free(model);
    }

    STEP("dequant edge cases (hand-crafted blocks)");
    {
        /* These check the kernel against the per-block CPU oracle on blocks
         * that the seeded random builders are unlikely to hit:
         *   - IQ2_XXS with d=0  -> all-zero contribution
         *   - IQ2_XXS with sign_idx=127 (full kmask) -> every lane flips
         *   - IQ2_XXS with grid index 0 (all 0x08 bytes) -> dequant = dl*8
         *   - Q2_K with dmin=0, all scales packed at max nibble (0xF in low,
         *     0xF in high) -> stresses the (sc & 0xF) and (sc >> 4) splits
         *   - Q2_K with d=0 -> only -dmin*sum_y contribution survives
         */
        const uint32_t in_dim = 256;
        const uint32_t mid_dim = 256;
        const uint32_t out_dim = 256;
        const uint32_t n_used = 1;
        const uint32_t n_total = 256;
        const float clamp = 100.0f;  /* effectively no clamp */

        const uint64_t gate_row_bytes = sizeof(moe_block_iq2_xxs);
        const uint64_t gate_expert_bytes = (uint64_t)mid_dim * gate_row_bytes;
        const uint64_t down_row_bytes = sizeof(moe_block_q2_K);
        const uint64_t down_expert_bytes = (uint64_t)out_dim * down_row_bytes;
        const uint64_t gate_pool = (uint64_t)n_total * gate_expert_bytes;
        const uint64_t down_pool = (uint64_t)n_total * down_expert_bytes;
        const uint64_t model_size = gate_pool + gate_pool + down_pool;
        uint8_t *model = (uint8_t *)calloc(1, (size_t)model_size);
        require(model != NULL, "edge-case model alloc");

        moe_block_iq2_xxs *gate_pool_p = (moe_block_iq2_xxs *)(model + 0);
        moe_block_iq2_xxs *up_pool_p   = (moe_block_iq2_xxs *)(model + gate_pool);
        moe_block_q2_K    *down_pool_p = (moe_block_q2_K    *)(model + gate_pool + gate_pool);

        const int32_t selected_vals[1] = { 7 };
        const float weights_vals[1] = { 1.0f };

        /* Gate row 0..mid_dim-1 cycles through edge-case patterns. */
        for (uint32_t r = 0; r < mid_dim; r++) {
            moe_block_iq2_xxs *g = &gate_pool_p[(uint64_t)selected_vals[0] * mid_dim + r];
            moe_block_iq2_xxs *u = &up_pool_p[(uint64_t)selected_vals[0] * mid_dim + r];
            const uint32_t mode = r % 4u;
            if (mode == 0) {
                /* d=0 -> dot must be exactly 0 regardless of qs/scales. */
                g->d = 0; u->d = 0;
                for (int i = 0; i < 32; i++) { g->qs[i] = (uint16_t)i; u->qs[i] = (uint16_t)(i + 1); }
            } else if (mode == 1) {
                /* sign_idx=127 across the whole block -> every lane is negated. */
                g->d = f32_to_f16(0.04f);
                u->d = f32_to_f16(0.04f);
                for (int ib32 = 0; ib32 < 8; ib32++) {
                    uint32_t aux_g = 0u;  /* grid idx 0 = all-8 grid */
                    uint32_t aux_s = (127u << 0) | (127u << 7) | (127u << 14) | (127u << 21);
                    aux_s |= (15u << 28);  /* max scale offset 15 -> dl = d * 15.5 * 0.25 */
                    g->qs[ib32 * 4 + 0] = (uint16_t)(aux_g & 0xFFFFu);
                    g->qs[ib32 * 4 + 1] = (uint16_t)((aux_g >> 16) & 0xFFFFu);
                    g->qs[ib32 * 4 + 2] = (uint16_t)(aux_s & 0xFFFFu);
                    g->qs[ib32 * 4 + 3] = (uint16_t)((aux_s >> 16) & 0xFFFFu);
                    u->qs[ib32 * 4 + 0] = (uint16_t)(aux_g & 0xFFFFu);
                    u->qs[ib32 * 4 + 1] = (uint16_t)((aux_g >> 16) & 0xFFFFu);
                    u->qs[ib32 * 4 + 2] = (uint16_t)(aux_s & 0xFFFFu);
                    u->qs[ib32 * 4 + 3] = (uint16_t)((aux_s >> 16) & 0xFFFFu);
                }
            } else if (mode == 2) {
                /* grid idx 0 (all-8) with sign_idx=0 -> all positive 8s. */
                g->d = f32_to_f16(0.025f);
                u->d = f32_to_f16(0.025f);
                for (int ib32 = 0; ib32 < 8; ib32++) {
                    uint32_t aux_g = 0u, aux_s = 0u;
                    g->qs[ib32 * 4 + 0] = (uint16_t)(aux_g & 0xFFFFu);
                    g->qs[ib32 * 4 + 1] = (uint16_t)((aux_g >> 16) & 0xFFFFu);
                    g->qs[ib32 * 4 + 2] = (uint16_t)(aux_s & 0xFFFFu);
                    g->qs[ib32 * 4 + 3] = (uint16_t)((aux_s >> 16) & 0xFFFFu);
                    u->qs[ib32 * 4 + 0] = (uint16_t)(aux_g & 0xFFFFu);
                    u->qs[ib32 * 4 + 1] = (uint16_t)((aux_g >> 16) & 0xFFFFu);
                    u->qs[ib32 * 4 + 2] = (uint16_t)(aux_s & 0xFFFFu);
                    u->qs[ib32 * 4 + 3] = (uint16_t)((aux_s >> 16) & 0xFFFFu);
                }
            } else {
                /* Standard seeded block as a control. */
                moe_build_iq2xxs_block(g, 5000u + r);
                moe_build_iq2xxs_block(u, 6000u + r);
            }
        }

        /* Down row 0..out_dim-1 cycles through Q2_K edge cases. */
        for (uint32_t r = 0; r < out_dim; r++) {
            moe_block_q2_K *d = &down_pool_p[(uint64_t)selected_vals[0] * out_dim + r];
            const uint32_t mode = r % 4u;
            if (mode == 0) {
                /* dmin=0, all scales = max-low (0xF), all mins = 0. */
                d->d    = f32_to_f16(0.02f);
                d->dmin = 0;
                for (int i = 0; i < 16; i++) d->scales[i] = 0x0F;
                for (int i = 0; i < 64; i++) d->qs[i] = (uint8_t)((i * 7u) & 0xFFu);
            } else if (mode == 1) {
                /* d=0 -> only -dmin*sum_y contribution survives. */
                d->d    = 0;
                d->dmin = f32_to_f16(0.01f);
                for (int i = 0; i < 16; i++) d->scales[i] = (uint8_t)((0xFu << 4) | 0xFu);
                for (int i = 0; i < 64; i++) d->qs[i] = 0xFFu;
            } else if (mode == 2) {
                /* All scales/mins zero -> dot must be exactly 0. */
                d->d    = f32_to_f16(0.03f);
                d->dmin = f32_to_f16(0.01f);
                memset(d->scales, 0, 16);
                for (int i = 0; i < 64; i++) d->qs[i] = (uint8_t)i;
            } else {
                moe_build_q2_K_block(d, 7000u + r);
            }
        }

        float xv[256];
        for (uint32_t i = 0; i < 256; i++) xv[i] = 0.012f * (float)((int)(i % 13) - 6);

        ds4_metal_tensor *tx = ds4_metal_tensor_alloc(sizeof(xv));
        ds4_metal_tensor *tsel = ds4_metal_tensor_alloc(sizeof(selected_vals));
        ds4_metal_tensor *twts = ds4_metal_tensor_alloc(sizeof(weights_vals));
        ds4_metal_tensor *tmid = ds4_metal_tensor_alloc((uint64_t)n_used * mid_dim * sizeof(float));
        ds4_metal_tensor *tout_moe = ds4_metal_tensor_alloc((uint64_t)out_dim * sizeof(float));
        require(tx && tsel && twts && tmid && tout_moe, "edge-case tensor alloc");
        require(ds4_metal_tensor_write(tx, 0, xv, sizeof(xv)), "");
        require(ds4_metal_tensor_write(tsel, 0, selected_vals, sizeof(selected_vals)), "");
        require(ds4_metal_tensor_write(twts, 0, weights_vals, sizeof(weights_vals)), "");

        require(ds4_metal_routed_moe_one_tensor(tout_moe, NULL, NULL, tmid, NULL,
                                                 model, model_size,
                                                 0, gate_pool, gate_pool + gate_pool,
                                                 16u, 10u,
                                                 gate_expert_bytes, gate_row_bytes,
                                                 down_expert_bytes, down_row_bytes,
                                                 in_dim, mid_dim, out_dim,
                                                 tsel, twts, n_used, clamp, tx),
                "routed_moe_one edge cases");

        float ref_mid[256];
        for (uint32_t r = 0; r < mid_dim; r++) {
            float gate_v = moe_iq2xxs_dot_block_cpu(&gate_pool_p[(uint64_t)selected_vals[0] * mid_dim + r], xv);
            float up_v   = moe_iq2xxs_dot_block_cpu(&up_pool_p[(uint64_t)selected_vals[0] * mid_dim + r], xv);
            const float silu = gate_v / (1.0f + expf(-gate_v));
            ref_mid[r] = silu * up_v * weights_vals[0];
            /* For mode 0 (d=0): expected exactly 0. */
            if (r % 4u == 0) require(nearf(ref_mid[r], 0.0f, 1e-12f), "edge case d=0 oracle");
        }
        float got_mid[256];
        read_tensor(tmid, got_mid, mid_dim);
        for (uint32_t i = 0; i < mid_dim; i++) {
            require(nearf(got_mid[i], ref_mid[i], 1e-4f), "edge case mid value");
        }

        float ref_out[256];
        for (uint32_t r = 0; r < out_dim; r++) {
            ref_out[r] = moe_q2_K_dot_block_cpu(&down_pool_p[(uint64_t)selected_vals[0] * out_dim + r], ref_mid);
            if (r % 4u == 2) require(nearf(ref_out[r], 0.0f, 1e-12f), "edge case all-zero scales");
        }
        float got_out[256];
        read_tensor(tout_moe, got_out, out_dim);
        for (uint32_t i = 0; i < out_dim; i++) {
            require(nearf(got_out[i], ref_out[i], 1e-3f), "edge case out value");
        }

        ds4_metal_tensor_free(tout_moe);
        ds4_metal_tensor_free(tmid);
        ds4_metal_tensor_free(twts);
        ds4_metal_tensor_free(tsel);
        ds4_metal_tensor_free(tx);
        free(model);
    }

    STEP("routed_moe_one with registered model map (zero-copy path)");
    {
        const uint32_t in_dim = 256;
        const uint32_t mid_dim = 256;
        const uint32_t out_dim = 256;
        const uint32_t n_used = 2;
        const uint32_t n_total = 256;
        const float clamp = 2.5f;
        const uint64_t gate_row_bytes = sizeof(moe_block_iq2_xxs);
        const uint64_t gate_expert_bytes = (uint64_t)mid_dim * gate_row_bytes;
        const uint64_t down_row_bytes = sizeof(moe_block_q2_K);
        const uint64_t down_expert_bytes = (uint64_t)out_dim * down_row_bytes;
        const uint64_t gate_pool = (uint64_t)n_total * gate_expert_bytes;
        const uint64_t down_pool = (uint64_t)n_total * down_expert_bytes;
        const uint64_t model_size = gate_pool + gate_pool + down_pool;
        /* Page-align so hipHostRegister is happy. */
        const size_t page = 4096;
        const size_t model_alloc = (model_size + page - 1) & ~(page - 1);
        void *model_raw = NULL;
        require(posix_memalign(&model_raw, page, model_alloc) == 0,
                "moe registered model alloc");
        uint8_t *model = (uint8_t *)model_raw;
        memset(model, 0, model_alloc);

        moe_block_iq2_xxs *gate_pool_p = (moe_block_iq2_xxs *)(model + 0);
        moe_block_iq2_xxs *up_pool_p   = (moe_block_iq2_xxs *)(model + gate_pool);
        moe_block_q2_K    *down_pool_p = (moe_block_q2_K    *)(model + gate_pool + gate_pool);

        const int32_t selected_vals[2] = { 11, 250 };
        const float weights_vals[2] = { 0.55f, 0.45f };
        for (uint32_t s = 0; s < n_used; s++) {
            const int32_t e = selected_vals[s];
            for (uint32_t r = 0; r < mid_dim; r++) {
                moe_build_iq2xxs_block(&gate_pool_p[(uint64_t)e * mid_dim + r],
                                       (uint32_t)e * 2003u + r * 17u);
                moe_build_iq2xxs_block(&up_pool_p[(uint64_t)e * mid_dim + r],
                                       (uint32_t)e * 2011u + r * 19u);
            }
            for (uint32_t r = 0; r < out_dim; r++) {
                moe_build_q2_K_block(&down_pool_p[(uint64_t)e * out_dim + r],
                                     (uint32_t)e * 2017u + r * 23u);
            }
        }

        require(ds4_metal_set_model_map_range(model, model_alloc, 0, model_alloc),
                "register synthetic model map");

        float xv[256];
        for (uint32_t i = 0; i < 256; i++) xv[i] = 0.018f * (float)((int)(i % 17) - 8);

        ds4_metal_tensor *tx = ds4_metal_tensor_alloc(sizeof(xv));
        ds4_metal_tensor *tsel = ds4_metal_tensor_alloc(sizeof(selected_vals));
        ds4_metal_tensor *twts = ds4_metal_tensor_alloc(sizeof(weights_vals));
        ds4_metal_tensor *tmid = ds4_metal_tensor_alloc((uint64_t)n_used * mid_dim * sizeof(float));
        ds4_metal_tensor *tout_moe = ds4_metal_tensor_alloc((uint64_t)out_dim * sizeof(float));
        require(tx && tsel && twts && tmid && tout_moe, "moe registered tensor alloc");
        require(ds4_metal_tensor_write(tx, 0, xv, sizeof(xv)), "moe reg x write");
        require(ds4_metal_tensor_write(tsel, 0, selected_vals, sizeof(selected_vals)), "moe reg sel write");
        require(ds4_metal_tensor_write(twts, 0, weights_vals, sizeof(weights_vals)), "moe reg wts write");

        require(ds4_metal_routed_moe_one_tensor(tout_moe, NULL, NULL, tmid, NULL,
                                                 model, model_alloc,
                                                 0, gate_pool, gate_pool + gate_pool,
                                                 16u, 10u,
                                                 gate_expert_bytes, gate_row_bytes,
                                                 down_expert_bytes, down_row_bytes,
                                                 in_dim, mid_dim, out_dim,
                                                 tsel, twts, n_used, clamp, tx),
                "routed_moe_one registered");

        float ref_mid[2 * 256];
        for (uint32_t s = 0; s < n_used; s++) {
            const int32_t e = selected_vals[s];
            for (uint32_t r = 0; r < mid_dim; r++) {
                float gate_v = moe_iq2xxs_dot_block_cpu(&gate_pool_p[(uint64_t)e * mid_dim + r], xv);
                float up_v   = moe_iq2xxs_dot_block_cpu(&up_pool_p[(uint64_t)e * mid_dim + r], xv);
                if (gate_v > clamp) gate_v = clamp;
                if (up_v > clamp) up_v = clamp;
                if (up_v < -clamp) up_v = -clamp;
                const float silu = gate_v / (1.0f + expf(-gate_v));
                ref_mid[s * mid_dim + r] = silu * up_v * weights_vals[s];
            }
        }
        float got_mid[2 * 256];
        read_tensor(tmid, got_mid, n_used * mid_dim);
        for (uint32_t i = 0; i < n_used * mid_dim; i++) {
            require(nearf(got_mid[i], ref_mid[i], 1e-4f), "routed_moe_one registered mid value");
        }

        float ref_out[256] = {0};
        for (uint32_t r = 0; r < out_dim; r++) {
            float acc = 0.0f;
            for (uint32_t s = 0; s < n_used; s++) {
                const int32_t e = selected_vals[s];
                acc += moe_q2_K_dot_block_cpu(&down_pool_p[(uint64_t)e * out_dim + r],
                                              ref_mid + (uint64_t)s * mid_dim);
            }
            ref_out[r] = acc;
        }
        float got_out[256];
        read_tensor(tout_moe, got_out, out_dim);
        for (uint32_t i = 0; i < out_dim; i++) {
            require(nearf(got_out[i], ref_out[i], 1e-3f), "routed_moe_one registered out value");
        }

        ds4_metal_tensor_free(tout_moe);
        ds4_metal_tensor_free(tmid);
        ds4_metal_tensor_free(twts);
        ds4_metal_tensor_free(tsel);
        ds4_metal_tensor_free(tx);
        /* `model` is intentionally leaked: ds4_metal_cleanup() runs after this
         * step and unregisters it; freeing it here would unpin live device
         * mappings.  The test exits right after cleanup so the OS reclaims. */
    }

    STEP("hc_split_weighted_sum_norm with non-trivial params (multi-token)");
    {
        const uint32_t n_embd = 16;
        const uint32_t n_hc = 4;
        const uint32_t n_tokens = 3;
        const uint32_t mix_hc = 2u * n_hc + n_hc * n_hc; /* 24 */
        const uint32_t sinkhorn_iters = 2;
        const float eps = 1e-6f;
        const float norm_eps = 1e-5f;

        float scales[3] = { 0.75f, 1.25f, -0.5f };
        float bases[24];
        for (uint32_t i = 0; i < 24; i++) bases[i] = 0.05f * (float)i - 0.4f;
        float norm_w[16];
        for (uint32_t i = 0; i < 16; i++) norm_w[i] = 0.5f + 0.1f * (float)i;

        float mix[3 * 24];
        float residual[3 * 4 * 16];
        for (uint32_t t = 0; t < n_tokens; t++) {
            for (uint32_t i = 0; i < mix_hc; i++)
                mix[t * mix_hc + i] = 0.2f * sinf((float)(t * 31 + i + 1));
            for (uint32_t h = 0; h < n_hc; h++) {
                for (uint32_t d = 0; d < n_embd; d++) {
                    residual[(t * n_hc + h) * n_embd + d] =
                        0.1f * cosf((float)(t * 17 + h * 7 + d) * 0.3f);
                }
            }
        }

        ds4_metal_tensor *tmix2  = ds4_metal_tensor_alloc(sizeof(mix));
        ds4_metal_tensor *tres2  = ds4_metal_tensor_alloc(sizeof(residual));
        ds4_metal_tensor *tsplit2 = ds4_metal_tensor_alloc(sizeof(mix));
        ds4_metal_tensor *thc2   = ds4_metal_tensor_alloc((uint64_t)n_tokens * n_embd * sizeof(float));
        ds4_metal_tensor *tnorm2 = ds4_metal_tensor_alloc((uint64_t)n_tokens * n_embd * sizeof(float));
        require(tmix2 && tres2 && tsplit2 && thc2 && tnorm2, "hc nontrivial alloc");
        require(ds4_metal_tensor_write(tmix2, 0, mix, sizeof(mix)), "hc nontrivial mix");
        require(ds4_metal_tensor_write(tres2, 0, residual, sizeof(residual)), "hc nontrivial res");

        uint8_t blob[sizeof(scales) + sizeof(bases) + sizeof(norm_w)];
        memcpy(blob, scales, sizeof(scales));
        memcpy(blob + sizeof(scales), bases, sizeof(bases));
        memcpy(blob + sizeof(scales) + sizeof(bases), norm_w, sizeof(norm_w));
        require(ds4_metal_hc_split_weighted_sum_norm_tensor(
                    thc2, tnorm2, tsplit2, tmix2, tres2, blob, sizeof(blob),
                    0, sizeof(scales), sizeof(scales) + sizeof(bases),
                    n_embd, n_hc, sinkhorn_iters, eps, norm_eps),
                "hc nontrivial");

        float ref_split[3 * 24];
        float ref_out[3 * 16];
        float ref_norm[3 * 16];
        for (uint32_t t = 0; t < n_tokens; t++) {
            float *sp = ref_split + t * mix_hc;
            const float *mp = mix + t * mix_hc;
            float pre_arr[4];
            for (uint32_t h = 0; h < 4; h++) {
                float z = mp[h] * scales[0] + bases[h];
                pre_arr[h] = 1.0f / (1.0f + expf(-z)) + eps;
                sp[h] = pre_arr[h];
            }
            for (uint32_t h = 0; h < 4; h++) {
                float z = mp[4 + h] * scales[1] + bases[4 + h];
                sp[4 + h] = 2.0f / (1.0f + expf(-z));
            }
            float r[4][4];
            for (uint32_t i = 0; i < 4; i++) {
                float maxv = -3.4028234663852886e38f;
                for (uint32_t j = 0; j < 4; j++) {
                    r[i][j] = mp[8 + i * 4 + j] * scales[2] + bases[8 + i * 4 + j];
                    if (r[i][j] > maxv) maxv = r[i][j];
                }
                float row_sum = 0.0f;
                for (uint32_t j = 0; j < 4; j++) {
                    r[i][j] = expf(r[i][j] - maxv);
                    row_sum += r[i][j];
                }
                for (uint32_t j = 0; j < 4; j++) r[i][j] = r[i][j] / row_sum + eps;
            }
            /* Matches Metal kernel_dsv4_hc_split_weighted_sum_norm4: one col-norm
             * then (sinkhorn_iters - 1) of (row, col), with +eps in every denom. */
            for (uint32_t j = 0; j < 4; j++) {
                float col_sum = eps;
                for (uint32_t i = 0; i < 4; i++) col_sum += r[i][j];
                for (uint32_t i = 0; i < 4; i++) r[i][j] /= col_sum;
            }
            for (uint32_t it = 1; it < sinkhorn_iters; it++) {
                for (uint32_t i = 0; i < 4; i++) {
                    float row_sum = eps;
                    for (uint32_t j = 0; j < 4; j++) row_sum += r[i][j];
                    for (uint32_t j = 0; j < 4; j++) r[i][j] /= row_sum;
                }
                for (uint32_t j = 0; j < 4; j++) {
                    float col_sum = eps;
                    for (uint32_t i = 0; i < 4; i++) col_sum += r[i][j];
                    for (uint32_t i = 0; i < 4; i++) r[i][j] /= col_sum;
                }
            }
            for (uint32_t i = 0; i < 4; i++)
                for (uint32_t j = 0; j < 4; j++) sp[8 + i * 4 + j] = r[i][j];

            float ss = 0.0f;
            for (uint32_t d = 0; d < n_embd; d++) {
                float v = 0.0f;
                for (uint32_t h = 0; h < 4; h++)
                    v += residual[(t * 4 + h) * n_embd + d] * pre_arr[h];
                ref_out[t * n_embd + d] = v;
                ss += v * v;
            }
            float ns = 1.0f / sqrtf(ss / (float)n_embd + norm_eps);
            for (uint32_t d = 0; d < n_embd; d++) {
                ref_norm[t * n_embd + d] = ref_out[t * n_embd + d] * ns * norm_w[d];
            }
        }

        float got_split[3 * 24], got_out[3 * 16], got_norm[3 * 16];
        read_tensor(tsplit2, got_split, n_tokens * mix_hc);
        read_tensor(thc2, got_out, n_tokens * n_embd);
        read_tensor(tnorm2, got_norm, n_tokens * n_embd);
        for (uint32_t i = 0; i < n_tokens * mix_hc; i++)
            require(nearf(got_split[i], ref_split[i], 1e-5f), "hc nontrivial split");
        for (uint32_t i = 0; i < n_tokens * n_embd; i++)
            require(nearf(got_out[i], ref_out[i], 1e-5f), "hc nontrivial out");
        for (uint32_t i = 0; i < n_tokens * n_embd; i++)
            require(nearf(got_norm[i], ref_norm[i], 1e-5f), "hc nontrivial norm");

        ds4_metal_tensor_free(tnorm2);
        ds4_metal_tensor_free(thc2);
        ds4_metal_tensor_free(tsplit2);
        ds4_metal_tensor_free(tres2);
        ds4_metal_tensor_free(tmix2);
    }

    STEP("output_hc_weights with non-trivial bias");
    {
        const uint32_t n_hc = 4;
        const uint32_t n_tokens = 5;
        const float eps = 1e-6f;
        float pre_in[5 * 4];
        for (uint32_t i = 0; i < 20; i++) pre_in[i] = 0.5f * sinf((float)(i + 1) * 0.4f);
        float scale = 1.5f;
        float base[4] = { -0.3f, 0.4f, 0.0f, 0.7f };

        ds4_metal_tensor *tpre = ds4_metal_tensor_alloc(sizeof(pre_in));
        ds4_metal_tensor *touw = ds4_metal_tensor_alloc(sizeof(pre_in));
        require(tpre && touw, "output_hc alloc");
        require(ds4_metal_tensor_write(tpre, 0, pre_in, sizeof(pre_in)), "output_hc pre");

        uint8_t blob[sizeof(scale) + sizeof(base)];
        memcpy(blob, &scale, sizeof(scale));
        memcpy(blob + sizeof(scale), base, sizeof(base));
        require(ds4_metal_output_hc_weights_tensor(touw, tpre, blob, sizeof(blob),
                                                   0, sizeof(scale), n_hc, eps),
                "output_hc");
        float got[20];
        read_tensor(touw, got, 20);
        for (uint32_t t = 0; t < n_tokens; t++) {
            for (uint32_t h = 0; h < 4; h++) {
                float z = pre_in[t * 4 + h] * scale + base[h];
                float ref = 1.0f / (1.0f + expf(-z)) + eps;
                require(nearf(got[t * 4 + h], ref, 1e-6f), "output_hc value");
            }
        }
        ds4_metal_tensor_free(touw);
        ds4_metal_tensor_free(tpre);
    }

    STEP("hc_weighted_sum with non-trivial weights (multi-token)");
    {
        const uint32_t n_embd = 12;
        const uint32_t n_hc = 4;
        const uint32_t n_tokens = 3;
        float resid[3 * 4 * 12];
        float w[3 * 4];
        for (uint32_t t = 0; t < n_tokens; t++) {
            for (uint32_t h = 0; h < 4; h++) {
                w[t * 4 + h] = 0.1f + 0.2f * (float)(t * 4 + h);
                for (uint32_t d = 0; d < n_embd; d++) {
                    resid[(t * 4 + h) * n_embd + d] = sinf((float)(t * 11 + h * 5 + d) * 0.7f);
                }
            }
        }
        ds4_metal_tensor *tres3 = ds4_metal_tensor_alloc(sizeof(resid));
        ds4_metal_tensor *tw3 = ds4_metal_tensor_alloc(sizeof(w));
        ds4_metal_tensor *tout3 = ds4_metal_tensor_alloc((uint64_t)n_tokens * n_embd * sizeof(float));
        require(tres3 && tw3 && tout3, "hc wsum alloc");
        require(ds4_metal_tensor_write(tres3, 0, resid, sizeof(resid)), "hc wsum res");
        require(ds4_metal_tensor_write(tw3, 0, w, sizeof(w)), "hc wsum w");
        require(ds4_metal_hc_weighted_sum_tensor(tout3, tres3, tw3, n_embd, n_hc),
                "hc weighted_sum");
        float got[3 * 12];
        read_tensor(tout3, got, n_tokens * n_embd);
        for (uint32_t t = 0; t < n_tokens; t++) {
            for (uint32_t d = 0; d < n_embd; d++) {
                float ref = 0.0f;
                for (uint32_t h = 0; h < 4; h++)
                    ref += resid[(t * 4 + h) * n_embd + d] * w[t * 4 + h];
                require(nearf(got[t * n_embd + d], ref, 1e-5f), "hc weighted_sum value");
            }
        }
        ds4_metal_tensor_free(tout3);
        ds4_metal_tensor_free(tw3);
        ds4_metal_tensor_free(tres3);
    }

    STEP("hc_expand_split (block + comb*residual via split layout)");
    {
        const uint32_t n_embd = 8;
        const uint32_t n_hc = 4;
        const uint32_t n_tokens = 2;
        const uint32_t mix_hc = 2u * n_hc + n_hc * n_hc; /* 24 */
        float block_in[2 * 8];
        float resid[2 * 4 * 8];
        float split_in[2 * 24];
        for (uint32_t t = 0; t < n_tokens; t++) {
            for (uint32_t d = 0; d < n_embd; d++)
                block_in[t * n_embd + d] = 0.5f * sinf((float)(t * 13 + d) * 0.5f);
            for (uint32_t h = 0; h < n_hc; h++)
                for (uint32_t d = 0; d < n_embd; d++)
                    resid[(t * n_hc + h) * n_embd + d] = 0.3f * cosf((float)(t * 7 + h * 3 + d) * 0.4f);
            for (uint32_t i = 0; i < mix_hc; i++)
                split_in[t * mix_hc + i] = 0.1f * (float)((int)i - 11) + 0.05f * (float)t;
        }
        ds4_metal_tensor *tblk = ds4_metal_tensor_alloc(sizeof(block_in));
        ds4_metal_tensor *tres4 = ds4_metal_tensor_alloc(sizeof(resid));
        ds4_metal_tensor *tspl = ds4_metal_tensor_alloc(sizeof(split_in));
        ds4_metal_tensor *tout4 = ds4_metal_tensor_alloc(sizeof(resid));
        require(tblk && tres4 && tspl && tout4, "hc expand_split alloc");
        require(ds4_metal_tensor_write(tblk, 0, block_in, sizeof(block_in)), "");
        require(ds4_metal_tensor_write(tres4, 0, resid, sizeof(resid)), "");
        require(ds4_metal_tensor_write(tspl, 0, split_in, sizeof(split_in)), "");
        require(ds4_metal_hc_expand_split_tensor(tout4, tblk, tres4, tspl, n_embd, n_hc),
                "hc expand_split");
        float got[2 * 4 * 8];
        read_tensor(tout4, got, n_tokens * n_hc * n_embd);
        for (uint32_t t = 0; t < n_tokens; t++) {
            const float *post = split_in + t * mix_hc + n_hc;
            const float *comb = split_in + t * mix_hc + 2u * n_hc;
            for (uint32_t dst_hc = 0; dst_hc < 4; dst_hc++) {
                for (uint32_t d = 0; d < n_embd; d++) {
                    float ref = block_in[t * n_embd + d] * post[dst_hc];
                    for (uint32_t src_hc = 0; src_hc < 4; src_hc++)
                        ref += comb[dst_hc * 4 + src_hc] * resid[(t * n_hc + src_hc) * n_embd + d];
                    require(nearf(got[(t * n_hc + dst_hc) * n_embd + d], ref, 1e-5f),
                            "hc expand_split value");
                }
            }
        }
        ds4_metal_tensor_free(tout4);
        ds4_metal_tensor_free(tspl);
        ds4_metal_tensor_free(tres4);
        ds4_metal_tensor_free(tblk);
    }

    STEP("hc_expand_add_split (block_out + block_add)");
    {
        const uint32_t n_embd = 6;
        const uint32_t n_hc = 4;
        const uint32_t n_tokens = 2;
        const uint32_t mix_hc = 2u * n_hc + n_hc * n_hc;
        float bo[2 * 6], ba[2 * 6];
        float resid[2 * 4 * 6];
        float split_in[2 * 24];
        for (uint32_t t = 0; t < n_tokens; t++) {
            for (uint32_t d = 0; d < n_embd; d++) {
                bo[t * n_embd + d] = 0.4f * cosf((float)(t * 5 + d) * 0.6f);
                ba[t * n_embd + d] = 0.3f * sinf((float)(t * 9 + d) * 0.5f);
            }
            for (uint32_t h = 0; h < n_hc; h++)
                for (uint32_t d = 0; d < n_embd; d++)
                    resid[(t * n_hc + h) * n_embd + d] = 0.2f * sinf((float)(t * 3 + h * 11 + d) * 0.7f);
            for (uint32_t i = 0; i < mix_hc; i++)
                split_in[t * mix_hc + i] = 0.05f * (float)((int)i - 12) - 0.02f * (float)t;
        }
        ds4_metal_tensor *tbo = ds4_metal_tensor_alloc(sizeof(bo));
        ds4_metal_tensor *tba = ds4_metal_tensor_alloc(sizeof(ba));
        ds4_metal_tensor *tres5 = ds4_metal_tensor_alloc(sizeof(resid));
        ds4_metal_tensor *tspl5 = ds4_metal_tensor_alloc(sizeof(split_in));
        ds4_metal_tensor *tout5 = ds4_metal_tensor_alloc(sizeof(resid));
        require(tbo && tba && tres5 && tspl5 && tout5, "hc expand_add alloc");
        require(ds4_metal_tensor_write(tbo, 0, bo, sizeof(bo)), "");
        require(ds4_metal_tensor_write(tba, 0, ba, sizeof(ba)), "");
        require(ds4_metal_tensor_write(tres5, 0, resid, sizeof(resid)), "");
        require(ds4_metal_tensor_write(tspl5, 0, split_in, sizeof(split_in)), "");
        require(ds4_metal_hc_expand_add_split_tensor(tout5, tbo, tba, tres5, tspl5, n_embd, n_hc),
                "hc expand_add_split");
        float got[2 * 4 * 6];
        read_tensor(tout5, got, n_tokens * n_hc * n_embd);
        for (uint32_t t = 0; t < n_tokens; t++) {
            const float *post = split_in + t * mix_hc + n_hc;
            const float *comb = split_in + t * mix_hc + 2u * n_hc;
            for (uint32_t dst_hc = 0; dst_hc < 4; dst_hc++) {
                for (uint32_t d = 0; d < n_embd; d++) {
                    float bv = bo[t * n_embd + d] + ba[t * n_embd + d];
                    float ref = bv * post[dst_hc];
                    for (uint32_t src_hc = 0; src_hc < 4; src_hc++)
                        ref += comb[dst_hc * 4 + src_hc] * resid[(t * n_hc + src_hc) * n_embd + d];
                    require(nearf(got[(t * n_hc + dst_hc) * n_embd + d], ref, 1e-5f),
                            "hc expand_add_split value");
                }
            }
        }
        ds4_metal_tensor_free(tout5);
        ds4_metal_tensor_free(tspl5);
        ds4_metal_tensor_free(tres5);
        ds4_metal_tensor_free(tba);
        ds4_metal_tensor_free(tbo);
    }

    STEP("matmul_f16_pair (two independent F16 matmuls)");
    {
        const uint32_t in_dim = 8;
        const uint32_t out_dim = 5;
        const uint32_t n_tok = 3;
        uint16_t Wa[5 * 8], Wb[5 * 8];
        float xv[3 * 8];
        for (uint32_t r = 0; r < out_dim; r++) {
            for (uint32_t c = 0; c < in_dim; c++) {
                Wa[r * in_dim + c] = f32_to_f16(0.1f * (float)(r + 1) + 0.05f * (float)c);
                Wb[r * in_dim + c] = f32_to_f16(-0.2f * (float)(r + 1) + 0.03f * (float)c);
            }
        }
        for (uint32_t i = 0; i < n_tok * in_dim; i++) xv[i] = sinf((float)(i + 1) * 0.4f);
        uint8_t blob[2 * sizeof(Wa)];
        memcpy(blob, Wa, sizeof(Wa));
        memcpy(blob + sizeof(Wa), Wb, sizeof(Wb));
        ds4_metal_tensor *tx_p = ds4_metal_tensor_alloc(sizeof(xv));
        ds4_metal_tensor *toa = ds4_metal_tensor_alloc((uint64_t)n_tok * out_dim * sizeof(float));
        ds4_metal_tensor *tob = ds4_metal_tensor_alloc((uint64_t)n_tok * out_dim * sizeof(float));
        require(tx_p && toa && tob, "matmul_f16_pair alloc");
        require(ds4_metal_tensor_write(tx_p, 0, xv, sizeof(xv)), "matmul_f16_pair x");
        require(ds4_metal_matmul_f16_pair_tensor(toa, tob, blob, sizeof(blob),
                                                  0, sizeof(Wa), in_dim, out_dim, tx_p, n_tok),
                "matmul_f16_pair");
        float ga[3 * 5], gb[3 * 5];
        read_tensor(toa, ga, n_tok * out_dim);
        read_tensor(tob, gb, n_tok * out_dim);
        for (uint32_t t = 0; t < n_tok; t++) {
            for (uint32_t r = 0; r < out_dim; r++) {
                float refa = 0.0f, refb = 0.0f;
                for (uint32_t c = 0; c < in_dim; c++) {
                    refa += f16_to_f32(Wa[r * in_dim + c]) * xv[t * in_dim + c];
                    refb += f16_to_f32(Wb[r * in_dim + c]) * xv[t * in_dim + c];
                }
                require(nearf(ga[t * out_dim + r], refa, 1e-3f), "f16_pair a value");
                require(nearf(gb[t * out_dim + r], refb, 1e-3f), "f16_pair b value");
            }
        }
        ds4_metal_tensor_free(tob);
        ds4_metal_tensor_free(toa);
        ds4_metal_tensor_free(tx_p);
    }

    /* Decode-shape fused matmul_f16_pair: in_dim multiple of 256, out_dim
     * multiple of 8 + >=256 hits the NROWS=8 fused path that computes both
     * out_a and out_b from one shared LDS-x load. */
    STEP("matmul_f16_pair fused decode (in=256, out=256, n_tok=1)");
    {
        const uint32_t IN = 256, OUT = 256;
        uint16_t *Wa = (uint16_t *)malloc((size_t)IN * OUT * sizeof(uint16_t));
        uint16_t *Wb = (uint16_t *)malloc((size_t)IN * OUT * sizeof(uint16_t));
        float    *xv  = (float    *)malloc((size_t)IN * sizeof(float));
        float    *refa = (float    *)malloc((size_t)OUT * sizeof(float));
        float    *refb = (float    *)malloc((size_t)OUT * sizeof(float));
        float    *gota = (float    *)malloc((size_t)OUT * sizeof(float));
        float    *gotb = (float    *)malloc((size_t)OUT * sizeof(float));
        require(Wa && Wb && xv && refa && refb && gota && gotb, "f16_pair fused alloc");
        for (uint32_t i = 0; i < IN; i++) xv[i] = (float)((int)(i % 17) - 8) * 0.125f;
        for (uint32_t r = 0; r < OUT; r++) {
            for (uint32_t i = 0; i < IN; i++) {
                Wa[r * IN + i] = f32_to_f16((float)((int)((r + i) % 23) - 11) * 0.0625f);
                Wb[r * IN + i] = f32_to_f16((float)((int)((r * 5 + i) % 19) - 9) * 0.0625f);
            }
        }
        for (uint32_t r = 0; r < OUT; r++) {
            float sa = 0.0f, sb = 0.0f;
            for (uint32_t i = 0; i < IN; i++) {
                sa += f16_to_f32(Wa[r * IN + i]) * xv[i];
                sb += f16_to_f32(Wb[r * IN + i]) * xv[i];
            }
            refa[r] = sa;
            refb[r] = sb;
        }
        const size_t wsize = (size_t)IN * OUT * sizeof(uint16_t);
        uint8_t *blob = (uint8_t *)malloc(2 * wsize);
        require(blob != NULL, "f16_pair fused blob alloc");
        memcpy(blob,         Wa, wsize);
        memcpy(blob + wsize, Wb, wsize);
        ds4_metal_tensor *txw = ds4_metal_tensor_alloc(IN * sizeof(float));
        ds4_metal_tensor *toa = ds4_metal_tensor_alloc(OUT * sizeof(float));
        ds4_metal_tensor *tob = ds4_metal_tensor_alloc(OUT * sizeof(float));
        require(txw && toa && tob, "f16_pair fused tensor alloc");
        require(ds4_metal_tensor_write(txw, 0, xv, IN * sizeof(float)),
                "f16_pair fused x write");
        require(ds4_metal_matmul_f16_pair_tensor(toa, tob, blob, 2 * wsize,
                                                  0, wsize, IN, OUT, txw, 1),
                "matmul_f16_pair fused 8x");
        require(ds4_metal_tensor_read(toa, 0, gota, OUT * sizeof(float)),
                "f16_pair fused a read");
        require(ds4_metal_tensor_read(tob, 0, gotb, OUT * sizeof(float)),
                "f16_pair fused b read");
        for (uint32_t r = 0; r < OUT; r++) {
            require(nearf(gota[r], refa[r], 1e-2f), "f16_pair fused a value");
            require(nearf(gotb[r], refb[r], 1e-2f), "f16_pair fused b value");
        }
        ds4_metal_tensor_free(tob);
        ds4_metal_tensor_free(toa);
        ds4_metal_tensor_free(txw);
        free(blob);
        free(gotb); free(gota); free(refb); free(refa);
        free(xv); free(Wb); free(Wa);
    }

    /* Same fused path but out_dim=20 (multiple of 4, not 8) -> NROWS=4. */
    STEP("matmul_f16_pair fused decode (in=256, out=20, n_tok=1)");
    {
        const uint32_t IN = 256, OUT = 20;
        uint16_t *Wa = (uint16_t *)malloc((size_t)IN * OUT * sizeof(uint16_t));
        uint16_t *Wb = (uint16_t *)malloc((size_t)IN * OUT * sizeof(uint16_t));
        float    *xv  = (float    *)malloc((size_t)IN * sizeof(float));
        float    *refa = (float    *)malloc((size_t)OUT * sizeof(float));
        float    *refb = (float    *)malloc((size_t)OUT * sizeof(float));
        float    *gota = (float    *)malloc((size_t)OUT * sizeof(float));
        float    *gotb = (float    *)malloc((size_t)OUT * sizeof(float));
        require(Wa && Wb && xv && refa && refb && gota && gotb,
                "f16_pair fused 4x alloc");
        for (uint32_t i = 0; i < IN; i++) xv[i] = (float)((int)(i % 13) - 6) * 0.125f;
        for (uint32_t r = 0; r < OUT; r++) {
            for (uint32_t i = 0; i < IN; i++) {
                Wa[r * IN + i] = f32_to_f16((float)((int)((r * 3 + i) % 19) - 9) * 0.0625f);
                Wb[r * IN + i] = f32_to_f16((float)((int)((r * 7 + i) % 17) - 8) * 0.0625f);
            }
        }
        for (uint32_t r = 0; r < OUT; r++) {
            float sa = 0.0f, sb = 0.0f;
            for (uint32_t i = 0; i < IN; i++) {
                sa += f16_to_f32(Wa[r * IN + i]) * xv[i];
                sb += f16_to_f32(Wb[r * IN + i]) * xv[i];
            }
            refa[r] = sa;
            refb[r] = sb;
        }
        const size_t wsize = (size_t)IN * OUT * sizeof(uint16_t);
        uint8_t *blob = (uint8_t *)malloc(2 * wsize);
        require(blob != NULL, "f16_pair fused 4x blob alloc");
        memcpy(blob,         Wa, wsize);
        memcpy(blob + wsize, Wb, wsize);
        ds4_metal_tensor *txw = ds4_metal_tensor_alloc(IN * sizeof(float));
        ds4_metal_tensor *toa = ds4_metal_tensor_alloc(OUT * sizeof(float));
        ds4_metal_tensor *tob = ds4_metal_tensor_alloc(OUT * sizeof(float));
        require(txw && toa && tob, "f16_pair fused 4x tensor alloc");
        require(ds4_metal_tensor_write(txw, 0, xv, IN * sizeof(float)),
                "f16_pair fused 4x x write");
        require(ds4_metal_matmul_f16_pair_tensor(toa, tob, blob, 2 * wsize,
                                                  0, wsize, IN, OUT, txw, 1),
                "matmul_f16_pair fused 4x");
        require(ds4_metal_tensor_read(toa, 0, gota, OUT * sizeof(float)),
                "f16_pair fused 4x a read");
        require(ds4_metal_tensor_read(tob, 0, gotb, OUT * sizeof(float)),
                "f16_pair fused 4x b read");
        for (uint32_t r = 0; r < OUT; r++) {
            require(nearf(gota[r], refa[r], 1e-2f), "f16_pair fused 4x a value");
            require(nearf(gotb[r], refb[r], 1e-2f), "f16_pair fused 4x b value");
        }
        ds4_metal_tensor_free(tob);
        ds4_metal_tensor_free(toa);
        ds4_metal_tensor_free(txw);
        free(blob);
        free(gotb); free(gota); free(refb); free(refa);
        free(xv); free(Wb); free(Wa);
    }

    /* Fused (rms_norm + matmul_f16) at the canonical HC-mixing shape
     * (in=hc_dim, out=mix_hc).  Uses small dims for fast unit testing;
     * the actual fused kernel covers any in_dim % 32 == 0 with
     * out_dim divisible by 4 or 8. */
    STEP("rms_matmul_f16_fused (decode HC-mixing shape, in=256, out=8)");
    {
        const uint32_t IN = 256, OUT = 8;
        uint16_t *W   = (uint16_t *)malloc((size_t)IN * OUT * sizeof(uint16_t));
        float    *xv  = (float    *)malloc((size_t)IN * sizeof(float));
        float    *ref = (float    *)malloc((size_t)OUT * sizeof(float));
        float    *got = (float    *)malloc((size_t)OUT * sizeof(float));
        require(W && xv && ref && got, "rms_matmul_f16 alloc");
        for (uint32_t i = 0; i < IN; i++) xv[i] = (float)((int)(i % 11) - 5) * 0.25f;
        for (uint32_t r = 0; r < OUT; r++) {
            for (uint32_t i = 0; i < IN; i++) {
                W[r * IN + i] = f32_to_f16((float)((int)((r * 5 + i) % 17) - 8) * 0.125f);
            }
        }
        const float eps = 1.0e-6f;
        float sq = 0.0f;
        for (uint32_t i = 0; i < IN; i++) sq += xv[i] * xv[i];
        const float rms = 1.0f / sqrtf(sq / (float)IN + eps);
        for (uint32_t r = 0; r < OUT; r++) {
            float s = 0.0f;
            for (uint32_t i = 0; i < IN; i++) {
                s += f16_to_f32(W[r * IN + i]) * (xv[i] * rms);
            }
            ref[r] = s;
        }
        const size_t wsize = (size_t)IN * OUT * sizeof(uint16_t);
        ds4_metal_tensor *txw = ds4_metal_tensor_alloc(IN * sizeof(float));
        ds4_metal_tensor *tow = ds4_metal_tensor_alloc(OUT * sizeof(float));
        require(txw && tow, "rms_matmul_f16 tensor alloc");
        require(ds4_metal_tensor_write(txw, 0, xv, IN * sizeof(float)),
                "rms_matmul_f16 x write");
        require(ds4_metal_rms_matmul_f16_fused_tensor(tow, W, wsize, 0,
                                                       IN, OUT, txw, eps),
                "rms_matmul_f16_fused");
        require(ds4_metal_tensor_read(tow, 0, got, OUT * sizeof(float)),
                "rms_matmul_f16 read");
        for (uint32_t r = 0; r < OUT; r++) {
            require(nearf(got[r], ref[r], 1e-2f), "rms_matmul_f16 value");
        }
        ds4_metal_tensor_free(tow);
        ds4_metal_tensor_free(txw);
        free(got); free(ref); free(xv); free(W);
    }

    /* Q8_0 pair-matmul (qr + kv_raw shape).  In=256 lets us use the
     * wave-per-row kernel without WMMA gating; out_dim_a=256 + out_dim_b=8
     * exercises the asymmetric out_dim path (need NROWS=8 since both >= 256
     * not required for NROWS=4, but small out_dim_b=8 still hits 4-wave). */
    STEP("matmul_q8_0_pair (asymmetric out_dim, NROWS=4)");
    {
        const uint32_t IN = 256, OUT_A = 64, OUT_B = 12;
        const uint64_t row_bytes = ((uint64_t)IN / 32u) * 34u;
        uint8_t *Wa = (uint8_t *)malloc((size_t)row_bytes * OUT_A);
        uint8_t *Wb = (uint8_t *)malloc((size_t)row_bytes * OUT_B);
        float   *xv   = (float *)malloc((size_t)IN * sizeof(float));
        float   *refa = (float *)malloc((size_t)OUT_A * sizeof(float));
        float   *refb = (float *)malloc((size_t)OUT_B * sizeof(float));
        float   *gota = (float *)malloc((size_t)OUT_A * sizeof(float));
        float   *gotb = (float *)malloc((size_t)OUT_B * sizeof(float));
        require(Wa && Wb && xv && refa && refb && gota && gotb, "q8_0 pair alloc");
        for (uint32_t i = 0; i < IN; i++) xv[i] = (float)((int)(i % 17) - 8) * 0.125f;
        for (uint32_t r = 0; r < OUT_A; r++) {
            uint16_t one = f32_to_f16(0.125f);
            for (uint32_t b = 0; b < IN / 32u; b++) {
                uint8_t *blk = Wa + (uint64_t)r * row_bytes + (uint64_t)b * 34u;
                blk[0] = (uint8_t)(one & 0xffu); blk[1] = (uint8_t)(one >> 8);
                for (uint32_t i = 0; i < 32; i++) {
                    blk[2 + i] = (uint8_t)(int8_t)(((int)((r * 3 + b * 32 + i) % 23) - 11));
                }
            }
        }
        for (uint32_t r = 0; r < OUT_B; r++) {
            uint16_t one = f32_to_f16(0.25f);
            for (uint32_t b = 0; b < IN / 32u; b++) {
                uint8_t *blk = Wb + (uint64_t)r * row_bytes + (uint64_t)b * 34u;
                blk[0] = (uint8_t)(one & 0xffu); blk[1] = (uint8_t)(one >> 8);
                for (uint32_t i = 0; i < 32; i++) {
                    blk[2 + i] = (uint8_t)(int8_t)(((int)((r * 7 + b * 32 + i) % 19) - 9));
                }
            }
        }
        for (uint32_t r = 0; r < OUT_A; r++) {
            float s = 0.0f;
            for (uint32_t b = 0; b < IN / 32u; b++) {
                const uint8_t *blk = Wa + (uint64_t)r * row_bytes + (uint64_t)b * 34u;
                float d = f16_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
                for (uint32_t i = 0; i < 32; i++) s += d * (float)(int8_t)blk[2 + i] * xv[b * 32 + i];
            }
            refa[r] = s;
        }
        for (uint32_t r = 0; r < OUT_B; r++) {
            float s = 0.0f;
            for (uint32_t b = 0; b < IN / 32u; b++) {
                const uint8_t *blk = Wb + (uint64_t)r * row_bytes + (uint64_t)b * 34u;
                float d = f16_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
                for (uint32_t i = 0; i < 32; i++) s += d * (float)(int8_t)blk[2 + i] * xv[b * 32 + i];
            }
            refb[r] = s;
        }
        const size_t wsize_a = (size_t)row_bytes * OUT_A;
        const size_t wsize_b = (size_t)row_bytes * OUT_B;
        uint8_t *blob = (uint8_t *)malloc(wsize_a + wsize_b);
        require(blob != NULL, "q8_0 pair blob alloc");
        memcpy(blob,           Wa, wsize_a);
        memcpy(blob + wsize_a, Wb, wsize_b);
        ds4_metal_tensor *txw = ds4_metal_tensor_alloc(IN * sizeof(float));
        ds4_metal_tensor *toa = ds4_metal_tensor_alloc(OUT_A * sizeof(float));
        ds4_metal_tensor *tob = ds4_metal_tensor_alloc(OUT_B * sizeof(float));
        require(txw && toa && tob, "q8_0 pair tensor alloc");
        require(ds4_metal_tensor_write(txw, 0, xv, IN * sizeof(float)),
                "q8_0 pair x write");
        require(ds4_metal_matmul_q8_0_pair_tensor(toa, tob, blob, wsize_a + wsize_b,
                                                  0, wsize_a, IN, OUT_A, OUT_B, txw),
                "matmul_q8_0_pair");
        require(ds4_metal_tensor_read(toa, 0, gota, OUT_A * sizeof(float)),
                "q8_0 pair a read");
        require(ds4_metal_tensor_read(tob, 0, gotb, OUT_B * sizeof(float)),
                "q8_0 pair b read");
        for (uint32_t r = 0; r < OUT_A; r++) {
            require(nearf(gota[r], refa[r], 1e-2f), "q8_0 pair a value");
        }
        for (uint32_t r = 0; r < OUT_B; r++) {
            require(nearf(gotb[r], refb[r], 1e-2f), "q8_0 pair b value");
        }
        ds4_metal_tensor_free(tob);
        ds4_metal_tensor_free(toa);
        ds4_metal_tensor_free(txw);
        free(blob);
        free(gotb); free(gota); free(refb); free(refa);
        free(xv); free(Wb); free(Wa);
    }

    /* Q8_0 pair-matmul at production qr+kv_raw scale: in=7168, out=1536 + 576.
     * out_dim_a=1536 satisfies NROWS=8 (both >=256, multiple of 8); the
     * out_b=576 path bounds-checks correctly when grid runs past it. */
    STEP("matmul_q8_0_pair (production qr+kv_raw shape, NROWS=8)");
    {
        const uint32_t IN = 7168, OUT_A = 1536, OUT_B = 576;
        const uint64_t row_bytes = ((uint64_t)IN / 32u) * 34u;
        uint8_t *Wa = (uint8_t *)malloc((size_t)row_bytes * OUT_A);
        uint8_t *Wb = (uint8_t *)malloc((size_t)row_bytes * OUT_B);
        float   *xv   = (float *)malloc((size_t)IN * sizeof(float));
        float   *refa = (float *)malloc((size_t)OUT_A * sizeof(float));
        float   *refb = (float *)malloc((size_t)OUT_B * sizeof(float));
        float   *gota = (float *)malloc((size_t)OUT_A * sizeof(float));
        float   *gotb = (float *)malloc((size_t)OUT_B * sizeof(float));
        require(Wa && Wb && xv && refa && refb && gota && gotb, "q8_0 pair prod alloc");
        for (uint32_t i = 0; i < IN; i++) xv[i] = (float)((int)(i % 31) - 15) * 0.0625f;
        for (uint32_t r = 0; r < OUT_A; r++) {
            uint16_t one = f32_to_f16(0.0625f);
            for (uint32_t b = 0; b < IN / 32u; b++) {
                uint8_t *blk = Wa + (uint64_t)r * row_bytes + (uint64_t)b * 34u;
                blk[0] = (uint8_t)(one & 0xffu); blk[1] = (uint8_t)(one >> 8);
                for (uint32_t i = 0; i < 32; i++) {
                    blk[2 + i] = (uint8_t)(int8_t)(((int)((r + b * 5 + i) % 11) - 5));
                }
            }
        }
        for (uint32_t r = 0; r < OUT_B; r++) {
            uint16_t one = f32_to_f16(0.125f);
            for (uint32_t b = 0; b < IN / 32u; b++) {
                uint8_t *blk = Wb + (uint64_t)r * row_bytes + (uint64_t)b * 34u;
                blk[0] = (uint8_t)(one & 0xffu); blk[1] = (uint8_t)(one >> 8);
                for (uint32_t i = 0; i < 32; i++) {
                    blk[2 + i] = (uint8_t)(int8_t)(((int)((r * 2 + b + i) % 13) - 6));
                }
            }
        }
        for (uint32_t r = 0; r < OUT_A; r++) {
            float s = 0.0f;
            for (uint32_t b = 0; b < IN / 32u; b++) {
                const uint8_t *blk = Wa + (uint64_t)r * row_bytes + (uint64_t)b * 34u;
                float d = f16_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
                for (uint32_t i = 0; i < 32; i++) s += d * (float)(int8_t)blk[2 + i] * xv[b * 32 + i];
            }
            refa[r] = s;
        }
        for (uint32_t r = 0; r < OUT_B; r++) {
            float s = 0.0f;
            for (uint32_t b = 0; b < IN / 32u; b++) {
                const uint8_t *blk = Wb + (uint64_t)r * row_bytes + (uint64_t)b * 34u;
                float d = f16_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
                for (uint32_t i = 0; i < 32; i++) s += d * (float)(int8_t)blk[2 + i] * xv[b * 32 + i];
            }
            refb[r] = s;
        }
        const size_t wsize_a = (size_t)row_bytes * OUT_A;
        const size_t wsize_b = (size_t)row_bytes * OUT_B;
        uint8_t *blob = (uint8_t *)malloc(wsize_a + wsize_b);
        require(blob != NULL, "q8_0 pair prod blob alloc");
        memcpy(blob,           Wa, wsize_a);
        memcpy(blob + wsize_a, Wb, wsize_b);
        ds4_metal_tensor *txw = ds4_metal_tensor_alloc(IN * sizeof(float));
        ds4_metal_tensor *toa = ds4_metal_tensor_alloc(OUT_A * sizeof(float));
        ds4_metal_tensor *tob = ds4_metal_tensor_alloc(OUT_B * sizeof(float));
        require(txw && toa && tob, "q8_0 pair prod tensor alloc");
        require(ds4_metal_tensor_write(txw, 0, xv, IN * sizeof(float)),
                "q8_0 pair prod x write");
        require(ds4_metal_matmul_q8_0_pair_tensor(toa, tob, blob, wsize_a + wsize_b,
                                                  0, wsize_a, IN, OUT_A, OUT_B, txw),
                "matmul_q8_0_pair prod");
        require(ds4_metal_tensor_read(toa, 0, gota, OUT_A * sizeof(float)),
                "q8_0 pair prod a read");
        require(ds4_metal_tensor_read(tob, 0, gotb, OUT_B * sizeof(float)),
                "q8_0 pair prod b read");
        for (uint32_t r = 0; r < OUT_A; r++) {
            require(nearf(gota[r], refa[r], 5e-2f), "q8_0 pair prod a value");
        }
        for (uint32_t r = 0; r < OUT_B; r++) {
            require(nearf(gotb[r], refb[r], 5e-2f), "q8_0 pair prod b value");
        }
        ds4_metal_tensor_free(tob);
        ds4_metal_tensor_free(toa);
        ds4_metal_tensor_free(txw);
        free(blob);
        free(gotb); free(gota); free(refb); free(refa);
        free(xv); free(Wb); free(Wa);
    }

    STEP("kv_fp8_store_raw (FP8 nope + identity rot, ring offset)");
    {
        const uint32_t head_dim = 16;
        const uint32_t n_rot = 4;
        const uint32_t n_nope = head_dim - n_rot;
        const uint32_t raw_cap = 4;
        const uint32_t row = 2;
        float kv_in[16];
        for (uint32_t i = 0; i < head_dim; i++) kv_in[i] = 0.6f * sinf((float)(i + 1) * 0.5f);
        float kv_orig[16];
        memcpy(kv_orig, kv_in, sizeof(kv_in));

        ds4_metal_tensor *tkv = ds4_metal_tensor_alloc(sizeof(kv_in));
        ds4_metal_tensor *traw = ds4_metal_tensor_alloc((uint64_t)raw_cap * head_dim * sizeof(float));
        require(tkv && traw, "kv_fp8_store alloc");
        float zeros[4 * 16] = {0};
        require(ds4_metal_tensor_write(traw, 0, zeros, sizeof(zeros)), "kv_fp8 raw zero");
        require(ds4_metal_tensor_write(tkv, 0, kv_in, sizeof(kv_in)), "kv_fp8 kv");

        require(ds4_metal_kv_fp8_store_raw_tensor(tkv, traw, raw_cap, row, head_dim, n_rot),
                "kv_fp8_store_raw");

        float got_kv[16], got_raw[4 * 16];
        read_tensor(tkv, got_kv, head_dim);
        read_tensor(traw, got_raw, raw_cap * head_dim);
        /* Reconstruct expected FP8 round-trip on n_nope. */
        float amax = 0.0f;
        for (uint32_t i = 0; i < n_nope; i++) if (fabsf(kv_orig[i]) > amax) amax = fabsf(kv_orig[i]);
        if (amax < 1.0e-4f) amax = 1.0e-4f;
        float scale = exp2f(ceilf(log2f(amax / 448.0f)));
        for (uint32_t i = 0; i < n_nope; i++) {
            float ref = e4m3fn_dequant(fminf(fmaxf(kv_orig[i] / scale, -448.0f), 448.0f)) * scale;
            require(nearf(got_kv[i], ref, 1e-6f), "kv_fp8 kv nope");
            /* raw stores f16(ref); for small E4M3 magnitudes this round-trip is lossless. */
            float raw_ref = f16_to_f32(f32_to_f16(ref));
            require(nearf(got_raw[row * head_dim + i], raw_ref, 1e-6f), "kv_fp8 raw nope");
        }
        for (uint32_t i = n_nope; i < head_dim; i++) {
            /* kv is unchanged for the RoPE region; raw stores f16(kv). */
            require(nearf(got_kv[i], kv_orig[i], 1e-6f), "kv_fp8 kv rot identity");
            float raw_ref = f16_to_f32(f32_to_f16(kv_orig[i]));
            require(nearf(got_raw[row * head_dim + i], raw_ref, 1e-6f), "kv_fp8 raw rot identity");
        }
        /* Other ring rows must remain zero. */
        for (uint32_t r = 0; r < raw_cap; r++) {
            if (r == row) continue;
            for (uint32_t i = 0; i < head_dim; i++) {
                require(nearf(got_raw[r * head_dim + i], 0.0f, 1e-6f), "kv_fp8 other rows untouched");
            }
        }
        ds4_metal_tensor_free(traw);
        ds4_metal_tensor_free(tkv);
    }

    STEP("attention_output_low_q8 (per-group Q8 matvec)");
    {
        const uint64_t group_dim = 32;          /* one Q8 block per row */
        const uint64_t rank = 4;
        const uint32_t n_groups = 3;
        const uint64_t row_bytes = ((group_dim + 31u) / 32u) * 34u; /* 34 */
        const uint64_t group_w_bytes = row_bytes * rank;
        const uint64_t total_w = (uint64_t)n_groups * group_w_bytes;
        uint8_t *W = (uint8_t *)calloc(1, (size_t)total_w);
        require(W != NULL, "out_low_q8 W alloc");

        struct gen_ctx { uint32_t group; } ctx_dummy = {0}; (void)ctx_dummy;
        for (uint32_t g = 0; g < n_groups; g++) {
            /* deterministic but distinct values per (g, r, c) */
            for (uint32_t r = 0; r < rank; r++) {
                /* build one row using a closure-ish manual pass */
                const uint64_t blocks_per_row = (group_dim + 31u) / 32u;
                const uint64_t this_row_bytes = blocks_per_row * 34u;
                uint8_t *dst = W + (uint64_t)g * group_w_bytes + (uint64_t)r * this_row_bytes;
                float vals[32];
                float amax = 0.0f;
                for (uint32_t c = 0; c < group_dim; c++) {
                    vals[c] = 0.1f * sinf((float)(g * 17 + r * 5 + c) * 0.3f);
                    if (fabsf(vals[c]) > amax) amax = fabsf(vals[c]);
                }
                float scale = amax / 127.0f;
                if (scale == 0.0f) scale = 1.0f;
                uint16_t hf = f32_to_f16(scale);
                dst[0] = (uint8_t)(hf & 0xffu);
                dst[1] = (uint8_t)(hf >> 8);
                for (uint32_t c = 0; c < 32; c++) {
                    int q = (int)roundf(vals[c] / scale);
                    if (q < -127) q = -127;
                    if (q > 127) q = 127;
                    dst[2 + c] = (uint8_t)(int8_t)q;
                }
            }
        }

        float heads[3 * 32];
        for (uint32_t i = 0; i < n_groups * group_dim; i++) heads[i] = cosf((float)(i + 1) * 0.4f);
        ds4_metal_tensor *theads = ds4_metal_tensor_alloc(sizeof(heads));
        ds4_metal_tensor *tlow = ds4_metal_tensor_alloc((uint64_t)n_groups * rank * sizeof(float));
        require(theads && tlow, "out_low_q8 tensor alloc");
        require(ds4_metal_tensor_write(theads, 0, heads, sizeof(heads)), "out_low_q8 heads");
        require(ds4_metal_attention_output_low_q8_tensor(tlow, W, total_w, 0,
                                                          group_dim, rank, n_groups, theads),
                "attention_output_low_q8");
        float got[3 * 4];
        read_tensor(tlow, got, n_groups * rank);
        for (uint32_t g = 0; g < n_groups; g++) {
            const uint8_t *Wg = W + (uint64_t)g * group_w_bytes;
            const float *xg = heads + (uint64_t)g * group_dim;
            for (uint32_t r = 0; r < rank; r++) {
                float ref = q8_0_row_dot(Wg, group_dim, r, xg);
                require(nearf(got[g * rank + r], ref, 1e-3f), "attn out_low_q8 value");
            }
        }
        ds4_metal_tensor_free(tlow);
        ds4_metal_tensor_free(theads);
        free(W);
    }

    STEP("shared_gate_up_swiglu_q8_0 (Q8 gate, Q8 up, SwiGLU fuse)");
    {
        const uint64_t in_dim = 32;
        const uint64_t out_dim = 6;
        const uint64_t blocks_per_row = (in_dim + 31u) / 32u;
        const uint64_t row_bytes = blocks_per_row * 34u;
        const uint64_t weight_bytes = row_bytes * out_dim;
        uint8_t *W = (uint8_t *)calloc(1, 2u * (size_t)weight_bytes);
        require(W != NULL, "shared_gate_up W alloc");
        /* Build Wg in [0, weight_bytes), Wu in [weight_bytes, 2*weight_bytes). */
        for (uint32_t r = 0; r < out_dim; r++) {
            float gvals[32], uvals[32];
            float amax_g = 0.0f, amax_u = 0.0f;
            for (uint32_t c = 0; c < in_dim; c++) {
                gvals[c] = 0.15f * cosf((float)(r * 7 + c) * 0.5f);
                uvals[c] = 0.12f * sinf((float)(r * 11 + c) * 0.4f);
                if (fabsf(gvals[c]) > amax_g) amax_g = fabsf(gvals[c]);
                if (fabsf(uvals[c]) > amax_u) amax_u = fabsf(uvals[c]);
            }
            float sg = amax_g / 127.0f; if (sg == 0.0f) sg = 1.0f;
            float su = amax_u / 127.0f; if (su == 0.0f) su = 1.0f;
            uint16_t hg = f32_to_f16(sg);
            uint16_t hu = f32_to_f16(su);
            uint8_t *dg = W + (uint64_t)r * row_bytes;
            uint8_t *du = W + weight_bytes + (uint64_t)r * row_bytes;
            dg[0] = (uint8_t)(hg & 0xff); dg[1] = (uint8_t)(hg >> 8);
            du[0] = (uint8_t)(hu & 0xff); du[1] = (uint8_t)(hu >> 8);
            for (uint32_t c = 0; c < 32; c++) {
                int qg = (int)roundf(gvals[c] / sg); if (qg<-127)qg=-127; if (qg>127)qg=127;
                int qu = (int)roundf(uvals[c] / su); if (qu<-127)qu=-127; if (qu>127)qu=127;
                dg[2 + c] = (uint8_t)(int8_t)qg;
                du[2 + c] = (uint8_t)(int8_t)qu;
            }
        }
        float xv[32];
        for (uint32_t i = 0; i < in_dim; i++) xv[i] = 0.2f * sinf((float)(i + 1) * 0.6f);
        ds4_metal_tensor *tx_s = ds4_metal_tensor_alloc(sizeof(xv));
        ds4_metal_tensor *tg = ds4_metal_tensor_alloc((uint64_t)out_dim * sizeof(float));
        ds4_metal_tensor *tu = ds4_metal_tensor_alloc((uint64_t)out_dim * sizeof(float));
        ds4_metal_tensor *tmid_s = ds4_metal_tensor_alloc((uint64_t)out_dim * sizeof(float));
        require(tx_s && tg && tu && tmid_s, "shared_gate_up tensor alloc");
        require(ds4_metal_tensor_write(tx_s, 0, xv, sizeof(xv)), "shared_gate_up x");
        require(ds4_metal_shared_gate_up_swiglu_q8_0_tensor(
                    tg, tu, tmid_s, W, 2u * weight_bytes,
                    0, weight_bytes, in_dim, out_dim, tx_s),
                "shared_gate_up_swiglu_q8_0");
        float gg[6], gu[6], gmid[6];
        read_tensor(tg, gg, out_dim);
        read_tensor(tu, gu, out_dim);
        read_tensor(tmid_s, gmid, out_dim);
        for (uint32_t r = 0; r < out_dim; r++) {
            float refg = q8_0_row_dot(W, in_dim, r, xv);
            float refu = q8_0_row_dot(W + weight_bytes, in_dim, r, xv);
            require(nearf(gg[r], refg, 1e-3f), "shared_gate value");
            require(nearf(gu[r], refu, 1e-3f), "shared_up value");
            float silu = refg / (1.0f + expf(-refg));
            require(nearf(gmid[r], silu * refu, 1e-3f), "shared_mid swiglu");
        }
        ds4_metal_tensor_free(tmid_s);
        ds4_metal_tensor_free(tu);
        ds4_metal_tensor_free(tg);
        ds4_metal_tensor_free(tx_s);
        free(W);
    }

    STEP("matmul_q8_0_hc_expand (Q8 matmul + hc_expand_split)");
    {
        const uint64_t in_dim = 32;
        const uint32_t n_embd = 8;
        const uint64_t out_dim = n_embd;
        const uint32_t n_hc = 4;
        const uint32_t mix_hc = 2u * n_hc + n_hc * n_hc; /* 24 */
        const uint64_t blocks_per_row = (in_dim + 31u) / 32u;
        const uint64_t row_bytes = blocks_per_row * 34u;
        const uint64_t weight_bytes = row_bytes * out_dim;
        uint8_t *W = (uint8_t *)calloc(1, (size_t)weight_bytes);
        require(W != NULL, "q8_hc_expand W alloc");
        for (uint32_t r = 0; r < out_dim; r++) {
            float vals[32];
            float amax = 0.0f;
            for (uint32_t c = 0; c < in_dim; c++) {
                vals[c] = 0.1f * sinf((float)(r * 13 + c) * 0.4f);
                if (fabsf(vals[c]) > amax) amax = fabsf(vals[c]);
            }
            float sc = amax / 127.0f; if (sc == 0.0f) sc = 1.0f;
            uint16_t hf = f32_to_f16(sc);
            uint8_t *d = W + (uint64_t)r * row_bytes;
            d[0] = (uint8_t)(hf & 0xff); d[1] = (uint8_t)(hf >> 8);
            for (uint32_t c = 0; c < 32; c++) {
                int q = (int)roundf(vals[c] / sc); if (q<-127)q=-127; if (q>127)q=127;
                d[2 + c] = (uint8_t)(int8_t)q;
            }
        }
        float xv[32];
        for (uint32_t i = 0; i < in_dim; i++) xv[i] = 0.3f * cosf((float)(i + 1) * 0.7f);
        float resid[4 * 8];
        for (uint32_t i = 0; i < n_hc * n_embd; i++) resid[i] = 0.2f * sinf((float)(i + 1) * 0.5f);
        float split_in[24];
        for (uint32_t i = 0; i < mix_hc; i++) split_in[i] = 0.05f * (float)((int)i - 12);

        ds4_metal_tensor *tx_h = ds4_metal_tensor_alloc(sizeof(xv));
        ds4_metal_tensor *tres_h = ds4_metal_tensor_alloc(sizeof(resid));
        ds4_metal_tensor *tsplit_h = ds4_metal_tensor_alloc(sizeof(split_in));
        ds4_metal_tensor *tblock = ds4_metal_tensor_alloc((uint64_t)n_embd * sizeof(float));
        ds4_metal_tensor *tout_h = ds4_metal_tensor_alloc(sizeof(resid));
        require(tx_h && tres_h && tsplit_h && tblock && tout_h, "q8_hc alloc");
        require(ds4_metal_tensor_write(tx_h, 0, xv, sizeof(xv)), "");
        require(ds4_metal_tensor_write(tres_h, 0, resid, sizeof(resid)), "");
        require(ds4_metal_tensor_write(tsplit_h, 0, split_in, sizeof(split_in)), "");
        require(ds4_metal_matmul_q8_0_hc_expand_tensor(
                    tout_h, tblock, W, weight_bytes, 0, in_dim, out_dim, tx_h,
                    tres_h, tsplit_h, n_embd, n_hc),
                "matmul_q8_0_hc_expand");
        float gblock[8];
        read_tensor(tblock, gblock, n_embd);
        for (uint32_t r = 0; r < out_dim; r++) {
            float ref = q8_0_row_dot(W, in_dim, r, xv);
            require(nearf(gblock[r], ref, 1e-3f), "q8_hc block value");
        }
        float gout[4 * 8];
        read_tensor(tout_h, gout, n_hc * n_embd);
        const float *post = split_in + n_hc;
        const float *comb = split_in + 2u * n_hc;
        for (uint32_t dst_hc = 0; dst_hc < 4; dst_hc++) {
            for (uint32_t d = 0; d < n_embd; d++) {
                float ref = gblock[d] * post[dst_hc];
                for (uint32_t src_hc = 0; src_hc < 4; src_hc++)
                    ref += comb[dst_hc * 4 + src_hc] * resid[src_hc * n_embd + d];
                require(nearf(gout[dst_hc * n_embd + d], ref, 1e-3f), "q8_hc out value");
            }
        }
        ds4_metal_tensor_free(tout_h);
        ds4_metal_tensor_free(tblock);
        ds4_metal_tensor_free(tsplit_h);
        ds4_metal_tensor_free(tres_h);
        ds4_metal_tensor_free(tx_h);
        free(W);
    }

    STEP("shared_down_hc_expand_q8_0 (Q8 down + (routed+shared) expand)");
    {
        const uint64_t in_dim = 32;
        const uint32_t n_embd = 8;
        const uint64_t out_dim = n_embd;
        const uint32_t n_hc = 4;
        const uint32_t mix_hc = 2u * n_hc + n_hc * n_hc;
        const uint64_t blocks_per_row = (in_dim + 31u) / 32u;
        const uint64_t row_bytes = blocks_per_row * 34u;
        const uint64_t weight_bytes = row_bytes * out_dim;
        uint8_t *W = (uint8_t *)calloc(1, (size_t)weight_bytes);
        require(W != NULL, "shared_down W alloc");
        for (uint32_t r = 0; r < out_dim; r++) {
            float vals[32];
            float amax = 0.0f;
            for (uint32_t c = 0; c < in_dim; c++) {
                vals[c] = 0.08f * sinf((float)(r * 19 + c) * 0.55f);
                if (fabsf(vals[c]) > amax) amax = fabsf(vals[c]);
            }
            float sc = amax / 127.0f; if (sc == 0.0f) sc = 1.0f;
            uint16_t hf = f32_to_f16(sc);
            uint8_t *d = W + (uint64_t)r * row_bytes;
            d[0] = (uint8_t)(hf & 0xff); d[1] = (uint8_t)(hf >> 8);
            for (uint32_t c = 0; c < 32; c++) {
                int q = (int)roundf(vals[c] / sc); if (q<-127)q=-127; if (q>127)q=127;
                d[2 + c] = (uint8_t)(int8_t)q;
            }
        }
        float shared_mid[32];
        for (uint32_t i = 0; i < in_dim; i++) shared_mid[i] = 0.4f * sinf((float)(i + 1) * 0.3f);
        float routed_out[8];
        for (uint32_t i = 0; i < n_embd; i++) routed_out[i] = 0.3f * cosf((float)(i + 1) * 0.6f);
        float resid[4 * 8];
        for (uint32_t i = 0; i < n_hc * n_embd; i++) resid[i] = 0.15f * cosf((float)(i + 1) * 0.45f);
        float split_in[24];
        for (uint32_t i = 0; i < mix_hc; i++) split_in[i] = 0.04f * (float)((int)i - 11);

        ds4_metal_tensor *tmid_d = ds4_metal_tensor_alloc(sizeof(shared_mid));
        ds4_metal_tensor *trouted = ds4_metal_tensor_alloc(sizeof(routed_out));
        ds4_metal_tensor *tres_d = ds4_metal_tensor_alloc(sizeof(resid));
        ds4_metal_tensor *tspl_d = ds4_metal_tensor_alloc(sizeof(split_in));
        ds4_metal_tensor *tshared = ds4_metal_tensor_alloc((uint64_t)n_embd * sizeof(float));
        ds4_metal_tensor *tout_d = ds4_metal_tensor_alloc(sizeof(resid));
        require(tmid_d && trouted && tres_d && tspl_d && tshared && tout_d, "shared_down alloc");
        require(ds4_metal_tensor_write(tmid_d, 0, shared_mid, sizeof(shared_mid)), "");
        require(ds4_metal_tensor_write(trouted, 0, routed_out, sizeof(routed_out)), "");
        require(ds4_metal_tensor_write(tres_d, 0, resid, sizeof(resid)), "");
        require(ds4_metal_tensor_write(tspl_d, 0, split_in, sizeof(split_in)), "");
        require(ds4_metal_shared_down_hc_expand_q8_0_tensor(
                    tout_d, tshared, W, weight_bytes, 0, in_dim, out_dim,
                    tmid_d, trouted, tres_d, tspl_d, n_embd, n_hc),
                "shared_down_hc_expand_q8_0");
        float gshared[8];
        read_tensor(tshared, gshared, n_embd);
        for (uint32_t r = 0; r < out_dim; r++) {
            float ref = q8_0_row_dot(W, in_dim, r, shared_mid);
            require(nearf(gshared[r], ref, 1e-3f), "shared_down shared_out value");
        }
        float gout[4 * 8];
        read_tensor(tout_d, gout, n_hc * n_embd);
        const float *post = split_in + n_hc;
        const float *comb = split_in + 2u * n_hc;
        for (uint32_t dst_hc = 0; dst_hc < 4; dst_hc++) {
            for (uint32_t d = 0; d < n_embd; d++) {
                float bv = routed_out[d] + gshared[d];
                float ref = bv * post[dst_hc];
                for (uint32_t src_hc = 0; src_hc < 4; src_hc++)
                    ref += comb[dst_hc * 4 + src_hc] * resid[src_hc * n_embd + d];
                require(nearf(gout[dst_hc * n_embd + d], ref, 1e-3f), "shared_down out value");
            }
        }
        ds4_metal_tensor_free(tout_d);
        ds4_metal_tensor_free(tshared);
        ds4_metal_tensor_free(tspl_d);
        ds4_metal_tensor_free(tres_d);
        ds4_metal_tensor_free(trouted);
        ds4_metal_tensor_free(tmid_d);
        free(W);
    }

    STEP("dsv4_qkv_rms_norm_rows (paired weighted RMS)");
    {
        const uint32_t qn = 8;
        const uint32_t kvn = 12;
        const uint32_t rows = 3;
        float qv[3 * 8], kvv[3 * 12];
        float qw[8], kvw[12];
        for (uint32_t i = 0; i < rows * qn; i++) qv[i] = sinf((float)(i + 1) * 0.4f);
        for (uint32_t i = 0; i < rows * kvn; i++) kvv[i] = cosf((float)(i + 1) * 0.5f);
        for (uint32_t i = 0; i < qn; i++) qw[i] = 0.5f + 0.1f * (float)i;
        for (uint32_t i = 0; i < kvn; i++) kvw[i] = -0.3f + 0.05f * (float)i;
        uint8_t blob[sizeof(qw) + sizeof(kvw)];
        memcpy(blob, qw, sizeof(qw));
        memcpy(blob + sizeof(qw), kvw, sizeof(kvw));
        ds4_metal_tensor *tq_n = ds4_metal_tensor_alloc(sizeof(qv));
        ds4_metal_tensor *tkv_n = ds4_metal_tensor_alloc(sizeof(kvv));
        ds4_metal_tensor *tq_o = ds4_metal_tensor_alloc(sizeof(qv));
        ds4_metal_tensor *tkv_o = ds4_metal_tensor_alloc(sizeof(kvv));
        require(tq_n && tkv_n && tq_o && tkv_o, "dsv4_qkv alloc");
        require(ds4_metal_tensor_write(tq_n, 0, qv, sizeof(qv)), "");
        require(ds4_metal_tensor_write(tkv_n, 0, kvv, sizeof(kvv)), "");
        require(ds4_metal_dsv4_qkv_rms_norm_rows_tensor(
                    tq_o, tq_n, blob, sizeof(blob), 0, qn,
                    tkv_o, tkv_n, sizeof(qw), kvn, rows, 1e-6f),
                "dsv4_qkv_rms_norm_rows");
        float gq[3 * 8], gkv[3 * 12];
        read_tensor(tq_o, gq, rows * qn);
        read_tensor(tkv_o, gkv, rows * kvn);
        for (uint32_t r = 0; r < rows; r++) {
            float ss = 0.0f;
            for (uint32_t i = 0; i < qn; i++) ss += qv[r * qn + i] * qv[r * qn + i];
            float sc = 1.0f / sqrtf(ss / (float)qn + 1e-6f);
            for (uint32_t i = 0; i < qn; i++)
                require(nearf(gq[r * qn + i], qv[r * qn + i] * sc * qw[i], 1e-5f), "dsv4_qkv q value");
        }
        for (uint32_t r = 0; r < rows; r++) {
            float ss = 0.0f;
            for (uint32_t i = 0; i < kvn; i++) ss += kvv[r * kvn + i] * kvv[r * kvn + i];
            float sc = 1.0f / sqrtf(ss / (float)kvn + 1e-6f);
            for (uint32_t i = 0; i < kvn; i++)
                require(nearf(gkv[r * kvn + i], kvv[r * kvn + i] * sc * kvw[i], 1e-5f), "dsv4_qkv kv value");
        }
        ds4_metal_tensor_free(tkv_o);
        ds4_metal_tensor_free(tq_o);
        ds4_metal_tensor_free(tkv_n);
        ds4_metal_tensor_free(tq_n);
    }

    STEP("head_rms_norm (per-head independent RMS)");
    {
        const uint32_t n_tok = 2;
        const uint32_t n_head = 3;
        const uint32_t head_dim = 8;
        float xv[2 * 3 * 8];
        for (uint32_t i = 0; i < n_tok * n_head * head_dim; i++)
            xv[i] = 0.4f * sinf((float)(i + 1) * 0.3f);
        float ref[2 * 3 * 8];
        memcpy(ref, xv, sizeof(xv));
        ds4_metal_tensor *thx = ds4_metal_tensor_alloc(sizeof(xv));
        require(thx != NULL, "head_rms alloc");
        require(ds4_metal_tensor_write(thx, 0, xv, sizeof(xv)), "head_rms x");
        require(ds4_metal_head_rms_norm_tensor(thx, n_tok, n_head, head_dim, 1e-6f),
                "head_rms_norm");
        float got[2 * 3 * 8];
        read_tensor(thx, got, n_tok * n_head * head_dim);
        for (uint32_t t = 0; t < n_tok; t++) {
            for (uint32_t h = 0; h < n_head; h++) {
                float ss = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) {
                    float v = ref[(t * n_head + h) * head_dim + d];
                    ss += v * v;
                }
                float sc = 1.0f / sqrtf(ss / (float)head_dim + 1e-6f);
                for (uint32_t d = 0; d < head_dim; d++) {
                    float v = ref[(t * n_head + h) * head_dim + d];
                    require(nearf(got[(t * n_head + h) * head_dim + d], v * sc, 1e-5f),
                            "head_rms value");
                }
            }
        }
        ds4_metal_tensor_free(thx);
    }

    STEP("store_raw_kv (FP16 round-trip per row)");
    {
        const uint32_t head_dim = 8;
        const uint32_t raw_cap = 5;
        float kv_in[8] = { 0.5f, -0.25f, 1.125f, -2.5f, 0.0f, 3.75f, -0.0625f, 100.5f };
        ds4_metal_tensor *tkv = ds4_metal_tensor_alloc(sizeof(kv_in));
        ds4_metal_tensor *traw = ds4_metal_tensor_alloc((uint64_t)raw_cap * head_dim * sizeof(float));
        require(tkv && traw, "store_raw_kv alloc");
        float zeros[5 * 8] = {0};
        require(ds4_metal_tensor_write(traw, 0, zeros, sizeof(zeros)), "store_raw_kv raw zero");
        require(ds4_metal_tensor_write(tkv, 0, kv_in, sizeof(kv_in)), "store_raw_kv write");
        const uint32_t target_row = 3;
        require(ds4_metal_store_raw_kv_tensor(traw, tkv, raw_cap, target_row, head_dim),
                "store_raw_kv");
        float got[5 * 8];
        read_tensor(traw, got, raw_cap * head_dim);
        for (uint32_t i = 0; i < head_dim; i++) {
            float ref = f16_to_f32(f32_to_f16(kv_in[i]));
            require(nearf(got[target_row * head_dim + i], ref, 1e-6f), "store_raw_kv row value");
        }
        for (uint32_t r = 0; r < raw_cap; r++) {
            if (r == target_row) continue;
            for (uint32_t i = 0; i < head_dim; i++) {
                require(nearf(got[r * head_dim + i], 0.0f, 1e-6f), "store_raw_kv other row untouched");
            }
        }
        ds4_metal_tensor_free(traw);
        ds4_metal_tensor_free(tkv);
    }

    STEP("store_raw_kv_batch (consecutive rows starting at pos0)");
    {
        const uint32_t head_dim = 4;
        const uint32_t raw_cap = 6;
        const uint32_t pos0 = 2;
        const uint32_t n_tokens = 3;
        float kv_in[3 * 4];
        for (uint32_t i = 0; i < n_tokens * head_dim; i++) kv_in[i] = 0.5f * (float)(i + 1) - 1.5f;
        ds4_metal_tensor *tkv2 = ds4_metal_tensor_alloc(sizeof(kv_in));
        ds4_metal_tensor *traw2 = ds4_metal_tensor_alloc((uint64_t)raw_cap * head_dim * sizeof(float));
        require(tkv2 && traw2, "store_raw_kv_batch alloc");
        float zeros[6 * 4] = {0};
        require(ds4_metal_tensor_write(traw2, 0, zeros, sizeof(zeros)), "");
        require(ds4_metal_tensor_write(tkv2, 0, kv_in, sizeof(kv_in)), "");
        require(ds4_metal_store_raw_kv_batch_tensor(traw2, tkv2, raw_cap, pos0, n_tokens, head_dim),
                "store_raw_kv_batch");
        float got[6 * 4];
        read_tensor(traw2, got, raw_cap * head_dim);
        for (uint32_t r = 0; r < raw_cap; r++) {
            for (uint32_t i = 0; i < head_dim; i++) {
                if (r >= pos0 && r < pos0 + n_tokens) {
                    float ref = f16_to_f32(f32_to_f16(kv_in[(r - pos0) * head_dim + i]));
                    require(nearf(got[r * head_dim + i], ref, 1e-6f), "store_raw_kv_batch value");
                } else {
                    require(nearf(got[r * head_dim + i], 0.0f, 1e-6f), "store_raw_kv_batch outside");
                }
            }
        }
        ds4_metal_tensor_free(traw2);
        ds4_metal_tensor_free(tkv2);
    }

    /* SWA cache is a ring; chunked-prefill chunks beyond the first have
     * pos0 + n_tokens > raw_cap, which must wrap modulo raw_cap. The
     * single-chunk test above only covered the non-wrapping case; this
     * one drives a span that wraps and validates each physical row. */
    STEP("store_raw_kv_batch (wraps around the SWA ring)");
    {
        const uint32_t head_dim = 4;
        const uint32_t raw_cap  = 6;
        const uint32_t pos0     = 4;     /* starts near the end of the ring */
        const uint32_t n_tokens = 5;     /* writes physical rows 4,5,0,1,2 */
        float kv_in[5 * 4];
        for (uint32_t i = 0; i < n_tokens * head_dim; i++) kv_in[i] = 0.25f * (float)(i + 3) - 0.7f;
        ds4_metal_tensor *tkv3  = ds4_metal_tensor_alloc(sizeof(kv_in));
        ds4_metal_tensor *traw3 = ds4_metal_tensor_alloc((uint64_t)raw_cap * head_dim * sizeof(float));
        require(tkv3 && traw3, "store_raw_kv_batch wrap alloc");
        float zeros[6 * 4] = {0};
        require(ds4_metal_tensor_write(traw3, 0, zeros, sizeof(zeros)), "store_raw_kv_batch wrap zero");
        require(ds4_metal_tensor_write(tkv3, 0, kv_in, sizeof(kv_in)), "store_raw_kv_batch wrap write");
        require(ds4_metal_store_raw_kv_batch_tensor(traw3, tkv3, raw_cap, pos0, n_tokens, head_dim),
                "store_raw_kv_batch wrap");
        float got[6 * 4];
        read_tensor(traw3, got, raw_cap * head_dim);
        /* Build the expected ring image: row (pos0+t) % raw_cap holds kv_in[t]. */
        float expect[6 * 4] = {0};
        for (uint32_t t = 0; t < n_tokens; t++) {
            const uint32_t r = (pos0 + t) % raw_cap;
            for (uint32_t i = 0; i < head_dim; i++) {
                expect[r * head_dim + i] = f16_to_f32(f32_to_f16(kv_in[t * head_dim + i]));
            }
        }
        for (uint32_t i = 0; i < raw_cap * head_dim; i++) {
            require(nearf(got[i], expect[i], 1e-6f), "store_raw_kv_batch wrap value");
        }
        ds4_metal_tensor_free(traw3);
        ds4_metal_tensor_free(tkv3);
    }

    STEP("router_select_tensor (defaults n_expert=256, n_used=6, no bias, no hash)");
    {
        const uint32_t n_expert = 256;
        const uint32_t n_used = 6;
        float logits[256];
        for (uint32_t i = 0; i < n_expert; i++)
            logits[i] = 0.5f * sinf((float)(i + 1) * 0.1f) - 0.2f * (float)((i / 32) % 5);

        ds4_metal_tensor *tlog = ds4_metal_tensor_alloc(sizeof(logits));
        ds4_metal_tensor *tsel = ds4_metal_tensor_alloc((uint64_t)n_used * sizeof(int32_t));
        ds4_metal_tensor *twts = ds4_metal_tensor_alloc((uint64_t)n_used * sizeof(float));
        ds4_metal_tensor *tprobs = ds4_metal_tensor_alloc(sizeof(logits));
        require(tlog && tsel && twts && tprobs, "router_select alloc");
        require(ds4_metal_tensor_write(tlog, 0, logits, sizeof(logits)), "");
        /* model_map can be a small dummy buffer since has_bias=false, hash_mode=false. */
        uint8_t dummy[16] = {0};
        require(ds4_metal_router_select_tensor(tsel, twts, tprobs, dummy, sizeof(dummy),
                                                0, 0, 0, 0,
                                                /*n_expert_groups=*/ 0, /*n_group_used=*/ 0,
                                                /*has_bias=*/ false, /*hash_mode=*/ false,
                                                tlog),
                "router_select default");

        float ref_probs[256];
        for (uint32_t i = 0; i < n_expert; i++) {
            float l = logits[i];
            float sp = l > 20.0f ? l : logf(1.0f + expf(l));
            ref_probs[i] = sqrtf(sp > 0.0f ? sp : 0.0f);
        }
        float got_probs[256];
        read_tensor(tprobs, got_probs, n_expert);
        for (uint32_t i = 0; i < n_expert; i++)
            require(nearf(got_probs[i], ref_probs[i], 1e-5f), "router_select probs");

        /* Reference top-k by probs (no bias). */
        int32_t ref_sel[6];
        int taken[256] = {0};
        for (uint32_t k = 0; k < n_used; k++) {
            float best = -3.4028234663852886e38f; int32_t best_i = -1;
            for (uint32_t e = 0; e < n_expert; e++) {
                if (taken[e]) continue;
                if (ref_probs[e] > best) { best = ref_probs[e]; best_i = (int32_t)e; }
            }
            ref_sel[k] = best_i;
            taken[best_i] = 1;
        }
        int32_t got_sel[6];
        ds4_metal_tensor_read(tsel, 0, got_sel, sizeof(got_sel));
        for (uint32_t k = 0; k < n_used; k++)
            require(got_sel[k] == ref_sel[k], "router_select selected");

        float sum = 0.0f;
        for (uint32_t k = 0; k < n_used; k++) sum += ref_probs[ref_sel[k]];
        if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;
        float ref_w[6];
        for (uint32_t k = 0; k < n_used; k++) ref_w[k] = ref_probs[ref_sel[k]] / sum * 1.5f;
        float got_w[6];
        read_tensor(twts, got_w, n_used);
        for (uint32_t k = 0; k < n_used; k++)
            require(nearf(got_w[k], ref_w[k], 1e-5f), "router_select weight");

        ds4_metal_tensor_free(tprobs);
        ds4_metal_tensor_free(twts);
        ds4_metal_tensor_free(tsel);
        ds4_metal_tensor_free(tlog);
    }

    STEP("router_select_tensor with bias");
    {
        const uint32_t n_expert = 256;
        const uint32_t n_used = 6;
        float logits[256];
        float bias[256];
        for (uint32_t i = 0; i < n_expert; i++) {
            logits[i] = 0.3f * cosf((float)(i + 1) * 0.07f);
            bias[i] = (i % 17 == 0) ? 5.0f : 0.0f; /* boost specific experts */
        }
        uint8_t blob[256 * sizeof(float)];
        memcpy(blob, bias, sizeof(bias));

        ds4_metal_tensor *tlog = ds4_metal_tensor_alloc(sizeof(logits));
        ds4_metal_tensor *tsel = ds4_metal_tensor_alloc((uint64_t)n_used * sizeof(int32_t));
        ds4_metal_tensor *twts = ds4_metal_tensor_alloc((uint64_t)n_used * sizeof(float));
        ds4_metal_tensor *tprobs = ds4_metal_tensor_alloc(sizeof(logits));
        require(tlog && tsel && twts && tprobs, "router_select bias alloc");
        require(ds4_metal_tensor_write(tlog, 0, logits, sizeof(logits)), "");
        require(ds4_metal_router_select_tensor(tsel, twts, tprobs, blob, sizeof(blob),
                                                0, 0, 0, 0, 0, 0, true, false, tlog),
                "router_select bias");
        float ref_probs[256];
        for (uint32_t i = 0; i < n_expert; i++) {
            float l = logits[i];
            float sp = l > 20.0f ? l : logf(1.0f + expf(l));
            ref_probs[i] = sqrtf(sp > 0.0f ? sp : 0.0f);
        }
        int32_t ref_sel[6];
        int taken[256] = {0};
        for (uint32_t k = 0; k < n_used; k++) {
            float best = -3.4028234663852886e38f; int32_t best_i = -1;
            for (uint32_t e = 0; e < n_expert; e++) {
                if (taken[e]) continue;
                float v = ref_probs[e] + bias[e];
                if (v > best) { best = v; best_i = (int32_t)e; }
            }
            ref_sel[k] = best_i;
            taken[best_i] = 1;
        }
        int32_t got_sel[6];
        ds4_metal_tensor_read(tsel, 0, got_sel, sizeof(got_sel));
        for (uint32_t k = 0; k < n_used; k++)
            require(got_sel[k] == ref_sel[k], "router_select bias selected");
        /* The biased experts (every 17th) should appear among the picks. */
        int found_biased = 0;
        for (uint32_t k = 0; k < n_used; k++) if (got_sel[k] % 17 == 0) found_biased = 1;
        require(found_biased, "router_select bias actually applied");

        ds4_metal_tensor_free(tprobs);
        ds4_metal_tensor_free(twts);
        ds4_metal_tensor_free(tsel);
        ds4_metal_tensor_free(tlog);
    }

    /* Reference top-K: pick the top_k largest scores by value (descending),
     * skipping the -inf sentinel; output the picked indices sorted ascending,
     * with -1 sentinels for empty slots pushed to the end. Mirrors the
     * semantics enforced by both rocm_indexer_topk_parallel_kernel and
     * rocm_indexer_topk_serial_kernel. */
    /* (Helper inline to keep the test self-contained.) */
    #define TOPK_REFERENCE(scores_, n_comp_, top_k_, out_) do { \
        for (uint32_t _k = 0; _k < (top_k_); _k++) (out_)[_k] = -1; \
        for (uint32_t _k = 0; _k < (top_k_); _k++) { \
            float _bv = -3.4028234663852886e38f; \
            int32_t _bi = -1; \
            for (uint32_t _c = 0; _c < (n_comp_); _c++) { \
                float _v = (scores_)[_c]; \
                if (_v == -3.4028234663852886e38f) continue; \
                int _taken = 0; \
                for (uint32_t _kk = 0; _kk < _k; _kk++) if ((out_)[_kk] == (int32_t)_c) { _taken = 1; break; } \
                if (_taken) continue; \
                if (_v > _bv) { _bv = _v; _bi = (int32_t)_c; } \
            } \
            (out_)[_k] = _bi; \
        } \
        /* sort ascending with sentinels at end */ \
        for (uint32_t _i = 0; _i < (top_k_); _i++) { \
            for (uint32_t _j = _i + 1; _j < (top_k_); _j++) { \
                int32_t _a = (out_)[_i], _b = (out_)[_j]; \
                int _swap = 0; \
                if (_a < 0 && _b >= 0) _swap = 1; \
                else if (_a >= 0 && _b >= 0 && _b < _a) _swap = 1; \
                if (_swap) { (out_)[_i] = _b; (out_)[_j] = _a; } \
            } \
        } \
    } while (0)

    STEP("indexer_topk_tensor parallel (top_k=4, n_comp=64, single token)");
    {
        const uint32_t TK = 4, NC = 64, NT = 1;
        float *scores = (float *)malloc(NC * sizeof(float));
        int32_t ref[4], got[4];
        for (uint32_t i = 0; i < NC; i++) scores[i] = sinf((float)(i + 1) * 0.37f) * 2.0f;
        TOPK_REFERENCE(scores, NC, TK, ref);
        ds4_metal_tensor *tsc = ds4_metal_tensor_alloc((uint64_t)NT * NC * sizeof(float));
        ds4_metal_tensor *tdt = ds4_metal_tensor_alloc((uint64_t)NT * TK * sizeof(int32_t));
        require(tsc && tdt, "topk parallel alloc");
        require(ds4_metal_tensor_write(tsc, 0, scores, NC * sizeof(float)),
                "topk parallel scores write");
        require(ds4_metal_indexer_topk_tensor(tdt, tsc, NC, NT, TK),
                "indexer_topk parallel small");
        require(ds4_metal_tensor_read(tdt, 0, got, TK * sizeof(int32_t)),
                "topk parallel read");
        for (uint32_t k = 0; k < TK; k++) {
            require(got[k] == ref[k], "topk parallel small value");
        }
        ds4_metal_tensor_free(tdt);
        ds4_metal_tensor_free(tsc);
        free(scores);
    }

    STEP("indexer_topk_tensor parallel (top_k=7 non-pow2, batched n_tokens=3)");
    {
        const uint32_t TK = 7, NC = 100, NT = 3;
        float *scores = (float *)malloc((size_t)NT * NC * sizeof(float));
        int32_t ref[7], got[7];
        for (uint32_t t = 0; t < NT; t++) {
            for (uint32_t i = 0; i < NC; i++) {
                scores[t * NC + i] = cosf((float)(t * NC + i + 1) * 0.21f) * 1.5f;
            }
        }
        ds4_metal_tensor *tsc = ds4_metal_tensor_alloc((uint64_t)NT * NC * sizeof(float));
        ds4_metal_tensor *tdt = ds4_metal_tensor_alloc((uint64_t)NT * TK * sizeof(int32_t));
        require(tsc && tdt, "topk parallel batched alloc");
        require(ds4_metal_tensor_write(tsc, 0, scores, (size_t)NT * NC * sizeof(float)),
                "topk parallel batched scores write");
        require(ds4_metal_indexer_topk_tensor(tdt, tsc, NC, NT, TK),
                "indexer_topk parallel batched");
        for (uint32_t t = 0; t < NT; t++) {
            TOPK_REFERENCE(scores + t * NC, NC, TK, ref);
            require(ds4_metal_tensor_read(tdt, t * TK * sizeof(int32_t), got, TK * sizeof(int32_t)),
                    "topk parallel batched read");
            for (uint32_t k = 0; k < TK; k++) {
                require(got[k] == ref[k], "topk parallel batched value");
            }
        }
        ds4_metal_tensor_free(tdt);
        ds4_metal_tensor_free(tsc);
        free(scores);
    }

    STEP("indexer_topk_tensor parallel (top_k=512, n_comp=8194, production size)");
    {
        const uint32_t TK = 512, NC = 8194, NT = 1;
        float *scores = (float *)malloc((size_t)NC * sizeof(float));
        int32_t *ref = (int32_t *)malloc((size_t)TK * sizeof(int32_t));
        int32_t *got = (int32_t *)malloc((size_t)TK * sizeof(int32_t));
        require(scores && ref && got, "topk parallel large alloc");
        for (uint32_t i = 0; i < NC; i++) scores[i] = sinf((float)(i + 17) * 0.013f) * 5.0f;
        TOPK_REFERENCE(scores, NC, TK, ref);
        ds4_metal_tensor *tsc = ds4_metal_tensor_alloc(NC * sizeof(float));
        ds4_metal_tensor *tdt = ds4_metal_tensor_alloc(TK * sizeof(int32_t));
        require(tsc && tdt, "topk parallel large tensor alloc");
        require(ds4_metal_tensor_write(tsc, 0, scores, NC * sizeof(float)),
                "topk parallel large scores write");
        require(ds4_metal_indexer_topk_tensor(tdt, tsc, NC, NT, TK),
                "indexer_topk parallel large");
        require(ds4_metal_tensor_read(tdt, 0, got, TK * sizeof(int32_t)),
                "topk parallel large read");
        for (uint32_t k = 0; k < TK; k++) {
            require(got[k] == ref[k], "topk parallel large value");
        }
        ds4_metal_tensor_free(tdt);
        ds4_metal_tensor_free(tsc);
        free(got); free(ref); free(scores);
    }

    STEP("indexer_topk_tensor parallel (n_comp < top_k, sentinels at end)");
    {
        const uint32_t TK = 16, NC = 5, NT = 1;
        float scores[5] = { 1.0f, 5.0f, 2.0f, 4.0f, 3.0f };
        int32_t ref[16], got[16];
        TOPK_REFERENCE(scores, NC, TK, ref);
        ds4_metal_tensor *tsc = ds4_metal_tensor_alloc(NC * sizeof(float));
        ds4_metal_tensor *tdt = ds4_metal_tensor_alloc(TK * sizeof(int32_t));
        require(tsc && tdt, "topk parallel sentinel alloc");
        require(ds4_metal_tensor_write(tsc, 0, scores, sizeof(scores)),
                "topk parallel sentinel scores write");
        require(ds4_metal_indexer_topk_tensor(tdt, tsc, NC, NT, TK),
                "indexer_topk parallel sentinel");
        require(ds4_metal_tensor_read(tdt, 0, got, TK * sizeof(int32_t)),
                "topk parallel sentinel read");
        for (uint32_t k = 0; k < TK; k++) {
            require(got[k] == ref[k], "topk parallel sentinel value");
        }
        ds4_metal_tensor_free(tdt);
        ds4_metal_tensor_free(tsc);
    }

    STEP("indexer_topk_tensor parallel (-inf scores skipped)");
    {
        const uint32_t TK = 4, NC = 16, NT = 1;
        float scores[16];
        for (uint32_t i = 0; i < NC; i++) scores[i] = (float)i * 0.1f;
        scores[3]  = -3.4028234663852886e38f;
        scores[10] = -3.4028234663852886e38f;
        scores[15] = -3.4028234663852886e38f;
        int32_t ref[4], got[4];
        TOPK_REFERENCE(scores, NC, TK, ref);
        ds4_metal_tensor *tsc = ds4_metal_tensor_alloc(NC * sizeof(float));
        ds4_metal_tensor *tdt = ds4_metal_tensor_alloc(TK * sizeof(int32_t));
        require(tsc && tdt, "topk parallel inf alloc");
        require(ds4_metal_tensor_write(tsc, 0, scores, sizeof(scores)),
                "topk parallel inf scores write");
        require(ds4_metal_indexer_topk_tensor(tdt, tsc, NC, NT, TK),
                "indexer_topk parallel inf");
        require(ds4_metal_tensor_read(tdt, 0, got, TK * sizeof(int32_t)),
                "topk parallel inf read");
        for (uint32_t k = 0; k < TK; k++) {
            require(got[k] == ref[k], "topk parallel inf value");
        }
        for (uint32_t k = 0; k < TK; k++) {
            require(got[k] != 3 && got[k] != 10 && got[k] != 15,
                    "topk parallel inf excluded");
        }
        ds4_metal_tensor_free(tdt);
        ds4_metal_tensor_free(tsc);
    }

    #undef TOPK_REFERENCE

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
