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
