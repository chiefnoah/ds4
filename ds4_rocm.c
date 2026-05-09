#include <stdbool.h>
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ds4_metal_tensor {
    void    *host_ptr;
    void    *dev_ptr;
    uint64_t offset;
    uint64_t bytes;
    int      owner;
    int      host;
} ds4_metal_tensor;

static int g_initialized;
static int g_batch_open;
static bool g_quality;
static uint64_t g_live_bytes;
static uint64_t g_peak_bytes;

int ds4_metal_rms_norm_plain_rows_tensor(ds4_metal_tensor *out,
                                          const ds4_metal_tensor *x,
                                          uint32_t n,
                                          uint32_t rows,
                                          float eps);
int ds4_metal_rms_norm_weight_rows_tensor(ds4_metal_tensor *out,
                                           const ds4_metal_tensor *x,
                                           const void *model_map,
                                           uint64_t model_size,
                                           uint64_t weight_offset,
                                           uint32_t n,
                                           uint32_t rows,
                                           float eps);

static int rocm_ok(hipError_t rc, const char *what) {
    if (rc == hipSuccess) return 1;
    fprintf(stderr, "ds4: ROCm %s failed: %s\n", what, hipGetErrorString(rc));
    return 0;
}

static int rocm_trace(void) {
    const char *v = getenv("DS4_ROCM_TRACE");
    return v && v[0] && v[0] != '0';
}

static void rocm_unimplemented(const char *name) {
    fprintf(stderr, "ds4: ROCm kernel not implemented yet: %s\n", name);
}

static const uint8_t *tensor_u8_const(const ds4_metal_tensor *t) {
    return (const uint8_t *)t->dev_ptr + t->offset;
}

static uint8_t *tensor_u8(ds4_metal_tensor *t) {
    return (uint8_t *)t->dev_ptr + t->offset;
}

static const uint8_t *tensor_host_u8_const(const ds4_metal_tensor *t) {
    return (const uint8_t *)t->host_ptr + t->offset;
}

static uint8_t *tensor_host_u8(ds4_metal_tensor *t) {
    return (uint8_t *)t->host_ptr + t->offset;
}

static __device__ float rocm_f16_to_f32(uint16_t bits) {
    __half_raw h;
    h.x = bits;
    return __half2float(h);
}

static __device__ float rocm_rope_yarn_ramp(float low, float high, int i0) {
    const float y = ((float)i0 / 2.0f - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}

static __device__ float rocm_rope_yarn_corr_factor(int n_dims,
                                                   int n_ctx_orig,
                                                   float n_rot,
                                                   float base) {
    return (float)n_dims * logf((float)n_ctx_orig / (n_rot * 6.2831853071795864769f)) /
           (2.0f * logf(base));
}

static __device__ void rocm_rope_yarn_corr_dims(int n_dims,
                                                int n_ctx_orig,
                                                float freq_base,
                                                float beta_fast,
                                                float beta_slow,
                                                float dims[2]) {
    dims[0] = fmaxf(0.0f, floorf(rocm_rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_fast, freq_base)));
    dims[1] = fminf((float)n_dims - 1.0f,
                    ceilf(rocm_rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_slow, freq_base)));
}

static __device__ void rocm_rope_yarn(float theta_extrap,
                                      float freq_scale,
                                      const float corr_dims[2],
                                      int i0,
                                      float ext_factor,
                                      float mscale,
                                      float *cos_theta,
                                      float *sin_theta) {
    const float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        const float ramp_mix = rocm_rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    *cos_theta = cosf(theta) * mscale;
    *sin_theta = sinf(theta) * mscale;
}

static __device__ float rocm_e4m3fn_value(int i) {
    const int exp = (i >> 3) & 0x0f;
    const int mant = i & 0x07;
    if (exp == 0) return (float)mant * 0.001953125f;
    return (1.0f + (float)mant * 0.125f) * exp2f((float)exp - 7.0f);
}

static __device__ float rocm_e4m3fn_dequant(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = fminf(fabsf(x), 448.0f);
    int lo = 0;
    int hi = 126;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (rocm_e4m3fn_value(mid) <= ax) lo = mid;
        else hi = mid - 1;
    }
    int best = lo;
    if (best < 126) {
        const float best_diff = fabsf(ax - rocm_e4m3fn_value(best));
        const float next_diff = fabsf(ax - rocm_e4m3fn_value(best + 1));
        if (next_diff < best_diff ||
            (next_diff == best_diff && (((best + 1) & 1) == 0) && ((best & 1) != 0))) {
            best++;
        }
    }
    return sign * rocm_e4m3fn_value(best);
}

static int rocm_upload_mapped(const void *src,
                              uint64_t bytes,
                              void **host_out,
                              void **dev_out,
                              const char *what) {
    *host_out = NULL;
    *dev_out = NULL;
    if (bytes == 0) return 0;
    if (!rocm_ok(hipHostMalloc(host_out, (size_t)bytes, hipHostMallocMapped), what)) {
        return 0;
    }
    if (!rocm_ok(hipHostGetDevicePointer(dev_out, *host_out, 0), "mapped upload device pointer")) {
        (void)hipHostFree(*host_out);
        *host_out = NULL;
        return 0;
    }
    memcpy(*host_out, src, (size_t)bytes);
    return 1;
}

__global__ static void rocm_embed_token_hc_kernel(float *out,
                                                  const uint16_t *rows,
                                                  uint32_t n_embd,
                                                  uint32_t n_hc) {
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t)n_embd * n_hc;
    if (idx >= total) return;
    const uint32_t embd = (uint32_t)(idx % n_embd);
    __half_raw h;
    h.x = rows[embd];
    out[idx] = __half2float(h);
}

__global__ static void rocm_embed_tokens_hc_kernel(float *out,
                                                   const uint16_t *rows,
                                                   uint32_t n_tokens,
                                                   uint32_t n_embd,
                                                   uint32_t n_hc) {
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t)n_tokens * n_hc * n_embd;
    if (idx >= total) return;
    const uint32_t embd = (uint32_t)(idx % n_embd);
    const uint64_t token_pos = idx / ((uint64_t)n_hc * n_embd);
    __half_raw h;
    h.x = rows[token_pos * n_embd + embd];
    out[idx] = __half2float(h);
}

__global__ static void rocm_repeat_hc_kernel(float *out,
                                             const float *row,
                                             uint32_t n_embd,
                                             uint32_t n_hc) {
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t)n_embd * n_hc;
    if (idx >= total) return;
    out[idx] = row[idx % n_embd];
}

__global__ static void rocm_add_kernel(float *out,
                                       const float *a,
                                       const float *b,
                                       uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ static void rocm_swiglu_kernel(float *out,
                                          const float *gate,
                                          const float *up,
                                          uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = gate[i];
    out[i] = (x / (1.0f + expf(-x))) * up[i];
}

__global__ static void rocm_rms_norm_plain_kernel(float *out,
                                                  const float *x,
                                                  uint32_t n,
                                                  float eps) {
    extern __shared__ float sh[];
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const float *xr = x + (uint64_t)row * n;
    float *yr = out + (uint64_t)row * n;

    float sum = 0.0f;
    for (uint32_t i = tid; i < n; i += blockDim.x) {
        const float v = xr[i];
        sum += v * v;
    }
    sh[tid] = sum;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sh[tid] += sh[tid + stride];
        __syncthreads();
    }

    const float scale = rsqrtf(sh[0] / (float)n + eps);
    for (uint32_t i = tid; i < n; i += blockDim.x) {
        yr[i] = xr[i] * scale;
    }
}

__global__ static void rocm_rms_norm_weight_kernel(float *out,
                                                   const float *x,
                                                   const float *weight,
                                                   uint32_t n,
                                                   float eps) {
    extern __shared__ float sh[];
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const float *xr = x + (uint64_t)row * n;
    float *yr = out + (uint64_t)row * n;

    float sum = 0.0f;
    for (uint32_t i = tid; i < n; i += blockDim.x) {
        const float v = xr[i];
        sum += v * v;
    }
    sh[tid] = sum;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sh[tid] += sh[tid + stride];
        __syncthreads();
    }

    const float scale = rsqrtf(sh[0] / (float)n + eps);
    for (uint32_t i = tid; i < n; i += blockDim.x) {
        yr[i] = xr[i] * scale * weight[i];
    }
}

__global__ static void rocm_matmul_f16_kernel(float *out,
                                              const uint16_t *w,
                                              const float *x,
                                              uint64_t in_dim,
                                              uint64_t out_dim) {
    extern __shared__ float sh[];
    const uint64_t row = blockIdx.x;
    const uint64_t tok = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    if (row >= out_dim) return;

    const uint16_t *wr = w + row * in_dim;
    const float *xr = x + tok * in_dim;
    float sum = 0.0f;
    for (uint64_t i = tid; i < in_dim; i += blockDim.x) {
        sum += rocm_f16_to_f32(wr[i]) * xr[i];
    }
    sh[tid] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sh[tid] += sh[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out[tok * out_dim + row] = sh[0];
}

__global__ static void rocm_matmul_f32_kernel(float *out,
                                              const float *w,
                                              const float *x,
                                              uint64_t in_dim,
                                              uint64_t out_dim) {
    extern __shared__ float sh[];
    const uint64_t row = blockIdx.x;
    const uint64_t tok = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    if (row >= out_dim) return;

    const float *wr = w + row * in_dim;
    const float *xr = x + tok * in_dim;
    float sum = 0.0f;
    for (uint64_t i = tid; i < in_dim; i += blockDim.x) {
        sum += wr[i] * xr[i];
    }
    sh[tid] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sh[tid] += sh[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out[tok * out_dim + row] = sh[0];
}

__global__ static void rocm_matmul_q8_0_kernel(float *out,
                                               const uint8_t *w,
                                               const float *x,
                                               uint64_t in_dim,
                                               uint64_t out_dim,
                                               uint64_t row_bytes) {
    extern __shared__ float sh[];
    const uint64_t row = blockIdx.x;
    const uint64_t tok = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    if (row >= out_dim) return;

    const uint8_t *wr = w + row * row_bytes;
    const float *xr = x + tok * in_dim;
    const uint64_t blocks = (in_dim + 31u) / 32u;
    float sum = 0.0f;
    for (uint64_t b = tid; b < blocks; b += blockDim.x) {
        const uint8_t *blk = wr + b * 34u;
        const uint16_t scale_bits = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
        const int8_t *qs = (const int8_t *)(blk + 2);
        const uint64_t i0 = b * 32u;
        const uint64_t n = in_dim - i0 < 32u ? in_dim - i0 : 32u;
        float dot = 0.0f;
        for (uint64_t j = 0; j < n; j++) dot += (float)qs[j] * xr[i0 + j];
        sum += rocm_f16_to_f32(scale_bits) * dot;
    }
    sh[tid] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sh[tid] += sh[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out[tok * out_dim + row] = sh[0];
}

__global__ static void rocm_hc_split_weighted_sum_norm_kernel(float *out,
                                                              float *norm_out,
                                                              float *split,
                                                              const float *mix,
                                                              const float *residual_hc,
                                                              const float *scale,
                                                              const float *base,
                                                              const float *norm_weight,
                                                              uint32_t n_embd,
                                                              uint32_t n_hc,
                                                              uint32_t sinkhorn_iters,
                                                              float eps,
                                                              float norm_eps) {
    extern __shared__ float sh[];
    float *pre = sh;
    float *sum_sh = sh + 4;
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (n_hc != 4 || n_embd == 0) return;

    const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    const float *mix_row = mix + (uint64_t)row * mix_hc;
    float *split_row = split + (uint64_t)row * mix_hc;

    if (tid == 0) {
        for (uint32_t h = 0; h < 4; h++) {
            const float z = mix_row[h] * scale[0] + base[h];
            pre[h] = 1.0f / (1.0f + expf(-z)) + eps;
            split_row[h] = pre[h];
        }
        for (uint32_t h = 0; h < 4; h++) {
            const float z = mix_row[4 + h] * scale[1] + base[4 + h];
            split_row[4 + h] = 2.0f / (1.0f + expf(-z));
        }

        float r[4][4];
        for (uint32_t i = 0; i < 4; i++) {
            float max_v = -3.4028234663852886e38f;
            for (uint32_t j = 0; j < 4; j++) {
                const uint32_t k = 8 + i * 4 + j;
                r[i][j] = mix_row[k] * scale[2] + base[k];
                if (r[i][j] > max_v) max_v = r[i][j];
            }
            float row_sum = 0.0f;
            for (uint32_t j = 0; j < 4; j++) {
                r[i][j] = expf(r[i][j] - max_v);
                row_sum += r[i][j];
            }
            for (uint32_t j = 0; j < 4; j++) r[i][j] = r[i][j] / row_sum + eps;
        }
        for (uint32_t iter = 0; iter < sinkhorn_iters; iter++) {
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
        for (uint32_t i = 0; i < 4; i++) {
            for (uint32_t j = 0; j < 4; j++) split_row[8 + i * 4 + j] = r[i][j];
        }
    }
    __syncthreads();

    float sum = 0.0f;
    for (uint32_t i = tid; i < n_embd; i += blockDim.x) {
        const uint64_t row_base = (uint64_t)row * n_hc * n_embd;
        float v = 0.0f;
        for (uint32_t h = 0; h < 4; h++) {
            v += residual_hc[row_base + (uint64_t)h * n_embd + i] * pre[h];
        }
        out[(uint64_t)row * n_embd + i] = v;
        sum += v * v;
    }
    sum_sh[tid] = sum;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sum_sh[tid] += sum_sh[tid + stride];
        __syncthreads();
    }

    const float norm_scale = rsqrtf(sum_sh[0] / (float)n_embd + norm_eps);
    for (uint32_t i = tid; i < n_embd; i += blockDim.x) {
        const float v = out[(uint64_t)row * n_embd + i];
        norm_out[(uint64_t)row * n_embd + i] = v * norm_scale * norm_weight[i];
    }
}

__global__ static void rocm_rope_tail_kernel(float *x,
                                             uint32_t n_tok,
                                             uint32_t n_head,
                                             uint32_t head_dim,
                                             uint32_t n_rot,
                                             uint32_t pos0,
                                             uint32_t n_ctx_orig,
                                             int inverse,
                                             float freq_base,
                                             float freq_scale,
                                             float ext_factor,
                                             float attn_factor,
                                             float beta_fast,
                                             float beta_slow) {
    const uint64_t pair = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t n_nope = head_dim - n_rot;
    const uint32_t n_pairs = n_rot / 2u;
    const uint64_t total = (uint64_t)n_tok * n_head * n_pairs;
    if (pair >= total) return;

    const uint32_t ic = (uint32_t)(pair % n_pairs);
    const uint64_t head_index = pair / n_pairs;
    const uint32_t head = (uint32_t)(head_index % n_head);
    const uint32_t tok = (uint32_t)(head_index / n_head);
    float *row = x + ((uint64_t)tok * n_head + head) * head_dim;

    float corr_dims[2];
    rocm_rope_yarn_corr_dims((int)n_rot, (int)n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    const uint32_t r = ic * 2u;
    const float theta_base = (float)(pos0 + tok);
    const float theta = theta_base * powf(freq_base, (-1.0f / (float)n_rot) * (float)r);
    float cos_theta;
    float sin_theta;
    rocm_rope_yarn(theta, freq_scale, corr_dims, (int)r, ext_factor, attn_factor,
                   &cos_theta, &sin_theta);
    if (inverse) sin_theta = -sin_theta;

    const uint32_t j0 = n_nope + r;
    const uint32_t j1 = j0 + 1u;
    const float x0 = row[j0];
    const float x1 = row[j1];
    row[j0] = x0 * cos_theta - x1 * sin_theta;
    row[j1] = x0 * sin_theta + x1 * cos_theta;
}

__global__ static void rocm_fp8_kv_quantize_kernel(float *x,
                                                   uint32_t n_tok,
                                                   uint32_t head_dim,
                                                   uint32_t n_rot) {
    __shared__ float scratch[64];
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (row >= n_tok || tid >= 64) return;
    const uint32_t n_nope = head_dim - n_rot;
    float *xr = x + (uint64_t)row * head_dim;
    for (uint32_t off = 0; off < n_nope; off += 64u) {
        float v = 0.0f;
        if (off + tid < n_nope) {
            v = xr[off + tid];
            scratch[tid] = fabsf(v);
        } else {
            scratch[tid] = 0.0f;
        }
        __syncthreads();
        for (uint32_t stride = 32; stride > 0; stride >>= 1) {
            if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
            __syncthreads();
        }
        const float amax = fmaxf(scratch[0], 1.0e-4f);
        const float scale = exp2f(ceilf(log2f(amax / 448.0f)));
        if (off + tid < n_nope) {
            const float q = rocm_e4m3fn_dequant(fminf(fmaxf(v / scale, -448.0f), 448.0f)) * scale;
            xr[off + tid] = q;
        }
        __syncthreads();
    }
}

int ds4_metal_init(void) {
    if (g_initialized) return 1;

    int count = 0;
    if (!rocm_ok(hipGetDeviceCount(&count), "device query")) return 0;
    if (count <= 0) {
        fprintf(stderr, "ds4: ROCm backend found no HIP devices\n");
        return 0;
    }
    if (!rocm_ok(hipSetDevice(0), "device select")) return 0;
    g_initialized = 1;
    return 1;
}

void ds4_metal_cleanup(void) {
    if (!g_initialized) return;
    (void)hipDeviceSynchronize();
    g_initialized = 0;
    g_batch_open = 0;
    g_live_bytes = 0;
    g_peak_bytes = 0;
}

ds4_metal_tensor *ds4_metal_tensor_alloc(uint64_t bytes) {
    if (!g_initialized && !ds4_metal_init()) return NULL;
    if (bytes == 0) return NULL;

    ds4_metal_tensor *t = (ds4_metal_tensor *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    /*
     * Bring-up uses mapped host allocations because hipMemcpy(H2D) into
     * hipMalloc memory faults on the current test host before returning.  The
     * ABI still hides this choice, so hot tensors can move back to device-local
     * memory after kernel correctness is established.
     */
    if (!rocm_ok(hipHostMalloc(&t->host_ptr, (size_t)bytes, hipHostMallocMapped),
                 "tensor allocation")) {
        free(t);
        return NULL;
    }
    if (!rocm_ok(hipHostGetDevicePointer(&t->dev_ptr, t->host_ptr, 0),
                 "tensor device pointer")) {
        (void)hipHostFree(t->host_ptr);
        free(t);
        return NULL;
    }
    if (rocm_trace()) {
        fprintf(stderr, "ds4: ROCm tensor alloc host-mapped host=%p dev=%p bytes=%llu\n",
                t->host_ptr,
                t->dev_ptr,
                (unsigned long long)bytes);
    }
    t->bytes = bytes;
    t->owner = 1;
    t->host = 1;
    g_live_bytes += bytes;
    if (g_live_bytes > g_peak_bytes) g_peak_bytes = g_live_bytes;
    return t;
}

ds4_metal_tensor *ds4_metal_tensor_view(const ds4_metal_tensor *base, uint64_t offset, uint64_t bytes) {
    if (!base) return NULL;
    if (offset > base->bytes || bytes > base->bytes - offset) return NULL;
    if (base->offset > UINT64_MAX - offset) return NULL;

    ds4_metal_tensor *t = (ds4_metal_tensor *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    t->host_ptr = base->host_ptr;
    t->dev_ptr = base->dev_ptr;
    t->offset = base->offset + offset;
    t->bytes = bytes;
    t->owner = 0;
    t->host = base->host;
    return t;
}

void ds4_metal_tensor_free(ds4_metal_tensor *tensor) {
    if (!tensor) return;
    if (tensor->owner && (tensor->host_ptr || tensor->dev_ptr)) {
        if (tensor->host) (void)hipHostFree(tensor->host_ptr);
        else (void)hipFree(tensor->dev_ptr);
        g_live_bytes = tensor->bytes <= g_live_bytes ? g_live_bytes - tensor->bytes : 0;
    }
    free(tensor);
}

uint64_t ds4_metal_tensor_bytes(const ds4_metal_tensor *tensor) {
    return tensor ? tensor->bytes : 0;
}

void *ds4_metal_tensor_contents(ds4_metal_tensor *tensor) {
    if (!tensor || !tensor->host || !tensor->host_ptr) return NULL;
    return tensor_host_u8(tensor);
}

int ds4_metal_tensor_write(ds4_metal_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes) {
    if (!tensor || (!data && bytes != 0)) return 0;
    if (offset > tensor->bytes || bytes > tensor->bytes - offset) return 0;
    if (bytes == 0) return 1;
    void *dst = tensor->host ?
                (void *)(tensor_host_u8(tensor) + offset) :
                (void *)(tensor_u8(tensor) + offset);
    if (rocm_trace()) {
        fprintf(stderr,
                "ds4: ROCm tensor write dst=%p host_base=%p dev_base=%p tensor_off=%llu off=%llu bytes=%llu src=%p\n",
                dst,
                tensor->host_ptr,
                tensor->dev_ptr,
                (unsigned long long)tensor->offset,
                (unsigned long long)offset,
                (unsigned long long)bytes,
                data);
    }
    hipError_t rc = hipSuccess;
    if (tensor->host) {
        memcpy(dst, data, (size_t)bytes);
    } else {
        rc = hipMemcpy(dst, data, (size_t)bytes, hipMemcpyHostToDevice);
    }
    if (rocm_trace()) {
        fprintf(stderr, "ds4: ROCm tensor write rc=%d %s\n", (int)rc, hipGetErrorString(rc));
    }
    return rocm_ok(rc, "tensor write");
}

int ds4_metal_tensor_read(const ds4_metal_tensor *tensor, uint64_t offset, void *data, uint64_t bytes) {
    if (!tensor || (!data && bytes != 0)) return 0;
    if (offset > tensor->bytes || bytes > tensor->bytes - offset) return 0;
    if (bytes == 0) return 1;
    const void *src = tensor->host ?
                      (const void *)(tensor_host_u8_const(tensor) + offset) :
                      (const void *)(tensor_u8_const(tensor) + offset);
    if (tensor->host) {
        memcpy(data, src, (size_t)bytes);
        return 1;
    }
    return rocm_ok(hipMemcpy(data, src, (size_t)bytes, hipMemcpyDeviceToHost), "tensor read");
}

int ds4_metal_tensor_copy(ds4_metal_tensor *dst, uint64_t dst_offset,
                          const ds4_metal_tensor *src, uint64_t src_offset,
                          uint64_t bytes) {
    if (!dst || !src) return 0;
    if (dst_offset > dst->bytes || bytes > dst->bytes - dst_offset) return 0;
    if (src_offset > src->bytes || bytes > src->bytes - src_offset) return 0;
    if (bytes == 0) return 1;
    if (dst->host && src->host) {
        void *dp = tensor_host_u8(dst) + dst_offset;
        const void *sp = tensor_host_u8_const(src) + src_offset;
        memmove(dp, sp, (size_t)bytes);
        return 1;
    }
    hipMemcpyKind kind = src->host ? hipMemcpyHostToDevice :
                         dst->host ? hipMemcpyDeviceToHost : hipMemcpyDeviceToDevice;
    void *dp = dst->host ?
               (void *)(tensor_host_u8(dst) + dst_offset) :
               (void *)(tensor_u8(dst) + dst_offset);
    const void *sp = src->host ?
                     (const void *)(tensor_host_u8_const(src) + src_offset) :
                     (const void *)(tensor_u8_const(src) + src_offset);
    return rocm_ok(hipMemcpy(dp, sp, (size_t)bytes, kind), "tensor copy");
}

int ds4_metal_begin_commands(void) {
    if (!g_initialized && !ds4_metal_init()) return 0;
    if (g_batch_open) return 0;
    g_batch_open = 1;
    return 1;
}

int ds4_metal_flush_commands(void) {
    if (!g_batch_open) return 0;
    return rocm_ok(hipDeviceSynchronize(), "command flush");
}

int ds4_metal_end_commands(void) {
    if (!g_batch_open) return 0;
    g_batch_open = 0;
    return rocm_ok(hipDeviceSynchronize(), "command completion");
}

int ds4_metal_synchronize(void) {
    if (!g_initialized && !ds4_metal_init()) return 0;
    return rocm_ok(hipDeviceSynchronize(), "synchronize");
}

int ds4_metal_set_model_map(const void *model_map, uint64_t model_size) {
    (void)model_map;
    (void)model_size;
    return 1;
}

int ds4_metal_set_model_map_range(const void *model_map, uint64_t model_size,
                                  uint64_t map_offset, uint64_t map_size) {
    (void)model_map;
    (void)model_size;
    (void)map_offset;
    (void)map_size;
    return 1;
}

void ds4_metal_set_quality(bool quality) {
    g_quality = quality;
}

void ds4_metal_print_memory_report(const char *label) {
    fprintf(stderr,
            "ds4: ROCm memory %s live %.2f MiB peak %.2f MiB quality=%s\n",
            label ? label : "",
            (double)g_live_bytes / (1024.0 * 1024.0),
            (double)g_peak_bytes / (1024.0 * 1024.0),
            g_quality ? "on" : "off");
}

int ds4_metal_embed_token_hc_tensor(
        ds4_metal_tensor *out_hc,
        const void       *model_map,
        uint64_t          model_size,
        uint64_t          weight_offset,
        uint32_t          n_vocab,
        uint32_t          token,
        uint32_t          n_embd,
        uint32_t          n_hc) {
    if (!g_initialized && !ds4_metal_init()) return 0;
    if (!out_hc || !model_map || n_vocab == 0 || token >= n_vocab || n_embd == 0 || n_hc == 0) return 0;
    const uint64_t out_bytes = (uint64_t)n_embd * n_hc * sizeof(float);
    const uint64_t row_bytes = (uint64_t)n_embd * sizeof(uint16_t);
    const uint64_t weight_bytes = (uint64_t)n_vocab * row_bytes;
    if (out_hc->bytes < out_bytes || weight_offset > model_size || weight_bytes > model_size - weight_offset) return 0;

    uint16_t *row_host = NULL;
    uint16_t *row_dev = NULL;
    if (!rocm_ok(hipHostMalloc(&row_host, (size_t)row_bytes, hipHostMallocMapped),
                 "embedding row allocation")) {
        return 0;
    }
    if (!rocm_ok(hipHostGetDevicePointer((void **)&row_dev, row_host, 0),
                 "embedding row device pointer")) {
        (void)hipHostFree(row_host);
        return 0;
    }
    const uint8_t *src = (const uint8_t *)model_map + weight_offset + (uint64_t)token * row_bytes;
    memcpy(row_host, src, (size_t)row_bytes);
    int ok = 1;
    if (ok) {
        const uint64_t total = (uint64_t)n_embd * n_hc;
        const dim3 block(256);
        const dim3 grid((unsigned)((total + block.x - 1u) / block.x));
        hipLaunchKernelGGL(rocm_embed_token_hc_kernel,
                           grid,
                           block,
                           0,
                           0,
                           (float *)tensor_u8(out_hc),
                           row_dev,
                           n_embd,
                           n_hc);
        ok = rocm_ok(hipGetLastError(), "embed token launch") &&
             rocm_ok(hipDeviceSynchronize(), "embed token completion");
    }
    (void)hipHostFree(row_host);
    return ok;
}

int ds4_metal_embed_tokens_hc_tensor(
        ds4_metal_tensor       *out_hc,
        const ds4_metal_tensor *tokens,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n_vocab,
        uint32_t                n_tokens,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (!g_initialized && !ds4_metal_init()) return 0;
    if (!out_hc || !tokens || !model_map || n_vocab == 0 || n_tokens == 0 || n_embd == 0 || n_hc == 0) return 0;
    const uint64_t out_bytes = (uint64_t)n_tokens * n_hc * n_embd * sizeof(float);
    const uint64_t token_bytes = (uint64_t)n_tokens * sizeof(int32_t);
    const uint64_t row_bytes = (uint64_t)n_embd * sizeof(uint16_t);
    const uint64_t rows_bytes = (uint64_t)n_tokens * row_bytes;
    const uint64_t weight_bytes = (uint64_t)n_vocab * row_bytes;
    if (out_hc->bytes < out_bytes || tokens->bytes < token_bytes ||
        weight_offset > model_size || weight_bytes > model_size - weight_offset) return 0;

    int32_t *host_tokens = (int32_t *)malloc((size_t)token_bytes);
    uint16_t *host_rows = NULL;
    uint16_t *rows_dev = NULL;
    if (!rocm_ok(hipHostMalloc(&host_rows, (size_t)rows_bytes, hipHostMallocMapped),
                 "embedding rows allocation")) {
        free(host_tokens);
        return 0;
    }
    if (!rocm_ok(hipHostGetDevicePointer((void **)&rows_dev, host_rows, 0),
                 "embedding rows device pointer")) {
        (void)hipHostFree(host_rows);
        free(host_tokens);
        return 0;
    }
    if (!host_tokens || !host_rows) {
        free(host_tokens);
        if (host_rows) (void)hipHostFree(host_rows);
        return 0;
    }
    int ok = ds4_metal_tensor_read(tokens, 0, host_tokens, token_bytes);
    const uint8_t *weights = (const uint8_t *)model_map + weight_offset;
    for (uint32_t i = 0; ok && i < n_tokens; i++) {
        if (host_tokens[i] < 0 || (uint32_t)host_tokens[i] >= n_vocab) {
            ok = 0;
            break;
        }
        memcpy(host_rows + (uint64_t)i * n_embd,
               weights + (uint64_t)(uint32_t)host_tokens[i] * row_bytes,
               (size_t)row_bytes);
    }
    if (ok) {
        const uint64_t total = (uint64_t)n_tokens * n_hc * n_embd;
        const dim3 block(256);
        const dim3 grid((unsigned)((total + block.x - 1u) / block.x));
        hipLaunchKernelGGL(rocm_embed_tokens_hc_kernel,
                           grid,
                           block,
                           0,
                           0,
                           (float *)tensor_u8(out_hc),
                           rows_dev,
                           n_tokens,
                           n_embd,
                           n_hc);
        ok = rocm_ok(hipGetLastError(), "embed tokens launch") &&
             rocm_ok(hipDeviceSynchronize(), "embed tokens completion");
    }
    if (host_rows) (void)hipHostFree(host_rows);
    free(host_tokens);
    return ok;
}

int ds4_metal_repeat_hc_tensor(
        ds4_metal_tensor       *out,
        const ds4_metal_tensor *row,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (!g_initialized && !ds4_metal_init()) return 0;
    if (!out || !row || n_embd == 0 || n_hc == 0) return 0;
    const uint64_t row_bytes = (uint64_t)n_embd * sizeof(float);
    const uint64_t out_bytes = row_bytes * n_hc;
    if (row->bytes < row_bytes || out->bytes < out_bytes) return 0;

    const uint64_t total = (uint64_t)n_embd * n_hc;
    const dim3 block(256);
    const dim3 grid((unsigned)((total + block.x - 1u) / block.x));
    hipLaunchKernelGGL(rocm_repeat_hc_kernel,
                       grid,
                       block,
                       0,
                       0,
                       (float *)tensor_u8(out),
                       (const float *)tensor_u8_const(row),
                       n_embd,
                       n_hc);
    return rocm_ok(hipGetLastError(), "repeat launch") &&
           rocm_ok(hipDeviceSynchronize(), "repeat completion");
}

int ds4_metal_rms_norm_plain_tensor(
        ds4_metal_tensor       *out,
        const ds4_metal_tensor *x,
        uint32_t                n,
        float                   eps) {
    return ds4_metal_rms_norm_plain_rows_tensor(out, x, n, 1, eps);
}

int ds4_metal_rms_norm_plain_rows_tensor(
        ds4_metal_tensor       *out,
        const ds4_metal_tensor *x,
        uint32_t                n,
        uint32_t                rows,
        float                   eps) {
    if (!g_initialized && !ds4_metal_init()) return 0;
    if (!out || !x || n == 0 || rows == 0 || (n & 3u) != 0) return 0;
    const uint64_t bytes = (uint64_t)n * rows * sizeof(float);
    if (out->bytes < bytes || x->bytes < bytes) return 0;

    unsigned threads = 256;
    while (threads > 1 && threads / 2 >= n) threads >>= 1;
    const dim3 block(threads);
    const dim3 grid(rows);
    hipLaunchKernelGGL(rocm_rms_norm_plain_kernel,
                       grid,
                       block,
                       threads * sizeof(float),
                       0,
                       (float *)tensor_u8(out),
                       (const float *)tensor_u8_const(x),
                       n,
                       eps);
    return rocm_ok(hipGetLastError(), "plain RMS norm launch") &&
           rocm_ok(hipDeviceSynchronize(), "plain RMS norm completion");
}

int ds4_metal_swiglu_tensor(
        ds4_metal_tensor       *out,
        const ds4_metal_tensor *gate,
        const ds4_metal_tensor *up,
        uint32_t                n,
        float                   clamp,
        float                   weight) {
    if (!g_initialized && !ds4_metal_init()) return 0;
    if (!out || !gate || !up || n == 0) return 0;
    if (fabsf(clamp) > 1.0e-12f || fabsf(weight - 1.0f) > 1.0e-12f) {
        fprintf(stderr, "ds4: ROCm SwiGLU kernel does not support clamp/weight\n");
        return 0;
    }
    const uint64_t bytes = (uint64_t)n * sizeof(float);
    if (out->bytes < bytes || gate->bytes < bytes || up->bytes < bytes) return 0;

    const dim3 block(256);
    const dim3 grid((n + block.x - 1u) / block.x);
    hipLaunchKernelGGL(rocm_swiglu_kernel,
                       grid,
                       block,
                       0,
                       0,
                       (float *)tensor_u8(out),
                       (const float *)tensor_u8_const(gate),
                       (const float *)tensor_u8_const(up),
                       n);
    return rocm_ok(hipGetLastError(), "SwiGLU launch") &&
           rocm_ok(hipDeviceSynchronize(), "SwiGLU completion");
}

int ds4_metal_add_tensor(
        ds4_metal_tensor       *out,
        const ds4_metal_tensor *a,
        const ds4_metal_tensor *b,
        uint32_t                n) {
    if (!g_initialized && !ds4_metal_init()) return 0;
    if (!out || !a || !b || n == 0) return 0;
    const uint64_t bytes = (uint64_t)n * sizeof(float);
    if (out->bytes < bytes || a->bytes < bytes || b->bytes < bytes) return 0;

    const dim3 block(256);
    const dim3 grid((n + block.x - 1u) / block.x);
    hipLaunchKernelGGL(rocm_add_kernel,
                       grid,
                       block,
                       0,
                       0,
                       (float *)tensor_u8(out),
                       (const float *)tensor_u8_const(a),
                       (const float *)tensor_u8_const(b),
                       n);
    return rocm_ok(hipGetLastError(), "add launch") &&
           rocm_ok(hipDeviceSynchronize(), "add completion");
}

int ds4_metal_matmul_f16_tensor(
        ds4_metal_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_metal_tensor *x,
        uint64_t                n_tok) {
    if (!g_initialized && !ds4_metal_init()) return 0;
    if (!out || !model_map || !x || in_dim == 0 || out_dim == 0 || n_tok == 0) return 0;
    const uint64_t x_bytes = n_tok * in_dim * sizeof(float);
    const uint64_t out_bytes = n_tok * out_dim * sizeof(float);
    const uint64_t weight_bytes = in_dim * out_dim * sizeof(uint16_t);
    if (x->bytes < x_bytes || out->bytes < out_bytes ||
        weight_offset > model_size || weight_bytes > model_size - weight_offset) {
        return 0;
    }

    void *w_host = NULL;
    void *w_dev = NULL;
    const void *src = (const uint8_t *)model_map + weight_offset;
    if (!rocm_upload_mapped(src, weight_bytes, &w_host, &w_dev, "F16 weight upload")) return 0;

    unsigned threads = 256;
    while (threads > 1 && threads / 2 >= in_dim) threads >>= 1;
    const dim3 block(threads);
    const dim3 grid((unsigned)out_dim, (unsigned)n_tok);
    hipLaunchKernelGGL(rocm_matmul_f16_kernel,
                       grid,
                       block,
                       threads * sizeof(float),
                       0,
                       (float *)tensor_u8(out),
                       (const uint16_t *)w_dev,
                       (const float *)tensor_u8_const(x),
                       in_dim,
                       out_dim);
    int ok = rocm_ok(hipGetLastError(), "F16 matmul launch") &&
             rocm_ok(hipDeviceSynchronize(), "F16 matmul completion");
    (void)hipHostFree(w_host);
    return ok;
}

int ds4_metal_matmul_f16_pair_tensor(
        ds4_metal_tensor       *out_a,
        ds4_metal_tensor       *out_b,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_a_offset,
        uint64_t                weight_b_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_metal_tensor *x,
        uint64_t                n_tok) {
    return ds4_metal_matmul_f16_tensor(out_a, model_map, model_size,
                                       weight_a_offset, in_dim, out_dim, x, n_tok) &&
           ds4_metal_matmul_f16_tensor(out_b, model_map, model_size,
                                       weight_b_offset, in_dim, out_dim, x, n_tok);
}

int ds4_metal_matmul_f32_tensor(
        ds4_metal_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_metal_tensor *x,
        uint64_t                n_tok) {
    if (!g_initialized && !ds4_metal_init()) return 0;
    if (!out || !model_map || !x || in_dim == 0 || out_dim == 0 || n_tok == 0) return 0;
    const uint64_t x_bytes = n_tok * in_dim * sizeof(float);
    const uint64_t out_bytes = n_tok * out_dim * sizeof(float);
    const uint64_t weight_bytes = in_dim * out_dim * sizeof(float);
    if (x->bytes < x_bytes || out->bytes < out_bytes ||
        weight_offset > model_size || weight_bytes > model_size - weight_offset) {
        return 0;
    }

    void *w_host = NULL;
    void *w_dev = NULL;
    const void *src = (const uint8_t *)model_map + weight_offset;
    if (!rocm_upload_mapped(src, weight_bytes, &w_host, &w_dev, "F32 weight upload")) return 0;

    unsigned threads = 256;
    while (threads > 1 && threads / 2 >= in_dim) threads >>= 1;
    const dim3 block(threads);
    const dim3 grid((unsigned)out_dim, (unsigned)n_tok);
    hipLaunchKernelGGL(rocm_matmul_f32_kernel,
                       grid,
                       block,
                       threads * sizeof(float),
                       0,
                       (float *)tensor_u8(out),
                       (const float *)w_dev,
                       (const float *)tensor_u8_const(x),
                       in_dim,
                       out_dim);
    int ok = rocm_ok(hipGetLastError(), "F32 matmul launch") &&
             rocm_ok(hipDeviceSynchronize(), "F32 matmul completion");
    (void)hipHostFree(w_host);
    return ok;
}

int ds4_metal_matmul_q8_0_tensor(
        ds4_metal_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_metal_tensor *x,
        uint64_t                n_tok) {
    if (!g_initialized && !ds4_metal_init()) return 0;
    if (!out || !model_map || !x || in_dim == 0 || out_dim == 0 || n_tok == 0) return 0;
    const uint64_t x_bytes = n_tok * in_dim * sizeof(float);
    const uint64_t out_bytes = n_tok * out_dim * sizeof(float);
    const uint64_t row_bytes = ((in_dim + 31u) / 32u) * 34u;
    const uint64_t weight_bytes = row_bytes * out_dim;
    if (x->bytes < x_bytes || out->bytes < out_bytes ||
        weight_offset > model_size || weight_bytes > model_size - weight_offset) {
        return 0;
    }

    void *w_host = NULL;
    void *w_dev = NULL;
    const void *src = (const uint8_t *)model_map + weight_offset;
    if (!rocm_upload_mapped(src, weight_bytes, &w_host, &w_dev, "Q8_0 weight upload")) return 0;

    const dim3 block(256);
    const dim3 grid((unsigned)out_dim, (unsigned)n_tok);
    hipLaunchKernelGGL(rocm_matmul_q8_0_kernel,
                       grid,
                       block,
                       block.x * sizeof(float),
                       0,
                       (float *)tensor_u8(out),
                       (const uint8_t *)w_dev,
                       (const float *)tensor_u8_const(x),
                       in_dim,
                       out_dim,
                       row_bytes);
    int ok = rocm_ok(hipGetLastError(), "Q8_0 matmul launch") &&
             rocm_ok(hipDeviceSynchronize(), "Q8_0 matmul completion");
    (void)hipHostFree(w_host);
    return ok;
}

int ds4_metal_rms_norm_weight_tensor(
        ds4_metal_tensor       *out,
        const ds4_metal_tensor *x,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        float                   eps) {
    return ds4_metal_rms_norm_weight_rows_tensor(out, x, model_map, model_size,
                                                 weight_offset, n, 1, eps);
}

int ds4_metal_rms_norm_weight_rows_tensor(
        ds4_metal_tensor       *out,
        const ds4_metal_tensor *x,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        uint32_t                rows,
        float                   eps) {
    if (!g_initialized && !ds4_metal_init()) return 0;
    if (!out || !x || !model_map || n == 0 || rows == 0 || (n & 3u) != 0) return 0;
    const uint64_t bytes = (uint64_t)n * rows * sizeof(float);
    const uint64_t weight_bytes = (uint64_t)n * sizeof(float);
    if (out->bytes < bytes || x->bytes < bytes ||
        weight_offset > model_size || weight_bytes > model_size - weight_offset) {
        return 0;
    }

    void *w_host = NULL;
    void *w_dev = NULL;
    const void *src = (const uint8_t *)model_map + weight_offset;
    if (!rocm_upload_mapped(src, weight_bytes, &w_host, &w_dev, "RMS weight upload")) return 0;

    unsigned threads = 256;
    while (threads > 1 && threads / 2 >= n) threads >>= 1;
    const dim3 block(threads);
    const dim3 grid(rows);
    hipLaunchKernelGGL(rocm_rms_norm_weight_kernel,
                       grid,
                       block,
                       threads * sizeof(float),
                       0,
                       (float *)tensor_u8(out),
                       (const float *)tensor_u8_const(x),
                       (const float *)w_dev,
                       n,
                       eps);
    int ok = rocm_ok(hipGetLastError(), "weighted RMS norm launch") &&
             rocm_ok(hipDeviceSynchronize(), "weighted RMS norm completion");
    (void)hipHostFree(w_host);
    return ok;
}

int ds4_metal_dsv4_qkv_rms_norm_rows_tensor(
        ds4_metal_tensor       *q_out,
        const ds4_metal_tensor *q,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                q_weight_offset,
        uint32_t                q_n,
        ds4_metal_tensor       *kv_out,
        const ds4_metal_tensor *kv,
        uint64_t                kv_weight_offset,
        uint32_t                kv_n,
        uint32_t                rows,
        float                   eps) {
    return ds4_metal_rms_norm_weight_rows_tensor(q_out, q, model_map, model_size,
                                                 q_weight_offset, q_n, rows, eps) &&
           ds4_metal_rms_norm_weight_rows_tensor(kv_out, kv, model_map, model_size,
                                                 kv_weight_offset, kv_n, rows, eps);
}

int ds4_metal_head_rms_norm_tensor(
        ds4_metal_tensor *x,
        uint32_t          n_tok,
        uint32_t          n_head,
        uint32_t          head_dim,
        float             eps) {
    if (!x || n_tok == 0 || n_head == 0 || head_dim == 0) return 0;
    return ds4_metal_rms_norm_plain_rows_tensor(x, x, head_dim, n_tok * n_head, eps);
}

int ds4_metal_hc_split_weighted_sum_norm_tensor(
        ds4_metal_tensor       *out,
        ds4_metal_tensor       *norm_out,
        ds4_metal_tensor       *split,
        const ds4_metal_tensor *mix,
        const ds4_metal_tensor *residual_hc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint64_t                norm_weight_offset,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps,
        float                   norm_eps) {
    if (!g_initialized && !ds4_metal_init()) return 0;
    if (!out || !norm_out || !split || !mix || !residual_hc || !model_map ||
        n_embd == 0 || n_hc != 4) {
        return 0;
    }
    const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    const uint64_t mix_bytes = mix_hc * sizeof(float);
    const uint64_t out_row_bytes = (uint64_t)n_embd * sizeof(float);
    const uint64_t residual_row_bytes = (uint64_t)n_hc * out_row_bytes;
    const uint64_t rows = out->bytes / out_row_bytes;
    if (rows == 0 || out->bytes < rows * out_row_bytes ||
        norm_out->bytes < rows * out_row_bytes ||
        mix->bytes < rows * mix_bytes ||
        split->bytes < rows * mix_bytes ||
        residual_hc->bytes < rows * residual_row_bytes) {
        return 0;
    }
    if (scale_offset > model_size || 3ull * sizeof(float) > model_size - scale_offset ||
        base_offset > model_size || mix_bytes > model_size - base_offset ||
        norm_weight_offset > model_size || out_row_bytes > model_size - norm_weight_offset) {
        return 0;
    }

    void *scale_host = NULL;
    void *scale_dev = NULL;
    void *base_host = NULL;
    void *base_dev = NULL;
    void *norm_host = NULL;
    void *norm_dev = NULL;
    int ok = rocm_upload_mapped((const uint8_t *)model_map + scale_offset,
                                3ull * sizeof(float),
                                &scale_host,
                                &scale_dev,
                                "HC scale upload") &&
             rocm_upload_mapped((const uint8_t *)model_map + base_offset,
                                mix_bytes,
                                &base_host,
                                &base_dev,
                                "HC base upload") &&
             rocm_upload_mapped((const uint8_t *)model_map + norm_weight_offset,
                                out_row_bytes,
                                &norm_host,
                                &norm_dev,
                                "HC norm weight upload");
    if (!ok) {
        if (scale_host) (void)hipHostFree(scale_host);
        if (base_host) (void)hipHostFree(base_host);
        if (norm_host) (void)hipHostFree(norm_host);
        return 0;
    }

    unsigned threads = 256;
    while (threads > 1 && threads / 2 >= n_embd) threads >>= 1;
    const dim3 block(threads);
    const dim3 grid((unsigned)rows);
    hipLaunchKernelGGL(rocm_hc_split_weighted_sum_norm_kernel,
                       grid,
                       block,
                       (4u + threads) * sizeof(float),
                       0,
                       (float *)tensor_u8(out),
                       (float *)tensor_u8(norm_out),
                       (float *)tensor_u8(split),
                       (const float *)tensor_u8_const(mix),
                       (const float *)tensor_u8_const(residual_hc),
                       (const float *)scale_dev,
                       (const float *)base_dev,
                       (const float *)norm_dev,
                       n_embd,
                       n_hc,
                       sinkhorn_iters,
                       eps,
                       norm_eps);
    ok = rocm_ok(hipGetLastError(), "HC split/sum/norm launch") &&
         rocm_ok(hipDeviceSynchronize(), "HC split/sum/norm completion");
    (void)hipHostFree(scale_host);
    (void)hipHostFree(base_host);
    (void)hipHostFree(norm_host);
    return ok;
}

int ds4_metal_rope_tail_tensor(
        ds4_metal_tensor *x,
        uint32_t          n_tok,
        uint32_t          n_head,
        uint32_t          head_dim,
        uint32_t          n_rot,
        uint32_t          pos0,
        uint32_t          n_ctx_orig,
        bool              inverse,
        float             freq_base,
        float             freq_scale,
        float             ext_factor,
        float             attn_factor,
        float             beta_fast,
        float             beta_slow) {
    if (!g_initialized && !ds4_metal_init()) return 0;
    if (!x || n_tok == 0 || n_head == 0 || head_dim == 0 || n_rot > head_dim || (n_rot & 1u) != 0) {
        return 0;
    }
    if (n_rot == 0) return 1;
    const uint64_t bytes = (uint64_t)n_tok * n_head * head_dim * sizeof(float);
    if (x->bytes < bytes) return 0;

    const uint64_t total = (uint64_t)n_tok * n_head * (n_rot / 2u);
    const dim3 block(256);
    const dim3 grid((unsigned)((total + block.x - 1u) / block.x));
    hipLaunchKernelGGL(rocm_rope_tail_kernel,
                       grid,
                       block,
                       0,
                       0,
                       (float *)tensor_u8(x),
                       n_tok,
                       n_head,
                       head_dim,
                       n_rot,
                       pos0,
                       n_ctx_orig,
                       inverse ? 1 : 0,
                       freq_base,
                       freq_scale,
                       ext_factor,
                       attn_factor,
                       beta_fast,
                       beta_slow);
    return rocm_ok(hipGetLastError(), "RoPE tail launch") &&
           rocm_ok(hipDeviceSynchronize(), "RoPE tail completion");
}

int ds4_metal_dsv4_fp8_kv_quantize_tensor(
        ds4_metal_tensor *x,
        uint32_t          n_tok,
        uint32_t          head_dim,
        uint32_t          n_rot) {
    if (!g_initialized && !ds4_metal_init()) return 0;
    if (!x || n_tok == 0 || head_dim == 0 || n_rot > head_dim) return 0;
    if (n_rot == head_dim) return 1;
    const uint64_t bytes = (uint64_t)n_tok * head_dim * sizeof(float);
    if (x->bytes < bytes) return 0;

    const dim3 block(64);
    const dim3 grid(n_tok);
    hipLaunchKernelGGL(rocm_fp8_kv_quantize_kernel,
                       grid,
                       block,
                       0,
                       0,
                       (float *)tensor_u8(x),
                       n_tok,
                       head_dim,
                       n_rot);
    return rocm_ok(hipGetLastError(), "FP8 KV quantize launch") &&
           rocm_ok(hipDeviceSynchronize(), "FP8 KV quantize completion");
}

/* =========================================================================
 * HC kernels.
 * ========================================================================= */

__global__ static void rocm_hc_weighted_sum_kernel(float *out,
                                                   const float *x,
                                                   const float *w,
                                                   uint32_t n_embd,
                                                   uint32_t n_hc,
                                                   uint32_t n_tokens,
                                                   uint64_t w_row_stride_floats) {
    const uint64_t total = (uint64_t)n_embd * n_tokens;
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;
    const uint32_t d = (uint32_t)(gid % n_embd);
    const uint32_t t = (uint32_t)(gid / n_embd);
    const float *xrow = x + (uint64_t)t * n_hc * n_embd;
    const float *wrow = w + (uint64_t)t * w_row_stride_floats;
    float acc = 0.0f;
    for (uint32_t h = 0; h < n_hc; h++) acc += xrow[(uint64_t)h * n_embd + d] * wrow[h];
    out[(uint64_t)t * n_embd + d] = acc;
}

static int rocm_hc_weighted_sum_strided(ds4_metal_tensor       *out,
                                        const ds4_metal_tensor *residual_hc,
                                        const ds4_metal_tensor *weights,
                                        uint64_t                weight_offset_bytes,
                                        uint64_t                weight_row_stride,
                                        uint32_t                n_embd,
                                        uint32_t                n_hc,
                                        const char             *label) {
    if (!g_initialized && !ds4_metal_init()) return 0;
    if (!out || !residual_hc || !weights || n_embd == 0 || n_hc == 0) return 0;
    const uint64_t out_row_bytes = (uint64_t)n_embd * sizeof(float);
    if (out->bytes < out_row_bytes || out->bytes % out_row_bytes) return 0;
    const uint64_t n_tokens = out->bytes / out_row_bytes;
    const uint64_t hc_bytes = n_tokens * (uint64_t)n_hc * out_row_bytes;
    const uint64_t w_last = weight_offset_bytes + (n_tokens - 1u) * weight_row_stride + (uint64_t)n_hc * sizeof(float);
    if (residual_hc->bytes < hc_bytes || weights->bytes < w_last) return 0;
    if ((weight_row_stride % sizeof(float)) != 0 || (weight_offset_bytes % sizeof(float)) != 0) return 0;

    const uint64_t total = (uint64_t)n_embd * n_tokens;
    const dim3 block(256);
    const dim3 grid((unsigned)((total + block.x - 1u) / block.x));
    hipLaunchKernelGGL(rocm_hc_weighted_sum_kernel,
                       grid, block, 0, 0,
                       (float *)tensor_u8(out),
                       (const float *)tensor_u8_const(residual_hc),
                       (const float *)(tensor_u8_const(weights) + weight_offset_bytes),
                       n_embd, n_hc, (uint32_t)n_tokens,
                       weight_row_stride / sizeof(float));
    return rocm_ok(hipGetLastError(), label) &&
           rocm_ok(hipDeviceSynchronize(), label);
}

int ds4_metal_hc_weighted_sum_tensor(
        ds4_metal_tensor       *out,
        const ds4_metal_tensor *residual_hc,
        const ds4_metal_tensor *weights,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    return rocm_hc_weighted_sum_strided(out, residual_hc, weights, 0,
                                        (uint64_t)n_hc * sizeof(float),
                                        n_embd, n_hc, "HC weighted sum");
}

int ds4_metal_hc_weighted_sum_split_tensor(
        ds4_metal_tensor       *out,
        const ds4_metal_tensor *residual_hc,
        const ds4_metal_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    return rocm_hc_weighted_sum_strided(out, residual_hc, split, 0,
                                        mix_hc * sizeof(float),
                                        n_embd, n_hc, "HC weighted sum split");
}

__global__ static void rocm_hc_split_sinkhorn_kernel(float *out,
                                                     const float *mix,
                                                     const float *scale,
                                                     const float *base,
                                                     uint32_t n_hc,
                                                     uint32_t sinkhorn_iters,
                                                     float eps,
                                                     uint32_t n_rows,
                                                     uint32_t mix_hc) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    const float *mr = mix + (uint64_t)row * mix_hc;
    float *out_r = out + (uint64_t)row * mix_hc;

    const float pre_s = scale[0], post_s = scale[1], comb_s = scale[2];
    for (uint32_t h = 0; h < n_hc; h++) {
        const float z = mr[h] * pre_s + base[h];
        out_r[h] = 1.0f / (1.0f + expf(-z)) + eps;
    }
    for (uint32_t h = 0; h < n_hc; h++) {
        const float z = mr[n_hc + h] * post_s + base[n_hc + h];
        out_r[n_hc + h] = 2.0f / (1.0f + expf(-z));
    }
    /* Sinkhorn over n_hc x n_hc; use stack matrix sized for n_hc <= 16. */
    float c[16 * 16];
    for (uint32_t i = 0; i < n_hc; i++) {
        float row_max = -3.4028234663852886e38f;
        for (uint32_t j = 0; j < n_hc; j++) {
            const uint32_t idx = i * n_hc + j;
            const float v = mr[2 * n_hc + idx] * comb_s + base[2 * n_hc + idx];
            c[idx] = v;
            if (v > row_max) row_max = v;
        }
        float row_sum = 0.0f;
        for (uint32_t j = 0; j < n_hc; j++) {
            const uint32_t idx = i * n_hc + j;
            c[idx] = expf(c[idx] - row_max);
            row_sum += c[idx];
        }
        const float inv = 1.0f / row_sum;
        for (uint32_t j = 0; j < n_hc; j++) c[i * n_hc + j] = c[i * n_hc + j] * inv + eps;
    }
    /* Sinkhorn iters: column-normalise then row-normalise n_hc-1 more times. */
    for (uint32_t j = 0; j < n_hc; j++) {
        float col_sum = eps;
        for (uint32_t i = 0; i < n_hc; i++) col_sum += c[i * n_hc + j];
        const float inv = 1.0f / col_sum;
        for (uint32_t i = 0; i < n_hc; i++) c[i * n_hc + j] *= inv;
    }
    for (uint32_t iter = 1; iter < sinkhorn_iters; iter++) {
        for (uint32_t i = 0; i < n_hc; i++) {
            float row_sum = eps;
            for (uint32_t j = 0; j < n_hc; j++) row_sum += c[i * n_hc + j];
            const float inv = 1.0f / row_sum;
            for (uint32_t j = 0; j < n_hc; j++) c[i * n_hc + j] *= inv;
        }
        for (uint32_t j = 0; j < n_hc; j++) {
            float col_sum = eps;
            for (uint32_t i = 0; i < n_hc; i++) col_sum += c[i * n_hc + j];
            const float inv = 1.0f / col_sum;
            for (uint32_t i = 0; i < n_hc; i++) c[i * n_hc + j] *= inv;
        }
    }
    for (uint32_t i = 0; i < n_hc * n_hc; i++) out_r[2 * n_hc + i] = c[i];
}

int ds4_metal_hc_split_sinkhorn_tensor(
        ds4_metal_tensor       *out,
        const ds4_metal_tensor *mix,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps) {
    if (!g_initialized && !ds4_metal_init()) return 0;
    if (!out || !mix || !model_map || n_hc == 0 || n_hc > 16) return 0;
    const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    const uint64_t row_bytes = mix_hc * sizeof(float);
    if (out->bytes < row_bytes || out->bytes % row_bytes) return 0;
    const uint64_t rows = out->bytes / row_bytes;
    if (mix->bytes < rows * row_bytes) return 0;
    if (scale_offset > model_size || 3ull * sizeof(float) > model_size - scale_offset) return 0;
    if (base_offset > model_size || row_bytes > model_size - base_offset) return 0;

    void *sh = NULL, *sd = NULL, *bh = NULL, *bd = NULL;
    int ok = rocm_upload_mapped((const uint8_t *)model_map + scale_offset, 3ull * sizeof(float),
                                &sh, &sd, "sinkhorn scale upload") &&
             rocm_upload_mapped((const uint8_t *)model_map + base_offset, row_bytes,
                                &bh, &bd, "sinkhorn base upload");
    if (!ok) {
        if (sh) (void)hipHostFree(sh);
        if (bh) (void)hipHostFree(bh);
        return 0;
    }

    const dim3 block(64);
    const dim3 grid((unsigned)((rows + block.x - 1u) / block.x));
    hipLaunchKernelGGL(rocm_hc_split_sinkhorn_kernel, grid, block, 0, 0,
                       (float *)tensor_u8(out),
                       (const float *)tensor_u8_const(mix),
                       (const float *)sd,
                       (const float *)bd,
                       n_hc, sinkhorn_iters, eps, (uint32_t)rows, (uint32_t)mix_hc);
    ok = rocm_ok(hipGetLastError(), "HC sinkhorn launch") &&
         rocm_ok(hipDeviceSynchronize(), "HC sinkhorn completion");
    (void)hipHostFree(sh);
    (void)hipHostFree(bh);
    return ok;
}

int ds4_metal_hc_split_weighted_sum_tensor(
        ds4_metal_tensor       *out,
        ds4_metal_tensor       *split,
        const ds4_metal_tensor *mix,
        const ds4_metal_tensor *residual_hc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps) {
    /* Compose split (sinkhorn) + weighted_sum_split.  The fused norm variant
     * already lives in a dedicated kernel; this two-stage path matches the
     * Metal reference fallback used by the graph driver. */
    return ds4_metal_hc_split_sinkhorn_tensor(split, mix, model_map, model_size,
                                              scale_offset, base_offset,
                                              n_hc, sinkhorn_iters, eps) &&
           ds4_metal_hc_weighted_sum_split_tensor(out, residual_hc, split, n_embd, n_hc);
}

__global__ static void rocm_output_hc_weights_kernel(float *out,
                                                     const float *pre,
                                                     const float *scale,
                                                     const float *base,
                                                     uint32_t n_hc,
                                                     uint32_t n_tokens,
                                                     float eps) {
    const uint64_t total = (uint64_t)n_hc * n_tokens;
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;
    const uint32_t h = (uint32_t)(gid % n_hc);
    const float z = pre[gid] * scale[0] + base[h];
    out[gid] = 1.0f / (1.0f + expf(-z)) + eps;
}

int ds4_metal_output_hc_weights_tensor(
        ds4_metal_tensor       *out,
        const ds4_metal_tensor *pre,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_hc,
        float                   eps) {
    if (!g_initialized && !ds4_metal_init()) return 0;
    if (!out || !pre || !model_map || n_hc == 0) return 0;
    const uint64_t row_bytes = (uint64_t)n_hc * sizeof(float);
    if (out->bytes < row_bytes || out->bytes % row_bytes) return 0;
    const uint64_t rows = out->bytes / row_bytes;
    if (pre->bytes < rows * row_bytes) return 0;
    if (scale_offset > model_size || sizeof(float) > model_size - scale_offset) return 0;
    if (base_offset > model_size || row_bytes > model_size - base_offset) return 0;

    void *sh = NULL, *sd = NULL, *bh = NULL, *bd = NULL;
    int ok = rocm_upload_mapped((const uint8_t *)model_map + scale_offset, sizeof(float),
                                &sh, &sd, "output_hc scale") &&
             rocm_upload_mapped((const uint8_t *)model_map + base_offset, row_bytes,
                                &bh, &bd, "output_hc base");
    if (!ok) { if (sh) (void)hipHostFree(sh); if (bh) (void)hipHostFree(bh); return 0; }

    const uint64_t total = rows * n_hc;
    const dim3 block(256);
    const dim3 grid((unsigned)((total + block.x - 1u) / block.x));
    hipLaunchKernelGGL(rocm_output_hc_weights_kernel, grid, block, 0, 0,
                       (float *)tensor_u8(out),
                       (const float *)tensor_u8_const(pre),
                       (const float *)sd, (const float *)bd,
                       n_hc, (uint32_t)rows, eps);
    ok = rocm_ok(hipGetLastError(), "output_hc launch") &&
         rocm_ok(hipDeviceSynchronize(), "output_hc completion");
    (void)hipHostFree(sh); (void)hipHostFree(bh);
    return ok;
}

/* HC=4 expand: out[d, dst_hc, t] = (block[d,t] (+ add[d,t])) * post[dst_hc,t]
 *                                + sum_src comb[dst_hc, src_hc, t] * residual[d, src_hc, t] */
__global__ static void rocm_hc_expand4_kernel(float *out,
                                              const float *block_out,
                                              const float *block_add,
                                              const float *residual,
                                              const float *post,
                                              const float *comb,
                                              uint32_t n_embd,
                                              uint32_t n_tokens,
                                              uint32_t post_stride_t,
                                              uint32_t comb_stride_t,
                                              int has_add) {
    const uint64_t total = (uint64_t)n_embd * n_tokens;
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;
    const uint32_t d = (uint32_t)(gid % n_embd);
    const uint32_t t = (uint32_t)(gid / n_embd);
    float bv = block_out[(uint64_t)t * n_embd + d];
    if (has_add) bv += block_add[(uint64_t)t * n_embd + d];
    const float *res = residual + (uint64_t)t * (uint64_t)4 * n_embd;
    const float r0 = res[0 * n_embd + d];
    const float r1 = res[1 * n_embd + d];
    const float r2 = res[2 * n_embd + d];
    const float r3 = res[3 * n_embd + d];
    const float *post_t = post + (uint64_t)t * post_stride_t;
    const float *comb_t = comb + (uint64_t)t * comb_stride_t;
    float *out_t = out + (uint64_t)t * (uint64_t)4 * n_embd;
    for (uint32_t dst_hc = 0; dst_hc < 4; dst_hc++) {
        float acc = bv * post_t[dst_hc];
        acc += comb_t[dst_hc * 4 + 0] * r0;
        acc += comb_t[dst_hc * 4 + 1] * r1;
        acc += comb_t[dst_hc * 4 + 2] * r2;
        acc += comb_t[dst_hc * 4 + 3] * r3;
        out_t[dst_hc * n_embd + d] = acc;
    }
}

static int rocm_hc_expand_dispatch(
        ds4_metal_tensor       *out_hc,
        const ds4_metal_tensor *block_out,
        const ds4_metal_tensor *block_add,
        const ds4_metal_tensor *residual_hc,
        const ds4_metal_tensor *post,
        const ds4_metal_tensor *comb,
        uint64_t                post_stride_floats,
        uint64_t                comb_stride_floats,
        uint32_t                n_embd,
        uint32_t                n_hc,
        int                     has_add,
        const char             *label) {
    if (!g_initialized && !ds4_metal_init()) return 0;
    if (!out_hc || !block_out || !residual_hc || !post || !comb || n_embd == 0 || n_hc != 4) {
        fprintf(stderr, "ds4: ROCm HC expand currently requires n_hc=4\n");
        return 0;
    }
    const uint64_t hc_row = (uint64_t)n_hc * n_embd * sizeof(float);
    if (out_hc->bytes < hc_row || out_hc->bytes % hc_row) return 0;
    const uint64_t n_tokens = out_hc->bytes / hc_row;
    const uint64_t block_bytes = n_tokens * (uint64_t)n_embd * sizeof(float);
    if (block_out->bytes < block_bytes || residual_hc->bytes < n_tokens * hc_row) return 0;
    if (has_add && (!block_add || block_add->bytes < block_bytes)) return 0;

    const uint64_t total = (uint64_t)n_embd * n_tokens;
    const dim3 block_d(256);
    const dim3 grid_d((unsigned)((total + block_d.x - 1u) / block_d.x));
    hipLaunchKernelGGL(rocm_hc_expand4_kernel, grid_d, block_d, 0, 0,
                       (float *)tensor_u8(out_hc),
                       (const float *)tensor_u8_const(block_out),
                       has_add ? (const float *)tensor_u8_const(block_add) : NULL,
                       (const float *)tensor_u8_const(residual_hc),
                       (const float *)tensor_u8_const(post),
                       (const float *)tensor_u8_const(comb),
                       n_embd, (uint32_t)n_tokens,
                       (uint32_t)post_stride_floats,
                       (uint32_t)comb_stride_floats,
                       has_add);
    return rocm_ok(hipGetLastError(), label) &&
           rocm_ok(hipDeviceSynchronize(), label);
}

int ds4_metal_hc_expand_tensor(
        ds4_metal_tensor       *out_hc,
        const ds4_metal_tensor *block_out,
        const ds4_metal_tensor *residual_hc,
        const ds4_metal_tensor *post,
        const ds4_metal_tensor *comb,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    /* post is [n_hc, n_tokens] dense, comb is [n_hc, n_hc, n_tokens] dense. */
    return rocm_hc_expand_dispatch(out_hc, block_out, NULL, residual_hc, post, comb,
                                   (uint64_t)n_hc, (uint64_t)n_hc * n_hc,
                                   n_embd, n_hc, 0, "HC expand");
}

int ds4_metal_hc_expand_split_tensor(
        ds4_metal_tensor       *out_hc,
        const ds4_metal_tensor *block_out,
        const ds4_metal_tensor *residual_hc,
        const ds4_metal_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    /* split layout per row: [pre[n_hc] | post[n_hc] | comb[n_hc*n_hc]] */
    if (!split) return 0;
    const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    ds4_metal_tensor *post_view = ds4_metal_tensor_view(split, (uint64_t)n_hc * sizeof(float),
                                                        split->bytes - (uint64_t)n_hc * sizeof(float));
    ds4_metal_tensor *comb_view = ds4_metal_tensor_view(split, 2ull * n_hc * sizeof(float),
                                                        split->bytes - 2ull * n_hc * sizeof(float));
    if (!post_view || !comb_view) {
        ds4_metal_tensor_free(post_view); ds4_metal_tensor_free(comb_view);
        return 0;
    }
    const int ok = rocm_hc_expand_dispatch(out_hc, block_out, NULL, residual_hc,
                                           post_view, comb_view,
                                           mix_hc, mix_hc, n_embd, n_hc, 0,
                                           "HC expand split");
    ds4_metal_tensor_free(post_view); ds4_metal_tensor_free(comb_view);
    return ok;
}

int ds4_metal_hc_expand_add_split_tensor(
        ds4_metal_tensor       *out_hc,
        const ds4_metal_tensor *block_out,
        const ds4_metal_tensor *block_add,
        const ds4_metal_tensor *residual_hc,
        const ds4_metal_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (!split) return 0;
    const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    ds4_metal_tensor *post_view = ds4_metal_tensor_view(split, (uint64_t)n_hc * sizeof(float),
                                                        split->bytes - (uint64_t)n_hc * sizeof(float));
    ds4_metal_tensor *comb_view = ds4_metal_tensor_view(split, 2ull * n_hc * sizeof(float),
                                                        split->bytes - 2ull * n_hc * sizeof(float));
    if (!post_view || !comb_view) {
        ds4_metal_tensor_free(post_view); ds4_metal_tensor_free(comb_view);
        return 0;
    }
    const int ok = rocm_hc_expand_dispatch(out_hc, block_out, block_add, residual_hc,
                                           post_view, comb_view,
                                           mix_hc, mix_hc, n_embd, n_hc, 1,
                                           "HC expand add split");
    ds4_metal_tensor_free(post_view); ds4_metal_tensor_free(comb_view);
    return ok;
}

/* =========================================================================
 * Raw KV / FP8 store kernels.
 * ========================================================================= */

__global__ static void rocm_store_raw_kv_kernel(float *raw,
                                                const float *kv,
                                                uint32_t head_dim) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= head_dim) return;
    raw[i] = (float)((__half)kv[i]);
}

int ds4_metal_store_raw_kv_tensor(
        ds4_metal_tensor       *raw_cache,
        const ds4_metal_tensor *kv,
        uint32_t                raw_cap,
        uint32_t                row,
        uint32_t                head_dim) {
    if (!g_initialized && !ds4_metal_init()) return 0;
    if (!raw_cache || !kv || raw_cap == 0 || row >= raw_cap || head_dim == 0) return 0;
    const uint64_t needed = ((uint64_t)row + 1u) * head_dim * sizeof(float);
    if (raw_cache->bytes < needed || kv->bytes < (uint64_t)head_dim * sizeof(float)) return 0;
    float *raw = (float *)tensor_u8(raw_cache) + (uint64_t)row * head_dim;
    const dim3 block(256);
    const dim3 grid((head_dim + block.x - 1u) / block.x);
    hipLaunchKernelGGL(rocm_store_raw_kv_kernel, grid, block, 0, 0,
                       raw, (const float *)tensor_u8_const(kv), head_dim);
    return rocm_ok(hipGetLastError(), "store raw kv launch") &&
           rocm_ok(hipDeviceSynchronize(), "store raw kv completion");
}

int ds4_metal_store_raw_kv_batch_tensor(
        ds4_metal_tensor       *raw_cache,
        const ds4_metal_tensor *kv,
        uint32_t                raw_cap,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                head_dim) {
    if (!g_initialized && !ds4_metal_init()) return 0;
    if (!raw_cache || !kv || head_dim == 0 || n_tokens == 0) return 0;
    if ((uint64_t)pos0 + n_tokens > raw_cap) return 0;
    if (kv->bytes < (uint64_t)n_tokens * head_dim * sizeof(float)) return 0;
    if (raw_cache->bytes < ((uint64_t)pos0 + n_tokens) * head_dim * sizeof(float)) return 0;
    /* Process row by row (small n_tokens path; prefill) */
    for (uint32_t i = 0; i < n_tokens; i++) {
        ds4_metal_tensor src = *kv;
        src.offset = kv->offset + (uint64_t)i * head_dim * sizeof(float);
        src.bytes = (uint64_t)head_dim * sizeof(float);
        if (!ds4_metal_store_raw_kv_tensor(raw_cache, &src, raw_cap, pos0 + i, head_dim)) return 0;
    }
    return 1;
}

__global__ static void rocm_kv_fp8_store_raw_kernel(float *kv,
                                                    float *raw,
                                                    uint32_t head_dim,
                                                    uint32_t n_rot) {
    __shared__ float scratch[64];
    const uint32_t tid = threadIdx.x;
    if (tid >= 64) return;
    const uint32_t n_nope = head_dim - n_rot;
    for (uint32_t off = 0; off < n_nope; off += 64u) {
        float v = 0.0f;
        if (off + tid < n_nope) {
            v = kv[off + tid];
            scratch[tid] = fabsf(v);
        } else {
            scratch[tid] = 0.0f;
        }
        __syncthreads();
        for (uint32_t s = 32; s > 0; s >>= 1) {
            if (tid < s) scratch[tid] = fmaxf(scratch[tid], scratch[tid + s]);
            __syncthreads();
        }
        const float amax = fmaxf(scratch[0], 1.0e-4f);
        const float scale = exp2f(ceilf(log2f(amax / 448.0f)));
        if (off + tid < n_nope) {
            const float q = rocm_e4m3fn_dequant(fminf(fmaxf(v / scale, -448.0f), 448.0f)) * scale;
            kv[off + tid] = q;
            raw[off + tid] = (float)((__half)q);
        }
        __syncthreads();
    }
    for (uint32_t i = n_nope + tid; i < head_dim; i += 64u) {
        raw[i] = (float)((__half)kv[i]);
    }
}

int ds4_metal_kv_fp8_store_raw_tensor(
        ds4_metal_tensor *kv,
        ds4_metal_tensor *raw_cache,
        uint32_t          raw_cap,
        uint32_t          row,
        uint32_t          head_dim,
        uint32_t          n_rot) {
    if (!g_initialized && !ds4_metal_init()) return 0;
    if (!kv || !raw_cache || head_dim == 0 || n_rot > head_dim || row >= raw_cap) return 0;
    if (kv->bytes < (uint64_t)head_dim * sizeof(float)) return 0;
    if (raw_cache->bytes < ((uint64_t)row + 1u) * head_dim * sizeof(float)) return 0;
    float *raw = (float *)tensor_u8(raw_cache) + (uint64_t)row * head_dim;
    hipLaunchKernelGGL(rocm_kv_fp8_store_raw_kernel, dim3(1), dim3(64), 0, 0,
                       (float *)tensor_u8(kv), raw, head_dim, n_rot);
    return rocm_ok(hipGetLastError(), "kv fp8 store launch") &&
           rocm_ok(hipDeviceSynchronize(), "kv fp8 store completion");
}

/* =========================================================================
 * Shared expert SwiGLU (Q8_0 weights).
 * ========================================================================= */

int ds4_metal_shared_gate_up_swiglu_q8_0_tensor(
        ds4_metal_tensor       *gate,
        ds4_metal_tensor       *up,
        ds4_metal_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_metal_tensor *x) {
    if (!gate || !up || !mid) return 0;
    if (!ds4_metal_matmul_q8_0_tensor(gate, model_map, model_size, gate_offset, in_dim, out_dim, x, 1)) return 0;
    if (!ds4_metal_matmul_q8_0_tensor(up, model_map, model_size, up_offset, in_dim, out_dim, x, 1)) return 0;
    return ds4_metal_swiglu_tensor(mid, gate, up, (uint32_t)out_dim, 0.0f, 1.0f);
}

/* =========================================================================
 * Attention output projections (Q8_0).
 * ========================================================================= */

int ds4_metal_attention_output_low_q8_tensor(
        ds4_metal_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        const ds4_metal_tensor *heads) {
    /* heads is laid out as n_groups blocks of group_dim; produce one column of
     * size n_groups * rank by running n_groups independent Q8_0 matvecs.
     * Each matvec consumes group_dim inputs and produces `rank` outputs.
     * The Metal kernel 'kernel_dsv4_attn_out_low_q8_0_f32' walks groups in a
     * single threadgroup, but for correctness we just dispatch per-group.
     */
    if (!low || !heads || group_dim == 0 || rank == 0 || n_groups == 0) return 0;
    if (low->bytes < (uint64_t)n_groups * rank * sizeof(float)) return 0;
    if (heads->bytes < (uint64_t)n_groups * group_dim * sizeof(float)) return 0;
    const uint64_t row_bytes = ((group_dim + 31u) / 32u) * 34u;
    const uint64_t group_w_bytes = row_bytes * rank;
    if (out_a_offset > model_size || (uint64_t)n_groups * group_w_bytes > model_size - out_a_offset) return 0;

    for (uint32_t g = 0; g < n_groups; g++) {
        ds4_metal_tensor x_view = *heads;
        x_view.offset = heads->offset + (uint64_t)g * group_dim * sizeof(float);
        x_view.bytes = group_dim * sizeof(float);
        ds4_metal_tensor low_view = *low;
        low_view.offset = low->offset + (uint64_t)g * rank * sizeof(float);
        low_view.bytes = rank * sizeof(float);
        if (!ds4_metal_matmul_q8_0_tensor(&low_view, model_map, model_size,
                                          out_a_offset + (uint64_t)g * group_w_bytes,
                                          group_dim, rank, &x_view, 1)) return 0;
    }
    return 1;
}

int ds4_metal_attention_output_q8_batch_tensor(
        ds4_metal_tensor       *out,
        ds4_metal_tensor       *low,
        ds4_metal_tensor       *group_tmp,
        ds4_metal_tensor       *low_tmp,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                out_b_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        uint64_t                out_dim,
        const ds4_metal_tensor *heads,
        uint32_t                n_tokens) {
    /* For each token, compute low = Wa @ heads (per-group Q8 matvec), then
     * out = Wb @ low (single Q8 matvec of dim n_groups*rank -> out_dim). */
    (void)group_tmp;
    (void)low_tmp;
    if (!out || !low || !heads || n_tokens == 0) return 0;
    const uint64_t low_per = (uint64_t)n_groups * rank;
    if (out->bytes < (uint64_t)n_tokens * out_dim * sizeof(float)) return 0;
    if (low->bytes < (uint64_t)n_tokens * low_per * sizeof(float)) return 0;
    if (heads->bytes < (uint64_t)n_tokens * n_groups * group_dim * sizeof(float)) return 0;

    for (uint32_t t = 0; t < n_tokens; t++) {
        ds4_metal_tensor heads_view = *heads;
        heads_view.offset = heads->offset + (uint64_t)t * n_groups * group_dim * sizeof(float);
        heads_view.bytes = (uint64_t)n_groups * group_dim * sizeof(float);
        ds4_metal_tensor low_view = *low;
        low_view.offset = low->offset + (uint64_t)t * low_per * sizeof(float);
        low_view.bytes = low_per * sizeof(float);
        if (!ds4_metal_attention_output_low_q8_tensor(&low_view, model_map, model_size,
                                                      out_a_offset, group_dim, rank,
                                                      n_groups, &heads_view)) return 0;
        ds4_metal_tensor out_view = *out;
        out_view.offset = out->offset + (uint64_t)t * out_dim * sizeof(float);
        out_view.bytes = out_dim * sizeof(float);
        if (!ds4_metal_matmul_q8_0_tensor(&out_view, model_map, model_size, out_b_offset,
                                          low_per, out_dim, &low_view, 1)) return 0;
    }
    return 1;
}

int ds4_metal_matmul_q8_0_hc_expand_tensor(
        ds4_metal_tensor       *out_hc,
        ds4_metal_tensor       *block_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_metal_tensor *x,
        const ds4_metal_tensor *residual_hc,
        const ds4_metal_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    /* Compose Q8 matmul (block_out = W . x) with HC expand (split layout). */
    if (!ds4_metal_matmul_q8_0_tensor(block_out, model_map, model_size, weight_offset,
                                      in_dim, out_dim, x, 1)) return 0;
    return ds4_metal_hc_expand_split_tensor(out_hc, block_out, residual_hc, split, n_embd, n_hc);
}

int ds4_metal_shared_down_hc_expand_q8_0_tensor(
        ds4_metal_tensor       *out_hc,
        ds4_metal_tensor       *shared_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_metal_tensor *shared_mid,
        const ds4_metal_tensor *routed_out,
        const ds4_metal_tensor *residual_hc,
        const ds4_metal_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    /* shared_out = W . shared_mid (Q8); then HC expand uses (routed_out + shared_out) as block. */
    if (!ds4_metal_matmul_q8_0_tensor(shared_out, model_map, model_size, weight_offset,
                                      in_dim, out_dim, shared_mid, 1)) return 0;
    return ds4_metal_hc_expand_add_split_tensor(out_hc, routed_out, shared_out, residual_hc,
                                                split, n_embd, n_hc);
}

/* =========================================================================
 * Indexer scoring and top-k selection.
 *
 * Score formula (per token t, compressed row c):
 *   score[t,c] = sum_h max(dot(Q[t,h], K[c]), 0) * W[t,h] * scale
 * Then mask with -inf for compressed rows beyond floor((pos0 + t + 1)/ratio).
 * ========================================================================= */

__global__ static void rocm_indexer_scores_kernel(float *scores,
                                                  const float *q,
                                                  const float *weights,
                                                  const float *kcomp,
                                                  uint32_t n_comp,
                                                  uint32_t n_tokens,
                                                  uint32_t n_head,
                                                  uint32_t head_dim,
                                                  uint32_t pos0,
                                                  uint32_t ratio,
                                                  float scale) {
    /* one threadgroup per (token, comp); reduce along head_dim within a block. */
    extern __shared__ float sh[];
    const uint32_t tok = blockIdx.y;
    const uint32_t comp = blockIdx.x;
    if (tok >= n_tokens || comp >= n_comp) return;
    const uint32_t visible = ratio == 0 ? n_comp : ((pos0 + tok + 1u) / ratio);
    if (comp >= visible) {
        if (threadIdx.x == 0) {
            scores[(uint64_t)tok * n_comp + comp] = -3.4028234663852886e38f;
        }
        return;
    }
    const float *kv = kcomp + (uint64_t)comp * head_dim;
    const float *wrow = weights + (uint64_t)tok * n_head;

    float acc = 0.0f;
    for (uint32_t h = 0; h < n_head; h++) {
        const float *qrow = q + ((uint64_t)tok * n_head + h) * head_dim;
        float dotv = 0.0f;
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            dotv += qrow[d] * kv[d];
        }
        sh[threadIdx.x] = dotv;
        __syncthreads();
        for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const float r = sh[0] > 0.0f ? sh[0] : 0.0f;
            acc += r * (wrow[h] * scale);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) scores[(uint64_t)tok * n_comp + comp] = acc;
}

static int rocm_indexer_scores_dispatch(
        ds4_metal_tensor       *scores,
        const ds4_metal_tensor *q,
        const ds4_metal_tensor *weights,
        const ds4_metal_tensor *kcomp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                pos0,
        uint32_t                ratio,
        float                   scale) {
    if (!g_initialized && !ds4_metal_init()) return 0;
    if (!scores || !q || !weights || !kcomp || n_comp == 0 || n_tokens == 0 || n_head == 0 || head_dim == 0) return 0;
    if (scores->bytes < (uint64_t)n_tokens * n_comp * sizeof(float)) return 0;
    if (q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float)) return 0;
    if (weights->bytes < (uint64_t)n_tokens * n_head * sizeof(float)) return 0;
    if (kcomp->bytes < (uint64_t)n_comp * head_dim * sizeof(float)) return 0;

    unsigned threads = 128u;
    while (threads > 1u && threads / 2u >= head_dim) threads >>= 1;
    const dim3 block_d(threads);
    const dim3 grid_d(n_comp, n_tokens);
    hipLaunchKernelGGL(rocm_indexer_scores_kernel,
                       grid_d, block_d, threads * sizeof(float), 0,
                       (float *)tensor_u8(scores),
                       (const float *)tensor_u8_const(q),
                       (const float *)tensor_u8_const(weights),
                       (const float *)tensor_u8_const(kcomp),
                       n_comp, n_tokens, n_head, head_dim,
                       pos0, ratio, scale);
    return rocm_ok(hipGetLastError(), "indexer scores launch") &&
           rocm_ok(hipDeviceSynchronize(), "indexer scores completion");
}

int ds4_metal_indexer_score_one_tensor(
        ds4_metal_tensor       *scores,
        const ds4_metal_tensor *q,
        const ds4_metal_tensor *weights,
        const ds4_metal_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_head,
        uint32_t                head_dim,
        float                   scale) {
    /* one-token decode at any position; visibility handled by caller via n_comp.
     * We pass ratio=0 so all rows up to n_comp are visible. */
    return rocm_indexer_scores_dispatch(scores, q, weights, index_comp,
                                        n_comp, 1, n_head, head_dim,
                                        0, 0, scale);
}

int ds4_metal_indexer_scores_prefill_tensor(
        ds4_metal_tensor       *scores,
        const ds4_metal_tensor *q,
        const ds4_metal_tensor *weights,
        const ds4_metal_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale) {
    return rocm_indexer_scores_dispatch(scores, q, weights, index_comp,
                                        n_comp, n_tokens, n_head, head_dim,
                                        0, ratio, scale);
}

int ds4_metal_indexer_scores_decode_batch_tensor(
        ds4_metal_tensor       *scores,
        const ds4_metal_tensor *q,
        const ds4_metal_tensor *weights,
        const ds4_metal_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale) {
    return rocm_indexer_scores_dispatch(scores, q, weights, index_comp,
                                        n_comp, n_tokens, n_head, head_dim,
                                        pos0, ratio, scale);
}

__global__ static void rocm_indexer_topk_kernel(int32_t *selected,
                                                const float *scores,
                                                uint32_t n_comp,
                                                uint32_t top_k) {
    /* one threadgroup per token, top-K via repeated argmax (top_k <= 512). */
    const uint32_t tok = blockIdx.x;
    const float *src = scores + (uint64_t)tok * n_comp;
    int32_t *dst = selected + (uint64_t)tok * top_k;
    if (threadIdx.x != 0) return;
    /* First pass: copy scores to a working area in shared mem too small if
     * n_comp huge.  For correctness, use direct pass with -inf marking on a
     * scratch buffer in global memory; we mark visited entries in a tiny stack
     * here using bookkeeping in selected. */
    for (uint32_t k = 0; k < top_k; k++) dst[k] = -1;
    for (uint32_t k = 0; k < top_k; k++) {
        float best_v = -3.4028234663852886e38f;
        int32_t best_i = -1;
        for (uint32_t c = 0; c < n_comp; c++) {
            float v = src[c];
            if (v == -3.4028234663852886e38f) continue;
            int taken = 0;
            for (uint32_t kk = 0; kk < k; kk++) if (dst[kk] == (int32_t)c) { taken = 1; break; }
            if (taken) continue;
            if (v > best_v) { best_v = v; best_i = (int32_t)c; }
        }
        dst[k] = best_i;
    }
    /* Sort selected ascending by row id, keeping -1 sentinels at end. */
    for (uint32_t i = 0; i < top_k; i++) {
        for (uint32_t j = i + 1; j < top_k; j++) {
            int32_t a = dst[i], b = dst[j];
            int swap = 0;
            if (a < 0 && b >= 0) swap = 1;
            else if (a >= 0 && b >= 0 && b < a) swap = 1;
            if (swap) { dst[i] = b; dst[j] = a; }
        }
    }
}

int ds4_metal_indexer_topk_tensor(
        ds4_metal_tensor       *selected,
        const ds4_metal_tensor *scores,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k) {
    if (!g_initialized && !ds4_metal_init()) return 0;
    if (!selected || !scores || n_comp == 0 || n_tokens == 0 || top_k == 0) return 0;
    if (selected->bytes < (uint64_t)n_tokens * top_k * sizeof(int32_t)) return 0;
    if (scores->bytes < (uint64_t)n_tokens * n_comp * sizeof(float)) return 0;

    /* Note: this O(n_comp * top_k) per-token loop is correct but slow; revisit
     * with a proper bitonic top-k once correctness is validated. */
    hipLaunchKernelGGL(rocm_indexer_topk_kernel, dim3(n_tokens), dim3(1), 0, 0,
                       (int32_t *)tensor_u8(selected),
                       (const float *)tensor_u8_const(scores),
                       n_comp, top_k);
    return rocm_ok(hipGetLastError(), "indexer topk launch") &&
           rocm_ok(hipDeviceSynchronize(), "indexer topk completion");
}

__global__ static void rocm_topk_mask_init_kernel(float *mask,
                                                  uint32_t n_comp,
                                                  uint32_t n_tokens) {
    const uint64_t total = (uint64_t)n_comp * n_tokens;
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;
    mask[gid] = -3.4028234663852886e38f;
}

__global__ static void rocm_topk_mask_scatter_kernel(float *mask,
                                                     const int32_t *topk,
                                                     uint32_t n_comp,
                                                     uint32_t n_tokens,
                                                     uint32_t top_k) {
    const uint64_t total = (uint64_t)top_k * n_tokens;
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;
    const uint32_t ik = (uint32_t)(gid % top_k);
    const uint32_t it = (uint32_t)(gid / top_k);
    const int32_t idx = topk[(uint64_t)it * top_k + ik];
    if (idx >= 0 && (uint32_t)idx < n_comp) {
        mask[(uint64_t)it * n_comp + (uint32_t)idx] = 0.0f;
    }
}

int ds4_metal_dsv4_topk_mask_tensor(
        ds4_metal_tensor       *mask,
        const ds4_metal_tensor *topk,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k) {
    if (!g_initialized && !ds4_metal_init()) return 0;
    if (!mask || !topk || n_comp == 0 || n_tokens == 0 || top_k == 0) return 0;
    if (mask->bytes < (uint64_t)n_comp * n_tokens * sizeof(float)) return 0;
    if (topk->bytes < (uint64_t)n_tokens * top_k * sizeof(int32_t)) return 0;

    {
        const uint64_t total = (uint64_t)n_comp * n_tokens;
        const dim3 block(256);
        const dim3 grid((unsigned)((total + block.x - 1u) / block.x));
        hipLaunchKernelGGL(rocm_topk_mask_init_kernel, grid, block, 0, 0,
                           (float *)tensor_u8(mask), n_comp, n_tokens);
        if (!rocm_ok(hipGetLastError(), "topk mask init launch") ||
            !rocm_ok(hipDeviceSynchronize(), "topk mask init completion")) return 0;
    }
    {
        const uint64_t total = (uint64_t)top_k * n_tokens;
        const dim3 block(256);
        const dim3 grid((unsigned)((total + block.x - 1u) / block.x));
        hipLaunchKernelGGL(rocm_topk_mask_scatter_kernel, grid, block, 0, 0,
                           (float *)tensor_u8(mask), (const int32_t *)tensor_u8_const(topk),
                           n_comp, n_tokens, top_k);
        return rocm_ok(hipGetLastError(), "topk mask scatter launch") &&
               rocm_ok(hipDeviceSynchronize(), "topk mask scatter completion");
    }
}

/* =========================================================================
 * Router selection.
 * ========================================================================= */

__global__ static void rocm_router_select_one_kernel(int32_t *selected,
                                                     float *weights,
                                                     const float *probs,
                                                     const float *bias,
                                                     const int32_t *hash,
                                                     uint32_t n_expert,
                                                     uint32_t n_used,
                                                     uint32_t hash_rows,
                                                     uint32_t token,
                                                     int has_bias,
                                                     int hash_mode) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    if (hash_mode) {
        const uint32_t row = token < hash_rows ? token : hash_rows - 1u;
        for (uint32_t i = 0; i < n_used; i++) selected[i] = hash[row * n_used + i];
    } else {
        /* repeated argmax; n_expert <= 256 so this is fine for one-token decode */
        for (uint32_t k = 0; k < n_used; k++) {
            float best_v = -3.4028234663852886e38f;
            int32_t best_i = -1;
            for (uint32_t e = 0; e < n_expert; e++) {
                float v = probs[e];
                if (has_bias) v += bias[e];
                int taken = 0;
                for (uint32_t kk = 0; kk < k; kk++) if (selected[kk] == (int32_t)e) { taken = 1; break; }
                if (taken) continue;
                if (v > best_v) { best_v = v; best_i = (int32_t)e; }
            }
            selected[k] = best_i;
        }
    }

    float sum = 0.0f;
    for (uint32_t i = 0; i < n_used; i++) sum += probs[selected[i]];
    if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;
    for (uint32_t i = 0; i < n_used; i++) weights[i] = probs[selected[i]] / sum * 1.5f;
}

int ds4_metal_router_select_tensor(
        ds4_metal_tensor       *selected,
        ds4_metal_tensor       *weights,
        ds4_metal_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        uint64_t                hash_offset,
        uint32_t                hash_rows,
        uint32_t                token,
        uint32_t                n_expert_groups,
        uint32_t                n_group_used,
        bool                    has_bias,
        bool                    hash_mode,
        const ds4_metal_tensor *logits) {
    if (!g_initialized && !ds4_metal_init()) return 0;
    if (!selected || !weights || !probs || !logits || n_expert_groups == 0 || n_group_used == 0) return 0;
    if (selected->bytes < (uint64_t)n_group_used * sizeof(int32_t)) return 0;
    if (weights->bytes < (uint64_t)n_group_used * sizeof(float)) return 0;
    if (probs->bytes < (uint64_t)n_expert_groups * sizeof(float)) return 0;

    /* probs := sqrt(softplus(logits)).  This matches the reference path so the
     * caller doesn't need to pre-compute it.  The softmax-style probability
     * shape is mandated by DS4. */
    {
        /* In-place transform on probs from logits (probs may equal logits). */
        const float *src = (const float *)tensor_u8_const(logits);
        float *dst = (float *)tensor_u8(probs);
        for (uint32_t i = 0; i < n_expert_groups; i++) {
            const float l = src[i];
            const float sp = l > 20.0f ? l : logf(1.0f + expf(l));
            dst[i] = sqrtf(sp > 0.0f ? sp : 0.0f);
        }
        /* Avoid a sync since these were host-mapped writes; downstream kernel
         * reads through the same mapped pages. */
    }

    void *bh = NULL, *bd = NULL;
    void *hh = NULL, *hd = NULL;
    if (has_bias) {
        if (bias_offset > model_size || (uint64_t)n_expert_groups * sizeof(float) > model_size - bias_offset) return 0;
        if (!rocm_upload_mapped((const uint8_t *)model_map + bias_offset,
                                (uint64_t)n_expert_groups * sizeof(float),
                                &bh, &bd, "router bias upload")) return 0;
    }
    if (hash_mode) {
        const uint64_t hash_bytes = (uint64_t)hash_rows * n_group_used * sizeof(int32_t);
        if (hash_offset > model_size || hash_bytes > model_size - hash_offset) {
            if (bh) (void)hipHostFree(bh); return 0;
        }
        if (!rocm_upload_mapped((const uint8_t *)model_map + hash_offset, hash_bytes,
                                &hh, &hd, "router hash upload")) {
            if (bh) (void)hipHostFree(bh); return 0;
        }
    }

    hipLaunchKernelGGL(rocm_router_select_one_kernel, dim3(1), dim3(1), 0, 0,
                       (int32_t *)tensor_u8(selected),
                       (float *)tensor_u8(weights),
                       (const float *)tensor_u8_const(probs),
                       (const float *)bd,
                       (const int32_t *)hd,
                       n_expert_groups, n_group_used, hash_rows, token,
                       has_bias ? 1 : 0, hash_mode ? 1 : 0);
    int ok = rocm_ok(hipGetLastError(), "router select launch") &&
             rocm_ok(hipDeviceSynchronize(), "router select completion");
    if (bh) (void)hipHostFree(bh);
    if (hh) (void)hipHostFree(hh);
    return ok;
}

int ds4_metal_router_select_batch_tensor(
        ds4_metal_tensor       *selected,
        ds4_metal_tensor       *weights,
        ds4_metal_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        uint64_t                hash_offset,
        uint32_t                hash_rows,
        uint32_t                n_expert_groups,
        uint32_t                n_group_used,
        bool                    has_bias,
        bool                    hash_mode,
        const ds4_metal_tensor *logits,
        const ds4_metal_tensor *tokens,
        uint32_t                n_tokens) {
    if (!selected || !weights || !probs || !logits || !tokens) return 0;
    /* Per-token loop is fine for small batch sizes; prefill hits hash-mode anyway. */
    const int32_t *toks = (const int32_t *)tensor_u8_const(tokens);
    for (uint32_t t = 0; t < n_tokens; t++) {
        ds4_metal_tensor sel_v = *selected;
        sel_v.offset = selected->offset + (uint64_t)t * n_group_used * sizeof(int32_t);
        sel_v.bytes = (uint64_t)n_group_used * sizeof(int32_t);
        ds4_metal_tensor w_v = *weights;
        w_v.offset = weights->offset + (uint64_t)t * n_group_used * sizeof(float);
        w_v.bytes = (uint64_t)n_group_used * sizeof(float);
        ds4_metal_tensor p_v = *probs;
        p_v.offset = probs->offset + (uint64_t)t * n_expert_groups * sizeof(float);
        p_v.bytes = (uint64_t)n_expert_groups * sizeof(float);
        ds4_metal_tensor l_v = *logits;
        l_v.offset = logits->offset + (uint64_t)t * n_expert_groups * sizeof(float);
        l_v.bytes = (uint64_t)n_expert_groups * sizeof(float);
        if (!ds4_metal_router_select_tensor(&sel_v, &w_v, &p_v, model_map, model_size,
                                            bias_offset, hash_offset, hash_rows,
                                            (uint32_t)toks[t], n_expert_groups,
                                            n_group_used, has_bias, hash_mode, &l_v)) {
            return 0;
        }
    }
    return 1;
}

/* =========================================================================
 * Compressor kernels.
 * =========================================================================
 *
 * The DSv4 compressor maintains a per-layer rolling state with two halves
 * (when ratio == 4) of `ratio` rows each.  store_batch fills the "current"
 * half (for ratio == 4) or the only half (for ratio < 4) by routing each
 * input row to row index `(coff == 2 ? ratio + pos_mod : pos_mod)` and
 * adding the absolute positional embedding (APE) row for that pos_mod onto
 * the score channel.  When several input tokens map to the same state row
 * (n_tokens > ratio) the LAST writer wins, matching Metal's `set_rows`
 * semantics; we resolve this without races by iterating per state row and
 * looking up the latest source token from each thread block.
 */

__global__ static void rocm_compressor_store_batch_kernel(
        float *state_kv,
        float *state_score,
        const float *kv,
        const float *sc,
        const float *ape_f32,
        const __half *ape_f16,
        uint32_t width,
        uint32_t ratio,
        uint32_t coff,
        uint32_t pos0,
        uint32_t n_tokens) {
    const uint32_t row = blockIdx.x;
    const uint32_t lane = blockIdx.y * blockDim.x + threadIdx.x;
    if (lane >= width) return;

    int pos_mod;
    if (coff == 2u) {
        if (row < ratio) return;            /* lower half is left untouched */
        pos_mod = (int)(row - ratio);
    } else {
        pos_mod = (int)row;
    }

    const int t0 = ((pos_mod - (int)(pos0 % ratio)) + (int)ratio) % (int)ratio;
    if ((uint32_t)t0 >= n_tokens) return;
    const uint32_t k_max = (n_tokens - 1u - (uint32_t)t0) / ratio;
    const uint32_t t = (uint32_t)t0 + k_max * ratio;

    const uint64_t in_off = (uint64_t)t * width + lane;
    const uint64_t state_off = (uint64_t)row * width + lane;
    const uint64_t ape_off = (uint64_t)pos_mod * width + lane;

    state_kv[state_off] = kv[in_off];
    const float ape = ape_f16 ? __half2float(ape_f16[ape_off]) : ape_f32[ape_off];
    state_score[state_off] = sc[in_off] + ape;
}

int ds4_metal_compressor_store_batch_tensor(
        const ds4_metal_tensor *kv,
        const ds4_metal_tensor *sc,
        ds4_metal_tensor       *state_kv,
        ds4_metal_tensor       *state_score,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_tokens) {
    if (!g_initialized && !ds4_metal_init()) return 0;
    if (!kv || !sc || !state_kv || !state_score || !model_map ||
        head_dim == 0 || n_tokens == 0 ||
        (ratio != 1u && ratio != 2u && ratio != 4u) ||
        (ape_type != 0u && ape_type != 1u)) return 0;

    const uint32_t coff = (ratio == 4u) ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint64_t kv_bytes = (uint64_t)n_tokens * width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t ape_bytes = (uint64_t)ratio * width * elem_ape;

    if (kv->bytes < kv_bytes || sc->bytes < kv_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes ||
        ape_offset > model_size || ape_bytes > model_size - ape_offset) return 0;

    void *ape_h = NULL, *ape_d = NULL;
    if (!rocm_upload_mapped((const uint8_t *)model_map + ape_offset,
                            ape_bytes, &ape_h, &ape_d, "compressor APE upload")) return 0;

    const uint32_t threads = (width < 256u) ? width : 256u;
    const uint32_t lane_blocks = (width + threads - 1u) / threads;
    dim3 grid(state_rows, lane_blocks, 1);
    dim3 block(threads, 1, 1);

    hipLaunchKernelGGL(rocm_compressor_store_batch_kernel, grid, block, 0, 0,
                       (float *)tensor_u8(state_kv),
                       (float *)tensor_u8(state_score),
                       (const float *)tensor_u8_const(kv),
                       (const float *)tensor_u8_const(sc),
                       ape_type == 0u ? (const float *)ape_d : (const float *)NULL,
                       ape_type == 1u ? (const __half *)ape_d : (const __half *)NULL,
                       width, ratio, coff, pos0, n_tokens);
    int ok = rocm_ok(hipGetLastError(), "compressor store batch launch") &&
             rocm_ok(hipDeviceSynchronize(), "compressor store batch completion");
    (void)hipHostFree(ape_h);
    return ok;
}

/* Per-token compressor update.  Stores the projected kv/sc row into the
 * recurrent state and, on ratio boundaries, pools the state into a fresh
 * comp_cache row, applies weighted RMS norm + RoPE tail, then shifts the
 * upper half of the state into the lower half (ratio == 4 only). */
__global__ static void rocm_compressor_pool_kernel(
        float *out,
        const float *state_kv,
        const float *state_score,
        uint32_t head_dim,
        uint32_t ratio,
        uint32_t coff) {
    const uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= head_dim) return;
    const uint32_t width = coff * head_dim;

    float maxs = -1.0e30f;
    if (coff == 2u) {
        for (uint32_t r = 0; r < ratio; r++) {
            const float sp = state_score[(uint64_t)r * width + j];
            const float sc = state_score[(uint64_t)(ratio + r) * width + head_dim + j];
            if (sp > maxs) maxs = sp;
            if (sc > maxs) maxs = sc;
        }
    } else {
        for (uint32_t r = 0; r < ratio; r++) {
            const float s = state_score[(uint64_t)r * width + j];
            if (s > maxs) maxs = s;
        }
    }
    if (maxs <= -5.0e29f) {
        out[j] = 0.0f;
        return;
    }
    float denom = 0.0f, sum = 0.0f;
    if (coff == 2u) {
        for (uint32_t r = 0; r < ratio; r++) {
            const float wp = expf(state_score[(uint64_t)r * width + j] - maxs);
            const float wc = expf(state_score[(uint64_t)(ratio + r) * width + head_dim + j] - maxs);
            denom += wp + wc;
            sum += wp * state_kv[(uint64_t)r * width + j];
            sum += wc * state_kv[(uint64_t)(ratio + r) * width + head_dim + j];
        }
    } else {
        for (uint32_t r = 0; r < ratio; r++) {
            const float w = expf(state_score[(uint64_t)r * width + j] - maxs);
            denom += w;
            sum += w * state_kv[(uint64_t)r * width + j];
        }
    }
    out[j] = denom > 0.0f ? sum / denom : 0.0f;
}

__global__ static void rocm_compressor_ratio4_shift_kernel(
        float *state_kv,
        float *state_score,
        uint32_t width) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t n = 4u * width;
    if (i >= n) return;
    state_kv[i] = state_kv[n + i];
    state_score[i] = state_score[n + i];
}

static int rocm_compressor_pool_dispatch(ds4_metal_tensor *out,
                                          const ds4_metal_tensor *state_kv,
                                          const ds4_metal_tensor *state_score,
                                          uint32_t head_dim,
                                          uint32_t ratio) {
    const uint32_t coff = (ratio == 4u) ? 2u : 1u;
    const uint32_t threads = head_dim < 256u ? head_dim : 256u;
    const uint32_t blocks = (head_dim + threads - 1u) / threads;
    hipLaunchKernelGGL(rocm_compressor_pool_kernel, dim3(blocks), dim3(threads), 0, 0,
                       (float *)tensor_u8(out),
                       (const float *)tensor_u8_const(state_kv),
                       (const float *)tensor_u8_const(state_score),
                       head_dim, ratio, coff);
    return rocm_ok(hipGetLastError(), "compressor pool launch") &&
           rocm_ok(hipDeviceSynchronize(), "compressor pool completion");
}

static int rocm_compressor_ratio4_shift_dispatch(ds4_metal_tensor *state_kv,
                                                  ds4_metal_tensor *state_score,
                                                  uint32_t width) {
    const uint32_t n = 4u * width;
    const uint32_t threads = n < 256u ? n : 256u;
    const uint32_t blocks = (n + threads - 1u) / threads;
    hipLaunchKernelGGL(rocm_compressor_ratio4_shift_kernel, dim3(blocks), dim3(threads), 0, 0,
                       (float *)tensor_u8(state_kv),
                       (float *)tensor_u8(state_score),
                       width);
    return rocm_ok(hipGetLastError(), "compressor shift launch") &&
           rocm_ok(hipDeviceSynchronize(), "compressor shift completion");
}

int ds4_metal_compressor_update_tensor(
        const ds4_metal_tensor *kv_cur,
        const ds4_metal_tensor *sc_cur,
        ds4_metal_tensor       *state_kv,
        ds4_metal_tensor       *state_score,
        ds4_metal_tensor       *comp_cache,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos,
        uint32_t                comp_row,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps) {
    if (!g_initialized && !ds4_metal_init()) return 0;
    if (!kv_cur || !sc_cur || !state_kv || !state_score || !comp_cache || !model_map ||
        head_dim == 0 || (ratio != 1u && ratio != 2u && ratio != 4u) ||
        n_rot > head_dim || (n_rot & 1u) != 0 ||
        (ape_type != 0u && ape_type != 1u) || norm_type != 0u) return 0;

    if (!ds4_metal_compressor_store_batch_tensor(kv_cur, sc_cur, state_kv, state_score,
                                                 model_map, model_size,
                                                 ape_offset, ape_type,
                                                 head_dim, ratio, pos, 1)) return 0;

    const uint32_t emit = ((pos + 1u) % ratio) == 0u ? 1u : 0u;
    if (!emit) return 1;

    const uint64_t comp_row_bytes = (uint64_t)head_dim * sizeof(float);
    if ((uint64_t)comp_row * head_dim * sizeof(float) >= comp_cache->bytes ||
        comp_row_bytes > comp_cache->bytes - (uint64_t)comp_row * head_dim * sizeof(float)) return 0;

    ds4_metal_tensor *comp_row_view = ds4_metal_tensor_view(
            comp_cache,
            (uint64_t)comp_row * head_dim * sizeof(float),
            comp_row_bytes);
    if (!comp_row_view) return 0;

    int ok = rocm_compressor_pool_dispatch(comp_row_view, state_kv, state_score, head_dim, ratio);
    if (ok) ok = ds4_metal_rms_norm_weight_rows_tensor(comp_row_view, comp_row_view,
                                                       model_map, model_size, norm_offset,
                                                       head_dim, 1, rms_eps) != 0;
    if (ok) {
        const uint32_t comp_pos = pos + 1u - ratio;
        ok = ds4_metal_rope_tail_tensor(comp_row_view, 1, 1, head_dim, n_rot, comp_pos,
                                        n_ctx_orig, false, freq_base, freq_scale,
                                        ext_factor, attn_factor, beta_fast, beta_slow) != 0;
    }
    if (ok && ratio == 4u) {
        const uint32_t coff = 2u;
        const uint32_t width = coff * head_dim;
        ok = rocm_compressor_ratio4_shift_dispatch(state_kv, state_score, width);
    }
    ds4_metal_tensor_free(comp_row_view);
    return ok;
}

__global__ static void rocm_fill_f32_kernel(float *p, float v, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}

static int rocm_fill_f32_dispatch(void *p, float v, uint64_t n) {
    if (n == 0) return 1;
    const uint64_t threads = 256;
    const uint64_t blocks = (n + threads - 1) / threads;
    hipLaunchKernelGGL(rocm_fill_f32_kernel, dim3((uint32_t)blocks), dim3((uint32_t)threads), 0, 0,
                       (float *)p, v, n);
    return rocm_ok(hipGetLastError(), "fill f32 launch") &&
           rocm_ok(hipDeviceSynchronize(), "fill f32 completion");
}

/* Multi-token prefill compressor pass.  Resets state, then iterates tokens
 * via single-token compressor_update; each emit boundary writes one comp
 * row (optionally FP8-quantized) and, for ratio==4, shifts state. */
int ds4_metal_compressor_prefill_tensor(
        ds4_metal_tensor       *comp_cache,
        ds4_metal_tensor       *state_kv,
        ds4_metal_tensor       *state_score,
        const ds4_metal_tensor *kv,
        const ds4_metal_tensor *sc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        bool                    quantize_fp8,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps) {
    if (!g_initialized && !ds4_metal_init()) return 0;
    if (!comp_cache || !state_kv || !state_score || !kv || !sc || !model_map ||
        head_dim == 0 || (ratio != 1u && ratio != 2u && ratio != 4u) || n_tokens == 0 ||
        n_rot > head_dim || (n_rot & 1u) != 0 ||
        (ape_type != 0u && ape_type != 1u) || norm_type != 0u) return 0;

    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint64_t state_elts = (uint64_t)state_rows * width;
    const uint64_t kv_bytes = (uint64_t)n_tokens * width * sizeof(float);
    const uint32_t n_comp = n_tokens / ratio;
    const uint64_t comp_bytes = (uint64_t)n_comp * head_dim * sizeof(float);
    if (kv->bytes < kv_bytes || sc->bytes < kv_bytes ||
        state_kv->bytes < state_elts * sizeof(float) ||
        state_score->bytes < state_elts * sizeof(float) ||
        (n_comp != 0 && comp_cache->bytes < comp_bytes)) return 0;

    if (!rocm_fill_f32_dispatch(tensor_u8(state_kv), 0.0f, state_elts)) return 0;
    if (!rocm_fill_f32_dispatch(tensor_u8(state_score), -1.0e30f, state_elts)) return 0;

    uint32_t comp_row = 0;
    int ok = 1;
    for (uint32_t t = 0; t < n_tokens && ok; t++) {
        ds4_metal_tensor *kv_v = ds4_metal_tensor_view(kv, (uint64_t)t * width * sizeof(float),
                                                       (uint64_t)width * sizeof(float));
        ds4_metal_tensor *sc_v = ds4_metal_tensor_view(sc, (uint64_t)t * width * sizeof(float),
                                                       (uint64_t)width * sizeof(float));
        if (!kv_v || !sc_v) { ok = 0; }
        else {
            ok = ds4_metal_compressor_update_tensor(kv_v, sc_v, state_kv, state_score, comp_cache,
                                                    model_map, model_size, ape_offset, ape_type,
                                                    norm_offset, norm_type, head_dim, ratio,
                                                    pos0 + t, comp_row, n_rot, n_ctx_orig,
                                                    freq_base, freq_scale, ext_factor, attn_factor,
                                                    beta_fast, beta_slow, rms_eps);
            if (ok && ((pos0 + t + 1u) % ratio) == 0u) {
                if (quantize_fp8 && head_dim == 128u /* DS4_N_HEAD_DIM */) {
                    ds4_metal_tensor *comp_v = ds4_metal_tensor_view(
                            comp_cache,
                            (uint64_t)comp_row * head_dim * sizeof(float),
                            (uint64_t)head_dim * sizeof(float));
                    if (!comp_v) ok = 0;
                    else {
                        ok = ds4_metal_dsv4_fp8_kv_quantize_tensor(comp_v, 1, head_dim, n_rot) != 0;
                        ds4_metal_tensor_free(comp_v);
                    }
                }
                comp_row++;
            }
        }
        ds4_metal_tensor_free(sc_v);
        ds4_metal_tensor_free(kv_v);
    }
    return ok;
}

/* Re-establish ratio==4 compressor state from the last 4 prefill tokens'
 * projected (kv,sc) rows, leaving the upper half empty (0,-inf).  Used to
 * place the state into the post-shift configuration so that decode can
 * keep filling the upper half. */
__global__ static void rocm_compressor_prefill_state_ratio4_kernel(
        float *state_kv,
        float *state_score,
        const float *kv,
        const float *sc,
        const float *ape_f32,
        const __half *ape_f16,
        uint32_t width,
        uint32_t pos0) {
    const uint32_t r = blockIdx.x;            /* 0..3 (target lower row) */
    const uint32_t lane = blockIdx.y * blockDim.x + threadIdx.x;
    if (r >= 4u || lane >= width) return;
    const uint32_t pos_mod = (pos0 + r) % 4u;
    const uint64_t in_off = (uint64_t)r * width + lane;
    const uint64_t state_off = (uint64_t)r * width + lane;
    const uint64_t ape_off = (uint64_t)pos_mod * width + lane;
    state_kv[state_off] = kv[in_off];
    const float ape = ape_f16 ? __half2float(ape_f16[ape_off]) : ape_f32[ape_off];
    state_score[state_off] = sc[in_off] + ape;
}

int ds4_metal_compressor_prefill_state_ratio4_tensor(
        ds4_metal_tensor       *state_kv,
        ds4_metal_tensor       *state_score,
        const ds4_metal_tensor *kv_tail,
        const ds4_metal_tensor *sc_tail,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                head_dim,
        uint32_t                pos0) {
    if (!g_initialized && !ds4_metal_init()) return 0;
    if (!state_kv || !state_score || !kv_tail || !sc_tail || !model_map ||
        head_dim == 0 || (ape_type != 0u && ape_type != 1u)) return 0;

    const uint32_t ratio = 4u;
    const uint32_t width = 2u * head_dim;
    const uint32_t state_rows = 8u;
    const uint64_t tail_bytes = (uint64_t)ratio * width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t ape_bytes = (uint64_t)ratio * width * elem_ape;
    if (kv_tail->bytes < tail_bytes || sc_tail->bytes < tail_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes ||
        ape_offset > model_size || ape_bytes > model_size - ape_offset) return 0;

    if (!rocm_fill_f32_dispatch(tensor_u8(state_kv), 0.0f, (uint64_t)state_rows * width)) return 0;
    if (!rocm_fill_f32_dispatch(tensor_u8(state_score), -1.0e30f, (uint64_t)state_rows * width)) return 0;

    void *ape_h = NULL, *ape_d = NULL;
    if (!rocm_upload_mapped((const uint8_t *)model_map + ape_offset,
                            ape_bytes, &ape_h, &ape_d, "ratio4 state APE upload")) return 0;

    const uint32_t threads = width < 256u ? width : 256u;
    const uint32_t lane_blocks = (width + threads - 1u) / threads;
    dim3 grid(ratio, lane_blocks, 1);
    dim3 block(threads, 1, 1);
    hipLaunchKernelGGL(rocm_compressor_prefill_state_ratio4_kernel, grid, block, 0, 0,
                       (float *)tensor_u8(state_kv),
                       (float *)tensor_u8(state_score),
                       (const float *)tensor_u8_const(kv_tail),
                       (const float *)tensor_u8_const(sc_tail),
                       ape_type == 0u ? (const float *)ape_d : (const float *)NULL,
                       ape_type == 1u ? (const __half *)ape_d : (const __half *)NULL,
                       width, pos0);
    int ok = rocm_ok(hipGetLastError(), "compressor ratio4 state launch") &&
             rocm_ok(hipDeviceSynchronize(), "compressor ratio4 state completion");
    (void)hipHostFree(ape_h);
    return ok;
}

#define DS4_ROCM_STUB(name) int name() { rocm_unimplemented(#name); return 0; }

int ds4_metal_compressor_prefill_ratio4_replay_tensor(
        ds4_metal_tensor       *comp_cache,
        ds4_metal_tensor       *state_kv,
        ds4_metal_tensor       *state_score,
        const ds4_metal_tensor *kv,
        const ds4_metal_tensor *sc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        bool                    quantize_fp8,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps) {
    /* Replay is a ratio-4 prefill that requires aligned pos0 and full-period
     * n_tokens.  Semantically identical to compressor_prefill with ratio==4. */
    if (n_tokens == 0 || (n_tokens & 3u) != 0 || (pos0 & 3u) != 0) return 0;
    return ds4_metal_compressor_prefill_tensor(comp_cache, state_kv, state_score, kv, sc,
                                                model_map, model_size,
                                                ape_offset, ape_type, norm_offset, norm_type,
                                                head_dim, 4u, pos0, n_tokens, n_rot, n_ctx_orig,
                                                quantize_fp8, freq_base, freq_scale,
                                                ext_factor, attn_factor, beta_fast, beta_slow,
                                                rms_eps);
}

DS4_ROCM_STUB(ds4_metal_attention_decode_heads_tensor)
DS4_ROCM_STUB(ds4_metal_attention_prefill_raw_heads_tensor)
DS4_ROCM_STUB(ds4_metal_attention_decode_raw_batch_heads_tensor)
DS4_ROCM_STUB(ds4_metal_attention_decode_mixed_batch_heads_tensor)
DS4_ROCM_STUB(ds4_metal_attention_indexed_mixed_batch_heads_tensor)
DS4_ROCM_STUB(ds4_metal_attention_prefill_static_mixed_heads_tensor)
DS4_ROCM_STUB(ds4_metal_attention_prefill_masked_mixed_heads_tensor)
DS4_ROCM_STUB(ds4_metal_routed_moe_one_tensor)
DS4_ROCM_STUB(ds4_metal_routed_moe_batch_tensor)

#ifdef __cplusplus
}
#endif
