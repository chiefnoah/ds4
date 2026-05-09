#include <stdbool.h>
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <sys/mman.h>

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

/* Registered model-map regions.  Each entry mirrors a slice of the mmap'd
 * model file into a device-resident buffer so kernels can dereference the
 * registered device pointer without per-call uploads.
 *
 * Two registration modes:
 *   - "host_pinned": host_owned is non-NULL.  hipHostRegister'd a slice of
 *     the source mmap; dev_base aliases the same physical pages.  Used for
 *     small synthetic test maps that fit in pinned host RAM.
 *   - "device_copy": host_owned is NULL.  hipMalloc'd a separate device
 *     allocation and hipMemcpy'd the source bytes in.  Used for production
 *     models larger than physical host RAM (BIOS reserves most of the
 *     APU's UMA for GPU use, leaving only ~30 GB host-pinnable). */
#define DS4_ROCM_MAX_REGISTRATIONS 4
typedef struct {
    const uint8_t *host_base;     /* Source pointer the kernels supply. */
    uint8_t       *dev_base;      /* Device pointer to use instead. */
    uint64_t       size;
    void          *host_owned;    /* Non-NULL when hipHostRegister'd. */
    void          *dev_owned;     /* Non-NULL when hipMalloc'd a copy. */
} rocm_registration;
static rocm_registration g_registrations[DS4_ROCM_MAX_REGISTRATIONS];
static int               g_registration_count;

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

static int rocm_sync_each(void);
static int rocm_launch_done(const char *launch_label, const char *sync_label);
static void rocm_drain_pending_frees(void);
static void rocm_defer_free(void *host_ptr);

static int rocm_ok(hipError_t rc, const char *what) {
    if (rc == hipSuccess) return 1;
    fprintf(stderr, "ds4: ROCm %s failed: %s\n", what, hipGetErrorString(rc));
    return 0;
}

static int rocm_trace(void) {
    const char *v = getenv("DS4_ROCM_TRACE");
    return v && v[0] && v[0] != '0';
}

/* Per-kernel timing.  Enabled by setting DS4_ROCM_TIME=1.  Each wrapper
 * brackets its work with rocm_time_begin / rocm_time_end_named, which
 * accumulates wall-clock time per kernel name.  At exit, ds4_metal_cleanup
 * dumps a sorted histogram to stderr so we can see where the GPU time is
 * actually going.  Kept minimal so the disabled path is just an env-var
 * read per call. */
#define ROCM_TIME_MAX_BUCKETS 64
struct rocm_time_bucket {
    const char *name;       /* string literal; no copy */
    uint64_t    count;
    double      total_sec;
};
static struct rocm_time_bucket g_time_buckets[ROCM_TIME_MAX_BUCKETS];
static int                     g_time_n_buckets;
static int                     g_time_enabled = -1;  /* -1 = unchecked */

static int rocm_time_on(void) {
    if (g_time_enabled < 0) {
        const char *v = getenv("DS4_ROCM_TIME");
        g_time_enabled = (v && v[0] && v[0] != '0') ? 1 : 0;
    }
    return g_time_enabled;
}

static double rocm_now_sec(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1.0e-9 * (double)t.tv_nsec;
}

static void rocm_time_record(const char *name, double seconds) {
    if (!rocm_time_on()) return;
    /* Linear search is fine: <50 unique names total. */
    int i = 0;
    for (; i < g_time_n_buckets; i++) {
        if (g_time_buckets[i].name == name) break;  /* literal-pointer compare */
    }
    if (i == g_time_n_buckets) {
        if (g_time_n_buckets >= ROCM_TIME_MAX_BUCKETS) return;
        g_time_buckets[i].name = name;
        g_time_buckets[i].count = 0;
        g_time_buckets[i].total_sec = 0.0;
        g_time_n_buckets++;
    }
    g_time_buckets[i].count++;
    g_time_buckets[i].total_sec += seconds;
}

/* RAII-style scope: instances are auto-cleaned via __attribute__((cleanup)),
 * recording elapsed time against the bound label as soon as control leaves
 * the enclosing block (including via early `return`).  All wrappers can
 * therefore opt in with a single `ROCM_TIME_SCOPE("name");` line. */
struct rocm_time_scope {
    const char *name;
    double      t0;
};
static inline void rocm_time_scope_end(struct rocm_time_scope *s) {
    if (rocm_time_on()) rocm_time_record(s->name, rocm_now_sec() - s->t0);
}
#define ROCM_TIME_SCOPE(label) \
    __attribute__((cleanup(rocm_time_scope_end))) \
    struct rocm_time_scope _rocm_scope_ ## __LINE__ = { \
        .name = (label), \
        .t0 = rocm_time_on() ? rocm_now_sec() : 0.0 \
    }

static void rocm_time_dump(void) {
    if (!rocm_time_on() || g_time_n_buckets == 0) return;
    /* Sort by descending total_sec via simple insertion sort. */
    for (int i = 1; i < g_time_n_buckets; i++) {
        struct rocm_time_bucket b = g_time_buckets[i];
        int j = i - 1;
        while (j >= 0 && g_time_buckets[j].total_sec < b.total_sec) {
            g_time_buckets[j + 1] = g_time_buckets[j];
            j--;
        }
        g_time_buckets[j + 1] = b;
    }
    double total = 0.0;
    for (int i = 0; i < g_time_n_buckets; i++) total += g_time_buckets[i].total_sec;
    fprintf(stderr, "ds4: ROCm timing histogram (%.3f sec total):\n", total);
    for (int i = 0; i < g_time_n_buckets; i++) {
        const struct rocm_time_bucket *b = &g_time_buckets[i];
        const double pct = total > 0.0 ? 100.0 * b->total_sec / total : 0.0;
        fprintf(stderr,
                "  %6.2f%%  %8.3f sec  %7llu calls  %8.3f ms/call  %s\n",
                pct, b->total_sec, (unsigned long long)b->count,
                1000.0 * b->total_sec / (double)b->count,
                b->name);
    }
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

/* Returns the registered device pointer for a host range if it lies inside a
 * pinned model-map registration; otherwise NULL.  A NULL return is the signal
 * to fall back to a temporary mapped upload. */
static uint8_t *rocm_registered_devptr(const void *host, uint64_t bytes) {
    if (!host || bytes == 0) return NULL;
    const uint8_t *h = (const uint8_t *)host;
    for (int i = 0; i < g_registration_count; i++) {
        const rocm_registration *r = &g_registrations[i];
        if (h >= r->host_base && (uint64_t)(h - r->host_base) <= r->size &&
            bytes <= r->size - (uint64_t)(h - r->host_base)) {
            return r->dev_base + (uint64_t)(h - r->host_base);
        }
    }
    return NULL;
}

/* Resolve src to a device pointer.  If src lies in a registered model-map
 * region, *host_out stays NULL (no allocation) and the caller must skip
 * hipHostFree.  Otherwise allocates a pinned mapped buffer, copies src into
 * it, and returns the device pointer.  Callers are expected to NULL-check
 * *host_out before freeing. */
static int rocm_upload_mapped(const void *src,
                              uint64_t bytes,
                              void **host_out,
                              void **dev_out,
                              const char *what) {
    *host_out = NULL;
    *dev_out = NULL;
    if (bytes == 0) return 0;
    uint8_t *registered = rocm_registered_devptr(src, bytes);
    if (registered) {
        *dev_out = registered;
        return 1;
    }
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
                                                   const uint16_t *table,
                                                   const int32_t *tokens,
                                                   uint32_t n_tokens,
                                                   uint32_t n_embd,
                                                   uint32_t n_hc,
                                                   uint32_t n_vocab) {
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t)n_tokens * n_hc * n_embd;
    if (idx >= total) return;
    const uint32_t embd = (uint32_t)(idx % n_embd);
    const uint64_t token_pos = idx / ((uint64_t)n_hc * n_embd);
    const int32_t tok = tokens[token_pos];
    if (tok < 0 || (uint32_t)tok >= n_vocab) {
        out[idx] = 0.0f;
        return;
    }
    __half_raw h;
    h.x = table[(uint64_t)tok * n_embd + embd];
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

/* Strix Halo (RDNA 3.5) optimised Q8_0 matmul.
 *
 *   - blockDim.x = N_ROWS_PER_BLOCK * 32: one wavefront per output row.
 *   - Each block computes N_ROWS_PER_BLOCK rows of one token.
 *   - x[tok, :] is cooperatively loaded into LDS once per block, then
 *     every wave reuses it from LDS instead of re-reading from global.
 *   - In-wave reduction uses __shfl_down (no LDS, no syncthreads).
 *
 * Constraints: in_dim divisible by 32 (true for all DSv4 matmuls -- the
 * scheduler aligns dims to DS4_QK_K=256), in_dim*sizeof(float) <= LDS
 * budget (we pick a safe ceiling of 16 KiB = 4096 floats per token).
 * The dispatcher falls back to the legacy kernel when those don't hold.
 *
 * Three concrete instantiations: 8 / 4 / 1 rows per block.  Plain
 * functions (not C++ templates) because this TU is wrapped in extern "C". */

static __device__ __forceinline__ float rocm_q8_0_block_dot(
        const uint8_t *__restrict__ blk, const float *__restrict__ x_lds_block) {
    const uint16_t scale_bits = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
    const int8_t *qs = (const int8_t *)(blk + 2);
    float dot = 0.0f;
    #pragma unroll
    for (int j = 0; j < 32; j++) dot += (float)qs[j] * x_lds_block[j];
    return rocm_f16_to_f32(scale_bits) * dot;
}

static __device__ __forceinline__ float rocm_wave32_sum(float v) {
    v += __shfl_down(v, 16);
    v += __shfl_down(v,  8);
    v += __shfl_down(v,  4);
    v += __shfl_down(v,  2);
    v += __shfl_down(v,  1);
    return v;
}

#define ROCM_DEFINE_MATMUL_Q8_0_LDS(NROWS) \
__global__ static void rocm_matmul_q8_0_lds_##NROWS##_kernel( \
        float *out, const uint8_t *w, const float *x, \
        uint32_t in_dim, uint32_t out_dim, uint64_t row_bytes) { \
    extern __shared__ float x_lds[]; \
    const uint32_t tid     = threadIdx.x; \
    const uint32_t row_idx = tid >> 5; \
    const uint32_t lane    = tid & 31u; \
    const uint32_t row     = blockIdx.x * (uint32_t)(NROWS) + row_idx; \
    const uint32_t tok     = blockIdx.y; \
    const float *xr = x + (uint64_t)tok * in_dim; \
    for (uint32_t i = tid; i < in_dim; i += (uint32_t)blockDim.x) x_lds[i] = xr[i]; \
    __syncthreads(); \
    if (row >= out_dim) return; \
    const uint32_t blocks = in_dim >> 5; \
    const uint8_t *wr = w + (uint64_t)row * row_bytes; \
    float sum = 0.0f; \
    for (uint32_t b = lane; b < blocks; b += 32u) { \
        sum += rocm_q8_0_block_dot(wr + (uint64_t)b * 34u, x_lds + (b << 5)); \
    } \
    sum = rocm_wave32_sum(sum); \
    if (lane == 0) out[(uint64_t)tok * out_dim + row] = sum; \
}

ROCM_DEFINE_MATMUL_Q8_0_LDS(8)
ROCM_DEFINE_MATMUL_Q8_0_LDS(4)
ROCM_DEFINE_MATMUL_Q8_0_LDS(1)

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
        /* Match Metal kernel_dsv4_hc_split_weighted_sum_norm4: do one col-norm
         * after the row-softmax-with-eps init, then (sinkhorn_iters - 1) more
         * iterations of (row, col).  The fused-loop variant we used to have did
         * an extra row-norm before the first col-norm, which silently diverged
         * from Metal numerics across 43+ layers. */
        for (uint32_t j = 0; j < 4; j++) {
            float col_sum = eps;
            for (uint32_t i = 0; i < 4; i++) col_sum += r[i][j];
            for (uint32_t i = 0; i < 4; i++) r[i][j] /= col_sum;
        }
        for (uint32_t iter = 1; iter < sinkhorn_iters; iter++) {
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
    rocm_drain_pending_frees();
    rocm_time_dump();
    for (int i = 0; i < g_registration_count; i++) {
        if (g_registrations[i].host_owned) {
            (void)hipHostUnregister(g_registrations[i].host_owned);
        }
        if (g_registrations[i].dev_owned) {
            (void)hipFree(g_registrations[i].dev_owned);
        }
        g_registrations[i].host_base = NULL;
        g_registrations[i].dev_base = NULL;
        g_registrations[i].size = 0;
        g_registrations[i].host_owned = NULL;
        g_registrations[i].dev_owned = NULL;
    }
    g_registration_count = 0;
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
    /* Reads need fresh data: drain any in-flight kernels we deferred in
     * batched mode before sampling the tensor, then release any pending
     * temporary host allocations that those kernels were still using.
     * This is the only place outside the explicit batch boundaries where
     * CPU consumes results. */
    if (g_batch_open && !rocm_sync_each()) {
        if (!rocm_ok(hipDeviceSynchronize(), "tensor read drain")) return 0;
        rocm_drain_pending_frees();
    }
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
    int ok = rocm_ok(hipDeviceSynchronize(), "command flush");
    rocm_drain_pending_frees();
    return ok;
}

int ds4_metal_end_commands(void) {
    if (!g_batch_open) return 0;
    g_batch_open = 0;
    int ok = rocm_ok(hipDeviceSynchronize(), "command completion");
    rocm_drain_pending_frees();
    return ok;
}

int ds4_metal_synchronize(void) {
    if (!g_initialized && !ds4_metal_init()) return 0;
    int ok = rocm_ok(hipDeviceSynchronize(), "synchronize");
    rocm_drain_pending_frees();
    return ok;
}

/* Deferred-free queue: temporary hipHostMalloc'd staging buffers (router
 * bias, attention sinks, MoE expert pools, weight-upload fallbacks) used
 * to be hipHostFree'd immediately after their kernel was launched.  In
 * batched mode the launch returns before the kernel runs, so an immediate
 * free races the queued kernel.  Production paths almost never hit the
 * temp-buffer path (rocm_upload_mapped finds the registered model map
 * first), so this is mostly defense-in-depth, but we still need it
 * correct for tests and any future caller that bypasses the registration
 * cache. */
#define ROCM_PENDING_FREE_CAP 4096
static void *g_pending_frees[ROCM_PENDING_FREE_CAP];
static int   g_pending_n;

static void rocm_defer_free(void *host_ptr) {
    if (!host_ptr) return;
    if (!g_batch_open || rocm_sync_each()) {
        (void)hipHostFree(host_ptr);
        return;
    }
    if (g_pending_n >= ROCM_PENDING_FREE_CAP) {
        /* Queue full: flush stream and drain so we can reuse the slot. */
        (void)hipDeviceSynchronize();
        for (int i = 0; i < g_pending_n; i++) (void)hipHostFree(g_pending_frees[i]);
        g_pending_n = 0;
    }
    g_pending_frees[g_pending_n++] = host_ptr;
}

static void rocm_drain_pending_frees(void) {
    if (g_pending_n == 0) return;
    /* Caller is responsible for syncing the stream before calling us. */
    for (int i = 0; i < g_pending_n; i++) (void)hipHostFree(g_pending_frees[i]);
    g_pending_n = 0;
}

/* Per-kernel launch validation.
 *
 * Historically every wrapper finished with `rocm_ok(hipGetLastError(), ...) &&
 * rocm_ok(hipDeviceSynchronize(), ...)` -- a full device-wide barrier after
 * every single kernel.  With ~30k kernel launches per inference, that's a
 * massive amount of CPU/GPU drain time that scales with launch count, not
 * useful work.  In batched mode we now only check the launch's
 * hipGetLastError and rely on default-stream FIFO ordering to keep
 * dependent kernels correct; the explicit sync is deferred to the
 * begin/flush/end/synchronize boundaries that ds4.c already brackets graphs
 * with.  Set DS4_ROCM_SYNC_EACH=1 to restore per-kernel sync (debug). */
static int g_sync_each = -1;
static int rocm_sync_each(void) {
    if (g_sync_each < 0) {
        const char *v = getenv("DS4_ROCM_SYNC_EACH");
        g_sync_each = (v && v[0] && v[0] != '0') ? 1 : 0;
    }
    return g_sync_each;
}

static int rocm_launch_done(const char *launch_label, const char *sync_label) {
    if (!rocm_ok(hipGetLastError(), launch_label)) return 0;
    if (g_batch_open && !rocm_sync_each()) return 1;
    return rocm_ok(hipDeviceSynchronize(), sync_label);
}

int ds4_metal_set_model_map_range(const void *model_map, uint64_t model_size,
                                  uint64_t map_offset, uint64_t map_size);

/* Walk /proc/self/maps to find the file backing a host pointer.  Returns the
 * length of the path written to out (0 if no file-backed mapping covers
 * addr).  Used to recover the source GGUF path so we can pread() it at full
 * disk bandwidth instead of relying on synchronous mmap page faults. */
static int rocm_resolve_mmap_path(const void *addr, char *out, size_t out_len) {
    if (!out || out_len == 0) return 0;
    out[0] = '\0';
    FILE *f = fopen("/proc/self/maps", "r");
    if (!f) return 0;
    const uintptr_t target = (uintptr_t)addr;
    char line[4096];
    int found = 0;
    while (fgets(line, sizeof(line), f)) {
        unsigned long lo, hi;
        if (sscanf(line, "%lx-%lx", &lo, &hi) != 2) continue;
        if (target < lo || target >= hi) continue;
        char *path = strchr(line, '/');
        if (!path) break;
        char *nl = strchr(path, '\n');
        if (nl) *nl = '\0';
        const size_t n = strlen(path);
        if (n + 1 > out_len) break;
        memcpy(out, path, n + 1);
        found = (int)n;
        break;
    }
    fclose(f);
    return found;
}

int ds4_metal_set_model_map(const void *model_map, uint64_t model_size) {
    return ds4_metal_set_model_map_range(model_map, model_size, 0, model_size);
}

int ds4_metal_set_model_map_range(const void *model_map, uint64_t model_size,
                                  uint64_t map_offset, uint64_t map_size) {
    if (!g_initialized && !ds4_metal_init()) return 0;
    if (!model_map || map_size == 0) return 0;
    if (map_offset > model_size || map_size > model_size - map_offset) return 0;
    if (g_registration_count >= DS4_ROCM_MAX_REGISTRATIONS) {
        fprintf(stderr, "ds4: ROCm registration table full (max %d)\n",
                DS4_ROCM_MAX_REGISTRATIONS);
        return 0;
    }
    uint8_t *base = (uint8_t *)((uintptr_t)model_map + (uintptr_t)map_offset);
    for (int i = 0; i < g_registration_count; i++) {
        if (g_registrations[i].host_base == base &&
            g_registrations[i].size == map_size) {
            return 1;
        }
    }
    rocm_registration *r = &g_registrations[g_registration_count];

    /* First try host pinning -- cheapest path, no device-side memcpy.  Caps
     * out at the system memlock budget (Strix Halo BIOS reserves most of
     * UMA for the iGPU, so only ~30 GB is host-pinnable in practice). */
    if (hipHostRegister(base, (size_t)map_size, hipHostRegisterMapped) == hipSuccess) {
        void *dev_base = NULL;
        if (!rocm_ok(hipHostGetDevicePointer(&dev_base, base, 0),
                     "model map device pointer")) {
            (void)hipHostUnregister(base);
            return 0;
        }
        r->host_base  = base;
        r->dev_base   = (uint8_t *)dev_base;
        r->size       = map_size;
        r->host_owned = base;
        r->dev_owned  = NULL;
        g_registration_count++;
        if (rocm_trace()) {
            fprintf(stderr,
                    "ds4: ROCm pinned model map host=%p dev=%p bytes=%llu\n",
                    (const void *)base, (const void *)dev_base,
                    (unsigned long long)map_size);
        }
        return 1;
    }
    (void)hipGetLastError();  /* Clear sticky error from the failed attempt. */

    /* Fall back to a device-resident copy.  Allocates from the GPU's UMA
     * partition (the 96 GB reserved by firmware), then streams the mmap'd
     * file bytes in via hipMemcpy.  One-time cost at engine startup. */
    void *dev_alloc = NULL;
    if (!rocm_ok(hipMalloc(&dev_alloc, (size_t)map_size),
                 "model map device allocation")) {
        return 0;
    }
    fprintf(stderr,
            "ds4: ROCm copying %.2f GiB of model into device memory (one-time at startup)...\n",
            (double)map_size / (1024.0 * 1024.0 * 1024.0));
    /* Recover the underlying file path from /proc/self/maps so we can read
     * with explicit pread().  hipMemcpy from the mmap'd source bottlenecks
     * because each 4 KiB page faults synchronously before DMA starts; an
     * explicit read() into a pinned staging buffer issues the disk reads
     * at full NVMe bandwidth and overlaps with the device copy. */
    char src_path[4096] = {0};
    const int fd = rocm_resolve_mmap_path(base, src_path, sizeof(src_path)) > 0
        ? open(src_path, O_RDONLY)
        : -1;
    /* Pinned staging buffer.  Sized to amortize per-call overhead; on
     * Strix Halo (UMA) the H2D "DMA" is just a memcpy from pinned host
     * to GPU partition, so adding parallelism between disk reads and
     * H2D doesn't add bandwidth -- they share the same memory bus.
     * Sequential single-buffer with kernel-driven readahead measured
     * fastest in A/B testing (~2.7 GiB/s vs ~2.2 GiB/s for a 3-buffer
     * threaded pipeline that paid for extra pinned RAM with no win). */
    const uint64_t stage_bytes = 256ull * 1024ull * 1024ull;
    void *stage_host = NULL;
    if (fd >= 0 && hipHostMalloc(&stage_host, (size_t)stage_bytes,
                                 hipHostMallocDefault) != hipSuccess) {
        stage_host = NULL;
    }
    int load_ok = 0;
    if (fd >= 0 && stage_host) {
        (void)posix_fadvise(fd, 0, (off_t)map_size, POSIX_FADV_SEQUENTIAL);
        (void)posix_fadvise(fd, 0, (off_t)map_size, POSIX_FADV_WILLNEED);
        const off_t base_off = (off_t)((const uint8_t *)base -
                                        (const uint8_t *)model_map);
        struct timespec t_start;
        clock_gettime(CLOCK_MONOTONIC, &t_start);
        load_ok = 1;
        for (uint64_t off = 0; off < map_size; off += stage_bytes) {
            const uint64_t n = (map_size - off < stage_bytes)
                ? (map_size - off) : stage_bytes;
            ssize_t got = pread(fd, stage_host, (size_t)n,
                                base_off + (off_t)off);
            if (got != (ssize_t)n) {
                fprintf(stderr,
                        "ds4: ROCm model read at offset %llu got %zd of %llu bytes\n",
                        (unsigned long long)(base_off + (off_t)off),
                        got, (unsigned long long)n);
                load_ok = 0;
                break;
            }
            if (!rocm_ok(hipMemcpy((uint8_t *)dev_alloc + off,
                                   stage_host, (size_t)n,
                                   hipMemcpyHostToDevice),
                         "model map staged copy")) {
                load_ok = 0;
                break;
            }
            if ((off + n) % (4ull * stage_bytes) == 0 || off + n == map_size) {
                struct timespec t_now;
                clock_gettime(CLOCK_MONOTONIC, &t_now);
                const double elapsed =
                    (double)(t_now.tv_sec - t_start.tv_sec) +
                    1.0e-9 * (double)(t_now.tv_nsec - t_start.tv_nsec);
                const double gibps = elapsed > 0.0
                    ? ((double)(off + n) / (1024.0 * 1024.0 * 1024.0)) / elapsed
                    : 0.0;
                fprintf(stderr,
                        "ds4: ROCm model copy %llu / %llu MiB (%.2f GiB/s avg)\n",
                        (unsigned long long)((off + n) >> 20),
                        (unsigned long long)(map_size >> 20), gibps);
            }
        }
    }
    if (fd >= 0) (void)close(fd);
    if (stage_host) (void)hipHostFree(stage_host);

    if (!load_ok) {
        /* Fallback: source mmap is the only thing we can read from.  Used
         * when /proc/self/maps lookup fails (e.g. test buffers that
         * aren't backed by a file). */
        (void)madvise(base, (size_t)map_size, MADV_SEQUENTIAL);
        (void)madvise(base, (size_t)map_size, MADV_WILLNEED);
        const uint64_t chunk = 1ull * 1024ull * 1024ull * 1024ull;
        for (uint64_t off = 0; off < map_size; off += chunk) {
            const uint64_t n = (map_size - off < chunk) ? (map_size - off) : chunk;
            if (!rocm_ok(hipMemcpy((uint8_t *)dev_alloc + off,
                                   base + off, (size_t)n,
                                   hipMemcpyHostToDevice),
                         "model map device copy")) {
                (void)hipFree(dev_alloc);
                return 0;
            }
            if ((off + n) % (8ull * chunk) == 0 || off + n == map_size) {
                fprintf(stderr, "ds4: ROCm model copy %llu / %llu MiB\n",
                        (unsigned long long)((off + n) >> 20),
                        (unsigned long long)(map_size >> 20));
            }
        }
        (void)madvise(base, (size_t)map_size, MADV_DONTNEED);
    }
    r->host_base  = base;
    r->dev_base   = (uint8_t *)dev_alloc;
    r->size       = map_size;
    r->host_owned = NULL;
    r->dev_owned  = dev_alloc;
    g_registration_count++;
    if (rocm_trace()) {
        fprintf(stderr,
                "ds4: ROCm device-mirrored model map host=%p dev=%p bytes=%llu\n",
                (const void *)base, (const void *)dev_alloc,
                (unsigned long long)map_size);
    }
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
    if (rocm_trace()) fprintf(stderr, "ds4: ROCm enter ds4_metal_embed_token_hc_tensor\n");
    ROCM_TIME_SCOPE("ds4_metal_embed_token_hc_tensor");
    if (!g_initialized && !ds4_metal_init()) return 0;
    if (!out_hc || !model_map || n_vocab == 0 || token >= n_vocab || n_embd == 0 || n_hc == 0) return 0;
    const uint64_t out_bytes = (uint64_t)n_embd * n_hc * sizeof(float);
    const uint64_t row_bytes = (uint64_t)n_embd * sizeof(uint16_t);
    const uint64_t weight_bytes = (uint64_t)n_vocab * row_bytes;
    if (out_hc->bytes < out_bytes || weight_offset > model_size || weight_bytes > model_size - weight_offset) return 0;

    const uint8_t *src = (const uint8_t *)model_map + weight_offset + (uint64_t)token * row_bytes;
    void *row_host = NULL, *row_dev = NULL;
    if (!rocm_upload_mapped(src, row_bytes, &row_host, &row_dev, "embed token row")) return 0;
    const uint64_t total = (uint64_t)n_embd * n_hc;
    const dim3 block(256);
    const dim3 grid((unsigned)((total + block.x - 1u) / block.x));
    hipLaunchKernelGGL(rocm_embed_token_hc_kernel,
                       grid,
                       block,
                       0,
                       0,
                       (float *)tensor_u8(out_hc),
                       (const uint16_t *)row_dev,
                       n_embd,
                       n_hc);
    int ok = rocm_launch_done("embed token launch", "embed token completion");
    rocm_defer_free(row_host);
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
    if (rocm_trace()) fprintf(stderr, "ds4: ROCm enter ds4_metal_embed_tokens_hc_tensor\n");
    ROCM_TIME_SCOPE("ds4_metal_embed_tokens_hc_tensor");
    if (!g_initialized && !ds4_metal_init()) return 0;
    if (!out_hc || !tokens || !model_map || n_vocab == 0 || n_tokens == 0 || n_embd == 0 || n_hc == 0) return 0;
    const uint64_t out_bytes = (uint64_t)n_tokens * n_hc * n_embd * sizeof(float);
    const uint64_t token_bytes = (uint64_t)n_tokens * sizeof(int32_t);
    const uint64_t row_bytes = (uint64_t)n_embd * sizeof(uint16_t);
    const uint64_t weight_bytes = (uint64_t)n_vocab * row_bytes;
    if (out_hc->bytes < out_bytes || tokens->bytes < token_bytes ||
        weight_offset > model_size || weight_bytes > model_size - weight_offset) return 0;

    const uint8_t *weights = (const uint8_t *)model_map + weight_offset;
    void *table_host = NULL, *table_dev = NULL;
    if (!rocm_upload_mapped(weights, weight_bytes, &table_host, &table_dev,
                            "embed tokens table")) return 0;
    const uint64_t total = (uint64_t)n_tokens * n_hc * n_embd;
    const dim3 block(256);
    const dim3 grid((unsigned)((total + block.x - 1u) / block.x));
    hipLaunchKernelGGL(rocm_embed_tokens_hc_kernel,
                       grid,
                       block,
                       0,
                       0,
                       (float *)tensor_u8(out_hc),
                       (const uint16_t *)table_dev,
                       (const int32_t *)tensor_u8_const(tokens),
                       n_tokens,
                       n_embd,
                       n_hc,
                       n_vocab);
    int ok = rocm_launch_done("embed tokens launch", "embed tokens completion");
    rocm_defer_free(table_host);
    return ok;
}

int ds4_metal_repeat_hc_tensor(
        ds4_metal_tensor       *out,
        const ds4_metal_tensor *row,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (rocm_trace()) fprintf(stderr, "ds4: ROCm enter ds4_metal_repeat_hc_tensor\n");
    ROCM_TIME_SCOPE("ds4_metal_repeat_hc_tensor");
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
    return rocm_launch_done("repeat launch", "repeat completion");
}

int ds4_metal_rms_norm_plain_tensor(
        ds4_metal_tensor       *out,
        const ds4_metal_tensor *x,
        uint32_t                n,
        float                   eps) {
    if (rocm_trace()) fprintf(stderr, "ds4: ROCm enter ds4_metal_rms_norm_plain_tensor\n");
    ROCM_TIME_SCOPE("ds4_metal_rms_norm_plain_tensor");
    return ds4_metal_rms_norm_plain_rows_tensor(out, x, n, 1, eps);
}

int ds4_metal_rms_norm_plain_rows_tensor(
        ds4_metal_tensor       *out,
        const ds4_metal_tensor *x,
        uint32_t                n,
        uint32_t                rows,
        float                   eps) {
    if (rocm_trace()) fprintf(stderr, "ds4: ROCm enter ds4_metal_rms_norm_plain_rows_tensor\n");
    ROCM_TIME_SCOPE("ds4_metal_rms_norm_plain_rows_tensor");
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
    return rocm_launch_done("plain RMS norm launch", "plain RMS norm completion");
}

int ds4_metal_swiglu_tensor(
        ds4_metal_tensor       *out,
        const ds4_metal_tensor *gate,
        const ds4_metal_tensor *up,
        uint32_t                n,
        float                   clamp,
        float                   weight) {
    if (rocm_trace()) fprintf(stderr, "ds4: ROCm enter ds4_metal_swiglu_tensor\n");
    ROCM_TIME_SCOPE("ds4_metal_swiglu_tensor");
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
    return rocm_launch_done("SwiGLU launch", "SwiGLU completion");
}

int ds4_metal_add_tensor(
        ds4_metal_tensor       *out,
        const ds4_metal_tensor *a,
        const ds4_metal_tensor *b,
        uint32_t                n) {
    if (rocm_trace()) fprintf(stderr, "ds4: ROCm enter ds4_metal_add_tensor\n");
    ROCM_TIME_SCOPE("ds4_metal_add_tensor");
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
    return rocm_launch_done("add launch", "add completion");
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
    if (rocm_trace()) fprintf(stderr, "ds4: ROCm enter ds4_metal_matmul_f16_tensor\n");
    ROCM_TIME_SCOPE("ds4_metal_matmul_f16_tensor");
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
    int ok = rocm_launch_done("F16 matmul launch", "F16 matmul completion");
    rocm_defer_free(w_host);
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
    if (rocm_trace()) fprintf(stderr, "ds4: ROCm enter ds4_metal_matmul_f16_pair_tensor\n");
    ROCM_TIME_SCOPE("ds4_metal_matmul_f16_pair_tensor");
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
    if (rocm_trace()) fprintf(stderr, "ds4: ROCm enter ds4_metal_matmul_f32_tensor\n");
    ROCM_TIME_SCOPE("ds4_metal_matmul_f32_tensor");
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
    int ok = rocm_launch_done("F32 matmul launch", "F32 matmul completion");
    rocm_defer_free(w_host);
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
    if (rocm_trace()) fprintf(stderr, "ds4: ROCm enter ds4_metal_matmul_q8_0_tensor\n");
    ROCM_TIME_SCOPE("ds4_metal_matmul_q8_0_tensor");
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
                       grid, block, block.x * sizeof(float), 0,
                       (float *)tensor_u8(out),
                       (const uint8_t *)w_dev,
                       (const float *)tensor_u8_const(x),
                       in_dim, out_dim, row_bytes);
    int ok = rocm_launch_done("Q8_0 matmul launch", "Q8_0 matmul completion");
    rocm_defer_free(w_host);
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
    if (rocm_trace()) fprintf(stderr, "ds4: ROCm enter ds4_metal_rms_norm_weight_tensor\n");
    ROCM_TIME_SCOPE("ds4_metal_rms_norm_weight_tensor");
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
    if (rocm_trace()) fprintf(stderr, "ds4: ROCm enter ds4_metal_rms_norm_weight_rows_tensor\n");
    ROCM_TIME_SCOPE("ds4_metal_rms_norm_weight_rows_tensor");
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
    int ok = rocm_launch_done("weighted RMS norm launch", "weighted RMS norm completion");
    rocm_defer_free(w_host);
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
    if (rocm_trace()) fprintf(stderr, "ds4: ROCm enter ds4_metal_dsv4_qkv_rms_norm_rows_tensor\n");
    ROCM_TIME_SCOPE("ds4_metal_dsv4_qkv_rms_norm_rows_tensor");
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
    if (rocm_trace()) fprintf(stderr, "ds4: ROCm enter ds4_metal_head_rms_norm_tensor\n");
    ROCM_TIME_SCOPE("ds4_metal_head_rms_norm_tensor");
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
    if (rocm_trace()) fprintf(stderr, "ds4: ROCm enter ds4_metal_hc_split_weighted_sum_norm_tensor\n");
    ROCM_TIME_SCOPE("ds4_metal_hc_split_weighted_sum_norm_tensor");
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
        rocm_defer_free(scale_host);
        rocm_defer_free(base_host);
        rocm_defer_free(norm_host);
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
    ok = rocm_launch_done("HC split/sum/norm launch", "HC split/sum/norm completion");
    rocm_defer_free(scale_host);
    rocm_defer_free(base_host);
    rocm_defer_free(norm_host);
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
    if (rocm_trace()) fprintf(stderr, "ds4: ROCm enter ds4_metal_rope_tail_tensor\n");
    ROCM_TIME_SCOPE("ds4_metal_rope_tail_tensor");
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
    return rocm_launch_done("RoPE tail launch", "RoPE tail completion");
}

int ds4_metal_dsv4_fp8_kv_quantize_tensor(
        ds4_metal_tensor *x,
        uint32_t          n_tok,
        uint32_t          head_dim,
        uint32_t          n_rot) {
    if (rocm_trace()) fprintf(stderr, "ds4: ROCm enter ds4_metal_dsv4_fp8_kv_quantize_tensor\n");
    ROCM_TIME_SCOPE("ds4_metal_dsv4_fp8_kv_quantize_tensor");
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
    return rocm_launch_done("FP8 KV quantize launch", "FP8 KV quantize completion");
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
    return rocm_launch_done(label, label);
}

int ds4_metal_hc_weighted_sum_tensor(
        ds4_metal_tensor       *out,
        const ds4_metal_tensor *residual_hc,
        const ds4_metal_tensor *weights,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (rocm_trace()) fprintf(stderr, "ds4: ROCm enter ds4_metal_hc_weighted_sum_tensor\n");
    ROCM_TIME_SCOPE("ds4_metal_hc_weighted_sum_tensor");
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
    if (rocm_trace()) fprintf(stderr, "ds4: ROCm enter ds4_metal_hc_weighted_sum_split_tensor\n");
    ROCM_TIME_SCOPE("ds4_metal_hc_weighted_sum_split_tensor");
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
    if (rocm_trace()) fprintf(stderr, "ds4: ROCm enter ds4_metal_hc_split_sinkhorn_tensor\n");
    ROCM_TIME_SCOPE("ds4_metal_hc_split_sinkhorn_tensor");
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
        rocm_defer_free(sh);
        rocm_defer_free(bh);
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
    ok = rocm_launch_done("HC sinkhorn launch", "HC sinkhorn completion");
    rocm_defer_free(sh);
    rocm_defer_free(bh);
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
    if (rocm_trace()) fprintf(stderr, "ds4: ROCm enter ds4_metal_hc_split_weighted_sum_tensor\n");
    ROCM_TIME_SCOPE("ds4_metal_hc_split_weighted_sum_tensor");
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
    if (rocm_trace()) fprintf(stderr, "ds4: ROCm enter ds4_metal_output_hc_weights_tensor\n");
    ROCM_TIME_SCOPE("ds4_metal_output_hc_weights_tensor");
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
    ok = rocm_launch_done("output_hc launch", "output_hc completion");
    rocm_defer_free(sh);
    rocm_defer_free(bh);
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
    return rocm_launch_done(label, label);
}

int ds4_metal_hc_expand_tensor(
        ds4_metal_tensor       *out_hc,
        const ds4_metal_tensor *block_out,
        const ds4_metal_tensor *residual_hc,
        const ds4_metal_tensor *post,
        const ds4_metal_tensor *comb,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (rocm_trace()) fprintf(stderr, "ds4: ROCm enter ds4_metal_hc_expand_tensor\n");
    ROCM_TIME_SCOPE("ds4_metal_hc_expand_tensor");
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
    if (rocm_trace()) fprintf(stderr, "ds4: ROCm enter ds4_metal_hc_expand_split_tensor\n");
    ROCM_TIME_SCOPE("ds4_metal_hc_expand_split_tensor");
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
    if (rocm_trace()) fprintf(stderr, "ds4: ROCm enter ds4_metal_hc_expand_add_split_tensor\n");
    ROCM_TIME_SCOPE("ds4_metal_hc_expand_add_split_tensor");
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
    if (rocm_trace()) fprintf(stderr, "ds4: ROCm enter ds4_metal_store_raw_kv_tensor\n");
    ROCM_TIME_SCOPE("ds4_metal_store_raw_kv_tensor");
    if (!g_initialized && !ds4_metal_init()) return 0;
    if (!raw_cache || !kv || raw_cap == 0 || row >= raw_cap || head_dim == 0) return 0;
    const uint64_t needed = ((uint64_t)row + 1u) * head_dim * sizeof(float);
    if (raw_cache->bytes < needed || kv->bytes < (uint64_t)head_dim * sizeof(float)) return 0;
    float *raw = (float *)tensor_u8(raw_cache) + (uint64_t)row * head_dim;
    const dim3 block(256);
    const dim3 grid((head_dim + block.x - 1u) / block.x);
    hipLaunchKernelGGL(rocm_store_raw_kv_kernel, grid, block, 0, 0,
                       raw, (const float *)tensor_u8_const(kv), head_dim);
    return rocm_launch_done("store raw kv launch", "store raw kv completion");
}

int ds4_metal_store_raw_kv_batch_tensor(
        ds4_metal_tensor       *raw_cache,
        const ds4_metal_tensor *kv,
        uint32_t                raw_cap,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                head_dim) {
    if (rocm_trace()) fprintf(stderr, "ds4: ROCm enter ds4_metal_store_raw_kv_batch_tensor\n");
    ROCM_TIME_SCOPE("ds4_metal_store_raw_kv_batch_tensor");
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
    if (rocm_trace()) fprintf(stderr, "ds4: ROCm enter ds4_metal_kv_fp8_store_raw_tensor\n");
    ROCM_TIME_SCOPE("ds4_metal_kv_fp8_store_raw_tensor");
    if (!g_initialized && !ds4_metal_init()) return 0;
    if (!kv || !raw_cache || head_dim == 0 || n_rot > head_dim || row >= raw_cap) return 0;
    if (kv->bytes < (uint64_t)head_dim * sizeof(float)) return 0;
    if (raw_cache->bytes < ((uint64_t)row + 1u) * head_dim * sizeof(float)) return 0;
    float *raw = (float *)tensor_u8(raw_cache) + (uint64_t)row * head_dim;
    hipLaunchKernelGGL(rocm_kv_fp8_store_raw_kernel, dim3(1), dim3(64), 0, 0,
                       (float *)tensor_u8(kv), raw, head_dim, n_rot);
    return rocm_launch_done("kv fp8 store launch", "kv fp8 store completion");
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
    if (rocm_trace()) fprintf(stderr, "ds4: ROCm enter ds4_metal_shared_gate_up_swiglu_q8_0_tensor\n");
    ROCM_TIME_SCOPE("ds4_metal_shared_gate_up_swiglu_q8_0_tensor");
    if (!gate || !up || !mid) return 0;
    if (!ds4_metal_matmul_q8_0_tensor(gate, model_map, model_size, gate_offset, in_dim, out_dim, x, 1)) return 0;
    if (!ds4_metal_matmul_q8_0_tensor(up, model_map, model_size, up_offset, in_dim, out_dim, x, 1)) return 0;
    return ds4_metal_swiglu_tensor(mid, gate, up, (uint32_t)out_dim, 0.0f, 1.0f);
}

/* =========================================================================
 * Attention output projections (Q8_0).
 * ========================================================================= */

/* Fused per-group Q8_0 matvec for the attention output low projection.
 *
 *   grid = (rank, n_groups, n_tokens)
 *   block = 256 threads = 8 wavefronts on RDNA 3.5
 *
 * Each block computes one output element low[t, g, r] by partial-summing
 * across the group_dim axis with all 256 threads, finishing with a
 * shared-memory reduction.  Replaces the n_groups CPU loop launching
 * one matmul_q8_0_tensor per group, which paid the per-launch overhead
 * 12-128x per call. */
__global__ static void rocm_attn_out_low_q8_fused_kernel(
        float *low,
        const uint8_t *w,
        const float *heads,
        uint32_t group_dim,
        uint32_t rank,
        uint32_t n_groups,
        uint64_t row_bytes,
        uint64_t group_w_bytes) {
    extern __shared__ float sh[];
    const uint32_t r   = blockIdx.x;
    const uint32_t g   = blockIdx.y;
    const uint32_t tok = blockIdx.z;
    const uint32_t tid = threadIdx.x;
    const uint32_t bs  = blockDim.x;
    if (r >= rank) return;

    const float   *xr  = heads + ((uint64_t)tok * n_groups + g) * group_dim;
    const uint8_t *wr  = w + (uint64_t)g * group_w_bytes + (uint64_t)r * row_bytes;
    const uint32_t qblocks = (group_dim + 31u) >> 5;

    float sum = 0.0f;
    for (uint32_t b = tid; b < qblocks; b += bs) {
        const uint8_t *blk = wr + (uint64_t)b * 34u;
        const uint16_t scale_bits = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
        const int8_t *qs = (const int8_t *)(blk + 2);
        const uint32_t i0 = b << 5;
        const uint32_t n  = (i0 + 32u <= group_dim) ? 32u : (group_dim - i0);
        float dot = 0.0f;
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            if ((uint32_t)j < n) dot += (float)qs[j] * xr[i0 + j];
        }
        sum += rocm_f16_to_f32(scale_bits) * dot;
    }
    sh[tid] = sum;
    __syncthreads();
    for (uint32_t stride = bs / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) sh[tid] += sh[tid + stride];
        __syncthreads();
    }
    if (tid == 0) low[((uint64_t)tok * n_groups + g) * rank + r] = sh[0];
}

static int rocm_attention_output_low_q8_dispatch(
        ds4_metal_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        uint32_t                n_tokens,
        const ds4_metal_tensor *heads) {
    if (!low || !heads || group_dim == 0 || rank == 0 || n_groups == 0 || n_tokens == 0) return 0;
    const uint64_t per_token_out = (uint64_t)n_groups * rank;
    const uint64_t per_token_in  = (uint64_t)n_groups * group_dim;
    if (low->bytes  < (uint64_t)n_tokens * per_token_out * sizeof(float)) return 0;
    if (heads->bytes < (uint64_t)n_tokens * per_token_in  * sizeof(float)) return 0;
    const uint64_t row_bytes = ((group_dim + 31u) / 32u) * 34u;
    const uint64_t group_w_bytes = row_bytes * rank;
    if (out_a_offset > model_size ||
        (uint64_t)n_groups * group_w_bytes > model_size - out_a_offset) return 0;

    void *w_host = NULL;
    void *w_dev  = NULL;
    if (!rocm_upload_mapped((const uint8_t *)model_map + out_a_offset,
                             (uint64_t)n_groups * group_w_bytes,
                             &w_host, &w_dev,
                             "attn out low Q8 weight upload")) return 0;

    /* Same blockDim=256 as the legacy matmul_q8_0_kernel; on this host the
     * 8-wave block hides scalar-reduction latency well even if the inner
     * loop has half the threads idle for tiny qblocks. */
    const uint32_t threads = 256u;
    dim3 grid((unsigned)rank, (unsigned)n_groups, (unsigned)n_tokens);
    dim3 block(threads, 1, 1);
    hipLaunchKernelGGL(rocm_attn_out_low_q8_fused_kernel,
                       grid, block, threads * sizeof(float), 0,
                       (float *)tensor_u8(low),
                       (const uint8_t *)w_dev,
                       (const float *)tensor_u8_const(heads),
                       (uint32_t)group_dim, (uint32_t)rank, n_groups,
                       row_bytes, group_w_bytes);
    int ok = rocm_launch_done("attn out low Q8 launch",
                              "attn out low Q8 completion");
    rocm_defer_free(w_host);
    return ok;
}

int ds4_metal_attention_output_low_q8_tensor(
        ds4_metal_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        const ds4_metal_tensor *heads) {
    if (rocm_trace()) fprintf(stderr, "ds4: ROCm enter ds4_metal_attention_output_low_q8_tensor\n");
    ROCM_TIME_SCOPE("ds4_metal_attention_output_low_q8_tensor");
    return rocm_attention_output_low_q8_dispatch(low, model_map, model_size,
                                                 out_a_offset, group_dim, rank,
                                                 n_groups, 1, heads);
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
    if (rocm_trace()) fprintf(stderr, "ds4: ROCm enter ds4_metal_attention_output_q8_batch_tensor\n");
    ROCM_TIME_SCOPE("ds4_metal_attention_output_q8_batch_tensor");
    /* Two stages:
     *   low = Wa @ heads  (per-group Q8 matvec, batched across tokens)
     *   out = Wb @ low    (single Q8 matvec of dim n_groups*rank -> out_dim)
     * Used to be a per-token CPU loop launching 2 kernels each iteration; for
     * a 2048-token prefill chunk that was 4096 launches per layer.  Both
     * stages now batch across n_tokens with a single launch each. */
    (void)group_tmp;
    (void)low_tmp;
    if (!out || !low || !heads || n_tokens == 0) return 0;
    const uint64_t low_per = (uint64_t)n_groups * rank;
    if (out->bytes < (uint64_t)n_tokens * out_dim * sizeof(float)) return 0;
    if (low->bytes < (uint64_t)n_tokens * low_per * sizeof(float)) return 0;
    if (heads->bytes < (uint64_t)n_tokens * n_groups * group_dim * sizeof(float)) return 0;

    if (!rocm_attention_output_low_q8_dispatch(low, model_map, model_size,
                                               out_a_offset, group_dim, rank,
                                               n_groups, n_tokens, heads)) return 0;
    return ds4_metal_matmul_q8_0_tensor(out, model_map, model_size, out_b_offset,
                                        low_per, out_dim, low, n_tokens);
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
    if (rocm_trace()) fprintf(stderr, "ds4: ROCm enter ds4_metal_matmul_q8_0_hc_expand_tensor\n");
    ROCM_TIME_SCOPE("ds4_metal_matmul_q8_0_hc_expand_tensor");
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
    if (rocm_trace()) fprintf(stderr, "ds4: ROCm enter ds4_metal_shared_down_hc_expand_q8_0_tensor\n");
    ROCM_TIME_SCOPE("ds4_metal_shared_down_hc_expand_q8_0_tensor");
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
    return rocm_launch_done("indexer scores launch", "indexer scores completion");
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
    if (rocm_trace()) fprintf(stderr, "ds4: ROCm enter ds4_metal_indexer_score_one_tensor\n");
    ROCM_TIME_SCOPE("ds4_metal_indexer_score_one_tensor");
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
    if (rocm_trace()) fprintf(stderr, "ds4: ROCm enter ds4_metal_indexer_scores_prefill_tensor\n");
    ROCM_TIME_SCOPE("ds4_metal_indexer_scores_prefill_tensor");
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
    if (rocm_trace()) fprintf(stderr, "ds4: ROCm enter ds4_metal_indexer_scores_decode_batch_tensor\n");
    ROCM_TIME_SCOPE("ds4_metal_indexer_scores_decode_batch_tensor");
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
    if (rocm_trace()) fprintf(stderr, "ds4: ROCm enter ds4_metal_indexer_topk_tensor\n");
    ROCM_TIME_SCOPE("ds4_metal_indexer_topk_tensor");
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
    return rocm_launch_done("indexer topk launch", "indexer topk completion");
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
    if (rocm_trace()) fprintf(stderr, "ds4: ROCm enter ds4_metal_dsv4_topk_mask_tensor\n");
    ROCM_TIME_SCOPE("ds4_metal_dsv4_topk_mask_tensor");
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
        if (!rocm_launch_done("topk mask init launch", "topk mask init completion")) return 0;
    }
    {
        const uint64_t total = (uint64_t)top_k * n_tokens;
        const dim3 block(256);
        const dim3 grid((unsigned)((total + block.x - 1u) / block.x));
        hipLaunchKernelGGL(rocm_topk_mask_scatter_kernel, grid, block, 0, 0,
                           (float *)tensor_u8(mask), (const int32_t *)tensor_u8_const(topk),
                           n_comp, n_tokens, top_k);
        return rocm_launch_done("topk mask scatter launch", "topk mask scatter completion");
    }
}

/* =========================================================================
 * Router selection.
 * ========================================================================= */

/* probs[i] := sqrt(softplus(logits[i])).  Replaces the CPU-side loop the
 * router_select wrapper used to do; that loop forced a stream-drain in
 * batched mode and dwarfed everything else when fired hundreds of times
 * per token across 43 layers. */
__global__ static void rocm_router_softplus_sqrt_kernel(float *probs,
                                                        const float *logits,
                                                        uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float l = logits[i];
    const float sp = l > 20.0f ? l : logf(1.0f + expf(l));
    probs[i] = sqrtf(sp > 0.0f ? sp : 0.0f);
}

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
    if (rocm_trace()) fprintf(stderr, "ds4: ROCm enter ds4_metal_router_select_tensor\n");
    ROCM_TIME_SCOPE("ds4_metal_router_select_tensor");
    if (!g_initialized && !ds4_metal_init()) return 0;
    if (!selected || !weights || !probs || !logits) return 0;
    /* Call sites pass (0, 0) to mean "use DS4 defaults" -- the Metal driver
     * also hardcodes these.  256 expert groups, 6 used per token. */
    if (n_expert_groups == 0) n_expert_groups = 256u;
    if (n_group_used == 0) n_group_used = 6u;
    if (selected->bytes < (uint64_t)n_group_used * sizeof(int32_t)) return 0;
    if (weights->bytes < (uint64_t)n_group_used * sizeof(float)) return 0;
    if (probs->bytes < (uint64_t)n_expert_groups * sizeof(float)) return 0;

    /* probs := sqrt(softplus(logits)) on the GPU.  Done as a launched
     * kernel rather than a CPU loop so we don't force a stream drain in
     * batched mode (router_select fires ~6 times per layer; a CPU drain
     * each time was 99% of inference wall in DS4_ROCM_TIME profiles). */
    {
        const uint32_t threads = 64u < n_expert_groups ? 64u : n_expert_groups;
        const uint32_t blocks = (n_expert_groups + threads - 1u) / threads;
        hipLaunchKernelGGL(rocm_router_softplus_sqrt_kernel,
                           dim3(blocks), dim3(threads), 0, 0,
                           (float *)tensor_u8(probs),
                           (const float *)tensor_u8_const(logits),
                           n_expert_groups);
        if (!rocm_launch_done("router softplus_sqrt launch",
                              "router softplus_sqrt completion")) return 0;
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
    int ok = rocm_launch_done("router select launch", "router select completion");
    rocm_defer_free(bh);
    rocm_defer_free(hh);
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
    if (rocm_trace()) fprintf(stderr, "ds4: ROCm enter ds4_metal_router_select_batch_tensor\n");
    ROCM_TIME_SCOPE("ds4_metal_router_select_batch_tensor");
    if (!selected || !weights || !probs || !logits || !tokens) return 0;
    if (n_expert_groups == 0) n_expert_groups = 256u;
    if (n_group_used == 0) n_group_used = 6u;
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
    if (rocm_trace()) fprintf(stderr, "ds4: ROCm enter ds4_metal_compressor_store_batch_tensor\n");
    ROCM_TIME_SCOPE("ds4_metal_compressor_store_batch_tensor");
    if (!g_initialized && !ds4_metal_init()) return 0;
    if (!kv || !sc || !state_kv || !state_score || !model_map ||
        head_dim == 0 || n_tokens == 0 || ratio == 0u ||
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
    int ok = rocm_launch_done("compressor store batch launch", "compressor store batch completion");
    rocm_defer_free(ape_h);
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
    return rocm_launch_done("compressor pool launch", "compressor pool completion");
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
    return rocm_launch_done("compressor shift launch", "compressor shift completion");
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
    if (rocm_trace()) fprintf(stderr, "ds4: ROCm enter ds4_metal_compressor_update_tensor\n");
    ROCM_TIME_SCOPE("ds4_metal_compressor_update_tensor");
    if (!g_initialized && !ds4_metal_init()) return 0;
    if (!kv_cur || !sc_cur || !state_kv || !state_score || !comp_cache || !model_map ||
        head_dim == 0 || ratio == 0u ||
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
    return rocm_launch_done("fill f32 launch", "fill f32 completion");
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
    if (rocm_trace()) fprintf(stderr, "ds4: ROCm enter ds4_metal_compressor_prefill_tensor head_dim=%u ratio=%u pos0=%u n_tokens=%u n_rot=%u ape_type=%u norm_type=%u\n",
                              head_dim, ratio, pos0, n_tokens, n_rot, ape_type, norm_type);
    ROCM_TIME_SCOPE("ds4_metal_compressor_prefill_tensor");
    if (!g_initialized && !ds4_metal_init()) return 0;
    if (!comp_cache || !state_kv || !state_score || !kv || !sc || !model_map ||
        head_dim == 0 || ratio == 0u || n_tokens == 0 ||
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
    if (rocm_trace()) fprintf(stderr, "ds4: ROCm enter ds4_metal_compressor_prefill_state_ratio4_tensor\n");
    ROCM_TIME_SCOPE("ds4_metal_compressor_prefill_state_ratio4_tensor");
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
    int ok = rocm_launch_done("compressor ratio4 state launch", "compressor ratio4 state completion");
    rocm_defer_free(ape_h);
    return ok;
}

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
    if (rocm_trace()) fprintf(stderr, "ds4: ROCm enter ds4_metal_compressor_prefill_ratio4_replay_tensor\n");
    ROCM_TIME_SCOPE("ds4_metal_compressor_prefill_ratio4_replay_tensor");
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

/* =========================================================================
 * Sink-aware attention kernels.
 * =========================================================================
 *
 * decode_heads computes one query token's per-head attention over a ring
 * buffer of raw KV plus a list of compressed KV rows.  Each head shares the
 * sink logit (read from the model map) and contributes one term to the
 * softmax denominator without injecting a value vector.
 */

__global__ static void rocm_attn_decode_heads_kernel(
        float *out,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const float *comp_mask,
        const float *sinks,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t use_mask) {
    extern __shared__ float smem[];
    const uint32_t h = blockIdx.x;
    if (h >= n_head) return;
    const uint32_t tid = threadIdx.x;
    const uint32_t bs = blockDim.x;
    const uint32_t n_kv = n_raw + n_comp;
    float *scores = smem;                /* n_kv */
    float *part   = smem + n_kv;         /* bs */

    const float *qh = q + (uint64_t)h * head_dim;
    const float kq_scale = rsqrtf((float)head_dim);

    /* Step 1: compute scores. */
    for (uint32_t r = tid; r < n_raw; r += bs) {
        const uint32_t actual = (raw_start + r) % raw_cap;
        const float *kv = raw_kv + (uint64_t)actual * head_dim;
        float s = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++) s += qh[d] * kv[d];
        scores[r] = s * kq_scale;
    }
    for (uint32_t r = tid; r < n_comp; r += bs) {
        const float *kv = comp_kv + (uint64_t)r * head_dim;
        float s = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++) s += qh[d] * kv[d];
        s = s * kq_scale + (use_mask ? comp_mask[r] : 0.0f);
        scores[n_raw + r] = s;
    }
    __syncthreads();

    /* Step 2: max reduction including sink. */
    float local_max = -1.0e30f;
    for (uint32_t r = tid; r < n_kv; r += bs) {
        if (scores[r] > local_max) local_max = scores[r];
    }
    if (tid == 0 && sinks[h] > local_max) local_max = sinks[h];
    part[tid] = local_max;
    __syncthreads();
    for (uint32_t s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) part[tid] = fmaxf(part[tid], part[tid + s]);
        __syncthreads();
    }
    const float maxs = part[0];

    /* Step 3: exp weights into scores in place. */
    for (uint32_t r = tid; r < n_kv; r += bs) {
        scores[r] = expf(scores[r] - maxs);
    }
    __syncthreads();

    /* Step 4: denom (sink + sum weights). */
    float local_sum = (tid == 0) ? expf(sinks[h] - maxs) : 0.0f;
    for (uint32_t r = tid; r < n_kv; r += bs) {
        local_sum += scores[r];
    }
    part[tid] = local_sum;
    __syncthreads();
    for (uint32_t s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) part[tid] += part[tid + s];
        __syncthreads();
    }
    const float denom = part[0];

    /* Step 5: weighted sum per dim. */
    float *out_h = out + (uint64_t)h * head_dim;
    for (uint32_t d = tid; d < head_dim; d += bs) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < n_raw; r++) {
            const uint32_t actual = (raw_start + r) % raw_cap;
            acc += scores[r] * raw_kv[(uint64_t)actual * head_dim + d];
        }
        for (uint32_t r = 0; r < n_comp; r++) {
            acc += scores[n_raw + r] * comp_kv[(uint64_t)r * head_dim + d];
        }
        out_h[d] = acc / denom;
    }
}

int ds4_metal_attention_decode_heads_tensor(
        ds4_metal_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_metal_tensor *q,
        const ds4_metal_tensor *raw_kv,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        const ds4_metal_tensor *comp_kv,
        uint32_t                n_comp,
        const ds4_metal_tensor *comp_mask,
        uint32_t                use_mask,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (rocm_trace()) fprintf(stderr, "ds4: ROCm enter ds4_metal_attention_decode_heads_tensor\n");
    ROCM_TIME_SCOPE("ds4_metal_attention_decode_heads_tensor");
    if (!g_initialized && !ds4_metal_init()) return 0;
    if (!heads || !model_map || !q || !raw_kv ||
        n_raw == 0 || n_head == 0 || head_dim == 0 ||
        raw_cap < n_raw || raw_start >= raw_cap ||
        n_raw > UINT32_MAX - n_comp || n_raw + n_comp > 8192u ||
        (n_comp != 0 && !comp_kv) || (use_mask != 0 && !comp_mask)) return 0;

    const uint64_t q_bytes = (uint64_t)n_head * head_dim * sizeof(float);
    const uint64_t raw_bytes = (uint64_t)raw_cap * head_dim * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)n_comp * head_dim * sizeof(float);
    const uint64_t sink_bytes = (uint64_t)n_head * sizeof(float);
    if (sinks_offset > model_size || sink_bytes > model_size - sinks_offset ||
        q->bytes < q_bytes || raw_kv->bytes < raw_bytes || heads->bytes < q_bytes ||
        (n_comp != 0 && comp_kv->bytes < comp_bytes) ||
        (use_mask && comp_mask->bytes < (uint64_t)n_comp * sizeof(float))) return 0;

    void *sink_h = NULL, *sink_d = NULL;
    if (!rocm_upload_mapped((const uint8_t *)model_map + sinks_offset, sink_bytes,
                            &sink_h, &sink_d, "attention sinks upload")) return 0;

    const uint32_t threads = 128;
    const uint32_t n_kv = n_raw + n_comp;
    const uint64_t smem_bytes = ((uint64_t)n_kv + threads) * sizeof(float);
    hipLaunchKernelGGL(rocm_attn_decode_heads_kernel, dim3(n_head), dim3(threads),
                       (uint32_t)smem_bytes, 0,
                       (float *)tensor_u8(heads),
                       (const float *)tensor_u8_const(q),
                       (const float *)tensor_u8_const(raw_kv),
                       n_comp ? (const float *)tensor_u8_const(comp_kv) : (const float *)NULL,
                       use_mask ? (const float *)tensor_u8_const(comp_mask) : (const float *)NULL,
                       (const float *)sink_d,
                       n_head, head_dim, n_raw, raw_cap, raw_start, n_comp, use_mask);
    int ok = rocm_launch_done("attn decode heads launch", "attn decode heads completion");
    rocm_defer_free(sink_h);
    return ok;
}

/* prefill_raw_heads: n_tokens parallel decode steps where token t attends
 * to a sliding window of the linearly-stored raw KV (no ring).  Each grid
 * tile is one (token, head) pair. */
__global__ static void rocm_attn_prefill_raw_heads_kernel(
        float *out,
        const float *q,
        const float *raw_kv,
        const float *sinks,
        uint32_t n_tokens,
        uint32_t window,
        uint32_t n_head,
        uint32_t head_dim) {
    extern __shared__ float smem[];
    const uint32_t t = blockIdx.x;
    const uint32_t h = blockIdx.y;
    if (t >= n_tokens || h >= n_head) return;
    const uint32_t tid = threadIdx.x;
    const uint32_t bs = blockDim.x;

    const uint32_t r0 = (t + 1u > window) ? t + 1u - window : 0u;
    const uint32_t n_kv = t + 1u - r0;
    float *scores = smem;
    float *part = smem + n_kv;

    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    const float kq_scale = rsqrtf((float)head_dim);

    for (uint32_t i = tid; i < n_kv; i += bs) {
        const float *kv = raw_kv + (uint64_t)(r0 + i) * head_dim;
        float s = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++) s += qh[d] * kv[d];
        scores[i] = s * kq_scale;
    }
    __syncthreads();

    float lmax = -1.0e30f;
    for (uint32_t i = tid; i < n_kv; i += bs) if (scores[i] > lmax) lmax = scores[i];
    if (tid == 0 && sinks[h] > lmax) lmax = sinks[h];
    part[tid] = lmax;
    __syncthreads();
    for (uint32_t s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) part[tid] = fmaxf(part[tid], part[tid + s]);
        __syncthreads();
    }
    const float maxs = part[0];

    for (uint32_t i = tid; i < n_kv; i += bs) scores[i] = expf(scores[i] - maxs);
    __syncthreads();

    float lsum = (tid == 0) ? expf(sinks[h] - maxs) : 0.0f;
    for (uint32_t i = tid; i < n_kv; i += bs) lsum += scores[i];
    part[tid] = lsum;
    __syncthreads();
    for (uint32_t s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) part[tid] += part[tid + s];
        __syncthreads();
    }
    const float denom = part[0];

    float *out_th = out + ((uint64_t)t * n_head + h) * head_dim;
    for (uint32_t d = tid; d < head_dim; d += bs) {
        float acc = 0.0f;
        for (uint32_t i = 0; i < n_kv; i++) acc += scores[i] * raw_kv[(uint64_t)(r0 + i) * head_dim + d];
        out_th[d] = acc / denom;
    }
}

int ds4_metal_attention_prefill_raw_heads_tensor(
        ds4_metal_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_metal_tensor *q,
        const ds4_metal_tensor *raw_kv,
        uint32_t                n_tokens,
        uint32_t                window,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (rocm_trace()) fprintf(stderr, "ds4: ROCm enter ds4_metal_attention_prefill_raw_heads_tensor\n");
    ROCM_TIME_SCOPE("ds4_metal_attention_prefill_raw_heads_tensor");
    if (!g_initialized && !ds4_metal_init()) return 0;
    if (!heads || !model_map || !q || !raw_kv ||
        n_tokens == 0 || window == 0 || n_head == 0 || head_dim == 0) return 0;

    const uint64_t q_bytes = (uint64_t)n_tokens * n_head * head_dim * sizeof(float);
    const uint64_t kv_bytes = (uint64_t)n_tokens * head_dim * sizeof(float);
    const uint64_t sink_bytes = (uint64_t)n_head * sizeof(float);
    if (sinks_offset > model_size || sink_bytes > model_size - sinks_offset ||
        q->bytes < q_bytes || raw_kv->bytes < kv_bytes || heads->bytes < q_bytes) return 0;

    void *sink_h = NULL, *sink_d = NULL;
    if (!rocm_upload_mapped((const uint8_t *)model_map + sinks_offset, sink_bytes,
                            &sink_h, &sink_d, "prefill raw sinks upload")) return 0;

    const uint32_t threads = 128;
    const uint64_t smem_bytes = ((uint64_t)window + threads) * sizeof(float);
    dim3 grid(n_tokens, n_head, 1);
    dim3 block(threads, 1, 1);
    hipLaunchKernelGGL(rocm_attn_prefill_raw_heads_kernel, grid, block,
                       (uint32_t)smem_bytes, 0,
                       (float *)tensor_u8(heads),
                       (const float *)tensor_u8_const(q),
                       (const float *)tensor_u8_const(raw_kv),
                       (const float *)sink_d,
                       n_tokens, window, n_head, head_dim);
    int ok = rocm_launch_done("attn prefill raw launch", "attn prefill raw completion");
    rocm_defer_free(sink_h);
    return ok;
}

/* Batch attention kernel.  Handles all five remaining attention variants by
 * parameter selection:
 *   - mode_static: raw_kv is laid out (n_tokens × head_dim) with key k at
 *     position k.  Otherwise raw_kv is a ring buffer (raw_cap × head_dim)
 *     whose n_raw valid rows occupy logical positions
 *     [pos0 + n_tokens - n_raw, pos0 + n_tokens) and start at index raw_start.
 *   - use_topk: only the per-token indices in topk[t, 0..top_k) participate
 *     from comp_kv; all other comp rows mask out for that query.
 *   - use_comp_mask: comp_mask is added to each comp score (used to mask
 *     dropped indexer top-k slots in non-indexed prefill).
 */
__global__ static void rocm_attn_batch_kernel(
        float *out,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const float *comp_mask,
        const int32_t *topk,
        const float *sinks,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t top_k,
        int use_topk,
        int use_comp_mask,
        int mode_static) {
    extern __shared__ float smem[];
    const uint32_t t = blockIdx.x;
    const uint32_t h = blockIdx.y;
    if (t >= n_tokens || h >= n_head) return;
    const uint32_t tid = threadIdx.x;
    const uint32_t bs = blockDim.x;
    const uint32_t n_keys = n_raw + n_comp;
    float *scores = smem;
    float *part = smem + n_keys;

    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    const float kq_scale = rsqrtf((float)head_dim);
    const uint32_t qpos = pos0 + t;
    const uint32_t first_raw_pos = mode_static ? 0u : pos0 + n_tokens - n_raw;
    const uint32_t n_visible_comp = ratio == 0u ? 0u : (qpos + 1u) / ratio;

    /* Raw scores. */
    for (uint32_t r = tid; r < n_raw; r += bs) {
        const uint32_t kpos = first_raw_pos + r;
        if (kpos > qpos || (window != 0u && qpos - kpos >= window)) {
            scores[r] = -1.0e30f;
            continue;
        }
        const uint32_t actual = mode_static ? r : (raw_start + r) % raw_cap;
        const float *kv = raw_kv + (uint64_t)actual * head_dim;
        float s = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++) s += qh[d] * kv[d];
        scores[r] = s * kq_scale;
    }

    /* Comp scores. */
    if (use_topk) {
        for (uint32_t c = tid; c < n_comp; c += bs) scores[n_raw + c] = -1.0e30f;
        __syncthreads();
        for (uint32_t k = tid; k < top_k; k += bs) {
            const int32_t c = topk[(uint64_t)t * top_k + k];
            if (c < 0 || (uint32_t)c >= n_comp) continue;
            if ((uint32_t)c >= n_visible_comp) continue;
            const float *kv = comp_kv + (uint64_t)c * head_dim;
            float s = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) s += qh[d] * kv[d];
            scores[n_raw + c] = s * kq_scale;
        }
    } else {
        for (uint32_t c = tid; c < n_comp; c += bs) {
            if (c >= n_visible_comp) {
                scores[n_raw + c] = -1.0e30f;
                continue;
            }
            const float *kv = comp_kv + (uint64_t)c * head_dim;
            float s = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) s += qh[d] * kv[d];
            scores[n_raw + c] = s * kq_scale + (use_comp_mask ? comp_mask[c] : 0.0f);
        }
    }
    __syncthreads();

    /* Max with sink. */
    float lmax = -1.0e30f;
    for (uint32_t r = tid; r < n_keys; r += bs) {
        if (scores[r] > lmax) lmax = scores[r];
    }
    if (tid == 0 && sinks[h] > lmax) lmax = sinks[h];
    part[tid] = lmax;
    __syncthreads();
    for (uint32_t s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) part[tid] = fmaxf(part[tid], part[tid + s]);
        __syncthreads();
    }
    const float maxs = part[0];

    /* Weights into scores in place; mask-out turns into 0. */
    for (uint32_t r = tid; r < n_keys; r += bs) {
        scores[r] = (scores[r] <= -5.0e29f) ? 0.0f : expf(scores[r] - maxs);
    }
    __syncthreads();

    /* Denom: sink + sum(weights). */
    float lsum = (tid == 0) ? expf(sinks[h] - maxs) : 0.0f;
    for (uint32_t r = tid; r < n_keys; r += bs) lsum += scores[r];
    part[tid] = lsum;
    __syncthreads();
    for (uint32_t s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) part[tid] += part[tid + s];
        __syncthreads();
    }
    const float denom = part[0];

    /* Weighted sum per output dim. */
    float *out_th = out + ((uint64_t)t * n_head + h) * head_dim;
    for (uint32_t d = tid; d < head_dim; d += bs) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < n_raw; r++) {
            const uint32_t actual = mode_static ? r : (raw_start + r) % raw_cap;
            acc += scores[r] * raw_kv[(uint64_t)actual * head_dim + d];
        }
        for (uint32_t c = 0; c < n_comp; c++) {
            acc += scores[n_raw + c] * comp_kv[(uint64_t)c * head_dim + d];
        }
        out_th[d] = acc / denom;
    }
}

static int rocm_attn_batch_dispatch(
        ds4_metal_tensor *heads,
        const void *model_map,
        uint64_t model_size,
        uint64_t sinks_offset,
        const ds4_metal_tensor *q,
        const ds4_metal_tensor *raw_kv,
        const ds4_metal_tensor *comp_kv,
        const ds4_metal_tensor *comp_mask,
        const ds4_metal_tensor *topk,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t top_k,
        int use_topk,
        int use_comp_mask,
        int mode_static,
        const char *what) {
    const uint64_t sink_bytes = (uint64_t)n_head * sizeof(float);
    const uint64_t q_bytes = (uint64_t)n_tokens * n_head * head_dim * sizeof(float);
    const uint64_t raw_bytes = (uint64_t)(mode_static ? n_tokens : raw_cap) * head_dim * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)n_comp * head_dim * sizeof(float);
    if (sinks_offset > model_size || sink_bytes > model_size - sinks_offset ||
        q->bytes < q_bytes || raw_kv->bytes < raw_bytes || heads->bytes < q_bytes ||
        (n_comp != 0 && comp_kv->bytes < comp_bytes) ||
        (use_topk && topk->bytes < (uint64_t)n_tokens * top_k * sizeof(int32_t)) ||
        (use_comp_mask && comp_mask->bytes < (uint64_t)n_comp * sizeof(float))) return 0;

    void *sink_h = NULL, *sink_d = NULL;
    if (!rocm_upload_mapped((const uint8_t *)model_map + sinks_offset, sink_bytes,
                            &sink_h, &sink_d, "attention sinks upload")) return 0;

    const uint32_t threads = 128;
    const uint32_t n_keys = n_raw + n_comp;
    const uint64_t smem_bytes = ((uint64_t)n_keys + threads) * sizeof(float);
    dim3 grid(n_tokens, n_head, 1);
    dim3 block(threads, 1, 1);
    hipLaunchKernelGGL(rocm_attn_batch_kernel, grid, block, (uint32_t)smem_bytes, 0,
                       (float *)tensor_u8(heads),
                       (const float *)tensor_u8_const(q),
                       (const float *)tensor_u8_const(raw_kv),
                       n_comp ? (const float *)tensor_u8_const(comp_kv) : (const float *)NULL,
                       use_comp_mask ? (const float *)tensor_u8_const(comp_mask) : (const float *)NULL,
                       use_topk ? (const int32_t *)tensor_u8_const(topk) : (const int32_t *)NULL,
                       (const float *)sink_d,
                       n_tokens, n_head, head_dim, pos0, n_raw, raw_cap, raw_start,
                       n_comp, window, ratio, top_k, use_topk, use_comp_mask, mode_static);
    char launch_msg[64];
    snprintf(launch_msg, sizeof(launch_msg), "attn batch %s launch", what);
    char done_msg[64];
    snprintf(done_msg, sizeof(done_msg), "attn batch %s completion", what);
    int ok = rocm_launch_done(launch_msg, done_msg);
    rocm_defer_free(sink_h);
    return ok;
}

int ds4_metal_attention_decode_raw_batch_heads_tensor(
        ds4_metal_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_metal_tensor *q,
        const ds4_metal_tensor *raw_kv,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                window,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (rocm_trace()) fprintf(stderr, "ds4: ROCm enter ds4_metal_attention_decode_raw_batch_heads_tensor\n");
    ROCM_TIME_SCOPE("ds4_metal_attention_decode_raw_batch_heads_tensor");
    if (!g_initialized && !ds4_metal_init()) return 0;
    if (!heads || !model_map || !q || !raw_kv ||
        n_tokens == 0 || n_raw == 0 || n_head == 0 || head_dim == 0 ||
        raw_cap < n_raw || raw_start >= raw_cap) return 0;
    return rocm_attn_batch_dispatch(heads, model_map, model_size, sinks_offset,
                                    q, raw_kv, NULL, NULL, NULL,
                                    n_tokens, n_head, head_dim, pos0,
                                    n_raw, raw_cap, raw_start,
                                    0u, window, 1u, 0u, 0, 0, 0,
                                    "decode_raw");
}

int ds4_metal_attention_decode_mixed_batch_heads_tensor(
        ds4_metal_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_metal_tensor *q,
        const ds4_metal_tensor *raw_kv,
        const ds4_metal_tensor *comp_kv,
        const ds4_metal_tensor *comp_mask,
        uint32_t                use_comp_mask,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (rocm_trace()) fprintf(stderr, "ds4: ROCm enter ds4_metal_attention_decode_mixed_batch_heads_tensor\n");
    ROCM_TIME_SCOPE("ds4_metal_attention_decode_mixed_batch_heads_tensor");
    if (!g_initialized && !ds4_metal_init()) return 0;
    if (!heads || !model_map || !q || !raw_kv ||
        n_tokens == 0 || n_raw == 0 || n_head == 0 || head_dim == 0 ||
        raw_cap < n_raw || raw_start >= raw_cap || ratio == 0u ||
        (n_comp != 0 && !comp_kv) || (use_comp_mask != 0 && !comp_mask)) return 0;
    return rocm_attn_batch_dispatch(heads, model_map, model_size, sinks_offset,
                                    q, raw_kv, comp_kv, comp_mask, NULL,
                                    n_tokens, n_head, head_dim, pos0,
                                    n_raw, raw_cap, raw_start,
                                    n_comp, window, ratio, 0u, 0,
                                    use_comp_mask ? 1 : 0, 0,
                                    "decode_mixed");
}

int ds4_metal_attention_indexed_mixed_batch_heads_tensor(
        ds4_metal_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_metal_tensor *q,
        const ds4_metal_tensor *raw_kv,
        const ds4_metal_tensor *comp_kv,
        const ds4_metal_tensor *topk,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                top_k,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (rocm_trace()) fprintf(stderr, "ds4: ROCm enter ds4_metal_attention_indexed_mixed_batch_heads_tensor\n");
    ROCM_TIME_SCOPE("ds4_metal_attention_indexed_mixed_batch_heads_tensor");
    if (!g_initialized && !ds4_metal_init()) return 0;
    if (!heads || !model_map || !q || !raw_kv || !comp_kv || !topk ||
        n_tokens == 0 || n_raw == 0 || n_head == 0 || head_dim == 0 ||
        raw_cap < n_raw || raw_start >= raw_cap || ratio == 0u ||
        n_comp == 0 || top_k == 0) return 0;
    return rocm_attn_batch_dispatch(heads, model_map, model_size, sinks_offset,
                                    q, raw_kv, comp_kv, NULL, topk,
                                    n_tokens, n_head, head_dim, pos0,
                                    n_raw, raw_cap, raw_start,
                                    n_comp, window, ratio, top_k, 1, 0, 0,
                                    "indexed_mixed");
}

int ds4_metal_attention_prefill_static_mixed_heads_tensor(
        ds4_metal_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_metal_tensor *q,
        const ds4_metal_tensor *raw_kv,
        const ds4_metal_tensor *comp_kv,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (rocm_trace()) fprintf(stderr, "ds4: ROCm enter ds4_metal_attention_prefill_static_mixed_heads_tensor\n");
    ROCM_TIME_SCOPE("ds4_metal_attention_prefill_static_mixed_heads_tensor");
    if (!g_initialized && !ds4_metal_init()) return 0;
    if (!heads || !model_map || !q || !raw_kv ||
        n_tokens == 0 || n_head == 0 || head_dim == 0 || ratio == 0u ||
        (n_comp != 0 && !comp_kv)) return 0;
    /* mode_static: raw_kv has n_tokens rows at logical positions [0, n_tokens). */
    return rocm_attn_batch_dispatch(heads, model_map, model_size, sinks_offset,
                                    q, raw_kv, comp_kv, NULL, NULL,
                                    n_tokens, n_head, head_dim,
                                    /*pos0=*/0u, /*n_raw=*/n_tokens, /*raw_cap=*/n_tokens,
                                    /*raw_start=*/0u, n_comp, window, ratio,
                                    0u, 0, 0, 1,
                                    "prefill_static");
}

int ds4_metal_attention_prefill_masked_mixed_heads_tensor(
        ds4_metal_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_metal_tensor *q,
        const ds4_metal_tensor *raw_kv,
        const ds4_metal_tensor *comp_kv,
        const ds4_metal_tensor *comp_mask,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (rocm_trace()) fprintf(stderr, "ds4: ROCm enter ds4_metal_attention_prefill_masked_mixed_heads_tensor\n");
    ROCM_TIME_SCOPE("ds4_metal_attention_prefill_masked_mixed_heads_tensor");
    if (!g_initialized && !ds4_metal_init()) return 0;
    if (!heads || !model_map || !q || !raw_kv || !comp_mask ||
        n_tokens == 0 || n_head == 0 || head_dim == 0 || ratio == 0u ||
        (n_comp != 0 && !comp_kv)) return 0;
    return rocm_attn_batch_dispatch(heads, model_map, model_size, sinks_offset,
                                    q, raw_kv, comp_kv, comp_mask, NULL,
                                    n_tokens, n_head, head_dim,
                                    /*pos0=*/0u, /*n_raw=*/n_tokens, /*raw_cap=*/n_tokens,
                                    /*raw_start=*/0u, n_comp, window, ratio,
                                    0u, 0, 1, 1,
                                    "prefill_masked");
}

/* =========================================================================
 * Routed MoE kernels (IQ2_XXS / Q2_K / Q4_K weights, FP32 activations).
 * =========================================================================
 *
 * Mirrors layer_routed_moe_one() / layer_routed_moe_batch() in ds4.c.  Two
 * stages per token:
 *   1. gate_up: for each (slot, mid_row), dot the per-slot expert's gate and
 *      up rows against the activation, clamp, then write
 *          mid[slot, mid_row] = silu(gate_clamped) * up_clamped * weight[slot]
 *   2. down: for each out_row, accumulate over slots:
 *          out[out_row] = sum_s dot(down_expert[selected[s], out_row], mid[s])
 *
 * For bring-up the host wrappers pre-pack just the n_used selected experts per
 * token (CPU-side gather) and upload that compact slab via the existing
 * mapped-host-memory path.  This keeps the per-call upload bounded by
 * n_used*per-expert-bytes instead of the full 256-expert pool.  Activations
 * stay in FP32 — we skip the Q8_K staging the CPU path uses, which is more
 * accurate and removes a kernel.
 */

#define DS4_QK_K 256

typedef struct {
    uint8_t  scales[DS4_QK_K / 16];
    uint8_t  qs[DS4_QK_K / 4];
    uint16_t d;
    uint16_t dmin;
} rocm_block_q2_K;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qs[DS4_QK_K / 2];
} rocm_block_q4_K;

typedef struct {
    uint16_t d;
    uint16_t qs[DS4_QK_K / 8];
} rocm_block_iq2_xxs;

#define DS4_TENSOR_TYPE_Q2_K     10u
#define DS4_TENSOR_TYPE_Q4_K     12u
#define DS4_TENSOR_TYPE_IQ2_XXS  16u

__device__ static const uint64_t rocm_iq2xxs_grid[256] = {
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

__device__ static const uint8_t rocm_ksigns_iq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

/* Per-block dot helpers.  Each computes sum_i (dequant_w_i * y_i) for one
 * 256-element block, accumulating in float against an FP32 activation slice. */

__device__ __forceinline__ static float rocm_iq2xxs_dot_block(
        const rocm_block_iq2_xxs *xb, const float *y) {
    const float d = rocm_f16_to_f32(xb->d);
    float sum = 0.0f;
    const uint16_t *q = xb->qs;
    for (int ib32 = 0; ib32 < 8; ib32++) {
        const uint32_t aux32_g = (uint32_t)q[0] | ((uint32_t)q[1] << 16);
        const uint32_t aux32_s = (uint32_t)q[2] | ((uint32_t)q[3] << 16);
        q += 4;
        const float dl = d * (0.5f + (float)(aux32_s >> 28)) * 0.25f;
        const uint8_t a0 = (uint8_t)(aux32_g >>  0);
        const uint8_t a1 = (uint8_t)(aux32_g >>  8);
        const uint8_t a2 = (uint8_t)(aux32_g >> 16);
        const uint8_t a3 = (uint8_t)(aux32_g >> 24);
        const uint8_t s0 = rocm_ksigns_iq2xs[(aux32_s >>  0) & 127u];
        const uint8_t s1 = rocm_ksigns_iq2xs[(aux32_s >>  7) & 127u];
        const uint8_t s2 = rocm_ksigns_iq2xs[(aux32_s >> 14) & 127u];
        const uint8_t s3 = rocm_ksigns_iq2xs[(aux32_s >> 21) & 127u];
        const uint8_t *g0 = (const uint8_t *)(rocm_iq2xxs_grid + a0);
        const uint8_t *g1 = (const uint8_t *)(rocm_iq2xxs_grid + a1);
        const uint8_t *g2 = (const uint8_t *)(rocm_iq2xxs_grid + a2);
        const uint8_t *g3 = (const uint8_t *)(rocm_iq2xxs_grid + a3);
        float sub = 0.0f;
        for (int j = 0; j < 8; j++) {
            const float v0 = (float)g0[j] * (((s0 >> j) & 1u) ? -1.0f : 1.0f);
            const float v1 = (float)g1[j] * (((s1 >> j) & 1u) ? -1.0f : 1.0f);
            const float v2 = (float)g2[j] * (((s2 >> j) & 1u) ? -1.0f : 1.0f);
            const float v3 = (float)g3[j] * (((s3 >> j) & 1u) ? -1.0f : 1.0f);
            sub += y[j +  0] * v0 + y[j +  8] * v1 + y[j + 16] * v2 + y[j + 24] * v3;
        }
        sum += dl * sub;
        y += 32;
    }
    return sum;
}

__device__ __forceinline__ static float rocm_q2_K_dot_block(
        const rocm_block_q2_K *xb, const float *y) {
    const float d = rocm_f16_to_f32(xb->d);
    const float dmin = rocm_f16_to_f32(xb->dmin);
    const uint8_t *q = xb->qs;
    const uint8_t *sc = xb->scales;
    float sum_d = 0.0f, sum_m = 0.0f;
    int is = 0;
    for (int k = 0; k < 2; k++) {
        int shift = 0;
        for (int j = 0; j < 4; j++) {
            const float scale_a = (float)(sc[is] & 0x0F);
            const float min_a   = (float)(sc[is] >> 4);
            float dot_a = 0.0f, sumy_a = 0.0f;
            for (int i = 0; i < 16; i++) {
                const float yv = y[i];
                dot_a += yv * (float)((q[i] >> shift) & 0x3);
                sumy_a += yv;
            }
            sum_d += scale_a * dot_a;
            sum_m += min_a * sumy_a;
            is++;

            const float scale_b = (float)(sc[is] & 0x0F);
            const float min_b   = (float)(sc[is] >> 4);
            float dot_b = 0.0f, sumy_b = 0.0f;
            for (int i = 0; i < 16; i++) {
                const float yv = y[i + 16];
                dot_b += yv * (float)((q[i + 16] >> shift) & 0x3);
                sumy_b += yv;
            }
            sum_d += scale_b * dot_b;
            sum_m += min_b * sumy_b;
            is++;

            shift += 2;
            y += 32;
        }
        q += 32;
    }
    return d * sum_d - dmin * sum_m;
}

__device__ __forceinline__ static void rocm_q4_K_get_scale_min(
        int j, const uint8_t *q, uint8_t *scale, uint8_t *minv) {
    if (j < 4) {
        *scale = q[j] & 0x3F;
        *minv  = q[j + 4] & 0x3F;
    } else {
        *scale = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        *minv  = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
}

__device__ __forceinline__ static float rocm_q4_K_dot_block(
        const rocm_block_q4_K *xb, const float *y) {
    const float d = rocm_f16_to_f32(xb->d);
    const float dmin = rocm_f16_to_f32(xb->dmin);
    const uint8_t *q = xb->qs;
    float sum_d = 0.0f, sum_m = 0.0f;
    for (int j = 0; j < 8; j++) {
        uint8_t scale, mn;
        rocm_q4_K_get_scale_min(j, xb->scales, &scale, &mn);
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

__device__ __forceinline__ static float rocm_quant_dot_block(
        uint32_t type, const uint8_t *block, const float *y) {
    if (type == DS4_TENSOR_TYPE_IQ2_XXS) {
        return rocm_iq2xxs_dot_block((const rocm_block_iq2_xxs *)block, y);
    }
    if (type == DS4_TENSOR_TYPE_Q2_K) {
        return rocm_q2_K_dot_block((const rocm_block_q2_K *)block, y);
    }
    return rocm_q4_K_dot_block((const rocm_block_q4_K *)block, y);
}

__device__ __forceinline__ static uint32_t rocm_quant_block_bytes(uint32_t type) {
    if (type == DS4_TENSOR_TYPE_IQ2_XXS) return (uint32_t)sizeof(rocm_block_iq2_xxs);
    if (type == DS4_TENSOR_TYPE_Q2_K)    return (uint32_t)sizeof(rocm_block_q2_K);
    return (uint32_t)sizeof(rocm_block_q4_K);
}

static uint64_t rocm_quant_block_bytes_host(uint32_t type) {
    if (type == DS4_TENSOR_TYPE_IQ2_XXS) return sizeof(rocm_block_iq2_xxs);
    if (type == DS4_TENSOR_TYPE_Q2_K)    return sizeof(rocm_block_q2_K);
    if (type == DS4_TENSOR_TYPE_Q4_K)    return sizeof(rocm_block_q4_K);
    return 0;
}

__device__ __forceinline__ static float rocm_silu(float x) {
    return x / (1.0f + expf(-x));
}

/* The kernel reads experts directly from the full model-map pool, indexed by
 * selected[t*n_used + s].  When the model map is registered (production path)
 * the pool pointer comes for free; for tests we upload the whole pool once
 * via rocm_upload_mapped.
 *
 * Activations are LDS-staged one 256-element block at a time.  Every thread
 * in a workgroup shares the same (slot, token) and so reads the same x
 * slice; with LDS staging the per-block load is cooperative (threads in the
 * block split the 256 elements) instead of each thread re-reading from
 * global. */

__global__ static void rocm_moe_gate_up_matvec_kernel(
        float *mid,                    /* [n_tokens * n_used * mid_dim]   */
        const uint8_t *gate_pool,      /* full [n_total_experts * gate_expert_bytes] */
        const uint8_t *up_pool,        /* full [n_total_experts * up_expert_bytes]   */
        uint32_t gate_type,
        uint32_t up_type,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t up_expert_bytes,
        uint64_t up_row_bytes,
        const float *x,                /* [n_tokens * in_dim]             */
        const int32_t *selected,       /* [n_tokens * n_used]             */
        const float *weights,          /* [n_tokens * n_used]             */
        uint32_t in_dim,
        uint32_t mid_dim,
        uint32_t n_used,
        uint32_t n_tokens,
        uint32_t n_total_experts,
        float clamp) {
    __shared__ float x_lds[DS4_QK_K];

    const uint32_t slot   = blockIdx.x;
    const uint32_t row    = blockIdx.y * blockDim.x + threadIdx.x;
    const uint32_t token  = blockIdx.z;
    if (slot >= n_used || token >= n_tokens) return;

    const int32_t e = selected[(uint64_t)token * n_used + slot];
    const bool valid_expert = (e >= 0 && (uint32_t)e < n_total_experts);
    const bool active_row = (row < mid_dim);

    if (!valid_expert) {
        if (active_row) mid[((uint64_t)token * n_used + slot) * mid_dim + row] = 0.0f;
        return;
    }

    const uint32_t nb = in_dim / DS4_QK_K;
    const uint32_t gate_bb = rocm_quant_block_bytes(gate_type);
    const uint32_t up_bb   = rocm_quant_block_bytes(up_type);
    const uint8_t *gate_row_p = active_row
        ? gate_pool + (uint64_t)e * gate_expert_bytes + (uint64_t)row * gate_row_bytes
        : NULL;
    const uint8_t *up_row_p = active_row
        ? up_pool + (uint64_t)e * up_expert_bytes + (uint64_t)row * up_row_bytes
        : NULL;
    const float *x_token_base = x + (uint64_t)token * in_dim;

    float gate_v = 0.0f, up_v = 0.0f;
    for (uint32_t b = 0; b < nb; b++) {
        const float *x_src = x_token_base + (uint64_t)b * DS4_QK_K;
        for (uint32_t i = threadIdx.x; i < DS4_QK_K; i += blockDim.x) {
            x_lds[i] = x_src[i];
        }
        __syncthreads();
        if (active_row) {
            gate_v += rocm_quant_dot_block(gate_type, gate_row_p + (uint64_t)b * gate_bb, x_lds);
            up_v   += rocm_quant_dot_block(up_type,   up_row_p   + (uint64_t)b * up_bb,   x_lds);
        }
        __syncthreads();
    }

    if (!active_row) return;
    if (clamp > 1.0e-6f) {
        if (gate_v > clamp) gate_v = clamp;
        if (up_v > clamp) up_v = clamp;
        if (up_v < -clamp) up_v = -clamp;
    }
    const float w = weights[(uint64_t)token * n_used + slot];
    mid[((uint64_t)token * n_used + slot) * mid_dim + row] = rocm_silu(gate_v) * up_v * w;
}

__global__ static void rocm_moe_down_matvec_kernel(
        float *out,                    /* [n_tokens * out_dim]            */
        const uint8_t *down_pool,      /* full [n_total_experts * down_expert_bytes] */
        uint32_t down_type,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        const float *mid,              /* [n_tokens * n_used * mid_dim]   */
        const int32_t *selected,       /* [n_tokens * n_used]             */
        uint32_t mid_dim,
        uint32_t out_dim,
        uint32_t n_used,
        uint32_t n_tokens,
        uint32_t n_total_experts) {
    __shared__ float mid_lds[DS4_QK_K];

    const uint32_t row   = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t token = blockIdx.y;
    if (token >= n_tokens) return;
    const bool active_row = (row < out_dim);

    const uint32_t nb = mid_dim / DS4_QK_K;
    const uint32_t down_bb = rocm_quant_block_bytes(down_type);

    float acc = 0.0f;
    for (uint32_t s = 0; s < n_used; s++) {
        const int32_t e = selected[(uint64_t)token * n_used + s];
        const bool valid = (e >= 0 && (uint32_t)e < n_total_experts);
        const uint8_t *down_row_p = (valid && active_row)
            ? down_pool + (uint64_t)e * down_expert_bytes + (uint64_t)row * down_row_bytes
            : NULL;
        const float *mid_slot = mid + ((uint64_t)token * n_used + s) * mid_dim;
        for (uint32_t b = 0; b < nb; b++) {
            const float *mid_src = mid_slot + (uint64_t)b * DS4_QK_K;
            for (uint32_t i = threadIdx.x; i < DS4_QK_K; i += blockDim.x) {
                mid_lds[i] = mid_src[i];
            }
            __syncthreads();
            if (valid && active_row) {
                acc += rocm_quant_dot_block(down_type, down_row_p + (uint64_t)b * down_bb, mid_lds);
            }
            __syncthreads();
        }
    }

    if (active_row) out[(uint64_t)token * out_dim + row] = acc;
}

static int rocm_moe_run(
        ds4_metal_tensor       *out,
        ds4_metal_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_metal_tensor *selected,
        const ds4_metal_tensor *weights,
        uint32_t                n_used,
        uint32_t                n_tokens,
        float                   clamp,
        const ds4_metal_tensor *x) {
    if (!out || !mid || !x || !selected || !weights || !model_map) return 0;
    if (n_used == 0 || n_tokens == 0 || expert_in_dim == 0 || expert_mid_dim == 0 || out_dim == 0) return 0;
    if ((expert_in_dim % DS4_QK_K) != 0 || (expert_mid_dim % DS4_QK_K) != 0) return 0;
    const uint64_t gate_bb = rocm_quant_block_bytes_host(gate_type);
    const uint64_t down_bb = rocm_quant_block_bytes_host(down_type);
    if (gate_bb == 0 || down_bb == 0) {
        fprintf(stderr, "ds4: ROCm routed MoE unsupported quant types gate=%u down=%u\n",
                gate_type, down_type);
        return 0;
    }
    /* Up shares gate's row layout and quant type. */
    const uint32_t up_type = gate_type;
    const uint64_t up_expert_bytes = gate_expert_bytes;
    const uint64_t up_row_bytes = gate_row_bytes;

    /* Pool size: total experts in the layer (all routed-MoE layers use 256). */
    const uint64_t total_experts = 256ull;
    const uint64_t gate_pool = total_experts * gate_expert_bytes;
    const uint64_t up_pool   = total_experts * up_expert_bytes;
    const uint64_t down_pool = total_experts * down_expert_bytes;
    if (gate_offset > model_size || gate_pool > model_size - gate_offset) return 0;
    if (up_offset   > model_size || up_pool   > model_size - up_offset)   return 0;
    if (down_offset > model_size || down_pool > model_size - down_offset) return 0;

    if (mid->bytes < (uint64_t)n_tokens * n_used * expert_mid_dim * sizeof(float)) return 0;
    if (out->bytes < (uint64_t)n_tokens * out_dim * sizeof(float)) return 0;
    if (x->bytes < (uint64_t)n_tokens * expert_in_dim * sizeof(float)) return 0;
    if (selected->bytes < (uint64_t)n_tokens * n_used * sizeof(int32_t)) return 0;
    if (weights->bytes < (uint64_t)n_tokens * n_used * sizeof(float)) return 0;

    /* Resolve each pool to a device pointer.  Hits the registered model-map
     * fast path on production runs; falls back to a one-shot pinned upload
     * (e.g. in tests) per pool.  No per-call packing or per-token copies. */
    void *gate_h = NULL, *up_h = NULL, *down_h = NULL;
    void *gate_d = NULL, *up_d = NULL, *down_d = NULL;
    int ok = rocm_upload_mapped((const uint8_t *)model_map + gate_offset, gate_pool,
                                &gate_h, &gate_d, "moe gate pool")
          && rocm_upload_mapped((const uint8_t *)model_map + up_offset, up_pool,
                                &up_h, &up_d, "moe up pool")
          && rocm_upload_mapped((const uint8_t *)model_map + down_offset, down_pool,
                                &down_h, &down_d, "moe down pool");
    if (!ok) {
        rocm_defer_free(gate_h);
        rocm_defer_free(up_h);
        rocm_defer_free(down_h);
        return 0;
    }

    {
        const uint32_t threads = 64;
        const uint32_t row_blocks = (expert_mid_dim + threads - 1) / threads;
        dim3 grid(n_used, row_blocks, n_tokens);
        dim3 block(threads, 1, 1);
        hipLaunchKernelGGL(rocm_moe_gate_up_matvec_kernel, grid, block, 0, 0,
                           (float *)tensor_u8(mid),
                           (const uint8_t *)gate_d,
                           (const uint8_t *)up_d,
                           gate_type, up_type,
                           gate_expert_bytes, gate_row_bytes,
                           up_expert_bytes, up_row_bytes,
                           (const float *)tensor_u8_const(x),
                           (const int32_t *)tensor_u8_const(selected),
                           (const float *)tensor_u8_const(weights),
                           expert_in_dim, expert_mid_dim, n_used, n_tokens,
                           (uint32_t)total_experts, clamp);
    }
    ok = rocm_launch_done("moe gate_up launch", "moe gate_up completion");

    if (ok) {
        const uint32_t threads = 64;
        const uint32_t row_blocks = (out_dim + threads - 1) / threads;
        dim3 grid(row_blocks, n_tokens, 1);
        dim3 block(threads, 1, 1);
        hipLaunchKernelGGL(rocm_moe_down_matvec_kernel, grid, block, 0, 0,
                           (float *)tensor_u8(out),
                           (const uint8_t *)down_d,
                           down_type,
                           down_expert_bytes, down_row_bytes,
                           (const float *)tensor_u8_const(mid),
                           (const int32_t *)tensor_u8_const(selected),
                           expert_mid_dim, out_dim, n_used, n_tokens,
                           (uint32_t)total_experts);
        ok = rocm_launch_done("moe down launch", "moe down completion");
    }

    rocm_defer_free(gate_h);
    if (up_h)   (void)hipHostFree(up_h);
    rocm_defer_free(down_h);
    return ok;
}

int ds4_metal_routed_moe_one_tensor(
        ds4_metal_tensor       *out,
        ds4_metal_tensor       *gate,
        ds4_metal_tensor       *up,
        ds4_metal_tensor       *mid,
        ds4_metal_tensor       *experts,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_metal_tensor *selected,
        const ds4_metal_tensor *weights,
        uint32_t                n_expert,
        float                   clamp,
        const ds4_metal_tensor *x) {
    if (rocm_trace()) fprintf(stderr, "ds4: ROCm enter ds4_metal_routed_moe_one_tensor\n");
    ROCM_TIME_SCOPE("ds4_metal_routed_moe_one_tensor");
    (void)gate; (void)up; (void)experts;
    return rocm_moe_run(out, mid, model_map, model_size,
                        gate_offset, up_offset, down_offset,
                        gate_type, down_type,
                        gate_expert_bytes, gate_row_bytes,
                        down_expert_bytes, down_row_bytes,
                        expert_in_dim, expert_mid_dim, out_dim,
                        selected, weights, n_expert, 1u, clamp, x);
}

int ds4_metal_routed_moe_batch_tensor(
        ds4_metal_tensor       *out,
        ds4_metal_tensor       *gate,
        ds4_metal_tensor       *up,
        ds4_metal_tensor       *mid,
        ds4_metal_tensor       *experts,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_metal_tensor *selected,
        const ds4_metal_tensor *weights,
        uint32_t                n_expert,
        float                   clamp,
        const ds4_metal_tensor *x,
        uint32_t                n_tokens) {
    if (rocm_trace()) fprintf(stderr, "ds4: ROCm enter ds4_metal_routed_moe_batch_tensor\n");
    ROCM_TIME_SCOPE("ds4_metal_routed_moe_batch_tensor");
    (void)gate; (void)up; (void)experts;
    return rocm_moe_run(out, mid, model_map, model_size,
                        gate_offset, up_offset, down_offset,
                        gate_type, down_type,
                        gate_expert_bytes, gate_row_bytes,
                        down_expert_bytes, down_row_bytes,
                        expert_in_dim, expert_mid_dim, out_dim,
                        selected, weights, n_expert, n_tokens, clamp, x);
}

#ifdef __cplusplus
}
#endif
