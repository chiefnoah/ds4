/* Minimal HIP graph capture probe.  Validates that capture+replay works
 * on gfx1151 / ROCm 7 before we wire it into ds4_rocm.c.  Launches a
 * trivial elementwise kernel N times directly (no ds4 wrappers), then
 * captures the same N launches into a graph and times replay.
 *
 * Builds with hipcc directly so we can call hipStreamBeginCapture etc.
 * without touching the public ds4_metal facade. */
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>

__global__ void noop_add(float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] + 1.0f;
}

static double now_sec() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1.0e-9 * (double)t.tv_nsec;
}

#define HIP_CHECK(x) do { hipError_t e = (x); if (e) { \
    fprintf(stderr, "hip err %s at %s:%d: %s\n", #x, __FILE__, __LINE__, hipGetErrorString(e)); \
    return 1; }} while (0)

int main() {
    const int N = 1024;          /* tiny payload so per-call cost is launch overhead */
    const int LAUNCHES = 300;    /* matches per-token kernel count in DSv4 decode */
    const int OUTER = 200;

    int dev_count = 0;
    HIP_CHECK(hipGetDeviceCount(&dev_count));
    if (dev_count == 0) { printf("no hip device\n"); return 77; }

    float *d = nullptr;
    HIP_CHECK(hipMalloc(&d, N * sizeof(float)));
    HIP_CHECK(hipMemset(d, 0, N * sizeof(float)));

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    /* warm */
    for (int j = 0; j < 50; j++) {
        hipLaunchKernelGGL(noop_add, dim3((N+63)/64), dim3(64), 0, stream, d, N);
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    /* baseline: direct launches */
    double t0 = now_sec();
    for (int o = 0; o < OUTER; o++) {
        for (int j = 0; j < LAUNCHES; j++) {
            hipLaunchKernelGGL(noop_add, dim3((N+63)/64), dim3(64), 0, stream, d, N);
        }
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    double dt_direct = now_sec() - t0;

    /* capture LAUNCHES kernels into a graph */
    HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
    for (int j = 0; j < LAUNCHES; j++) {
        hipLaunchKernelGGL(noop_add, dim3((N+63)/64), dim3(64), 0, stream, d, N);
    }
    hipGraph_t graph;
    HIP_CHECK(hipStreamEndCapture(stream, &graph));

    hipGraphExec_t exec;
    HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

    /* warm graph */
    for (int o = 0; o < 5; o++) HIP_CHECK(hipGraphLaunch(exec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    /* replay */
    t0 = now_sec();
    for (int o = 0; o < OUTER; o++) {
        HIP_CHECK(hipGraphLaunch(exec, stream));
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    double dt_graph = now_sec() - t0;

    const double total = (double)OUTER * (double)LAUNCHES;
    printf("direct: %.3f ms total, %.3f us / launch\n",
           dt_direct * 1000.0, dt_direct * 1e6 / total);
    printf("graph : %.3f ms total, %.3f us / launch (%.2fx vs direct)\n",
           dt_graph  * 1000.0, dt_graph  * 1e6 / total,
           dt_direct / dt_graph);

    hipGraphExecDestroy(exec);
    hipGraphDestroy(graph);
    hipStreamDestroy(stream);
    hipFree(d);
    return 0;
}
