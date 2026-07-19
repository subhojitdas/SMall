// vector_add.cu
//
// My first CUDA kernel: element-wise vector addition, C = A + B.
//
// The mental model coming from CPU code:
//   On a CPU you'd write a loop:  for (i = 0; i < N; i++) C[i] = A[i] + B[i];
//   On a GPU you launch thousands of threads, and EACH thread computes ONE i.
//   The loop disappears; the parallelism replaces it.
//
// Build:   nvcc -O3 vector_add.cu -o vector_add
// Run:     ./vector_add

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Error-checking helper.
//
// Almost every CUDA runtime call returns a cudaError_t. Beginners skip checking
// these and then spend hours confused. Wrap every call. This macro prints the
// file/line and the human-readable error, then exits.
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s:%d: '%s' -> %s\n",                 \
                    __FILE__, __LINE__, #call, cudaGetErrorString(err__));    \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// ---------------------------------------------------------------------------
// The kernel.
//
// __global__  means: this function runs ON the GPU (device) and is LAUNCHED
//             FROM the CPU (host). It must return void.
//
// Every thread that runs this kernel executes the same code, but with a unique
// index. We compute that index from three built-in variables:
//
//   blockIdx.x  - which block this thread is in       (0 .. gridDim.x-1)
//   blockDim.x  - how many threads per block           (a number you choose)
//   threadIdx.x - this thread's position in its block  (0 .. blockDim.x-1)
//
// So the global index of a thread is:
//   i = blockIdx.x * blockDim.x + threadIdx.x
//
// This is THE most important line in beginner CUDA. Draw it out on paper once:
// block 0 owns indices [0, blockDim), block 1 owns [blockDim, 2*blockDim), etc.
//
// The `if (i < n)` guard matters because we usually launch a few more threads
// than elements (n is rarely a perfect multiple of the block size). Threads
// with i >= n must do nothing, or they'd read/write out of bounds.
// ---------------------------------------------------------------------------
__global__ void vectorAdd(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    const int N = 1 << 20;              // ~1M elements
    const size_t bytes = N * sizeof(float);

    // --- 1. Allocate host (CPU) memory and fill inputs -------------------
    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // --- 2. Allocate device (GPU) memory ---------------------------------
    // The GPU has its own separate memory. Pointers from cudaMalloc are device
    // pointers; you cannot dereference them on the CPU.
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    // --- 3. Copy inputs from host to device ------------------------------
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // --- 4. Launch the kernel --------------------------------------------
    // Choose a block size (threads per block). 256 is a common, safe default.
    // Then compute how many blocks we need to cover all N elements, rounding up.
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Warm-up launch. The FIRST kernel launch in a program pays one-time costs
    // (context creation, module load, JIT). If you time that launch you measure
    // startup, not the kernel. So we run once untimed to "warm up" the device.
    // The <<<blocks, threads>>> syntax is the kernel launch configuration.
    // This launch is asynchronous: the CPU keeps going immediately.
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());        // catch launch-config errors
    CUDA_CHECK(cudaDeviceSynchronize());   // wait + catch kernel runtime errors

    // --- 4b. Time the kernel with CUDA events ----------------------------
    // You cannot time a GPU kernel with a CPU clock reliably, because the launch
    // is asynchronous. CUDA events are timestamps recorded IN the GPU's stream,
    // so the elapsed time between two events is real on-device execution time.
    //
    // We run many iterations and average, because a single measurement of a
    // ~microsecond kernel is dominated by noise.
    const int iters = 100;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < iters; it++) {
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop)); // wait until the stop event completes

    float totalMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&totalMs, start, stop)); // milliseconds
    float msPerIter = totalMs / iters;

    // Effective bandwidth: vector add reads A and B and writes C, so it moves
    // 3 * N * sizeof(float) bytes per launch. Convert bytes/ms to GB/s.
    //   bytes / (ms * 1e-3)  ->  bytes/sec;  / 1e9  ->  GB/s.
    double bytesMoved = 3.0 * (double)N * sizeof(float);
    double gbPerSec = (bytesMoved / (msPerIter * 1.0e-3)) / 1.0e9;

    printf("\nKernel time: %.4f ms/launch (avg over %d iters)\n", msPerIter, iters);
    printf("Effective bandwidth: %.1f GB/s\n", gbPerSec);
    printf("(T4 peak is ~320 GB/s; this kernel is memory-bound, so that ratio\n"
           " is your real efficiency number.)\n");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // --- 5. Copy the result back to the host -----------------------------
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // --- 6. Verify correctness against the CPU expectation ---------------
    // A habit worth forming now: always validate GPU results.
    float maxError = 0.0f;
    for (int i = 0; i < N; i++) {
        maxError = fmaxf(maxError, fabsf(h_C[i] - 3.0f));
    }
    printf("Max error: %f (expected 0.0)\n", maxError);

    // --- Print a small sample of the result ------------------------------
    // The vector has N (~1M) elements, so we print only the first few.
    // Printing all of them would flood the terminal.
    const int sample = 20;
    printf("First %d elements of C = A + B:\n", sample);
    for (int i = 0; i < sample && i < N; i++) {
        printf("  C[%d] = %.1f + %.1f = %.1f\n", i, h_A[i], h_B[i], h_C[i]);
    }

    // --- 7. Free everything ----------------------------------------------
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
