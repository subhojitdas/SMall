// vector_add.cu
//
// My first CUDA kernel: element-wise vector addition, C = A + B.
//
// The mental model coming from CPU code:
//   On a CPU you'd write a loop:  for (i = 0; i < N; i++) C[i] = A[i] + B[i];
//   On a GPU you launch thousands of threads, and EACH thread computes ONE i.
//   The loop disappears; the parallelism replaces it.
//
// Build:   nvcc -O3 -Xcompiler -fopenmp vector_add.cu -o vector_add
//          (-Xcompiler -fopenmp passes the OpenMP flag to the host g++ so the
//           multi-core CPU version below actually runs in parallel. Without it
//           the code still compiles and runs, just single-threaded.)
// Run:     ./vector_add

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#ifdef _OPENMP
#include <omp.h>
#endif

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

// ---------------------------------------------------------------------------
// The CPU reference: the exact same operation, written the "normal" way.
//
// This is the sequential loop that the GPU kernel replaces. One core walks the
// array element by element. It's the baseline we compare the GPU against.
// ---------------------------------------------------------------------------
// The `bias` parameter looks pointless, but it is what stops the compiler from
// cheating. Because bias changes every iteration in the timed loop below, each
// call computes a DIFFERENT result, so the compiler cannot hoist the work out
// of the loop and run it just once. Combined with reading the result back
// (see the checksum in main), this forces all iterations to actually execute.
void vectorAddCPU(const float* A, const float* B, float* C, int n, float bias) {
    for (int i = 0; i < n; i++) {
        C[i] = A[i] + B[i] + bias;
    }
}

// ---------------------------------------------------------------------------
// The multi-core CPU version. Identical loop, but the OpenMP pragma tells the
// compiler to split the iteration space across all available CPU threads.
//
// This is the CPU's real answer to the GPU: don't use one core, use them all.
// It's the fair middle ground between a single core and thousands of GPU
// threads. If OpenMP isn't enabled at compile time, the pragma is ignored and
// this runs single-threaded (still correct, just not parallel).
// ---------------------------------------------------------------------------
void vectorAddCPU_MP(const float* A, const float* B, float* C, int n, float bias) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        C[i] = A[i] + B[i] + bias;
    }
}

int main() {
    // 1<<25 = ~33M elements. Working set is 3 arrays * 33M * 4B = 384 MB, which
    // is ~10x larger than the CPU's ~35 MB L3 cache. This FORCES the CPU to read
    // from DRAM instead of cache, so its measured bandwidth becomes honest.
    // (Try 1<<20 vs 1<<25 and watch the CPU "bandwidth" collapse to reality.)
    const int N = 1 << 25;              // ~33M elements
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

    // --- 4c. Time the SAME operation on the CPU --------------------------
    // We use std::chrono here (a plain CPU clock) because this work runs
    // entirely on the CPU synchronously, so a CPU timer is exactly right.
    // Same recipe as the GPU: one warm-up pass, then average over many iters.
    float* h_C_cpu = (float*)malloc(bytes);

    vectorAddCPU(h_A, h_B, h_C_cpu, N, 0.0f);  // warm-up

    // `sink` is a running checksum we print at the end. Reading one element of
    // the result each iteration makes the output observable, so the compiler
    // cannot delete the computation as dead code.
    volatile float sink = 0.0f;
    auto cpuStart = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; it++) {
        vectorAddCPU(h_A, h_B, h_C_cpu, N, (float)it);  // bias varies each iter
        sink += h_C_cpu[it % N];
    }
    auto cpuEnd = std::chrono::high_resolution_clock::now();

    double cpuTotalMs =
        std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();
    double cpuMsPerIter = cpuTotalMs / iters;
    double cpuGbPerSec = (bytesMoved / (cpuMsPerIter * 1.0e-3)) / 1.0e9;

    // --- 4c-mp. Time the multi-core (OpenMP) CPU version -----------------
    int numThreads = 1;
#ifdef _OPENMP
    numThreads = omp_get_max_threads();
#endif

    vectorAddCPU_MP(h_A, h_B, h_C_cpu, N, 0.0f);  // warm-up (also spins up threads)

    volatile float sinkMp = 0.0f;
    auto mpStart = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; it++) {
        vectorAddCPU_MP(h_A, h_B, h_C_cpu, N, (float)it);
        sinkMp += h_C_cpu[it % N];
    }
    auto mpEnd = std::chrono::high_resolution_clock::now();

    double mpTotalMs =
        std::chrono::duration<double, std::milli>(mpEnd - mpStart).count();
    double mpMsPerIter = mpTotalMs / iters;
    double mpGbPerSec = (bytesMoved / (mpMsPerIter * 1.0e-3)) / 1.0e9;

    // --- 4d. Side-by-side comparison -------------------------------------
    // Two honest views:
    //   * "kernel only": GPU compute vs CPU compute. This flatters the GPU
    //     because it ignores the cost of shipping data over PCIe.
    //   * See the note below about the transfer cost, which is the number that
    //     decides whether offloading to the GPU is actually worth it.
    char cpuMpLabel[32];
    snprintf(cpuMpLabel, sizeof(cpuMpLabel), "CPU (%d cores)", numThreads);

    printf("\n================ CPU vs GPU (vector add, N = %d) ================\n", N);
    printf("%-16s %14s %18s\n", "", "time (ms)", "bandwidth (GB/s)");
    printf("%-16s %14.4f %18.1f\n", "CPU (1 core)", cpuMsPerIter, cpuGbPerSec);
    printf("%-16s %14.4f %18.1f\n", cpuMpLabel,     mpMsPerIter,  mpGbPerSec);
    printf("%-16s %14.4f %18.1f\n", "GPU kernel",   msPerIter,    gbPerSec);
    printf("----------------------------------------------------------------\n");
    printf("GPU vs 1 CPU core:      %.1fx faster\n", cpuMsPerIter / msPerIter);
    printf("GPU vs %d CPU cores:    %.1fx faster\n", numThreads, mpMsPerIter / msPerIter);
    printf("Multi-core CPU scaling: %.1fx over 1 core (ideal would be %dx)\n",
           cpuMsPerIter / mpMsPerIter, numThreads);
    printf("NOTE: this compares compute only. It ignores host<->device copies\n"
           " (the two H2D + one D2H cudaMemcpy above). For a workload this cheap\n"
           " per element, that PCIe transfer usually costs MORE than the compute,\n"
           " so a real end-to-end vector add is often NOT worth offloading. The\n"
           " GPU wins when you do lots of work per byte moved, or keep data\n"
           " resident on the device across many kernels.\n");

    printf("(CPU checksums sink=%.1f sinkMp=%.1f — printed only so the compiler\n"
           " can't optimize the timed loops away)\n", (float)sink, (float)sinkMp);

    free(h_C_cpu);

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
