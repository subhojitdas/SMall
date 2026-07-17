# GPU Architecture and CUDA Programming: Complete Reference

A comprehensive guide to understanding GPU hardware architecture, the CUDA programming model, and performance optimization — everything you need to start writing CUDA code.

> Based on Modal's GPU Glossary (https://modal.com/gpu-glossary) and NVIDIA documentation.

---

## Table of Contents

1. [The Big Picture: CPU vs GPU](#the-big-picture-cpu-vs-gpu)
2. [GPU Hardware Architecture](#gpu-hardware-architecture)
3. [The Streaming Multiprocessor (SM) — Deep Dive](#the-streaming-multiprocessor-sm--deep-dive)
4. [Execution Units Inside the SM](#execution-units-inside-the-sm)
5. [Memory Hierarchy](#memory-hierarchy)
6. [The CUDA Programming Model](#the-cuda-programming-model)
7. [Thread Hierarchy](#thread-hierarchy)
8. [Execution Model: How Code Actually Runs](#execution-model-how-code-actually-runs)
9. [Performance Concepts](#performance-concepts)
10. [CUDA C++ Programming Basics](#cuda-c-programming-basics)
11. [Putting It All Together: Mental Model](#putting-it-all-together-mental-model)
12. [Key GPU Specifications (H100 Reference)](#key-gpu-specifications-h100-reference)
13. [NVIDIA GPU Architecture Evolution](#nvidia-gpu-architecture-evolution)
14. [Next Steps for Learning CUDA](#next-steps-for-learning-cuda)

---

## The Big Picture: CPU vs GPU

| Aspect | CPU | GPU |
|--------|-----|-----|
| Core count | 8–192 powerful cores | Thousands of simple cores |
| Core complexity | Out-of-order execution, branch prediction, speculation | In-order, pipelined, no speculation |
| Thread switching | ~microseconds (expensive context save/restore) | ~1 nanosecond (registers stay in place) |
| Parallelism strategy | Task parallelism (few threads, complex work) | Data parallelism (massive threads, simple work) |
| Strength | Low-latency serial work | High-throughput parallel work |
| Power per thread | ~1.25 W (AMD EPYC 9965: 500W / 384 threads) | ~0.05 W (H100: 700W / ~16,000 parallel threads) |

A GPU is not "a faster CPU." It is a *throughput machine* — it trades single-thread performance for the ability to run tens of thousands of threads concurrently, hiding memory latency by always having other work ready to execute.

CPUs hide latency from end-users using large hardware-managed caches and sophisticated instruction prediction. This extra hardware limits the fraction of silicon, power, and heat budget that CPUs can allocate to pure computation. GPUs take the opposite approach: for programs where the programmer can express cache behavior explicitly (like neural network inference or matrix multiplication), the result is much higher throughput.

---

## GPU Hardware Architecture

An NVIDIA GPU is organized in a hierarchy of physical components:

```
GPU Die
├── GPU Processing Clusters (GPCs)          [8 on H100]
│   ├── Texture Processing Clusters (TPCs)  [pairs of SMs]
│   │   ├── Streaming Multiprocessor (SM)   ← the fundamental compute unit
│   │   └── Streaming Multiprocessor (SM)
│   └── ...more TPCs
├── ...more GPCs
├── L2 Cache (shared across all SMs)        [50 MiB on H100]
└── GPU RAM (HBM / GDDR — "global memory")  [80 GiB on H100]
```

### GPU Processing Cluster (GPC)

The largest subdivision of compute on the die. A GPC contains multiple TPCs plus a raster engine (for graphics). The name used to stand for "Graphics Processing Cluster" but is now expanded as "GPU Processing Cluster" in NVIDIA documentation.

Since Hopper (compute capability 9.0), there is a "cluster" level in the CUDA thread hierarchy: thread blocks scheduled onto the same GPC can access each other's shared memory via "distributed shared memory."

### Texture Processing Cluster (TPC)

A TPC is a pair of adjacent Streaming Multiprocessors. In older architectures, TPCs were invisible to the programmer — just a physical grouping. Starting with Blackwell (5th-gen Tensor Cores), TPCs became programmable:
- A "CTA pair" in the PTX thread hierarchy maps directly onto a TPC
- PTX instructions include a `.cta_group` field: `.cta_group::1` targets one SM, `::2` targets both SMs in a TPC pair
- These map to `1SM` and `2SM` variants of SASS MMA instructions

### GPU RAM (VRAM)

The bottom-level memory addressable by all SMs:
- Uses Dynamic RAM (DRAM) cells — slower but denser than Static RAM (SRAM) used in caches
- Modern data-center GPUs (H100, B200) use **High-Bandwidth Memory (HBM)** placed on a shared interposer alongside compute dies, reducing latency and increasing bandwidth
- Consumer GPUs use GDDR (Double Data Rate) memory
- Implements "global memory" in the CUDA programming model
- Also stores register data that "spills" from the register file
- H100: 80 GiB HBM3, ~3.35 TB/s bandwidth

---

## The Streaming Multiprocessor (SM) — Deep Dive

The SM is the fundamental processing unit of NVIDIA GPUs — roughly analogous to a CPU core, but designed for massive parallelism rather than single-thread speed.

### What makes an SM different from a CPU core?

- **No speculative execution** — instructions execute in order, pipelined within each instruction
- **No branch prediction** — branches are resolved, not predicted
- **Manages thousands of threads simultaneously** — where a CPU core runs 1–2 threads
- **Context switches in one clock cycle** — registers are pre-allocated per thread, so switching between warps is essentially free (no save/restore)

### SM Internal Components

```
Streaming Multiprocessor (SM)
├── Warp Schedulers (4 per SM on H100)
│   └── Each manages warps of 32 threads
├── CUDA Cores (128 FP32 per SM on H100)
│   ├── FP32 units (single-precision float)
│   ├── INT32 units (integer arithmetic)
│   └── FP64 units (double-precision float)
├── Tensor Cores (4 per SM on H100)
│   └── Matrix multiply-accumulate engines
├── Special Function Units (SFUs)
│   └── Transcendental math (sin, cos, exp, sqrt)
├── Load/Store Units (LSUs)
│   └── Interface between cores and memory subsystems
├── Register File (65,536 × 32-bit registers on H100)
│   └── Fastest storage, private to each thread
└── L1 Data Cache / Shared Memory (256 KiB on H100)
    └── Fast on-chip SRAM shared within a thread block
```

### The H100 SM in Numbers

| Resource | Per SM | Total (132 SMs) |
|----------|--------|------------------|
| Warp Schedulers | 4 | 528 |
| FP32 CUDA Cores | 128 | 16,896 |
| Tensor Cores | 4 (one per scheduler) | 528 |
| Register File | 256 KiB (65,536 × 32-bit) | 33 MiB |
| L1 Cache / Shared Memory | 256 KiB | 33 MiB |
| Max Concurrent Threads | 2,048 (64 warps × 32) | ~270,000 |
| Max Parallel Threads/cycle | 128 (4 schedulers × 32) | 16,896 |

---

## Execution Units Inside the SM

### CUDA Cores

CUDA Cores execute scalar arithmetic instructions. Unlike CPU cores that run independently, groups of CUDA cores are issued the same instruction simultaneously by the Warp Scheduler but apply that instruction to different data (different registers).

Key points:
- The term "CUDA Core" is imprecise — it conflates different hardware units (FP32, INT32, FP64)
- The H100 has 128 "FP32 CUDA Cores" per SM, but this only counts the 32-bit float units — the number of INT32 and FP64 units differs
- For performance estimation, look at the specific unit counts for your operation type
- Historically, GPUs had specialized compute units mapped onto shader pipelines; CUDA Cores represent the move to general-purpose scalar computation
- Groups are commonly 32 threads (a warp), but contemporary GPUs can issue to as few as one thread at a performance cost

### Tensor Cores

Tensor Cores operate on entire matrices with each instruction, delivering approximately **100x more FLOPS** than CUDA Cores for matrix operations.

**How they work:**
- Execute matrix multiply-accumulate (MMA): `D = A × B + C`
- Example: the `HMMA.16816.F32` SASS instruction performs 16×8×16 = 2,048 multiply-accumulate operations
  - `HMMA16` = half-precision (16-bit) inputs
  - `F32` = single-precision (32-bit) output accumulation
  - `16`, `8`, `16` = matrix dimensions (m, n, k)
- All 32 threads in a warp cooperate to produce results (~64 MACs per thread per instruction)
- Power efficiency comes from amortizing instruction fetch/decode across massive data

**Compilation pipeline example (16×16 matmul):**
```
CUDA C++:    wmma::mma_sync(c, a, b, c);
    ↓ nvcc
PTX:         wmma.mma.sync.aligned.col.row.m16n16k16.f32.f32 {...}
    ↓ ptxas  
SASS:        HMMA.1688.F32 R20, R12, R11, RZ   // D = A @ B + 0
             HMMA.1688.F32 R24, R12, R17, RZ   // (4 instructions partition
             HMMA.1688.F32 R20, R14, R16, R20  //  the 16×16 into smaller
             HMMA.1688.F32 R24, R14, R18, R24  //  sub-multiplications)
```

**Precision formats supported (varies by architecture):**
- FP16, BF16, TF32, FP8, INT8, INT4, FP4 (Blackwell)

**Why they matter for AI:**
- Neural network training and inference are dominated by matrix multiplications
- Introduced in Volta (V100) — made NVIDIA GPUs dominant for deep learning
- Only 4 per SM (one per warp scheduler) — large but powerful
- Internal architecture is proprietary (theorized to be systolic arrays)
- Programming: via WMMA intrinsics, CUTLASS/CuTe, cuBLAS, or CuTe DSL (Python)

**Important:** Programming Hopper and Blackwell Tensor Cores for maximum performance cannot be done in pure CUDA C++ — it requires PTX intrinsics for both computation and memory. Use cuBLAS, CUTLASS, or CuTe DSL instead.

### Special Function Units (SFUs)

Hardware accelerators for transcendental math operations:
- `exp` (exponential), `sin`, `cos`, `sqrt`
- Accessed via SASS instructions with `MUFU` prefix (e.g., `MUFU.EX2`, `MUFU.SQRT`)
- Important for neural network activations (softmax uses exp, GELU uses erf)
- Lower throughput than CUDA Cores — use sparingly in hot loops

### Load/Store Units (LSUs)

Dispatch requests to load or store data between:
- The SM's on-chip L1 data cache (fast, SRAM)
- GPU RAM / global memory (slow, DRAM)

They bridge the fastest and slowest levels of the memory hierarchy. All data-dependent computation flows through LSUs. Critical for CUDA programmers because they manage data movement — the most common performance bottleneck.

### Warp Schedulers

The traffic controllers of the SM. Each clock cycle (~1 ns), a Warp Scheduler:
1. Checks which warps have all operands ready (not stalled on memory or dependencies)
2. Selects an eligible warp
3. Issues its next instruction to the execution units

**Why this is the GPU's secret weapon:**
- Each thread's registers are pre-allocated from the register file (never saved/restored)
- Switching between warps costs exactly **one clock cycle** — over 1000x faster than a CPU context switch
- CPU context switches degrade performance through cache miss rates; GPU warp switches don't (L1 caches are programmer-managed and shared between co-scheduled warps)
- This is the foundation of **latency hiding** — the key to GPU performance

The Warp Scheduler also manages execution state: tracking which warps are active, stalled, eligible (ready to issue), or selected (currently issuing).

---

## Memory Hierarchy

The GPU memory system trades speed for capacity at every level:

```
Speed       Memory Level          Scope              Capacity (H100)    Latency
──────────────────────────────────────────────────────────────────────────────────
Fastest  →  Registers             Per-thread         ~255 regs/thread   ~1 cycle
            ↓
         →  L1 / Shared Memory    Per-SM (block)     256 KiB / SM       ~30 cycles
            ↓
         →  L2 Cache              Shared (all SMs)   50 MiB             ~200 cycles
            ↓
Slowest  →  GPU RAM (Global)      Shared (all SMs)   80 GiB             ~400 cycles
```

### Registers (Thread-Private)

- The fastest memory — ~10x faster than L1
- Backed by the SM's physical register file (SRAM)
- 32-bit wide; dynamically combined for 64-bit or split for 16-bit data
- Physical registers back virtual registers defined in PTX
- Allocated at compile time by `ptxas` (PTX → SASS compiler)
- **Trade-off:** more registers per thread → fewer threads fit on SM → lower occupancy → less latency hiding

### L1 Data Cache / Shared Memory (Block-Scope)

- Physically stored in SRAM within the SM
- Co-located with and ~10x slower than registers
- Each SM has 256 KiB on H100 (33 MiB total across all 132 SMs)
- **Programmer-managed** — unlike CPU L1 caches which are hardware-managed
- Partitioned among thread blocks scheduled onto the SM
- Accessed by Load/Store Units

**Shared Memory** is the programmer-visible portion of the L1 cache:
- Shared across all threads in a thread block
- Typical usage pattern:
  1. Load a tile of data from global memory → shared memory
  2. Synchronize threads (`__syncthreads()`)
  3. Compute using the fast shared memory
  4. Write results back to global memory
- Organized into **32 banks** — simultaneous accesses to the same bank serialize (bank conflicts)

### L2 Cache (Shared Across All SMs)

- 50 MiB on H100
- Hardware-managed (unlike shared memory)
- Automatically caches global memory accesses
- Shared by all SMs on the device

### Global Memory (Grid-Scope)

- Physically stored in GPU RAM (HBM or GDDR)
- "Global in scope and lifetime" — accessible by every thread in the grid, persists for program duration
- Allocated from the host via `cudaMalloc` / CUDA Driver API
- Synchronized via atomic operations or barriers
- **Memory coalescing**: consecutive threads accessing consecutive addresses merge into efficient bulk transfers; scattered patterns waste bandwidth
- Confusing naming: the `__global__` keyword in CUDA C++ denotes kernel functions, not global memory

### Tensor Memory Accelerator (TMA) — Hopper/Blackwell Only

Specialized hardware that accelerates access to multi-dimensional arrays in GPU RAM:
- Loads data directly from global memory to shared memory/L1, **bypassing registers**
- Hardware calculates addresses for bulk affine memory operations (`addr = width * base + offset`)
- Reduces register pressure and CUDA Core demand for address computation
- **Asynchronous**: a single thread triggers a large copy, then rejoins its warp for other work
- Enables producer-consumer patterns with async completion detection
- Despite the name, TMA does NOT accelerate operations using Tensor Memory (a separate Blackwell feature)

### Tensor Memory (Blackwell Only)

Specialized memory in Blackwell SMs for storing Tensor Core inputs/outputs:
- Data moved collectively by four warps (a warpgroup)
- Highly restricted access patterns
- For MMA operations: accumulator D must be in Tensor Memory, left matrix A may be in Tensor Memory or shared memory, right matrix B must be in shared memory
- Rationale: accumulators are accessed more frequently during matmuls, benefiting from shorter wiring to Tensor Cores

---

## The CUDA Programming Model

CUDA provides a programming abstraction that maps naturally onto GPU hardware. The key insight: you write one function (a kernel) that describes what a single thread does, then launch millions of instances organized into a hierarchy.

### What is a Kernel?

A kernel is "the unit of CUDA code that programmers typically write and compose, akin to a procedure or function in languages targeting CPUs." But unlike a CPU function:
- It launches once but executes across many threads simultaneously
- Executions occur concurrently with non-deterministic ordering
- All threads form a "thread block grid" — the highest level of the thread hierarchy

```cuda
// Kernels are marked with __global__
__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Launched from the host with <<<grid, block>>> syntax
vectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);
```

Key characteristics:
- Accept pointers to device (GPU) global memory
- Return nothing (`void`) — they mutate memory
- Each thread computes its unique index to determine which data it operates on
- Optimization means mapping work onto the thread hierarchy to leverage shared memory and increase arithmetic intensity

---

## Thread Hierarchy

CUDA organizes threads in three levels that map directly to hardware:

```
Grid (entire kernel launch)              → runs across ALL SMs
├── Block 0 (thread block / CTA)         → scheduled onto ONE SM
│   ├── Warp 0 (threads 0–31)            → scheduled together by Warp Scheduler
│   ├── Warp 1 (threads 32–63)           → scheduled together
│   └── ...more warps
├── Block 1                              → another SM (maybe same, maybe different)
│   ├── Warp 0
│   └── ...
└── ...more blocks (execute in ANY order)
```

### Thread

The finest granularity of execution:
- Has a unique ID within its block (`threadIdx.x/y/z`)
- Owns private registers from the SM's register file
- Executes the kernel code
- Grouped into warps of 32 for scheduling
- "Little else" beyond registers — lightweight by design

### Warp (32 Threads)

The actual unit of hardware execution:
- All threads execute the same instruction each cycle (**SIMT**: Single-Instruction, Multiple-Thread)
- **Not part of CUDA's official programming model** — it's a hardware implementation detail (like CPU cache lines)
- When threads take different branches ("warp divergence"), both paths execute serially
- A warp awaiting operands (e.g., from a memory load) is "stalled"
- Switching to another warp happens in one cycle — this is latency hiding
- Warp size is technically machine-dependent but always 32 in practice
- Equivalents: subgroups (WebGPU), waves (DirectX), simdgroups (Metal)

### Thread Block (Cooperative Thread Array / CTA)

A group of threads (up to 1024) that:
- Are **guaranteed to run on the same SM**
- Share the SM's L1 cache / shared memory
- Can synchronize with `__syncthreads()` barriers
- Have block-level IDs (`blockIdx.x/y/z`)
- Composed of one or more warps

**Resource constraints determine how many blocks fit on one SM:**
- Register file capacity (computed at compile time per block)
- Shared memory capacity
- Maximum warp slots

**Important:** blocking on another CTA can easily lead to deadlock because CTA scheduling order is non-deterministic.

### Grid

The complete set of thread blocks for a kernel launch:
- Blocks execute with **no guaranteed order** (fully serial to fully parallel)
- Blocks **cannot synchronize** with each other via barriers
- Communication between blocks requires atomic operations on global memory
- Your code must be correct regardless of block execution order
- Can be 1D, 2D, or 3D

### Warpgroup (Hopper+)

A set of four contiguous warps (128 threads) where the first warp's rank is a multiple of 4. Used for advanced Tensor Core programming on Hopper and later architectures.

### Why This Design?

The block-order-independence constraint means the GPU can schedule blocks onto SMs in any order and any degree of parallelism. The same CUDA code scales from a small GPU (running blocks sequentially) to a massive GPU (running all blocks in parallel) without code changes.

---

## Execution Model: How Code Actually Runs

Here's what happens when you launch a kernel:

1. **Host launches kernel** with grid dimensions (number of blocks) and block dimensions (threads per block)
2. **GPU scheduler** distributes thread blocks across available SMs
3. **Each SM** receives one or more blocks and partitions their threads into warps of 32
4. **Warp Schedulers** (4 per SM on H100) pick eligible warps each cycle and issue instructions
5. **Cores execute** — CUDA Cores for scalar math, Tensor Cores for matrix ops, SFUs for transcendentals
6. **Memory accesses** go through LSUs → L1 cache → (L2 cache) → GPU RAM
7. **When a warp stalls** (waiting for memory or dependencies), the scheduler instantly switches to another ready warp — this is **latency hiding**
8. **Results** are written back to global memory where the host can read them

### Latency Hiding: The GPU's Superpower

CPUs hide latency with caches and speculative execution. GPUs hide latency with massive parallelism:

- A global memory load takes ~400 cycles
- The warp scheduler switches to another warp in **1 cycle**
- If you have enough warps resident, the SM is always doing useful work while other warps wait
- This is why **occupancy** matters — more resident warps = more opportunities to hide latency
- GPUs are throughput machines: they don't make individual operations faster, they keep all hardware busy

### The Compilation Pipeline

```
CUDA C++ source code
    ↓ nvcc (NVIDIA CUDA Compiler Driver) or NVRTC (runtime compiler)
PTX (Parallel Thread eXecution) — virtual ISA, portable across architectures
    ↓ ptxas (part of nvcc, or JIT at runtime)
SASS (Streaming ASSembler) — architecture-specific machine code
    ↓ executed by SMs
```

- **PTX** is like LLVM IR for GPUs — portable, versioned by "compute capability"
- **SASS** is what actually runs — inspect it for real instruction scheduling and memory operations
- Compatibility: SASS from one SM architecture is NOT guaranteed to run on another major version
- PTX has forward compatibility: old PTX runs on new GPUs (the "onion layer model")

### Compute Capability

The versioning system for GPU instruction set compatibility:
- Major.minor format (e.g., 9.0 for Hopper)
- Forward compatible: old PTX code runs on new GPUs
- NOT backward compatible: new SASS won't run on older hardware
- Target with `nvcc -arch=sm_90` (for SASS) or `-gencode arch=compute_90,code=sm_90`
- Hopper introduced suffix `a` (9.0a) for features without future compatibility guarantee
- Blackwell introduced suffix `f` (10.0f) with SemVer-style compatibility

---

## Performance Concepts

### Arithmetic Intensity and the Roofline Model

The ratio of compute operations to memory bytes accessed:

```
Arithmetic Intensity = FLOPs / Bytes Transferred
```

This determines whether your kernel is:
- **Compute-bound** (high arithmetic intensity): limited by CUDA/Tensor Core throughput
- **Memory-bound** (low arithmetic intensity): limited by memory bandwidth

The **Roofline Model** visualizes this as a plot: arithmetic intensity on x-axis, achievable FLOPS on y-axis. Performance is bounded by:
```
Achievable FLOPS = min(Peak Compute, Memory Bandwidth × Arithmetic Intensity)
```

Most naive kernels are memory-bound. Optimization usually means increasing arithmetic intensity via tiling and data reuse in shared memory/registers.

### Occupancy

```
Occupancy = Active Warps on SM / Maximum Possible Warps on SM
```

Higher occupancy generally means more latency-hiding opportunities. Occupancy is limited by:
- **Register usage per thread** — more registers = fewer threads fit = lower occupancy
- **Shared memory per block** — more shared memory = fewer blocks fit
- **Block size** — must be a multiple of 32 (warp size) for efficiency

**Important nuance:** maximum occupancy does not always mean maximum performance. Sometimes using more registers/shared memory per thread gives better overall performance despite lower occupancy (better data reuse outweighs less latency hiding).

### Memory Coalescing

When threads in a warp access consecutive memory addresses, the hardware coalesces these into a single efficient transaction. Scattered access patterns waste bandwidth.

```
Good (coalesced):   Thread 0 → addr[0], Thread 1 → addr[1], Thread 2 → addr[2], ...
Bad (scattered):    Thread 0 → addr[0], Thread 1 → addr[1000], Thread 2 → addr[7], ...
```

This is why **Structure of Arrays (SoA)** often outperforms **Array of Structures (AoS)** on GPUs.

### Warp Divergence

When threads in a warp take different branches:
- Both paths execute serially (not in parallel)
- Threads not on the current path are masked (disabled but consume a slot)
- Performance degrades proportionally to divergence

```cuda
// BAD: threads in same warp diverge (odd/even split)
if (threadIdx.x % 2 == 0) { doA(); } else { doB(); }

// BETTER: divergence across warp boundaries (entire warps take one path)
if (threadIdx.x / 32 < N) { doA(); } else { doB(); }
```

### Bank Conflicts

Shared memory is organized in **32 banks** (one per thread in a warp). When multiple threads simultaneously access different addresses in the same bank, accesses serialize. Design access patterns so each thread hits a different bank.

### Register Pressure

When a kernel uses many registers per thread:
- Fewer threads can be resident on the SM (lower occupancy)
- Less latency hiding possible
- The compiler may "spill" registers to slow local memory (backed by GPU RAM!)

Control with `__launch_bounds__` or `maxrregcount` compiler flags.

### Scoreboard Stalls

A warp stalls when an instruction cannot issue because it depends on the result of a prior instruction that hasn't completed. Common causes:
- Waiting for global memory loads (~400 cycles)
- Waiting for shared memory loads (~30 cycles)
- Waiting for long-latency arithmetic (FP64, SFU operations)

Solution: ensure enough warps are resident to hide these stalls.

### Little's Law (Applied to GPUs)

```
Concurrency Required = Latency × Throughput
```

To fully utilize a pipe with throughput T and latency L, you need L × T operations in flight. This quantifies how much parallelism you need to saturate hardware resources.

### Performance at NVIDIA

"Performance is the product." — NVIDIA internal slogan

For GPU programming, correctness is necessary but not sufficient. If you cannot achieve superior performance (per second, per dollar, or per watt), the application has failed. GPU programming is too hard and too expensive for anything else to be the case.

---

## CUDA C++ Programming Basics

CUDA C++ extends standard C++ with GPU-specific constructs:

### Function Qualifiers

```cuda
__global__ void kernel(...)    // Launched from host, runs on device (the kernel)
__device__ void helper(...)    // Called from device code only
__host__   void hostFunc(...)  // Runs on host (default, can omit)
__host__ __device__ void both(...)  // Compiled for both host and device
```

### Built-in Variables (available inside kernels)

```cuda
threadIdx.x, threadIdx.y, threadIdx.z   // Thread's position within its block
blockIdx.x, blockIdx.y, blockIdx.z      // Block's position within the grid
blockDim.x, blockDim.y, blockDim.z      // Dimensions of each block
gridDim.x, gridDim.y, gridDim.z         // Dimensions of the grid
warpSize                                 // Always 32 on current hardware
```

### Memory Management

```cuda
// Allocate GPU memory
float* d_data;
cudaMalloc(&d_data, n * sizeof(float));

// Copy host → device
cudaMemcpy(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice);

// Copy device → host
cudaMemcpy(h_data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost);

// Free GPU memory
cudaFree(d_data);
```

### Shared Memory Declaration

```cuda
__global__ void kernel() {
    __shared__ float tile[256];      // Static: size known at compile time
    extern __shared__ float dyn[];   // Dynamic: size set at kernel launch

    // Load data into shared memory
    tile[threadIdx.x] = globalData[globalIdx];
    __syncthreads();  // Barrier — ALL threads in block must reach here before any continue

    // Now safe to read other threads' data from tile[]
    float val = tile[threadIdx.x + 1];
}
```

### Kernel Launch Syntax

```cuda
// <<<gridDim, blockDim, sharedMemBytes, stream>>>
myKernel<<<numBlocks, threadsPerBlock>>>(arg1, arg2);

// With dynamic shared memory and CUDA stream
myKernel<<<grid, block, sharedBytes, stream>>>(args...);

// 2D grid and block example
dim3 grid(width/16, height/16);
dim3 block(16, 16);
matMul<<<grid, block>>>(A, B, C, width);
```

### Synchronization Primitives

```cuda
__syncthreads();           // Block-level barrier (all threads in block must reach)
__syncwarp();              // Warp-level barrier (all threads in warp)
atomicAdd(&counter, 1);    // Atomic operation (safe across all threads)
__threadfence();           // Memory fence (ordering guarantee within device)
__threadfence_block();     // Memory fence within block
```

### Warp-Level Primitives (Modern CUDA)

```cuda
// Warp shuffle — exchange data between threads without shared memory
float val = __shfl_down_sync(0xFFFFFFFF, myVal, offset);
float val = __shfl_xor_sync(mask, myVal, laneMask);

// Warp vote — collective decisions
bool allTrue = __all_sync(mask, predicate);
bool anyTrue = __any_sync(mask, predicate);
unsigned ballot = __ballot_sync(mask, predicate);
```

### Error Checking

```cuda
cudaError_t err = cudaMalloc(&ptr, size);
if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
}

// After kernel launch (asynchronous — must sync first)
myKernel<<<grid, block>>>(args);
cudaDeviceSynchronize();
err = cudaGetLastError();
```

### Complete Example: Vector Addition

```cuda
#include <stdio.h>

__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int n = 1 << 20;  // 1M elements
    size_t bytes = n * sizeof(float);

    // Allocate host memory
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);

    // Initialize
    for (int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Copy to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    vectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);

    // Copy result back
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Verify
    for (int i = 0; i < n; i++) {
        if (h_c[i] != 3.0f) { printf("Error at %d!\n", i); break; }
    }
    printf("Success!\n");

    // Cleanup
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}
```

### Complete Example: Tiled Matrix Multiplication (Shared Memory)

```cuda
#define TILE_SIZE 16

__global__ void matMul(const float* A, const float* B, float* C, int N) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    // Loop over tiles along the shared dimension
    for (int t = 0; t < N / TILE_SIZE; t++) {
        // Collaboratively load tiles into shared memory
        tileA[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        __syncthreads();

        // Compute partial dot product from this tile
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        __syncthreads();
    }

    C[row * N + col] = sum;
}
```

This achieves much higher arithmetic intensity than the naive version because each element loaded into shared memory is reused TILE_SIZE times.

---

## Putting It All Together: Mental Model

When you write a CUDA kernel, think of it as:

1. **You write one function** that describes what a single thread does
2. **You launch millions of instances** organized as blocks in a grid
3. **The hardware maps blocks to SMs** — each SM runs one or more blocks
4. **Threads within a block** are partitioned into warps of 32, executed by warp schedulers
5. **Memory access is your bottleneck** — use shared memory to stage data close to compute
6. **Latency is hidden by parallelism** — keep enough warps resident so the SM always has work

### The Optimization Loop

```
1. Write a correct kernel
2. Profile it (Nsight Compute for kernel metrics, Nsight Systems for timeline)
3. Identify the bottleneck:
   - Memory-bound? → Improve coalescing, tile into shared memory, reduce global accesses
   - Compute-bound? → Use Tensor Cores, reduce unnecessary operations, vectorize
   - Latency-bound? → Increase occupancy, reduce register pressure, prefetch
   - Launch-overhead-bound? → Fuse kernels, use CUDA Graphs, batch work
4. Apply fix, re-profile, repeat
```

### Common Optimization Patterns

| Pattern | When to Use | How |
|---------|-------------|-----|
| Shared memory tiling | Matrix ops, stencils | Load a tile into `__shared__`, reuse many times |
| Memory coalescing | Any global memory access | Ensure consecutive threads access consecutive addresses |
| Warp-level reduction | Sum/max/min across threads | Use `__shfl_down_sync` instead of shared memory |
| Loop unrolling | Inner loops with known bounds | `#pragma unroll` or manual unroll |
| Occupancy tuning | Latency-bound kernels | Adjust block size, limit registers |
| Tensor Core usage | Matrix multiplication | Use WMMA, CUTLASS, or cuBLAS |
| CUDA Graphs | Repeated kernel sequences | Capture and replay entire workflows |
| Async memory (TMA) | Large data movement (Hopper+) | Overlap compute with asynchronous copies |

---

## Key GPU Specifications (H100 Reference)

| Specification | H100 SXM5 |
|---------------|------------|
| Architecture | Hopper |
| SMs | 132 |
| CUDA Cores (FP32) | 16,896 (128/SM) |
| Tensor Cores (4th gen) | 528 (4/SM) |
| Warp Schedulers | 528 (4/SM) |
| Max Threads per SM | 2,048 |
| Max Concurrent Threads | ~270,000 |
| Register File per SM | 256 KiB (65,536 × 32-bit) |
| L1 Cache / Shared Mem per SM | 256 KiB |
| L2 Cache | 50 MiB |
| GPU RAM | 80 GiB HBM3 |
| Memory Bandwidth | 3.35 TB/s |
| FP32 TFLOPS | 67 |
| TF32 Tensor Core TFLOPS | 989 |
| FP16 Tensor Core TFLOPS | 1,979 |
| FP8 Tensor Core TFLOPS | 3,958 |
| Interconnect | NVLink 4.0 (900 GB/s) |
| TDP | 700W |
| Compute Capability | 9.0 |
| Key Features | TMA, FP8, Thread Block Clusters |

---

## NVIDIA GPU Architecture Evolution

| Architecture | Year | SM Version | Key Innovation | Example GPU |
|-------------|------|------------|----------------|-------------|
| Tesla | 2008 | 1.x | First CUDA architecture, unified shaders | GTX 280 |
| Fermi | 2010 | 2.x | L1/L2 caches, ECC, concurrent kernels | GTX 480 |
| Kepler | 2012 | 3.x | Dynamic parallelism, Hyper-Q, SMX | K80 |
| Maxwell | 2014 | 5.x | Energy efficiency, shared memory redesign | GTX 980 |
| Pascal | 2016 | 6.x | HBM2, NVLink, unified memory, FP16 | P100 |
| Volta | 2017 | 7.0 | **Tensor Cores**, independent thread scheduling | V100 |
| Turing | 2018 | 7.5 | RT Cores, INT8/INT4 Tensor Cores | RTX 2080 |
| Ampere | 2020 | 8.x | TF32, structural sparsity, 3rd-gen Tensor Cores | A100 |
| Hopper | 2022 | 9.0 | TMA, FP8, Thread Block Clusters, DPX | H100 |
| Blackwell | 2024 | 10.x | 5th-gen Tensor Cores, TPC programming, FP4 | B200 |

### Key Milestones for AI/CUDA:
- **Volta (2017)**: Tensor Cores made NVIDIA dominant for deep learning
- **Ampere (2020)**: TF32 format = "free" speedup for existing FP32 code on Tensor Cores
- **Hopper (2022)**: TMA enables async data movement; FP8 training became practical
- **Blackwell (2024)**: FP4 inference; TPC-level cooperative operations

---

## Host Software Stack

### The Driver Stack

```
Your Application (Python/C++/etc.)
    ↓
CUDA Runtime API (libcudart.so)     ← Higher-level, auto-manages contexts
    ↓
CUDA Driver API (libcuda.so)        ← Lower-level, explicit context management
    ↓
nvidia.ko (kernel module)           ← OS kernel driver
    ↓
GPU Hardware
```

- **nvidia-smi**: CLI to query GPU state (utilization, temperature, memory, power)
- **NVML** (libnvml.so): Management library behind nvidia-smi
- **nvcc**: Compiler driver that orchestrates host and device compilation
- **NVRTC**: Runtime compilation — compile CUDA at runtime (JIT)
- **Nsight Systems**: Timeline profiler (where time goes across CPU and GPU)
- **Nsight Compute**: Kernel profiler (detailed metrics for a single kernel)
- **CUPTI**: Profiling tools interface (timestamps, counters, traces)

### Key Libraries

| Library | Purpose |
|---------|---------|
| **cuBLAS** | Optimized BLAS (matrix multiply, etc.) — uses Tensor Cores automatically |
| **cuDNN** | Deep learning primitives (convolution, normalization, activation) |
| **CUTLASS** | C++ templates for high-performance linear algebra kernels |
| **CuTe** | Header-only C++ library for tensor layout algebra (within CUTLASS) |
| **CuTe DSL** | Python DSL for writing high-performance kernels with CuTe abstractions |

### CUDA Graphs

A CUDA Graph is a pre-recorded graph of kernel launches and other work submitted to the device all at once. Benefits:
- Eliminates per-kernel launch overhead (significant for many small kernels)
- GPU can schedule the entire graph optimally
- Ideal for repeated workloads (inference, simulation steps)

---

## Next Steps for Learning CUDA

### Beginner Path

1. **Install CUDA Toolkit** — get `nvcc`, `cuda-gdb`, `nsight-compute`
2. **Vector Addition** — the "hello world" (one thread per element)
3. **Matrix Addition** — practice 2D indexing
4. **Reduction (sum)** — learn shared memory and `__syncthreads()`
5. **Matrix Multiplication (naive)** — understand why it's slow
6. **Matrix Multiplication (tiled)** — the "aha moment" for shared memory

### Intermediate Path

7. **Profile with Nsight Compute** — understand where time goes
8. **Memory coalescing experiments** — AoS vs SoA, strided vs sequential
9. **Warp-level primitives** — `__shfl_down_sync` for fast reductions
10. **Streams and async** — overlap compute with memory transfers
11. **CUDA Graphs** — reduce launch overhead for repeated kernel sequences
12. **cuBLAS / cuDNN** — use optimized libraries as building blocks

### Advanced Path

13. **Tensor Core programming** — via WMMA API or CUTLASS
14. **CuTe (CUTLASS)** — layout algebra for high-performance kernels
15. **CuTe DSL (Python)** — high-performance kernels with Python productivity
16. **PTX inline assembly** — when you need precise hardware control
17. **TMA (Hopper+)** — asynchronous bulk data movement
18. **Multi-GPU / NVLink** — scaling beyond one device
19. **Custom PyTorch CUDA extensions** — integrate with deep learning frameworks

### Essential Resources

| Resource | What It's For |
|----------|--------------|
| [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) | Canonical reference |
| [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) | Performance tuning |
| [Modal GPU Glossary](https://modal.com/gpu-glossary) | Clear, interlinked explanations |
| [Nsight Compute Docs](https://docs.nvidia.com/nsight-compute/) | Kernel profiling |
| [CUTLASS GitHub](https://github.com/NVIDIA/cutlass) | High-performance GEMM templates |
| [GPU MODE Discord](https://discord.gg/gpumode) | Active community |
| [Fabien Sanglard's CUDA History](https://fabiensanglard.net/cuda/) | Architecture evolution |
| [What Every Programmer Should Know About Memory](https://people.freebsd.org/~lstewart/articles/cpumemory.pdf) | Memory fundamentals |
| [Lindholm et al., 2008](https://www.cs.cmu.edu/afs/cs/academic/class/15869-f11/www/readings/lindholm08_tesla.pdf) | Original CUDA architecture paper |
| [H100 Whitepaper](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c) | Current-gen details |
| [GPU Glossary on GitHub](https://github.com/modal-labs/gpu-glossary) | Contribute corrections |

---

## Glossary Quick Reference

| Term | Definition |
|------|-----------|
| SM | Streaming Multiprocessor — the fundamental compute unit |
| Warp | 32 threads scheduled and executed together (SIMT) |
| CTA | Cooperative Thread Array = thread block |
| SIMT | Single Instruction, Multiple Threads |
| Occupancy | Active warps / max possible warps on an SM |
| Coalescing | Merging thread memory accesses into bulk transactions |
| Bank conflict | Threads hitting same shared memory bank (serializes) |
| Register spill | Overflow from registers to slow local memory |
| Latency hiding | Warp switching to stay busy while waiting |
| Arithmetic intensity | FLOPs / bytes transferred |
| TMA | Tensor Memory Accelerator (async bulk data, Hopper+) |
| PTX | Parallel Thread eXecution (virtual ISA) |
| SASS | Streaming ASSembler (actual machine code) |
| Compute capability | SM architecture version number |
| CUDA Graph | Pre-recorded kernel launch sequence |
| NVLink | High-bandwidth GPU interconnect |
| HBM | High-Bandwidth Memory (stacked DRAM) |
| Roofline | Model bounding perf by min(compute, bandwidth × intensity) |
| SFU | Special Function Unit (sin, cos, exp, sqrt) |
| LSU | Load/Store Unit (memory access dispatch) |
| TPC | Texture Processing Cluster (pair of SMs) |
| GPC | GPU Processing Cluster (group of TPCs) |
