# GPU Architecture, CUDA C++, PTX, and SASS Learning Roadmap

This roadmap is intended for an experienced software engineer who already understands
Java, Python, distributed services, deep learning, and transformer architecture. The
goal is not only to write correct CUDA kernels, but to explain and improve their
performance using measurements.

Suggested pace: **8-10 hours per week for 16 weeks**.

## Learning Order

1. GPU architecture and the GPU performance model
2. CUDA C++ programming
3. Profiling and performance engineering
4. PTX inspection and limited inline PTX
5. SASS analysis
6. Tensor Cores and transformer-oriented kernels

PTX is a virtual instruction set that NVIDIA's driver compiles for a target GPU. SASS
is the machine instruction set for a particular NVIDIA architecture. Learn CUDA C++
and the performance model first; use PTX and SASS to answer specific compiler and
hardware questions.

## Prerequisites and Setup

You need:

- An NVIDIA GPU supported by a current CUDA Toolkit
- A Linux environment, preferably Ubuntu
- A recent C++ compiler supported by the installed CUDA Toolkit
- CUDA Toolkit tools: `nvcc`, `ncu`, `nsys`, `compute-sanitizer`, `cuobjdump`, and
  `nvdisasm`
- CMake or a simple Makefile for repeatable builds
- Python and PyTorch for reference implementations and result validation

Start here:

- [CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [CUDA Toolkit documentation](https://docs.nvidia.com/cuda/)
- [CUDA GPU compute capability table](https://developer.nvidia.com/cuda-gpus)
- [CUDA Samples](https://github.com/NVIDIA/cuda-samples)

Record the GPU model, compute capability, driver version, CUDA version, compiler
flags, input sizes, and warm-up policy with every benchmark.

## Weeks 1-2: GPU Architecture and Performance Model

### Learn

- CPU latency optimization versus GPU throughput optimization
- Streaming Multiprocessors (SMs), warp schedulers, and execution units
- Grids, thread blocks, warps, and threads
- SIMT execution and branch divergence
- Registers, shared memory, L1/L2 cache, and global memory
- Coalesced memory access and shared-memory bank conflicts
- Latency hiding, occupancy, and register pressure
- Arithmetic intensity and the Roofline model

### Study

- [Existing architecture reference in this repository](gpu_architecture_cuda_pytorch_reference.md)
- [CUDA Programming Guide: Programming Model](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programming-model)
- [CUDA Programming Guide: Hardware Implementation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#hardware-implementation)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Nsight Compute Roofline analysis](https://developer.nvidia.com/blog/accelerating-hpc-applications-with-nsight-compute-roofline-analysis/)
- [Programming Massively Parallel Processors, 4th edition](https://www.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0)
- [Stanford CS149: Parallel Computing](https://gfxcourses.stanford.edu/cs149/fall23/)

For architecture-specific details, read the tuning guide matching the available GPU:

- [Ampere tuning guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/)
- [Ada tuning guide](https://docs.nvidia.com/cuda/ada-tuning-guide/)
- [Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/)
- [Blackwell tuning guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/)

### Implement

- Vector addition
- SAXPY: `y = a * x + y`
- Copy and memory-bandwidth benchmark
- Naive and coalesced matrix transpose

### Exit Criteria

You should be able to explain:

- Why adjacent threads should usually access adjacent memory
- Why more occupancy is not always better
- How a GPU hides memory latency
- Whether a measured kernel is likely memory-bound or compute-bound

## Weeks 3-5: CUDA C++ Fundamentals

### Learn

- CUDA compilation and kernel launch syntax
- Thread and block indexing
- Device memory allocation and host-device transfers
- Unified memory versus explicit memory management
- Shared memory and `__syncthreads()`
- Streams, events, asynchronous copies, and pinned host memory
- Atomics and race conditions
- Warp primitives such as `__shfl_sync`
- CUDA error checking

### Study

- [CUDA Programming Guide: CUDA Programming Model](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-programming-model)
- [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/)
- [CUDA C++ Best Practices: Memory Optimizations](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#memory-optimizations)
- [CUDA C++ Best Practices: Execution Configuration](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#execution-configuration-optimizations)
- [NVIDIA reduction tutorial](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
- [Efficient matrix transpose in CUDA C++](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)
- [Cooperative Groups introduction](https://developer.nvidia.com/blog/cooperative-groups/)
- [CUDA Samples: asynchronous API example](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/0_Introduction/simpleStreams)

### Implement

Implement and optimize each operation in multiple stages:

1. Reduction: naive, shared-memory, and warp-shuffle versions
2. Histogram: global atomics followed by block-local aggregation
3. Prefix sum
4. Tiled matrix multiplication

For each kernel:

- Validate results against a CPU or PyTorch implementation
- Run multiple warm-up and measured iterations
- Report median or percentile latency rather than one timing
- Measure effective bandwidth or achieved FLOP/s
- Compare with the previous implementation

### Exit Criteria

You should be able to write a race-free tiled kernel, reason about synchronization,
and select a defensible block size based on resource usage rather than habit.

## Weeks 6-8: Profiling and Performance Engineering

### Learn

- Application timelines and CPU-GPU synchronization
- Kernel launch overhead
- Memory-bound versus compute-bound behavior
- Warp issue efficiency and stall reasons
- Register allocation, spills, and local memory
- Occupancy limitations
- Shared-memory bank conflicts
- Instruction throughput and dependency chains
- Overlapping transfers and execution

### Tools and Study

- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/)
- [Nsight Compute documentation](https://docs.nvidia.com/nsight-compute/)
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
- [Compute Sanitizer documentation](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/)
- [CUDA Programming Guide: Performance Guidelines](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#performance-guidelines)
- [CUDA Best Practices: Performance Metrics](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#performance-metrics)

Useful commands:

```bash
compute-sanitizer --tool memcheck ./program
compute-sanitizer --tool racecheck ./program
nsys profile --stats=true ./program
ncu --set full ./program
nvcc --resource-usage kernel.cu -o kernel
```

Do not optimize from a full profiler report at random. Form a hypothesis, select the
metrics needed to test it, make one change, and measure again.

### Implement

- Numerically stable softmax
- Layer normalization
- Fused bias plus activation
- Batched reductions

Compare against PyTorch and, where applicable, cuBLAS or cuDNN. Include both latency
and numerical error in the results.

### Exit Criteria

Given a slow kernel, you should be able to use profiler evidence to identify its
dominant bottleneck and propose a testable optimization.

## Weeks 9-10: PTX

### Learn

- CUDA compilation pipeline: CUDA C++ to PTX to SASS
- PTX registers, types, predicates, and address spaces
- Load/store widths and cache operators
- Predicated execution
- Arithmetic, conversion, barrier, and warp instructions
- Inline PTX constraints and portability risks

### Study

- [PTX ISA documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [CUDA Programming Guide: Inline PTX Assembly](https://docs.nvidia.com/cuda/inline-ptx-assembly/)
- [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/)
- [CUDA compilation trajectory](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#cuda-compilation-trajectory)

Generate and retain intermediate output:

```bash
nvcc -O3 --keep kernel.cu
nvcc -O3 -lineinfo -ptx kernel.cu -o kernel.ptx
nvcc -O3 -Xptxas=-v kernel.cu -o kernel
```

### Exercises

- Map a simple CUDA kernel to its PTX
- Find global and shared-memory operations
- Identify predicated instructions produced by a branch
- Compare scalar and vectorized memory accesses
- Write one small inline PTX example, such as a special instruction or controlled
  load, and compare it with compiler-generated code

Avoid writing a substantial application directly in PTX. PTX is most useful for
inspection and narrowly targeted functionality.

### Exit Criteria

You should be able to explain how a CUDA source construct maps to PTX and recognize
when the compiler introduced conversions, scalarized an operation, or changed an
address space.

## Weeks 11-12: SASS

### Learn

- Architecture-specific machine instructions
- Register operands, predicates, and control flow
- Global, shared, and local-memory instructions
- Address calculation overhead
- Floating-point and integer pipelines
- Tensor Core instructions
- Register spills
- Why instruction names and scheduling details change by architecture

### Study

- [CUDA Binary Utilities: cuobjdump](https://docs.nvidia.com/cuda/cuda-binary-utilities/#cuobjdump)
- [CUDA Binary Utilities: nvdisasm](https://docs.nvidia.com/cuda/cuda-binary-utilities/#nvdisasm)
- [Nsight Compute Source page](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html#source-page)
- [NVIDIA developer tools documentation](https://developer.nvidia.com/tools-overview)

Inspect binaries:

```bash
cuobjdump --dump-sass ./program
cuobjdump --dump-ptx ./program
nvcc -O3 -cubin kernel.cu -o kernel.cubin
nvdisasm -g kernel.cubin
```

### Exercises

Use SASS to answer focused questions:

- Did the compiler generate vectorized memory instructions?
- Did a local-memory access reveal a register spill?
- Which instructions implement `exp`, `sqrt`, or fused multiply-add?
- Did matrix multiplication generate Tensor Core instructions?
- Did a source-level branch become predication or control flow?

### Exit Criteria

You should be able to correlate CUDA source, PTX, SASS, and profiler output without
assuming that a particular SASS mnemonic is portable to another GPU generation.

## Weeks 13-16: Tensor Cores and Transformer Kernels

### Learn

- GEMM tiling across threads, warps, and thread blocks
- Tensor Core matrix multiply-accumulate operations
- Mixed precision and accumulation precision
- Data layout, alignment, and vectorized movement
- Software pipelining and asynchronous copies
- Kernel fusion
- Online softmax and IO-aware attention

### Study

- [CUDA Programming Guide: WMMA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#warp-matrix-functions)
- [cuBLAS documentation](https://docs.nvidia.com/cuda/cublas/)
- [CUTLASS repository](https://github.com/NVIDIA/cutlass)
- [CUTLASS GEMM API](https://docs.nvidia.com/cutlass/media/docs/cpp/gemm_api.html)
- [CuTe quick start](https://docs.nvidia.com/cutlass/media/docs/cpp/cute/00_quickstart.html)
- [CUDA Mode lectures and exercises](https://github.com/cuda-mode/lectures)
- [FlashAttention paper](https://arxiv.org/abs/2205.14135)
- [FlashAttention implementation](https://github.com/Dao-AILab/flash-attention)
- [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)

Start with cuBLAS as the performance and correctness baseline. Then use CUTLASS/CuTe
to understand production-quality tiling instead of trying to reproduce an entire GEMM
framework immediately.

### Capstone

Build a small transformer inference kernel suite:

1. Fused residual addition and layer normalization
2. Optimized causal or non-causal softmax
3. Tensor Core GEMM using CUTLASS
4. Simplified fused attention using online softmax
5. End-to-end benchmark against equivalent PyTorch operations

For every operation, document:

- Supported shapes and data types
- Numerical tolerance and reference implementation
- GPU and software environment
- Latency and throughput over several realistic shapes
- Roofline classification
- Profiler evidence for the main bottleneck
- PTX or SASS evidence for important compiler decisions

### Exit Criteria

You should be able to explain the performance of a transformer-oriented kernel in
terms of data movement, arithmetic intensity, instruction selection, launch overhead,
and available parallelism.

## Weekly Working Routine

Use this allocation as a default:

| Activity | Time |
|---|---:|
| Architecture, documentation, or papers | 2 hours |
| Implementation | 4 hours |
| Profiling and experiments | 2 hours |
| PTX/SASS inspection | 1 hour |
| Written performance notes | 1 hour |

Keep each optimization as a separate commit. For each experiment, write:

1. Current result
2. Bottleneck hypothesis
3. Proposed change
4. Expected effect on specific profiler metrics
5. Measured result
6. Explanation of agreement or disagreement with the hypothesis

## Suggested Repository Structure

```text
gpu-learning/
├── common/                  # Error checking, timing, result validation
├── 01-vector-add/
├── 02-memory-access/
├── 03-reduction/
├── 04-histogram/
├── 05-scan/
├── 06-matmul/
├── 07-softmax/
├── 08-layernorm/
├── 09-ptx/
├── 10-sass/
├── 11-cutlass-gemm/
├── 12-attention/
└── reports/                 # Benchmark and profiler notes
```

## Progress Checkpoints

### After 2 Weeks

Explain coalescing, divergence, latency hiding, occupancy, and the memory hierarchy.
Measure the bandwidth difference between naive and coalesced access.

### After 6 Weeks

Implement and explain a reduction and tiled GEMM. Demonstrate correctness and show why
each optimization changes performance.

### After 10 Weeks

Profile softmax or layer normalization, identify the primary bottleneck, and correlate
the CUDA source with generated PTX.

### After 12 Weeks

Use SASS to verify an instruction-selection or register-spill hypothesis.

### After 16 Weeks

Present the capstone as a short engineering report with reproducible benchmarks,
correctness tests, profiler evidence, and comparison against optimized libraries.

## Topics to Defer

These are valuable, but postpone them until the core roadmap is complete:

- Multi-GPU programming, NCCL, and topology-aware communication
- CUDA Graphs
- Dynamic parallelism
- Thread-block clusters and distributed shared memory
- Architecture-specific PTX for advanced Tensor Core pipelines
- Writing substantial kernels directly in PTX or attempting to hand-author SASS

The first milestone is not memorizing instruction names. It is being able to implement
a reduction and tiled GEMM, measure them correctly, and explain their performance from
the source code down to the hardware.
