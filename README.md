# NVMath.Net

[![Build](https://github.com/zcsizmadia/NVMath.Net/actions/workflows/build.yml/badge.svg)](https://github.com/zcsizmadia/NVMath.Net/actions/workflows/build.yml)
[![NuGet](https://img.shields.io/nuget/v/NVMath.Net.svg)](https://www.nuget.org/packages/NVMath.Net)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/zcsizmadia/NVMath.Net/blob/main/LICENSE)
[![Target frameworks](https://img.shields.io/badge/.NET-8.0%20%7C%209.0%20%7C%2010.0-512BD4?logo=dotnet)](https://dotnet.microsoft.com/download)

**NVIDIA's CUDA math libraries for .NET.** NVMath.Net is a C# counterpart to
[nvmath-python](https://github.com/NVIDIA/nvmath-python): idiomatic, `async`-friendly wrappers over **cuFFT, cuBLASLt,
cuBLAS, cuSPARSE, cuDSS, cuTENSOR, cuSOLVER and cuRAND**, with the same *plan → execute* workflow, explicit resource
ownership and no native build step (the NVIDIA libraries are loaded at run time).

📖 **Documentation:** <https://zcsizmadia.github.io/NVMath.Net/> — guides, per-module references and the generated API reference.

---

## Features

| Area | Wraps | Highlights |
|---|---|---|
| **FFT** | cuFFT | C2C / R2C / C2R, N-D and batched plans, leading- or trailing-axis layouts, output normalization (`None`, `Backward`, `Forward`, `Orthonormal`) |
| **Matrix multiplication** | cuBLASLt, cuBLAS | `Matmul` with epilogs (bias, ReLU, GELU, aux outputs), batching, FP8 scale pointers, algorithm autotuning with a plan cache, conjugate transpose and complex matmul; GEMM, TRSM, SYMM, DGMM |
| **Sparse** | cuSPARSE | CSR, COO, BSR and CSC matrices (real and complex); SpMV, SpMM, SpSV, SpSM, SDDMM, SpGEMM, CSR↔CSC |
| **Sparse direct solver** | cuDSS | `SparseDirectSolver<T>`: analyse / factorise / refactorise / solve; LU, LDLᵀ, Cholesky; real and complex (Hermitian) |
| **Tensors** | cuTENSOR 2 | Contractions, element-wise unary / binary / trinary operations, permutation, reduction |
| **Dense solvers** | cuSOLVER, cuBLAS | LU, QR, least squares, Cholesky, SVD, eigenvalues; batched Cholesky / LU / Jacobi SVD; complex (double) variants including SVD |
| **Random numbers** | cuRAND | Pseudo- and quasi-random generators, uniform / normal / log-normal / Poisson, integers, skip-ahead |
| **Infrastructure** | CUDA runtime | `DeviceBuffer<T>`, streams, events, multi-GPU helper, pluggable device memory managers, dependency-free logging hook |

---

## Prerequisites

| Requirement | Version |
|---|---|
| .NET | 8.0, 9.0 or 10.0 |
| NVIDIA GPU | Compute capability 7.0+ recommended |
| CUDA Toolkit | 13.x (runtime, cuBLAS, cuBLASLt, cuFFT 12, cuSPARSE 12, cuSOLVER 12, cuRAND 10) |
| cuTENSOR (tensor module) | 2.x (`cutensor.dll` / `libcutensor.so.2`) |
| cuDSS (`SparseDirectSolver`) | 0.x (`cudss64_0.dll` / `libcudss.so.0`) |

Libraries are loaded **lazily on first use**, so you only need the ones for the modules you call. On Windows the CUDA
Toolkit, cuTENSOR and cuDSS default install folders are searched automatically; otherwise put the folder containing the
libraries on `PATH` (Windows) or `LD_LIBRARY_PATH` (Linux). Details:
[Installing the native libraries](https://zcsizmadia.github.io/NVMath.Net/articles/native-libraries.html).

---

## Installation

```bash
dotnet add package NVMath.Net
```

or add a project reference to `src/NVMathNet/NVMathNet.csproj`.

---

## Quick start

Matrices and tensors are **column-major** (the cuBLAS convention); FFT shapes are **row-major** (the last axis is contiguous). Always dispose GPU resources (`using`).

### FFT

```csharp
using NVMathNet;
using NVMathNet.Fft;

using var input = new DeviceBuffer<System.Numerics.Complex>(512);
input.CopyFrom(hostData);

// One-shot helper: plans, executes, returns the output buffer
using var output = await Fft.FftAsync(input, new long[] { 512 });
var spectrum = output.ToArray();
```

For repeated transforms use the plan-based `Fft` class, which also supports N-D shapes, batching, `axes`, and
`FftOptions.Normalization`.

### Matrix multiplication (cuBLASLt)

```csharp
using NVMathNet;
using NVMathNet.LinAlg;

// D = alpha * A * B + beta * C   (M = N = K = 4)
using var matmul = new Matmul(m: 4, n: 4, k: 4, options: new MatmulOptions { Blocking = true });
await matmul.PlanAsync();

using var a = new DeviceBuffer<float>(16);
using var b = new DeviceBuffer<float>(16);
using var c = new DeviceBuffer<float>(16);
using var d = new DeviceBuffer<float>(16);
// ... fill a and b (column-major), c.Clear() ...
await matmul.ExecuteAsync<float>(a, b, c, d, alpha: 1.0, beta: 0.0);
```

### Sparse matrices and a sparse direct solver

```csharp
using NVMathNet;
using NVMathNet.Sparse;

// 3x3 identity in CSR
var rowOffsets = new DeviceBuffer<int>(4);
var colIndices = new DeviceBuffer<int>(3);
var values     = new DeviceBuffer<float>(3);
rowOffsets.CopyFrom(new[] { 0, 1, 2, 3 });
colIndices.CopyFrom(new[] { 0, 1, 2 });
values.CopyFrom(new[] { 1f, 1f, 1f });
using var matrix = new CsrMatrix<float>(3, 3, rowOffsets, colIndices, values);   // owns the buffers

using var x = new DeviceBuffer<float>(3);
using var y = new DeviceBuffer<float>(3);
x.CopyFrom(new[] { 4f, 5f, 6f });
await SparseLinearAlgebra.SpMVAsync(matrix, x, y);            // y = A x

// Solve A z = x with cuDSS
using var z = new DeviceBuffer<float>(3);
using var solver = new SparseDirectSolver<float>(matrix);
solver.Solve(x, z);
```

### Tensors (cuTENSOR)

```csharp
using NVMathNet;
using NVMathNet.Tensor;

// C[m,n] = A[m,k] * B[k,n]   (mode labels: m=0, n=1, k=2)
using var contraction = new TensorContraction(
    [4, 4], [0, 2],     // A
    [4, 4], [2, 1],     // B
    [4, 4], [0, 1],     // C
    [4, 4], [0, 1],     // D
    TensorDataType.Float32,
    new TensorContractionOptions { Blocking = true });
await contraction.PlanAsync();
await contraction.ExecuteAsync(a.PointerAsInt, b.PointerAsInt, c.PointerAsInt, d.PointerAsInt, alpha: 1.0, beta: 0.0);

// Row sums of a 3x4 tensor: D[i] = sum_j A[i,j]
using var reduce = new TensorReduction([3, 4], [0, 1], [3], [0]);
await reduce.ExecuteAsync(a3x4, rowSums);
```

### Dense solvers (cuSOLVER)

```csharp
using NVMathNet;
using NVMathNet.Solver;

await DenseSolver.SolveAsync(dA, dB, n);                          // LU solve; B is overwritten with X
await DenseSolver.LeastSquaresAsync(dA, dB, m, n);                // min ||A x - b||
await DenseSolver.EigAsync(dA, dW, n, computeVectors: true);      // symmetric eigenproblem
```

### Random numbers (cuRAND)

```csharp
using NVMathNet;
using NVMathNet.Rand;

using var rng = new CudaRandom(seed: 42);
using var buf = new DeviceBuffer<float>(10_000);
await rng.FillUniformAsync(buf);                                  // (0, 1]
await rng.FillNormalAsync(buf, mean: 0f, stddev: 1f);
```

### Multiple GPUs

```csharp
using NVMathNet;

using var ctx = new MultiGpuContext();                            // all visible devices
await ctx.ForEachDeviceAsync(async (index, deviceId, stream) =>
{
    using var buf = new DeviceBuffer<float>(1024);
    // ... GPU work on device `deviceId` ...
    await stream.SynchronizeAsync();
});
```

---

## Documentation

- **Website:** <https://zcsizmadia.github.io/NVMath.Net/>
- **Guides:** [Getting started](https://zcsizmadia.github.io/NVMath.Net/articles/getting-started.html),
  [Core concepts](https://zcsizmadia.github.io/NVMath.Net/articles/concepts.html),
  [Coming from nvmath-python](https://zcsizmadia.github.io/NVMath.Net/articles/nvmath-python.html),
  [Troubleshooting](https://zcsizmadia.github.io/NVMath.Net/articles/troubleshooting.html)
- **Module references** (also in the repository under [`docs/api`](https://github.com/zcsizmadia/NVMath.Net/tree/main/docs/api)):
  [Core](https://github.com/zcsizmadia/NVMath.Net/blob/main/docs/api/Core.md) ·
  [FFT](https://github.com/zcsizmadia/NVMath.Net/blob/main/docs/api/Fft.md) ·
  [Linear algebra](https://github.com/zcsizmadia/NVMath.Net/blob/main/docs/api/Linalg.md) ·
  [Sparse](https://github.com/zcsizmadia/NVMath.Net/blob/main/docs/api/Sparse.md) ·
  [Tensor](https://github.com/zcsizmadia/NVMath.Net/blob/main/docs/api/Tensor.md) ·
  [Solver](https://github.com/zcsizmadia/NVMath.Net/blob/main/docs/api/Solver.md) ·
  [Random](https://github.com/zcsizmadia/NVMath.Net/blob/main/docs/api/Rand.md)
- **Samples:** 15 runnable projects in [`samples/`](https://github.com/zcsizmadia/NVMath.Net/tree/main/samples), indexed in the [samples guide](https://zcsizmadia.github.io/NVMath.Net/articles/samples.html); they verify their own results.

To build the documentation site locally:

```bash
dotnet tool restore
dotnet docfx docs/docfx.json --serve
```

---

## Building from source

```bash
git clone https://github.com/zcsizmadia/NVMath.Net.git
cd NVMath.Net
dotnet build
```

The test suite needs a CUDA-capable GPU and the native libraries (cuTENSOR and cuDSS tests need those libraries too):

```bash
dotnet test
```

The hosted CI has no GPU: it builds the library, tests and samples and packs the NuGet package, but does not run the
tests. Run them locally before sending a change.

Local builds target `net10.0` only; with `CI=true` the library and tests target `net8.0;net9.0;net10.0`.

---

## Project structure

```
src/NVMathNet/
  Interop/      Raw bindings for each native library (public *Native classes) and exception types
  Fft/          cuFFT wrapper
  LinAlg/       cuBLASLt Matmul, GEMM, TRSM, SYMM, DGMM
  Sparse/       CSR/COO/BSR/CSC matrices, cuSPARSE operations, cuDSS direct solver
  Tensor/       cuTENSOR contraction, element-wise, permutation, reduction
  Solver/       cuSOLVER dense solvers (single, batched, complex)
  Rand/         cuRAND generators
  Memory/       Pluggable device memory managers
  *.cs          DeviceBuffer, PinnedBuffer, CudaStream, CudaEvent, CudaDevice, MultiGpuContext
tests/NVMathNet.Tests/   TUnit test suite (requires a GPU)
samples/                 One runnable project per feature area
docs/                    Documentation site (DocFX) and per-module guides
```

---

## Design notes

Native libraries are loaded at run time through `System.Runtime.InteropServices.NativeLibrary` and function pointers
instead of `[DllImport]`. A single code path loads the differently named binaries on each platform
(`cufft64_12.dll` on Windows, `libcufft.so.12` on Linux) and defers loading until a module is first used.
Function signatures and enum values are checked against the NVIDIA headers.

---

## Contributing

Issues and pull requests are welcome. Public members need XML documentation (missing docs are a build error), and new
functionality needs a test, a section in the matching `docs/api` guide, and where appropriate a sample.

## License

[MIT](https://github.com/zcsizmadia/NVMath.Net/blob/main/LICENSE)
