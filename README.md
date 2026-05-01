# NVMath.Net

A .NET wrapper library for NVIDIA CUDA math libraries — cuFFT, cuBLASLt, cuBLAS, cuSPARSE, cuTENSOR, cuSOLVER, and cuRAND — modelled after Python's [nvmath-python](https://github.com/NVIDIA/nvmath-python).

---

## Prerequisites

| Requirement | Version |
|---|---|
| .NET | 8.0, 9.0, or 10.0 |
| CUDA Toolkit | 13.x (CUDA Runtime 13, cuBLAS 13, cuBLASLt 13) |
| cuFFT | 12.x (`cufft64_12.dll` / `libcufft.so.12`) |
| cuSPARSE | 12.x (`cusparse64_12.dll` / `libcusparse.so.12`) |
| cuTENSOR | 2.x (`cutensor.dll` / `libcutensor.so.2`) |
| cuSOLVER | 12.x (`cusolver64_12.dll` / `libcusolver.so.12`) |
| cuRAND | 10.x (`curand64_10.dll` / `libcurand.so.10`) |
| NVIDIA GPU | Compute capability 7.0+ recommended |

Make sure the native library directories are on your system's `PATH` (Windows) or `LD_LIBRARY_PATH` (Linux) before running.

---

## Installation

Add a reference to the library projects or the NuGet package (once published):

```xml
<PackageReference Include="NVMath.Net" Version="1.0.0" />
```

---

## Quick Start

### FFT

```csharp
using NVMathNet;
using NVMathNet.Fft;

// 1-D complex-to-complex FFT
using var input = new DeviceBuffer<System.Numerics.Complex>(512);
input.CopyFrom(hostData);

// One-shot static helper: allocates plan, executes, returns output buffer
using var output = await Fft.FftAsync(input, new long[] { 512 });
var result = output.ToArray();
```

### Matrix Multiplication (cuBLASLt)

```csharp
using NVMathNet;
using NVMathNet.LinAlg;

// D = alpha * A * B + beta * C   (M=4, N=4, K=4)
// All matrices use column-major layout
using var matmul = new Matmul(m: 4, n: 4, k: 4);
await matmul.PlanAsync();

using var a = new DeviceBuffer<float>(16);
using var b = new DeviceBuffer<float>(16);
using var c = new DeviceBuffer<float>(16);
using var d = new DeviceBuffer<float>(16);
// ... fill a and b (column-major) ...
await matmul.ExecuteAsync<float>(a, b, c, d, alpha: 1f, beta: 0f);
```

### Sparse Matrix-Vector Multiply (cuSPARSE)

```csharp
using NVMathNet;
using NVMathNet.Sparse;

// Build a 3×3 CSR identity matrix
var rowOffsets = new DeviceBuffer<int>(4);   // rows+1
var colIndices = new DeviceBuffer<int>(3);   // nnz
var values     = new DeviceBuffer<float>(3); // nnz
// ... fill buffers ...

var matrix = new CsrMatrix<float>(3, 3, rowOffsets, colIndices, values);

using var x = new DeviceBuffer<float>(3);
using var y = new DeviceBuffer<float>(3);
await SparseLinearAlgebra.SpMVAsync<float>(matrix, x, y, alpha: 1f, beta: 0f);
```

### Tensor Contraction (cuTENSOR)

```csharp
using NVMathNet;
using NVMathNet.Tensor;

// Contract A[m,k] * B[k,n] -> C[m,n]  (like a matrix multiply)
var extA = new long[] { 4, 4 };
var extB = new long[] { 4, 4 };
var extC = new long[] { 4, 4 };

using var tc = new TensorContraction(
    extA, modeA: new int[] { 'm', 'k' },
    extB, modeB: new int[] { 'k', 'n' },
    extC, modeC: new int[] { 'm', 'n' },
    extC, modeD: new int[] { 'm', 'n' },
    TensorDataType.Float32);

await tc.PlanAsync();

using var a = new DeviceBuffer<float>(16);
using var c = new DeviceBuffer<float>(16);
// ...
await tc.ExecuteAsync<float>(a, a, c, c, alpha: 1.0, beta: 0.0);
```

### Dense Solver (cuSOLVER)

```csharp
using NVMathNet;
using NVMathNet.Solver;

// Solve A*x = b using LU factorisation (column-major)
using var dA = new DeviceBuffer<float>(n * n);
using var dB = new DeviceBuffer<float>(n);
dA.CopyFrom(hostA);
dB.CopyFrom(hostB);

await DenseSolver.SolveAsync(dA, dB, n);
float[] solution = dB.ToArray(); // b is overwritten with x

// Eigenvalue decomposition of symmetric matrix
using var dW = new DeviceBuffer<float>(n); // eigenvalues
await DenseSolver.EigAsync(dA, dW, n, computeVectors: true);
```

### Random Number Generation (cuRAND)

```csharp
using NVMathNet;
using NVMathNet.Rand;

using var rng = new CudaRandom(seed: 42);
using var dBuf = new DeviceBuffer<float>(10_000);

// Fill with uniform (0, 1]
await rng.FillUniformAsync(dBuf);

// Fill with normal distribution (mean=0, stddev=1)
await rng.FillNormalAsync(dBuf, mean: 0f, stddev: 1f);
```

### Structured Matrix Operations (cuBLAS)

```csharp
using NVMathNet;
using NVMathNet.LinAlg;

// Triangular solve: solve A*X = B where A is lower-triangular
await TriangularSolve.StrsmAsync(dA, dB, m: 4, n: 1, upper: false);

// Symmetric multiply: C = alpha * A * B + beta * C where A is symmetric
await SymmetricMultiply.SsymmAsync(dA, dB, dC, m: 4, n: 4);
```

### Multi-GPU

```csharp
using NVMathNet;

// Create a context managing all available GPUs
using var ctx = new MultiGpuContext();

// Run work concurrently — one thread per GPU
await ctx.ForEachDeviceAsync(async (index, deviceId, stream) =>
{
    using var buf = new DeviceBuffer<float>(1024);
    // ... do GPU work on device 'deviceId' ...
    await stream.SynchronizeAsync();
});

ctx.SynchronizeAll();
```

---

## Building from Source

```bash
git clone https://github.com/zcsizmadia/NVMath.Net.git
cd NVMath.Net
dotnet build
```

Run the tests (requires a CUDA-capable GPU at runtime):

```bash
dotnet test
```

---

## Project Structure

```
src/
  NVMathNet/              # High-level .NET API
    Interop/              # Raw P/Invoke bindings (cuFFT, cuBLASLt, cuSPARSE, cuTENSOR, cuSOLVER, cuRAND)
    Fft/                  # cuFFT wrapper
    LinAlg/               # cuBLASLt matrix multiplication, GEMM, TRSM, SYMM
    Solver/               # cuSOLVER dense solver (LU, Cholesky, SVD, eigenvalues)
    Rand/                 # cuRAND random number generation
    Sparse/               # cuSPARSE sparse linear algebra (CSR, COO, BSR)
    Tensor/               # cuTENSOR v2 contractions + element-wise operations
tests/
  NVMathNet.Tests/        # TUnit test suite
samples/
  ChainingSample/         # Multi-stream chaining patterns
  FftSample/              # FFT examples
  MatmulSample/           # Matrix multiplication examples
  SparseSample/           # Sparse linear algebra examples
  TensorSample/           # Tensor contraction examples
docs/
  API.md                  # Full API reference
```

---

## Design Notes

Native libraries are loaded at runtime via `System.Runtime.InteropServices.NativeLibrary` + delegate-based dispatch instead of `[DllImport]` attributes. This allows loading differently named platform-specific binaries (`cufft64_12.dll` on Windows, `libcufft.so.12` on Linux) from a single code path without per-platform conditional compilation.

---

## License

[MIT](LICENSE)
