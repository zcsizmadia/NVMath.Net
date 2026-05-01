# NVMath.Net

A .NET wrapper library for NVIDIA CUDA math libraries — cuFFT, cuBLASLt, cuSPARSE, and cuTENSOR — modelled after Python's [nvmath-python](https://github.com/NVIDIA/nvmath-python).

---

## Prerequisites

| Requirement | Version |
|---|---|
| .NET | 8.0, 9.0, or 10.0 |
| CUDA Toolkit | 12.x (cuBLASLt 12, cuSPARSE 12, CUDA Runtime 12) |
| cuFFT | 11.x (`cufft64_11.dll` / `libcufft.so.11`) |
| cuTENSOR | 2.x |
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
await using var stream = new CudaStream();
using var input  = new DeviceBuffer<System.Numerics.Complex>(512);
using var output = new DeviceBuffer<System.Numerics.Complex>(512);

// Copy host data to device
input.CopyFrom(hostData);

// Plan and execute
var plan = await Fft.FftAsync(input, new long[] { 512 }, stream: stream);
await plan.SynchronizeAsync();
```

### Matrix Multiplication (cuBLASLt)

```csharp
using NVMathNet;
using NVMathNet.LinAlg;

// C = alpha * A * B + beta * C   (M=4, N=4, K=4, batch=1)
await using var stream = new CudaStream();
using var matmul = new Matmul(m: 4, n: 4, k: 4);
await matmul.PlanAsync(stream);

using var a = new DeviceBuffer<float>(16);
using var b = new DeviceBuffer<float>(16);
using var c = new DeviceBuffer<float>(16);
// ... fill a and b ...
await matmul.ExecuteAsync<float>(a, b, c, c, alpha: 1f, beta: 0f);
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
    extC, modeD: new int[] { 'm', 'n' });

await tc.PlanAsync();

using var a = new DeviceBuffer<float>(16);
using var c = new DeviceBuffer<float>(16);
// ...
await tc.ExecuteAsync<float>(a, a, c, c, alpha: 1f, beta: 0f);
```

---

## Building from Source

```bash
git clone https://github.com/nvmath/NVMath.Net.git
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
  NVMath.Net.Interop/   # Raw P/Invoke bindings (cuFFT, cuBLASLt, cuSPARSE, cuTENSOR)
  NVMath.Net/           # High-level .NET API
    Fft/                # cuFFT wrapper
    LinAlg/             # cuBLASLt matrix multiplication
    Sparse/             # cuSPARSE sparse linear algebra
    Tensor/             # cuTENSOR contractions
tests/
  NVMath.Net.Tests/     # TUnit test suite
samples/
  FftSample/            # FFT examples
  MatmulSample/         # Matrix multiplication examples
  SparseSample/         # Sparse linear algebra examples
  TensorSample/         # Tensor contraction examples
```

---

## Design Notes

Native libraries are loaded at runtime via `System.Runtime.InteropServices.NativeLibrary` + delegate-based dispatch instead of `[DllImport]` attributes. This allows loading differently named platform-specific binaries (`cufft64_11.dll` on Windows, `libcufft.so.11` on Linux) from a single code path without per-platform conditional compilation.

---

## License

[MIT](LICENSE)
