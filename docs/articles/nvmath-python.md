# Coming from nvmath-python

NVMath.Net follows the structure of [nvmath-python](https://github.com/NVIDIA/nvmath-python): stateful objects with
`plan` and `execute` steps, option objects, and thin stateless helpers. This page maps the concepts. It describes
NVMath.Net as it is today; where nvmath-python has something this library does not, that is stated.

## Concept mapping

| nvmath-python | NVMath.Net |
|---|---|
| NumPy / CuPy / PyTorch tensors | `DeviceBuffer<T>` (explicit GPU memory); shapes are passed separately |
| `obj.plan()` | `Plan()` / `PlanAsync()` |
| `obj.execute()` | `Execute()` / `ExecuteAsync()` |
| `obj.free()` / context manager | `Dispose()` / `using` |
| `options={…}` dict or dataclass | `MatmulOptions`, `FftOptions`, `TensorContractionOptions`, `DirectSolverOptions` |
| `stream=` argument | optional `CudaStream` constructor / method argument |
| `blocking="auto"` | `Blocking` option (default: non-blocking, except where noted) |
| Row-major arrays (NumPy default) | **Column-major** buffers; tensors treat the first mode as fastest |

## Module mapping

| nvmath-python | NVMath.Net | Notes |
|---|---|---|
| `nvmath.fft.fft`, `ifft`, `rfft`, `irfft` | `Fft.FftAsync`, `IFftAsync`, `RFftAsync`, `IRFftAsync` | One-shot helpers |
| `nvmath.fft.FFT` | `Fft` class | N-D and batched plans; `FftOptions.Normalization` replaces norm scaling. Prolog/epilog callbacks are not available yet |
| `nvmath.linalg.advanced.matmul` / `Matmul` | `Matmul` | Epilogs (`MatmulEpilog`), bias, aux outputs, batching, FP8 scale pointers, `Autotune` |
| `nvmath.linalg` BLAS-style helpers | `Gemm`, `TriangularSolve`, `SymmetricMultiply`, `DiagonalMultiply` | Typed per precision (`Sgemm`, `Dgemm`, …) |
| `nvmath.sparse.advanced.DirectSolver` | `SparseDirectSolver<T>` | cuDSS; analyse / factorise / refactorise / solve |
| `nvmath.sparse` multiplication | `SparseLinearAlgebra` (`SpMVAsync`, `SpMMAsync`, `SpGemmAsync`, …) | CSR, COO, BSR, CSC |
| `nvmath.tensor.contract` / contraction objects | `TensorContraction` | cuTENSOR 2.x plans |
| `nvmath.tensor` element-wise ops | `ElementwiseBinary`, `ElementwiseTrinary` | Operator enum `CuTensorOperator` |
| `nvmath.tensor` permutation / reduction | `TensorPermutation`, `TensorReduction` | |
| `nvmath.bindings.*` | `NVMathNet.Interop.*Native` classes | Raw bindings, public |
| `nvmath.device.*` (cuBLASDx, cuFFTDx, …) | — | Device-side APIs need a device compiler; not offered |
| `nvmath.distributed.*` | — | Multi-process / NVSHMEM support is not offered. `MultiGpuContext` covers single-process multi-GPU |

## Beyond nvmath-python

NVMath.Net also wraps libraries that nvmath-python exposes only through bindings or not at all: dense solvers
(`DenseSolver`: LU, QR, least squares, Cholesky, SVD, eigenvalues, batched and complex variants) and random number
generation (`CudaRandom`).

## A side-by-side example

nvmath-python:

```python
import cupy as cp, nvmath
a = cp.random.rand(128, 64, dtype=cp.float32)
b = cp.random.rand(64, 32, dtype=cp.float32)
d = nvmath.linalg.advanced.matmul(a, b)
```

NVMath.Net:

```csharp
using NVMathNet;
using NVMathNet.LinAlg;
using NVMathNet.Rand;

const int M = 128, K = 64, N = 32;
using var a = new DeviceBuffer<float>(M * K);
using var b = new DeviceBuffer<float>(K * N);
using var c = new DeviceBuffer<float>(M * N);
using var d = new DeviceBuffer<float>(M * N);

using var rng = new CudaRandom(seed: 1);
rng.FillUniform(a);
rng.FillUniform(b);
c.Clear();

using var matmul = new Matmul(M, N, K, options: new MatmulOptions { Blocking = true });
await matmul.PlanAsync();
await matmul.ExecuteAsync<float>(a, b, c, d);
```
