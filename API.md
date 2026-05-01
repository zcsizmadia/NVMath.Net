# NVMath.Net API Reference

## Namespaces

| Namespace | Purpose |
|---|---|
| [`NVMathNet`](docs/api/Core.md) | Core CUDA infrastructure (device, stream, event, buffers, multi-GPU) |
| [`NVMathNet.Fft`](docs/api/Fft.md) | cuFFT wrapper |
| [`NVMathNet.LinAlg`](docs/api/Linalg.md) | cuBLASLt matrix multiplication, GEMM, structured matrices |
| [`NVMathNet.Solver`](docs/api/Solver.md) | cuSOLVER dense solver (LU, Cholesky, SVD, eigenvalues) |
| [`NVMathNet.Rand`](docs/api/Rand.md) | cuRAND random number generation |
| [`NVMathNet.Sparse`](docs/api/Sparse.md) | cuSPARSE sparse linear algebra |
| [`NVMathNet.Tensor`](docs/api/Tensor.md) | cuTENSOR v2 tensor contractions and element-wise operations |
| [`NVMathNet.Interop`](docs/api/Interop.md) | Raw P/Invoke bindings (internal use) |

## Quick Links

- [Core Infrastructure](docs/api/core.md) — `CudaDevice`, `CudaStream`, `CudaEvent`, `DeviceBuffer<T>`, `PinnedBuffer<T>`
- [FFT](docs/api/fft.md) — `Fft`, `FftOptions`, static helpers (`FftAsync`, `IFftAsync`, `RFftAsync`, `IRFftAsync`)
- [Linear Algebra](docs/api/linalg.md) — `Matmul`, `Gemm`, `TriangularSolve`, `SymmetricMultiply`, `DiagonalMultiply`
- [Dense Solver](docs/api/solver.md) — `DenseSolver` (LU, Cholesky, SVD, Eig)
- [Random](docs/api/rand.md) — `CudaRandom`, `CuRandRngType`
- [Sparse](docs/api/sparse.md) — `CsrMatrix<T>`, `CooMatrix<T>`, `BsrMatrix<T>`, `SparseLinearAlgebra`
- [Tensor](docs/api/tensor.md) — `TensorContraction`, `ElementwiseTrinary`
- [Multi-GPU](docs/api/core.md#multigpucontext) — `MultiGpuContext`
- [Interop / Exceptions](docs/api/interop.md) — Raw native bindings and exception types

## nvmath-python Parity

NVMath.Net covers the nvmath-python HOST API (non-distributed) as follows:

| nvmath-python Module | NVMath.Net Equivalent | Notes |
|---|---|---|
| `nvmath.fft` (fft/ifft/rfft/irfft) | `Fft` class + static factories | Full parity |
| `nvmath.linalg` (GeneralMatrixQualifier) | `Gemm` (Sgemm/Dgemm/Hgemm) | Full parity |
| `nvmath.linalg` (TriangularMatrixQualifier) | `TriangularSolve` | Full parity |
| `nvmath.linalg` (SymmetricMatrixQualifier) | `SymmetricMultiply` | Full parity |
| `nvmath.linalg` (DiagonalMatrixQualifier) | `DiagonalMultiply` | Full parity |
| `nvmath.linalg.advanced` (Matmul) | `Matmul` with epilog/compute type | Full parity |
| `nvmath.sparse` (matmul/SpMV/SpMM) | `SparseLinearAlgebra` | Full parity (incl. transpose) |
| `nvmath.tensor` (binary/ternary contraction) | `TensorContraction` / `ElementwiseTrinary` | Full parity |

**NVMath.Net extras** (beyond nvmath-python HOST API): `DenseSolver` (cuSOLVER), `CudaRandom` (cuRAND), `MultiGpuContext`.

**Not applicable in .NET**: FFT LTO-IR callbacks, sparse matmul callbacks (require CUDA compiler), cuDSS (separate library), UST DSL (Python-specific), CPU execution (NVPL).
