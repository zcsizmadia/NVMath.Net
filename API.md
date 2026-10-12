# NVMath.Net API Overview

The complete, searchable documentation lives on the website: **<https://zcsizmadia.github.io/NVMath.Net/>**
(guides, per-module references and the generated API reference). This page is a short map of the repository.

## Namespaces and module guides

| Namespace | Purpose | Guide |
|---|---|---|
| `NVMathNet` | Devices, streams, events, `DeviceBuffer<T>`, `PinnedBuffer<T>`, multi-GPU | [Core](docs/api/Core.md) |
| `NVMathNet.Memory` | Pluggable device memory managers | [Core](docs/api/Core.md) |
| `NVMathNet.Fft` | cuFFT | [FFT](docs/api/Fft.md) |
| `NVMathNet.LinAlg` | cuBLASLt `Matmul`, GEMM, TRSM, SYMM, DGMM | [Linear algebra](docs/api/Linalg.md) |
| `NVMathNet.Sparse` | cuSPARSE matrices and operations, cuDSS `SparseDirectSolver` | [Sparse](docs/api/Sparse.md) |
| `NVMathNet.Tensor` | cuTENSOR contraction, element-wise, permutation, reduction | [Tensor](docs/api/Tensor.md) |
| `NVMathNet.Solver` | cuSOLVER dense solvers (single, batched, complex) | [Solver](docs/api/Solver.md) |
| `NVMathNet.Rand` | cuRAND generators and distributions | [Random](docs/api/Rand.md) |
| `NVMathNet.Interop` | Raw native bindings and exception types | [Interop](docs/api/Interop.md) |

## nvmath-python comparison

NVMath.Net covers the host (single-process) API of [nvmath-python](https://github.com/NVIDIA/nvmath-python):

| nvmath-python | NVMath.Net | Status |
|---|---|---|
| `nvmath.fft` — `fft`, `ifft`, `rfft`, `irfft`, `FFT` | `Fft` (plan-based and static helpers) | Covered: C2C/R2C/C2R, N-D, batched, axes, normalization. **Not covered:** prolog/epilog callbacks |
| `nvmath.linalg.advanced.Matmul` | `Matmul` | Covered: epilogs, bias, aux outputs, batching, FP8 scale pointers, autotune, plan cache, conjugate transpose, complex matmul. **Not covered:** block-scaled MXFP8/NVFP4 (needs Blackwell hardware to verify) |
| `nvmath.linalg` BLAS-style routines | `Gemm`, `TriangularSolve`, `SymmetricMultiply`, `DiagonalMultiply` | Covered (typed per precision) |
| `nvmath.sparse` multiplication | `SparseLinearAlgebra` | Covered for CSR/COO/BSR/CSC where cuSPARSE supports it |
| `nvmath.sparse.advanced.DirectSolver` | `SparseDirectSolver<T>` (cuDSS) | Covered for float/double/complex; no batched or multi-GPU solves |
| `nvmath.tensor` contraction and element-wise operations | `TensorContraction`, `ElementwiseBinary`, `ElementwiseTrinary`, `TensorPermutation`, `TensorReduction` | Covered |
| `nvmath.bindings` | `NVMathNet.Interop.*Native` | Raw bindings are public |

**Beyond nvmath-python:** `DenseSolver` (cuSOLVER), `CudaRandom` (cuRAND), `MultiGpuContext`, pluggable memory managers.

**Out of scope for .NET:** the device-side APIs (`nvmath.device`: cuBLASDx, cuFFTDx, cuSOLVERDx, device cuRAND) and
the distributed APIs (`nvmath.distributed`: NVSHMEM, cuFFTMp), as well as Python-specific features (UST DSL,
NumPy/CuPy/PyTorch interop, CPU execution through NVPL).

Deferred work is tracked under the [`post-1.0`](https://github.com/zcsizmadia/NVMath.Net/labels/post-1.0) label.
