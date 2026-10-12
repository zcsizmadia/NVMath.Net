# Samples

Runnable examples, one project per feature area. Each sample verifies its own results and exits with a non-zero code on
failure, so they double as smoke tests. They need a CUDA-capable GPU and the native libraries for the modules they use
(see [Installing the native libraries](https://zcsizmadia.github.io/NVMath.Net/articles/native-libraries.html)).
The tensor and cuDSS parts skip themselves with a message when cuTENSOR or cuDSS is not installed.

```bash
dotnet run --project samples/FftAdvancedSample
```

| Sample | Shows |
|---|---|
| [`FftSample`](FftSample) | Basic 1-D FFT and the one-shot helpers |
| [`FftAdvancedSample`](FftAdvancedSample) | N-D and batched transforms, leading-axis layouts, real-to-complex round trips, output normalization |
| [`MatmulSample`](MatmulSample) | A first cuBLASLt matrix product |
| [`MatmulAdvancedSample`](MatmulAdvancedSample) | Bias epilog, strided-batch matmul, autotuning with the plan cache, complex matmul with conjugate transpose |
| [`ChainingSample`](ChainingSample) | Chaining operations on one stream, multi-stream patterns |
| [`SparseSample`](SparseSample) | SpMV and SpMM on a CSR matrix |
| [`SparseAdvancedSample`](SparseAdvancedSample) | Triangular solves, SpGEMM, CSR/CSC/COO and transpose |
| [`DirectSolverSample`](DirectSolverSample) | cuDSS sparse direct solver: analyse / factorise / solve |
| [`TensorSample`](TensorSample) | A cuTENSOR contraction |
| [`TensorOpsSample`](TensorOpsSample) | Permutation, reduction, element-wise binary and contraction with mode labels |
| [`SolverSample`](SolverSample) | LU, Cholesky, SVD and eigenvalues |
| [`LeastSquaresSample`](LeastSquaresSample) | QR, polynomial least-squares fitting, factor-once/solve-many LU, batched LU and SVD |
| [`ComplexSample`](ComplexSample) | Complex dense solvers, SVD, Hermitian eigenproblem, complex sparse products, Hermitian cuDSS solve |
| [`RandSample`](RandSample) | cuRAND generators and distributions |
| [`MemoryAndLoggingSample`](MemoryAndLoggingSample) | The caching memory manager, `DeviceMemory.Default` and the `NVMathLog` hook |

Layout reminder: BLAS-style modules (matmul, solvers, sparse dense operands) and tensors are **column-major**; the FFT
module follows cuFFT and is **row-major** (the last axis is contiguous).
