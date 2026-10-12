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
| [`FftSample`](https://github.com/zcsizmadia/NVMath.Net/tree/main/samples/FftSample) | Basic 1-D FFT and the one-shot helpers |
| [`FftAdvancedSample`](https://github.com/zcsizmadia/NVMath.Net/tree/main/samples/FftAdvancedSample) | N-D and batched transforms, leading-axis layouts, real-to-complex round trips, output normalization |
| [`MatmulSample`](https://github.com/zcsizmadia/NVMath.Net/tree/main/samples/MatmulSample) | A first cuBLASLt matrix product |
| [`MatmulAdvancedSample`](https://github.com/zcsizmadia/NVMath.Net/tree/main/samples/MatmulAdvancedSample) | Bias epilog, strided-batch matmul, autotuning with the plan cache, complex matmul with conjugate transpose |
| [`ChainingSample`](https://github.com/zcsizmadia/NVMath.Net/tree/main/samples/ChainingSample) | Chaining operations on one stream, multi-stream patterns |
| [`SparseSample`](https://github.com/zcsizmadia/NVMath.Net/tree/main/samples/SparseSample) | SpMV and SpMM on a CSR matrix |
| [`SparseAdvancedSample`](https://github.com/zcsizmadia/NVMath.Net/tree/main/samples/SparseAdvancedSample) | Triangular solves, SpGEMM, CSR/CSC/COO and transpose |
| [`DirectSolverSample`](https://github.com/zcsizmadia/NVMath.Net/tree/main/samples/DirectSolverSample) | cuDSS sparse direct solver: analyse / factorise / solve |
| [`TensorSample`](https://github.com/zcsizmadia/NVMath.Net/tree/main/samples/TensorSample) | A cuTENSOR contraction |
| [`TensorOpsSample`](https://github.com/zcsizmadia/NVMath.Net/tree/main/samples/TensorOpsSample) | Permutation, reduction, element-wise binary and contraction with mode labels |
| [`SolverSample`](https://github.com/zcsizmadia/NVMath.Net/tree/main/samples/SolverSample) | LU, Cholesky, SVD and eigenvalues |
| [`LeastSquaresSample`](https://github.com/zcsizmadia/NVMath.Net/tree/main/samples/LeastSquaresSample) | QR, polynomial least-squares fitting, factor-once/solve-many LU, batched LU and SVD |
| [`ComplexSample`](https://github.com/zcsizmadia/NVMath.Net/tree/main/samples/ComplexSample) | Complex dense solvers, SVD, Hermitian eigenproblem, complex sparse products, Hermitian cuDSS solve |
| [`RandSample`](https://github.com/zcsizmadia/NVMath.Net/tree/main/samples/RandSample) | cuRAND generators and distributions |
| [`MemoryAndLoggingSample`](https://github.com/zcsizmadia/NVMath.Net/tree/main/samples/MemoryAndLoggingSample) | The caching memory manager, `DeviceMemory.Default` and the `NVMathLog` hook |

Layout reminder: BLAS-style modules (matmul, solvers, sparse dense operands) and tensors are **column-major**; the FFT
module follows cuFFT and is **row-major** (the last axis is contiguous).
