# Changelog

All notable changes to NVMath.Net are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to
[Semantic Versioning](https://semver.org/).

## [1.0.0] - unreleased

First stable release. The public API is now considered stable; breaking changes will require a new major version.

### Breaking changes since 0.9.0

- **`Matmul` scalars are `double`.** `Execute`, `ExecuteAsync` and `ExecuteAsync<T>` take `double alpha, double beta`
  (previously `float`) and encode them according to `MatmulOptions.ScaleType`. Source compatible for callers passing
  `float`; assemblies compiled against 0.9.0 must be rebuilt.
- **cuTENSOR enums corrected.** `CuTensorOperator` had wrong values for `Add`, `Mul`, `Max`, `Min`, `Rcp`, `Sigmoid`,
  `Tanh` and `Sin`, and `RcpSqrt` (not a cuTENSOR operator) is removed. `CuTensorStatus` values from `CudaError` onward
  were off by one.
- **`MatmulEpilog` values corrected:** `GeluAux` is 160 and `GeluBias` is 36 (they were 96 and 100).
- **Dense solvers throw on failure.** `DenseSolver.SolveAsync`, `CholeskyAsync`, `SvdAsync` and `EigAsync` now check
  cuSOLVER's `devInfo` and throw `InvalidOperationException` for a singular matrix, a matrix that is not positive
  definite, or a decomposition that did not converge. Previously they returned silently wrong results.
- **`Fft` validates `axes`.** Axes must be ascending, contiguous and include the first or last dimension (negative
  values are accepted); anything else throws. Leading-axis transforms are now planned with the correct strided layout.
- **`CudaRandom` rejects a non-zero seed for quasirandom generators** with an `ArgumentException` (cuRAND does not
  support seeding them).

### Added

- **FFT:** N-D and batched plans with leading- or trailing-axis layouts; `FftOptions.Normalization`
  (`None`, `Backward`, `Forward`, `Orthonormal`).
- **Matmul (cuBLASLt):** strided-batch matmul, `SetScalePointers` (FP8 A/B/C/D scales and amax), `SetBias`,
  `SetEpilogAux`, `MatmulOptions.FastAccum`, additional epilog values, algorithm candidates with `AlgorithmCount`,
  `SelectAlgorithm` and `Autotune`/`AutotuneAsync`.
- **Sparse (cuSPARSE):** SpSV, SpSM, SDDMM, SpGEMM, `CsrMatrix.Transpose()`, `CscMatrix<T>` with `CsrMatrix.ToCsc()`,
  SpMV/SpMM for COO, BSR and CSC, SpSV/SpSM for COO, empty SpGEMM products.
- **Sparse direct solver (cuDSS):** `SparseDirectSolver<T>` with analyse / factorise / refactorise / solve, general,
  symmetric and SPD structures, multiple right-hand sides, reordering and iterative-refinement options.
- **Tensors (cuTENSOR 2):** `TensorPermutation`, `ElementwiseBinary`, `TensorReduction` on a shared `TensorOperation`
  base; all unary operators from the cuTENSOR header.
- **Dense solvers (cuSOLVER):** `LuFactorAsync`/`LuSolveAsync`, `QrAsync`/`QrFormQAsync`, `LeastSquaresAsync`,
  batched Cholesky, batched LU, batched Jacobi SVD, and complex (double) LU, QR, least squares, Cholesky and
  `HermitianEigAsync`.
- **Random numbers (cuRAND):** `FillUInt32`, `FillUInt64`, `FillPoisson`, `SetOffset` (skip-ahead),
  `SetQuasiRandomDimensions`, `RngType`.
- **Memory:** `IDeviceMemoryManager`, `DeviceMemory.Default`, `CachingMemoryManager`, and a `DeviceBuffer<T>` constructor
  taking a manager. Internal workspaces (cuFFT, cuBLASLt, cuTENSOR, cuSPARSE) use the same manager.
- **Native loading:** the default cuDSS install folder and `CUDSS_ROOT` are probed on Windows, like cuTENSOR.
- **Documentation:** a DocFX website (guides, module references and generated API reference) published to GitHub
  Pages, a rewritten README and `API.md`, and a new `DirectSolverSample`.

### Fixed

- **cuBLASLt:** the transpose, epilog and implementation-mask attribute IDs were wrong (transposition was written to
  the compute-type attribute), and `batchCount` was accepted but never applied.
- **Matmul / SpMV / SpMM / cuTENSOR scalars:** `float` scalars were passed regardless of the operation's precision,
  producing wrong results for double (and 16F) operations.
- **cuTENSOR:** `ElementwiseTrinary` could never be constructed because its default `Add` operator had the value of a
  different operator; it now also resolves its compute descriptor from the data type.
- **cuSPARSE:** `BsrMatrix` could not be created (invalid block-order argument); the `CreateDnMat` documentation had
  the order values wrong.
- **FFT:** batch distances for R2C and C2R used the full extent for the half-spectrum operand.
- **Build:** the `TUnit` meta package was left at an old version while `TUnit.Core`/`Assertions` moved on; Dependabot
  now updates the `TUnit*` packages together. CI builds the test and sample projects as well as the library.

### Not included (tracked after 1.0)

Block-scaled MXFP8/NVFP4 matmul, Matmul plan cache and matrix qualifiers, FFT prolog/epilog callbacks, complex SVD
and complex batched solvers, further cuRAND distributions, and an optional logging hook. See the issues labelled
[`post-1.0`](https://github.com/zcsizmadia/NVMath.Net/labels/post-1.0).

## [0.9.0]

Initial public release.
