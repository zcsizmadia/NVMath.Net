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

- **`CsrMatrix<T>` constructor changed.** The old five-argument constructor was replaced (binary breaking; rebuild).
  The `Interop` enum value corrections above change the numeric values of public interop enums.
- **Single-/double-precision real FFT helpers.** `Fft.RFftAsync(DeviceBuffer<float>)` previously returned a
  `DeviceBuffer<Complex>` of garbage (single-precision data in a double-complex buffer). It now returns interleaved
  `(re, im)` floats; `RFftAsync(DeviceBuffer<double>)` returns `DeviceBuffer<Complex>`. `IRFftAsync` has matching
  `Complex`->`double` and interleaved `float`->`float` overloads.
- **`TensorReduction.Execute(pA, pD, ...)` renamed `ExecuteInPlace`** (the shorter overload silently captured calls
  meant for the four-pointer form).
- **In-place R2C/C2R `Fft` throws `NotSupportedException`** (it needs a padded real layout that was never implemented).

### Added

- **FFT:** N-D and batched plans with leading- or trailing-axis layouts; `FftOptions.Normalization`
  (`None`, `Backward`, `Forward`, `Orthonormal`).
- **Matmul (cuBLASLt):** strided-batch matmul, `SetScalePointers` (FP8 A/B/C/D scales and amax), `SetBias`,
  `SetEpilogAux`, `MatmulOptions.FastAccum`, additional epilog values, algorithm candidates with `AlgorithmCount`,
  `SelectAlgorithm` and `Autotune`/`AutotuneAsync`, `MatmulPlanCache` (repeated plans skip the heuristic and reuse
  autotuned choices), conjugate transpose of A/B (`ConjugateTransposeA/B`), complex element/scale types
  (`Complex64`, `Complex128`) and complex-scalar `Execute` overloads.
- **Sparse (cuSPARSE):** SpSV, SpSM, SDDMM, SpGEMM, `CsrMatrix.Transpose()`, `CscMatrix<T>` with `CsrMatrix.ToCsc()`,
  SpMV/SpMM for COO, BSR and CSC, SpSV/SpSM for COO, empty SpGEMM products, complex (double) values for all
  operations cuSPARSE supports them for, and `T`-typed scalar overloads of SpMV/SpMM (complex `alpha`/`beta`).
- **Sparse direct solver (cuDSS):** `SparseDirectSolver<T>` with analyse / factorise / refactorise / solve, general,
  symmetric, Hermitian and positive-definite structures, `float`/`double`/`Complex` values, multiple right-hand sides,
  reordering and iterative-refinement options.
- **Tensors (cuTENSOR 2):** `TensorPermutation`, `ElementwiseBinary`, `TensorReduction` on a shared `TensorOperation`
  base; all unary operators from the cuTENSOR header.
- **Dense solvers (cuSOLVER):** `LuFactorAsync`/`LuSolveAsync`, `QrAsync`/`QrFormQAsync`, `LeastSquaresAsync`,
  batched Cholesky, batched LU, batched Jacobi SVD, and complex (double) LU, QR, least squares, Cholesky, SVD,
  `HermitianEigAsync` and the batched Cholesky / LU / Jacobi SVD routines.
- **Random numbers (cuRAND):** `FillUInt32`, `FillUInt64`, `FillPoisson`, `SetOffset` (skip-ahead),
  `SetQuasiRandomDimensions`, `RngType`.
- **Memory:** `IDeviceMemoryManager`, `DeviceMemory.Default`, `CachingMemoryManager`, and a `DeviceBuffer<T>` constructor
  taking a manager. Internal workspaces (cuFFT, cuBLASLt, cuTENSOR, cuSPARSE) use the same manager.
- **Logging:** a dependency-free hook, `NVMathLog` (`Handler`, `MinimumLevel`), that reports failed native calls,
  library loads, autotune results and plan details; bridge it to `ILogger` or any logger.
- **Native loading:** the default cuDSS install folder and `CUDSS_ROOT` are probed on Windows, like cuTENSOR.
- **Documentation:** a DocFX website (guides, module references and generated API reference) published to GitHub
  Pages, a rewritten README and `API.md`, and a new `DirectSolverSample`.

### Fixed

- **Complex scalars and alignment:** `cuDoubleComplex` needs 16-byte alignment but `System.Numerics.Complex` only
  guarantees 8, so passing a complex scalar by address could access-violate inside cuSPARSE or cuBLAS depending on stack
  placement. Host scalars of generic or complex type are now stored in 16-byte aligned memory.
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

- **Review fixes (FFT / cuTENSOR):** static FFT helpers dropped `Normalization` and mutated the caller's options;
  `FftOptions` is now snapshotted at construction; Float16 contractions use 32F compute with float scalars; complex
  tensor scalars no longer over-read the stack; plans synchronise their stream before releasing workspace; constructors
  and helpers no longer leak streams or output buffers on failure.

- **Review fixes (sparse / dense solvers / cuRAND):** sparse constructors and operations validate buffer sizes
  (an undersized dense operand used to corrupt memory); `Transpose()`/`ToCsc()` work for matrices with no stored
  entries (`CscMatrix` gained an optional `nnz`); workspaces are released only after the stream has finished with them;
  `SparseDirectSolver` no longer solves with stale factors after a failed `Refactorize` or after `SetValues`; dense
  solvers validate dimensions and buffer sizes before calling cuSOLVER (a bad size used to leave the CUDA context in a
  sticky error state) and the real `SvdAsync` rejects `m < n` with a clear message; `CudaRandom` no longer leaks its
  generator when configuration fails and `FillPoisson` rejects non-finite or oversized lambda.

### Not included (tracked after 1.0)

Block-scaled MXFP8/NVFP4 matmul (needs Blackwell hardware to verify) and FFT prolog/epilog callbacks (needs
NVRTC/LTO-IR). See the issues labelled [`post-1.0`](https://github.com/zcsizmadia/NVMath.Net/labels/post-1.0).

Some things are intentionally absent because the NVIDIA libraries do not offer them: binomial and histogram discrete
distributions in the cuRAND host API (the entry points are not exported), triangular solves for BSR matrices in
cuSPARSE, SpGEMM/SDDMM for COO and BSR, and a structured-matrix matmul in cuBLASLt (use `TriangularSolve` and
`SymmetricMultiply`).

## [0.9.0]

Initial public release.
