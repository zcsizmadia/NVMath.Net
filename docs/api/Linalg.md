# Linear Algebra — `NVMathNet.LinAlg`

## Enums

```csharp
enum MatmulComputeType
{
    Compute16F = 64, Compute16FPedantic = 65,
    Compute32F = 68, Compute32FPedantic = 69,
    Compute32FFast16F = 74, Compute32FFast16BF = 75, Compute32FFastTF32 = 77,
    Compute64F = 70, Compute64FPedantic = 71,
    Compute32I = 72, Compute32IPedantic = 73,
}

enum MatrixDataType
{
    Float32 = 0, Float64 = 1, Float16 = 2, Int8 = 3,
    Int32 = 10, BFloat16 = 14, FP8E4M3 = 28, FP8E5M2 = 29,
}

enum MatmulEpilog { Default = 1, Relu = 2, Bias = 4, ReluBias = 6, Gelu = 32, GeluAux = 96, GeluBias = 100 }
```

---

## Specialized Matmul (`Matmul`)

### `MatmulOptions`

| Property | Type | Default | Description |
|---|---|---|---|
| `ComputeType` | `MatmulComputeType` | `Compute32F` | Accumulation precision. |
| `TypeA/B/C/D` | `MatrixDataType` | `Float32` | Per-operand data types. |
| `ScaleType` | `MatrixDataType` | `Float32` | Alpha/beta scalar type. |
| `Epilog` | `MatmulEpilog` | `Default` | Post-multiply activation. |
| `MaxWorkspaceBytes` | `nuint` | `4 MB` | Max workspace size. |
| `TransposeA` | `bool` | `false` | Transpose A before multiply. |
| `TransposeB` | `bool` | `false` | Transpose B before multiply. |
| `FastAccum` | `bool` | `false` | FP8 fast accumulation. |
| `Blocking` | `bool` | `true` | Synchronise after execution. |

### `MatmulPlanPreferences`

| Property | Type | Default | Description |
|---|---|---|---|
| `RequestedAlgoCount` | `int` | `4` | Number of algorithms to evaluate. |
| `NumericalImplMask` | `ulong` | `0` | Implementation mask. |
| `MaxWorkspaceBytes` | `nuint?` | `null` | Override workspace limit for planning. |

### `Matmul`

Stateful cuBLASLt matrix multiplication wrapper. Computes `D = α·op(A)·op(B) + β·C`.
All matrices use **column-major** layout.
Implements `IAsyncDisposable`, `IDisposable`.

| Member | Description |
|---|---|
| `Matmul(long m, long n, long k, long batchCount = 1, MatmulOptions? options = null, CudaStream? stream = null)` | Creates the wrapper. |
| `Task PlanAsync(MatmulPlanPreferences? prefs = null, CancellationToken ct = default)` | Plans the operation (selects algorithm). |
| `void Plan(MatmulPlanPreferences? prefs = null)` | Synchronous plan. |
| `Task ExecuteAsync<T>(DeviceBuffer<T> a, b, c, d, double alpha = 1, double beta = 0, CancellationToken ct = default)` | Typed async execution. |
| `void Execute(nint pA, pB, pC, pD, double alpha = 1, double beta = 0)` | Raw pointer execution. |
| `Task SynchronizeAsync(CancellationToken ct = default)` | Synchronises the stream. |
| `void SetScalePointers(nint aScale = 0, bScale = 0, cScale = 0, dScale = 0, amaxD = 0)` | Device pointers to FP8 scale factors and `amax(D)` output. |
| `void SetBias(nint bias, MatrixDataType? biasType = null, long batchStride = 0)` | Bias vector for the `Bias` epilogs. |
| `void SetEpilogAux(nint aux, long ld, MatrixDataType? auxType = null, long batchStride = 0)` | Aux output for the `*Aux` epilogs (GELU: pre-activation values, `ld` a multiple of 8; ReLU: bit mask, `ld` counts bits, multiple of 128). Call before `Plan`. |
| `int AlgorithmCount`, `int SelectedAlgorithm` | Number of planned candidate algorithms (best-first) and the one `Execute` uses. Request several via `MatmulPlanPreferences.RequestedAlgoCount`. |
| `void SelectAlgorithm(int index)` | Picks a candidate algorithm. |
| `float[] Autotune(nint pA, pB, pC, pD, double alpha = 1, double beta = 0, int iterations = 3)` / `AutotuneAsync(...)` | Times every candidate and selects the fastest; returns average ms per candidate. **D is overwritten** by the trial runs. |

**Batched matmul:** pass `batchCount > 1`; A, B, C and D are expected packed back-to-back
(strides `m*k`, `k*n`, `m*n`, `m*n` elements).

---

## Generic GEMM (`Gemm`)

Convenience wrappers for single-shot GEMM using cuBLAS. Column-major layout.

| Method | Description |
|---|---|
| `static Task SgemmAsync(DeviceBuffer<float> a, b, c, int m, n, k, ...)` | C = α·op(A)·op(B) + β·C (float). |
| `static Task DgemmAsync(...)` | Same for double. |
| `static Task HgemmAsync(...)` | Same for Half. |
| `static void Sgemm(...)` | Synchronous float GEMM. |
| `static void Dgemm(...)` | Synchronous double GEMM. |

---

## Structured Matrix Operations

### `TriangularSolve` (static)

Solves triangular systems using cuBLAS TRSM: `B = alpha * inv(op(A)) * B`.

| Member | Description |
|---|---|
| `Task StrsmAsync(DeviceBuffer<float> a, b, int m, n, ...)` | Single-precision triangular solve. |
| `Task DtrsmAsync(DeviceBuffer<double> a, b, int m, n, ...)` | Double-precision triangular solve. |

Parameters: `alpha`, `leftSide`, `upper`, `transpose`, `unitDiag`, `stream`, `ct`.

### `SymmetricMultiply` (static)

Symmetric matrix-matrix multiply using cuBLAS SYMM: `C = alpha * A * B + beta * C` where A is symmetric.

| Member | Description |
|---|---|
| `Task SsymmAsync(DeviceBuffer<float> a, b, c, int m, n, ...)` | Single-precision symmetric multiply. |
| `Task DsymmAsync(DeviceBuffer<double> a, b, c, int m, n, ...)` | Double-precision symmetric multiply. |

Parameters: `alpha`, `beta`, `leftSide`, `upper`, `stream`, `ct`.

### `DiagonalMultiply` (static)

Diagonal matrix multiply using cuBLAS DGMM: `C = diag(x) * A` (left) or `C = A * diag(x)` (right).

| Member | Description |
|---|---|
| `Task SdgmmAsync(DeviceBuffer<float> a, x, c, int m, n, ...)` | Single-precision diagonal multiply. |
| `Task DdgmmAsync(DeviceBuffer<double> a, x, c, int m, n, ...)` | Double-precision diagonal multiply. |

Parameters: `leftSide`, `stream`, `ct`.

Alpha and beta are `double` and are encoded according to `MatmulOptions.ScaleType` (`Float32`, `Float64`, `Float16`, `BFloat16` or `Int32`).

### Conjugate transpose, complex matmul and the plan cache

| Member | Description |
|---|---|
| `MatmulOptions.ConjugateTransposeA` / `ConjugateTransposeB` | Use `Aᴴ` / `Bᴴ` (implies a transpose; takes precedence over `TransposeA/B`). |
| `MatrixDataType.Complex64` / `Complex128` | Complex element and scale types. Use `Compute64F` with `Complex128` (`DeviceBuffer<System.Numerics.Complex>`). |
| `Execute(..., Complex alpha, Complex beta)`, `ExecuteAsync<T>(..., Complex alpha, Complex beta)` | Complex scalars; for real scale types only the real part is used. |
| `MatmulPlanCache.Enabled` | Caches planning results process-wide (default `true`). A repeated `Matmul` with the same shape, types, options, epilog configuration and preferences on the same device skips the cuBLASLt heuristic and reuses the algorithm chosen by `Autotune`. |
| `MatmulPlanCache.Count`, `Hits`, `Misses`, `Clear()` | Observability and reset. |

Symmetric and triangular operands are handled by the dedicated `SymmetricMultiply` and `TriangularSolve` classes rather than by matmul qualifiers, because cuBLASLt has no structured-matrix matmul.
