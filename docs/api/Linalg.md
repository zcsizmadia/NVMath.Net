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
| `Task ExecuteAsync<T>(DeviceBuffer<T> a, b, c, d, float alpha = 1f, float beta = 0f, CancellationToken ct = default)` | Typed async execution. |
| `void Execute(nint pA, pB, pC, pD, float alpha = 1f, float beta = 0f)` | Raw pointer execution. |
| `Task SynchronizeAsync(CancellationToken ct = default)` | Synchronises the stream. |

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
