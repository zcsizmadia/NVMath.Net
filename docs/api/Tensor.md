# Tensor — `NVMathNet.Tensor`

## Enums

```csharp
enum TensorDataType
{
    Float32 = 0, Float64 = 1, Float16 = 2,
    Complex64 = 4, Complex128 = 5, BFloat16 = 14,
}

enum TensorAlgorithm
{
    DefaultPatient = -6, Gett = -4, Tgett = -3, Ttgt = -2, Default = -1,
}

enum WorksizePreference { Min = 1, Recommended = 2, Max = 3 }
enum ContractionAutotuneMode { None = 0, Auto = 1, Benchmark = 2 }
```

## `TensorContractionOptions`

| Property | Type | Default | Description |
|---|---|---|---|
| `WorksizePreference` | `WorksizePreference` | `Recommended` | Workspace size strategy. |
| `Algorithm` | `TensorAlgorithm` | `Default` | Algorithm selection hint. |
| `AutotuneMode` | `ContractionAutotuneMode` | `None` | Auto-tuning mode. |
| `Blocking` | `bool` | `true` | Synchronise after execution. |

## `TensorContraction`

Stateful cuTENSOR v2 contraction wrapper. Implements `IAsyncDisposable`, `IDisposable`.

| Member | Description |
|---|---|
| `TensorContraction(long[] extentA, int[] modeA, long[] extentB, int[] modeB, long[] extentC, int[] modeC, long[] extentD, int[] modeD, TensorDataType dataType = Float32, TensorContractionOptions? options = null, CudaStream? stream = null)` | Creates the contraction. |
| `Task PlanAsync(CancellationToken ct = default)` | Plans (selects algorithm, allocates workspace). |
| `void Plan()` | Synchronous plan. |
| `Task ExecuteAsync<T>(DeviceBuffer<T> a, b, c, d, double alpha = 1.0, double beta = 0.0, CancellationToken ct = default)` | Typed async execution. |
| `void Execute(nint pA, pB, pC, pD, double alpha = 1.0, double beta = 0.0)` | Raw pointer execution. |
| `Task SynchronizeAsync(CancellationToken ct = default)` | Synchronises the stream. |

## `ElementwiseTrinary`

Stateful cuTENSOR v2 element-wise trinary: `D = opABC(α·opA(A), opAB(β·opB(B), γ·opC(C)))`.
Implements `IAsyncDisposable`, `IDisposable`.

Unary operators: `Identity`, `Sqrt`, `Relu`, `Conj`, `Rcp`, `Sigmoid`, `Tanh`, `Exp`, `Log`, `Abs`, `Neg`, `Sin`, `Cos`, `Tan`, `Sinh`, `Cosh`, `Asin`, `Acos`, `Atan`, `Asinh`, `Acosh`, `Atanh`, `Ceil`, `Floor`, `Mish`, `Swish`, `SoftPlus`, `SoftSign`. Binary operators (for `opAB` / `opABC`): `Add`, `Mul`, `Max`, `Min`. Values match `cutensor/types.h`. Scalars (`alpha`, `beta`, `gamma`) are encoded as half, float or double according to the data type's compute descriptor.

| Member | Description |
|---|---|
| `ElementwiseTrinary(long[] extentA, int[] modeA, ..., TensorDataType dataType = Float32, CuTensorOperator opA = Identity, opB = Identity, opC = Identity, opAB = Add, opABC = Add, ...)` | Creates the operation. |
| `Task ExecuteAsync<T>(DeviceBuffer<T> a, b, c, d, double alpha = 1.0, double beta = 1.0, double gamma = 1.0, CancellationToken ct = default)` | Typed async execution. |
| `Task SynchronizeAsync(CancellationToken ct = default)` | Synchronises the stream. |

## Permutation, element-wise binary and reduction

Tensors use a packed **column-major** layout (the first mode varies fastest). Modes are integer labels shared between operands; the same label means the same dimension. All three classes derive from `TensorOperation` (`IDisposable`, `IAsyncDisposable`, `SynchronizeAsync`) and accept `TensorContractionOptions` (`Blocking`) and an optional `CudaStream`.

| Type | Computes | Notes |
|---|---|---|
| `TensorPermutation(long[] extentA, int[] modeA, int[] modeB, TensorDataType dataType = Float32, CuTensorOperator opA = Identity, ...)` | `B[modeB] = alpha · opA(A[modeA])` | `modeB` is a reordering of `modeA` (generalised transpose). `Execute(pA, pB, alpha)` / `ExecuteAsync<T>(a, b, alpha)`. |
| `ElementwiseBinary(extentA, modeA, extentC, modeC, extentD, modeD, dataType, opA, opC, opAC = Add, ...)` | `D = opAC(alpha · opA(A), gamma · opC(C))` | Only A may be permuted: C must have the same modes and extents as D (cuTENSOR restriction). `Execute(pA, pC, pD, alpha, gamma)`. |
| `TensorReduction(extentA, modeA, extentD, modeD, dataType, opReduce = Add, opA = Identity, opC = Identity, ...)` | `D = alpha · opReduce(opA(A)) + beta · opC(C)` | Reduces the modes of A that are not in `modeD` (a subset of `modeA`). `opReduce` is `Add`, `Mul`, `Max` or `Min`. `ExecuteInPlace(pA, pD, alpha, beta)` works in place on D; `Execute(pA, pC, pD, ...)` uses a separate C. |

```csharp
using var transpose = new TensorPermutation([2, 3], [0, 1], [1, 0]);   // 2×3 -> 3×2
using var rowSums = new TensorReduction([3, 4], [0, 1], [3], [0]);      // sum over mode 1
```
