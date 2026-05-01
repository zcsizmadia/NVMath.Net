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

Available operators: `Identity`, `Sqrt`, `ReLU`, `Conj`, `RcpSqrt`, `Sin`, `Rcp`, `Sigmoid`, `Tanh`, `Exp`, `Log`, `Abs`, `Neg`, `Add`, `Mul`, `Max`, `Min`.

| Member | Description |
|---|---|
| `ElementwiseTrinary(long[] extentA, int[] modeA, ..., TensorDataType dataType = Float32, CuTensorOperator opA = Identity, opB = Identity, opC = Identity, opAB = Add, opABC = Add, ...)` | Creates the operation. |
| `Task ExecuteAsync<T>(DeviceBuffer<T> a, b, c, d, double alpha = 1.0, double beta = 1.0, double gamma = 1.0, CancellationToken ct = default)` | Typed async execution. |
| `Task SynchronizeAsync(CancellationToken ct = default)` | Synchronises the stream. |
