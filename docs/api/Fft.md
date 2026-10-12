# FFT — `NVMathNet.Fft`

## Memory layout

Unlike the BLAS-style modules, FFT shapes are **row-major** (cuFFT convention): in a shape such as `[batch, n]` the **last** axis is contiguous in memory. `axes` selects the dimensions to transform; they must be ascending, contiguous and include the first or last dimension. Transforming a leading axis (for example `shape [8, 3]`, `axes [0]`) uses a strided layout. For R2C the shape is the real input; for C2R it is the real output, and the half-spectrum has `n/2 + 1` bins along the last transformed axis.

## Enums

```csharp
enum FftDirection { Forward = -1, Inverse = 1 }
enum FftType { C2C, R2C, C2R }
enum LastAxisParity { Even, Odd }
enum FftNormalization { None, Backward, Forward, Orthonormal }
```

## `FftOptions`

| Property | Type | Default | Description |
|---|---|---|---|
| `FftType` | `FftType?` | `null` (auto) | Transform type. |
| `InPlace` | `bool` | `false` | Execute in-place. |
| `LastAxisParity` | `LastAxisParity` | `Even` | Parity of the last axis for R2C/C2R. |
| `DeviceId` | `int` | `0` | GPU device index. |
| `Blocking` | `bool` | `false` | Synchronise after execution. |
| `Normalization` | `FftNormalization` | `None` | Output scaling: `None` (cuFFT behaviour), `Backward` (1/N on inverse), `Forward` (1/N on forward), `Orthonormal` (1/√N on both). R2C counts as forward and C2R as inverse. |

## `Fft` (class)

Stateful cuFFT plan wrapper. Implements `IAsyncDisposable`, `IDisposable`.

| Member | Description |
|---|---|
| `Fft(long[] shape, int[]? axes = null, FftOptions? options = null, CudaStream? stream = null, bool doublePrecision = false)` | Creates a plan descriptor. `axes` selects which dimensions to transform (`null` = all); they must be ascending, contiguous and include the first or last dimension (negative values allowed). Batched trailing- and leading-axis layouts are supported. |
| `Task PlanAsync(CancellationToken ct = default)` | Builds the cuFFT plan. |
| `void Plan()` | Synchronous plan creation. |
| `Task ExecuteAsync(FftDirection direction = Forward, CancellationToken ct = default)` | Executes the transform. |
| `void Execute(FftDirection direction = Forward)` | Synchronous execution. |
| `void ResetOperand(nint inputPtr, nint outputPtr)` | Changes data pointers without replanning. |
| `Task SynchronizeAsync(CancellationToken ct = default)` | Synchronises the stream. |

### Static Helpers

| Method | Description |
|---|---|
| `static Task<DeviceBuffer<Complex>> FftAsync(...)` | One-shot forward C2C FFT. Returns new output buffer. |
| `static Task<DeviceBuffer<Complex>> IFftAsync(...)` | One-shot inverse C2C FFT. |
| `static Task<DeviceBuffer<Complex>> RFftAsync(DeviceBuffer<float> input, ...)` | Real-to-complex FFT. Output length = `n/2+1`. |
| `static Task<DeviceBuffer<float>> IRFftAsync(DeviceBuffer<Complex> input, long[] outputShape, ...)` | Complex-to-real inverse FFT. |
