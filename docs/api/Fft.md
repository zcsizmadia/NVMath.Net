# FFT — `NVMathNet.Fft`

## Enums

```csharp
enum FftDirection { Forward = -1, Inverse = 1 }
enum FftType { C2C, R2C, C2R }
enum LastAxisParity { Even, Odd }
```

## `FftOptions`

| Property | Type | Default | Description |
|---|---|---|---|
| `FftType` | `FftType?` | `null` (auto) | Transform type. |
| `InPlace` | `bool` | `false` | Execute in-place. |
| `LastAxisParity` | `LastAxisParity` | `Even` | Parity of the last axis for R2C/C2R. |
| `DeviceId` | `int` | `0` | GPU device index. |
| `Blocking` | `bool` | `false` | Synchronise after execution. |

## `Fft` (class)

Stateful cuFFT plan wrapper. Implements `IAsyncDisposable`, `IDisposable`.

| Member | Description |
|---|---|
| `Fft(long[] shape, int[]? axes = null, FftOptions? options = null, CudaStream? stream = null, bool doublePrecision = false)` | Creates a plan descriptor. `axes` selects which dimensions to transform (`null` = all). |
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
