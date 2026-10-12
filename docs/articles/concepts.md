# Core concepts

## Device buffers

`DeviceBuffer<T>` is a typed block of GPU memory. `T` is an unmanaged numeric type (`float`, `double`,
`Half`, `int`, `System.Numerics.Complex`, …). It owns its allocation: **dispose it** (with `using`) to free
the memory.

```csharp
using var buf = new DeviceBuffer<float>(1024);
buf.CopyFrom(hostArray);          // host -> device
float[] back = buf.ToArray();     // device -> host
buf.Clear();                      // zero fill
```

`PointerAsInt` / `Pointer` expose the raw device pointer for interop with other CUDA code. Pinned host memory
for faster transfers is available as `PinnedBuffer<T>`.

## Layout: column-major, except FFT

Matrices and tensors are **column-major** (the first index varies fastest), matching cuBLAS, cuSOLVER and cuSPARSE.
For tensors the same rule holds: the first mode has stride 1. Dense matrices passed to sparse routines
(`SpMM`, `SpSM`, …) are column-major too.

The **FFT module is the exception**: it follows cuFFT, so a shape such as `[batch, n]` is **row-major** — the *last* axis
is contiguous in memory, and `axes` selects which of those dimensions are transformed. See the [FFT guide](../api/Fft.md).

## Streams, async and blocking

GPU work is queued on a `CudaStream`. Most operation classes create a private stream unless you pass one in, and
only dispose a stream they created.

- `Execute…` methods **launch** work. By default the call returns when the work is queued.
- Set `Blocking = true` in the options to make each call wait for completion.
- `SynchronizeAsync()` awaits all work queued on the stream; `Synchronize()` blocks.
- Static one-shot helpers (`SparseLinearAlgebra.SpMVAsync`, `DenseSolver.SolveAsync`, …) synchronise before they return.

Passing the **same `CudaStream` to several operations** chains them in order without host synchronisation, and
different streams run concurrently. See the `ChainingSample` project for a worked example.

```csharp
await using var stream = new CudaStream();
using var op1 = new Matmul(m, n, k, options: nonBlocking, stream: stream);
using var op2 = new Matmul(m, n, k, options: nonBlocking, stream: stream);
// … launch op1, then op2 on the same stream …
await stream.SynchronizeAsync();
```

## The plan → execute lifecycle

Most classes (`Fft`, `Matmul`, `TensorContraction`, `SparseDirectSolver`, …) separate **planning** from
**execution**. Planning inspects shapes and types and picks algorithms; execution runs on the GPU. Create the
object once, plan once, execute repeatedly. Scalars such as `alpha` and `beta` are passed at execution time.

## Resource ownership

- Anything holding GPU resources implements `IDisposable` (and usually `IAsyncDisposable`). Always use `using`.
- Wrapper classes accept an external `CudaStream` or create their own, and dispose only the stream they created.
- Sparse matrix types (`CsrMatrix<T>`, `CooMatrix<T>`, `BsrMatrix<T>`, `CscMatrix<T>`) **own** the buffers you give them
  and dispose them in `Dispose()`.
- `SparseDirectSolver<T>` borrows its matrix: dispose the solver first, then the matrix.

## Memory managers

By default `DeviceBuffer<T>` allocates with `cudaMalloc`. Plug in a different allocator, for example the
included `CachingMemoryManager`, which reuses freed blocks:

```csharp
using NVMathNet.Memory;

using var cache = new CachingMemoryManager();
DeviceMemory.Default = cache;               // all new buffers and internal workspaces use it
using var x = new DeviceBuffer<float>(1 << 20);
```

A buffer returns its memory to the manager it came from, so changing `DeviceMemory.Default` never affects
existing buffers. You can also pass a manager to a single buffer: `new DeviceBuffer<float>(n, manager)`.
Library-internal scratch memory for cuFFT, cuBLASLt, cuTENSOR and cuSPARSE goes through the same manager.

## Errors

Native failures surface as typed exceptions that carry the raw status code:

| Exception | Library |
|---|---|
| `CudaException` | CUDA runtime |
| `CuFftException` | cuFFT |
| `CuBlasException` | cuBLAS / cuBLASLt |
| `CuSparseException` | cuSPARSE |
| `CuSolverException` | cuSOLVER |
| `CuRandException` | cuRAND |
| `CuTensorException` | cuTENSOR |
| `CuDssException` | cuDSS |

Numerical failures, such as a singular matrix passed to `DenseSolver.SolveAsync`, throw `InvalidOperationException`
or are reported through the return value, as documented per method. Argument problems throw the usual
`ArgumentException` family.

## Multiple GPUs

`MultiGpuContext` creates one stream per device and runs work on all of them concurrently:

```csharp
using var ctx = new MultiGpuContext();                    // all visible GPUs
await ctx.ForEachDeviceAsync(async (index, deviceId, stream) =>
{
    using var buf = new DeviceBuffer<float>(1024);
    // … work on device `deviceId` using `stream` …
    await stream.SynchronizeAsync();
});
```

## Precision and scalar types

`alpha`/`beta` parameters are `double` (or `T`, where noted) and are converted to the type the native routine
expects, usually the compute type of the operation: half, float or double. Complex data uses
`System.Numerics.Complex` (double precision) where supported; .NET has no single-precision complex type.

## Logging

The library is silent by default. Install a handler on `NVMathLog` to receive diagnostics, for example to forward
them to `Microsoft.Extensions.Logging` (NVMath.Net itself has no logging dependency):

```csharp
NVMathLog.MinimumLevel = NVMathLogLevel.Information;
NVMathLog.Handler = (level, category, message) =>
    logger.Log(level switch
    {
        NVMathLogLevel.Error => LogLevel.Error,
        NVMathLogLevel.Warning => LogLevel.Warning,
        NVMathLogLevel.Information => LogLevel.Information,
        _ => LogLevel.Debug,
    }, "{Category}: {Message}", category, message);
```

| Level | What is logged |
|---|---|
| `Error` | Every failed native call (library, function and status), immediately before the typed exception is thrown |
| `Information` | Native libraries being loaded, `Matmul.Autotune` results |
| `Debug` | Plan details: chosen shapes, candidate algorithms, workspace sizes, cuDSS phases |

The handler runs synchronously on the calling thread (possibly from several threads at once), so keep it fast and
thread-safe. Exceptions thrown by the handler are swallowed so logging can never break a GPU operation.
