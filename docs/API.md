# NVMath.Net API Reference

## Namespaces

| Namespace | Purpose |
|---|---|
| `NVMath` | Core CUDA infrastructure (device, stream, event, buffers) |
| `NVMath.Fft` | cuFFT wrapper |
| `NVMath.LinAlg` | cuBLASLt matrix multiplication |
| `NVMath.Sparse` | cuSPARSE sparse linear algebra |
| `NVMath.Tensor` | cuTENSOR tensor contractions |
| `NVMath.Interop` | Raw P/Invoke bindings (internal use) |

---

## NVMath — Core Infrastructure

### `CudaDevice`

Represents a CUDA-capable device.

| Member | Description |
|---|---|
| `static int Count` | Number of available CUDA devices. |
| `static CudaDevice Current` | Currently selected device. |
| `int Index` | Zero-based device index. |
| `string Name` | Device name string. |
| `static void SetCurrent(int index)` | Selects a device for the calling thread. |
| `static CudaDevice Get(int index)` | Returns the `CudaDevice` for the given index. |

---

### `CudaStream`

Wraps a CUDA stream. Implements `IAsyncDisposable`.

| Member | Description |
|---|---|
| `CudaStream()` | Creates a new stream on the current device. |
| `nint Handle` | Raw CUDA stream handle. |
| `Task SynchronizeAsync(CancellationToken ct = default)` | Waits for all operations in the stream to complete. |
| `void Synchronize()` | Synchronous version of `SynchronizeAsync`. |
| `ValueTask DisposeAsync()` | Destroys the stream asynchronously. |

---

### `CudaEvent`

Wraps a CUDA event for timing and synchronisation. Implements `IDisposable`.

| Member | Description |
|---|---|
| `CudaEvent(bool disableTiming = false)` | Creates an event. |
| `void Record(CudaStream stream)` | Records the event in a stream. |
| `float ElapsedMilliseconds(CudaEvent start)` | Returns elapsed time between two recorded events. |
| `void Synchronize()` | Waits until the event is complete. |

---

### `DeviceBuffer<T>`

Strongly-typed GPU device buffer. Implements `IDisposable`.

| Member | Description |
|---|---|
| `DeviceBuffer(long length)` | Allocates `length * sizeof(T)` bytes on the device. |
| `static Task<DeviceBuffer<T>> AllocAsync(long length, CudaStream stream)` | Allocates asynchronously. |
| `long Length` | Number of elements. |
| `nint PointerAsInt` | Raw device pointer as `nint`. |
| `unsafe void* Pointer` | Raw device pointer. |
| `void CopyFrom(ReadOnlySpan<T> host, CudaStream? stream = null)` | Copies host → device. |
| `void CopyTo(Span<T> host, CudaStream? stream = null)` | Copies device → host. |
| `Task CopyFromAsync(T[] host, CudaStream stream, CancellationToken ct = default)` | Async copy host → device. |
| `Task CopyToAsync(T[] host, CudaStream stream, CancellationToken ct = default)` | Async copy device → host. |
| `void CopyFrom(DeviceBuffer<T> src, CudaStream stream)` | Device-to-device copy. |
| `void Clear(CudaStream? stream = null)` | Fills the buffer with zeros. |
| `T[] ToArray()` | Copies the buffer to a new host array. |
| `void Dispose()` | Frees the device memory. |

---

### `PinnedBuffer<T>`

Page-locked (pinned) host memory for fast DMA transfers. Implements `IDisposable`.

| Member | Description |
|---|---|
| `PinnedBuffer(long length)` | Allocates pinned host memory. |
| `long Length` | Number of elements. |
| `nint PointerAsInt` | Raw pointer as `nint`. |
| `unsafe void* Pointer` | Raw pointer. |
| `unsafe Span<T> Span` | Span over the pinned memory. |
| `void CopyToDevice(DeviceBuffer<T> dst, CudaStream stream)` | Host → device DMA. |
| `void CopyFromDevice(DeviceBuffer<T> src, CudaStream stream)` | Device → host DMA. |
| `void Dispose()` | Frees pinned memory. |

---

## NVMath.Fft

### `FftDirection`

```csharp
enum FftDirection { Forward = -1, Inverse = 1 }
```

### `FftType`

```csharp
enum FftType { C2C, R2C, C2R }
```

### `FftOptions`

| Property | Type | Default | Description |
|---|---|---|---|
| `FftType` | `FftType?` | `null` (auto) | Transform type. |
| `InPlace` | `bool` | `false` | Execute in-place. |
| `LastAxisParity` | `LastAxisParity` | `Even` | Parity of the last axis for R2C/C2R. |
| `DeviceId` | `int` | `0` | GPU device index. |
| `Blocking` | `bool` | `false` | Synchronise after execution. |

### `Fft` (class)

Stateful cuFFT plan wrapper. Implements `IAsyncDisposable`.

| Member | Description |
|---|---|
| `Fft(long[] shape, FftOptions? options = null, CudaStream? stream = null)` | Creates a plan. |
| `Task PlanAsync(CancellationToken ct = default)` | Builds the cuFFT plan. |
| `void Plan()` | Synchronous plan creation. |
| `Task ExecuteAsync(FftDirection direction, CancellationToken ct = default)` | Executes the transform. |
| `void Execute(FftDirection direction)` | Synchronous execution. |
| `void ResetOperand(nint inputPtr, nint outputPtr)` | Changes data pointers without replanning. |
| `Task SynchronizeAsync(CancellationToken ct = default)` | Synchronises the stream. |
| `static Task<Fft> FftAsync(DeviceBuffer<Complex> input, long[] shape, ...)` | Helper: plan + execute forward FFT. |
| `static Task<Fft> IFftAsync(DeviceBuffer<Complex> input, long[] shape, ...)` | Helper: plan + execute inverse FFT. |

---

## NVMath.LinAlg

### `MatmulOptions`

| Property | Type | Default | Description |
|---|---|---|---|
| `ComputeType` | `CuBlasComputeType` | `F32` | Accumulation precision. |
| `ScaleType` | `CudaDataType` | `R_32F` | Alpha/beta scalar type. |
| `Epilogue` | `CuBlasLtEpilogue` | `Default` | Post-multiply activation. |
| `WorkspaceBytes` | `nuint` | `4MB` | Max workspace size. |

### `Matmul`

Stateful cuBLASLt matrix multiplication wrapper. Implements `IAsyncDisposable`.

| Member | Description |
|---|---|
| `Matmul(int m, int n, int k, int batchCount = 1, MatmulOptions? options = null, CudaStream? stream = null)` | Creates the wrapper. |
| `Task PlanAsync(CudaStream? stream = null, CancellationToken ct = default)` | Plans the operation. |
| `void Plan(CudaStream? stream = null)` | Synchronous plan. |
| `Task ExecuteAsync<T>(DeviceBuffer<T> a, b, c, d, float alpha, float beta, CancellationToken ct = default)` | Typed async execution. |
| `void Execute(nint pA, pB, pC, pD, float alpha, float beta)` | Raw pointer execution. |

---

## NVMath.Sparse

### `CsrMatrix<T>`

CSR sparse matrix backed by `DeviceBuffer<>`.

```csharp
CsrMatrix(int rows, int cols,
          DeviceBuffer<int> rowOffsets,
          DeviceBuffer<int> colIndices,
          DeviceBuffer<T> values)
```

### `CooMatrix<T>`

COO sparse matrix backed by `DeviceBuffer<>`.

```csharp
CooMatrix(int rows, int cols,
          DeviceBuffer<int> rowIndices,
          DeviceBuffer<int> colIndices,
          DeviceBuffer<T> values)
```

### `SparseLinearAlgebra` (static)

| Method | Description |
|---|---|
| `Task SpMVAsync<T>(CsrMatrix<T> A, DeviceBuffer<T> x, y, T alpha, T beta, CudaStream? stream = null, CancellationToken ct = default)` | Sparse matrix-vector: y = α·A·x + β·y. |
| `Task SpMMAsync<T>(CsrMatrix<T> A, DeviceBuffer<T> B, C, int n, T alpha, T beta, CudaStream? stream = null, CancellationToken ct = default)` | Sparse matrix-dense matrix: C = α·A·B + β·C. `n` is the number of columns in B/C. |

---

## NVMath.Tensor

### `TensorDataType`

```csharp
enum TensorDataType { Float16, Float32, Float64, BFloat16, Int8, Int32 }
```

### `TensorContractionOptions`

| Property | Type | Default | Description |
|---|---|---|---|
| `Algorithm` | `TensorAlgorithm` | `Default` | Algorithm hint. |
| `WorksizePreference` | `WorksizePreference` | `Recommended` | Workspace preference. |

### `TensorContraction`

Stateful cuTENSOR contraction wrapper. Implements `IAsyncDisposable`.

| Member | Description |
|---|---|
| `TensorContraction(long[] extA, int[] modeA, long[] extB, int[] modeB, long[] extC, int[] modeC, long[] extD, int[] modeD, TensorDataType dataType = Float32, TensorContractionOptions? options = null, CudaStream? stream = null)` | Creates the contraction. |
| `Task PlanAsync(CancellationToken ct = default)` | Plans the contraction. |
| `void Plan()` | Synchronous plan. |
| `Task ExecuteAsync<T>(DeviceBuffer<T> a, b, c, d, float alpha, float beta, CancellationToken ct = default)` | Typed async execution. |
| `void Execute(nint pA, pB, pC, pD, float alpha, float beta)` | Raw pointer execution. |
| `Task SynchronizeAsync(CancellationToken ct = default)` | Stream synchronisation. |

---

## NVMath.Interop

Raw P/Invoke bindings. Not intended for direct use by application code.

| Class | CUDA library |
|---|---|
| `CuFftNative` | cuFFT |
| `CuBlasNative` | cuBLASLt |
| `CuSparseNative` | cuSPARSE |
| `CuTensorNative` | cuTENSOR |
| `NativeLibraryLoader` | Cross-platform library loader |

### Exceptions

| Exception | Thrown by |
|---|---|
| `CudaException` | CUDA runtime errors |
| `CuFftException` | cuFFT errors |
| `CuBlasException` | cuBLASLt errors |
| `CuSparseException` | cuSPARSE errors |
| `CuTensorException` | cuTENSOR errors |

All exceptions expose an `int ErrorCode` property with the native status code.
