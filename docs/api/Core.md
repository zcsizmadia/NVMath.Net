# Core Infrastructure — `NVMathNet`

## `CudaDevice`

Represents a CUDA-capable device.

| Member | Description |
|---|---|
| `CudaDevice(int deviceId = 0)` | Constructs a device handle for the given index. |
| `int DeviceId` | Zero-based device index. |
| `static int GetDeviceCount()` | Number of available CUDA devices. |
| `static CudaDevice Get(int deviceId = 0)` | Returns the `CudaDevice` for the given index. |
| `static CudaDevice GetCurrent()` | Returns the currently selected device. |
| `void SetAsCurrent()` | Selects this device for the calling thread. |
| `void Synchronize()` | Blocks until all work on the device completes. |
| `static int GetDriverVersion()` | Returns the CUDA driver version. |
| `static int GetRuntimeVersion()` | Returns the CUDA runtime version. |

---

## `CudaStream`

Wraps a CUDA stream. Implements `IAsyncDisposable`, `IDisposable`.

| Member | Description |
|---|---|
| `CudaStream(bool nonBlocking = true)` | Creates a new stream on the current device. |
| `static CudaStream FromHandle(nint handle)` | Wraps an existing CUDA stream handle. |
| `nint Handle` | Raw CUDA stream handle. |
| `Task SynchronizeAsync(CancellationToken ct = default)` | Waits for all operations in the stream to complete. |
| `void Synchronize()` | Synchronous version of `SynchronizeAsync`. |
| `void WaitEvent(CudaEvent event)` | Makes the stream wait for an event (GPU-side). |
| `ValueTask DisposeAsync()` | Destroys the stream asynchronously. |
| `void Dispose()` | Destroys the stream. |

---

## `CudaEvent`

Wraps a CUDA event for timing and synchronisation. Implements `IAsyncDisposable`, `IDisposable`.

| Member | Description |
|---|---|
| `CudaEvent(bool disableTiming = true)` | Creates an event. Pass `false` to enable timing. |
| `nint Handle` | Raw CUDA event handle. |
| `void Record(CudaStream stream)` | Records the event in a stream. |
| `void Record()` | Records the event in the default stream. |
| `float ElapsedMilliseconds(CudaEvent start)` | Returns elapsed ms between `start` and `this`. |
| `static float ElapsedMilliseconds(CudaEvent start, CudaEvent stop)` | Static overload. |
| `Task SynchronizeAsync(CancellationToken ct = default)` | Waits until the event is complete. |
| `void Synchronize()` | Blocks until the event is complete. |
| `ValueTask DisposeAsync()` | Destroys the event asynchronously. |
| `void Dispose()` | Destroys the event. |

---

## `DeviceBuffer<T>`

Strongly-typed GPU device buffer. Implements `IDisposable`, `IAsyncDisposable`.
Constraint: `T : unmanaged, INumberBase<T>`.

| Member | Description |
|---|---|
| `DeviceBuffer(long length)` | Allocates `length * sizeof(T)` bytes on the device. |
| `static DeviceBuffer<T> AllocAsync(long length, CudaStream stream)` | Allocates asynchronously on a stream. |
| `long Length` | Number of elements. |
| `nuint SizeInBytes` | Total size in bytes. |
| `nint PointerAsInt` | Raw device pointer as `nint`. |
| `unsafe void* Pointer` | Raw device pointer. |
| `void CopyFrom(ReadOnlySpan<T> source, CudaStream? stream = null)` | Copies host → device. |
| `void CopyTo(Span<T> destination, CudaStream? stream = null)` | Copies device → host. |
| `Task CopyFromAsync(T[] source, CudaStream stream, CancellationToken ct = default)` | Async host → device. |
| `Task CopyToAsync(T[] destination, CudaStream stream, CancellationToken ct = default)` | Async device → host. |
| `void CopyFrom(DeviceBuffer<T> other, CudaStream stream)` | Device-to-device copy. |
| `void Clear(CudaStream? stream = null)` | Fills the buffer with zeros. |
| `T[] ToArray()` | Copies the buffer to a new host array. |
| `void Dispose()` | Frees the device memory. |
| `ValueTask DisposeAsync()` | Frees the device memory asynchronously. |

---

## `PinnedBuffer<T>`

Page-locked (pinned) host memory for fast DMA transfers. Implements `IDisposable`.
Constraint: `T : unmanaged, INumberBase<T>`.

| Member | Description |
|---|---|
| `PinnedBuffer(long length)` | Allocates pinned host memory. |
| `long Length` | Number of elements. |
| `nuint SizeInBytes` | Total size in bytes. |
| `nint PointerAsInt` | Raw pointer as `nint`. |
| `unsafe void* Pointer` | Raw pointer. |
| `unsafe Span<T> Span` | Span over the pinned memory. |
| `void CopyToDevice(DeviceBuffer<T> dst, CudaStream stream)` | Host → device DMA. |
| `void CopyFromDevice(DeviceBuffer<T> src, CudaStream stream)` | Device → host DMA. |
| `void Dispose()` | Frees pinned memory. |

---

## `MultiGpuContext`

Manages a set of CUDA devices for single-process multi-GPU execution. Implements `IDisposable`.

| Member | Description |
|---|---|
| `MultiGpuContext(int[]? deviceIds = null)` | Creates context; defaults to all devices. Enables peer access where supported. |
| `IReadOnlyList<int> DeviceIds` | Managed device IDs. |
| `int DeviceCount` | Number of GPUs. |
| `CudaStream GetStream(int index)` | Stream for device at index. |
| `int GetDeviceId(int index)` | CUDA device ID at index. |
| `string GetDeviceName(int index)` | Device name. |
| `void SetDevice(int index)` | Sets active device for calling thread. |
| `void ForEachDevice(Action<int, int, CudaStream> action)` | Runs action per device in parallel. |
| `Task ForEachDeviceAsync(Func<int, int, CudaStream, Task> action, CancellationToken ct = default)` | Async per-device execution. |
| `void SynchronizeAll()` | Blocks until all device streams complete. |
| `Task SynchronizeAllAsync(CancellationToken ct = default)` | Async synchronise all. |
| `void CopyPeerAsync<T>(DeviceBuffer<T> dst, int dstIdx, DeviceBuffer<T> src, int srcIdx, CudaStream? stream = null)` | Peer-to-peer device copy. |
| `bool CanAccessPeer(int fromIndex, int toIndex)` | Whether peer access is possible. |
