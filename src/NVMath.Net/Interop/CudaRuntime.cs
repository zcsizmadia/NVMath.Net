// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the Apache 2.0 License.

using System.Runtime.InteropServices;

namespace NVMathNet.Interop;

/// <summary>
/// CUDA error codes as returned by the CUDA Runtime API.
/// </summary>
public enum CudaError : int
{
    /// <summary>Operation completed successfully.</summary>
    Success = 0,
    /// <summary>A parameter has an invalid value.</summary>
    InvalidValue = 1,
    /// <summary>GPU memory allocation failed.</summary>
    MemoryAllocation = 2,
    /// <summary>CUDA driver/runtime initialisation failed.</summary>
    InitializationError = 3,
    /// <summary>CUDA runtime is being unloaded.</summary>
    CudartUnloading = 4,
    /// <summary>Profiler is disabled.</summary>
    ProfilerDisabled = 5,
    /// <summary>The requested device function is invalid.</summary>
    InvalidDeviceFunction = 8,
    /// <summary>The specified device index is invalid.</summary>
    InvalidDevice = 10,
    /// <summary>The memory-copy direction is not valid.</summary>
    InvalidMemcpyDirection = 21,
    /// <summary>CUDA driver version is below the minimum required.</summary>
    InsufficientDriver = 35,
    /// <summary>No CUDA-capable device was detected.</summary>
    NoDevice = 100,
    /// <summary>The referenced surface object is invalid.</summary>
    InvalidSurface = 31,
    /// <summary>Operation is not supported on this device or configuration.</summary>
    NotSupported = 46,
    /// <summary>Peer-to-peer access has not been enabled between the devices.</summary>
    PeerAccessNotEnabled = 704,
    // Add more as needed
}

/// <summary>
/// Specifies the direction of a memory copy operation.
/// </summary>
public enum CudaMemcpyKind : int
{
    /// <summary>Host memory to host memory.</summary>
    HostToHost = 0,
    /// <summary>Host memory to device memory.</summary>
    HostToDevice = 1,
    /// <summary>Device memory to host memory.</summary>
    DeviceToHost = 2,
    /// <summary>Device memory to device memory.</summary>
    DeviceToDevice = 3,
    /// <summary>Direction is inferred from the pointer types.</summary>
    Default = 4,
}

/// <summary>
/// Raw P/Invoke bindings for the CUDA Runtime API (<c>cudart</c>).
/// All methods are loaded lazily via <see cref="NativeLibraryLoader"/>.
/// </summary>
public static unsafe class CudaRuntime
{
    // ── Library names ────────────────────────────────────────────────────────
    private const string LibWindows = "cudart64_12.dll";
    private const string LibLinux   = "libcudart.so.12";

    private static readonly string LibName =
        RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? LibWindows : LibLinux;

    // ── Delegate types ───────────────────────────────────────────────────────

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CudaError cudaGetDeviceCountDelegate(int* count);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CudaError cudaSetDeviceDelegate(int device);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CudaError cudaGetDeviceDelegate(int* device);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CudaError cudaMallocDelegate(void** devPtr, nuint size);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CudaError cudaMallocAsyncDelegate(void** devPtr, nuint size, nint stream);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CudaError cudaFreeDelegate(void* devPtr);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CudaError cudaFreeAsyncDelegate(void* devPtr, nint stream);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CudaError cudaMallocHostDelegate(void** ptr, nuint size);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CudaError cudaFreeHostDelegate(void* ptr);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CudaError cudaMemcpyDelegate(void* dst, void* src, nuint count, CudaMemcpyKind kind);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CudaError cudaMemcpyAsyncDelegate(void* dst, void* src, nuint count, CudaMemcpyKind kind, nint stream);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CudaError cudaMemsetDelegate(void* devPtr, int value, nuint count);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CudaError cudaMemsetAsyncDelegate(void* devPtr, int value, nuint count, nint stream);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CudaError cudaStreamCreateDelegate(nint* pStream);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CudaError cudaStreamCreateWithFlagsDelegate(nint* pStream, uint flags);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CudaError cudaStreamDestroyDelegate(nint stream);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CudaError cudaStreamSynchronizeDelegate(nint stream);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CudaError cudaStreamWaitEventDelegate(nint stream, nint @event, uint flags);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CudaError cudaEventCreateDelegate(nint* pEvent);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CudaError cudaEventCreateWithFlagsDelegate(nint* pEvent, uint flags);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CudaError cudaEventDestroyDelegate(nint @event);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CudaError cudaEventRecordDelegate(nint @event, nint stream);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CudaError cudaEventSynchronizeDelegate(nint @event);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CudaError cudaEventElapsedTimeDelegate(float* ms, nint start, nint end);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CudaError cudaDeviceSynchronizeDelegate();

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CudaError cudaGetLastErrorDelegate();

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate byte* cudaGetErrorStringDelegate(CudaError error);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CudaError cudaDriverGetVersionDelegate(int* driverVersion);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate CudaError cudaRuntimeGetVersionDelegate(int* runtimeVersion);

    // ── Lazy-loaded delegates ────────────────────────────────────────────────

    private static T Load<T>(string name) where T : Delegate =>
        Marshal.GetDelegateForFunctionPointer<T>(NativeLibraryLoader.GetExport(LibName, name));

    private static readonly Lazy<cudaGetDeviceCountDelegate>    _getDeviceCount  = new(() => Load<cudaGetDeviceCountDelegate>("cudaGetDeviceCount"));
    private static readonly Lazy<cudaSetDeviceDelegate>         _setDevice       = new(() => Load<cudaSetDeviceDelegate>("cudaSetDevice"));
    private static readonly Lazy<cudaGetDeviceDelegate>         _getDevice       = new(() => Load<cudaGetDeviceDelegate>("cudaGetDevice"));
    private static readonly Lazy<cudaMallocDelegate>            _malloc          = new(() => Load<cudaMallocDelegate>("cudaMalloc"));
    private static readonly Lazy<cudaMallocAsyncDelegate>       _mallocAsync     = new(() => Load<cudaMallocAsyncDelegate>("cudaMallocAsync"));
    private static readonly Lazy<cudaFreeDelegate>              _free            = new(() => Load<cudaFreeDelegate>("cudaFree"));
    private static readonly Lazy<cudaFreeAsyncDelegate>         _freeAsync       = new(() => Load<cudaFreeAsyncDelegate>("cudaFreeAsync"));
    private static readonly Lazy<cudaMallocHostDelegate>        _mallocHost      = new(() => Load<cudaMallocHostDelegate>("cudaMallocHost"));
    private static readonly Lazy<cudaFreeHostDelegate>          _freeHost        = new(() => Load<cudaFreeHostDelegate>("cudaFreeHost"));
    private static readonly Lazy<cudaMemcpyDelegate>            _memcpy          = new(() => Load<cudaMemcpyDelegate>("cudaMemcpy"));
    private static readonly Lazy<cudaMemcpyAsyncDelegate>       _memcpyAsync     = new(() => Load<cudaMemcpyAsyncDelegate>("cudaMemcpyAsync"));
    private static readonly Lazy<cudaMemsetDelegate>            _memset          = new(() => Load<cudaMemsetDelegate>("cudaMemset"));
    private static readonly Lazy<cudaMemsetAsyncDelegate>       _memsetAsync     = new(() => Load<cudaMemsetAsyncDelegate>("cudaMemsetAsync"));
    private static readonly Lazy<cudaStreamCreateDelegate>      _streamCreate    = new(() => Load<cudaStreamCreateDelegate>("cudaStreamCreate"));
    private static readonly Lazy<cudaStreamCreateWithFlagsDelegate> _streamCreateWithFlags = new(() => Load<cudaStreamCreateWithFlagsDelegate>("cudaStreamCreateWithFlags"));
    private static readonly Lazy<cudaStreamDestroyDelegate>     _streamDestroy   = new(() => Load<cudaStreamDestroyDelegate>("cudaStreamDestroy"));
    private static readonly Lazy<cudaStreamSynchronizeDelegate> _streamSync      = new(() => Load<cudaStreamSynchronizeDelegate>("cudaStreamSynchronize"));
    private static readonly Lazy<cudaStreamWaitEventDelegate>   _streamWaitEvent = new(() => Load<cudaStreamWaitEventDelegate>("cudaStreamWaitEvent"));
    private static readonly Lazy<cudaEventCreateDelegate>       _eventCreate     = new(() => Load<cudaEventCreateDelegate>("cudaEventCreate"));
    private static readonly Lazy<cudaEventCreateWithFlagsDelegate> _eventCreateWithFlags = new(() => Load<cudaEventCreateWithFlagsDelegate>("cudaEventCreateWithFlags"));
    private static readonly Lazy<cudaEventDestroyDelegate>      _eventDestroy    = new(() => Load<cudaEventDestroyDelegate>("cudaEventDestroy"));
    private static readonly Lazy<cudaEventRecordDelegate>       _eventRecord     = new(() => Load<cudaEventRecordDelegate>("cudaEventRecord"));
    private static readonly Lazy<cudaEventSynchronizeDelegate>  _eventSync       = new(() => Load<cudaEventSynchronizeDelegate>("cudaEventSynchronize"));
    private static readonly Lazy<cudaEventElapsedTimeDelegate>  _eventElapsed    = new(() => Load<cudaEventElapsedTimeDelegate>("cudaEventElapsedTime"));
    private static readonly Lazy<cudaDeviceSynchronizeDelegate> _deviceSync      = new(() => Load<cudaDeviceSynchronizeDelegate>("cudaDeviceSynchronize"));
    private static readonly Lazy<cudaGetLastErrorDelegate>      _getLastError    = new(() => Load<cudaGetLastErrorDelegate>("cudaGetLastError"));
    private static readonly Lazy<cudaGetErrorStringDelegate>    _getErrorString  = new(() => Load<cudaGetErrorStringDelegate>("cudaGetErrorString"));
    private static readonly Lazy<cudaDriverGetVersionDelegate>  _driverVersion   = new(() => Load<cudaDriverGetVersionDelegate>("cudaDriverGetVersion"));
    private static readonly Lazy<cudaRuntimeGetVersionDelegate> _runtimeVersion  = new(() => Load<cudaRuntimeGetVersionDelegate>("cudaRuntimeGetVersion"));

    // ── Public API ───────────────────────────────────────────────────────────

    /// <summary>
    /// Throws <see cref="CudaException"/> if <paramref name="error"/> is not
    /// <see cref="CudaError.Success"/>. Optionally prefixes the message with
    /// <paramref name="context"/>.
    /// </summary>
    public static void CheckError(CudaError error, string? context = null)
    {
        if (error == CudaError.Success) return;
        string msg = GetErrorString(error);
        throw new CudaException(error, context is null ? msg : $"{context}: {msg}");
    }

    /// <summary>Returns the human-readable description for a CUDA error code.</summary>
    public static string GetErrorString(CudaError error)
    {
        byte* ptr = _getErrorString.Value(error);
        return ptr == null ? $"CUDA error {(int)error}" : Marshal.PtrToStringAnsi((nint)ptr) ?? $"CUDA error {(int)error}";
    }

    /// <summary>Returns the number of CUDA-capable devices visible to the runtime.</summary>
    public static int GetDeviceCount()
    {
        int count;
        CheckError(_getDeviceCount.Value(&count));
        return count;
    }

    /// <summary>Sets the active CUDA device for the calling host thread.</summary>
    public static void SetDevice(int device) =>
        CheckError(_setDevice.Value(device));

    /// <summary>Returns the index of the device currently selected for the calling thread.</summary>
    public static int GetDevice()
    {
        int dev;
        CheckError(_getDevice.Value(&dev));
        return dev;
    }

    /// <summary>Allocates <paramref name="size"/> bytes of device memory synchronously.</summary>
    /// <returns>Raw device pointer.</returns>
    public static void* Malloc(nuint size)
    {
        void* ptr;
        CheckError(_malloc.Value(&ptr, size), "cudaMalloc");
        return ptr;
    }

    /// <summary>Allocates <paramref name="size"/> bytes of device memory asynchronously on <paramref name="stream"/>.</summary>
    /// <returns>Raw device pointer.</returns>
    public static void* MallocAsync(nuint size, nint stream)
    {
        void* ptr;
        CheckError(_mallocAsync.Value(&ptr, size, stream), "cudaMallocAsync");
        return ptr;
    }

    /// <summary>Frees device memory synchronously.</summary>
    public static void Free(void* devPtr) =>
        CheckError(_free.Value(devPtr), "cudaFree");

    /// <summary>Frees device memory asynchronously on <paramref name="stream"/>.</summary>
    public static void FreeAsync(void* devPtr, nint stream) =>
        CheckError(_freeAsync.Value(devPtr, stream), "cudaFreeAsync");

    /// <summary>Allocates <paramref name="size"/> bytes of page-locked (pinned) host memory.</summary>
    /// <returns>Raw host pointer.</returns>
    public static void* MallocHost(nuint size)
    {
        void* ptr;
        CheckError(_mallocHost.Value(&ptr, size), "cudaMallocHost");
        return ptr;
    }

    /// <summary>Frees page-locked host memory allocated with <see cref="MallocHost"/>.</summary>
    public static void FreeHost(void* ptr) =>
        CheckError(_freeHost.Value(ptr), "cudaFreeHost");

    /// <summary>Copies <paramref name="count"/> bytes synchronously.</summary>
    public static void Memcpy(void* dst, void* src, nuint count, CudaMemcpyKind kind) =>
        CheckError(_memcpy.Value(dst, src, count, kind), "cudaMemcpy");

    /// <summary>Copies <paramref name="count"/> bytes asynchronously on <paramref name="stream"/>.</summary>
    public static void MemcpyAsync(void* dst, void* src, nuint count, CudaMemcpyKind kind, nint stream) =>
        CheckError(_memcpyAsync.Value(dst, src, count, kind, stream), "cudaMemcpyAsync");

    /// <summary>Fills <paramref name="count"/> bytes of device memory with <paramref name="value"/> synchronously.</summary>
    public static void Memset(void* devPtr, int value, nuint count) =>
        CheckError(_memset.Value(devPtr, value, count), "cudaMemset");

    /// <summary>Fills <paramref name="count"/> bytes of device memory with <paramref name="value"/> asynchronously.</summary>
    public static void MemsetAsync(void* devPtr, int value, nuint count, nint stream) =>
        CheckError(_memsetAsync.Value(devPtr, value, count, stream), "cudaMemsetAsync");

    /// <summary>Creates a CUDA stream with default flags. Caller must call <see cref="StreamDestroy"/> when done.</summary>
    public static nint StreamCreate()
    {
        nint s;
        CheckError(_streamCreate.Value(&s), "cudaStreamCreate");
        return s;
    }

    /// <summary>
    /// Creates a CUDA stream with the given <paramref name="flags"/>.
    /// Use <see cref="StreamNonBlocking"/> to create a non-blocking stream.
    /// </summary>
    public static nint StreamCreateWithFlags(uint flags)
    {
        nint s;
        CheckError(_streamCreateWithFlags.Value(&s, flags), "cudaStreamCreateWithFlags");
        return s;
    }

    /// <summary>cudaStreamNonBlocking flag value.</summary>
    public const uint StreamNonBlocking = 0x01;

    /// <summary>Destroys a CUDA stream.</summary>
    public static void StreamDestroy(nint stream) =>
        CheckError(_streamDestroy.Value(stream), "cudaStreamDestroy");

    /// <summary>Blocks the calling thread until all operations in <paramref name="stream"/> complete.</summary>
    public static void StreamSynchronize(nint stream) =>
        CheckError(_streamSync.Value(stream), "cudaStreamSynchronize");

    /// <summary>Makes <paramref name="stream"/> wait until <paramref name="event"/> has been recorded.</summary>
    public static void StreamWaitEvent(nint stream, nint @event, uint flags = 0) =>
        CheckError(_streamWaitEvent.Value(stream, @event, flags), "cudaStreamWaitEvent");

    /// <summary>Creates a CUDA event with default flags. Caller must call <see cref="EventDestroy"/> when done.</summary>
    public static nint EventCreate()
    {
        nint e;
        CheckError(_eventCreate.Value(&e), "cudaEventCreate");
        return e;
    }

    /// <summary>
    /// Creates a CUDA event with the given <paramref name="flags"/>.
    /// Use <see cref="EventDisableTiming"/> to skip timing and reduce overhead.
    /// </summary>
    public static nint EventCreateWithFlags(uint flags)
    {
        nint e;
        CheckError(_eventCreateWithFlags.Value(&e, flags), "cudaEventCreateWithFlags");
        return e;
    }

    /// <summary>cudaEventDisableTiming flag value.</summary>
    public const uint EventDisableTiming = 0x02;

    /// <summary>Destroys a CUDA event.</summary>
    public static void EventDestroy(nint @event) =>
        CheckError(_eventDestroy.Value(@event), "cudaEventDestroy");

    /// <summary>Records <paramref name="event"/> in <paramref name="stream"/>.</summary>
    public static void EventRecord(nint @event, nint stream) =>
        CheckError(_eventRecord.Value(@event, stream), "cudaEventRecord");

    /// <summary>Blocks the calling thread until <paramref name="event"/> has been recorded.</summary>
    public static void EventSynchronize(nint @event) =>
        CheckError(_eventSync.Value(@event), "cudaEventSynchronize");

    /// <summary>Returns elapsed time in milliseconds between two recorded events.</summary>
    public static float EventElapsedTime(nint start, nint end)
    {
        float ms;
        CheckError(_eventElapsed.Value(&ms, start, end), "cudaEventElapsedTime");
        return ms;
    }

    /// <summary>Blocks until all preceding CUDA work on all streams of the active device completes.</summary>
    public static void DeviceSynchronize() =>
        CheckError(_deviceSync.Value(), "cudaDeviceSynchronize");

    /// <summary>Returns and clears the last CUDA error set on the calling host thread.</summary>
    public static CudaError GetLastError() =>
        _getLastError.Value();

    /// <summary>Returns the CUDA driver version (e.g. 12000 = v12.0).</summary>
    public static int GetDriverVersion()
    {
        int v;
        CheckError(_driverVersion.Value(&v));
        return v;
    }

    /// <summary>Returns the CUDA runtime version (e.g. 12000 = v12.0).</summary>
    public static int GetRuntimeVersion()
    {
        int v;
        CheckError(_runtimeVersion.Value(&v));
        return v;
    }
}
