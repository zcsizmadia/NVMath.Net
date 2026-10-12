// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the MIT License.

using NVMathNet.Interop;

namespace NVMathNet.Memory;

/// <summary>Process-wide settings for device memory allocation.</summary>
public static class DeviceMemory
{
    private static IDeviceMemoryManager _default = CudaMallocMemoryManager.Instance;

    /// <summary>
    /// The manager used by <see cref="DeviceBuffer{T}"/> constructors that do not take an explicit
    /// manager. Defaults to <see cref="CudaMallocMemoryManager.Instance"/>. A buffer remembers the
    /// manager it was allocated from, so changing this never affects existing buffers.
    /// </summary>
    public static IDeviceMemoryManager Default
    {
        get => Volatile.Read(ref _default);
        set
        {
            ArgumentNullException.ThrowIfNull(value);
            Volatile.Write(ref _default, value);
        }
    }
}

/// <summary>The default manager: plain <c>cudaMalloc</c> / <c>cudaFree</c>.</summary>
public sealed class CudaMallocMemoryManager : IDeviceMemoryManager
{
    /// <summary>The shared instance.</summary>
    public static CudaMallocMemoryManager Instance { get; } = new();

    private CudaMallocMemoryManager()
    {
    }

    /// <inheritdoc/>
    public unsafe nint Allocate(nuint sizeInBytes) => (nint)CudaRuntime.Malloc(sizeInBytes);

    /// <inheritdoc/>
    public unsafe void Free(nint pointer) => CudaRuntime.Free((void*)pointer);
}
