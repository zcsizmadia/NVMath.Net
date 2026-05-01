// Copyright (c) 2026 NVMath.Net Contributors. All rights reserved.
// Licensed under the Apache 2.0 License.

using System.Numerics;
using System.Runtime.CompilerServices;
using NVMathNet.Interop;

namespace NVMathNet;

/// <summary>
/// Page-locked (pinned) host memory for maximum-throughput, zero-copy transfers
/// between host and device.
/// </summary>
/// <typeparam name="T">Unmanaged numeric element type.</typeparam>
public sealed class PinnedBuffer<T> : IDisposable where T : unmanaged, INumberBase<T>
{
    private nint _ptr;
    private bool _disposed;

    /// <summary>Number of elements.</summary>
    public long Length { get; }

    /// <summary>Total size in bytes.</summary>
    public nuint SizeInBytes => (nuint)(Length * Unsafe.SizeOf<T>());

    /// <summary>Raw host pointer as <see cref="nint"/> (page-locked).</summary>
    public nint PointerAsInt
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _ptr;
        }
    }

    /// <summary>Raw host pointer as <c>void*</c>.</summary>
    public unsafe void* Pointer => (void*)PointerAsInt;

    /// <summary>A <see cref="Span{T}"/> over the pinned region. Valid as long as the buffer is alive.</summary>
    public unsafe Span<T> Span
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return new Span<T>((void*)_ptr, (int)Length);
        }
    }

    /// <summary>Allocates <paramref name="length"/> pinned host elements.</summary>
    public unsafe PinnedBuffer(long length)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(length);
        Length = length;
        _ptr = (nint)CudaRuntime.MallocHost((nuint)(length * Unsafe.SizeOf<T>()));
    }

    /// <summary>
    /// Copies the pinned buffer to a <see cref="DeviceBuffer{T}"/> using DMA.
    /// </summary>
    public unsafe void CopyToDevice(DeviceBuffer<T> device, CudaStream stream)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        nuint byteCount = (nuint)(Length * Unsafe.SizeOf<T>());
        CudaRuntime.MemcpyAsync(device.Pointer, (void*)_ptr, byteCount, CudaMemcpyKind.HostToDevice, stream.Handle);
    }

    /// <summary>
    /// Copies a <see cref="DeviceBuffer{T}"/> back into this pinned buffer.
    /// </summary>
    public unsafe void CopyFromDevice(DeviceBuffer<T> device, CudaStream stream)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        nuint byteCount = (nuint)(Length * Unsafe.SizeOf<T>());
        CudaRuntime.MemcpyAsync((void*)_ptr, device.Pointer, byteCount, CudaMemcpyKind.DeviceToHost, stream.Handle);
    }

    /// <inheritdoc/>
    public unsafe void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_ptr != 0)
        {
            CudaRuntime.FreeHost((void*)_ptr);
            _ptr = 0;
        }
    }
}
